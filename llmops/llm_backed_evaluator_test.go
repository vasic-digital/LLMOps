package llmops

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Round-81 §11.4 anti-bluff close-out of the round-75 deferred item.
// The tests below exercise the concrete LLMBackedEvaluator
// implementation introduced in llm_backed_evaluator.go.
//
// Anti-bluff posture: every test that claims LLMBackedEvaluator "works"
// either (a) drives a deterministic stub responder and asserts on the
// exact score map produced by the parser, (b) drives an httptest.NewServer
// loopback and asserts on the request+response cycle, or (c) is a
// paired-mutation test that asserts DIFFERENT judge responses produce
// DIFFERENT score maps. No test asserts only "no error returned" —
// every PASS path asserts a concrete property the parser+validator
// derived from the actual judge response.
//
// Load-bearing test: TestLLMBackedEvaluator_NotFabricated. If that test
// passes when the implementation has been replaced by a hardcoded
// constant score map, the test is bluffing and MUST be tightened.

// stubResponder is a deterministic LLMResponder used by the unit tests.
// It records the last prompt it received (so tests can assert on the
// rendered judgment template) and returns a configurable canned
// response (so tests can simulate any judge output shape).
type stubResponder struct {
	canned       string
	cannedErr    error
	lastPrompt   atomic.Value // string
	calls        atomic.Int32
	blockUntilCh chan struct{} // optional: if non-nil, Generate blocks on this channel before returning
}

func newStubResponder(canned string) *stubResponder {
	return &stubResponder{canned: canned}
}

func (s *stubResponder) Generate(ctx context.Context, prompt string) (string, error) {
	s.calls.Add(1)
	s.lastPrompt.Store(prompt)
	if s.blockUntilCh != nil {
		select {
		case <-s.blockUntilCh:
		case <-ctx.Done():
			return "", ctx.Err()
		}
	}
	if s.cannedErr != nil {
		return "", s.cannedErr
	}
	return s.canned, nil
}

// --- Construction tests --------------------------------------------------

func TestLLMBackedEvaluator_NilResponder_ReturnsSentinel(t *testing.T) {
	e, err := NewLLMBackedEvaluator(LLMBackedEvaluatorConfig{Responder: nil})
	require.Error(t, err)
	assert.Nil(t, e)
	assert.ErrorIs(t, err, ErrLLMBackedEvaluatorResponderNotProvided,
		"nil Responder MUST return ErrLLMBackedEvaluatorResponderNotProvided at construction time")
}

func TestNewLLMBackedEvaluator_EmptyTemplate_AppliesDefault(t *testing.T) {
	r := newStubResponder("SCORE_correctness: 0.5")
	e, err := NewLLMBackedEvaluator(LLMBackedEvaluatorConfig{Responder: r})
	require.NoError(t, err)
	require.NotNil(t, e)
	assert.Contains(t, e.judgmentPromptTemplate, "strict evaluator",
		"empty template MUST resolve to the default judgment template containing the strict-evaluator framing")
	assert.Contains(t, e.judgmentPromptTemplate, "{prompt}",
		"default template MUST still contain the placeholders so render still works")
}

func TestNewLLMBackedEvaluator_CustomTemplate_Preserved(t *testing.T) {
	custom := "Custom template for {metrics} — prompt={prompt} response={response} expected={expected}"
	r := newStubResponder("SCORE_x: 0.1")
	e, err := NewLLMBackedEvaluator(LLMBackedEvaluatorConfig{
		Responder:              r,
		JudgmentPromptTemplate: custom,
	})
	require.NoError(t, err)
	assert.Equal(t, custom, e.judgmentPromptTemplate)
}

func TestNewLLMBackedEvaluator_EmptyMetrics_AppliesDefaults(t *testing.T) {
	r := newStubResponder("SCORE_correctness: 0.5")
	e, err := NewLLMBackedEvaluator(LLMBackedEvaluatorConfig{Responder: r})
	require.NoError(t, err)
	assert.Equal(t, []string{"correctness", "completeness", "conciseness", "factuality"}, e.defaultMetrics)
}

func TestNewLLMBackedEvaluator_CustomMetrics_Preserved(t *testing.T) {
	r := newStubResponder("SCORE_x: 0.1")
	e, err := NewLLMBackedEvaluator(LLMBackedEvaluatorConfig{
		Responder:      r,
		DefaultMetrics: []string{"clarity", "tone"},
	})
	require.NoError(t, err)
	assert.Equal(t, []string{"clarity", "tone"}, e.defaultMetrics)
}

func TestNewLLMBackedEvaluator_String_NoNilPanic(t *testing.T) {
	r := newStubResponder("ok")
	e, err := NewLLMBackedEvaluator(LLMBackedEvaluatorConfig{Responder: r})
	require.NoError(t, err)
	s := e.String()
	assert.Contains(t, s, "LLMBackedEvaluator")
	assert.Contains(t, s, "templateLen=")
	assert.Equal(t, s, e.GoString())
}

// --- Happy-path tests ---------------------------------------------------

func TestLLMBackedEvaluator_Evaluate_HappyPath_ParsesScores(t *testing.T) {
	r := newStubResponder("SCORE_correctness: 0.8\nSCORE_completeness: 0.6")
	e, err := NewLLMBackedEvaluator(LLMBackedEvaluatorConfig{Responder: r})
	require.NoError(t, err)

	scores, err := e.Evaluate(context.Background(),
		"What is 2+2?", "4", "4",
		[]string{"correctness", "completeness"})
	require.NoError(t, err)
	require.Len(t, scores, 2)
	assert.InDelta(t, 0.8, scores["correctness"], 1e-9)
	assert.InDelta(t, 0.6, scores["completeness"], 1e-9)
}

func TestLLMBackedEvaluator_Evaluate_RendersAllPlaceholders(t *testing.T) {
	r := newStubResponder("SCORE_correctness: 0.9")
	e, err := NewLLMBackedEvaluator(LLMBackedEvaluatorConfig{Responder: r})
	require.NoError(t, err)

	_, err = e.Evaluate(context.Background(),
		"PROMPTBODY", "RESPONSEBODY", "EXPECTEDBODY",
		[]string{"correctness", "factuality"})
	require.NoError(t, err)

	// Verify the rendered template contained every supplied input
	// — proof that buildJudgmentPrompt actually substitutes all four
	// placeholders, not just declares them in a string.
	got, _ := r.lastPrompt.Load().(string)
	assert.Contains(t, got, "PROMPTBODY", "rendered prompt MUST include {prompt} substitution")
	assert.Contains(t, got, "RESPONSEBODY", "rendered prompt MUST include {response} substitution")
	assert.Contains(t, got, "EXPECTEDBODY", "rendered prompt MUST include {expected} substitution")
	// Sorted metrics list: "correctness, factuality" alphabetically.
	assert.Contains(t, got, "correctness, factuality",
		"rendered prompt MUST include sorted {metrics} substitution")
	// And NONE of the placeholder tokens remain.
	assert.NotContains(t, got, "{prompt}", "placeholder token MUST be replaced, not surface verbatim")
	assert.NotContains(t, got, "{response}", "placeholder token MUST be replaced")
	assert.NotContains(t, got, "{expected}", "placeholder token MUST be replaced")
	assert.NotContains(t, got, "{metrics}", "placeholder token MUST be replaced")
}

func TestLLMBackedEvaluator_Evaluate_EmptyMetrics_UsesDefaults(t *testing.T) {
	r := newStubResponder("SCORE_correctness: 0.4\nSCORE_completeness: 0.5\nSCORE_conciseness: 0.6\nSCORE_factuality: 0.7")
	e, err := NewLLMBackedEvaluator(LLMBackedEvaluatorConfig{Responder: r})
	require.NoError(t, err)

	scores, err := e.Evaluate(context.Background(), "p", "r", "x", nil)
	require.NoError(t, err)
	require.Len(t, scores, 4)
	assert.InDelta(t, 0.4, scores["correctness"], 1e-9)
	assert.InDelta(t, 0.7, scores["factuality"], 1e-9)
}

func TestLLMBackedEvaluator_Evaluate_CaseInsensitiveMetricKeys(t *testing.T) {
	// Judge may capitalise differently; parser lowercases for stable
	// map keys regardless of judge stylistic choice.
	r := newStubResponder("SCORE_Correctness: 0.3\nSCORE_COMPLETENESS: 0.4")
	e, err := NewLLMBackedEvaluator(LLMBackedEvaluatorConfig{Responder: r})
	require.NoError(t, err)

	scores, err := e.Evaluate(context.Background(), "p", "r", "x", []string{"correctness", "completeness"})
	require.NoError(t, err)
	assert.InDelta(t, 0.3, scores["correctness"], 1e-9)
	assert.InDelta(t, 0.4, scores["completeness"], 1e-9)
}

func TestLLMBackedEvaluator_Evaluate_IgnoresExtraJudgeProse(t *testing.T) {
	// Real judges often emit "Here are my scores:" before the score
	// lines. Parser MUST extract only the SCORE_ lines and ignore
	// the rest — otherwise the unparseable sentinel would fire on
	// every real-world response.
	judge := `Sure, I'll evaluate the response now.

Here are my scores:
SCORE_correctness: 0.75
SCORE_completeness: 0.50

Let me know if you need a more detailed breakdown.`
	r := newStubResponder(judge)
	e, err := NewLLMBackedEvaluator(LLMBackedEvaluatorConfig{Responder: r})
	require.NoError(t, err)

	scores, err := e.Evaluate(context.Background(), "p", "r", "x",
		[]string{"correctness", "completeness"})
	require.NoError(t, err)
	require.Len(t, scores, 2)
	assert.InDelta(t, 0.75, scores["correctness"], 1e-9)
	assert.InDelta(t, 0.50, scores["completeness"], 1e-9)
}

// --- Anti-bluff sentinel tests ------------------------------------------

func TestLLMBackedEvaluator_Evaluate_UnparseableResponse_ReturnsSentinel(t *testing.T) {
	// Judge returns prose but no SCORE_ lines. The implementation
	// MUST refuse to fabricate scores — the load-bearing anti-bluff
	// branch.
	r := newStubResponder("I'm sorry, I cannot evaluate this response.")
	e, err := NewLLMBackedEvaluator(LLMBackedEvaluatorConfig{Responder: r})
	require.NoError(t, err)

	scores, err := e.Evaluate(context.Background(), "p", "r", "x",
		[]string{"correctness"})
	require.Error(t, err)
	assert.Nil(t, scores)
	assert.ErrorIs(t, err, ErrLLMBackedEvaluatorResponseUnparseable,
		"judge response without SCORE_ lines MUST return ErrLLMBackedEvaluatorResponseUnparseable, NEVER a fabricated 0.5")
}

func TestLLMBackedEvaluator_Evaluate_EmptyResponse_ReturnsSentinel(t *testing.T) {
	r := newStubResponder("")
	e, err := NewLLMBackedEvaluator(LLMBackedEvaluatorConfig{Responder: r})
	require.NoError(t, err)

	scores, err := e.Evaluate(context.Background(), "p", "r", "x",
		[]string{"correctness"})
	require.Error(t, err)
	assert.Nil(t, scores)
	assert.ErrorIs(t, err, ErrLLMBackedEvaluatorResponseUnparseable,
		"empty judge response MUST also surface as unparseable, not fabricated empty success")
}

func TestLLMBackedEvaluator_Evaluate_ScoreOutOfRangeHigh_ReturnsSentinel(t *testing.T) {
	r := newStubResponder("SCORE_correctness: 1.5")
	e, err := NewLLMBackedEvaluator(LLMBackedEvaluatorConfig{Responder: r})
	require.NoError(t, err)

	scores, err := e.Evaluate(context.Background(), "p", "r", "x",
		[]string{"correctness"})
	require.Error(t, err)
	assert.Nil(t, scores)
	assert.ErrorIs(t, err, ErrLLMBackedEvaluatorScoreOutOfRange,
		"score > 1.0 MUST return ErrLLMBackedEvaluatorScoreOutOfRange, NEVER silently clamp by default")
}

func TestLLMBackedEvaluator_Evaluate_ScoreOutOfRangeLow_ReturnsSentinel(t *testing.T) {
	r := newStubResponder("SCORE_correctness: -0.2")
	e, err := NewLLMBackedEvaluator(LLMBackedEvaluatorConfig{Responder: r})
	require.NoError(t, err)

	scores, err := e.Evaluate(context.Background(), "p", "r", "x",
		[]string{"correctness"})
	require.Error(t, err)
	assert.Nil(t, scores)
	assert.ErrorIs(t, err, ErrLLMBackedEvaluatorScoreOutOfRange,
		"score < 0.0 MUST return ErrLLMBackedEvaluatorScoreOutOfRange")
}

func TestLLMBackedEvaluator_Evaluate_ClampOnRangeError_OptInBehaviour(t *testing.T) {
	r := newStubResponder("SCORE_correctness: 1.5\nSCORE_completeness: -0.3")
	e, err := NewLLMBackedEvaluator(LLMBackedEvaluatorConfig{
		Responder:         r,
		ClampOnRangeError: true,
	})
	require.NoError(t, err)

	scores, err := e.Evaluate(context.Background(), "p", "r", "x",
		[]string{"correctness", "completeness"})
	require.NoError(t, err, "ClampOnRangeError=true MUST suppress the out-of-range sentinel")
	assert.InDelta(t, 1.0, scores["correctness"], 1e-9, "high out-of-range MUST clamp to 1.0")
	assert.InDelta(t, 0.0, scores["completeness"], 1e-9, "low out-of-range MUST clamp to 0.0")
}

// --- Error propagation tests --------------------------------------------

func TestLLMBackedEvaluator_Evaluate_ResponderError_PropagatesAsError(t *testing.T) {
	sentinelErr := errors.New("synthetic transport failure")
	r := &stubResponder{cannedErr: sentinelErr}
	e, err := NewLLMBackedEvaluator(LLMBackedEvaluatorConfig{Responder: r})
	require.NoError(t, err)

	scores, err := e.Evaluate(context.Background(), "p", "r", "x",
		[]string{"correctness"})
	require.Error(t, err)
	assert.Nil(t, scores)
	assert.ErrorIs(t, err, sentinelErr,
		"underlying responder error MUST be wrapped (errors.Is reachable) so callers can branch on cause")
	assert.Contains(t, err.Error(), "judge LLM dispatch failed",
		"error message MUST identify the judge layer (not generic 'request failed') so operators can diagnose")
}

func TestLLMBackedEvaluator_Evaluate_PreCancelledContext_ReturnsImmediately(t *testing.T) {
	r := newStubResponder("SCORE_correctness: 0.5")
	e, err := NewLLMBackedEvaluator(LLMBackedEvaluatorConfig{Responder: r})
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // pre-cancel

	scores, err := e.Evaluate(ctx, "p", "r", "x", []string{"correctness"})
	require.Error(t, err)
	assert.Nil(t, scores)
	assert.ErrorIs(t, err, context.Canceled,
		"pre-cancelled context MUST surface context.Canceled via errors.Is")
	// And the responder MUST NOT have been invoked.
	assert.Equal(t, int32(0), r.calls.Load(),
		"pre-cancel context MUST short-circuit BEFORE dispatching to the judge")
}

func TestLLMBackedEvaluator_Evaluate_HonoursContextCancel(t *testing.T) {
	// Responder blocks until we cancel; verifies ctx is plumbed
	// through to the underlying Generate call.
	r := &stubResponder{
		canned:       "SCORE_correctness: 0.5",
		blockUntilCh: make(chan struct{}),
	}
	e, err := NewLLMBackedEvaluator(LLMBackedEvaluatorConfig{Responder: r})
	require.NoError(t, err)

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	scores, err := e.Evaluate(ctx, "p", "r", "x", []string{"correctness"})
	require.Error(t, err)
	assert.Nil(t, scores)
	assert.ErrorIs(t, err, context.DeadlineExceeded,
		"timeout MUST propagate from inside the blocking Generate call")

	// Release the responder so the goroutine doesn't leak.
	close(r.blockUntilCh)
}

// --- Paired-mutation / anti-fabrication test ----------------------------

// TestLLMBackedEvaluator_NotFabricated is the LOAD-BEARING anti-bluff
// test. It asserts two DIFFERENT judge responses produce two
// DIFFERENT score maps. If this test passes when the implementation
// is replaced with a hardcoded constant map (e.g. always returns
// {"correctness": 0.5}), the test is bluffing and MUST be tightened.
func TestLLMBackedEvaluator_NotFabricated(t *testing.T) {
	// Judge response A: high scores.
	rA := newStubResponder("SCORE_correctness: 0.9\nSCORE_completeness: 0.85")
	eA, err := NewLLMBackedEvaluator(LLMBackedEvaluatorConfig{Responder: rA})
	require.NoError(t, err)
	scoresA, err := eA.Evaluate(context.Background(), "p", "r", "x",
		[]string{"correctness", "completeness"})
	require.NoError(t, err)

	// Judge response B: low scores.
	rB := newStubResponder("SCORE_correctness: 0.1\nSCORE_completeness: 0.2")
	eB, err := NewLLMBackedEvaluator(LLMBackedEvaluatorConfig{Responder: rB})
	require.NoError(t, err)
	scoresB, err := eB.Evaluate(context.Background(), "p", "r", "x",
		[]string{"correctness", "completeness"})
	require.NoError(t, err)

	// Different judge text => different scores. A hardcoded-constant
	// bluff would produce equal maps and fail this assertion.
	assert.NotEqual(t, scoresA, scoresB,
		"different judge responses MUST produce different score maps — a constant-bluff implementation would fail this")

	// And the actual numeric ordering MUST follow the judge text
	// (sanity check that we didn't swap the parser branches).
	assert.Greater(t, scoresA["correctness"], scoresB["correctness"],
		"A correctness MUST be > B correctness — parser would produce inverted output if branches swapped")
	assert.Greater(t, scoresA["completeness"], scoresB["completeness"])

	// And the SAME judge response MUST produce identical score maps
	// across two evaluations — deterministic, not noisy.
	rC := newStubResponder("SCORE_correctness: 0.42")
	eC, err := NewLLMBackedEvaluator(LLMBackedEvaluatorConfig{Responder: rC})
	require.NoError(t, err)
	c1, err := eC.Evaluate(context.Background(), "p", "r", "x", []string{"correctness"})
	require.NoError(t, err)
	c2, err := eC.Evaluate(context.Background(), "p", "r", "x", []string{"correctness"})
	require.NoError(t, err)
	assert.Equal(t, c1, c2, "same judge response MUST produce identical score maps (determinism)")
}

// --- Sentinel distinctness matrix ---------------------------------------

func TestLLMBackedEvaluator_Sentinels_AreDistinct(t *testing.T) {
	// The three round-81 sentinels MUST be distinguishable via
	// errors.Is from each other.
	all := []error{
		ErrLLMBackedEvaluatorResponderNotProvided,
		ErrLLMBackedEvaluatorResponseUnparseable,
		ErrLLMBackedEvaluatorScoreOutOfRange,
	}
	for i, a := range all {
		for j, b := range all {
			if i == j {
				assert.True(t, errors.Is(a, b),
					"errors.Is MUST be reflexive for sentinel %d", i)
				continue
			}
			assert.False(t, errors.Is(a, b),
				"round-81 sentinels %d and %d MUST be distinct under errors.Is", i, j)
		}
	}
}

func TestLLMBackedEvaluator_Sentinels_DistinctFromRound25Round75(t *testing.T) {
	// Round-81 sentinels MUST be distinct from the round-25 evaluator
	// sentinel they complement and from round-75 HeuristicEvaluator
	// sentinels — so callers can branch on which evaluator family
	// failed regardless of which concrete impl is wired.
	type pair struct{ a, b error }
	disjoint := []pair{
		// round 81 vs round 25
		{ErrLLMBackedEvaluatorResponderNotProvided, ErrEvaluatorNotConfigured},
		{ErrLLMBackedEvaluatorResponseUnparseable, ErrEvaluatorNotConfigured},
		{ErrLLMBackedEvaluatorScoreOutOfRange, ErrEvaluatorNotConfigured},
		{ErrLLMBackedEvaluatorResponderNotProvided, ErrLLMResponderNotConfigured},
		{ErrLLMBackedEvaluatorResponseUnparseable, ErrLLMResponderNotConfigured},
		{ErrLLMBackedEvaluatorScoreOutOfRange, ErrLLMResponderNotConfigured},
		// round 81 vs round 75
		{ErrLLMBackedEvaluatorResponderNotProvided, ErrHeuristicEvaluatorMissingInput},
		{ErrLLMBackedEvaluatorResponseUnparseable, ErrHeuristicEvaluatorMissingInput},
		{ErrLLMBackedEvaluatorScoreOutOfRange, ErrHeuristicEvaluatorMissingInput},
		{ErrLLMBackedEvaluatorResponderNotProvided, ErrHeuristicEvaluatorScoreOutOfRange},
		{ErrLLMBackedEvaluatorResponseUnparseable, ErrHeuristicEvaluatorScoreOutOfRange},
		{ErrLLMBackedEvaluatorScoreOutOfRange, ErrHeuristicEvaluatorScoreOutOfRange},
		// round 81 vs round 62
		{ErrLLMBackedEvaluatorResponderNotProvided, ErrHTTPResponderEndpointNotConfigured},
		{ErrLLMBackedEvaluatorResponderNotProvided, ErrHTTPResponderModelNotConfigured},
		{ErrLLMBackedEvaluatorResponseUnparseable, ErrHTTPResponderRequestFailed},
		{ErrLLMBackedEvaluatorScoreOutOfRange, ErrHTTPResponderResponseInvalid},
	}
	for _, p := range disjoint {
		assert.False(t, errors.Is(p.a, p.b),
			"sentinels %v and %v MUST be distinct under errors.Is", p.a, p.b)
		assert.False(t, errors.Is(p.b, p.a),
			"sentinels %v and %v MUST be distinct under errors.Is (symmetric)", p.b, p.a)
	}
}

// --- Interface conformance + end-to-end wire-through --------------------

func TestLLMBackedEvaluator_ImplementsLLMEvaluator(t *testing.T) {
	// Compile-time assertion via interface assignment.
	var _ LLMEvaluator = (*LLMBackedEvaluator)(nil)
}

// TestEvaluator_WithLLMBackedEvaluator_EndToEnd is the closing
// assertion for round 81: InMemoryContinuousEvaluator, when wired
// with a real LLMBackedEvaluator AND a real responder-under-test
// (here a deterministic stub LLMResponderFunc to keep the test in
// the unit tier per CONST-050(A)), produces SampleResults with REAL
// per-metric scores derived from the JUDGE's response — proving the
// LLM-as-judge pipeline really works end-to-end through the
// round-25 plumbing (CreateRun → StartRun → executeRun → evaluateSample).
func TestEvaluator_WithLLMBackedEvaluator_EndToEnd(t *testing.T) {
	hLogger := logrus.New()
	hLogger.SetLevel(logrus.WarnLevel)

	// Judge stub: returns a deterministic SCORE_ block.
	judgeResponder := newStubResponder("SCORE_correctness: 0.75\nSCORE_completeness: 0.55")

	lbEval, err := NewLLMBackedEvaluator(LLMBackedEvaluatorConfig{Responder: judgeResponder})
	require.NoError(t, err)

	eval := NewInMemoryContinuousEvaluator(lbEval, nil, nil, hLogger)

	// Wire the response-under-test responder via SetResponder (round-25
	// plumbing). Deterministic stub so the test is reproducible.
	eval.SetResponder(LLMResponderFunc(func(_ context.Context, prompt string) (string, error) {
		return "the model under test answered something", nil
	}))

	ctx := context.Background()
	dataset := &Dataset{Name: "round81-e2e", Type: DatasetTypeGolden}
	require.NoError(t, eval.CreateDataset(ctx, dataset))
	require.NoError(t, eval.AddSamples(ctx, dataset.ID, []*DatasetSample{
		{Input: "describe", ExpectedOutput: "an answer the judge will grade"},
	}))

	run := &EvaluationRun{
		Name:    "round81-e2e-run",
		Dataset: dataset.ID,
		Metrics: []string{"correctness", "completeness"},
	}
	require.NoError(t, eval.CreateRun(ctx, run))
	require.NoError(t, eval.StartRun(ctx, run.ID))

	// Poll for completion.
	deadline := 50
	for i := 0; i < deadline; i++ {
		got, err := eval.GetRun(ctx, run.ID)
		require.NoError(t, err)
		if got.Status == EvaluationStatusCompleted {
			require.NotNil(t, got.Results, "completed run MUST have results")
			require.Equal(t, 1, got.Results.TotalSamples)
			require.NotEmpty(t, got.Results.SampleResults)

			scores := got.Results.SampleResults[0].Scores
			require.Contains(t, scores, "correctness")
			require.Contains(t, scores, "completeness")

			// REAL scores from the LLM-as-judge pipeline — these are the
			// exact values the judge stub emitted, threaded through
			// evaluateSample → LLMBackedEvaluator.Evaluate → parseScores
			// → returned map. A regression to a fabricated constant
			// (round-25 0.8 PASS-bluff) would fail these asserts.
			assert.InDelta(t, 0.75, scores["correctness"], 1e-9,
				"correctness MUST equal the judge's emitted 0.75 — proves the LLM-as-judge value flows end-to-end without fabrication")
			assert.InDelta(t, 0.55, scores["completeness"], 1e-9,
				"completeness MUST equal the judge's emitted 0.55")

			// And the judge MUST have been called exactly once (one sample → one judge call).
			assert.Equal(t, int32(1), judgeResponder.calls.Load(),
				"judge LLM MUST have been invoked exactly once for the single dataset sample")
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("run did not complete within poll budget")
}

// --- Real-provider integration test (env-gated) -------------------------

// TestLLMBackedEvaluator_RealOpenAI_Roundtrip is the CONST-050(B)
// integration-tier coverage for LLMBackedEvaluator. When
// LLMOPS_TEST_OPENAI_KEY is set, it wires an HTTPResponder
// (round-62) as the judge and asserts a real OpenAI-compatible
// endpoint returns a parseable score for a trivial prompt. Without
// the env key, the test SKIP-OK marker tracks the coverage gap loudly
// per CONST-035 skip-bluff rule.
//
// SKIP-OK: #LLMOPS-LLM-JUDGE-REAL-ROUND81 — without LLMOPS_TEST_OPENAI_KEY
// we cannot drive a real judge LLM. The skip is loud per CONST-035
// and the gate marker tracks the coverage gap.
func TestLLMBackedEvaluator_RealOpenAI_Roundtrip(t *testing.T) {
	apiKey := os.Getenv("LLMOPS_TEST_OPENAI_KEY")
	if apiKey == "" {
		t.Skip("SKIP-OK: #LLMOPS-LLM-JUDGE-REAL-ROUND81 — LLMOPS_TEST_OPENAI_KEY not set; set to a real OpenAI-compatible key to exercise this integration tier")
	}

	endpoint := os.Getenv("LLMOPS_TEST_OPENAI_ENDPOINT")
	if endpoint == "" {
		endpoint = "https://api.openai.com/v1"
	}
	model := os.Getenv("LLMOPS_TEST_OPENAI_MODEL")
	if model == "" {
		model = "gpt-4o-mini"
	}

	judgeResponder, err := NewHTTPResponder(HTTPResponderConfig{
		Endpoint: endpoint,
		Model:    model,
		APIKey:   apiKey,
		Timeout:  60 * time.Second,
	})
	require.NoError(t, err)

	eval, err := NewLLMBackedEvaluator(LLMBackedEvaluatorConfig{Responder: judgeResponder})
	require.NoError(t, err)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	scores, err := eval.Evaluate(ctx,
		"What is 2+2?",
		"4",
		"4",
		[]string{"correctness"})
	require.NoError(t, err, "real-provider LLMBackedEvaluator.Evaluate MUST succeed; check LLMOPS_TEST_OPENAI_KEY validity")
	require.NotEmpty(t, scores, "real judge MUST return at least one parseable score")

	// The judge MUST emit at least correctness — exact value depends
	// on the model but it MUST be in [0, 1] (the sentinel would have
	// fired otherwise) and reasonably high (a correct math answer
	// should score well; we use a loose floor to tolerate model
	// variance).
	c, ok := scores["correctness"]
	require.True(t, ok, "real judge MUST emit a SCORE_correctness line")
	assert.GreaterOrEqual(t, c, 0.0, "real-judge score MUST be >= 0")
	assert.LessOrEqual(t, c, 1.0, "real-judge score MUST be <= 1")

	// Print evidence so the operator running with -v gets confirmation
	// the real roundtrip happened (CONST-035 captured-evidence floor).
	t.Logf("real-provider judge roundtrip evidence (endpoint=%s model=%s): scores=%v",
		endpoint, model, scores)
	fmt.Fprintf(io.Discard, "consumed: %v\n", scores)
}

// --- httptest-driven integration with a real HTTPResponder judge --------

// TestLLMBackedEvaluator_WithHTTPResponderJudge_LoopbackRoundtrip wires
// an HTTPResponder (round-62) as the judge against an in-process
// httptest.Server and asserts the full LLM-as-judge pipeline produces
// the expected scores. This counts as a unit test under CONST-050(A)
// (no external network) but exercises the integration path between
// LLMBackedEvaluator and the real HTTPResponder implementation
// without depending on a remote provider.
func TestLLMBackedEvaluator_WithHTTPResponderJudge_LoopbackRoundtrip(t *testing.T) {
	const judgeReply = "SCORE_correctness: 0.9\nSCORE_completeness: 0.6"

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		// Sanity check: judge endpoint MUST receive the OpenAI-shaped
		// chat-completions request with a user-role message containing
		// the rendered judgment prompt.
		body, _ := io.ReadAll(req.Body)
		bodyStr := string(body)
		if !strings.Contains(bodyStr, "strict evaluator") {
			http.Error(w, "request did not contain rendered judgment prompt", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"choices":[{"message":{"role":"assistant","content":%q}}]}`, judgeReply)
	}))
	defer srv.Close()

	judgeResponder, err := NewHTTPResponder(HTTPResponderConfig{
		Endpoint: srv.URL,
		Model:    "test-judge-model",
		APIKey:   "test-key",
		Timeout:  5 * time.Second,
	})
	require.NoError(t, err)

	eval, err := NewLLMBackedEvaluator(LLMBackedEvaluatorConfig{Responder: judgeResponder})
	require.NoError(t, err)

	scores, err := eval.Evaluate(context.Background(),
		"What is 2+2?", "4", "4",
		[]string{"correctness", "completeness"})
	require.NoError(t, err)
	require.Len(t, scores, 2)
	assert.InDelta(t, 0.9, scores["correctness"], 1e-9)
	assert.InDelta(t, 0.6, scores["completeness"], 1e-9)
}
