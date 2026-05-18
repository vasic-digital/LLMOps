package llmops

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Round-75 §11.4 anti-bluff close-out of the round-25
// ErrEvaluatorNotConfigured sentinel. The tests below exercise the
// concrete HeuristicEvaluator implementation introduced in
// heuristic_evaluator.go.
//
// Anti-bluff posture: every test that claims HeuristicEvaluator "works"
// either (a) drives concrete response/expected inputs and asserts on
// the exact ordering / sign / magnitude relationships between scores,
// or (b) is a paired-mutation test that asserts DIFFERENT inputs
// produce DIFFERENT scores. No test asserts only "no error returned"
// — every PASS path asserts a concrete property the algorithm derived
// from the actual input.
//
// Load-bearing test: TestHeuristicEvaluator_NotFabricated. If that test
// passes when the implementation has been replaced by a hardcoded
// constant, the test is bluffing and MUST be tightened. Conversely if
// it fails on the correct implementation, the algorithm has lost
// input-dependency and MUST be fixed.

// --- Construction tests --------------------------------------------------

func TestNewHeuristicEvaluator_EmptyConfig_UsesDefaults(t *testing.T) {
	e, err := NewHeuristicEvaluator(HeuristicEvaluatorConfig{})
	require.NoError(t, err)
	require.NotNil(t, e)

	// Default profile must normalise to sum-1.0 across the three sub-scores.
	got := e.weights.jaccard + e.weights.keywordCoverage + e.weights.lengthRatio
	assert.InDelta(t, 1.0, got, 1e-9, "default profile weights MUST sum to 1.0")
}

func TestNewHeuristicEvaluator_CustomWeights_Normalised(t *testing.T) {
	e, err := NewHeuristicEvaluator(HeuristicEvaluatorConfig{
		Weights: map[string]float64{
			"jaccard":          2.0,
			"keyword_coverage": 1.0,
			"length_ratio":     1.0,
		},
	})
	require.NoError(t, err)
	assert.InDelta(t, 0.5, e.weights.jaccard, 1e-9)
	assert.InDelta(t, 0.25, e.weights.keywordCoverage, 1e-9)
	assert.InDelta(t, 0.25, e.weights.lengthRatio, 1e-9)
}

func TestNewHeuristicEvaluator_NegativeWeights_ClampedToZero(t *testing.T) {
	e, err := NewHeuristicEvaluator(HeuristicEvaluatorConfig{
		Weights: map[string]float64{
			"jaccard":          1.0,
			"keyword_coverage": -5.0,
			"length_ratio":     1.0,
		},
	})
	require.NoError(t, err)
	assert.InDelta(t, 0.5, e.weights.jaccard, 1e-9)
	assert.InDelta(t, 0.0, e.weights.keywordCoverage, 1e-9)
	assert.InDelta(t, 0.5, e.weights.lengthRatio, 1e-9)
}

func TestNewHeuristicEvaluator_AllZeroWeights_FallsBackToDefault(t *testing.T) {
	e, err := NewHeuristicEvaluator(HeuristicEvaluatorConfig{
		Weights: map[string]float64{"jaccard": 0, "keyword_coverage": 0, "length_ratio": 0},
	})
	require.NoError(t, err)
	// Fallback default is equal-thirds.
	assert.InDelta(t, 1.0/3.0, e.weights.jaccard, 1e-9)
	assert.InDelta(t, 1.0/3.0, e.weights.keywordCoverage, 1e-9)
	assert.InDelta(t, 1.0/3.0, e.weights.lengthRatio, 1e-9)
}

func TestNewHeuristicEvaluator_StringRedactsNothingButCompact(t *testing.T) {
	e, err := NewHeuristicEvaluator(HeuristicEvaluatorConfig{})
	require.NoError(t, err)
	s := e.String()
	assert.Contains(t, s, "HeuristicEvaluator")
	assert.Contains(t, s, "profiles=")
	// %#v mirrors %s.
	assert.Equal(t, s, e.GoString())
}

// --- Empty-input sentinel tests ------------------------------------------

func TestHeuristicEvaluator_EmptyActual_ReturnsSentinel(t *testing.T) {
	e, err := NewHeuristicEvaluator(HeuristicEvaluatorConfig{})
	require.NoError(t, err)
	scores, err := e.Evaluate(context.Background(), "prompt", "", "expected output", []string{"correctness"})
	require.Error(t, err)
	assert.Nil(t, scores)
	assert.ErrorIs(t, err, ErrHeuristicEvaluatorMissingInput,
		"empty actual MUST return ErrHeuristicEvaluatorMissingInput, NEVER a fabricated 0.0")
}

func TestHeuristicEvaluator_EmptyExpected_ReturnsSentinel(t *testing.T) {
	e, err := NewHeuristicEvaluator(HeuristicEvaluatorConfig{})
	require.NoError(t, err)
	scores, err := e.Evaluate(context.Background(), "prompt", "actual response", "", []string{"correctness"})
	require.Error(t, err)
	assert.Nil(t, scores)
	assert.ErrorIs(t, err, ErrHeuristicEvaluatorMissingInput,
		"empty expected MUST return ErrHeuristicEvaluatorMissingInput")
}

func TestHeuristicEvaluator_WhitespaceOnlyInputs_ReturnSentinel(t *testing.T) {
	e, err := NewHeuristicEvaluator(HeuristicEvaluatorConfig{})
	require.NoError(t, err)
	_, err = e.Evaluate(context.Background(), "prompt", "  \t\n  ", "expected", []string{"correctness"})
	assert.ErrorIs(t, err, ErrHeuristicEvaluatorMissingInput)
	_, err = e.Evaluate(context.Background(), "prompt", "actual", "  \t\n  ", []string{"correctness"})
	assert.ErrorIs(t, err, ErrHeuristicEvaluatorMissingInput)
}

func TestHeuristicEvaluator_ContextCancelled_ReturnsError(t *testing.T) {
	e, err := NewHeuristicEvaluator(HeuristicEvaluatorConfig{})
	require.NoError(t, err)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = e.Evaluate(ctx, "prompt", "actual", "expected", []string{"correctness"})
	require.Error(t, err)
	assert.ErrorIs(t, err, context.Canceled)
}

// --- Scoring property tests ----------------------------------------------

func TestHeuristicEvaluator_ExactMatch_HighScore(t *testing.T) {
	e, err := NewHeuristicEvaluator(HeuristicEvaluatorConfig{})
	require.NoError(t, err)
	text := "the quick brown fox jumps over the lazy dog"
	scores, err := e.Evaluate(context.Background(), "prompt", text, text, []string{"correctness", "completeness", "conciseness"})
	require.NoError(t, err)
	for metric, score := range scores {
		assert.InDelta(t, 1.0, score, 1e-9,
			"exact match MUST score 1.0 on every metric (got %.6f for %q)", score, metric)
	}
}

func TestHeuristicEvaluator_TotalMismatch_LowScore(t *testing.T) {
	e, err := NewHeuristicEvaluator(HeuristicEvaluatorConfig{})
	require.NoError(t, err)
	// Token sets share zero elements but length ratio is close to 1
	// (both 4 tokens). Expected score is bounded above by the length
	// ratio's contribution to the profile.
	scores, err := e.Evaluate(context.Background(), "prompt",
		"alpha beta gamma delta",
		"epsilon zeta eta theta",
		[]string{"correctness"})
	require.NoError(t, err)
	corr := scores["correctness"]
	// "correctness" profile: jaccard=0.60, keyword_coverage=0.25, length_ratio=0.15
	// jaccard=0, coverage=0, length_ratio=1.0 (4/4) => 0.15.
	assert.InDelta(t, 0.15, corr, 1e-6,
		"disjoint token sets with equal length MUST score = length_ratio_weight * 1.0 = 0.15 on 'correctness' (got %.6f)", corr)
}

func TestHeuristicEvaluator_PartialMatch_IntermediateScore(t *testing.T) {
	e, err := NewHeuristicEvaluator(HeuristicEvaluatorConfig{})
	require.NoError(t, err)
	// Response shares 2/4 tokens with expected; lengths are 4 vs 4 so
	// length ratio = 1.0; coverage = 2/4 = 0.5; jaccard = 2/(4+4-2) = 1/3.
	scores, err := e.Evaluate(context.Background(), "prompt",
		"alpha beta epsilon zeta",
		"alpha beta gamma delta",
		[]string{"correctness"})
	require.NoError(t, err)
	corr := scores["correctness"]
	// 0.60 * 0.333... + 0.25 * 0.5 + 0.15 * 1.0 = 0.2 + 0.125 + 0.15 = 0.475
	assert.InDelta(t, 0.475, corr, 1e-3,
		"partial match expected ~0.475 on 'correctness' (got %.6f)", corr)
	// Bounds sanity: strictly between disjoint (0.15) and exact (1.0).
	assert.Greater(t, corr, 0.15)
	assert.Less(t, corr, 1.0)
}

func TestHeuristicEvaluator_LengthPenalty(t *testing.T) {
	e, err := NewHeuristicEvaluator(HeuristicEvaluatorConfig{})
	require.NoError(t, err)
	// Same token content, vastly different lengths: response is 4
	// tokens, expected is 4 tokens repeated 5 times = 20 tokens. On
	// the 'conciseness' metric (length-ratio-heavy), this MUST score
	// significantly LOWER than an identical-length response.
	expectedLong := strings.Repeat("alpha beta gamma delta ", 5)
	shortIdentical := "alpha beta gamma delta"

	longScores, err := e.Evaluate(context.Background(), "p", expectedLong, expectedLong, []string{"conciseness"})
	require.NoError(t, err)
	shortVsLong, err := e.Evaluate(context.Background(), "p", shortIdentical, expectedLong, []string{"conciseness"})
	require.NoError(t, err)

	assert.Greater(t, longScores["conciseness"], shortVsLong["conciseness"],
		"matching length MUST score higher than length-mismatched on 'conciseness' (long-match=%.4f, mismatch=%.4f)",
		longScores["conciseness"], shortVsLong["conciseness"])

	// length_ratio = 4/20 = 0.20; coverage = 4/4 = 1.0 (token SET coverage);
	// jaccard = 4/4 = 1.0 (sets are equal even though counts differ).
	// 'conciseness' profile: j=0.25 kc=0.15 lr=0.60.
	// Expected = 0.25*1.0 + 0.15*1.0 + 0.60*0.20 = 0.25 + 0.15 + 0.12 = 0.52
	assert.InDelta(t, 0.52, shortVsLong["conciseness"], 1e-3,
		"length-mismatched 'conciseness' expected ~0.52 (got %.6f)", shortVsLong["conciseness"])
}

func TestHeuristicEvaluator_StructuralMarkers_BoostScore(t *testing.T) {
	e, err := NewHeuristicEvaluator(HeuristicEvaluatorConfig{})
	require.NoError(t, err)
	// Expected references 4 specific tokens; first response covers
	// all of them (plus filler); second covers none. On
	// 'completeness' (coverage-heavy), the first MUST score much higher.
	expected := "users tasks projects roles"
	withMarkers := "this system has users tasks projects and roles all wired together end to end"
	withoutMarkers := "this system is incomplete and unwired so far so good but nothing real"

	withScores, err := e.Evaluate(context.Background(), "p", withMarkers, expected, []string{"completeness"})
	require.NoError(t, err)
	withoutScores, err := e.Evaluate(context.Background(), "p", withoutMarkers, expected, []string{"completeness"})
	require.NoError(t, err)

	assert.Greater(t, withScores["completeness"], withoutScores["completeness"],
		"response containing all expected keywords MUST score higher on 'completeness' (with=%.4f, without=%.4f)",
		withScores["completeness"], withoutScores["completeness"])
}

func TestHeuristicEvaluator_OverallClamped_ToZeroToOne(t *testing.T) {
	e, err := NewHeuristicEvaluator(HeuristicEvaluatorConfig{})
	require.NoError(t, err)
	cases := []struct {
		name             string
		response, expect string
	}{
		{"tiny-vs-huge", "x", strings.Repeat("alpha beta gamma delta ", 100)},
		{"huge-vs-tiny", strings.Repeat("alpha beta gamma delta ", 100), "x"},
		{"identical-single", "alpha", "alpha"},
		{"punctuation-vs-text", "!!!", "alpha beta gamma"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			scores, err := e.Evaluate(context.Background(), "p", tc.response, tc.expect,
				[]string{"correctness", "completeness", "conciseness"})
			require.NoError(t, err, "tc=%s", tc.name)
			for metric, s := range scores {
				assert.GreaterOrEqual(t, s, 0.0, "%s/%s MUST be >= 0.0 (got %.6f)", tc.name, metric, s)
				assert.LessOrEqual(t, s, 1.0, "%s/%s MUST be <= 1.0 (got %.6f)", tc.name, metric, s)
			}
		})
	}
}

func TestHeuristicEvaluator_PerMetricProfiles_ProduceDifferentScores(t *testing.T) {
	e, err := NewHeuristicEvaluator(HeuristicEvaluatorConfig{})
	require.NoError(t, err)
	// Same inputs, different metric names. Because the three built-in
	// profile families weight the three sub-scores differently, the
	// returned scores MUST differ unless all three sub-scores happen
	// to be equal (a corner case we avoid here by construction).
	scores, err := e.Evaluate(context.Background(), "p",
		"alpha beta delta epsilon",
		"alpha beta gamma",
		[]string{"correctness", "completeness", "conciseness"})
	require.NoError(t, err)

	corr := scores["correctness"]
	comp := scores["completeness"]
	conc := scores["conciseness"]

	// At least one pair of metric scores MUST differ — proving the
	// per-metric profile mechanism actually produces variation.
	pairsEqual := approxEqual(corr, comp, 1e-9) && approxEqual(comp, conc, 1e-9) && approxEqual(corr, conc, 1e-9)
	assert.False(t, pairsEqual,
		"per-metric profile variation MUST produce DIFFERENT scores for DIFFERENT metric names from the same inputs (corr=%.6f comp=%.6f conc=%.6f)",
		corr, comp, conc)
}

func TestHeuristicEvaluator_UnknownMetric_UsesDefaultProfile(t *testing.T) {
	e, err := NewHeuristicEvaluator(HeuristicEvaluatorConfig{})
	require.NoError(t, err)
	scores, err := e.Evaluate(context.Background(), "p",
		"alpha beta gamma",
		"alpha beta gamma",
		[]string{"some_made_up_metric"})
	require.NoError(t, err)
	// Exact match => default profile aggregate = 1.0 regardless of weights.
	assert.InDelta(t, 1.0, scores["some_made_up_metric"], 1e-9)
}

func TestHeuristicEvaluator_NoMetricsRequested_ReturnsOverall(t *testing.T) {
	e, err := NewHeuristicEvaluator(HeuristicEvaluatorConfig{})
	require.NoError(t, err)
	scores, err := e.Evaluate(context.Background(), "p", "alpha beta", "alpha gamma", nil)
	require.NoError(t, err)
	require.Contains(t, scores, "_overall",
		"empty metrics list MUST return at least the synthetic _overall key (vs an empty map indistinguishable from 'scored zero on nothing')")
	assert.Greater(t, scores["_overall"], 0.0)
	assert.Less(t, scores["_overall"], 1.0)
}

// --- LOAD-BEARING anti-bluff test ---------------------------------------

// TestHeuristicEvaluator_NotFabricated is the §11.4 anti-bluff load-bearing
// guard: it proves the evaluator's scores are DERIVED from the actual
// inputs by changing the response by exactly one character / one token
// and asserting AT LEAST ONE score changes by a measurable amount.
//
// If this test ever passes when the Evaluate implementation has been
// replaced by `return map[string]float64{metric: 0.8}, nil` (the
// round-25 PASS-bluff that was removed), the test is itself a bluff
// and MUST be tightened.
func TestHeuristicEvaluator_NotFabricated(t *testing.T) {
	e, err := NewHeuristicEvaluator(HeuristicEvaluatorConfig{})
	require.NoError(t, err)

	expected := "the user account was created successfully"
	baseline := "the user account was created successfully"
	mutatedToken := "the user account was created unsuccessfully" // 1 token swap
	mutatedChar := "the user account was created successfullx"   // 1 char swap

	metrics := []string{"correctness", "completeness", "conciseness"}

	baseScores, err := e.Evaluate(context.Background(), "p", baseline, expected, metrics)
	require.NoError(t, err)
	tokScores, err := e.Evaluate(context.Background(), "p", mutatedToken, expected, metrics)
	require.NoError(t, err)
	charScores, err := e.Evaluate(context.Background(), "p", mutatedChar, expected, metrics)
	require.NoError(t, err)

	// PROOF 1: token-swap MUST change at least one metric score
	// measurably. With 1-token swap on a 6-token response the jaccard
	// drops from 1.0 to 5/7≈0.714 (and coverage drops to 5/6≈0.833)
	// so EVERY metric is affected.
	tokDiff := false
	for _, m := range metrics {
		if !approxEqual(baseScores[m], tokScores[m], 1e-6) {
			tokDiff = true
			t.Logf("token-mutation metric %q: baseline=%.6f mutated=%.6f delta=%.6f",
				m, baseScores[m], tokScores[m], baseScores[m]-tokScores[m])
		}
	}
	require.True(t, tokDiff,
		"§11.4 anti-bluff invariant violated: token-level input change produced ZERO score change across all metrics — Evaluate is fabricating a constant rather than deriving from input")

	// PROOF 2: char-swap (which produces a NEW token after tokenisation
	// because tokenise splits on non-letters) MUST also change at least
	// one metric score. "successfullx" is a different token from
	// "successfully" so the sets diverge.
	charDiff := false
	for _, m := range metrics {
		if !approxEqual(baseScores[m], charScores[m], 1e-6) {
			charDiff = true
			t.Logf("char-mutation metric %q: baseline=%.6f mutated=%.6f delta=%.6f",
				m, baseScores[m], charScores[m], baseScores[m]-charScores[m])
		}
	}
	require.True(t, charDiff,
		"§11.4 anti-bluff invariant violated: single-char input change produced ZERO score change across all metrics — Evaluate is fabricating a constant rather than deriving from input")

	// PROOF 3: the baseline (exact match) MUST score strictly higher
	// than the mutated variants on every metric. If a fabricator
	// returned a constant, this would FAIL because constant == constant.
	for _, m := range metrics {
		assert.Greater(t, baseScores[m], tokScores[m],
			"§11.4: exact-match baseline MUST score strictly higher than token-mutated on %q (base=%.6f tok=%.6f)",
			m, baseScores[m], tokScores[m])
		assert.Greater(t, baseScores[m], charScores[m],
			"§11.4: exact-match baseline MUST score strictly higher than char-mutated on %q (base=%.6f char=%.6f)",
			m, baseScores[m], charScores[m])
	}

	// PROOF 4 (defensive): no score equals the round-25 fabricated
	// constant 0.8 by coincidence on these specific inputs. (Not a
	// tight invariant — the algorithm could legitimately produce 0.8 on
	// other inputs — but on THESE inputs the math doesn't land there.)
	for _, m := range metrics {
		assert.False(t, approxEqual(0.8, tokScores[m], 1e-6),
			"§11.4: round-25 fabricated constant 0.8 surfaced for metric %q on mutated input (got %.6f) — possible regression to placeholder bluff",
			m, tokScores[m])
	}
}

// --- Sentinel distinctness tests ----------------------------------------

func TestHeuristicEvaluator_Sentinels_AreDistinct(t *testing.T) {
	// Each sentinel MUST be distinguishable via errors.Is from every
	// other sentinel exported by this package, so callers can branch
	// on failure mode without resorting to string matching.
	all := []error{
		ErrHeuristicEvaluatorMissingInput,
		ErrHeuristicEvaluatorScoreOutOfRange,
		ErrEvaluatorNotConfigured,
		ErrLLMResponderNotConfigured,
		ErrHTTPResponderEndpointNotConfigured,
		ErrHTTPResponderModelNotConfigured,
		ErrHTTPResponderRequestFailed,
		ErrHTTPResponderResponseInvalid,
	}
	for i, a := range all {
		for j, b := range all {
			if i == j {
				assert.True(t, errors.Is(a, b),
					"errors.Is must be reflexive for sentinel %d", i)
				continue
			}
			assert.False(t, errors.Is(a, b),
				"sentinels %d and %d MUST be distinct under errors.Is", i, j)
		}
	}
}

func TestHeuristicEvaluator_Sentinels_DistinctFromRound25(t *testing.T) {
	// Round-75 sentinels MUST be distinct from the round-25 evaluator
	// sentinel they complement (one fires when no evaluator wired; the
	// new ones fire when the heuristic evaluator IS wired but receives
	// bad input or hits a defensive guard).
	assert.False(t, errors.Is(ErrHeuristicEvaluatorMissingInput, ErrEvaluatorNotConfigured))
	assert.False(t, errors.Is(ErrHeuristicEvaluatorScoreOutOfRange, ErrEvaluatorNotConfigured))
	assert.False(t, errors.Is(ErrEvaluatorNotConfigured, ErrHeuristicEvaluatorMissingInput))
	assert.False(t, errors.Is(ErrEvaluatorNotConfigured, ErrHeuristicEvaluatorScoreOutOfRange))
}

// --- Interface conformance + end-to-end wire-through --------------------

func TestHeuristicEvaluator_ImplementsLLMEvaluator(t *testing.T) {
	// Compile-time assertion via interface assignment.
	var _ LLMEvaluator = (*HeuristicEvaluator)(nil)
}

// TestEvaluator_WithHeuristicEvaluator_EndToEnd is the closing assertion
// for round 75: the InMemoryContinuousEvaluator, when wired with a
// real HeuristicEvaluator AND a real responder (here a deterministic
// stub LLMResponderFunc to keep the test in the unit tier per
// CONST-050(A)), produces SampleResults with REAL per-metric scores
// derived from the response vs expected — NOT the fabricated 0.8
// constant the round-25 bluff used to produce.
func TestEvaluator_WithHeuristicEvaluator_EndToEnd(t *testing.T) {
	hLogger := logrus.New()
	hLogger.SetLevel(logrus.WarnLevel) // suppress info noise in test output

	hEval, err := NewHeuristicEvaluator(HeuristicEvaluatorConfig{})
	require.NoError(t, err)

	eval := NewInMemoryContinuousEvaluator(hEval, nil, nil, hLogger)

	// Wire a stub responder that echoes a partial-match string. We use
	// LLMResponderFunc to keep this test in the unit tier; the
	// integration tier uses HTTPResponder against a real provider.
	eval.SetResponder(LLMResponderFunc(func(_ context.Context, prompt string) (string, error) {
		// Return a response that shares ~half the expected tokens with
		// the expected output below. Deterministic for reproducibility.
		return "alpha beta epsilon zeta", nil
	}))

	ctx := context.Background()
	dataset := &Dataset{Name: "round75-e2e", Type: DatasetTypeGolden}
	require.NoError(t, eval.CreateDataset(ctx, dataset))
	require.NoError(t, eval.AddSamples(ctx, dataset.ID, []*DatasetSample{
		{Input: "describe", ExpectedOutput: "alpha beta gamma delta"},
	}))

	run := &EvaluationRun{
		Name:    "round75-e2e-run",
		Dataset: dataset.ID,
		Metrics: []string{"correctness", "completeness"},
	}
	require.NoError(t, eval.CreateRun(ctx, run))
	require.NoError(t, eval.StartRun(ctx, run.ID))

	// Poll for completion (StartRun is async per CLAUDE.md gotcha #1).
	// Budget: 50 iterations * 20ms = 1s wall-clock max.
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

			// REAL scores: must be strictly between 0 and 1 (partial
			// match), and the two metrics MUST differ because of the
			// per-metric profile weighting on the SAME sub-scores.
			assert.Greater(t, scores["correctness"], 0.0)
			assert.Less(t, scores["correctness"], 1.0)
			assert.Greater(t, scores["completeness"], 0.0)
			assert.Less(t, scores["completeness"], 1.0)
			assert.NotEqual(t, scores["correctness"], scores["completeness"],
				"per-metric profile MUST yield distinct scores for distinct metrics on the same inputs")

			// CRITICAL: NEVER the round-25 fabricated 0.8 constant.
			assert.False(t, approxEqual(0.8, scores["correctness"], 1e-6),
				"round-25 fabricated 0.8 constant surfaced for correctness (got %.6f) — regression to PASS-bluff",
				scores["correctness"])
			assert.False(t, approxEqual(0.8, scores["completeness"], 1e-6),
				"round-25 fabricated 0.8 constant surfaced for completeness (got %.6f) — regression to PASS-bluff",
				scores["completeness"])
			return
		}
		// Yield + sleep briefly so the executeRun goroutine gets CPU.
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("run did not complete within poll budget")
}

// --- helpers -------------------------------------------------------------

func approxEqual(a, b, eps float64) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d <= eps
}
