package llmops

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// LLMBackedEvaluator is a concrete LLMEvaluator implementation that scores
// the LLM response against the expected output using a SECOND LLM as the
// judge ("LLM-as-judge" pattern). It is the round-81 §11.4 anti-bluff
// close-out of the round-75 deferred item: round 75 shipped the
// deterministic HeuristicEvaluator (Token Jaccard + keyword coverage +
// length ratio) for callers who want a pure-Go, no-network evaluator;
// round 81 ships THIS LLM-as-judge variant for callers who want richer
// semantic judgment on free-form responses where lexical overlap is
// a poor proxy for correctness (open-ended QA, creative tasks,
// summarisation).
//
// Both evaluators coexist as sibling concrete impls of the
// LLMEvaluator interface and the round-25 ErrEvaluatorNotConfigured
// sentinel keeps its meaning (fires when no LLMEvaluator wired at all).
// Round-75 sentinels are HeuristicEvaluator-specific; the three new
// sentinels exported by THIS file are LLMBackedEvaluator-specific.
//
// Round 62 (commit f017e43) introduced the LLMResponder interface +
// HTTPResponder concrete impl used by InMemoryContinuousEvaluator to
// produce the response under test. LLMBackedEvaluator REUSES that
// same LLMResponder interface — the caller wires an HTTPResponder (or
// any other LLMResponder implementation) configured against their
// chosen judge model, and LLMBackedEvaluator dispatches the judgment
// prompt through it. A common deployment is "responder under test =
// small fast model; judge responder = larger frontier model" — but
// the two can also be the same model, the same endpoint with a
// different system prompt, or a local-vs-remote split.
//
// CONST-035 / Article XI §11.9 / CONST-050(A) load-bearing invariants
// (this implementation refuses to bluff three ways):
//
//  1. **Refuses to fabricate scores on unparseable judge response** —
//     if the judge LLM returns text that contains zero parseable
//     SCORE_<metric> lines, Evaluate returns
//     ErrLLMBackedEvaluatorResponseUnparseable. The round-25 PASS-bluff
//     equivalent would be returning 0.5 (or any fabricated constant)
//     for every requested metric — that would let the run report
//     completion with meaningless scores. The sentinel surfaces the
//     judge-side failure loudly so callers can branch (retry, switch
//     to HeuristicEvaluator fallback, abort the run).
//  2. **Refuses to silently clamp out-of-range scores** — if the judge
//     returns SCORE_correctness: 1.5 (or -0.2, or 3.7), Evaluate
//     returns ErrLLMBackedEvaluatorScoreOutOfRange rather than
//     clamping to [0, 1] without telling the caller. A judge that
//     emits out-of-range scores is either broken or has been misprompted
//     — silent clamping would hide the bug and let it accumulate.
//  3. **Refuses construction with nil responder** — NewLLMBackedEvaluator
//     returns ErrLLMBackedEvaluatorResponderNotProvided when called with
//     a nil Responder. Construction-time failure beats runtime failure.
//
// The paired-mutation test TestLLMBackedEvaluator_NotFabricated proves
// the input-dependency invariant: two different judge responses MUST
// produce two different score maps. If that test passes against a
// hardcoded-constant implementation, the test is bluffing and MUST
// be tightened.
//
// CONST-050(A): production code may import and construct
// LLMBackedEvaluator. The unit tests in llm_backed_evaluator_test.go
// drive it with a deterministic stub LLMResponderFunc (no external
// network) which counts as a unit test under CONST-050(A); the
// env-gated integration test
// TestLLMBackedEvaluator_RealOpenAI_Roundtrip exercises it against a
// real OpenAI-compatible endpoint when LLMOPS_TEST_OPENAI_KEY is set,
// satisfying CONST-050(B) coverage for the integration tier.
type LLMBackedEvaluator struct {
	responder              LLMResponder
	judgmentPromptTemplate string
	defaultMetrics         []string
	clampOnRangeError      bool
}

// LLMBackedEvaluatorConfig is the constructor input for NewLLMBackedEvaluator.
type LLMBackedEvaluatorConfig struct {
	// Responder is the LLM used as the JUDGE. REQUIRED — a nil
	// Responder causes NewLLMBackedEvaluator to return
	// ErrLLMBackedEvaluatorResponderNotProvided. Typically constructed
	// via NewHTTPResponder(HTTPResponderConfig{...}) targeting a
	// strong model with cheap latency (frontier model with low
	// max_tokens). Can be any LLMResponder implementation —
	// HTTPResponder, LLMResponderFunc, or a custom type wrapping a
	// non-OpenAI-shaped SDK.
	Responder LLMResponder

	// JudgmentPromptTemplate optionally overrides the default judgment
	// prompt sent to the judge LLM. The template MUST contain the four
	// placeholder tokens {prompt}, {response}, {expected}, and
	// {metrics} — they are replaced via strings.ReplaceAll before
	// dispatch. Empty string => the documented default template is
	// used (see defaultJudgmentPromptTemplate below).
	//
	// The template MUST instruct the judge to emit one
	// "SCORE_<metric>: <0.0-1.0>" line per requested metric — the
	// score parser is regex-based and depends on that shape. A
	// template that produces a different output shape will trigger
	// ErrLLMBackedEvaluatorResponseUnparseable on every Evaluate call.
	JudgmentPromptTemplate string

	// DefaultMetrics is the metric list used when Evaluate is invoked
	// with an empty metrics slice. Empty/nil => the package default
	// ["correctness", "completeness", "conciseness", "factuality"] is
	// used (the four metric families HeuristicEvaluator also defines
	// per-profile weights for, kept symmetric so a caller can swap
	// evaluators without re-thinking metric names).
	DefaultMetrics []string

	// ClampOnRangeError controls the out-of-range behaviour. Default
	// (false) is "refuse" — return ErrLLMBackedEvaluatorScoreOutOfRange
	// surfacing the judge's bug loudly. Setting to true changes
	// behaviour to "clamp to [0, 1] silently" for callers who would
	// rather accept the score than abort the run. The default is
	// REFUSE because silent clamping hides judge regressions and is
	// a PASS-bluff sibling pattern.
	ClampOnRangeError bool
}

// defaultJudgmentPromptTemplate is the prompt sent to the judge LLM when
// the caller does not supply a custom template. Placeholders:
//
//   - {prompt}   — the original prompt under evaluation
//   - {response} — the model-under-test's response to that prompt
//   - {expected} — the gold-standard expected output
//   - {metrics}  — comma-separated list of metric names to score
//
// The "Reply ONLY with score lines in the format SCORE_<metric>: <score>"
// instruction is load-bearing — it makes the parser's regex contract
// explicit to the judge model and reduces conversational chatter
// that the parser would otherwise have to skip.
const defaultJudgmentPromptTemplate = `You are a strict evaluator. Score the RESPONSE 0.0 to 1.0 against the EXPECTED output for each requested metric.

PROMPT:
{prompt}

RESPONSE:
{response}

EXPECTED:
{expected}

METRICS to score: {metrics}

Reply ONLY with score lines in the exact format below, one per metric, with no prose, no headers, no markdown:
SCORE_correctness: 0.85
SCORE_completeness: 0.72

Each score MUST be a decimal in [0.0, 1.0] (inclusive). 1.0 means perfect match; 0.0 means completely wrong; intermediate values reflect partial credit.`

// scorePattern matches "SCORE_<metric>: <number>" lines emitted by the
// judge. Allows leading whitespace, trailing comment text, and any
// case for the metric name. The number capture accepts "0.85", ".85",
// "1", "1.0", "-0.2" (rejected later by range check), "3.7" (also
// rejected). The non-greedy `[0-9.]+` is intentionally permissive
// at the parse level; numeric validation is the next stage.
//
// Multiline flag (?m) lets ^ and $ anchor against line boundaries
// inside the multi-line judge response.
var scorePattern = regexp.MustCompile(`(?m)^\s*SCORE_([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*([+-]?[0-9]*\.?[0-9]+)`)

// Default metric names used when the caller passes an empty metrics
// slice to Evaluate. Kept aligned with HeuristicEvaluator's named
// profile families (correctness / completeness / conciseness) plus
// factuality which is unique to LLM-as-judge (lexical heuristics
// cannot judge factual correctness).
var defaultMetricFamilies = []string{"correctness", "completeness", "conciseness", "factuality"}

// Sentinel errors covering the three failure modes Evaluate +
// NewLLMBackedEvaluator expose. All three are stable contract surfaces —
// consumers MAY errors.Is() against them to branch on failure mode.
var (
	// ErrLLMBackedEvaluatorResponderNotProvided fires from
	// NewLLMBackedEvaluator when Responder is nil. Distinct from
	// the round-25 ErrLLMResponderNotConfigured which fires from
	// evaluateSample when SetResponder was never called — that
	// sentinel is about the RESPONSE-GENERATION responder; THIS
	// sentinel is about the JUDGE responder. Two different responder
	// roles, two different sentinels.
	ErrLLMBackedEvaluatorResponderNotProvided = errors.New(
		"llmops: LLMBackedEvaluatorConfig.Responder is nil — wire an LLMResponder (typically NewHTTPResponder targeting a judge-capable model) before constructing LLMBackedEvaluator; the previous absence of this guard would have allowed a nil-pointer dereference at first Evaluate call",
	)

	// ErrLLMBackedEvaluatorResponseUnparseable fires from Evaluate
	// when the judge LLM responded successfully but the response
	// text contained zero parseable SCORE_<metric>: <number> lines.
	// This is the anti-bluff load-bearing sentinel: returning
	// fabricated 0.5 (or any constant) per metric in this case
	// would be the round-25 PASS-bluff pattern. The sentinel
	// surfaces the judge's malformed output loudly so callers can
	// branch (retry with stricter prompt, fall back to
	// HeuristicEvaluator, abort the run).
	ErrLLMBackedEvaluatorResponseUnparseable = errors.New(
		"llmops: LLMBackedEvaluator judge LLM responded but produced zero parseable SCORE_<metric>: <0.0-1.0> lines — the judge either ignored the format directive or returned chatty prose; returning fabricated scores here would be a §11.4 PASS-bluff sibling of round-25's simulated-response defect",
	)

	// ErrLLMBackedEvaluatorScoreOutOfRange fires from Evaluate when
	// the judge LLM emitted a SCORE_<metric>: <number> line with the
	// number outside [0, 1]. Defensive: silently clamping to [0, 1]
	// would hide a judge-side regression (model misprompted, scale
	// confusion, percentage-vs-fraction confusion). Callers who
	// would rather accept the score MAY set
	// LLMBackedEvaluatorConfig.ClampOnRangeError=true to opt into
	// clamp-on-error behaviour; the default is REFUSE.
	ErrLLMBackedEvaluatorScoreOutOfRange = errors.New(
		"llmops: LLMBackedEvaluator judge LLM emitted a score outside [0, 1] — silently clamping here would hide a judge-side regression; set LLMBackedEvaluatorConfig.ClampOnRangeError=true if you prefer clamping over surfacing",
	)
)

// NewLLMBackedEvaluator constructs an LLMBackedEvaluator from the given
// config. Returns ErrLLMBackedEvaluatorResponderNotProvided if Responder
// is nil. The returned evaluator is safe for concurrent use across
// goroutines — it holds no mutable state after construction and the
// wrapped LLMResponder is required to be concurrency-safe per its
// interface contract.
func NewLLMBackedEvaluator(cfg LLMBackedEvaluatorConfig) (*LLMBackedEvaluator, error) {
	if cfg.Responder == nil {
		return nil, ErrLLMBackedEvaluatorResponderNotProvided
	}

	tmpl := strings.TrimSpace(cfg.JudgmentPromptTemplate)
	if tmpl == "" {
		tmpl = defaultJudgmentPromptTemplate
	}

	metrics := cfg.DefaultMetrics
	if len(metrics) == 0 {
		// Copy the package default so a caller mutating their slice
		// later cannot mutate ours.
		metrics = append([]string(nil), defaultMetricFamilies...)
	} else {
		// Defensive copy — same reason.
		metrics = append([]string(nil), metrics...)
	}

	return &LLMBackedEvaluator{
		responder:              cfg.Responder,
		judgmentPromptTemplate: tmpl,
		defaultMetrics:         metrics,
		clampOnRangeError:      cfg.ClampOnRangeError,
	}, nil
}

// Evaluate satisfies the LLMEvaluator interface contract. It dispatches
// a judgment prompt to the wrapped judge LLMResponder and parses the
// response for SCORE_<metric>: <number> lines, returning a map keyed
// by the requested metric names.
//
// Behavioural guarantees:
//
//   - Honours ctx.Cancel + ctx.Deadline — the underlying responder's
//     Generate is invoked with the supplied ctx so cancellation
//     propagates to the in-flight HTTP request.
//   - Empty metrics slice => the configured DefaultMetrics list is
//     used (package default: correctness / completeness /
//     conciseness / factuality).
//   - Judge Generate error => wrapped and returned (caller can
//     errors.Unwrap to inspect the underlying HTTP / transport error).
//   - Judge responded but zero scores parseable =>
//     ErrLLMBackedEvaluatorResponseUnparseable (NEVER fabricated 0.5s).
//   - Judge returned score outside [0, 1] =>
//     ErrLLMBackedEvaluatorScoreOutOfRange (unless ClampOnRangeError=true).
//   - Returned map contains an entry for every metric the JUDGE
//     scored. Metrics the judge did not score are silently absent —
//     consumers that require all-metrics-or-error MUST check map
//     cardinality themselves. (Future strict-mode flag is a follow-up
//     deferred to round 82+ if operator demand emerges.)
func (e *LLMBackedEvaluator) Evaluate(ctx context.Context, prompt, response, expected string, metrics []string) (map[string]float64, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("llmops: LLMBackedEvaluator: context cancelled before judge dispatch: %w", err)
	}

	useMetrics := metrics
	if len(useMetrics) == 0 {
		useMetrics = e.defaultMetrics
	}

	judgmentPrompt := e.buildJudgmentPrompt(prompt, response, expected, useMetrics)

	rawJudgment, err := e.responder.Generate(ctx, judgmentPrompt)
	if err != nil {
		return nil, fmt.Errorf("llmops: LLMBackedEvaluator: judge LLM dispatch failed: %w", err)
	}

	scores, err := e.parseScores(rawJudgment)
	if err != nil {
		return nil, err
	}

	if len(scores) == 0 {
		// Anti-bluff load-bearing branch: judge returned text but the
		// parser extracted zero scores. Refuse to fabricate.
		// Include a truncated excerpt of the judge's response so
		// the operator can see what the judge actually said.
		excerpt := rawJudgment
		const maxExcerpt = 256
		if len(excerpt) > maxExcerpt {
			excerpt = excerpt[:maxExcerpt] + "...(truncated)"
		}
		return nil, fmt.Errorf("%w: judge response excerpt: %q", ErrLLMBackedEvaluatorResponseUnparseable, excerpt)
	}

	return scores, nil
}

// buildJudgmentPrompt renders the judgment template with the supplied
// values. Placeholder replacement is string-based (strings.ReplaceAll)
// rather than text/template because (a) the prompt body may legitimately
// contain { and } characters from code-block content the model has to
// judge, and (b) we want NO escape semantics — the judge sees the
// inputs verbatim so it can detect adversarial / boundary-confusion
// attempts in the response under test.
func (e *LLMBackedEvaluator) buildJudgmentPrompt(prompt, response, expected string, metrics []string) string {
	// Sort metrics for stable output so the judge sees them in a
	// consistent order regardless of how the caller built the slice.
	// Sorted output also makes the unit tests' golden assertions
	// deterministic without per-test sort calls.
	sorted := append([]string(nil), metrics...)
	sort.Strings(sorted)
	metricsList := strings.Join(sorted, ", ")

	out := e.judgmentPromptTemplate
	out = strings.ReplaceAll(out, "{prompt}", prompt)
	out = strings.ReplaceAll(out, "{response}", response)
	out = strings.ReplaceAll(out, "{expected}", expected)
	out = strings.ReplaceAll(out, "{metrics}", metricsList)
	return out
}

// parseScores extracts SCORE_<metric>: <number> pairs from the judge's
// response text. Returns the parsed map (lowercased metric keys) or
// ErrLLMBackedEvaluatorScoreOutOfRange if any number is outside [0, 1]
// AND ClampOnRangeError=false. Returns an empty map (NOT an error)
// when the judge response contained zero matches — the caller's
// len(scores)==0 branch turns that into the unparseable sentinel.
func (e *LLMBackedEvaluator) parseScores(judgment string) (map[string]float64, error) {
	matches := scorePattern.FindAllStringSubmatch(judgment, -1)
	if len(matches) == 0 {
		return map[string]float64{}, nil
	}

	scores := make(map[string]float64, len(matches))
	for _, m := range matches {
		// m[0] = full match, m[1] = metric name, m[2] = score literal
		metric := strings.ToLower(strings.TrimSpace(m[1]))
		raw := strings.TrimSpace(m[2])

		score, err := strconv.ParseFloat(raw, 64)
		if err != nil {
			// Defensive: the regex character class restricted to
			// digits + sign + dot, so ParseFloat shouldn't realistically
			// fail. Wrap defensively rather than panic so the contract
			// stays "every failure surfaces as a sentinel-wrapped error".
			return nil, fmt.Errorf("%w: failed to parse score literal %q for metric %q: %v",
				ErrLLMBackedEvaluatorScoreOutOfRange, raw, metric, err)
		}

		if score < 0.0 || score > 1.0 {
			if !e.clampOnRangeError {
				return nil, fmt.Errorf("%w: metric %q got score %g (expected [0, 1])",
					ErrLLMBackedEvaluatorScoreOutOfRange, metric, score)
			}
			// Opt-in clamp branch.
			if score < 0 {
				score = 0
			}
			if score > 1 {
				score = 1
			}
		}

		scores[metric] = score
	}
	return scores, nil
}

// String renders a non-secret summary of the evaluator for diagnostic
// logs. Does NOT include the wrapped responder's secret state (the
// responder's own String/GoString redact API keys per CONST-042; we
// defer to that here).
func (e *LLMBackedEvaluator) String() string {
	return fmt.Sprintf(
		"LLMBackedEvaluator{responder=%v defaultMetrics=%v clampOnRangeError=%t templateLen=%d}",
		e.responder, e.defaultMetrics, e.clampOnRangeError, len(e.judgmentPromptTemplate),
	)
}

// GoString mirrors String for %#v formatting consistency.
func (e *LLMBackedEvaluator) GoString() string { return e.String() }
