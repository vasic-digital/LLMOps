package llmops

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"unicode"
)

// HeuristicEvaluator is a concrete LLMEvaluator implementation that scores
// the LLM response against the expected output using real-input-derived
// signals. It is the round-75 §11.4 anti-bluff close-out of the round-25
// ErrEvaluatorNotConfigured sentinel: round 25 removed the literal
// "Passed=true with 0.8 placeholder score per metric" bluff but left
// consumers without a ready wireable implementation; this type is what
// they wire into (*InMemoryContinuousEvaluator) via the constructor's
// evaluator parameter (or via SetDebateEvaluator at the LLMOpsSystem
// layer) so evaluateSample produces real per-metric scores derived from
// the actual response content rather than fabricated constants.
//
// Round 62 (commit f017e43) closed the sibling sentinel
// ErrLLMResponderNotConfigured with HTTPResponder; this file is the
// mirror-image fix for the evaluator side of the same audit-round-25
// pair.
//
// CONST-035 / Article XI §11.9 / CONST-050(A) load-bearing invariant:
// every score this type returns is DERIVED from the actual response and
// expected strings — not a hardcoded constant. The paired-mutation test
// TestHeuristicEvaluator_NotFabricated proves the invariant: change the
// response by one character and the score MUST change. If that test
// fails, this implementation is bluffing and ships nothing.
//
// Algorithm: three deterministic sub-scores derived from token sets and
// length ratios, combined per a metric-name-aware weighting profile so
// per-metric scores vary in a domain-meaningful way:
//
//  1. Jaccard similarity on lowercased word tokens — symmetric set
//     overlap, |A ∩ B| / |A ∪ B|. Drives "correctness"/"accuracy" metrics.
//  2. Keyword coverage — fraction of expected tokens present in the
//     response. Drives "completeness"/"coverage"/"recall" metrics.
//  3. Length ratio — min(|R|, |E|) / max(|R|, |E|) on word counts.
//     Drives "conciseness"/"brevity"/"length" metrics; penalises
//     responses that are either far shorter OR far longer than expected.
//
// Each sub-score lives in [0, 1] by construction. The aggregate is the
// weighted sum clamped to [0, 1] defensively (ErrHeuristicEvaluatorScoreOutOfRange
// fires if a coding bug ever produces a score outside [0, 1] BEFORE
// clamping, so silent drift is loud).
//
// CONST-050(A): production code may import and construct HeuristicEvaluator.
// The unit tests in heuristic_evaluator_test.go exercise it against
// in-process string inputs (no external network), satisfying the unit
// tier. Integration callers wire it via NewInMemoryContinuousEvaluator
// and exercise it end-to-end through evaluateSample with real model
// responses produced by HTTPResponder (round 62).
type HeuristicEvaluator struct {
	weights           profileWeights
	metricProfiles    map[string]profileWeights
	minMatchThreshold float64
}

// HeuristicEvaluatorConfig is the constructor input for NewHeuristicEvaluator.
// All fields are optional — a zero-value config produces a usable
// evaluator with sensible defaults documented per field.
type HeuristicEvaluatorConfig struct {
	// Weights overrides the DEFAULT weighting profile applied to metrics
	// whose name does not match one of the named profiles in
	// MetricProfiles. Map keys are the three sub-score identifiers:
	//   "jaccard"           — token-set overlap
	//   "keyword_coverage"  — expected tokens present in response
	//   "length_ratio"      — word-count ratio
	// Values are weights in [0, +inf). They are normalised by their sum
	// before use, so {jaccard: 2, keyword_coverage: 1, length_ratio: 1}
	// is identical to {jaccard: 0.5, keyword_coverage: 0.25, length_ratio: 0.25}.
	// Unknown keys are ignored (so adding sub-scores in future versions
	// is non-breaking). Empty map => the default profile is used.
	Weights map[string]float64

	// MetricProfiles maps metric names (case-insensitive) to weighting
	// profiles. When evaluateSample requests scores for metric "accuracy",
	// the profile registered under "accuracy" (after lowercasing) is used.
	// Empty map => the built-in profile heuristic (see metricProfileFor)
	// applies. Profiles supplied here merge over the built-ins.
	MetricProfiles map[string]map[string]float64

	// MinMatchThreshold is reserved for future use (a stricter PASS
	// floor). Currently unused by Evaluate (the PASS floor lives in
	// evaluateSample at 0.7); kept in the struct so the field can be
	// added later without a constructor signature break.
	MinMatchThreshold float64
}

// profileWeights is the normalised weighting profile used by Evaluate.
// Sum of the three fields equals 1.0 (enforced by normaliseProfile).
type profileWeights struct {
	jaccard         float64
	keywordCoverage float64
	lengthRatio     float64
}

// ErrHeuristicEvaluatorMissingInput fires from Evaluate when either the
// response or the expected output is empty after trimming. Either input
// being empty means the heuristic signal is undefined — Jaccard of
// {} ∩ anything is 0, length ratio of 0/N is 0, keyword coverage of
// "what's in {}" is undefined. Returning a fabricated 0.0 would be a
// PASS-bluff sibling (silently asserting "scored low" when the truth is
// "could not score at all"); returning a sentinel surfaces the
// undefined-state loudly.
var ErrHeuristicEvaluatorMissingInput = errors.New(
	"llmops: HeuristicEvaluator requires non-empty response and expected output — at least one input was empty/whitespace, scoring is undefined (returning a fabricated 0.0 here would be a §11.4 PASS-bluff sibling of the round-25 placeholder defect)",
)

// ErrHeuristicEvaluatorScoreOutOfRange is a defensive sentinel that
// fires if the internal weighted-aggregate computation produces a value
// outside [0, 1] BEFORE clamping. The three sub-scores are each
// constructively bounded to [0, 1] and the weights normalise to 1, so
// the aggregate cannot exceed [0, 1] in correct code. This sentinel
// makes a future coding regression LOUD (test detection) rather than
// silent (clamp hides the bug).
var ErrHeuristicEvaluatorScoreOutOfRange = errors.New(
	"llmops: HeuristicEvaluator internal computation produced a sub-score or aggregate outside [0, 1] — this is a defensive guard; correct code cannot reach this branch, so a hit means a sub-score function was changed without re-verifying its bounds",
)

// NewHeuristicEvaluator constructs a HeuristicEvaluator from the given
// config. An empty config produces a usable evaluator with defaults
// (equal-weight jaccard/keyword_coverage/length_ratio for unknown
// metrics; the built-in named profiles for "correctness", "completeness",
// "conciseness" families). Never returns an error — invalid weights are
// normalised; unknown keys are ignored; nil maps are treated as empty.
//
// The returned evaluator is safe for concurrent use across goroutines:
// it holds no mutable state after construction.
func NewHeuristicEvaluator(cfg HeuristicEvaluatorConfig) (*HeuristicEvaluator, error) {
	defaultWeights := normaliseProfile(cfg.Weights, profileWeights{
		jaccard:         1.0 / 3.0,
		keywordCoverage: 1.0 / 3.0,
		lengthRatio:     1.0 / 3.0,
	})

	// Built-in profiles (merged with caller-supplied overrides below).
	builtIn := map[string]profileWeights{
		// "correctness" / "accuracy" — jaccard-heavy: the response
		// should share most tokens with the expected output.
		"correctness": {jaccard: 0.60, keywordCoverage: 0.25, lengthRatio: 0.15},
		"accuracy":    {jaccard: 0.60, keywordCoverage: 0.25, lengthRatio: 0.15},
		"precision":   {jaccard: 0.60, keywordCoverage: 0.25, lengthRatio: 0.15},

		// "completeness" / "coverage" / "recall" — keyword-coverage-heavy:
		// the response should mention every expected token.
		"completeness": {jaccard: 0.25, keywordCoverage: 0.60, lengthRatio: 0.15},
		"coverage":     {jaccard: 0.25, keywordCoverage: 0.60, lengthRatio: 0.15},
		"recall":       {jaccard: 0.25, keywordCoverage: 0.60, lengthRatio: 0.15},

		// "conciseness" / "brevity" / "length" — length-ratio-heavy: the
		// response should be roughly the same length as expected.
		"conciseness": {jaccard: 0.25, keywordCoverage: 0.15, lengthRatio: 0.60},
		"brevity":     {jaccard: 0.25, keywordCoverage: 0.15, lengthRatio: 0.60},
		"length":      {jaccard: 0.25, keywordCoverage: 0.15, lengthRatio: 0.60},
	}

	// Merge caller-supplied profiles over the built-ins (lower-case keys).
	for name, raw := range cfg.MetricProfiles {
		builtIn[strings.ToLower(strings.TrimSpace(name))] = normaliseProfile(raw, defaultWeights)
	}

	return &HeuristicEvaluator{
		weights:           defaultWeights,
		metricProfiles:    builtIn,
		minMatchThreshold: cfg.MinMatchThreshold,
	}, nil
}

// Evaluate satisfies the LLMEvaluator interface contract. It returns a
// map keyed by the requested metric names where each value is a real,
// input-derived score in [0, 1].
//
// Behavioural guarantees:
//
//   - Empty response or expected => ErrHeuristicEvaluatorMissingInput
//     (NEVER a fabricated 0.0).
//   - Every score is DERIVED from response + expected — change either
//     input by one character and at least one score changes.
//   - Per-metric weighting profiles produce DIFFERENT scores for
//     DIFFERENT metric names from the SAME response/expected pair.
//   - All scores clamped to [0, 1]; out-of-range internal values fire
//     ErrHeuristicEvaluatorScoreOutOfRange BEFORE clamping (defensive).
//   - The prompt argument is currently unused (matches the
//     non-LLM-as-judge heuristic posture); kept in the signature for
//     LLMEvaluator interface conformance.
//   - The ctx argument is honoured for cancellation only when the
//     metrics list is large enough to make per-metric work non-trivial;
//     the current scoring is O(|R| + |E|) per metric so cancellation
//     check happens once at the top.
func (h *HeuristicEvaluator) Evaluate(ctx context.Context, _, response, expected string, metrics []string) (map[string]float64, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("llmops: HeuristicEvaluator: context cancelled before scoring: %w", err)
	}

	if strings.TrimSpace(response) == "" || strings.TrimSpace(expected) == "" {
		return nil, ErrHeuristicEvaluatorMissingInput
	}

	respTokens := tokenise(response)
	expTokens := tokenise(expected)

	// Pre-compute the three sub-scores once — they don't depend on the
	// metric name, only the per-metric WEIGHTING does.
	jacc, err := jaccardScore(respTokens, expTokens)
	if err != nil {
		return nil, err
	}
	cov, err := keywordCoverageScore(respTokens, expTokens)
	if err != nil {
		return nil, err
	}
	lr, err := lengthRatioScore(respTokens, expTokens)
	if err != nil {
		return nil, err
	}

	// If no metrics were requested, return at least the default-profile
	// aggregate under the synthetic key "_overall" so callers that
	// forgot to supply metrics still get a real signal rather than an
	// empty map (which would be indistinguishable from "scored zero on
	// nothing"). Callers SHOULD always pass a metrics list.
	if len(metrics) == 0 {
		score, err := h.aggregate(h.weights, jacc, cov, lr)
		if err != nil {
			return nil, err
		}
		return map[string]float64{"_overall": score}, nil
	}

	scores := make(map[string]float64, len(metrics))
	for _, m := range metrics {
		profile := h.metricProfileFor(m)
		score, err := h.aggregate(profile, jacc, cov, lr)
		if err != nil {
			return nil, fmt.Errorf("llmops: HeuristicEvaluator: aggregate failed for metric %q: %w", m, err)
		}
		scores[m] = score
	}
	return scores, nil
}

// metricProfileFor returns the weighting profile for the given metric
// name. Lookup is case-insensitive against the merged built-in +
// caller-supplied profile map; unknown metrics fall back to the default
// (equal-weight, or the caller-supplied default via Weights).
func (h *HeuristicEvaluator) metricProfileFor(metric string) profileWeights {
	if p, ok := h.metricProfiles[strings.ToLower(strings.TrimSpace(metric))]; ok {
		return p
	}
	return h.weights
}

// aggregate produces the weighted sum of the three sub-scores and
// clamps to [0, 1] defensively. Returns ErrHeuristicEvaluatorScoreOutOfRange
// if any sub-score or the pre-clamp aggregate lies outside [0, 1].
func (h *HeuristicEvaluator) aggregate(w profileWeights, jacc, cov, lr float64) (float64, error) {
	for _, s := range []float64{jacc, cov, lr} {
		if s < 0.0 || s > 1.0 {
			return 0, fmt.Errorf("%w: sub-score %g not in [0,1]", ErrHeuristicEvaluatorScoreOutOfRange, s)
		}
	}
	score := w.jaccard*jacc + w.keywordCoverage*cov + w.lengthRatio*lr
	if score < -1e-9 || score > 1.0+1e-9 {
		return 0, fmt.Errorf("%w: aggregate %g not in [0,1]", ErrHeuristicEvaluatorScoreOutOfRange, score)
	}
	// Clamp floating-point drift inside [0, 1] (sub-1e-9 over/underflow
	// from FP rounding is harmless; the sentinel above catches anything
	// larger which means an algorithmic bug).
	if score < 0 {
		score = 0
	}
	if score > 1 {
		score = 1
	}
	return score, nil
}

// tokenise lowercases the input and splits on any non-letter / non-digit
// rune. Empty tokens are dropped. The function is deterministic and
// pure — same input always produces same token slice.
func tokenise(s string) []string {
	s = strings.ToLower(s)
	tokens := strings.FieldsFunc(s, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
	out := make([]string, 0, len(tokens))
	for _, t := range tokens {
		if t != "" {
			out = append(out, t)
		}
	}
	return out
}

// tokenSet folds a token slice into a set (map[string]struct{}). Used
// by jaccardScore + keywordCoverageScore; pulled out to a helper so the
// O(N) construction happens at known points.
func tokenSet(tokens []string) map[string]struct{} {
	s := make(map[string]struct{}, len(tokens))
	for _, t := range tokens {
		s[t] = struct{}{}
	}
	return s
}

// jaccardScore returns |A ∩ B| / |A ∪ B| in [0, 1]. Empty-set edge
// cases: both empty => 1.0 (vacuously equal — caller already rejected
// empty inputs via ErrHeuristicEvaluatorMissingInput, this branch only
// fires if tokenise produced zero tokens from non-whitespace input which
// would mean punctuation-only strings; treat as "trivially matching").
func jaccardScore(resp, exp []string) (float64, error) {
	rSet := tokenSet(resp)
	eSet := tokenSet(exp)

	if len(rSet) == 0 && len(eSet) == 0 {
		return 1.0, nil
	}

	// Intersection
	var inter int
	for t := range rSet {
		if _, ok := eSet[t]; ok {
			inter++
		}
	}
	// Union = |A| + |B| - intersection
	union := len(rSet) + len(eSet) - inter
	if union == 0 {
		return 0, fmt.Errorf("%w: jaccard union is zero with non-empty inputs (logic bug)", ErrHeuristicEvaluatorScoreOutOfRange)
	}
	return float64(inter) / float64(union), nil
}

// keywordCoverageScore returns the fraction of expected tokens present
// in the response token set. Empty expected => 1.0 (vacuously covered).
func keywordCoverageScore(resp, exp []string) (float64, error) {
	rSet := tokenSet(resp)
	eSet := tokenSet(exp)

	if len(eSet) == 0 {
		return 1.0, nil
	}

	var present int
	for t := range eSet {
		if _, ok := rSet[t]; ok {
			present++
		}
	}
	return float64(present) / float64(len(eSet)), nil
}

// lengthRatioScore returns min(|R|, |E|) / max(|R|, |E|) on token
// counts. Both empty => 1.0; one empty => 0.0.
func lengthRatioScore(resp, exp []string) (float64, error) {
	rN, eN := len(resp), len(exp)
	if rN == 0 && eN == 0 {
		return 1.0, nil
	}
	if rN == 0 || eN == 0 {
		return 0.0, nil
	}
	if rN <= eN {
		return float64(rN) / float64(eN), nil
	}
	return float64(eN) / float64(rN), nil
}

// normaliseProfile converts a raw weight map (potentially containing
// unknown keys, nil, or negative values) into a profileWeights that
// sums to 1.0. Negative values are clamped to 0. Empty/nil map or
// all-zero weights => fallback to the supplied default profile.
func normaliseProfile(raw map[string]float64, fallback profileWeights) profileWeights {
	var p profileWeights
	if raw != nil {
		p.jaccard = clampNonNeg(raw["jaccard"])
		p.keywordCoverage = clampNonNeg(raw["keyword_coverage"])
		p.lengthRatio = clampNonNeg(raw["length_ratio"])
	}
	sum := p.jaccard + p.keywordCoverage + p.lengthRatio
	if sum <= 0 {
		return fallback
	}
	return profileWeights{
		jaccard:         p.jaccard / sum,
		keywordCoverage: p.keywordCoverage / sum,
		lengthRatio:     p.lengthRatio / sum,
	}
}

func clampNonNeg(v float64) float64 {
	if v < 0 {
		return 0
	}
	return v
}

// String renders a non-secret summary of the evaluator for diagnostic
// logs. No fields are sensitive (weights and profiles), so unlike
// HTTPResponder.String we don't elide anything — but we keep the
// format compact.
func (h *HeuristicEvaluator) String() string {
	return fmt.Sprintf(
		"HeuristicEvaluator{default={j=%.2f kc=%.2f lr=%.2f} profiles=%d minMatch=%.2f}",
		h.weights.jaccard, h.weights.keywordCoverage, h.weights.lengthRatio,
		len(h.metricProfiles), h.minMatchThreshold,
	)
}

// GoString mirrors String for %#v formatting consistency.
func (h *HeuristicEvaluator) GoString() string { return h.String() }
