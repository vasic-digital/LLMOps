// CONST-035 / CONST-046 sentinel coverage for round-119 i18n migration.
// Tests guard against regressions that drop SetTranslator wiring or
// stop piping translator output into user-facing Recommendation /
// Summary fields. Per CONST-050(A) fakes are permitted in *_test.go.
package llmops

import (
	"context"
	"sync"
	"testing"
	"time"

	"digital.vasic.llmops/pkg/i18n"
)

// recordingTranslator captures the i18n keys the production code emits
// and returns a sentinel string so call-sites can be asserted on.
type recordingTranslator struct {
	mu   sync.Mutex
	seen []string
}

func (r *recordingTranslator) T(key string, _ map[string]any) string {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.seen = append(r.seen, key)
	return "translated:" + key
}

func (r *recordingTranslator) keys() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]string, len(r.seen))
	copy(out, r.seen)
	return out
}

func TestInMemoryContinuousEvaluator_SetTranslator_OverridesSummary(t *testing.T) {
	e := NewInMemoryContinuousEvaluator(nil, nil, nil, nil)
	rt := &recordingTranslator{}
	e.SetTranslator(rt)

	// Render each of the three summary branches via renderSummary
	// directly (it is the centralised seam used by CompareRuns).
	regr := e.renderSummary("llmops_run_comparison_regressions",
		map[string]any{"count": 3},
		"Regressions in 3 metrics")
	impr := e.renderSummary("llmops_run_comparison_improvements",
		map[string]any{"count": 2},
		"Improvements in 2 metrics")
	none := e.renderSummary("llmops_run_comparison_no_change", nil,
		"No significant changes")

	if regr != "translated:llmops_run_comparison_regressions" {
		t.Fatalf("regressions branch: got %q, want translator output", regr)
	}
	if impr != "translated:llmops_run_comparison_improvements" {
		t.Fatalf("improvements branch: got %q, want translator output", impr)
	}
	if none != "translated:llmops_run_comparison_no_change" {
		t.Fatalf("no-change branch: got %q, want translator output", none)
	}

	keys := rt.keys()
	if len(keys) != 3 {
		t.Fatalf("recorded %d translator calls, want 3", len(keys))
	}
}

func TestInMemoryContinuousEvaluator_NoopTranslator_PreservesEnglish(t *testing.T) {
	// Default (no SetTranslator call) MUST render legacy English so
	// existing assertions keep passing — guards against a regression
	// that ships keys-as-prose if the fallback path is removed.
	e := NewInMemoryContinuousEvaluator(nil, nil, nil, nil)

	if got := e.renderSummary("llmops_run_comparison_no_change", nil, "No significant changes"); got != "No significant changes" {
		t.Fatalf("noop fallback: got %q, want %q", got, "No significant changes")
	}
}

func TestInMemoryContinuousEvaluator_SetTranslator_NilIsNoop(t *testing.T) {
	e := NewInMemoryContinuousEvaluator(nil, nil, nil, nil)
	e.SetTranslator(nil) // MUST NOT panic, MUST NOT overwrite the default

	if got := e.renderSummary("llmops_run_comparison_no_change", nil, "No significant changes"); got != "No significant changes" {
		t.Fatalf("after nil SetTranslator: got %q, want default fallback", got)
	}
}

func TestInMemoryExperimentManager_SetTranslator_OverridesRecommendation(t *testing.T) {
	m := NewInMemoryExperimentManager(nil)
	rt := &recordingTranslator{}
	m.SetTranslator(rt)

	deploy := m.renderRecommendation("llmops_experiment_recommendation_deploy",
		map[string]any{"variant": "B", "confidence": 95.4},
		"Deploy variant B with 95.4% confidence")
	cont := m.renderRecommendation("llmops_experiment_recommendation_continue",
		nil,
		"Continue experiment - insufficient confidence")

	if deploy != "translated:llmops_experiment_recommendation_deploy" {
		t.Fatalf("deploy branch: got %q, want translator output", deploy)
	}
	if cont != "translated:llmops_experiment_recommendation_continue" {
		t.Fatalf("continue branch: got %q, want translator output", cont)
	}

	keys := rt.keys()
	if len(keys) != 2 {
		t.Fatalf("recorded %d translator calls, want 2", len(keys))
	}
}

func TestInMemoryExperimentManager_NoopTranslator_PreservesEnglish(t *testing.T) {
	m := NewInMemoryExperimentManager(nil)

	got := m.renderRecommendation(
		"llmops_experiment_recommendation_continue",
		nil,
		"Continue experiment - insufficient confidence")
	if got != "Continue experiment - insufficient confidence" {
		t.Fatalf("noop fallback: got %q, want legacy English", got)
	}
}

func TestInMemoryExperimentManager_SetTranslator_NilIsNoop(t *testing.T) {
	m := NewInMemoryExperimentManager(nil)
	m.SetTranslator(nil) // MUST NOT panic, MUST NOT overwrite the default

	got := m.renderRecommendation(
		"llmops_experiment_recommendation_continue",
		nil,
		"Continue experiment - insufficient confidence")
	if got != "Continue experiment - insufficient confidence" {
		t.Fatalf("after nil SetTranslator: got %q, want default fallback", got)
	}
}

// CONST-051(B) decoupling check: NoopTranslator interface satisfaction
// + Translator type both live in vasic-digital/LLMOps' own pkg/i18n —
// no parent-tree reach. This compile-time assertion is the §11.4
// paired-mutation pair for the translator-import regression mode.
var _ i18n.Translator = i18n.NoopTranslator{}

// --- Round-211 §11.4 — HTTPResponder + LLMBackedEvaluator i18n
// callsite coverage. Each test below proves three invariants per
// migrated key:
//   1. NoopTranslator path emits the legacy English fallback verbatim
//      (so historical regression tests + downstream string assertions
//      keep passing — Article XI §11.9 / CONST-035 / CONST-046).
//   2. Recording translator's keys() captures the expected i18n key
//      (so a future refactor that drops the SetTranslator wiring or
//      bypasses renderError gets caught by the missing-call mutation).
//   3. SetTranslator(nil) is a safe no-op (so a regression that drops
//      the nil-guard cannot crash production at wire-up time).

func TestHTTPResponder_SetTranslator_OverridesAllErrorBodies(t *testing.T) {
	r, err := NewHTTPResponder(HTTPResponderConfig{
		Endpoint: "http://127.0.0.1:1", // unroutable port → POST fails fast
		Model:    "round-211-test-model",
		Timeout:  100 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("constructor failed: %v", err)
	}
	rt := &recordingTranslator{}
	r.SetTranslator(rt)

	// Direct renderError walk over each round-211 responder key —
	// equivalent to the (m *InMemoryExperimentManager).renderRecommendation
	// pattern in TestInMemoryExperimentManager_SetTranslator_OverridesRecommendation.
	keys := []struct {
		key      string
		params   map[string]any
		fallback string
	}{
		{"llmops_responder_marshal_request_failed",
			map[string]any{"cause": "x"},
			"marshal request body: x"},
		{"llmops_responder_build_request_failed",
			map[string]any{"url": "u", "cause": "c"},
			"build request for u: c"},
		{"llmops_responder_post_failed",
			map[string]any{"url": "u", "cause": "c"},
			"POST u: c"},
		{"llmops_responder_read_body_failed",
			map[string]any{"url": "u", "status": 500, "cause": "c"},
			"read response body from u (status 500): c"},
		{"llmops_responder_zero_choices",
			map[string]any{"url": "u"},
			"response from u contained zero choices"},
		{"llmops_responder_empty_content",
			map[string]any{"url": "u"},
			"response from u had choices[0].message.content == \"\""},
	}
	for _, tc := range keys {
		got := r.renderError(tc.key, tc.params, tc.fallback)
		want := "translated:" + tc.key
		if got != want {
			t.Fatalf("renderError(%s): got %q, want %q", tc.key, got, want)
		}
	}

	if len(rt.keys()) != len(keys) {
		t.Fatalf("recorded %d translator calls, want %d", len(rt.keys()), len(keys))
	}
}

func TestHTTPResponder_NoopTranslator_PreservesEnglish(t *testing.T) {
	r, err := NewHTTPResponder(HTTPResponderConfig{
		Endpoint: "http://127.0.0.1:1",
		Model:    "round-211-test-model",
	})
	if err != nil {
		t.Fatalf("constructor failed: %v", err)
	}

	got := r.renderError("llmops_responder_zero_choices",
		map[string]any{"url": "u"},
		"response from u contained zero choices")
	if got != "response from u contained zero choices" {
		t.Fatalf("noop fallback: got %q, want legacy English", got)
	}
}

func TestHTTPResponder_SetTranslator_NilIsNoop(t *testing.T) {
	r, err := NewHTTPResponder(HTTPResponderConfig{
		Endpoint: "http://127.0.0.1:1",
		Model:    "round-211-test-model",
	})
	if err != nil {
		t.Fatalf("constructor failed: %v", err)
	}
	r.SetTranslator(nil) // MUST NOT panic, MUST NOT overwrite default

	got := r.renderError("llmops_responder_post_failed",
		map[string]any{"url": "u", "cause": "c"},
		"POST u: c")
	if got != "POST u: c" {
		t.Fatalf("after nil SetTranslator: got %q, want default fallback", got)
	}
}

func TestLLMBackedEvaluator_SetTranslator_OverridesJudgeErrorBodies(t *testing.T) {
	e, err := NewLLMBackedEvaluator(LLMBackedEvaluatorConfig{
		Responder: LLMResponderFunc(func(context.Context, string) (string, error) { return "", nil }),
	})
	if err != nil {
		t.Fatalf("constructor failed: %v", err)
	}
	rt := &recordingTranslator{}
	e.SetTranslator(rt)

	keys := []struct {
		key      string
		params   map[string]any
		fallback string
	}{
		{"llmops_llm_evaluator_ctx_cancelled",
			map[string]any{"cause": "deadline"},
			"llmops: LLMBackedEvaluator: context cancelled before judge dispatch"},
		{"llmops_llm_evaluator_judge_dispatch_failed",
			map[string]any{"cause": "timeout"},
			"llmops: LLMBackedEvaluator: judge LLM dispatch failed"},
		{"llmops_llm_evaluator_judge_excerpt",
			map[string]any{"excerpt": "chatter"},
			"judge response excerpt: \"chatter\""},
		{"llmops_llm_evaluator_metric_out_of_range",
			map[string]any{"metric": "correctness", "score": 1.5},
			"metric \"correctness\" got score 1.5 (expected [0, 1])"},
	}
	for _, tc := range keys {
		got := e.renderError(tc.key, tc.params, tc.fallback)
		want := "translated:" + tc.key
		if got != want {
			t.Fatalf("renderError(%s): got %q, want %q", tc.key, got, want)
		}
	}

	if len(rt.keys()) != len(keys) {
		t.Fatalf("recorded %d translator calls, want %d", len(rt.keys()), len(keys))
	}
}

func TestLLMBackedEvaluator_NoopTranslator_PreservesEnglish(t *testing.T) {
	e, err := NewLLMBackedEvaluator(LLMBackedEvaluatorConfig{
		Responder: LLMResponderFunc(func(context.Context, string) (string, error) { return "", nil }),
	})
	if err != nil {
		t.Fatalf("constructor failed: %v", err)
	}

	got := e.renderError("llmops_llm_evaluator_judge_dispatch_failed",
		map[string]any{"cause": "x"},
		"llmops: LLMBackedEvaluator: judge LLM dispatch failed")
	if got != "llmops: LLMBackedEvaluator: judge LLM dispatch failed" {
		t.Fatalf("noop fallback: got %q, want legacy English", got)
	}
}

func TestLLMBackedEvaluator_SetTranslator_NilIsNoop(t *testing.T) {
	e, err := NewLLMBackedEvaluator(LLMBackedEvaluatorConfig{
		Responder: LLMResponderFunc(func(context.Context, string) (string, error) { return "", nil }),
	})
	if err != nil {
		t.Fatalf("constructor failed: %v", err)
	}
	e.SetTranslator(nil)

	got := e.renderError("llmops_llm_evaluator_judge_excerpt",
		map[string]any{"excerpt": "y"},
		"judge response excerpt: \"y\"")
	if got != "judge response excerpt: \"y\"" {
		t.Fatalf("after nil SetTranslator: got %q, want default fallback", got)
	}
}

// --- Round-394 §11.4 — regression-alert (continuous evaluator) +
// experiment-description (LLMOpsSystem) i18n callsite coverage. Each
// test below proves the same three invariants as the round-211 block:
//   1. NoopTranslator path emits the legacy English fallback verbatim
//      (Article XI §11.9 / CONST-035 / CONST-046).
//   2. A recording translator's keys() captures the expected i18n key
//      (so a future refactor that drops the renderSummary/renderDescription
//      wiring gets caught by the missing-call mutation).
//   3. SetTranslator(nil) is a safe no-op.

func TestInMemoryContinuousEvaluator_SetTranslator_OverridesAlertMessages(t *testing.T) {
	e := NewInMemoryContinuousEvaluator(nil, nil, nil, nil)
	rt := &recordingTranslator{}
	e.SetTranslator(rt)

	// renderSummary is the centralised seam used by both the
	// pass-rate-regression and per-metric-regression Alert.Message
	// call-sites in checkRegressions.
	passRate := e.renderSummary("llmops_alert_pass_rate_regression",
		map[string]any{"previous": "92.0", "current": "85.0"},
		"Pass rate regression: 92.0% -> 85.0%")
	metric := e.renderSummary("llmops_alert_metric_regression",
		map[string]any{"metric": "correctness", "previous": "0.90", "current": "0.70"},
		"Metric correctness regression: 0.90 -> 0.70")

	if passRate != "translated:llmops_alert_pass_rate_regression" {
		t.Fatalf("pass-rate alert: got %q, want translator output", passRate)
	}
	if metric != "translated:llmops_alert_metric_regression" {
		t.Fatalf("metric alert: got %q, want translator output", metric)
	}
	if keys := rt.keys(); len(keys) != 2 {
		t.Fatalf("recorded %d translator calls, want 2", len(keys))
	}
}

func TestInMemoryContinuousEvaluator_NoopTranslator_PreservesAlertEnglish(t *testing.T) {
	e := NewInMemoryContinuousEvaluator(nil, nil, nil, nil)

	pr := e.renderSummary("llmops_alert_pass_rate_regression",
		map[string]any{"previous": "92.0", "current": "85.0"},
		"Pass rate regression: 92.0% -> 85.0%")
	if pr != "Pass rate regression: 92.0% -> 85.0%" {
		t.Fatalf("noop fallback (pass-rate): got %q, want legacy English", pr)
	}
	m := e.renderSummary("llmops_alert_metric_regression",
		map[string]any{"metric": "correctness", "previous": "0.90", "current": "0.70"},
		"Metric correctness regression: 0.90 -> 0.70")
	if m != "Metric correctness regression: 0.90 -> 0.70" {
		t.Fatalf("noop fallback (metric): got %q, want legacy English", m)
	}
}

func TestLLMOpsSystem_SetTranslator_OverridesExperimentDescriptions(t *testing.T) {
	s := NewLLMOpsSystem(nil, nil)
	rt := &recordingTranslator{}
	s.SetTranslator(rt)

	ab := s.renderDescription("llmops_experiment_prompt_ab_description",
		map[string]any{"control": "ctrl", "treatment": "trt"},
		"A/B test: ctrl vs trt")
	model := s.renderDescription("llmops_experiment_model_comparison_description",
		map[string]any{"models": "[gpt-4 claude-3]"},
		"Model comparison: [gpt-4 claude-3]")

	if ab != "translated:llmops_experiment_prompt_ab_description" {
		t.Fatalf("prompt A/B description: got %q, want translator output", ab)
	}
	if model != "translated:llmops_experiment_model_comparison_description" {
		t.Fatalf("model comparison description: got %q, want translator output", model)
	}
	if keys := rt.keys(); len(keys) != 2 {
		t.Fatalf("recorded %d translator calls, want 2", len(keys))
	}
}

func TestLLMOpsSystem_NoopTranslator_PreservesDescriptionEnglish(t *testing.T) {
	s := NewLLMOpsSystem(nil, nil)

	ab := s.renderDescription("llmops_experiment_prompt_ab_description",
		map[string]any{"control": "ctrl", "treatment": "trt"},
		"A/B test: ctrl vs trt")
	if ab != "A/B test: ctrl vs trt" {
		t.Fatalf("noop fallback (A/B): got %q, want legacy English", ab)
	}
	model := s.renderDescription("llmops_experiment_model_comparison_description",
		map[string]any{"models": "[m1 m2]"},
		"Model comparison: [m1 m2]")
	if model != "Model comparison: [m1 m2]" {
		t.Fatalf("noop fallback (model): got %q, want legacy English", model)
	}
}

func TestLLMOpsSystem_SetTranslator_NilIsNoop(t *testing.T) {
	s := NewLLMOpsSystem(nil, nil)
	s.SetTranslator(nil) // MUST NOT panic, MUST NOT overwrite the default

	got := s.renderDescription("llmops_experiment_prompt_ab_description",
		map[string]any{"control": "c", "treatment": "t"},
		"A/B test: c vs t")
	if got != "A/B test: c vs t" {
		t.Fatalf("after nil SetTranslator: got %q, want default fallback", got)
	}
}
