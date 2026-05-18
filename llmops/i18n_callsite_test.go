// CONST-035 / CONST-046 sentinel coverage for round-119 i18n migration.
// Tests guard against regressions that drop SetTranslator wiring or
// stop piping translator output into user-facing Recommendation /
// Summary fields. Per CONST-050(A) fakes are permitted in *_test.go.
package llmops

import (
	"sync"
	"testing"

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
