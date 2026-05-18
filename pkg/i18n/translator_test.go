// CONST-035 / Article XI §11.9 unit coverage for the LLMOps i18n
// translator seam. Tests assert observable behaviour (string equality on
// returned keys, behaviour invariance across nil/empty params) rather
// than presence of constructors.
package i18n_test

import (
	"sync"
	"testing"

	"digital.vasic.llmops/pkg/i18n"
)

func TestNoopTranslator_ReturnsKeyVerbatim(t *testing.T) {
	tr := i18n.NoopTranslator{}

	cases := []struct {
		key string
	}{
		{"llmops_experiment_recommendation_deploy"},
		{"llmops_experiment_recommendation_continue"},
		{"llmops_run_comparison_regressions"},
		{"llmops_run_comparison_improvements"},
		{"llmops_run_comparison_no_change"},
		{""},
		{"unknown_key_with_dots.and.colons:plus-dashes"},
	}

	for _, c := range cases {
		c := c
		t.Run(c.key, func(t *testing.T) {
			got := tr.T(c.key, nil)
			if got != c.key {
				t.Fatalf("NoopTranslator.T(%q, nil) = %q, want key verbatim", c.key, got)
			}
		})
	}
}

func TestNoopTranslator_IgnoresParams(t *testing.T) {
	tr := i18n.NoopTranslator{}
	key := "llmops_experiment_recommendation_deploy"
	params := map[string]any{
		"variant":    "B",
		"confidence": 95.4,
	}

	got := tr.T(key, params)
	if got != key {
		t.Fatalf("NoopTranslator.T(%q, params) = %q, want %q (params must be ignored)", key, got, key)
	}
}

func TestNoopTranslator_ConcurrentSafe(t *testing.T) {
	tr := i18n.NoopTranslator{}
	const goroutines = 64
	const iterations = 256

	var wg sync.WaitGroup
	wg.Add(goroutines)
	for i := 0; i < goroutines; i++ {
		go func() {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				if got := tr.T("llmops_run_comparison_no_change", nil); got != "llmops_run_comparison_no_change" {
					t.Errorf("concurrent T() returned %q", got)
					return
				}
			}
		}()
	}
	wg.Wait()
}

// fakeTranslator is a unit-test-only test double — permitted under
// CONST-050(A) because this file is *_test.go. It proves the Translator
// interface is satisfiable by a non-Noop implementation and gives the
// call-site tests a way to verify that SetTranslator actually rewires
// rendering.
type fakeTranslator struct {
	calls map[string]int
	mu    sync.Mutex
}

func (f *fakeTranslator) T(key string, _ map[string]any) string {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.calls == nil {
		f.calls = map[string]int{}
	}
	f.calls[key]++
	return "translated:" + key
}

func TestTranslator_InterfaceSatisfaction(t *testing.T) {
	var _ i18n.Translator = i18n.NoopTranslator{}
	var _ i18n.Translator = &fakeTranslator{}
}
