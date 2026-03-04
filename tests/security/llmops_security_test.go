package security

import (
	"context"
	"strings"
	"testing"

	"digital.vasic.llmops/llmops"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPromptRegistryEmptyInputs_Security(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping security test in short mode")
	}

	reg := llmops.NewInMemoryPromptRegistry(nil)
	ctx := context.Background()

	// Empty name
	err := reg.Create(ctx, &llmops.PromptVersion{Name: "", Version: "1.0.0", Content: "test"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "name")

	// Empty version
	err = reg.Create(ctx, &llmops.PromptVersion{Name: "p", Version: "", Content: "test"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "version")

	// Empty content
	err = reg.Create(ctx, &llmops.PromptVersion{Name: "p", Version: "1.0.0", Content: ""})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "content")
}

func TestPromptRegistryDuplicateVersion_Security(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping security test in short mode")
	}

	reg := llmops.NewInMemoryPromptRegistry(nil)
	ctx := context.Background()

	require.NoError(t, reg.Create(ctx, &llmops.PromptVersion{
		Name: "dup", Version: "1.0.0", Content: "original",
	}))

	err := reg.Create(ctx, &llmops.PromptVersion{
		Name: "dup", Version: "1.0.0", Content: "duplicate",
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "already exists")
}

func TestPromptRegistryDeleteActiveVersion_Security(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping security test in short mode")
	}

	reg := llmops.NewInMemoryPromptRegistry(nil)
	ctx := context.Background()

	require.NoError(t, reg.Create(ctx, &llmops.PromptVersion{
		Name: "del", Version: "1.0.0", Content: "content",
	}))

	err := reg.Delete(ctx, "del", "1.0.0")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "active")
}

func TestPromptRenderMissingRequiredVariable_Security(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping security test in short mode")
	}

	reg := llmops.NewInMemoryPromptRegistry(nil)
	ctx := context.Background()

	require.NoError(t, reg.Create(ctx, &llmops.PromptVersion{
		Name: "tmpl", Version: "1.0.0", Content: "Hello {{name}}",
		Variables: []llmops.PromptVariable{
			{Name: "name", Type: "string", Required: true},
		},
	}))

	_, err := reg.Render(ctx, "tmpl", "1.0.0", map[string]interface{}{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "required")
}

func TestExperimentManagerInsufficientVariants_Security(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping security test in short mode")
	}

	mgr := llmops.NewInMemoryExperimentManager(nil)
	ctx := context.Background()

	// Zero variants
	err := mgr.Create(ctx, &llmops.Experiment{Name: "bad", Variants: nil})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "2 variants")

	// One variant
	err = mgr.Create(ctx, &llmops.Experiment{
		Name:     "bad",
		Variants: []*llmops.Variant{{Name: "only"}},
	})
	require.Error(t, err)
}

func TestExperimentInvalidStateTransitions_Security(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping security test in short mode")
	}

	mgr := llmops.NewInMemoryExperimentManager(nil)
	ctx := context.Background()

	exp := &llmops.Experiment{
		Name: "state-test",
		Variants: []*llmops.Variant{
			{Name: "A"}, {Name: "B"},
		},
	}
	require.NoError(t, mgr.Create(ctx, exp))

	// Cannot pause a draft experiment
	err := mgr.Pause(ctx, exp.ID)
	require.Error(t, err)

	// Cannot assign variant to non-running experiment
	_, err = mgr.AssignVariant(ctx, exp.ID, "user-1")
	require.Error(t, err)

	// Complete and try to cancel
	require.NoError(t, mgr.Start(ctx, exp.ID))
	require.NoError(t, mgr.Complete(ctx, exp.ID, ""))

	err = mgr.Cancel(ctx, exp.ID)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "finalized")
}

func TestExperimentNonexistentLookup_Security(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping security test in short mode")
	}

	mgr := llmops.NewInMemoryExperimentManager(nil)
	ctx := context.Background()

	_, err := mgr.Get(ctx, "nonexistent")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not found")

	err = mgr.Start(ctx, "nonexistent")
	require.Error(t, err)

	_, err = mgr.GetResults(ctx, "nonexistent")
	require.Error(t, err)
}

func TestEvaluatorRunRequiresDataset_Security(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping security test in short mode")
	}

	eval := llmops.NewInMemoryContinuousEvaluator(nil, nil, nil, nil)
	ctx := context.Background()

	// Create run with nonexistent dataset
	err := eval.CreateRun(ctx, &llmops.EvaluationRun{
		Name:    "bad-run",
		Dataset: "nonexistent",
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not found")

	// Empty name
	err = eval.CreateRun(ctx, &llmops.EvaluationRun{
		Name: "", Dataset: "ds",
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "name")
}

func TestPromptInjectionResistance_Security(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping security test in short mode")
	}

	reg := llmops.NewInMemoryPromptRegistry(nil)
	ctx := context.Background()

	maliciousContent := strings.Repeat("A", 100000)
	err := reg.Create(ctx, &llmops.PromptVersion{
		Name: "inject", Version: "1.0.0", Content: maliciousContent,
	})
	require.NoError(t, err) // Should not panic

	prompt, err := reg.Get(ctx, "inject", "1.0.0")
	require.NoError(t, err)
	assert.Equal(t, maliciousContent, prompt.Content)
}
