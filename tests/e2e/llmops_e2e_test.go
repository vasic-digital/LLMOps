package e2e

import (
	"context"
	"fmt"
	"runtime"
	"testing"
	"time"

	"digital.vasic.llmops/llmops"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func init() {
	runtime.GOMAXPROCS(2)
}

func TestFullEvaluationWorkflow_E2E(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping e2e test in short mode")
	}

	// Initialize the full system
	system := llmops.NewLLMOpsSystem(nil, nil)
	err := system.Initialize()
	require.NoError(t, err)

	registry := system.GetPromptRegistry()
	require.NotNil(t, registry)

	evaluator := system.GetEvaluator()
	require.NotNil(t, evaluator)

	experimentMgr := system.GetExperimentManager()
	require.NotNil(t, experimentMgr)

	alertMgr := system.GetAlertManager()
	require.NotNil(t, alertMgr)

	// Create a prompt version
	prompt := &llmops.PromptVersion{
		Name:    "qa-prompt",
		Version: "1.0.0",
		Content: "Answer the following question: {{question}}",
		Variables: []llmops.PromptVariable{
			{Name: "question", Type: "string", Required: true},
		},
	}
	err = registry.Create(context.Background(), prompt)
	require.NoError(t, err)

	// Create a second version
	promptV2 := &llmops.PromptVersion{
		Name:    "qa-prompt",
		Version: "2.0.0",
		Content: "You are a helpful AI assistant. Answer: {{question}}",
		Variables: []llmops.PromptVariable{
			{Name: "question", Type: "string", Required: true},
		},
	}
	err = registry.Create(context.Background(), promptV2)
	require.NoError(t, err)

	// Activate v2
	err = registry.Activate(context.Background(), "qa-prompt", "2.0.0")
	require.NoError(t, err)

	// Verify latest
	latest, err := registry.GetLatest(context.Background(), "qa-prompt")
	require.NoError(t, err)
	assert.Equal(t, "2.0.0", latest.Version)
	assert.True(t, latest.IsActive)
}

func TestFullExperimentWorkflow_E2E(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping e2e test in short mode")
	}

	system := llmops.NewLLMOpsSystem(nil, nil)
	err := system.Initialize()
	require.NoError(t, err)

	registry := system.GetPromptRegistry()
	experimentMgr := system.GetExperimentManager()

	// Create two prompt versions for the experiment
	controlPrompt := &llmops.PromptVersion{
		Name:    "exp-control",
		Version: "1.0.0",
		Content: "Answer: {{q}}",
		Variables: []llmops.PromptVariable{
			{Name: "q", Type: "string", Required: true},
		},
	}
	err = registry.Create(context.Background(), controlPrompt)
	require.NoError(t, err)

	treatmentPrompt := &llmops.PromptVersion{
		Name:    "exp-treatment",
		Version: "1.0.0",
		Content: "You are an expert. Answer: {{q}}",
		Variables: []llmops.PromptVariable{
			{Name: "q", Type: "string", Required: true},
		},
	}
	err = registry.Create(context.Background(), treatmentPrompt)
	require.NoError(t, err)

	// Create experiment via system
	exp, err := system.CreatePromptExperiment(
		context.Background(),
		"prompt-ab-test",
		controlPrompt,
		treatmentPrompt,
		0.5,
	)
	require.NoError(t, err)
	assert.NotEmpty(t, exp.ID)
	assert.Equal(t, 2, len(exp.Variants))

	// Start experiment
	err = experimentMgr.Start(context.Background(), exp.ID)
	require.NoError(t, err)

	// Simulate traffic
	for i := 0; i < 10; i++ {
		variant, err := experimentMgr.AssignVariant(
			context.Background(), exp.ID, fmt.Sprintf("user-%d", i))
		require.NoError(t, err)
		require.NotNil(t, variant)

		_ = experimentMgr.RecordMetric(
			context.Background(), exp.ID, variant.ID, "quality", 0.75+float64(i)*0.01)
	}

	// Get results
	results, err := experimentMgr.GetResults(context.Background(), exp.ID)
	require.NoError(t, err)
	assert.Greater(t, results.TotalSamples, 0)

	// Complete experiment
	err = experimentMgr.Complete(context.Background(), exp.ID, "")
	require.NoError(t, err)

	fetched, err := experimentMgr.Get(context.Background(), exp.ID)
	require.NoError(t, err)
	assert.Equal(t, llmops.ExperimentStatusCompleted, fetched.Status)
}

func TestModelExperimentWorkflow_E2E(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping e2e test in short mode")
	}

	system := llmops.NewLLMOpsSystem(nil, nil)
	err := system.Initialize()
	require.NoError(t, err)

	exp, err := system.CreateModelExperiment(
		context.Background(),
		"model-comparison",
		[]string{"gpt-4", "claude-3"},
		map[string]interface{}{"temperature": 0.7},
	)
	require.NoError(t, err)
	assert.NotEmpty(t, exp.ID)
	assert.Equal(t, 2, len(exp.Variants))
	assert.Equal(t, "gpt-4", exp.Variants[0].ModelName)
	assert.Equal(t, "claude-3", exp.Variants[1].ModelName)
}

func TestDefaultLLMOpsConfig_E2E(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping e2e test in short mode")
	}

	cfg := llmops.DefaultLLMOpsConfig()
	assert.True(t, cfg.EnableAutoEvaluation)
	assert.Equal(t, 24*time.Hour, cfg.EvaluationInterval)
	assert.Equal(t, 100, cfg.MinSamplesForSignif)
	assert.True(t, cfg.EnableDebateEvaluation)
	assert.NotEmpty(t, cfg.AlertThresholds)
}

func TestPromptVersionComparison_E2E(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping e2e test in short mode")
	}

	registry := llmops.NewInMemoryPromptRegistry(nil)

	v1 := &llmops.PromptVersion{
		Name:    "compare-prompt",
		Version: "1.0.0",
		Content: "Hello {{name}}, how are you?",
		Variables: []llmops.PromptVariable{
			{Name: "name", Type: "string", Required: true},
		},
	}
	err := registry.Create(context.Background(), v1)
	require.NoError(t, err)

	v2 := &llmops.PromptVersion{
		Name:    "compare-prompt",
		Version: "2.0.0",
		Content: "Greetings {{name}}! How may I help you today? {{topic}}",
		Variables: []llmops.PromptVariable{
			{Name: "name", Type: "string", Required: true},
			{Name: "topic", Type: "string", Required: false},
		},
	}
	err = registry.Create(context.Background(), v2)
	require.NoError(t, err)

	comparator := llmops.NewPromptVersionComparator(registry, nil)
	diff, err := comparator.Compare(context.Background(), "compare-prompt", "1.0.0", "2.0.0")
	require.NoError(t, err)
	assert.Equal(t, "1.0.0", diff.OldVersion)
	assert.Equal(t, "2.0.0", diff.NewVersion)
	assert.Contains(t, diff.AddedVars, "topic")
	assert.NotEmpty(t, diff.ContentDiff)
}

func TestVerifierIntegration_E2E(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping e2e test in short mode")
	}

	scores := map[string]float64{
		"openai":  0.9,
		"claude":  0.85,
		"gemini":  0.7,
		"offline": 0.95,
	}
	healthy := map[string]bool{
		"openai":  true,
		"claude":  true,
		"gemini":  true,
		"offline": false,
	}

	vi := llmops.NewVerifierIntegration(
		func(name string) float64 { return scores[name] },
		func(name string) bool { return healthy[name] },
		nil,
	)

	best, score := vi.SelectBestProvider([]string{"openai", "claude", "gemini", "offline"})
	assert.Equal(t, "openai", best)
	assert.Equal(t, 0.9, score)
}

func TestEvaluationRunListWithFilter_E2E(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping e2e test in short mode")
	}

	evaluator := llmops.NewInMemoryContinuousEvaluator(nil, nil, nil, nil)

	dataset := &llmops.Dataset{Name: "filter-ds", Type: llmops.DatasetTypeBenchmark}
	err := evaluator.CreateDataset(context.Background(), dataset)
	require.NoError(t, err)

	for i := 0; i < 5; i++ {
		run := &llmops.EvaluationRun{
			Name:      fmt.Sprintf("run-%d", i),
			Dataset:   dataset.ID,
			ModelName: "test-model",
			Metrics:   []string{"accuracy"},
		}
		err = evaluator.CreateRun(context.Background(), run)
		require.NoError(t, err)
	}

	// List with filter
	runs, err := evaluator.ListRuns(context.Background(), &llmops.EvaluationFilter{
		ModelName: "test-model",
		Limit:     3,
	})
	require.NoError(t, err)
	assert.Equal(t, 3, len(runs))
}
