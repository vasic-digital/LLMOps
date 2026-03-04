package integration

import (
	"context"
	"runtime"
	"testing"

	"digital.vasic.llmops/llmops"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func init() {
	runtime.GOMAXPROCS(2)
}

func TestEvaluatorAndDatasetIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	evaluator := llmops.NewInMemoryContinuousEvaluator(nil, nil, nil, nil)

	// Create a dataset
	dataset := &llmops.Dataset{
		Name: "integration-test-dataset",
		Type: llmops.DatasetTypeGolden,
	}
	err := evaluator.CreateDataset(context.Background(), dataset)
	require.NoError(t, err)
	assert.NotEmpty(t, dataset.ID)

	// Add samples
	samples := []*llmops.DatasetSample{
		{Input: "What is Go?", ExpectedOutput: "Go is a programming language"},
		{Input: "What is Rust?", ExpectedOutput: "Rust is a systems programming language"},
	}
	err = evaluator.AddSamples(context.Background(), dataset.ID, samples)
	require.NoError(t, err)

	// Create evaluation run
	run := &llmops.EvaluationRun{
		Name:    "integration-eval-run",
		Dataset: dataset.ID,
		Metrics: []string{"accuracy", "relevance"},
	}
	err = evaluator.CreateRun(context.Background(), run)
	require.NoError(t, err)
	assert.NotEmpty(t, run.ID)

	// Get run back
	fetched, err := evaluator.GetRun(context.Background(), run.ID)
	require.NoError(t, err)
	assert.Equal(t, run.Name, fetched.Name)
	assert.Equal(t, llmops.EvaluationStatusPending, fetched.Status)
}

func TestExperimentLifecycleIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	mgr := llmops.NewInMemoryExperimentManager(nil)

	exp := &llmops.Experiment{
		Name: "integration-experiment",
		Variants: []*llmops.Variant{
			{Name: "Control", IsControl: true},
			{Name: "Treatment"},
		},
		Metrics:      []string{"quality"},
		TargetMetric: "quality",
	}
	err := mgr.Create(context.Background(), exp)
	require.NoError(t, err)
	assert.NotEmpty(t, exp.ID)
	assert.Equal(t, llmops.ExperimentStatusDraft, exp.Status)

	// Start the experiment
	err = mgr.Start(context.Background(), exp.ID)
	require.NoError(t, err)

	fetched, err := mgr.Get(context.Background(), exp.ID)
	require.NoError(t, err)
	assert.Equal(t, llmops.ExperimentStatusRunning, fetched.Status)

	// Assign variant
	variant, err := mgr.AssignVariant(context.Background(), exp.ID, "user-1")
	require.NoError(t, err)
	assert.NotNil(t, variant)

	// Record metric
	err = mgr.RecordMetric(context.Background(), exp.ID, variant.ID, "quality", 0.85)
	require.NoError(t, err)

	// Get results
	results, err := mgr.GetResults(context.Background(), exp.ID)
	require.NoError(t, err)
	assert.Equal(t, exp.ID, results.ExperimentID)
}

func TestPromptRegistryAndEvaluatorIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	registry := llmops.NewInMemoryPromptRegistry(nil)

	// Create prompt
	prompt := &llmops.PromptVersion{
		Name:    "test-prompt",
		Version: "1.0.0",
		Content: "Answer the question: {{question}}",
		Variables: []llmops.PromptVariable{
			{Name: "question", Type: "string", Required: true},
		},
	}
	err := registry.Create(context.Background(), prompt)
	require.NoError(t, err)

	// Verify it can be rendered
	rendered, err := registry.Render(context.Background(), "test-prompt", "1.0.0",
		map[string]interface{}{"question": "What is Go?"})
	require.NoError(t, err)
	assert.Contains(t, rendered, "What is Go?")

	// Create evaluator with the registry
	evaluator := llmops.NewInMemoryContinuousEvaluator(nil, registry, nil, nil)

	// Create dataset and run
	dataset := &llmops.Dataset{Name: "prompt-test-ds", Type: llmops.DatasetTypeGolden}
	err = evaluator.CreateDataset(context.Background(), dataset)
	require.NoError(t, err)

	run := &llmops.EvaluationRun{
		Name:          "prompt-eval",
		Dataset:       dataset.ID,
		PromptName:    "test-prompt",
		PromptVersion: "1.0.0",
		Metrics:       []string{"accuracy"},
	}
	err = evaluator.CreateRun(context.Background(), run)
	require.NoError(t, err)
	assert.NotEmpty(t, run.ID)
}

func TestAlertManagerIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	alertMgr := llmops.NewInMemoryAlertManager(nil)

	// Create alerts
	alert1 := &llmops.Alert{
		Type:     llmops.AlertTypeRegression,
		Severity: llmops.AlertSeverityWarning,
		Message:  "Pass rate dropped",
		Source:   "evaluation",
	}
	err := alertMgr.Create(context.Background(), alert1)
	require.NoError(t, err)
	assert.NotEmpty(t, alert1.ID)

	alert2 := &llmops.Alert{
		Type:     llmops.AlertTypeThreshold,
		Severity: llmops.AlertSeverityCritical,
		Message:  "Latency exceeded threshold",
		Source:   "monitoring",
	}
	err = alertMgr.Create(context.Background(), alert2)
	require.NoError(t, err)

	// List all alerts
	all, err := alertMgr.List(context.Background(), nil)
	require.NoError(t, err)
	assert.Equal(t, 2, len(all))

	// Filter by type
	filtered, err := alertMgr.List(context.Background(), &llmops.AlertFilter{
		Types: []llmops.AlertType{llmops.AlertTypeRegression},
	})
	require.NoError(t, err)
	assert.Equal(t, 1, len(filtered))

	// Acknowledge alert
	err = alertMgr.Acknowledge(context.Background(), alert1.ID)
	require.NoError(t, err)

	// Filter unacked
	unacked, err := alertMgr.List(context.Background(), &llmops.AlertFilter{
		Unacked: true,
	})
	require.NoError(t, err)
	assert.Equal(t, 1, len(unacked))
}

func TestExperimentPauseAndResumeIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	mgr := llmops.NewInMemoryExperimentManager(nil)

	exp := &llmops.Experiment{
		Name: "pause-resume-experiment",
		Variants: []*llmops.Variant{
			{Name: "A", IsControl: true},
			{Name: "B"},
		},
	}
	err := mgr.Create(context.Background(), exp)
	require.NoError(t, err)

	err = mgr.Start(context.Background(), exp.ID)
	require.NoError(t, err)

	// Pause
	err = mgr.Pause(context.Background(), exp.ID)
	require.NoError(t, err)

	fetched, err := mgr.Get(context.Background(), exp.ID)
	require.NoError(t, err)
	assert.Equal(t, llmops.ExperimentStatusPaused, fetched.Status)

	// Resume
	err = mgr.Start(context.Background(), exp.ID)
	require.NoError(t, err)

	fetched, err = mgr.Get(context.Background(), exp.ID)
	require.NoError(t, err)
	assert.Equal(t, llmops.ExperimentStatusRunning, fetched.Status)
}
