package benchmark

import (
	"context"
	"fmt"
	"runtime"
	"testing"

	"digital.vasic.llmops/llmops"
	"github.com/stretchr/testify/require"
)

func init() {
	runtime.GOMAXPROCS(2)
}

func BenchmarkPromptRegistryCreate(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping benchmark in short mode")
	}

	registry := llmops.NewInMemoryPromptRegistry(nil)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = registry.Create(context.Background(), &llmops.PromptVersion{
			Name:    fmt.Sprintf("bench-prompt-%d", i),
			Version: "1.0.0",
			Content: fmt.Sprintf("Prompt content %d: {{input}}", i),
			Variables: []llmops.PromptVariable{
				{Name: "input", Type: "string", Required: true},
			},
		})
	}
}

func BenchmarkPromptRegistryGet(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping benchmark in short mode")
	}

	registry := llmops.NewInMemoryPromptRegistry(nil)
	_ = registry.Create(context.Background(), &llmops.PromptVersion{
		Name:    "bench-get",
		Version: "1.0.0",
		Content: "content",
	})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = registry.Get(context.Background(), "bench-get", "1.0.0")
	}
}

func BenchmarkPromptRegistryRender(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping benchmark in short mode")
	}

	registry := llmops.NewInMemoryPromptRegistry(nil)
	_ = registry.Create(context.Background(), &llmops.PromptVersion{
		Name:    "bench-render",
		Version: "1.0.0",
		Content: "Hello {{name}}, your order {{order}} is ready.",
		Variables: []llmops.PromptVariable{
			{Name: "name", Type: "string", Required: true},
			{Name: "order", Type: "string", Required: true},
		},
	})

	vars := map[string]interface{}{
		"name":  "Alice",
		"order": "#12345",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = registry.Render(context.Background(), "bench-render", "1.0.0", vars)
	}
}

func BenchmarkPromptRegistryListAll(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping benchmark in short mode")
	}

	registry := llmops.NewInMemoryPromptRegistry(nil)
	for i := 0; i < 100; i++ {
		_ = registry.Create(context.Background(), &llmops.PromptVersion{
			Name:    fmt.Sprintf("prompt-%d", i),
			Version: "1.0.0",
			Content: "content",
		})
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = registry.ListAll(context.Background())
	}
}

func BenchmarkExperimentCreate(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping benchmark in short mode")
	}

	mgr := llmops.NewInMemoryExperimentManager(nil)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = mgr.Create(context.Background(), &llmops.Experiment{
			Name: fmt.Sprintf("bench-exp-%d", i),
			Variants: []*llmops.Variant{
				{Name: "Control", IsControl: true},
				{Name: "Treatment"},
			},
			Metrics: []string{"quality", "latency"},
		})
	}
}

func BenchmarkExperimentAssignVariant(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping benchmark in short mode")
	}

	mgr := llmops.NewInMemoryExperimentManager(nil)
	exp := &llmops.Experiment{
		Name: "bench-assign",
		Variants: []*llmops.Variant{
			{Name: "A", IsControl: true},
			{Name: "B"},
		},
	}
	err := mgr.Create(context.Background(), exp)
	require.NoError(b, err)
	err = mgr.Start(context.Background(), exp.ID)
	require.NoError(b, err)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = mgr.AssignVariant(context.Background(), exp.ID, fmt.Sprintf("user-%d", i))
	}
}

func BenchmarkExperimentRecordMetric(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping benchmark in short mode")
	}

	mgr := llmops.NewInMemoryExperimentManager(nil)
	exp := &llmops.Experiment{
		Name: "bench-metric",
		Variants: []*llmops.Variant{
			{Name: "A", IsControl: true},
			{Name: "B"},
		},
	}
	err := mgr.Create(context.Background(), exp)
	require.NoError(b, err)
	err = mgr.Start(context.Background(), exp.ID)
	require.NoError(b, err)

	variantID := exp.Variants[0].ID

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = mgr.RecordMetric(context.Background(), exp.ID, variantID,
			"quality", float64(i)/float64(b.N))
	}
}

func BenchmarkExperimentGetResults(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping benchmark in short mode")
	}

	mgr := llmops.NewInMemoryExperimentManager(nil)
	exp := &llmops.Experiment{
		Name: "bench-results",
		Variants: []*llmops.Variant{
			{Name: "A", IsControl: true},
			{Name: "B"},
		},
	}
	err := mgr.Create(context.Background(), exp)
	require.NoError(b, err)
	err = mgr.Start(context.Background(), exp.ID)
	require.NoError(b, err)

	// Record some metrics
	for i := 0; i < 50; i++ {
		_ = mgr.RecordMetric(context.Background(), exp.ID, exp.Variants[0].ID,
			"quality", 0.8+float64(i)*0.001)
		_ = mgr.RecordMetric(context.Background(), exp.ID, exp.Variants[1].ID,
			"quality", 0.75+float64(i)*0.002)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = mgr.GetResults(context.Background(), exp.ID)
	}
}

func BenchmarkDatasetCreate(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping benchmark in short mode")
	}

	evaluator := llmops.NewInMemoryContinuousEvaluator(nil, nil, nil, nil)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = evaluator.CreateDataset(context.Background(), &llmops.Dataset{
			Name: fmt.Sprintf("bench-ds-%d", i),
			Type: llmops.DatasetTypeGolden,
		})
	}
}

func BenchmarkDatasetAddSamples(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping benchmark in short mode")
	}

	evaluator := llmops.NewInMemoryContinuousEvaluator(nil, nil, nil, nil)
	ds := &llmops.Dataset{Name: "bench-samples", Type: llmops.DatasetTypeGolden}
	err := evaluator.CreateDataset(context.Background(), ds)
	require.NoError(b, err)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = evaluator.AddSamples(context.Background(), ds.ID, []*llmops.DatasetSample{
			{Input: fmt.Sprintf("input-%d", i), ExpectedOutput: "output"},
		})
	}
}

func BenchmarkEvaluationRunCreate(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping benchmark in short mode")
	}

	evaluator := llmops.NewInMemoryContinuousEvaluator(nil, nil, nil, nil)
	ds := &llmops.Dataset{Name: "bench-eval-ds", Type: llmops.DatasetTypeGolden}
	err := evaluator.CreateDataset(context.Background(), ds)
	require.NoError(b, err)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = evaluator.CreateRun(context.Background(), &llmops.EvaluationRun{
			Name:    fmt.Sprintf("bench-run-%d", i),
			Dataset: ds.ID,
			Metrics: []string{"accuracy", "relevance"},
		})
	}
}

func BenchmarkAlertCreate(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping benchmark in short mode")
	}

	alertMgr := llmops.NewInMemoryAlertManager(nil)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = alertMgr.Create(context.Background(), &llmops.Alert{
			Type:     llmops.AlertTypeRegression,
			Severity: llmops.AlertSeverityWarning,
			Message:  fmt.Sprintf("alert-%d", i),
			Source:   "benchmark",
		})
	}
}

func BenchmarkAlertListWithFilter(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping benchmark in short mode")
	}

	alertMgr := llmops.NewInMemoryAlertManager(nil)
	for i := 0; i < 200; i++ {
		severity := llmops.AlertSeverityWarning
		if i%3 == 0 {
			severity = llmops.AlertSeverityCritical
		}
		_ = alertMgr.Create(context.Background(), &llmops.Alert{
			Type:     llmops.AlertTypeRegression,
			Severity: severity,
			Message:  fmt.Sprintf("alert-%d", i),
			Source:   "benchmark",
		})
	}

	filter := &llmops.AlertFilter{
		Severities: []llmops.AlertSeverity{llmops.AlertSeverityCritical},
		Limit:      10,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = alertMgr.List(context.Background(), filter)
	}
}

func BenchmarkDefaultLLMOpsConfig(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping benchmark in short mode")
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = llmops.DefaultLLMOpsConfig()
	}
}
