package stress

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"

	"digital.vasic.llmops/llmops"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func init() {
	runtime.GOMAXPROCS(2)
}

func TestConcurrentEvaluationRunCreation_Stress(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping stress test in short mode")  // SKIP-OK: #short-mode
	}

	const goroutines = 75

	evaluator := llmops.NewInMemoryContinuousEvaluator(nil, nil, nil, nil)

	// Create a shared dataset
	dataset := &llmops.Dataset{Name: "stress-dataset", Type: llmops.DatasetTypeGolden}
	err := evaluator.CreateDataset(context.Background(), dataset)
	require.NoError(t, err)

	var wg sync.WaitGroup
	var successCount atomic.Int64

	wg.Add(goroutines)
	for i := 0; i < goroutines; i++ {
		go func(idx int) {
			defer wg.Done()

			run := &llmops.EvaluationRun{
				Name:    fmt.Sprintf("stress-run-%d", idx),
				Dataset: dataset.ID,
				Metrics: []string{"accuracy"},
			}
			if err := evaluator.CreateRun(context.Background(), run); err == nil {
				successCount.Add(1)
			}
		}(i)
	}

	wg.Wait()
	assert.Equal(t, int64(goroutines), successCount.Load(),
		"all runs should be created successfully")
}

func TestConcurrentExperimentCreation_Stress(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping stress test in short mode")  // SKIP-OK: #short-mode
	}

	const goroutines = 60

	mgr := llmops.NewInMemoryExperimentManager(nil)

	var wg sync.WaitGroup
	var successCount atomic.Int64

	wg.Add(goroutines)
	for i := 0; i < goroutines; i++ {
		go func(idx int) {
			defer wg.Done()

			exp := &llmops.Experiment{
				Name: fmt.Sprintf("stress-exp-%d", idx),
				Variants: []*llmops.Variant{
					{Name: fmt.Sprintf("ctrl-%d", idx), IsControl: true},
					{Name: fmt.Sprintf("treat-%d", idx)},
				},
				Metrics: []string{"quality"},
			}
			if err := mgr.Create(context.Background(), exp); err == nil {
				successCount.Add(1)
			}
		}(i)
	}

	wg.Wait()
	assert.Equal(t, int64(goroutines), successCount.Load())
}

func TestConcurrentExperimentLifecycle_Stress(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping stress test in short mode")  // SKIP-OK: #short-mode
	}

	const goroutines = 50

	mgr := llmops.NewInMemoryExperimentManager(nil)

	var wg sync.WaitGroup
	var completedCount atomic.Int64

	wg.Add(goroutines)
	for i := 0; i < goroutines; i++ {
		go func(idx int) {
			defer wg.Done()

			exp := &llmops.Experiment{
				Name: fmt.Sprintf("lifecycle-%d", idx),
				Variants: []*llmops.Variant{
					{Name: "A", IsControl: true},
					{Name: "B"},
				},
			}
			if err := mgr.Create(context.Background(), exp); err != nil {
				return
			}
			if err := mgr.Start(context.Background(), exp.ID); err != nil {
				return
			}

			// Assign variants and record metrics
			for j := 0; j < 5; j++ {
				variant, err := mgr.AssignVariant(
					context.Background(), exp.ID, fmt.Sprintf("user-%d-%d", idx, j))
				if err != nil {
					continue
				}
				_ = mgr.RecordMetric(
					context.Background(), exp.ID, variant.ID, "quality", 0.8)
			}

			if err := mgr.Complete(context.Background(), exp.ID, ""); err == nil {
				completedCount.Add(1)
			}
		}(i)
	}

	wg.Wait()
	assert.Equal(t, int64(goroutines), completedCount.Load())
}

func TestConcurrentPromptOperations_Stress(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping stress test in short mode")  // SKIP-OK: #short-mode
	}

	const goroutines = 80

	registry := llmops.NewInMemoryPromptRegistry(nil)

	var wg sync.WaitGroup
	var successCount atomic.Int64

	wg.Add(goroutines)
	for i := 0; i < goroutines; i++ {
		go func(idx int) {
			defer wg.Done()

			prompt := &llmops.PromptVersion{
				Name:    fmt.Sprintf("stress-prompt-%d", idx),
				Version: "1.0.0",
				Content: fmt.Sprintf("This is prompt %d: {{input}}", idx),
				Variables: []llmops.PromptVariable{
					{Name: "input", Type: "string", Required: true},
				},
			}
			if err := registry.Create(context.Background(), prompt); err == nil {
				successCount.Add(1)
			}
		}(i)
	}

	wg.Wait()
	assert.Equal(t, int64(goroutines), successCount.Load())

	// Verify all were created
	all, err := registry.ListAll(context.Background())
	require.NoError(t, err)
	assert.Equal(t, goroutines, len(all))
}

func TestConcurrentAlertCreation_Stress(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping stress test in short mode")  // SKIP-OK: #short-mode
	}

	const goroutines = 100

	alertMgr := llmops.NewInMemoryAlertManager(nil)

	var wg sync.WaitGroup
	var successCount atomic.Int64

	wg.Add(goroutines)
	for i := 0; i < goroutines; i++ {
		go func(idx int) {
			defer wg.Done()

			alert := &llmops.Alert{
				Type:     llmops.AlertTypeRegression,
				Severity: llmops.AlertSeverityWarning,
				Message:  fmt.Sprintf("alert-%d", idx),
				Source:   "stress-test",
			}
			if err := alertMgr.Create(context.Background(), alert); err == nil {
				successCount.Add(1)
			}
		}(i)
	}

	wg.Wait()
	assert.Equal(t, int64(goroutines), successCount.Load())

	all, err := alertMgr.List(context.Background(), nil)
	require.NoError(t, err)
	assert.Equal(t, goroutines, len(all))
}

func TestConcurrentMetricRecording_Stress(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping stress test in short mode")  // SKIP-OK: #short-mode
	}

	const goroutines = 100

	mgr := llmops.NewInMemoryExperimentManager(nil)

	exp := &llmops.Experiment{
		Name: "metric-stress",
		Variants: []*llmops.Variant{
			{Name: "A", IsControl: true},
			{Name: "B"},
		},
	}
	err := mgr.Create(context.Background(), exp)
	require.NoError(t, err)

	err = mgr.Start(context.Background(), exp.ID)
	require.NoError(t, err)

	variantA := exp.Variants[0]
	variantB := exp.Variants[1]

	var wg sync.WaitGroup
	var successCount atomic.Int64

	wg.Add(goroutines)
	for i := 0; i < goroutines; i++ {
		go func(idx int) {
			defer wg.Done()

			variant := variantA
			if idx%2 == 0 {
				variant = variantB
			}
			if err := mgr.RecordMetric(
				context.Background(), exp.ID, variant.ID,
				"quality", float64(idx)/100.0,
			); err == nil {
				successCount.Add(1)
			}
		}(i)
	}

	wg.Wait()
	assert.Equal(t, int64(goroutines), successCount.Load())
}

func TestConcurrentDatasetOperations_Stress(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping stress test in short mode")  // SKIP-OK: #short-mode
	}

	const goroutines = 60

	evaluator := llmops.NewInMemoryContinuousEvaluator(nil, nil, nil, nil)

	var wg sync.WaitGroup
	var successCount atomic.Int64

	wg.Add(goroutines)
	for i := 0; i < goroutines; i++ {
		go func(idx int) {
			defer wg.Done()

			ds := &llmops.Dataset{
				Name: fmt.Sprintf("stress-ds-%d", idx),
				Type: llmops.DatasetTypeGolden,
			}
			if err := evaluator.CreateDataset(context.Background(), ds); err != nil {
				return
			}

			samples := []*llmops.DatasetSample{
				{Input: fmt.Sprintf("input-%d-a", idx)},
				{Input: fmt.Sprintf("input-%d-b", idx)},
			}
			if err := evaluator.AddSamples(context.Background(), ds.ID, samples); err == nil {
				successCount.Add(1)
			}
		}(i)
	}

	wg.Wait()
	assert.Equal(t, int64(goroutines), successCount.Load())
}
