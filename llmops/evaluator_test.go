package llmops

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func init() {
	runtime.GOMAXPROCS(2)
}

// --- Mock helpers for evaluator tests ---

type mockLLMEvaluator struct {
	scores map[string]float64
	err    error
}

func (m *mockLLMEvaluator) Evaluate(_ context.Context, _, _, _ string, metrics []string) (map[string]float64, error) {
	if m.err != nil {
		return nil, m.err
	}
	result := make(map[string]float64, len(metrics))
	for _, metric := range metrics {
		if score, ok := m.scores[metric]; ok {
			result[metric] = score
		} else {
			result[metric] = 0.8
		}
	}
	return result, nil
}

type mockAlertManager struct {
	mu     sync.Mutex
	alerts []*Alert
}

func (m *mockAlertManager) Create(_ context.Context, alert *Alert) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.alerts = append(m.alerts, alert)
	return nil
}

func (m *mockAlertManager) List(_ context.Context, _ *AlertFilter) ([]*Alert, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.alerts, nil
}

func (m *mockAlertManager) Acknowledge(_ context.Context, _ string) error {
	return nil
}

func (m *mockAlertManager) Subscribe(_ context.Context, _ AlertCallback) error {
	return nil
}

func (m *mockAlertManager) getAlerts() []*Alert {
	m.mu.Lock()
	defer m.mu.Unlock()
	cp := make([]*Alert, len(m.alerts))
	copy(cp, m.alerts)
	return cp
}

// --- Helper to create a ready evaluator with dataset and samples ---

func newTestEvaluator(llmEval LLMEvaluator, alertMgr AlertManager) *InMemoryContinuousEvaluator {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel) // suppress logging in tests
	registry := NewInMemoryPromptRegistry(logger)
	return NewInMemoryContinuousEvaluator(llmEval, registry, alertMgr, logger)
}

func createTestDataset(t *testing.T, eval *InMemoryContinuousEvaluator, id, name string, samples []*DatasetSample) {
	t.Helper()
	ctx := context.Background()
	ds := &Dataset{ID: id, Name: name, Type: DatasetTypeGolden}
	require.NoError(t, eval.CreateDataset(ctx, ds))
	if len(samples) > 0 {
		require.NoError(t, eval.AddSamples(ctx, id, samples))
	}
}

func defaultSamples() []*DatasetSample {
	return []*DatasetSample{
		{ID: "s1", Input: "What is 2+2?", ExpectedOutput: "4"},
		{ID: "s2", Input: "What is Go?", ExpectedOutput: "A programming language"},
		{ID: "s3", Input: "Hello", ExpectedOutput: "Hi"},
	}
}

// --- Constructor tests ---

func TestNewInMemoryContinuousEvaluator_NilLogger(t *testing.T) {
	eval := NewInMemoryContinuousEvaluator(nil, nil, nil, nil)
	require.NotNil(t, eval)
	require.NotNil(t, eval.logger)
}

func TestNewInMemoryContinuousEvaluator_WithLogger(t *testing.T) {
	logger := logrus.New()
	eval := NewInMemoryContinuousEvaluator(nil, nil, nil, logger)
	require.NotNil(t, eval)
	assert.Equal(t, logger, eval.logger)
}

func TestNewInMemoryContinuousEvaluator_WithDependencies(t *testing.T) {
	llmEval := &mockLLMEvaluator{scores: map[string]float64{"accuracy": 0.9}}
	alertMgr := &mockAlertManager{}
	registry := NewInMemoryPromptRegistry(nil)
	eval := NewInMemoryContinuousEvaluator(llmEval, registry, alertMgr, nil)
	require.NotNil(t, eval)
	assert.Equal(t, llmEval, eval.evaluator)
	assert.Equal(t, alertMgr, eval.alertManager)
}

// --- CreateDataset tests ---

func TestInMemoryContinuousEvaluator_CreateDataset_Success(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()
	ds := &Dataset{Name: "test-dataset", Type: DatasetTypeGolden}
	err := eval.CreateDataset(ctx, ds)
	require.NoError(t, err)
	assert.NotEmpty(t, ds.ID)
	assert.False(t, ds.CreatedAt.IsZero())
	assert.False(t, ds.UpdatedAt.IsZero())
}

func TestInMemoryContinuousEvaluator_CreateDataset_WithExplicitID(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()
	ds := &Dataset{ID: "my-id", Name: "test-dataset", Type: DatasetTypeBenchmark}
	err := eval.CreateDataset(ctx, ds)
	require.NoError(t, err)
	assert.Equal(t, "my-id", ds.ID)
}

func TestInMemoryContinuousEvaluator_CreateDataset_EmptyName(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()
	ds := &Dataset{Type: DatasetTypeGolden}
	err := eval.CreateDataset(ctx, ds)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "dataset name is required")
}

// --- GetDataset tests ---

func TestInMemoryContinuousEvaluator_GetDataset_Success(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()
	ds := &Dataset{ID: "ds1", Name: "test"}
	require.NoError(t, eval.CreateDataset(ctx, ds))

	got, err := eval.GetDataset(ctx, "ds1")
	require.NoError(t, err)
	assert.Equal(t, "test", got.Name)
}

func TestInMemoryContinuousEvaluator_GetDataset_NotFound(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()
	_, err := eval.GetDataset(ctx, "nonexistent")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "dataset not found")
}

// --- AddSamples tests ---

func TestInMemoryContinuousEvaluator_AddSamples_Success(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()
	ds := &Dataset{ID: "ds1", Name: "test"}
	require.NoError(t, eval.CreateDataset(ctx, ds))

	samples := []*DatasetSample{
		{ID: "s1", Input: "hello"},
		{Input: "world"}, // no ID, should be auto-generated
	}
	err := eval.AddSamples(ctx, "ds1", samples)
	require.NoError(t, err)

	got, _ := eval.GetDataset(ctx, "ds1")
	assert.Equal(t, 2, got.SampleCount)
	assert.NotEmpty(t, samples[1].ID, "auto-generated ID expected")
}

func TestInMemoryContinuousEvaluator_AddSamples_DatasetNotFound(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()
	err := eval.AddSamples(ctx, "nonexistent", []*DatasetSample{{Input: "hello"}})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "dataset not found")
}

func TestInMemoryContinuousEvaluator_AddSamples_MultipleBatches(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()
	ds := &Dataset{ID: "ds1", Name: "test"}
	require.NoError(t, eval.CreateDataset(ctx, ds))

	require.NoError(t, eval.AddSamples(ctx, "ds1", []*DatasetSample{{ID: "s1", Input: "a"}}))
	require.NoError(t, eval.AddSamples(ctx, "ds1", []*DatasetSample{{ID: "s2", Input: "b"}, {ID: "s3", Input: "c"}}))

	got, _ := eval.GetDataset(ctx, "ds1")
	assert.Equal(t, 3, got.SampleCount)
}

// --- CreateRun tests ---

func TestInMemoryContinuousEvaluator_CreateRun_Success(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()
	createTestDataset(t, eval, "ds1", "test-ds", nil)

	run := &EvaluationRun{Name: "test-run", Dataset: "ds1", Metrics: []string{"accuracy"}}
	err := eval.CreateRun(ctx, run)
	require.NoError(t, err)
	assert.NotEmpty(t, run.ID)
	assert.Equal(t, EvaluationStatusPending, run.Status)
	assert.False(t, run.CreatedAt.IsZero())
}

func TestInMemoryContinuousEvaluator_CreateRun_WithExplicitID(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()
	createTestDataset(t, eval, "ds1", "test-ds", nil)

	run := &EvaluationRun{ID: "custom-id", Name: "test-run", Dataset: "ds1"}
	err := eval.CreateRun(ctx, run)
	require.NoError(t, err)
	assert.Equal(t, "custom-id", run.ID)
}

func TestInMemoryContinuousEvaluator_CreateRun_EmptyName(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()
	createTestDataset(t, eval, "ds1", "test-ds", nil)

	run := &EvaluationRun{Dataset: "ds1"}
	err := eval.CreateRun(ctx, run)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "run name is required")
}

func TestInMemoryContinuousEvaluator_CreateRun_EmptyDataset(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()

	run := &EvaluationRun{Name: "test-run"}
	err := eval.CreateRun(ctx, run)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "dataset is required")
}

func TestInMemoryContinuousEvaluator_CreateRun_DatasetNotFound(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()

	run := &EvaluationRun{Name: "test-run", Dataset: "nonexistent"}
	err := eval.CreateRun(ctx, run)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "dataset not found")
}

// --- StartRun tests ---

func TestInMemoryContinuousEvaluator_StartRun_Success(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()
	createTestDataset(t, eval, "ds1", "test-ds", defaultSamples())

	run := &EvaluationRun{ID: "run1", Name: "test-run", Dataset: "ds1", Metrics: []string{"accuracy"}}
	require.NoError(t, eval.CreateRun(ctx, run))

	err := eval.StartRun(ctx, "run1")
	require.NoError(t, err)

	// Wait for async execution to complete
	require.Eventually(t, func() bool {
		got, _ := eval.GetRun(ctx, "run1")
		return got.Status == EvaluationStatusCompleted
	}, 5*time.Second, 50*time.Millisecond)

	got, err := eval.GetRun(ctx, "run1")
	require.NoError(t, err)
	assert.Equal(t, EvaluationStatusCompleted, got.Status)
	assert.NotNil(t, got.StartTime)
}

func TestInMemoryContinuousEvaluator_StartRun_NotFound(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()
	err := eval.StartRun(ctx, "nonexistent")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "run not found")
}

func TestInMemoryContinuousEvaluator_StartRun_AlreadyStarted(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()
	createTestDataset(t, eval, "ds1", "test-ds", defaultSamples())

	run := &EvaluationRun{ID: "run1", Name: "test-run", Dataset: "ds1"}
	require.NoError(t, eval.CreateRun(ctx, run))
	require.NoError(t, eval.StartRun(ctx, "run1"))

	// Wait for completion
	require.Eventually(t, func() bool {
		got, _ := eval.GetRun(ctx, "run1")
		return got.Status == EvaluationStatusCompleted
	}, 5*time.Second, 50*time.Millisecond)

	// Try to start again
	err := eval.StartRun(ctx, "run1")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "run already started or completed")
}

// --- GetRun tests ---

func TestInMemoryContinuousEvaluator_GetRun_Success(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()
	createTestDataset(t, eval, "ds1", "test-ds", nil)

	run := &EvaluationRun{ID: "run1", Name: "test-run", Dataset: "ds1"}
	require.NoError(t, eval.CreateRun(ctx, run))

	got, err := eval.GetRun(ctx, "run1")
	require.NoError(t, err)
	assert.Equal(t, "run1", got.ID)
	assert.Equal(t, "test-run", got.Name)
}

func TestInMemoryContinuousEvaluator_GetRun_NotFound(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()
	_, err := eval.GetRun(ctx, "nonexistent")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "run not found")
}

func TestInMemoryContinuousEvaluator_GetRun_ReturnsShallowCopy(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()
	createTestDataset(t, eval, "ds1", "test-ds", nil)

	run := &EvaluationRun{ID: "run1", Name: "test-run", Dataset: "ds1"}
	require.NoError(t, eval.CreateRun(ctx, run))

	got1, _ := eval.GetRun(ctx, "run1")
	got2, _ := eval.GetRun(ctx, "run1")
	// Verify they are different pointers (shallow copies)
	assert.NotSame(t, got1, got2)
	assert.Equal(t, got1.ID, got2.ID)
}

// --- ListRuns tests ---

func TestInMemoryContinuousEvaluator_ListRuns_NilFilter(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()
	createTestDataset(t, eval, "ds1", "test-ds", nil)

	require.NoError(t, eval.CreateRun(ctx, &EvaluationRun{ID: "r1", Name: "run1", Dataset: "ds1"}))
	time.Sleep(time.Millisecond)
	require.NoError(t, eval.CreateRun(ctx, &EvaluationRun{ID: "r2", Name: "run2", Dataset: "ds1"}))

	runs, err := eval.ListRuns(ctx, nil)
	require.NoError(t, err)
	assert.Len(t, runs, 2)
	// Newest first
	assert.Equal(t, "r2", runs[0].ID)
	assert.Equal(t, "r1", runs[1].ID)
}

func TestInMemoryContinuousEvaluator_ListRuns_EmptyFilter(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()
	createTestDataset(t, eval, "ds1", "test-ds", nil)

	require.NoError(t, eval.CreateRun(ctx, &EvaluationRun{ID: "r1", Name: "run1", Dataset: "ds1"}))

	runs, err := eval.ListRuns(ctx, &EvaluationFilter{})
	require.NoError(t, err)
	assert.Len(t, runs, 1)
}

func TestInMemoryContinuousEvaluator_ListRuns_FilterByPromptName(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()
	createTestDataset(t, eval, "ds1", "test-ds", nil)

	require.NoError(t, eval.CreateRun(ctx, &EvaluationRun{ID: "r1", Name: "run1", Dataset: "ds1", PromptName: "prompt-a"}))
	require.NoError(t, eval.CreateRun(ctx, &EvaluationRun{ID: "r2", Name: "run2", Dataset: "ds1", PromptName: "prompt-b"}))

	runs, err := eval.ListRuns(ctx, &EvaluationFilter{PromptName: "prompt-a"})
	require.NoError(t, err)
	assert.Len(t, runs, 1)
	assert.Equal(t, "r1", runs[0].ID)
}

func TestInMemoryContinuousEvaluator_ListRuns_FilterByModelName(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()
	createTestDataset(t, eval, "ds1", "test-ds", nil)

	require.NoError(t, eval.CreateRun(ctx, &EvaluationRun{ID: "r1", Name: "run1", Dataset: "ds1", ModelName: "gpt-4"}))
	require.NoError(t, eval.CreateRun(ctx, &EvaluationRun{ID: "r2", Name: "run2", Dataset: "ds1", ModelName: "claude-3"}))

	runs, err := eval.ListRuns(ctx, &EvaluationFilter{ModelName: "gpt-4"})
	require.NoError(t, err)
	assert.Len(t, runs, 1)
	assert.Equal(t, "r1", runs[0].ID)
}

func TestInMemoryContinuousEvaluator_ListRuns_FilterByStatus(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()
	createTestDataset(t, eval, "ds1", "test-ds", nil)

	require.NoError(t, eval.CreateRun(ctx, &EvaluationRun{ID: "r1", Name: "run1", Dataset: "ds1"}))

	runs, err := eval.ListRuns(ctx, &EvaluationFilter{Status: EvaluationStatusPending})
	require.NoError(t, err)
	assert.Len(t, runs, 1)

	runs, err = eval.ListRuns(ctx, &EvaluationFilter{Status: EvaluationStatusRunning})
	require.NoError(t, err)
	assert.Len(t, runs, 0)
}

func TestInMemoryContinuousEvaluator_ListRuns_FilterByTimeRange(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()
	createTestDataset(t, eval, "ds1", "test-ds", nil)

	before := time.Now().Add(-time.Second)
	require.NoError(t, eval.CreateRun(ctx, &EvaluationRun{ID: "r1", Name: "run1", Dataset: "ds1"}))
	after := time.Now().Add(time.Second)

	runs, err := eval.ListRuns(ctx, &EvaluationFilter{StartTime: &before, EndTime: &after})
	require.NoError(t, err)
	assert.Len(t, runs, 1)

	// Out of range
	farPast := time.Now().Add(-24 * time.Hour)
	almostPast := time.Now().Add(-1 * time.Hour)
	runs, err = eval.ListRuns(ctx, &EvaluationFilter{StartTime: &farPast, EndTime: &almostPast})
	require.NoError(t, err)
	assert.Len(t, runs, 0)
}

func TestInMemoryContinuousEvaluator_ListRuns_FilterByEndTimeExclusion(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()
	createTestDataset(t, eval, "ds1", "test-ds", nil)

	require.NoError(t, eval.CreateRun(ctx, &EvaluationRun{ID: "r1", Name: "run1", Dataset: "ds1"}))

	// EndTime in the past — run was created AFTER end time, so it should be excluded
	pastEnd := time.Now().Add(-1 * time.Hour)
	runs, err := eval.ListRuns(ctx, &EvaluationFilter{EndTime: &pastEnd})
	require.NoError(t, err)
	assert.Len(t, runs, 0, "run created after EndTime should be excluded")
}

func TestInMemoryContinuousEvaluator_ListRuns_FilterByStartTimeExclusion(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()
	createTestDataset(t, eval, "ds1", "test-ds", nil)

	require.NoError(t, eval.CreateRun(ctx, &EvaluationRun{ID: "r1", Name: "run1", Dataset: "ds1"}))

	// StartTime in the future — run was created BEFORE start time, so it should be excluded
	futureStart := time.Now().Add(1 * time.Hour)
	runs, err := eval.ListRuns(ctx, &EvaluationFilter{StartTime: &futureStart})
	require.NoError(t, err)
	assert.Len(t, runs, 0, "run created before StartTime should be excluded")
}

func TestInMemoryContinuousEvaluator_ListRuns_FilterByLimit(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()
	createTestDataset(t, eval, "ds1", "test-ds", nil)

	for i := 0; i < 5; i++ {
		require.NoError(t, eval.CreateRun(ctx, &EvaluationRun{
			ID: fmt.Sprintf("r%d", i), Name: fmt.Sprintf("run%d", i), Dataset: "ds1",
		}))
		time.Sleep(time.Millisecond)
	}

	runs, err := eval.ListRuns(ctx, &EvaluationFilter{Limit: 3})
	require.NoError(t, err)
	assert.Len(t, runs, 3)
}

func TestInMemoryContinuousEvaluator_ListRuns_LimitZero(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()
	createTestDataset(t, eval, "ds1", "test-ds", nil)

	require.NoError(t, eval.CreateRun(ctx, &EvaluationRun{ID: "r1", Name: "run1", Dataset: "ds1"}))

	runs, err := eval.ListRuns(ctx, &EvaluationFilter{Limit: 0})
	require.NoError(t, err)
	assert.Len(t, runs, 1, "Limit=0 should not filter")
}

func TestInMemoryContinuousEvaluator_ListRuns_ReturnsCopies(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()
	createTestDataset(t, eval, "ds1", "test-ds", nil)

	require.NoError(t, eval.CreateRun(ctx, &EvaluationRun{ID: "r1", Name: "run1", Dataset: "ds1"}))

	runs, _ := eval.ListRuns(ctx, nil)
	require.Len(t, runs, 1)
	// Modify the returned run — should not affect the original
	runs[0].Name = "modified"
	got, _ := eval.GetRun(ctx, "r1")
	assert.Equal(t, "run1", got.Name)
}

// --- ExecuteRun with LLM evaluator ---

func TestInMemoryContinuousEvaluator_ExecuteRun_WithLLMEvaluator_AllPass(t *testing.T) {
	llmEval := &mockLLMEvaluator{scores: map[string]float64{"accuracy": 0.9, "relevance": 0.85}}
	eval := newTestEvaluator(llmEval, nil)
	ctx := context.Background()
	createTestDataset(t, eval, "ds1", "test-ds", defaultSamples())

	run := &EvaluationRun{ID: "run1", Name: "test-run", Dataset: "ds1", Metrics: []string{"accuracy", "relevance"}}
	require.NoError(t, eval.CreateRun(ctx, run))
	require.NoError(t, eval.StartRun(ctx, "run1"))

	require.Eventually(t, func() bool {
		got, _ := eval.GetRun(ctx, "run1")
		return got.Status == EvaluationStatusCompleted
	}, 5*time.Second, 50*time.Millisecond)

	got, _ := eval.GetRun(ctx, "run1")
	require.NotNil(t, got.Results)
	assert.Equal(t, 3, got.Results.TotalSamples)
	assert.Equal(t, 3, got.Results.PassedSamples)
	assert.Equal(t, 0, got.Results.FailedSamples)
	assert.InDelta(t, 1.0, got.Results.PassRate, 0.001)
	assert.InDelta(t, 0.9, got.Results.MetricScores["accuracy"], 0.001)
	assert.InDelta(t, 0.85, got.Results.MetricScores["relevance"], 0.001)
}

func TestInMemoryContinuousEvaluator_ExecuteRun_WithLLMEvaluator_SomeFail(t *testing.T) {
	// Scores below 0.7 threshold cause failure
	llmEval := &mockLLMEvaluator{scores: map[string]float64{"accuracy": 0.5}}
	eval := newTestEvaluator(llmEval, nil)
	ctx := context.Background()
	createTestDataset(t, eval, "ds1", "test-ds", defaultSamples())

	run := &EvaluationRun{ID: "run1", Name: "test-run", Dataset: "ds1", Metrics: []string{"accuracy"}}
	require.NoError(t, eval.CreateRun(ctx, run))
	require.NoError(t, eval.StartRun(ctx, "run1"))

	require.Eventually(t, func() bool {
		got, _ := eval.GetRun(ctx, "run1")
		return got.Status == EvaluationStatusCompleted
	}, 5*time.Second, 50*time.Millisecond)

	got, _ := eval.GetRun(ctx, "run1")
	require.NotNil(t, got.Results)
	assert.Equal(t, 3, got.Results.FailedSamples)
	assert.InDelta(t, 0.0, got.Results.PassRate, 0.001)
}

func TestInMemoryContinuousEvaluator_ExecuteRun_WithLLMEvaluator_Error(t *testing.T) {
	llmEval := &mockLLMEvaluator{err: fmt.Errorf("evaluation failed")}
	eval := newTestEvaluator(llmEval, nil)
	ctx := context.Background()
	createTestDataset(t, eval, "ds1", "test-ds", defaultSamples())

	run := &EvaluationRun{ID: "run1", Name: "test-run", Dataset: "ds1", Metrics: []string{"accuracy"}}
	require.NoError(t, eval.CreateRun(ctx, run))
	require.NoError(t, eval.StartRun(ctx, "run1"))

	require.Eventually(t, func() bool {
		got, _ := eval.GetRun(ctx, "run1")
		return got.Status == EvaluationStatusCompleted
	}, 5*time.Second, 50*time.Millisecond)

	got, _ := eval.GetRun(ctx, "run1")
	require.NotNil(t, got.Results)
	assert.Equal(t, 3, got.Results.FailedSamples)
	assert.Contains(t, got.Results.FailureReasons, "evaluation failed")
}

func TestInMemoryContinuousEvaluator_ExecuteRun_WithoutEvaluator(t *testing.T) {
	eval := newTestEvaluator(nil, nil) // no LLM evaluator
	ctx := context.Background()
	createTestDataset(t, eval, "ds1", "test-ds", defaultSamples())

	run := &EvaluationRun{ID: "run1", Name: "test-run", Dataset: "ds1", Metrics: []string{"accuracy"}}
	require.NoError(t, eval.CreateRun(ctx, run))
	require.NoError(t, eval.StartRun(ctx, "run1"))

	require.Eventually(t, func() bool {
		got, _ := eval.GetRun(ctx, "run1")
		return got.Status == EvaluationStatusCompleted
	}, 5*time.Second, 50*time.Millisecond)

	got, _ := eval.GetRun(ctx, "run1")
	require.NotNil(t, got.Results)
	assert.Equal(t, 3, got.Results.PassedSamples, "heuristic eval always passes")
	assert.InDelta(t, 1.0, got.Results.PassRate, 0.001)
	// Heuristic eval assigns 0.8 for all metrics
	assert.InDelta(t, 0.8, got.Results.MetricScores["accuracy"], 0.001)
}

func TestInMemoryContinuousEvaluator_ExecuteRun_EmptyDataset(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()
	createTestDataset(t, eval, "ds1", "test-ds", nil) // no samples

	run := &EvaluationRun{ID: "run1", Name: "test-run", Dataset: "ds1"}
	require.NoError(t, eval.CreateRun(ctx, run))
	require.NoError(t, eval.StartRun(ctx, "run1"))

	require.Eventually(t, func() bool {
		got, _ := eval.GetRun(ctx, "run1")
		return got.Status == EvaluationStatusCompleted
	}, 5*time.Second, 50*time.Millisecond)

	got, _ := eval.GetRun(ctx, "run1")
	require.NotNil(t, got.Results)
	assert.Equal(t, 0, got.Results.TotalSamples)
}

func TestInMemoryContinuousEvaluator_ExecuteRun_WithCancelledContext(t *testing.T) {
	// Use a slow evaluator to give time for cancellation
	slowEval := &mockLLMEvaluator{scores: map[string]float64{"accuracy": 0.9}}
	eval := newTestEvaluator(slowEval, nil)

	ctx := context.Background()
	manySamples := make([]*DatasetSample, 100)
	for i := range manySamples {
		manySamples[i] = &DatasetSample{ID: fmt.Sprintf("s%d", i), Input: fmt.Sprintf("input %d", i)}
	}
	createTestDataset(t, eval, "ds1", "test-ds", manySamples)

	cancelCtx, cancel := context.WithCancel(ctx)

	run := &EvaluationRun{ID: "run1", Name: "test-run", Dataset: "ds1", Metrics: []string{"accuracy"}}
	require.NoError(t, eval.CreateRun(ctx, run))
	require.NoError(t, eval.StartRun(cancelCtx, "run1"))

	// Cancel immediately
	cancel()

	// The run should eventually complete or fail
	require.Eventually(t, func() bool {
		got, _ := eval.GetRun(ctx, "run1")
		return got.Status == EvaluationStatusCompleted || got.Status == EvaluationStatusFailed
	}, 5*time.Second, 50*time.Millisecond)
}

func TestInMemoryContinuousEvaluator_ExecuteRun_SetsEndTime(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()
	createTestDataset(t, eval, "ds1", "test-ds", defaultSamples())

	run := &EvaluationRun{ID: "run1", Name: "test-run", Dataset: "ds1"}
	require.NoError(t, eval.CreateRun(ctx, run))
	require.NoError(t, eval.StartRun(ctx, "run1"))

	require.Eventually(t, func() bool {
		got, _ := eval.GetRun(ctx, "run1")
		return got.Status == EvaluationStatusCompleted
	}, 5*time.Second, 50*time.Millisecond)

	got, _ := eval.GetRun(ctx, "run1")
	assert.NotNil(t, got.EndTime)
}

// --- CompareRuns tests ---

func TestInMemoryContinuousEvaluator_CompareRuns_Success(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()
	createTestDataset(t, eval, "ds1", "test-ds", defaultSamples())

	run1 := &EvaluationRun{ID: "r1", Name: "run1", Dataset: "ds1", Metrics: []string{"accuracy"}}
	run2 := &EvaluationRun{ID: "r2", Name: "run2", Dataset: "ds1", Metrics: []string{"accuracy"}}
	require.NoError(t, eval.CreateRun(ctx, run1))
	require.NoError(t, eval.CreateRun(ctx, run2))
	require.NoError(t, eval.StartRun(ctx, "r1"))
	require.NoError(t, eval.StartRun(ctx, "r2"))

	require.Eventually(t, func() bool {
		g1, _ := eval.GetRun(ctx, "r1")
		g2, _ := eval.GetRun(ctx, "r2")
		return g1.Status == EvaluationStatusCompleted && g2.Status == EvaluationStatusCompleted
	}, 5*time.Second, 50*time.Millisecond)

	comp, err := eval.CompareRuns(ctx, "r1", "r2")
	require.NoError(t, err)
	assert.Equal(t, "r1", comp.Run1ID)
	assert.Equal(t, "r2", comp.Run2ID)
	assert.NotEmpty(t, comp.Summary)
}

func TestInMemoryContinuousEvaluator_CompareRuns_NotFoundRun1(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()
	_, err := eval.CompareRuns(ctx, "nonexistent", "r2")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "run not found")
}

func TestInMemoryContinuousEvaluator_CompareRuns_NotFoundRun2(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()
	createTestDataset(t, eval, "ds1", "test-ds", nil)
	require.NoError(t, eval.CreateRun(ctx, &EvaluationRun{ID: "r1", Name: "run1", Dataset: "ds1"}))

	_, err := eval.CompareRuns(ctx, "r1", "nonexistent")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "run not found")
}

func TestInMemoryContinuousEvaluator_CompareRuns_NotCompleted(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()
	createTestDataset(t, eval, "ds1", "test-ds", nil)

	require.NoError(t, eval.CreateRun(ctx, &EvaluationRun{ID: "r1", Name: "run1", Dataset: "ds1"}))
	require.NoError(t, eval.CreateRun(ctx, &EvaluationRun{ID: "r2", Name: "run2", Dataset: "ds1"}))

	_, err := eval.CompareRuns(ctx, "r1", "r2")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "both runs must be completed")
}

func TestInMemoryContinuousEvaluator_CompareRuns_WithRegressions(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()
	createTestDataset(t, eval, "ds1", "test-ds", defaultSamples())

	run1 := &EvaluationRun{ID: "r1", Name: "run1", Dataset: "ds1", Metrics: []string{"accuracy"}}
	require.NoError(t, eval.CreateRun(ctx, run1))
	require.NoError(t, eval.StartRun(ctx, "r1"))

	require.Eventually(t, func() bool {
		g, _ := eval.GetRun(ctx, "r1")
		return g.Status == EvaluationStatusCompleted
	}, 5*time.Second, 50*time.Millisecond)

	// Manually set up run2 with lower scores for regression comparison
	run2 := &EvaluationRun{
		ID: "r2", Name: "run2", Dataset: "ds1",
		Status: EvaluationStatusCompleted,
		Results: &EvaluationResults{
			PassRate:     0.3,
			MetricScores: map[string]float64{"accuracy": 0.3},
		},
	}
	eval.mu.Lock()
	run2.CreatedAt = time.Now()
	eval.runs["r2"] = run2
	eval.mu.Unlock()

	comp, err := eval.CompareRuns(ctx, "r1", "r2")
	require.NoError(t, err)
	// accuracy dropped from 0.8 to 0.3, more than -5% => regression
	assert.Contains(t, comp.Regressions, "accuracy")
	assert.Contains(t, comp.Summary, "Regressions")
}

func TestInMemoryContinuousEvaluator_CompareRuns_WithImprovements(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()
	createTestDataset(t, eval, "ds1", "test-ds", defaultSamples())

	// Set up run1 with lower scores
	run1 := &EvaluationRun{
		ID: "r1", Name: "run1", Dataset: "ds1",
		Status: EvaluationStatusCompleted,
		Results: &EvaluationResults{
			PassRate:     0.5,
			MetricScores: map[string]float64{"accuracy": 0.5},
		},
	}
	eval.mu.Lock()
	run1.CreatedAt = time.Now()
	eval.runs["r1"] = run1
	eval.mu.Unlock()

	// Set up run2 with higher scores
	run2 := &EvaluationRun{
		ID: "r2", Name: "run2", Dataset: "ds1",
		Status: EvaluationStatusCompleted,
		Results: &EvaluationResults{
			PassRate:     0.95,
			MetricScores: map[string]float64{"accuracy": 0.95},
		},
	}
	eval.mu.Lock()
	run2.CreatedAt = time.Now()
	eval.runs["r2"] = run2
	eval.mu.Unlock()

	comp, err := eval.CompareRuns(ctx, "r1", "r2")
	require.NoError(t, err)
	assert.Contains(t, comp.Improvements, "accuracy")
	assert.Contains(t, comp.Summary, "Improvements")
}

func TestInMemoryContinuousEvaluator_CompareRuns_NoSignificantChanges(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()

	// Two runs with identical scores
	run1 := &EvaluationRun{
		ID: "r1", Name: "run1", Dataset: "ds1",
		Status: EvaluationStatusCompleted,
		Results: &EvaluationResults{
			PassRate:     0.8,
			MetricScores: map[string]float64{"accuracy": 0.8},
		},
	}
	run2 := &EvaluationRun{
		ID: "r2", Name: "run2", Dataset: "ds1",
		Status: EvaluationStatusCompleted,
		Results: &EvaluationResults{
			PassRate:     0.8,
			MetricScores: map[string]float64{"accuracy": 0.8},
		},
	}
	eval.mu.Lock()
	run1.CreatedAt = time.Now()
	run2.CreatedAt = time.Now()
	eval.runs["r1"] = run1
	eval.runs["r2"] = run2
	eval.mu.Unlock()

	comp, err := eval.CompareRuns(ctx, "r1", "r2")
	require.NoError(t, err)
	assert.Equal(t, "No significant changes", comp.Summary)
	assert.Empty(t, comp.Regressions)
	assert.Empty(t, comp.Improvements)
}

// --- ScheduleRun tests ---

func TestInMemoryContinuousEvaluator_ScheduleRun_Success(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()
	createTestDataset(t, eval, "ds1", "test-ds", defaultSamples())

	run := &EvaluationRun{Name: "scheduled-run", Dataset: "ds1", Metrics: []string{"accuracy"}}
	err := eval.ScheduleRun(ctx, run, "0 * * * *")
	require.NoError(t, err)
	assert.NotEmpty(t, run.ID)

	eval.mu.RLock()
	sched, ok := eval.schedules[run.ID]
	eval.mu.RUnlock()
	assert.True(t, ok)
	assert.Equal(t, "0 * * * *", sched.cron)

	// Clean up: close the stop channel to terminate the scheduler goroutine
	close(sched.stopCh)
}

func TestInMemoryContinuousEvaluator_ScheduleRun_InvalidRun(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()

	run := &EvaluationRun{Name: "scheduled-run", Dataset: "nonexistent"}
	err := eval.ScheduleRun(ctx, run, "0 * * * *")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "dataset not found")
}

// --- Regression detection tests ---

func TestInMemoryContinuousEvaluator_CheckForRegressions_NoAlertManager(t *testing.T) {
	eval := newTestEvaluator(nil, nil) // no alert manager
	ctx := context.Background()
	createTestDataset(t, eval, "ds1", "test-ds", defaultSamples())

	run := &EvaluationRun{ID: "run1", Name: "test-run", Dataset: "ds1", Metrics: []string{"accuracy"}}
	require.NoError(t, eval.CreateRun(ctx, run))
	require.NoError(t, eval.StartRun(ctx, "run1"))

	require.Eventually(t, func() bool {
		got, _ := eval.GetRun(ctx, "run1")
		return got.Status == EvaluationStatusCompleted
	}, 5*time.Second, 50*time.Millisecond)
	// No panic — gracefully skipped
}

func TestInMemoryContinuousEvaluator_CheckForRegressions_PassRateWarning(t *testing.T) {
	alertMgr := &mockAlertManager{}
	eval := newTestEvaluator(nil, alertMgr)
	ctx := context.Background()
	createTestDataset(t, eval, "ds1", "test-ds", defaultSamples())

	// Create and complete a "previous" run with high pass rate
	prevRun := &EvaluationRun{
		ID: "prev", Name: "prev-run", Dataset: "ds1",
		PromptName: "test-prompt", ModelName: "gpt-4",
		Status: EvaluationStatusCompleted,
		Results: &EvaluationResults{
			PassRate:     0.95,
			MetricScores: map[string]float64{"accuracy": 0.95},
		},
	}
	eval.mu.Lock()
	prevRun.CreatedAt = time.Now().Add(-time.Hour)
	eval.runs["prev"] = prevRun
	eval.mu.Unlock()

	// Create a new run with same prompt/model but lower scores using a low-scoring evaluator
	lowEval := &mockLLMEvaluator{scores: map[string]float64{"accuracy": 0.5}}
	eval.evaluator = lowEval

	run := &EvaluationRun{
		ID: "run1", Name: "test-run", Dataset: "ds1",
		PromptName: "test-prompt", ModelName: "gpt-4",
		Metrics: []string{"accuracy"},
	}
	require.NoError(t, eval.CreateRun(ctx, run))
	require.NoError(t, eval.StartRun(ctx, "run1"))

	require.Eventually(t, func() bool {
		got, _ := eval.GetRun(ctx, "run1")
		return got.Status == EvaluationStatusCompleted
	}, 5*time.Second, 50*time.Millisecond)

	// Check that regression alerts were triggered
	alerts := alertMgr.getAlerts()
	assert.NotEmpty(t, alerts, "expected regression alerts")

	foundPassRateAlert := false
	for _, a := range alerts {
		if a.Type == AlertTypeRegression {
			foundPassRateAlert = true
			break
		}
	}
	assert.True(t, foundPassRateAlert, "expected pass rate regression alert")
}

func TestInMemoryContinuousEvaluator_CheckForRegressions_CriticalSeverity(t *testing.T) {
	alertMgr := &mockAlertManager{}
	eval := newTestEvaluator(nil, alertMgr)
	ctx := context.Background()
	createTestDataset(t, eval, "ds1", "test-ds", defaultSamples())

	// Previous run with very high pass rate
	prevRun := &EvaluationRun{
		ID: "prev", Name: "prev-run", Dataset: "ds1",
		PromptName: "test-prompt", ModelName: "gpt-4",
		Status: EvaluationStatusCompleted,
		Results: &EvaluationResults{
			PassRate:     1.0,
			MetricScores: map[string]float64{"accuracy": 1.0},
		},
	}
	eval.mu.Lock()
	prevRun.CreatedAt = time.Now().Add(-time.Hour)
	eval.runs["prev"] = prevRun
	eval.mu.Unlock()

	// Run with scores that cause > 10% regression in pass rate
	lowEval := &mockLLMEvaluator{scores: map[string]float64{"accuracy": 0.5}}
	eval.evaluator = lowEval

	run := &EvaluationRun{
		ID: "run1", Name: "test-run", Dataset: "ds1",
		PromptName: "test-prompt", ModelName: "gpt-4",
		Metrics: []string{"accuracy"},
	}
	require.NoError(t, eval.CreateRun(ctx, run))
	require.NoError(t, eval.StartRun(ctx, "run1"))

	require.Eventually(t, func() bool {
		got, _ := eval.GetRun(ctx, "run1")
		return got.Status == EvaluationStatusCompleted
	}, 5*time.Second, 50*time.Millisecond)

	alerts := alertMgr.getAlerts()
	foundCritical := false
	for _, a := range alerts {
		if a.Severity == AlertSeverityCritical {
			foundCritical = true
			break
		}
	}
	assert.True(t, foundCritical, "expected critical severity for > 10% regression")
}

func TestInMemoryContinuousEvaluator_CheckForRegressions_NoPreviousRun(t *testing.T) {
	alertMgr := &mockAlertManager{}
	eval := newTestEvaluator(nil, alertMgr)
	ctx := context.Background()
	createTestDataset(t, eval, "ds1", "test-ds", defaultSamples())

	run := &EvaluationRun{ID: "run1", Name: "test-run", Dataset: "ds1", Metrics: []string{"accuracy"}}
	require.NoError(t, eval.CreateRun(ctx, run))
	require.NoError(t, eval.StartRun(ctx, "run1"))

	require.Eventually(t, func() bool {
		got, _ := eval.GetRun(ctx, "run1")
		return got.Status == EvaluationStatusCompleted
	}, 5*time.Second, 50*time.Millisecond)

	// No regression alerts expected when no previous run exists
	alerts := alertMgr.getAlerts()
	assert.Empty(t, alerts)
}

// --- Concurrent access tests ---

func TestInMemoryContinuousEvaluator_ConcurrentCreateRuns(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()
	createTestDataset(t, eval, "ds1", "test-ds", nil)

	var wg sync.WaitGroup
	n := 20
	errs := make([]error, n)

	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			run := &EvaluationRun{
				ID:      fmt.Sprintf("concurrent-run-%d", idx),
				Name:    fmt.Sprintf("run-%d", idx),
				Dataset: "ds1",
			}
			errs[idx] = eval.CreateRun(ctx, run)
		}(i)
	}
	wg.Wait()

	for i, err := range errs {
		assert.NoError(t, err, "run %d", i)
	}

	runs, _ := eval.ListRuns(ctx, nil)
	assert.Len(t, runs, n)
}

func TestInMemoryContinuousEvaluator_ConcurrentCreateDatasets(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()

	var wg sync.WaitGroup
	n := 20
	errs := make([]error, n)

	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			ds := &Dataset{
				ID:   fmt.Sprintf("ds-%d", idx),
				Name: fmt.Sprintf("dataset-%d", idx),
			}
			errs[idx] = eval.CreateDataset(ctx, ds)
		}(i)
	}
	wg.Wait()

	for i, err := range errs {
		assert.NoError(t, err, "dataset %d", i)
	}
}

func TestInMemoryContinuousEvaluator_ConcurrentReadWrite(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()
	createTestDataset(t, eval, "ds1", "test-ds", defaultSamples())

	// Create some runs
	for i := 0; i < 5; i++ {
		require.NoError(t, eval.CreateRun(ctx, &EvaluationRun{
			ID: fmt.Sprintf("r%d", i), Name: fmt.Sprintf("run-%d", i), Dataset: "ds1",
		}))
	}

	var wg sync.WaitGroup

	// Concurrent readers
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _ = eval.ListRuns(ctx, nil)
			_, _ = eval.GetRun(ctx, "r0")
		}()
	}

	// Concurrent writers
	for i := 5; i < 15; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			_ = eval.CreateRun(ctx, &EvaluationRun{
				ID: fmt.Sprintf("rw%d", idx), Name: fmt.Sprintf("run-rw-%d", idx), Dataset: "ds1",
			})
		}(i)
	}

	wg.Wait()
	// No race conditions or panics
}

// --- matchesFilter tests (indirectly via ListRuns, but also edge cases) ---

func TestInMemoryContinuousEvaluator_MatchesFilter_CombinedFilters(t *testing.T) {
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()
	createTestDataset(t, eval, "ds1", "test-ds", nil)

	require.NoError(t, eval.CreateRun(ctx, &EvaluationRun{
		ID: "r1", Name: "run1", Dataset: "ds1", PromptName: "p1", ModelName: "m1",
	}))
	require.NoError(t, eval.CreateRun(ctx, &EvaluationRun{
		ID: "r2", Name: "run2", Dataset: "ds1", PromptName: "p1", ModelName: "m2",
	}))
	require.NoError(t, eval.CreateRun(ctx, &EvaluationRun{
		ID: "r3", Name: "run3", Dataset: "ds1", PromptName: "p2", ModelName: "m1",
	}))

	runs, err := eval.ListRuns(ctx, &EvaluationFilter{PromptName: "p1", ModelName: "m1"})
	require.NoError(t, err)
	assert.Len(t, runs, 1)
	assert.Equal(t, "r1", runs[0].ID)
}

// --- Prompt template integration in executeRun ---

func TestInMemoryContinuousEvaluator_ExecuteRun_WithPromptTemplate(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	registry := NewInMemoryPromptRegistry(logger)
	ctx := context.Background()

	// Create a prompt version
	require.NoError(t, registry.Create(ctx, &PromptVersion{
		Name:    "test-prompt",
		Version: "1.0",
		Content: "Please answer: {{question}}",
	}))

	eval := NewInMemoryContinuousEvaluator(nil, registry, nil, logger)
	createTestDataset(t, eval, "ds1", "test-ds", defaultSamples())

	run := &EvaluationRun{
		ID: "run1", Name: "test-run", Dataset: "ds1",
		PromptName: "test-prompt", PromptVersion: "1.0",
		Metrics: []string{"accuracy"},
	}
	require.NoError(t, eval.CreateRun(ctx, run))
	require.NoError(t, eval.StartRun(ctx, "run1"))

	require.Eventually(t, func() bool {
		got, _ := eval.GetRun(ctx, "run1")
		return got.Status == EvaluationStatusCompleted
	}, 5*time.Second, 50*time.Millisecond)

	got, _ := eval.GetRun(ctx, "run1")
	assert.Equal(t, EvaluationStatusCompleted, got.Status)
}

func TestInMemoryContinuousEvaluator_ExecuteRun_WithLatestPromptVersion(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	registry := NewInMemoryPromptRegistry(logger)
	ctx := context.Background()

	require.NoError(t, registry.Create(ctx, &PromptVersion{
		Name:    "test-prompt",
		Version: "1.0",
		Content: "Answer: {{question}}",
	}))

	eval := NewInMemoryContinuousEvaluator(nil, registry, nil, logger)
	createTestDataset(t, eval, "ds1", "test-ds", defaultSamples())

	// PromptVersion left empty => should use "latest"
	run := &EvaluationRun{
		ID: "run1", Name: "test-run", Dataset: "ds1",
		PromptName: "test-prompt",
		Metrics:    []string{"accuracy"},
	}
	require.NoError(t, eval.CreateRun(ctx, run))
	require.NoError(t, eval.StartRun(ctx, "run1"))

	require.Eventually(t, func() bool {
		got, _ := eval.GetRun(ctx, "run1")
		return got.Status == EvaluationStatusCompleted
	}, 5*time.Second, 50*time.Millisecond)
}

// --- Interface compliance ---

func TestInMemoryContinuousEvaluator_ImplementsContinuousEvaluator(t *testing.T) {
	var _ ContinuousEvaluator = (*InMemoryContinuousEvaluator)(nil)
}

func TestInMemoryContinuousEvaluator_ImplementsDatasetManager(t *testing.T) {
	// Verify key dataset methods exist (not a full interface — just functional check)
	eval := newTestEvaluator(nil, nil)
	ctx := context.Background()
	ds := &Dataset{ID: "ds1", Name: "test"}
	require.NoError(t, eval.CreateDataset(ctx, ds))
	got, err := eval.GetDataset(ctx, "ds1")
	require.NoError(t, err)
	assert.Equal(t, "ds1", got.ID)
}
