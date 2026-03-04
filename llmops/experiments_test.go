package llmops

import (
	"context"
	"fmt"
	"hash/fnv"
	"math"
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

// --- Helpers ---

func newTestExperimentManager() *InMemoryExperimentManager {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	return NewInMemoryExperimentManager(logger)
}

func defaultVariants() []*Variant {
	return []*Variant{
		{ID: "control", Name: "Control", IsControl: true, ModelName: "gpt-4"},
		{ID: "treatment", Name: "Treatment", ModelName: "gpt-4-turbo"},
	}
}

func defaultExperiment(id, name string) *Experiment {
	return &Experiment{
		ID:           id,
		Name:         name,
		Variants:     defaultVariants(),
		Metrics:      []string{"latency", "quality"},
		TargetMetric: "quality",
	}
}

// --- Constructor tests ---

func TestNewInMemoryExperimentManager_NilLogger(t *testing.T) {
	mgr := NewInMemoryExperimentManager(nil)
	require.NotNil(t, mgr)
	require.NotNil(t, mgr.logger)
}

func TestNewInMemoryExperimentManager_WithLogger(t *testing.T) {
	logger := logrus.New()
	mgr := NewInMemoryExperimentManager(logger)
	require.NotNil(t, mgr)
	assert.Equal(t, logger, mgr.logger)
}

// --- Create tests ---

func TestInMemoryExperimentManager_Create_Success(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	exp := defaultExperiment("", "test-exp")
	err := mgr.Create(ctx, exp)
	require.NoError(t, err)
	assert.NotEmpty(t, exp.ID)
	assert.Equal(t, ExperimentStatusDraft, exp.Status)
	assert.False(t, exp.CreatedAt.IsZero())
	assert.False(t, exp.UpdatedAt.IsZero())
}

func TestInMemoryExperimentManager_Create_WithExplicitID(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	exp := defaultExperiment("my-id", "test-exp")
	err := mgr.Create(ctx, exp)
	require.NoError(t, err)
	assert.Equal(t, "my-id", exp.ID)
}

func TestInMemoryExperimentManager_Create_EmptyName(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	exp := &Experiment{Variants: defaultVariants()}
	err := mgr.Create(ctx, exp)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "experiment name is required")
}

func TestInMemoryExperimentManager_Create_InsufficientVariants_Zero(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	exp := &Experiment{Name: "test", Variants: []*Variant{}}
	err := mgr.Create(ctx, exp)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "at least 2 variants required")
}

func TestInMemoryExperimentManager_Create_InsufficientVariants_One(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	exp := &Experiment{Name: "test", Variants: []*Variant{{ID: "v1", Name: "only"}}}
	err := mgr.Create(ctx, exp)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "at least 2 variants required")
}

func TestInMemoryExperimentManager_Create_AutoGeneratesVariantIDs(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	exp := &Experiment{
		Name: "test",
		Variants: []*Variant{
			{Name: "v1"},
			{Name: "v2"},
		},
	}
	err := mgr.Create(ctx, exp)
	require.NoError(t, err)
	assert.NotEmpty(t, exp.Variants[0].ID)
	assert.NotEmpty(t, exp.Variants[1].ID)
}

func TestInMemoryExperimentManager_Create_DefaultTrafficSplit(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	exp := defaultExperiment("exp1", "test")
	// No explicit traffic split
	exp.TrafficSplit = nil
	err := mgr.Create(ctx, exp)
	require.NoError(t, err)

	// Should default to equal split
	assert.InDelta(t, 0.5, exp.TrafficSplit["control"], 0.001)
	assert.InDelta(t, 0.5, exp.TrafficSplit["treatment"], 0.001)
}

func TestInMemoryExperimentManager_Create_DefaultTrafficSplit_ThreeVariants(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	exp := &Experiment{
		Name: "test",
		Variants: []*Variant{
			{ID: "v1", Name: "A"},
			{ID: "v2", Name: "B"},
			{ID: "v3", Name: "C"},
		},
	}
	err := mgr.Create(ctx, exp)
	require.NoError(t, err)
	for _, v := range exp.Variants {
		assert.InDelta(t, 1.0/3.0, exp.TrafficSplit[v.ID], 0.001)
	}
}

func TestInMemoryExperimentManager_Create_CustomTrafficSplit(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	exp := &Experiment{
		Name: "test",
		Variants: []*Variant{
			{ID: "v1", Name: "Control"},
			{ID: "v2", Name: "Treatment"},
		},
		TrafficSplit: map[string]float64{"v1": 0.7, "v2": 0.3},
	}
	err := mgr.Create(ctx, exp)
	require.NoError(t, err)
}

func TestInMemoryExperimentManager_Create_TrafficSplitNotSumToOne(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	exp := &Experiment{
		Name: "test",
		Variants: []*Variant{
			{ID: "v1", Name: "A"},
			{ID: "v2", Name: "B"},
		},
		TrafficSplit: map[string]float64{"v1": 0.5, "v2": 0.3},
	}
	err := mgr.Create(ctx, exp)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "traffic split must sum to 1.0")
}

func TestInMemoryExperimentManager_Create_TrafficSplitNegative(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	exp := &Experiment{
		Name: "test",
		Variants: []*Variant{
			{ID: "v1", Name: "A"},
			{ID: "v2", Name: "B"},
		},
		TrafficSplit: map[string]float64{"v1": 1.5, "v2": -0.5},
	}
	err := mgr.Create(ctx, exp)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "traffic split cannot be negative")
}

func TestInMemoryExperimentManager_Create_TrafficSplitMissingVariant(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	exp := &Experiment{
		Name: "test",
		Variants: []*Variant{
			{ID: "v1", Name: "A"},
			{ID: "v2", Name: "B"},
		},
		TrafficSplit: map[string]float64{"v1": 1.0}, // missing v2
	}
	err := mgr.Create(ctx, exp)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "missing traffic split for variant")
}

func TestInMemoryExperimentManager_Create_InitializesMetricMaps(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	exp := defaultExperiment("exp1", "test")
	require.NoError(t, mgr.Create(ctx, exp))

	mgr.mu.RLock()
	defer mgr.mu.RUnlock()
	assert.Contains(t, mgr.metrics, "exp1")
	assert.Contains(t, mgr.assignments, "exp1")
	for _, v := range exp.Variants {
		assert.Contains(t, mgr.metrics["exp1"], v.ID)
	}
}

// --- Get tests ---

func TestInMemoryExperimentManager_Get_Success(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	exp := defaultExperiment("exp1", "test")
	require.NoError(t, mgr.Create(ctx, exp))

	got, err := mgr.Get(ctx, "exp1")
	require.NoError(t, err)
	assert.Equal(t, "exp1", got.ID)
	assert.Equal(t, "test", got.Name)
}

func TestInMemoryExperimentManager_Get_NotFound(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	_, err := mgr.Get(ctx, "nonexistent")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "experiment not found")
}

// --- List tests ---

func TestInMemoryExperimentManager_List_All(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	require.NoError(t, mgr.Create(ctx, defaultExperiment("exp1", "test1")))
	time.Sleep(time.Millisecond)
	require.NoError(t, mgr.Create(ctx, defaultExperiment("exp2", "test2")))

	exps, err := mgr.List(ctx, "")
	require.NoError(t, err)
	assert.Len(t, exps, 2)
	// Newest first
	assert.Equal(t, "exp2", exps[0].ID)
}

func TestInMemoryExperimentManager_List_ByStatus(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	require.NoError(t, mgr.Create(ctx, defaultExperiment("exp1", "test1")))
	require.NoError(t, mgr.Create(ctx, defaultExperiment("exp2", "test2")))
	require.NoError(t, mgr.Start(ctx, "exp2"))

	drafts, err := mgr.List(ctx, ExperimentStatusDraft)
	require.NoError(t, err)
	assert.Len(t, drafts, 1)

	running, err := mgr.List(ctx, ExperimentStatusRunning)
	require.NoError(t, err)
	assert.Len(t, running, 1)
}

func TestInMemoryExperimentManager_List_Empty(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	exps, err := mgr.List(ctx, "")
	require.NoError(t, err)
	assert.Empty(t, exps)
}

// --- Start tests ---

func TestInMemoryExperimentManager_Start_FromDraft(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	exp := defaultExperiment("exp1", "test")
	require.NoError(t, mgr.Create(ctx, exp))

	err := mgr.Start(ctx, "exp1")
	require.NoError(t, err)

	got, _ := mgr.Get(ctx, "exp1")
	assert.Equal(t, ExperimentStatusRunning, got.Status)
	assert.NotNil(t, got.StartTime)
}

func TestInMemoryExperimentManager_Start_FromPaused(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	exp := defaultExperiment("exp1", "test")
	require.NoError(t, mgr.Create(ctx, exp))
	require.NoError(t, mgr.Start(ctx, "exp1"))
	require.NoError(t, mgr.Pause(ctx, "exp1"))

	err := mgr.Start(ctx, "exp1")
	require.NoError(t, err)

	got, _ := mgr.Get(ctx, "exp1")
	assert.Equal(t, ExperimentStatusRunning, got.Status)
}

func TestInMemoryExperimentManager_Start_DoesNotResetStartTime(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	exp := defaultExperiment("exp1", "test")
	require.NoError(t, mgr.Create(ctx, exp))
	require.NoError(t, mgr.Start(ctx, "exp1"))

	got1, _ := mgr.Get(ctx, "exp1")
	firstStart := *got1.StartTime

	require.NoError(t, mgr.Pause(ctx, "exp1"))
	require.NoError(t, mgr.Start(ctx, "exp1"))

	got2, _ := mgr.Get(ctx, "exp1")
	assert.Equal(t, firstStart, *got2.StartTime, "StartTime should not change on restart")
}

func TestInMemoryExperimentManager_Start_NotFound(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	err := mgr.Start(ctx, "nonexistent")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "experiment not found")
}

func TestInMemoryExperimentManager_Start_InvalidStatus(t *testing.T) {
	tests := []struct {
		name   string
		status ExperimentStatus
		setup  func(mgr *InMemoryExperimentManager, ctx context.Context)
	}{
		{
			name:   "already running",
			status: ExperimentStatusRunning,
			setup: func(mgr *InMemoryExperimentManager, ctx context.Context) {
				require.NoError(t, mgr.Create(ctx, defaultExperiment("exp1", "test")))
				require.NoError(t, mgr.Start(ctx, "exp1"))
			},
		},
		{
			name:   "completed",
			status: ExperimentStatusCompleted,
			setup: func(mgr *InMemoryExperimentManager, ctx context.Context) {
				require.NoError(t, mgr.Create(ctx, defaultExperiment("exp1", "test")))
				require.NoError(t, mgr.Start(ctx, "exp1"))
				require.NoError(t, mgr.Complete(ctx, "exp1", ""))
			},
		},
		{
			name:   "cancelled",
			status: ExperimentStatusCancelled,
			setup: func(mgr *InMemoryExperimentManager, ctx context.Context) {
				require.NoError(t, mgr.Create(ctx, defaultExperiment("exp1", "test")))
				require.NoError(t, mgr.Cancel(ctx, "exp1"))
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mgr := newTestExperimentManager()
			ctx := context.Background()
			tt.setup(mgr, ctx)
			err := mgr.Start(ctx, "exp1")
			require.Error(t, err)
			assert.Contains(t, err.Error(), "cannot start experiment")
		})
	}
}

// --- Pause tests ---

func TestInMemoryExperimentManager_Pause_Success(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	require.NoError(t, mgr.Create(ctx, defaultExperiment("exp1", "test")))
	require.NoError(t, mgr.Start(ctx, "exp1"))

	err := mgr.Pause(ctx, "exp1")
	require.NoError(t, err)

	got, _ := mgr.Get(ctx, "exp1")
	assert.Equal(t, ExperimentStatusPaused, got.Status)
}

func TestInMemoryExperimentManager_Pause_NotFound(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	err := mgr.Pause(ctx, "nonexistent")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "experiment not found")
}

func TestInMemoryExperimentManager_Pause_InvalidStatus(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	require.NoError(t, mgr.Create(ctx, defaultExperiment("exp1", "test")))
	// Still in draft
	err := mgr.Pause(ctx, "exp1")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "cannot pause experiment")
}

// --- Complete tests ---

func TestInMemoryExperimentManager_Complete_WithWinner(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	require.NoError(t, mgr.Create(ctx, defaultExperiment("exp1", "test")))
	require.NoError(t, mgr.Start(ctx, "exp1"))

	err := mgr.Complete(ctx, "exp1", "control")
	require.NoError(t, err)

	got, _ := mgr.Get(ctx, "exp1")
	assert.Equal(t, ExperimentStatusCompleted, got.Status)
	assert.Equal(t, "control", got.Winner)
	assert.NotNil(t, got.EndTime)
}

func TestInMemoryExperimentManager_Complete_WithoutWinner(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	require.NoError(t, mgr.Create(ctx, defaultExperiment("exp1", "test")))

	err := mgr.Complete(ctx, "exp1", "")
	require.NoError(t, err)

	got, _ := mgr.Get(ctx, "exp1")
	assert.Equal(t, ExperimentStatusCompleted, got.Status)
	assert.Empty(t, got.Winner)
}

func TestInMemoryExperimentManager_Complete_InvalidWinner(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	require.NoError(t, mgr.Create(ctx, defaultExperiment("exp1", "test")))

	err := mgr.Complete(ctx, "exp1", "nonexistent-variant")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "invalid winner variant")
}

func TestInMemoryExperimentManager_Complete_NotFound(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	err := mgr.Complete(ctx, "nonexistent", "")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "experiment not found")
}

// --- Cancel tests ---

func TestInMemoryExperimentManager_Cancel_FromDraft(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	require.NoError(t, mgr.Create(ctx, defaultExperiment("exp1", "test")))

	err := mgr.Cancel(ctx, "exp1")
	require.NoError(t, err)

	got, _ := mgr.Get(ctx, "exp1")
	assert.Equal(t, ExperimentStatusCancelled, got.Status)
	assert.NotNil(t, got.EndTime)
}

func TestInMemoryExperimentManager_Cancel_FromRunning(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	require.NoError(t, mgr.Create(ctx, defaultExperiment("exp1", "test")))
	require.NoError(t, mgr.Start(ctx, "exp1"))

	err := mgr.Cancel(ctx, "exp1")
	require.NoError(t, err)

	got, _ := mgr.Get(ctx, "exp1")
	assert.Equal(t, ExperimentStatusCancelled, got.Status)
}

func TestInMemoryExperimentManager_Cancel_FromPaused(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	require.NoError(t, mgr.Create(ctx, defaultExperiment("exp1", "test")))
	require.NoError(t, mgr.Start(ctx, "exp1"))
	require.NoError(t, mgr.Pause(ctx, "exp1"))

	err := mgr.Cancel(ctx, "exp1")
	require.NoError(t, err)

	got, _ := mgr.Get(ctx, "exp1")
	assert.Equal(t, ExperimentStatusCancelled, got.Status)
}

func TestInMemoryExperimentManager_Cancel_AlreadyCompleted(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	require.NoError(t, mgr.Create(ctx, defaultExperiment("exp1", "test")))
	require.NoError(t, mgr.Complete(ctx, "exp1", ""))

	err := mgr.Cancel(ctx, "exp1")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "experiment already finalized")
}

func TestInMemoryExperimentManager_Cancel_AlreadyCancelled(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	require.NoError(t, mgr.Create(ctx, defaultExperiment("exp1", "test")))
	require.NoError(t, mgr.Cancel(ctx, "exp1"))

	err := mgr.Cancel(ctx, "exp1")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "experiment already finalized")
}

func TestInMemoryExperimentManager_Cancel_NotFound(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	err := mgr.Cancel(ctx, "nonexistent")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "experiment not found")
}

// --- AssignVariant tests ---

func TestInMemoryExperimentManager_AssignVariant_Success(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	exp := defaultExperiment("exp1", "test")
	require.NoError(t, mgr.Create(ctx, exp))
	require.NoError(t, mgr.Start(ctx, "exp1"))

	variant, err := mgr.AssignVariant(ctx, "exp1", "user-1")
	require.NoError(t, err)
	require.NotNil(t, variant)
	assert.Contains(t, []string{"control", "treatment"}, variant.ID)
}

func TestInMemoryExperimentManager_AssignVariant_Deterministic(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	exp := defaultExperiment("exp1", "test")
	require.NoError(t, mgr.Create(ctx, exp))
	require.NoError(t, mgr.Start(ctx, "exp1"))

	// Assign the same user multiple times — should get same variant
	v1, err := mgr.AssignVariant(ctx, "exp1", "user-1")
	require.NoError(t, err)

	v2, err := mgr.AssignVariant(ctx, "exp1", "user-1")
	require.NoError(t, err)

	assert.Equal(t, v1.ID, v2.ID, "same user must get same variant")
}

func TestInMemoryExperimentManager_AssignVariant_DifferentUsers(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	exp := defaultExperiment("exp1", "test")
	exp.TrafficSplit = map[string]float64{"control": 0.5, "treatment": 0.5}
	require.NoError(t, mgr.Create(ctx, exp))
	require.NoError(t, mgr.Start(ctx, "exp1"))

	assignments := make(map[string]int)
	n := 100
	for i := 0; i < n; i++ {
		v, err := mgr.AssignVariant(ctx, "exp1", fmt.Sprintf("user-%d", i))
		require.NoError(t, err)
		assignments[v.ID]++
	}

	// With 50/50 split and FNV hash, we expect roughly equal distribution
	// Allow +-30% tolerance for hash distribution
	assert.Greater(t, assignments["control"], 20)
	assert.Greater(t, assignments["treatment"], 20)
}

func TestInMemoryExperimentManager_AssignVariant_NotRunning(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	require.NoError(t, mgr.Create(ctx, defaultExperiment("exp1", "test")))
	// Still in draft

	_, err := mgr.AssignVariant(ctx, "exp1", "user-1")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "experiment not running")
}

func TestInMemoryExperimentManager_AssignVariant_NotFound(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	_, err := mgr.AssignVariant(ctx, "nonexistent", "user-1")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "experiment not found")
}

func TestInMemoryExperimentManager_AssignVariant_FNVHashDeterminism(t *testing.T) {
	// Directly verify the hash-based selection logic
	mgr := newTestExperimentManager()

	exp := defaultExperiment("exp1", "test")
	exp.TrafficSplit = map[string]float64{"control": 0.5, "treatment": 0.5}

	// Compute expected variant from hash
	h := fnv.New32a()
	h.Write([]byte("exp1" + "stable-user"))
	hashValue := float64(h.Sum32()) / float64(^uint32(0))

	selected := mgr.selectVariant(exp, "stable-user")
	require.NotNil(t, selected)

	// Verify selection matches cumulative split logic
	if hashValue < 0.5 {
		assert.Equal(t, exp.Variants[0].ID, selected.ID)
	} else {
		assert.Equal(t, exp.Variants[1].ID, selected.ID)
	}
}

// --- RecordMetric tests ---

func TestInMemoryExperimentManager_RecordMetric_Success(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	require.NoError(t, mgr.Create(ctx, defaultExperiment("exp1", "test")))

	err := mgr.RecordMetric(ctx, "exp1", "control", "latency", 150.0)
	require.NoError(t, err)

	mgr.mu.RLock()
	samples := mgr.metrics["exp1"]["control"]
	mgr.mu.RUnlock()
	assert.Len(t, samples, 1)
	assert.InDelta(t, 150.0, samples[0].Value, 0.001)
}

func TestInMemoryExperimentManager_RecordMetric_MultipleSamples(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	require.NoError(t, mgr.Create(ctx, defaultExperiment("exp1", "test")))

	for i := 0; i < 10; i++ {
		require.NoError(t, mgr.RecordMetric(ctx, "exp1", "control", "latency", float64(100+i)))
	}

	mgr.mu.RLock()
	samples := mgr.metrics["exp1"]["control"]
	mgr.mu.RUnlock()
	assert.Len(t, samples, 10)
}

func TestInMemoryExperimentManager_RecordMetric_ExperimentNotFound(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	err := mgr.RecordMetric(ctx, "nonexistent", "v1", "latency", 100)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "experiment not found")
}

func TestInMemoryExperimentManager_RecordMetric_VariantNotFound(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	require.NoError(t, mgr.Create(ctx, defaultExperiment("exp1", "test")))

	err := mgr.RecordMetric(ctx, "exp1", "nonexistent-variant", "latency", 100)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "variant not found")
}

// --- GetResults tests ---

func TestInMemoryExperimentManager_GetResults_EmptyMetrics(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	require.NoError(t, mgr.Create(ctx, defaultExperiment("exp1", "test")))

	result, err := mgr.GetResults(ctx, "exp1")
	require.NoError(t, err)
	assert.Equal(t, "exp1", result.ExperimentID)
	assert.Equal(t, 0, result.TotalSamples)
}

func TestInMemoryExperimentManager_GetResults_NotFound(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	_, err := mgr.GetResults(ctx, "nonexistent")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "experiment not found")
}

func TestInMemoryExperimentManager_GetResults_WithMetrics(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	require.NoError(t, mgr.Create(ctx, defaultExperiment("exp1", "test")))
	require.NoError(t, mgr.Start(ctx, "exp1"))

	// Record several metric samples for each variant
	for i := 0; i < 50; i++ {
		require.NoError(t, mgr.RecordMetric(ctx, "exp1", "control", "primary", 0.7+float64(i)*0.001))
		require.NoError(t, mgr.RecordMetric(ctx, "exp1", "treatment", "primary", 0.8+float64(i)*0.001))
	}

	result, err := mgr.GetResults(ctx, "exp1")
	require.NoError(t, err)
	assert.Equal(t, 100, result.TotalSamples) // 50 per variant
	assert.Len(t, result.VariantResults, 2)

	// Control metrics
	controlResult := result.VariantResults["control"]
	require.NotNil(t, controlResult)
	assert.Equal(t, 50, controlResult.SampleCount)
	assert.NotNil(t, controlResult.MetricValues["primary"])

	// Treatment should have higher mean
	treatmentResult := result.VariantResults["treatment"]
	require.NotNil(t, treatmentResult)
	assert.Greater(t, treatmentResult.MetricValues["primary"].Value, controlResult.MetricValues["primary"].Value)
}

func TestInMemoryExperimentManager_GetResults_StartTimeEndTime(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	require.NoError(t, mgr.Create(ctx, defaultExperiment("exp1", "test")))
	require.NoError(t, mgr.Start(ctx, "exp1"))

	result, err := mgr.GetResults(ctx, "exp1")
	require.NoError(t, err)
	assert.False(t, result.StartTime.IsZero())
	// EndTime should be approximately now since experiment isn't completed
	assert.WithinDuration(t, time.Now(), result.EndTime, 2*time.Second)
}

func TestInMemoryExperimentManager_GetResults_CompletedExperiment(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	require.NoError(t, mgr.Create(ctx, defaultExperiment("exp1", "test")))
	require.NoError(t, mgr.Start(ctx, "exp1"))
	require.NoError(t, mgr.Complete(ctx, "exp1", "control"))

	result, err := mgr.GetResults(ctx, "exp1")
	require.NoError(t, err)
	assert.False(t, result.EndTime.IsZero())
}

func TestInMemoryExperimentManager_GetResults_InsufficientConfidence(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	require.NoError(t, mgr.Create(ctx, defaultExperiment("exp1", "test")))
	require.NoError(t, mgr.Start(ctx, "exp1"))

	// Only a few samples — not enough for significance
	for i := 0; i < 5; i++ {
		require.NoError(t, mgr.RecordMetric(ctx, "exp1", "control", "primary", 0.7))
		require.NoError(t, mgr.RecordMetric(ctx, "exp1", "treatment", "primary", 0.8))
	}

	result, err := mgr.GetResults(ctx, "exp1")
	require.NoError(t, err)
	assert.Contains(t, result.Recommendation, "Continue experiment")
	assert.Empty(t, result.Winner)
}

// --- calculateVariantResult tests ---

func TestInMemoryExperimentManager_CalculateVariantResult_EmptySamples(t *testing.T) {
	mgr := newTestExperimentManager()
	vr := mgr.calculateVariantResult("v1", []*metricSample{})
	assert.Equal(t, "v1", vr.VariantID)
	assert.Equal(t, 0, vr.SampleCount)
	assert.Empty(t, vr.MetricValues)
}

func TestInMemoryExperimentManager_CalculateVariantResult_SingleSample(t *testing.T) {
	mgr := newTestExperimentManager()
	samples := []*metricSample{{Value: 5.0, Timestamp: time.Now()}}
	vr := mgr.calculateVariantResult("v1", samples)

	assert.Equal(t, 1, vr.SampleCount)
	require.NotNil(t, vr.MetricValues["primary"])
	assert.InDelta(t, 5.0, vr.MetricValues["primary"].Value, 0.001)
	assert.InDelta(t, 5.0, vr.MetricValues["primary"].Min, 0.001)
	assert.InDelta(t, 5.0, vr.MetricValues["primary"].Max, 0.001)
	assert.InDelta(t, 0.0, vr.MetricValues["primary"].StdDev, 0.001)
}

func TestInMemoryExperimentManager_CalculateVariantResult_MultipleSamples(t *testing.T) {
	mgr := newTestExperimentManager()
	samples := []*metricSample{
		{Value: 10.0, Timestamp: time.Now()},
		{Value: 20.0, Timestamp: time.Now()},
		{Value: 30.0, Timestamp: time.Now()},
	}
	vr := mgr.calculateVariantResult("v1", samples)

	assert.Equal(t, 3, vr.SampleCount)
	mv := vr.MetricValues["primary"]
	require.NotNil(t, mv)
	assert.InDelta(t, 20.0, mv.Value, 0.001) // mean
	assert.InDelta(t, 10.0, mv.Min, 0.001)
	assert.InDelta(t, 30.0, mv.Max, 0.001)

	// StdDev for [10, 20, 30]: mean=20, variance = (100+0+100)/3 = 66.67, stddev ~= 8.16
	expectedVariance := (100.0 + 0.0 + 100.0) / 3.0
	expectedStdDev := math.Sqrt(expectedVariance)
	assert.InDelta(t, expectedStdDev, mv.StdDev, 0.01)
}

// --- calculateSignificance tests ---

func TestInMemoryExperimentManager_CalculateSignificance_InsufficientVariants(t *testing.T) {
	mgr := newTestExperimentManager()
	result := &ExperimentResult{
		VariantResults: map[string]*VariantResult{
			"v1": {SampleCount: 50, MetricValues: map[string]*MetricValue{"primary": {Value: 1.0, StdDev: 0.1, Count: 50}}},
		},
	}
	sig, conf := mgr.calculateSignificance(result)
	assert.InDelta(t, 0, sig, 0.001)
	assert.InDelta(t, 0, conf, 0.001)
}

func TestInMemoryExperimentManager_CalculateSignificance_InsufficientSamples(t *testing.T) {
	mgr := newTestExperimentManager()
	// SampleCount < 30 triggers "insufficient sample size" branch
	result := &ExperimentResult{
		VariantResults: map[string]*VariantResult{
			"v1": {SampleCount: 10, MetricValues: map[string]*MetricValue{"primary": {Value: 1.0, StdDev: 0.1, Count: 10}}},
			"v2": {SampleCount: 10, MetricValues: map[string]*MetricValue{"primary": {Value: 2.0, StdDev: 0.1, Count: 10}}},
		},
	}
	sig, conf := mgr.calculateSignificance(result)
	assert.InDelta(t, 0, sig, 0.001)
	assert.InDelta(t, 0.5, conf, 0.001)
}

func TestInMemoryExperimentManager_CalculateSignificance_ZeroPooledSE(t *testing.T) {
	mgr := newTestExperimentManager()
	// calculateSignificance uses vr.SampleCount (not MetricValue.Count) for n1/n2
	result := &ExperimentResult{
		VariantResults: map[string]*VariantResult{
			"v1": {SampleCount: 50, MetricValues: map[string]*MetricValue{"primary": {Value: 1.0, StdDev: 0.0, Count: 50}}},
			"v2": {SampleCount: 50, MetricValues: map[string]*MetricValue{"primary": {Value: 1.0, StdDev: 0.0, Count: 50}}},
		},
	}
	sig, conf := mgr.calculateSignificance(result)
	assert.InDelta(t, 0, sig, 0.001)
	assert.InDelta(t, 0, conf, 0.001)
}

func TestInMemoryExperimentManager_CalculateSignificance_HighConfidence(t *testing.T) {
	mgr := newTestExperimentManager()
	// Large difference, small stddev, many samples => high z-score
	// calculateSignificance uses vr.SampleCount for n1/n2
	result := &ExperimentResult{
		VariantResults: map[string]*VariantResult{
			"v1": {SampleCount: 100, MetricValues: map[string]*MetricValue{"primary": {Value: 1.0, StdDev: 0.1, Count: 100}}},
			"v2": {SampleCount: 100, MetricValues: map[string]*MetricValue{"primary": {Value: 2.0, StdDev: 0.1, Count: 100}}},
		},
	}
	_, conf := mgr.calculateSignificance(result)
	assert.GreaterOrEqual(t, conf, 0.95)
}

func TestInMemoryExperimentManager_CalculateSignificance_LowConfidence(t *testing.T) {
	mgr := newTestExperimentManager()
	// Small difference, large stddev => low z-score
	result := &ExperimentResult{
		VariantResults: map[string]*VariantResult{
			"v1": {SampleCount: 50, MetricValues: map[string]*MetricValue{"primary": {Value: 1.0, StdDev: 5.0, Count: 50}}},
			"v2": {SampleCount: 50, MetricValues: map[string]*MetricValue{"primary": {Value: 1.01, StdDev: 5.0, Count: 50}}},
		},
	}
	_, conf := mgr.calculateSignificance(result)
	assert.Less(t, conf, 0.90)
}

func TestInMemoryExperimentManager_CalculateSignificance_NoPrimaryMetric(t *testing.T) {
	mgr := newTestExperimentManager()
	result := &ExperimentResult{
		VariantResults: map[string]*VariantResult{
			"v1": {SampleCount: 50, MetricValues: map[string]*MetricValue{"other": {Value: 1.0, StdDev: 0.1, Count: 50}}},
			"v2": {SampleCount: 50, MetricValues: map[string]*MetricValue{"other": {Value: 2.0, StdDev: 0.1, Count: 50}}},
		},
	}
	sig, conf := mgr.calculateSignificance(result)
	assert.InDelta(t, 0, sig, 0.001)
	assert.InDelta(t, 0, conf, 0.001)
}

// --- determineWinner tests ---

func TestInMemoryExperimentManager_DetermineWinner_SelectsHighest(t *testing.T) {
	mgr := newTestExperimentManager()
	result := &ExperimentResult{
		VariantResults: map[string]*VariantResult{
			"v1": {VariantID: "v1", MetricValues: map[string]*MetricValue{"primary": {Value: 0.8}}},
			"v2": {VariantID: "v2", MetricValues: map[string]*MetricValue{"primary": {Value: 0.9}}},
		},
	}
	winner := mgr.determineWinner(result)
	assert.Equal(t, "v2", winner)
}

func TestInMemoryExperimentManager_DetermineWinner_NoPrimaryMetric(t *testing.T) {
	mgr := newTestExperimentManager()
	result := &ExperimentResult{
		VariantResults: map[string]*VariantResult{
			"v1": {VariantID: "v1", SampleCount: 10, MetricValues: map[string]*MetricValue{}},
		},
	}
	winner := mgr.determineWinner(result)
	assert.Empty(t, winner)
}

// --- Full lifecycle test ---

func TestInMemoryExperimentManager_FullLifecycle(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	// 1. Create
	exp := defaultExperiment("exp1", "A/B Test GPT-4 vs Turbo")
	require.NoError(t, mgr.Create(ctx, exp))
	assert.Equal(t, ExperimentStatusDraft, exp.Status)

	// 2. Start
	require.NoError(t, mgr.Start(ctx, "exp1"))
	got, _ := mgr.Get(ctx, "exp1")
	assert.Equal(t, ExperimentStatusRunning, got.Status)

	// 3. Assign variants
	for i := 0; i < 50; i++ {
		v, err := mgr.AssignVariant(ctx, "exp1", fmt.Sprintf("user-%d", i))
		require.NoError(t, err)
		// 4. Record metrics
		require.NoError(t, mgr.RecordMetric(ctx, "exp1", v.ID, "primary", 0.75+float64(i%10)*0.02))
	}

	// 5. Get results
	result, err := mgr.GetResults(ctx, "exp1")
	require.NoError(t, err)
	assert.Greater(t, result.TotalSamples, 0)

	// 6. Complete
	require.NoError(t, mgr.Complete(ctx, "exp1", "control"))
	got, _ = mgr.Get(ctx, "exp1")
	assert.Equal(t, ExperimentStatusCompleted, got.Status)
	assert.Equal(t, "control", got.Winner)
}

// --- GetResults with control improvement ---

func TestInMemoryExperimentManager_GetResults_ImprovementVsControl(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	require.NoError(t, mgr.Create(ctx, defaultExperiment("exp1", "test")))
	require.NoError(t, mgr.Start(ctx, "exp1"))

	// Record enough samples for both variants with slight variance
	for i := 0; i < 50; i++ {
		require.NoError(t, mgr.RecordMetric(ctx, "exp1", "control", "primary", 0.5+float64(i%5)*0.001))
		require.NoError(t, mgr.RecordMetric(ctx, "exp1", "treatment", "primary", 0.7+float64(i%5)*0.001))
	}

	result, err := mgr.GetResults(ctx, "exp1")
	require.NoError(t, err)

	treatmentResult := result.VariantResults["treatment"]
	require.NotNil(t, treatmentResult)
	// Improvement should be approximately (0.7 - 0.5) / 0.5 * 100 = 40%
	assert.InDelta(t, 40.0, treatmentResult.Improvement, 2.0)
}

// --- GetResults with statistical significance and winner determination ---

func TestInMemoryExperimentManager_GetResults_HighSignificanceWinner(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	require.NoError(t, mgr.Create(ctx, defaultExperiment("exp1", "test")))
	require.NoError(t, mgr.Start(ctx, "exp1"))

	// Very distinct distributions with slight variance for non-zero stddev
	for i := 0; i < 100; i++ {
		require.NoError(t, mgr.RecordMetric(ctx, "exp1", "control", "primary", 0.5+float64(i%3)*0.01))
		require.NoError(t, mgr.RecordMetric(ctx, "exp1", "treatment", "primary", 0.9+float64(i%3)*0.01))
	}

	result, err := mgr.GetResults(ctx, "exp1")
	require.NoError(t, err)
	assert.GreaterOrEqual(t, result.Confidence, 0.95)
	assert.NotEmpty(t, result.Winner)
	assert.Contains(t, result.Recommendation, "Deploy variant")
}

// --- Concurrent access tests ---

func TestInMemoryExperimentManager_ConcurrentOperations(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	require.NoError(t, mgr.Create(ctx, defaultExperiment("exp1", "test")))
	require.NoError(t, mgr.Start(ctx, "exp1"))

	var wg sync.WaitGroup

	// Concurrent assigns
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			_, _ = mgr.AssignVariant(ctx, "exp1", fmt.Sprintf("user-%d", idx))
		}(i)
	}

	// Concurrent metric recordings
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			_ = mgr.RecordMetric(ctx, "exp1", "control", "latency", float64(idx))
		}(i)
	}

	// Concurrent reads
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _ = mgr.Get(ctx, "exp1")
			_, _ = mgr.List(ctx, "")
			_, _ = mgr.GetResults(ctx, "exp1")
		}()
	}

	wg.Wait()
	// No panics or data races
}

func TestInMemoryExperimentManager_ConcurrentCreates(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	var wg sync.WaitGroup
	n := 20
	errs := make([]error, n)

	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			exp := defaultExperiment(fmt.Sprintf("exp-%d", idx), fmt.Sprintf("test-%d", idx))
			errs[idx] = mgr.Create(ctx, exp)
		}(i)
	}
	wg.Wait()

	for i, err := range errs {
		assert.NoError(t, err, "experiment %d", i)
	}

	all, _ := mgr.List(ctx, "")
	assert.Len(t, all, n)
}

// --- Z-test confidence level boundary tests ---

func TestInMemoryExperimentManager_CalculateSignificance_ZScoreBoundaries(t *testing.T) {
	mgr := newTestExperimentManager()

	tests := []struct {
		name           string
		mean1, mean2   float64
		std1, std2     float64
		n1, n2         int
		minConfidence  float64
		maxConfidence  float64
	}{
		{
			name:          "z >= 2.576 => 0.99 confidence",
			mean1:         1.0,
			mean2:         2.0,
			std1:          0.1,
			std2:          0.1,
			n1:            100,
			n2:            100,
			minConfidence: 0.99,
			maxConfidence: 1.0,
		},
		{
			name:          "z between 1.96 and 2.576 => 0.95",
			mean1:         1.0,
			mean2:         1.03,
			std1:          0.1,
			std2:          0.1,
			n1:            100,
			n2:            100,
			minConfidence: 0.90,
			maxConfidence: 0.99,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := &ExperimentResult{
				VariantResults: map[string]*VariantResult{
					"v1": {SampleCount: tt.n1, MetricValues: map[string]*MetricValue{"primary": {Value: tt.mean1, StdDev: tt.std1, Count: tt.n1}}},
					"v2": {SampleCount: tt.n2, MetricValues: map[string]*MetricValue{"primary": {Value: tt.mean2, StdDev: tt.std2, Count: tt.n2}}},
				},
			}
			_, conf := mgr.calculateSignificance(result)
			assert.GreaterOrEqual(t, conf, tt.minConfidence)
			assert.LessOrEqual(t, conf, tt.maxConfidence)
		})
	}
}

// --- Interface compliance ---

func TestInMemoryExperimentManager_ImplementsExperimentManager(t *testing.T) {
	var _ ExperimentManager = (*InMemoryExperimentManager)(nil)
}

// --- validateTrafficSplit edge cases ---

func TestInMemoryExperimentManager_ValidateTrafficSplit_EmptyMap(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	exp := &Experiment{
		Name: "test",
		Variants: []*Variant{
			{ID: "v1", Name: "A"},
			{ID: "v2", Name: "B"},
		},
		TrafficSplit: map[string]float64{}, // empty map
	}
	err := mgr.Create(ctx, exp)
	require.NoError(t, err)
	// Should default to equal split
	assert.InDelta(t, 0.5, exp.TrafficSplit["v1"], 0.001)
}

func TestInMemoryExperimentManager_ValidateTrafficSplit_SumSlightlyOff(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	// Within tolerance: math.Abs(total - 1.0) <= 0.01
	// Use 0.502 + 0.502 = 1.004 which is well within 0.01 tolerance
	exp := &Experiment{
		Name: "test",
		Variants: []*Variant{
			{ID: "v1", Name: "A"},
			{ID: "v2", Name: "B"},
		},
		TrafficSplit: map[string]float64{"v1": 0.502, "v2": 0.502},
	}
	err := mgr.Create(ctx, exp)
	require.NoError(t, err)
}

func TestInMemoryExperimentManager_ValidateTrafficSplit_SumFarOff(t *testing.T) {
	mgr := newTestExperimentManager()
	ctx := context.Background()

	exp := &Experiment{
		Name: "test",
		Variants: []*Variant{
			{ID: "v1", Name: "A"},
			{ID: "v2", Name: "B"},
		},
		TrafficSplit: map[string]float64{"v1": 0.4, "v2": 0.4}, // 0.8, not 1.0
	}
	err := mgr.Create(ctx, exp)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "traffic split must sum to 1.0")
}

// --- selectVariant fallback ---

func TestInMemoryExperimentManager_SelectVariant_FallbackToLast(t *testing.T) {
	mgr := newTestExperimentManager()

	// This tests the fallback path — if hash value falls beyond all cumulative splits,
	// which shouldn't normally happen, but the code covers it.
	exp := &Experiment{
		ID: "exp1",
		Variants: []*Variant{
			{ID: "v1", Name: "A"},
			{ID: "v2", Name: "B"},
		},
		TrafficSplit: map[string]float64{"v1": 0.5, "v2": 0.5},
	}

	// This simply verifies the function doesn't panic and returns a non-nil variant
	v := mgr.selectVariant(exp, "any-user")
	require.NotNil(t, v)
}
