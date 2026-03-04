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

// --- mock helpers specific to integration tests ---

type mockDebateLLMEvaluator struct {
	scores map[string]float64
	err    error
}

func (m *mockDebateLLMEvaluator) EvaluateWithDebate(_ context.Context, _, _, _ string, metrics []string) (map[string]float64, error) {
	if m.err != nil {
		return nil, m.err
	}
	result := make(map[string]float64, len(metrics))
	for _, metric := range metrics {
		if score, ok := m.scores[metric]; ok {
			result[metric] = score
		} else {
			result[metric] = 0.85
		}
	}
	return result, nil
}

// --- DefaultLLMOpsConfig tests ---

func TestDefaultLLMOpsConfig_ReturnsNonNil(t *testing.T) {
	cfg := DefaultLLMOpsConfig()
	require.NotNil(t, cfg)
}

func TestDefaultLLMOpsConfig_DefaultValues(t *testing.T) {
	cfg := DefaultLLMOpsConfig()
	assert.True(t, cfg.EnableAutoEvaluation)
	assert.Equal(t, 24*time.Hour, cfg.EvaluationInterval)
	assert.Equal(t, 100, cfg.MinSamplesForSignif)
	assert.True(t, cfg.EnableDebateEvaluation)
	require.NotNil(t, cfg.AlertThresholds)
	assert.InDelta(t, 0.85, cfg.AlertThresholds["pass_rate"], 0.001)
	assert.InDelta(t, 5000.0, cfg.AlertThresholds["latency_p99"], 0.001)
}

// --- NewVerifierIntegration tests ---

func TestNewVerifierIntegration_WithAllParams(t *testing.T) {
	getScore := func(name string) float64 { return 8.5 }
	isHealthy := func(name string) bool { return true }
	logger := logrus.New()

	vi := NewVerifierIntegration(getScore, isHealthy, logger)
	require.NotNil(t, vi)
	assert.NotNil(t, vi.getProviderScore)
	assert.NotNil(t, vi.isProviderHealthy)
	assert.Equal(t, logger, vi.logger)
}

func TestNewVerifierIntegration_NilFunctions(t *testing.T) {
	vi := NewVerifierIntegration(nil, nil, nil)
	require.NotNil(t, vi)
	assert.Nil(t, vi.getProviderScore)
	assert.Nil(t, vi.isProviderHealthy)
}

// --- SelectBestProvider tests ---

func TestVerifierIntegration_SelectBestProvider_NilScoreFunc_EmptyProviders(t *testing.T) {
	vi := NewVerifierIntegration(nil, nil, nil)
	provider, score := vi.SelectBestProvider([]string{})
	assert.Equal(t, "", provider)
	assert.InDelta(t, 0.0, score, 0.001)
}

func TestVerifierIntegration_SelectBestProvider_NilScoreFunc_WithProviders(t *testing.T) {
	vi := NewVerifierIntegration(nil, nil, nil)
	provider, score := vi.SelectBestProvider([]string{"openai", "claude"})
	assert.Equal(t, "openai", provider)
	assert.InDelta(t, 0.0, score, 0.001)
}

func TestVerifierIntegration_SelectBestProvider_SelectsHighestScore(t *testing.T) {
	scores := map[string]float64{
		"openai": 7.5,
		"claude": 9.0,
		"gemini": 8.0,
	}
	getScore := func(name string) float64 { return scores[name] }
	isHealthy := func(name string) bool { return true }

	vi := NewVerifierIntegration(getScore, isHealthy, nil)
	provider, score := vi.SelectBestProvider([]string{"openai", "claude", "gemini"})
	assert.Equal(t, "claude", provider)
	assert.InDelta(t, 9.0, score, 0.001)
}

func TestVerifierIntegration_SelectBestProvider_SkipsUnhealthy(t *testing.T) {
	scores := map[string]float64{
		"openai": 7.5,
		"claude": 9.0,
		"gemini": 8.0,
	}
	getScore := func(name string) float64 { return scores[name] }
	isHealthy := func(name string) bool { return name != "claude" }

	vi := NewVerifierIntegration(getScore, isHealthy, nil)
	provider, score := vi.SelectBestProvider([]string{"openai", "claude", "gemini"})
	assert.Equal(t, "gemini", provider)
	assert.InDelta(t, 8.0, score, 0.001)
}

func TestVerifierIntegration_SelectBestProvider_AllUnhealthy(t *testing.T) {
	getScore := func(name string) float64 { return 9.0 }
	isHealthy := func(name string) bool { return false }

	vi := NewVerifierIntegration(getScore, isHealthy, nil)
	provider, score := vi.SelectBestProvider([]string{"openai", "claude"})
	assert.Equal(t, "", provider)
	assert.InDelta(t, 0.0, score, 0.001)
}

func TestVerifierIntegration_SelectBestProvider_NilHealthFunc(t *testing.T) {
	scores := map[string]float64{"openai": 7.0, "claude": 9.0}
	getScore := func(name string) float64 { return scores[name] }

	vi := NewVerifierIntegration(getScore, nil, nil)
	provider, score := vi.SelectBestProvider([]string{"openai", "claude"})
	assert.Equal(t, "claude", provider)
	assert.InDelta(t, 9.0, score, 0.001)
}

func TestVerifierIntegration_SelectBestProvider_SingleProvider(t *testing.T) {
	getScore := func(name string) float64 { return 8.0 }
	isHealthy := func(name string) bool { return true }

	vi := NewVerifierIntegration(getScore, isHealthy, nil)
	provider, score := vi.SelectBestProvider([]string{"openai"})
	assert.Equal(t, "openai", provider)
	assert.InDelta(t, 8.0, score, 0.001)
}

// --- NewLLMOpsSystem tests ---

func TestNewLLMOpsSystem_WithNilConfig(t *testing.T) {
	sys := NewLLMOpsSystem(nil, nil)
	require.NotNil(t, sys)
	require.NotNil(t, sys.config)
	require.NotNil(t, sys.logger)
	// Should use defaults
	assert.True(t, sys.config.EnableAutoEvaluation)
}

func TestNewLLMOpsSystem_WithNilLogger(t *testing.T) {
	cfg := &LLMOpsConfig{EnableAutoEvaluation: false}
	sys := NewLLMOpsSystem(cfg, nil)
	require.NotNil(t, sys)
	require.NotNil(t, sys.logger)
	assert.False(t, sys.config.EnableAutoEvaluation)
}

func TestNewLLMOpsSystem_WithAllParams(t *testing.T) {
	cfg := DefaultLLMOpsConfig()
	logger := logrus.New()
	sys := NewLLMOpsSystem(cfg, logger)
	require.NotNil(t, sys)
	assert.Equal(t, cfg, sys.config)
	assert.Equal(t, logger, sys.logger)
}

// --- Initialize tests ---

func TestLLMOpsSystem_Initialize_Success(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	sys := NewLLMOpsSystem(nil, logger)
	err := sys.Initialize()
	require.NoError(t, err)

	assert.NotNil(t, sys.GetPromptRegistry())
	assert.NotNil(t, sys.GetExperimentManager())
	assert.NotNil(t, sys.GetEvaluator())
	assert.NotNil(t, sys.GetAlertManager())
}

func TestLLMOpsSystem_Initialize_WithoutDebateEvaluator(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	cfg := DefaultLLMOpsConfig()
	cfg.EnableDebateEvaluation = true
	sys := NewLLMOpsSystem(cfg, logger)
	// No debate evaluator set
	err := sys.Initialize()
	require.NoError(t, err)
	assert.NotNil(t, sys.GetEvaluator())
}

func TestLLMOpsSystem_Initialize_WithDebateEvaluator(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	cfg := DefaultLLMOpsConfig()
	cfg.EnableDebateEvaluation = true

	sys := NewLLMOpsSystem(cfg, logger)
	sys.SetDebateEvaluator(&mockDebateLLMEvaluator{scores: map[string]float64{"quality": 0.9}})

	err := sys.Initialize()
	require.NoError(t, err)
	assert.NotNil(t, sys.GetEvaluator())
}

func TestLLMOpsSystem_Initialize_DebateDisabled(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	cfg := DefaultLLMOpsConfig()
	cfg.EnableDebateEvaluation = false

	sys := NewLLMOpsSystem(cfg, logger)
	sys.SetDebateEvaluator(&mockDebateLLMEvaluator{scores: map[string]float64{"quality": 0.9}})

	err := sys.Initialize()
	require.NoError(t, err)
	// Even with evaluator set, it should not be used when debate is disabled
}

// --- SetDebateEvaluator tests ---

func TestLLMOpsSystem_SetDebateEvaluator_Success(t *testing.T) {
	sys := NewLLMOpsSystem(nil, nil)
	eval := &mockDebateLLMEvaluator{}
	sys.SetDebateEvaluator(eval)
	assert.Equal(t, eval, sys.debateEvaluator)
}

func TestLLMOpsSystem_SetDebateEvaluator_Nil(t *testing.T) {
	sys := NewLLMOpsSystem(nil, nil)
	sys.SetDebateEvaluator(nil)
	assert.Nil(t, sys.debateEvaluator)
}

// --- SetVerifierIntegration tests ---

func TestLLMOpsSystem_SetVerifierIntegration_Success(t *testing.T) {
	sys := NewLLMOpsSystem(nil, nil)
	vi := NewVerifierIntegration(nil, nil, nil)
	sys.SetVerifierIntegration(vi)
	assert.Equal(t, vi, sys.verifierIntegration)
}

func TestLLMOpsSystem_SetVerifierIntegration_Nil(t *testing.T) {
	sys := NewLLMOpsSystem(nil, nil)
	sys.SetVerifierIntegration(nil)
	assert.Nil(t, sys.verifierIntegration)
}

// --- Getter tests (before Initialize) ---

func TestLLMOpsSystem_GetPromptRegistry_BeforeInit(t *testing.T) {
	sys := NewLLMOpsSystem(nil, nil)
	assert.Nil(t, sys.GetPromptRegistry())
}

func TestLLMOpsSystem_GetExperimentManager_BeforeInit(t *testing.T) {
	sys := NewLLMOpsSystem(nil, nil)
	assert.Nil(t, sys.GetExperimentManager())
}

func TestLLMOpsSystem_GetEvaluator_BeforeInit(t *testing.T) {
	sys := NewLLMOpsSystem(nil, nil)
	assert.Nil(t, sys.GetEvaluator())
}

func TestLLMOpsSystem_GetAlertManager_BeforeInit(t *testing.T) {
	sys := NewLLMOpsSystem(nil, nil)
	assert.Nil(t, sys.GetAlertManager())
}

// --- Getter tests (after Initialize) ---

func TestLLMOpsSystem_GetPromptRegistry_AfterInit(t *testing.T) {
	sys := newInitializedSystem(t)
	assert.NotNil(t, sys.GetPromptRegistry())
}

func TestLLMOpsSystem_GetExperimentManager_AfterInit(t *testing.T) {
	sys := newInitializedSystem(t)
	assert.NotNil(t, sys.GetExperimentManager())
}

func TestLLMOpsSystem_GetEvaluator_AfterInit(t *testing.T) {
	sys := newInitializedSystem(t)
	assert.NotNil(t, sys.GetEvaluator())
}

func TestLLMOpsSystem_GetAlertManager_AfterInit(t *testing.T) {
	sys := newInitializedSystem(t)
	assert.NotNil(t, sys.GetAlertManager())
}

// --- CreatePromptExperiment tests ---

func TestLLMOpsSystem_CreatePromptExperiment_Success(t *testing.T) {
	sys := newInitializedSystem(t)
	ctx := context.Background()

	control := &PromptVersion{Name: "prompt-control", Version: "1.0", Content: "Control prompt"}
	treatment := &PromptVersion{Name: "prompt-treatment", Version: "1.0", Content: "Treatment prompt"}

	exp, err := sys.CreatePromptExperiment(ctx, "AB test", control, treatment, 0.5)
	require.NoError(t, err)
	require.NotNil(t, exp)
	assert.NotEmpty(t, exp.ID)
	assert.Equal(t, "AB test", exp.Name)
	assert.Len(t, exp.Variants, 2)
	assert.True(t, exp.Variants[0].IsControl)
	assert.False(t, exp.Variants[1].IsControl)
	assert.Equal(t, "prompt-control", exp.Variants[0].PromptName)
	assert.Equal(t, "prompt-treatment", exp.Variants[1].PromptName)
	assert.Contains(t, exp.Metrics, "quality")
	assert.Contains(t, exp.Metrics, "latency")
	assert.Contains(t, exp.Metrics, "satisfaction")
	assert.Equal(t, "quality", exp.TargetMetric)
}

func TestLLMOpsSystem_CreatePromptExperiment_TrafficSplitCorrect(t *testing.T) {
	sys := newInitializedSystem(t)
	ctx := context.Background()

	control := &PromptVersion{Name: "ctrl", Version: "1.0", Content: "Control"}
	treatment := &PromptVersion{Name: "treat", Version: "1.0", Content: "Treatment"}

	exp, err := sys.CreatePromptExperiment(ctx, "split-test", control, treatment, 0.3)
	require.NoError(t, err)

	controlSplit := exp.TrafficSplit[exp.Variants[0].ID]
	treatmentSplit := exp.TrafficSplit[exp.Variants[1].ID]
	assert.InDelta(t, 0.7, controlSplit, 0.001)
	assert.InDelta(t, 0.3, treatmentSplit, 0.001)
}

func TestLLMOpsSystem_CreatePromptExperiment_ControlPromptCreateFails(t *testing.T) {
	sys := newInitializedSystem(t)
	ctx := context.Background()

	// Create the control prompt first to trigger duplicate error
	controlPre := &PromptVersion{Name: "dup-prompt", Version: "1.0", Content: "Already exists"}
	require.NoError(t, sys.GetPromptRegistry().Create(ctx, controlPre))

	control := &PromptVersion{Name: "dup-prompt", Version: "1.0", Content: "Duplicate control"}
	treatment := &PromptVersion{Name: "treat-ok", Version: "1.0", Content: "Treatment"}

	_, err := sys.CreatePromptExperiment(ctx, "fail-test", control, treatment, 0.5)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "control prompt")
}

func TestLLMOpsSystem_CreatePromptExperiment_TreatmentPromptCreateFails(t *testing.T) {
	sys := newInitializedSystem(t)
	ctx := context.Background()

	// Pre-create the treatment to trigger duplicate error
	treatPre := &PromptVersion{Name: "dup-treat", Version: "1.0", Content: "Already exists"}
	require.NoError(t, sys.GetPromptRegistry().Create(ctx, treatPre))

	control := &PromptVersion{Name: "ctrl-ok", Version: "1.0", Content: "Control"}
	treatment := &PromptVersion{Name: "dup-treat", Version: "1.0", Content: "Duplicate treatment"}

	_, err := sys.CreatePromptExperiment(ctx, "fail-test", control, treatment, 0.5)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "treatment prompt")
}

// --- CreateModelExperiment tests ---

func TestLLMOpsSystem_CreateModelExperiment_Success(t *testing.T) {
	sys := newInitializedSystem(t)
	ctx := context.Background()

	models := []string{"gpt-4", "claude-3", "gemini-pro"}
	params := map[string]interface{}{"temperature": 0.7}

	exp, err := sys.CreateModelExperiment(ctx, "model-compare", models, params)
	require.NoError(t, err)
	require.NotNil(t, exp)
	assert.NotEmpty(t, exp.ID)
	assert.Equal(t, "model-compare", exp.Name)
	assert.Len(t, exp.Variants, 3)
	assert.True(t, exp.Variants[0].IsControl)
	assert.False(t, exp.Variants[1].IsControl)
	assert.False(t, exp.Variants[2].IsControl)
	assert.Equal(t, "gpt-4", exp.Variants[0].ModelName)
	assert.Equal(t, "claude-3", exp.Variants[1].ModelName)
	assert.Equal(t, "gemini-pro", exp.Variants[2].ModelName)
	assert.Contains(t, exp.Metrics, "quality")
	assert.Contains(t, exp.Metrics, "latency")
	assert.Contains(t, exp.Metrics, "cost")
}

func TestLLMOpsSystem_CreateModelExperiment_EqualTrafficSplit(t *testing.T) {
	sys := newInitializedSystem(t)
	ctx := context.Background()

	exp, err := sys.CreateModelExperiment(ctx, "split-test", []string{"a", "b", "c"}, nil)
	require.NoError(t, err)

	expectedSplit := 1.0 / 3.0
	for _, v := range exp.Variants {
		assert.InDelta(t, expectedSplit, exp.TrafficSplit[v.ID], 0.001)
	}
}

func TestLLMOpsSystem_CreateModelExperiment_TwoModels(t *testing.T) {
	sys := newInitializedSystem(t)
	ctx := context.Background()

	exp, err := sys.CreateModelExperiment(ctx, "two-model", []string{"a", "b"}, nil)
	require.NoError(t, err)
	assert.Len(t, exp.Variants, 2)
	for _, v := range exp.Variants {
		assert.InDelta(t, 0.5, exp.TrafficSplit[v.ID], 0.001)
	}
}

func TestLLMOpsSystem_CreateModelExperiment_InsufficientModels_Zero(t *testing.T) {
	sys := newInitializedSystem(t)
	ctx := context.Background()

	_, err := sys.CreateModelExperiment(ctx, "fail-test", []string{}, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "at least 2 models required")
}

func TestLLMOpsSystem_CreateModelExperiment_InsufficientModels_One(t *testing.T) {
	sys := newInitializedSystem(t)
	ctx := context.Background()

	_, err := sys.CreateModelExperiment(ctx, "fail-test", []string{"only-one"}, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "at least 2 models required")
}

func TestLLMOpsSystem_CreateModelExperiment_WithParameters(t *testing.T) {
	sys := newInitializedSystem(t)
	ctx := context.Background()

	params := map[string]interface{}{
		"temperature": 0.7,
		"max_tokens":  1024,
	}
	exp, err := sys.CreateModelExperiment(ctx, "param-test", []string{"a", "b"}, params)
	require.NoError(t, err)

	for _, v := range exp.Variants {
		assert.Equal(t, 0.7, v.Parameters["temperature"])
		assert.Equal(t, 1024, v.Parameters["max_tokens"])
	}
}

func TestLLMOpsSystem_CreateModelExperiment_NilParameters(t *testing.T) {
	sys := newInitializedSystem(t)
	ctx := context.Background()

	exp, err := sys.CreateModelExperiment(ctx, "nil-params", []string{"a", "b"}, nil)
	require.NoError(t, err)
	for _, v := range exp.Variants {
		assert.Nil(t, v.Parameters)
	}
}

// --- debateEvaluatorAdapter.Evaluate tests ---

func TestDebateEvaluatorAdapter_Evaluate_Success(t *testing.T) {
	mockEval := &mockDebateLLMEvaluator{
		scores: map[string]float64{"accuracy": 0.95, "relevance": 0.88},
	}
	adapter := &debateEvaluatorAdapter{evaluator: mockEval}

	ctx := context.Background()
	scores, err := adapter.Evaluate(ctx, "test prompt", "test response", "expected", []string{"accuracy", "relevance"})
	require.NoError(t, err)
	assert.InDelta(t, 0.95, scores["accuracy"], 0.001)
	assert.InDelta(t, 0.88, scores["relevance"], 0.001)
}

func TestDebateEvaluatorAdapter_Evaluate_Error(t *testing.T) {
	mockEval := &mockDebateLLMEvaluator{err: fmt.Errorf("debate failed")}
	adapter := &debateEvaluatorAdapter{evaluator: mockEval}

	ctx := context.Background()
	_, err := adapter.Evaluate(ctx, "test prompt", "test response", "expected", []string{"accuracy"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "debate failed")
}

// --- InMemoryAlertManager tests ---

func TestNewInMemoryAlertManager_WithLogger(t *testing.T) {
	logger := logrus.New()
	mgr := NewInMemoryAlertManager(logger)
	require.NotNil(t, mgr)
	assert.Equal(t, logger, mgr.logger)
	assert.NotNil(t, mgr.alerts)
	assert.NotNil(t, mgr.callbacks)
}

func TestNewInMemoryAlertManager_NilLogger(t *testing.T) {
	mgr := NewInMemoryAlertManager(nil)
	require.NotNil(t, mgr)
	require.NotNil(t, mgr.logger)
}

func TestInMemoryAlertManager_Create_Success(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	mgr := NewInMemoryAlertManager(logger)
	ctx := context.Background()

	alert := &Alert{
		Type:     AlertTypeRegression,
		Severity: AlertSeverityWarning,
		Message:  "Pass rate dropped",
		Source:   "evaluation",
	}
	err := mgr.Create(ctx, alert)
	require.NoError(t, err)
	assert.NotEmpty(t, alert.ID)
	assert.False(t, alert.CreatedAt.IsZero())
}

func TestInMemoryAlertManager_Create_WithExistingID(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	mgr := NewInMemoryAlertManager(logger)
	ctx := context.Background()

	alert := &Alert{
		ID:       "custom-alert-id",
		Type:     AlertTypeThreshold,
		Severity: AlertSeverityCritical,
		Message:  "Threshold breached",
	}
	err := mgr.Create(ctx, alert)
	require.NoError(t, err)
	assert.Equal(t, "custom-alert-id", alert.ID)
}

func TestInMemoryAlertManager_Create_WithExistingTimestamp(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	mgr := NewInMemoryAlertManager(logger)
	ctx := context.Background()

	customTime := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)
	alert := &Alert{
		Type:      AlertTypeAnomaly,
		Severity:  AlertSeverityInfo,
		Message:   "Anomaly detected",
		CreatedAt: customTime,
	}
	err := mgr.Create(ctx, alert)
	require.NoError(t, err)
	assert.Equal(t, customTime, alert.CreatedAt)
}

func TestInMemoryAlertManager_Create_NotifiesSubscribers(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	mgr := NewInMemoryAlertManager(logger)
	ctx := context.Background()

	var received sync.WaitGroup
	received.Add(1)
	var receivedAlert *Alert

	err := mgr.Subscribe(ctx, func(alert *Alert) error {
		receivedAlert = alert
		received.Done()
		return nil
	})
	require.NoError(t, err)

	alert := &Alert{
		Type:     AlertTypeRegression,
		Severity: AlertSeverityWarning,
		Message:  "Test alert",
	}
	require.NoError(t, mgr.Create(ctx, alert))

	received.Wait()
	assert.Equal(t, "Test alert", receivedAlert.Message)
}

func TestInMemoryAlertManager_Create_SubscriberError_DoesNotBlock(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	mgr := NewInMemoryAlertManager(logger)
	ctx := context.Background()

	err := mgr.Subscribe(ctx, func(alert *Alert) error {
		return fmt.Errorf("callback error")
	})
	require.NoError(t, err)

	alert := &Alert{
		Type:     AlertTypeRegression,
		Severity: AlertSeverityWarning,
		Message:  "Test alert",
	}
	// Should not panic or block even if callback errors
	require.NoError(t, mgr.Create(ctx, alert))
	// Give goroutine time to run
	time.Sleep(50 * time.Millisecond)
}

// --- InMemoryAlertManager List tests ---

func TestInMemoryAlertManager_List_NilFilter(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	mgr := NewInMemoryAlertManager(logger)
	ctx := context.Background()

	require.NoError(t, mgr.Create(ctx, &Alert{Type: AlertTypeRegression, Severity: AlertSeverityWarning, Message: "a"}))
	require.NoError(t, mgr.Create(ctx, &Alert{Type: AlertTypeThreshold, Severity: AlertSeverityCritical, Message: "b"}))

	alerts, err := mgr.List(ctx, nil)
	require.NoError(t, err)
	assert.Len(t, alerts, 2)
}

func TestInMemoryAlertManager_List_EmptyResult(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	mgr := NewInMemoryAlertManager(logger)
	ctx := context.Background()

	alerts, err := mgr.List(ctx, nil)
	require.NoError(t, err)
	assert.Empty(t, alerts)
}

func TestInMemoryAlertManager_List_FilterByTypes(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	mgr := NewInMemoryAlertManager(logger)
	ctx := context.Background()

	require.NoError(t, mgr.Create(ctx, &Alert{Type: AlertTypeRegression, Severity: AlertSeverityWarning, Message: "regression"}))
	require.NoError(t, mgr.Create(ctx, &Alert{Type: AlertTypeThreshold, Severity: AlertSeverityCritical, Message: "threshold"}))
	require.NoError(t, mgr.Create(ctx, &Alert{Type: AlertTypeAnomaly, Severity: AlertSeverityInfo, Message: "anomaly"}))

	alerts, err := mgr.List(ctx, &AlertFilter{Types: []AlertType{AlertTypeRegression}})
	require.NoError(t, err)
	assert.Len(t, alerts, 1)
	assert.Equal(t, AlertTypeRegression, alerts[0].Type)
}

func TestInMemoryAlertManager_List_FilterByMultipleTypes(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	mgr := NewInMemoryAlertManager(logger)
	ctx := context.Background()

	require.NoError(t, mgr.Create(ctx, &Alert{Type: AlertTypeRegression, Severity: AlertSeverityWarning, Message: "a"}))
	require.NoError(t, mgr.Create(ctx, &Alert{Type: AlertTypeThreshold, Severity: AlertSeverityCritical, Message: "b"}))
	require.NoError(t, mgr.Create(ctx, &Alert{Type: AlertTypeAnomaly, Severity: AlertSeverityInfo, Message: "c"}))

	alerts, err := mgr.List(ctx, &AlertFilter{Types: []AlertType{AlertTypeRegression, AlertTypeAnomaly}})
	require.NoError(t, err)
	assert.Len(t, alerts, 2)
}

func TestInMemoryAlertManager_List_FilterBySeverities(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	mgr := NewInMemoryAlertManager(logger)
	ctx := context.Background()

	require.NoError(t, mgr.Create(ctx, &Alert{Type: AlertTypeRegression, Severity: AlertSeverityWarning, Message: "warn"}))
	require.NoError(t, mgr.Create(ctx, &Alert{Type: AlertTypeThreshold, Severity: AlertSeverityCritical, Message: "crit"}))
	require.NoError(t, mgr.Create(ctx, &Alert{Type: AlertTypeAnomaly, Severity: AlertSeverityInfo, Message: "info"}))

	alerts, err := mgr.List(ctx, &AlertFilter{Severities: []AlertSeverity{AlertSeverityCritical}})
	require.NoError(t, err)
	assert.Len(t, alerts, 1)
	assert.Equal(t, AlertSeverityCritical, alerts[0].Severity)
}

func TestInMemoryAlertManager_List_FilterByMultipleSeverities(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	mgr := NewInMemoryAlertManager(logger)
	ctx := context.Background()

	require.NoError(t, mgr.Create(ctx, &Alert{Type: AlertTypeRegression, Severity: AlertSeverityWarning, Message: "warn"}))
	require.NoError(t, mgr.Create(ctx, &Alert{Type: AlertTypeThreshold, Severity: AlertSeverityCritical, Message: "crit"}))
	require.NoError(t, mgr.Create(ctx, &Alert{Type: AlertTypeAnomaly, Severity: AlertSeverityInfo, Message: "info"}))

	alerts, err := mgr.List(ctx, &AlertFilter{Severities: []AlertSeverity{AlertSeverityWarning, AlertSeverityInfo}})
	require.NoError(t, err)
	assert.Len(t, alerts, 2)
}

func TestInMemoryAlertManager_List_FilterBySource(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	mgr := NewInMemoryAlertManager(logger)
	ctx := context.Background()

	require.NoError(t, mgr.Create(ctx, &Alert{Type: AlertTypeRegression, Severity: AlertSeverityWarning, Message: "a", Source: "evaluation"}))
	require.NoError(t, mgr.Create(ctx, &Alert{Type: AlertTypeThreshold, Severity: AlertSeverityCritical, Message: "b", Source: "experiment"}))

	alerts, err := mgr.List(ctx, &AlertFilter{Source: "evaluation"})
	require.NoError(t, err)
	assert.Len(t, alerts, 1)
	assert.Equal(t, "evaluation", alerts[0].Source)
}

func TestInMemoryAlertManager_List_FilterByUnacked(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	mgr := NewInMemoryAlertManager(logger)
	ctx := context.Background()

	require.NoError(t, mgr.Create(ctx, &Alert{ID: "a1", Type: AlertTypeRegression, Severity: AlertSeverityWarning, Message: "unacked"}))
	require.NoError(t, mgr.Create(ctx, &Alert{ID: "a2", Type: AlertTypeThreshold, Severity: AlertSeverityCritical, Message: "will-ack"}))

	// Acknowledge one alert
	require.NoError(t, mgr.Acknowledge(ctx, "a2"))

	alerts, err := mgr.List(ctx, &AlertFilter{Unacked: true})
	require.NoError(t, err)
	assert.Len(t, alerts, 1)
	assert.Equal(t, "a1", alerts[0].ID)
}

func TestInMemoryAlertManager_List_FilterByStartTime(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	mgr := NewInMemoryAlertManager(logger)
	ctx := context.Background()

	oldTime := time.Now().Add(-2 * time.Hour)
	require.NoError(t, mgr.Create(ctx, &Alert{Type: AlertTypeRegression, Severity: AlertSeverityWarning, Message: "old", CreatedAt: oldTime}))
	require.NoError(t, mgr.Create(ctx, &Alert{Type: AlertTypeThreshold, Severity: AlertSeverityCritical, Message: "new"}))

	startTime := time.Now().Add(-1 * time.Hour)
	alerts, err := mgr.List(ctx, &AlertFilter{StartTime: &startTime})
	require.NoError(t, err)
	assert.Len(t, alerts, 1)
	assert.Equal(t, "new", alerts[0].Message)
}

func TestInMemoryAlertManager_List_FilterByLimit(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	mgr := NewInMemoryAlertManager(logger)
	ctx := context.Background()

	for i := 0; i < 5; i++ {
		require.NoError(t, mgr.Create(ctx, &Alert{
			Type:     AlertTypeRegression,
			Severity: AlertSeverityWarning,
			Message:  fmt.Sprintf("alert-%d", i),
		}))
	}

	alerts, err := mgr.List(ctx, &AlertFilter{Limit: 3})
	require.NoError(t, err)
	assert.Len(t, alerts, 3)
}

func TestInMemoryAlertManager_List_FilterLimitZero(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	mgr := NewInMemoryAlertManager(logger)
	ctx := context.Background()

	require.NoError(t, mgr.Create(ctx, &Alert{Type: AlertTypeRegression, Severity: AlertSeverityWarning, Message: "a"}))
	require.NoError(t, mgr.Create(ctx, &Alert{Type: AlertTypeThreshold, Severity: AlertSeverityCritical, Message: "b"}))

	alerts, err := mgr.List(ctx, &AlertFilter{Limit: 0})
	require.NoError(t, err)
	assert.Len(t, alerts, 2, "Limit=0 should not filter")
}

func TestInMemoryAlertManager_List_FilterLimitExceedsCount(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	mgr := NewInMemoryAlertManager(logger)
	ctx := context.Background()

	require.NoError(t, mgr.Create(ctx, &Alert{Type: AlertTypeRegression, Severity: AlertSeverityWarning, Message: "a"}))

	alerts, err := mgr.List(ctx, &AlertFilter{Limit: 100})
	require.NoError(t, err)
	assert.Len(t, alerts, 1)
}

func TestInMemoryAlertManager_List_CombinedFilters(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	mgr := NewInMemoryAlertManager(logger)
	ctx := context.Background()

	require.NoError(t, mgr.Create(ctx, &Alert{Type: AlertTypeRegression, Severity: AlertSeverityWarning, Source: "eval", Message: "match"}))
	require.NoError(t, mgr.Create(ctx, &Alert{Type: AlertTypeRegression, Severity: AlertSeverityCritical, Source: "eval", Message: "wrong-sev"}))
	require.NoError(t, mgr.Create(ctx, &Alert{Type: AlertTypeThreshold, Severity: AlertSeverityWarning, Source: "eval", Message: "wrong-type"}))
	require.NoError(t, mgr.Create(ctx, &Alert{Type: AlertTypeRegression, Severity: AlertSeverityWarning, Source: "exp", Message: "wrong-src"}))

	alerts, err := mgr.List(ctx, &AlertFilter{
		Types:      []AlertType{AlertTypeRegression},
		Severities: []AlertSeverity{AlertSeverityWarning},
		Source:     "eval",
	})
	require.NoError(t, err)
	assert.Len(t, alerts, 1)
	assert.Equal(t, "match", alerts[0].Message)
}

// --- matchesFilter tests (edge cases) ---

func TestInMemoryAlertManager_MatchesFilter_NilFilter(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	mgr := NewInMemoryAlertManager(logger)

	alert := &Alert{Type: AlertTypeRegression, Severity: AlertSeverityWarning}
	assert.True(t, mgr.matchesFilter(alert, nil))
}

func TestInMemoryAlertManager_MatchesFilter_EmptyFilter(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	mgr := NewInMemoryAlertManager(logger)

	alert := &Alert{Type: AlertTypeRegression, Severity: AlertSeverityWarning}
	assert.True(t, mgr.matchesFilter(alert, &AlertFilter{}))
}

func TestInMemoryAlertManager_MatchesFilter_TypeMismatch(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	mgr := NewInMemoryAlertManager(logger)

	alert := &Alert{Type: AlertTypeRegression, Severity: AlertSeverityWarning}
	assert.False(t, mgr.matchesFilter(alert, &AlertFilter{Types: []AlertType{AlertTypeThreshold}}))
}

func TestInMemoryAlertManager_MatchesFilter_SeverityMismatch(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	mgr := NewInMemoryAlertManager(logger)

	alert := &Alert{Type: AlertTypeRegression, Severity: AlertSeverityWarning}
	assert.False(t, mgr.matchesFilter(alert, &AlertFilter{Severities: []AlertSeverity{AlertSeverityCritical}}))
}

func TestInMemoryAlertManager_MatchesFilter_SourceMismatch(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	mgr := NewInMemoryAlertManager(logger)

	alert := &Alert{Type: AlertTypeRegression, Source: "eval"}
	assert.False(t, mgr.matchesFilter(alert, &AlertFilter{Source: "experiment"}))
}

func TestInMemoryAlertManager_MatchesFilter_AckedExcluded(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	mgr := NewInMemoryAlertManager(logger)

	now := time.Now()
	alert := &Alert{Type: AlertTypeRegression, AckedAt: &now}
	assert.False(t, mgr.matchesFilter(alert, &AlertFilter{Unacked: true}))
}

func TestInMemoryAlertManager_MatchesFilter_BeforeStartTime(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	mgr := NewInMemoryAlertManager(logger)

	startTime := time.Now()
	alert := &Alert{Type: AlertTypeRegression, CreatedAt: startTime.Add(-1 * time.Hour)}
	assert.False(t, mgr.matchesFilter(alert, &AlertFilter{StartTime: &startTime}))
}

// --- Acknowledge tests ---

func TestInMemoryAlertManager_Acknowledge_Success(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	mgr := NewInMemoryAlertManager(logger)
	ctx := context.Background()

	alert := &Alert{ID: "ack-test", Type: AlertTypeRegression, Severity: AlertSeverityWarning, Message: "test"}
	require.NoError(t, mgr.Create(ctx, alert))

	err := mgr.Acknowledge(ctx, "ack-test")
	require.NoError(t, err)

	alerts, _ := mgr.List(ctx, nil)
	assert.NotNil(t, alerts[0].AckedAt)
}

func TestInMemoryAlertManager_Acknowledge_NotFound(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	mgr := NewInMemoryAlertManager(logger)
	ctx := context.Background()

	err := mgr.Acknowledge(ctx, "nonexistent")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "alert not found")
}

func TestInMemoryAlertManager_Acknowledge_AlreadyAcked(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	mgr := NewInMemoryAlertManager(logger)
	ctx := context.Background()

	alert := &Alert{ID: "ack-twice", Type: AlertTypeRegression, Severity: AlertSeverityWarning, Message: "test"}
	require.NoError(t, mgr.Create(ctx, alert))

	require.NoError(t, mgr.Acknowledge(ctx, "ack-twice"))
	firstAck := alert.AckedAt

	require.NoError(t, mgr.Acknowledge(ctx, "ack-twice"))
	// AckedAt should be updated
	assert.NotNil(t, alert.AckedAt)
	assert.True(t, !alert.AckedAt.Before(*firstAck))
}

// --- Subscribe tests ---

func TestInMemoryAlertManager_Subscribe_Success(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	mgr := NewInMemoryAlertManager(logger)
	ctx := context.Background()

	err := mgr.Subscribe(ctx, func(alert *Alert) error {
		return nil
	})
	require.NoError(t, err)
	assert.Len(t, mgr.callbacks, 1)
}

func TestInMemoryAlertManager_Subscribe_Multiple(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	mgr := NewInMemoryAlertManager(logger)
	ctx := context.Background()

	for i := 0; i < 3; i++ {
		require.NoError(t, mgr.Subscribe(ctx, func(alert *Alert) error { return nil }))
	}
	assert.Len(t, mgr.callbacks, 3)
}

func TestInMemoryAlertManager_Subscribe_AllNotified(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	mgr := NewInMemoryAlertManager(logger)
	ctx := context.Background()

	var wg sync.WaitGroup
	count := 3
	wg.Add(count)

	for i := 0; i < count; i++ {
		require.NoError(t, mgr.Subscribe(ctx, func(alert *Alert) error {
			wg.Done()
			return nil
		}))
	}

	require.NoError(t, mgr.Create(ctx, &Alert{Type: AlertTypeRegression, Severity: AlertSeverityWarning, Message: "notify all"}))
	wg.Wait()
	// All 3 subscribers were notified
}

// --- Interface compliance ---

func TestInMemoryAlertManager_ImplementsAlertManager(t *testing.T) {
	var _ AlertManager = (*InMemoryAlertManager)(nil)
}

func TestDebateEvaluatorAdapter_ImplementsLLMEvaluator(t *testing.T) {
	var _ LLMEvaluator = (*debateEvaluatorAdapter)(nil)
}

// --- Concurrent access tests ---

func TestLLMOpsSystem_ConcurrentSetters(t *testing.T) {
	sys := NewLLMOpsSystem(nil, nil)

	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(2)
		go func() {
			defer wg.Done()
			sys.SetDebateEvaluator(&mockDebateLLMEvaluator{})
		}()
		go func() {
			defer wg.Done()
			sys.SetVerifierIntegration(NewVerifierIntegration(nil, nil, nil))
		}()
	}
	wg.Wait()
}

func TestLLMOpsSystem_ConcurrentGetters(t *testing.T) {
	sys := newInitializedSystem(t)

	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(4)
		go func() {
			defer wg.Done()
			_ = sys.GetPromptRegistry()
		}()
		go func() {
			defer wg.Done()
			_ = sys.GetExperimentManager()
		}()
		go func() {
			defer wg.Done()
			_ = sys.GetEvaluator()
		}()
		go func() {
			defer wg.Done()
			_ = sys.GetAlertManager()
		}()
	}
	wg.Wait()
}

func TestInMemoryAlertManager_ConcurrentCreateAndList(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	mgr := NewInMemoryAlertManager(logger)
	ctx := context.Background()

	var wg sync.WaitGroup
	n := 20

	// Concurrent creates
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			_ = mgr.Create(ctx, &Alert{
				Type:     AlertTypeRegression,
				Severity: AlertSeverityWarning,
				Message:  fmt.Sprintf("alert-%d", idx),
			})
		}(i)
	}

	// Concurrent reads
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _ = mgr.List(ctx, nil)
		}()
	}

	wg.Wait()
}

func TestInMemoryAlertManager_ConcurrentAcknowledge(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	mgr := NewInMemoryAlertManager(logger)
	ctx := context.Background()

	// Create some alerts
	for i := 0; i < 10; i++ {
		require.NoError(t, mgr.Create(ctx, &Alert{
			ID:       fmt.Sprintf("alert-%d", i),
			Type:     AlertTypeRegression,
			Severity: AlertSeverityWarning,
			Message:  fmt.Sprintf("alert-%d", i),
		}))
	}

	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			_ = mgr.Acknowledge(ctx, fmt.Sprintf("alert-%d", idx))
		}(i)
	}
	wg.Wait()
}

// --- Full lifecycle integration test ---

func TestLLMOpsSystem_FullLifecycle(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)

	// 1. Create system with config
	cfg := DefaultLLMOpsConfig()
	sys := NewLLMOpsSystem(cfg, logger)
	require.NotNil(t, sys)

	// 2. Set debate evaluator
	debateEval := &mockDebateLLMEvaluator{scores: map[string]float64{"quality": 0.9}}
	sys.SetDebateEvaluator(debateEval)

	// 3. Set verifier integration
	vi := NewVerifierIntegration(
		func(name string) float64 {
			scores := map[string]float64{"openai": 8.5, "claude": 9.0}
			return scores[name]
		},
		func(name string) bool { return true },
		logger,
	)
	sys.SetVerifierIntegration(vi)

	// 4. Initialize
	require.NoError(t, sys.Initialize())
	assert.NotNil(t, sys.GetPromptRegistry())
	assert.NotNil(t, sys.GetExperimentManager())
	assert.NotNil(t, sys.GetEvaluator())
	assert.NotNil(t, sys.GetAlertManager())

	// 5. Create a prompt experiment
	ctx := context.Background()
	control := &PromptVersion{Name: "ctrl-lc", Version: "1.0", Content: "Control: {{question}}"}
	treatment := &PromptVersion{Name: "treat-lc", Version: "1.0", Content: "Treatment: {{question}}"}

	exp, err := sys.CreatePromptExperiment(ctx, "lifecycle-test", control, treatment, 0.5)
	require.NoError(t, err)
	assert.NotEmpty(t, exp.ID)

	// 6. Create a model experiment
	modelExp, err := sys.CreateModelExperiment(ctx, "model-lifecycle", []string{"openai", "claude"}, nil)
	require.NoError(t, err)
	assert.NotEmpty(t, modelExp.ID)

	// 7. Use verifier to select best provider
	best, score := vi.SelectBestProvider([]string{"openai", "claude"})
	assert.Equal(t, "claude", best)
	assert.InDelta(t, 9.0, score, 0.001)

	// 8. Create an alert
	alertMgr := sys.GetAlertManager()
	require.NoError(t, alertMgr.Create(ctx, &Alert{
		Type:     AlertTypeExperiment,
		Severity: AlertSeverityInfo,
		Message:  "Experiment started",
		Source:   "lifecycle-test",
	}))

	alerts, err := alertMgr.List(ctx, nil)
	require.NoError(t, err)
	assert.Len(t, alerts, 1)
}

// --- helper functions ---

func newInitializedSystem(t *testing.T) *LLMOpsSystem {
	t.Helper()
	logger := logrus.New()
	logger.SetLevel(logrus.PanicLevel)
	sys := NewLLMOpsSystem(nil, logger)
	require.NoError(t, sys.Initialize())
	return sys
}
