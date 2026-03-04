package llmops

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/sirupsen/logrus"
)

// LLMOpsSystem is the main orchestrator for MLOps/LLMOps
type LLMOpsSystem struct {
	promptRegistry      PromptRegistry
	experimentManager   ExperimentManager
	evaluator           ContinuousEvaluator
	alertManager        AlertManager
	debateEvaluator     DebateLLMEvaluator
	verifierIntegration *VerifierIntegration
	config              *LLMOpsConfig
	logger              *logrus.Logger
	mu                  sync.RWMutex
}

// LLMOpsConfig configuration for LLMOps
type LLMOpsConfig struct {
	EnableAutoEvaluation   bool               `json:"enable_auto_evaluation"`
	EvaluationInterval     time.Duration      `json:"evaluation_interval"`
	MinSamplesForSignif    int                `json:"min_samples_for_significance"`
	AlertThresholds        map[string]float64 `json:"alert_thresholds"`
	EnableDebateEvaluation bool               `json:"enable_debate_evaluation"`
}

// DefaultLLMOpsConfig returns default configuration
func DefaultLLMOpsConfig() *LLMOpsConfig {
	return &LLMOpsConfig{
		EnableAutoEvaluation:   true,
		EvaluationInterval:     24 * time.Hour,
		MinSamplesForSignif:    100,
		EnableDebateEvaluation: true,
		AlertThresholds: map[string]float64{
			"pass_rate":   0.85,
			"latency_p99": 5000, // ms
		},
	}
}

// DebateLLMEvaluator uses debate service for LLM evaluation
type DebateLLMEvaluator interface {
	EvaluateWithDebate(ctx context.Context, prompt, response, expected string, metrics []string) (map[string]float64, error)
}

// VerifierIntegration integrates with LLMsVerifier
type VerifierIntegration struct {
	getProviderScore  func(name string) float64
	isProviderHealthy func(name string) bool
	logger            *logrus.Logger
}

// NewVerifierIntegration creates verifier integration
func NewVerifierIntegration(getScore func(string) float64, isHealthy func(string) bool, logger *logrus.Logger) *VerifierIntegration {
	return &VerifierIntegration{
		getProviderScore:  getScore,
		isProviderHealthy: isHealthy,
		logger:            logger,
	}
}

// SelectBestProvider selects the best provider for an experiment
func (v *VerifierIntegration) SelectBestProvider(providers []string) (string, float64) {
	if v.getProviderScore == nil {
		if len(providers) > 0 {
			return providers[0], 0
		}
		return "", 0
	}

	var bestProvider string
	var bestScore float64

	for _, p := range providers {
		if v.isProviderHealthy != nil && !v.isProviderHealthy(p) {
			continue
		}
		score := v.getProviderScore(p)
		if score > bestScore {
			bestScore = score
			bestProvider = p
		}
	}

	return bestProvider, bestScore
}

// NewLLMOpsSystem creates the main LLMOps system
func NewLLMOpsSystem(config *LLMOpsConfig, logger *logrus.Logger) *LLMOpsSystem {
	if config == nil {
		config = DefaultLLMOpsConfig()
	}
	if logger == nil {
		logger = logrus.New()
	}

	return &LLMOpsSystem{
		config: config,
		logger: logger,
	}
}

// Initialize sets up all components
func (s *LLMOpsSystem) Initialize() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Create prompt registry
	s.promptRegistry = NewInMemoryPromptRegistry(s.logger)

	// Create alert manager
	s.alertManager = NewInMemoryAlertManager(s.logger)

	// Create evaluator
	var llmEval LLMEvaluator
	if s.config.EnableDebateEvaluation && s.debateEvaluator != nil {
		llmEval = &debateEvaluatorAdapter{evaluator: s.debateEvaluator}
	}
	s.evaluator = NewInMemoryContinuousEvaluator(llmEval, s.promptRegistry, s.alertManager, s.logger)

	// Create experiment manager
	s.experimentManager = NewInMemoryExperimentManager(s.logger)

	s.logger.Info("LLMOps system initialized")
	return nil
}

// SetDebateEvaluator sets the debate-based evaluator
func (s *LLMOpsSystem) SetDebateEvaluator(evaluator DebateLLMEvaluator) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.debateEvaluator = evaluator
}

// SetVerifierIntegration sets the verifier integration
func (s *LLMOpsSystem) SetVerifierIntegration(vi *VerifierIntegration) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.verifierIntegration = vi
}

// GetPromptRegistry returns the prompt registry
func (s *LLMOpsSystem) GetPromptRegistry() PromptRegistry {
	return s.promptRegistry
}

// GetExperimentManager returns the experiment manager
func (s *LLMOpsSystem) GetExperimentManager() ExperimentManager {
	return s.experimentManager
}

// GetEvaluator returns the continuous evaluator
func (s *LLMOpsSystem) GetEvaluator() ContinuousEvaluator {
	return s.evaluator
}

// GetAlertManager returns the alert manager
func (s *LLMOpsSystem) GetAlertManager() AlertManager {
	return s.alertManager
}

// CreatePromptExperiment creates an A/B test for prompts
func (s *LLMOpsSystem) CreatePromptExperiment(ctx context.Context, name string, controlPrompt, treatmentPrompt *PromptVersion, trafficSplit float64) (*Experiment, error) {
	// Create prompt versions
	if err := s.promptRegistry.Create(ctx, controlPrompt); err != nil {
		return nil, fmt.Errorf("failed to create control prompt: %w", err)
	}
	if err := s.promptRegistry.Create(ctx, treatmentPrompt); err != nil {
		return nil, fmt.Errorf("failed to create treatment prompt: %w", err)
	}

	// Create experiment
	exp := &Experiment{
		ID:          uuid.New().String(),
		Name:        name,
		Description: fmt.Sprintf("A/B test: %s vs %s", controlPrompt.Name, treatmentPrompt.Name),
		Variants: []*Variant{
			{
				ID:            uuid.New().String(),
				Name:          "Control",
				PromptName:    controlPrompt.Name,
				PromptVersion: controlPrompt.Version,
				IsControl:     true,
			},
			{
				ID:            uuid.New().String(),
				Name:          "Treatment",
				PromptName:    treatmentPrompt.Name,
				PromptVersion: treatmentPrompt.Version,
				IsControl:     false,
			},
		},
		TrafficSplit: map[string]float64{},
		Metrics:      []string{"quality", "latency", "satisfaction"},
		TargetMetric: "quality",
	}

	// Set traffic split
	exp.TrafficSplit[exp.Variants[0].ID] = 1 - trafficSplit
	exp.TrafficSplit[exp.Variants[1].ID] = trafficSplit

	if err := s.experimentManager.Create(ctx, exp); err != nil {
		return nil, err
	}

	s.logger.WithFields(logrus.Fields{
		"experiment":       exp.ID,
		"control_prompt":   controlPrompt.Name,
		"treatment_prompt": treatmentPrompt.Name,
		"traffic_split":    trafficSplit,
	}).Info("Prompt experiment created")

	return exp, nil
}

// CreateModelExperiment creates an A/B test for models
func (s *LLMOpsSystem) CreateModelExperiment(ctx context.Context, name string, models []string, parameters map[string]interface{}) (*Experiment, error) {
	if len(models) < 2 {
		return nil, fmt.Errorf("at least 2 models required for experiment")
	}

	variants := make([]*Variant, len(models))
	trafficSplit := make(map[string]float64)
	splitPct := 1.0 / float64(len(models))

	for i, model := range models {
		variants[i] = &Variant{
			ID:         uuid.New().String(),
			Name:       model,
			ModelName:  model,
			Parameters: parameters,
			IsControl:  i == 0,
		}
		trafficSplit[variants[i].ID] = splitPct
	}

	exp := &Experiment{
		ID:           uuid.New().String(),
		Name:         name,
		Description:  fmt.Sprintf("Model comparison: %v", models),
		Variants:     variants,
		TrafficSplit: trafficSplit,
		Metrics:      []string{"quality", "latency", "cost"},
		TargetMetric: "quality",
	}

	if err := s.experimentManager.Create(ctx, exp); err != nil {
		return nil, err
	}

	return exp, nil
}

// debateEvaluatorAdapter adapts DebateLLMEvaluator to LLMEvaluator
type debateEvaluatorAdapter struct {
	evaluator DebateLLMEvaluator
}

func (a *debateEvaluatorAdapter) Evaluate(ctx context.Context, prompt, response, expected string, metrics []string) (map[string]float64, error) {
	return a.evaluator.EvaluateWithDebate(ctx, prompt, response, expected, metrics)
}

// InMemoryAlertManager implements AlertManager
type InMemoryAlertManager struct {
	alerts    []*Alert
	callbacks []AlertCallback
	mu        sync.RWMutex
	logger    *logrus.Logger
}

// NewInMemoryAlertManager creates a new alert manager
func NewInMemoryAlertManager(logger *logrus.Logger) *InMemoryAlertManager {
	if logger == nil {
		logger = logrus.New()
	}
	return &InMemoryAlertManager{
		alerts:    make([]*Alert, 0),
		callbacks: make([]AlertCallback, 0),
		logger:    logger,
	}
}

// Create creates a new alert
func (m *InMemoryAlertManager) Create(ctx context.Context, alert *Alert) error {
	m.mu.Lock()
	if alert.ID == "" {
		alert.ID = uuid.New().String()
	}
	if alert.CreatedAt.IsZero() {
		alert.CreatedAt = time.Now()
	}
	m.alerts = append(m.alerts, alert)
	callbacks := m.callbacks
	m.mu.Unlock()

	// Notify subscribers
	for _, cb := range callbacks {
		go func(callback AlertCallback) {
			if err := callback(alert); err != nil {
				m.logger.WithError(err).Warn("Alert callback failed")
			}
		}(cb)
	}

	m.logger.WithFields(logrus.Fields{
		"id":       alert.ID,
		"type":     alert.Type,
		"severity": alert.Severity,
		"message":  alert.Message,
	}).Info("Alert created")

	return nil
}

// List lists alerts
func (m *InMemoryAlertManager) List(ctx context.Context, filter *AlertFilter) ([]*Alert, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var result []*Alert
	for _, alert := range m.alerts {
		if m.matchesFilter(alert, filter) {
			result = append(result, alert)
		}
	}

	// Apply limit
	if filter != nil && filter.Limit > 0 && filter.Limit < len(result) {
		result = result[:filter.Limit]
	}

	return result, nil
}

func (m *InMemoryAlertManager) matchesFilter(alert *Alert, filter *AlertFilter) bool {
	if filter == nil {
		return true
	}

	if len(filter.Types) > 0 {
		found := false
		for _, t := range filter.Types {
			if alert.Type == t {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	if len(filter.Severities) > 0 {
		found := false
		for _, s := range filter.Severities {
			if alert.Severity == s {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	if filter.Source != "" && alert.Source != filter.Source {
		return false
	}

	if filter.Unacked && alert.AckedAt != nil {
		return false
	}

	if filter.StartTime != nil && alert.CreatedAt.Before(*filter.StartTime) {
		return false
	}

	return true
}

// Acknowledge acknowledges an alert
func (m *InMemoryAlertManager) Acknowledge(ctx context.Context, alertID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	for _, alert := range m.alerts {
		if alert.ID == alertID {
			now := time.Now()
			alert.AckedAt = &now
			return nil
		}
	}

	return fmt.Errorf("alert not found: %s", alertID)
}

// Subscribe subscribes to alerts
func (m *InMemoryAlertManager) Subscribe(ctx context.Context, callback AlertCallback) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.callbacks = append(m.callbacks, callback)
	return nil
}
