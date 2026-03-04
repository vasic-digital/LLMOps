package llmops

import (
	"context"
	"fmt"
	"hash/fnv"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/sirupsen/logrus"
)

// InMemoryExperimentManager implements ExperimentManager
type InMemoryExperimentManager struct {
	experiments map[string]*Experiment
	metrics     map[string]map[string][]*metricSample // exp -> variant -> samples
	assignments map[string]map[string]string          // exp -> user -> variant
	mu          sync.RWMutex
	logger      *logrus.Logger
}

type metricSample struct {
	Value     float64
	Timestamp time.Time
}

// NewInMemoryExperimentManager creates a new experiment manager
func NewInMemoryExperimentManager(logger *logrus.Logger) *InMemoryExperimentManager {
	if logger == nil {
		logger = logrus.New()
	}
	return &InMemoryExperimentManager{
		experiments: make(map[string]*Experiment),
		metrics:     make(map[string]map[string][]*metricSample),
		assignments: make(map[string]map[string]string),
		logger:      logger,
	}
}

// Create creates a new experiment
func (m *InMemoryExperimentManager) Create(ctx context.Context, exp *Experiment) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if exp.Name == "" {
		return fmt.Errorf("experiment name is required")
	}
	if len(exp.Variants) < 2 {
		return fmt.Errorf("at least 2 variants required")
	}

	// Generate ID if not present
	if exp.ID == "" {
		exp.ID = uuid.New().String()
	}

	// Ensure each variant has an ID
	for _, v := range exp.Variants {
		if v.ID == "" {
			v.ID = uuid.New().String()
		}
	}

	// Validate and normalize traffic split
	if err := m.validateTrafficSplit(exp); err != nil {
		return err
	}

	exp.Status = ExperimentStatusDraft
	exp.CreatedAt = time.Now()
	exp.UpdatedAt = time.Now()

	m.experiments[exp.ID] = exp
	m.metrics[exp.ID] = make(map[string][]*metricSample)
	m.assignments[exp.ID] = make(map[string]string)

	// Initialize metrics for each variant
	for _, v := range exp.Variants {
		m.metrics[exp.ID][v.ID] = make([]*metricSample, 0)
	}

	m.logger.WithFields(logrus.Fields{
		"id":       exp.ID,
		"name":     exp.Name,
		"variants": len(exp.Variants),
	}).Info("Experiment created")

	return nil
}

// Get retrieves an experiment
func (m *InMemoryExperimentManager) Get(ctx context.Context, id string) (*Experiment, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	exp, ok := m.experiments[id]
	if !ok {
		return nil, fmt.Errorf("experiment not found: %s", id)
	}

	return exp, nil
}

// List lists all experiments
func (m *InMemoryExperimentManager) List(ctx context.Context, status ExperimentStatus) ([]*Experiment, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var result []*Experiment
	for _, exp := range m.experiments {
		if status == "" || exp.Status == status {
			result = append(result, exp)
		}
	}

	// Sort by creation time (newest first)
	sort.Slice(result, func(i, j int) bool {
		return result[i].CreatedAt.After(result[j].CreatedAt)
	})

	return result, nil
}

// Start starts an experiment
func (m *InMemoryExperimentManager) Start(ctx context.Context, id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	exp, ok := m.experiments[id]
	if !ok {
		return fmt.Errorf("experiment not found: %s", id)
	}

	if exp.Status != ExperimentStatusDraft && exp.Status != ExperimentStatusPaused {
		return fmt.Errorf("cannot start experiment in status: %s", exp.Status)
	}

	now := time.Now()
	exp.Status = ExperimentStatusRunning
	if exp.StartTime == nil {
		exp.StartTime = &now
	}
	exp.UpdatedAt = now

	m.logger.WithField("id", id).Info("Experiment started")

	return nil
}

// Pause pauses an experiment
func (m *InMemoryExperimentManager) Pause(ctx context.Context, id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	exp, ok := m.experiments[id]
	if !ok {
		return fmt.Errorf("experiment not found: %s", id)
	}

	if exp.Status != ExperimentStatusRunning {
		return fmt.Errorf("cannot pause experiment in status: %s", exp.Status)
	}

	exp.Status = ExperimentStatusPaused
	exp.UpdatedAt = time.Now()

	m.logger.WithField("id", id).Info("Experiment paused")

	return nil
}

// Complete completes an experiment
func (m *InMemoryExperimentManager) Complete(ctx context.Context, id string, winner string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	exp, ok := m.experiments[id]
	if !ok {
		return fmt.Errorf("experiment not found: %s", id)
	}

	// Validate winner if specified
	if winner != "" {
		found := false
		for _, v := range exp.Variants {
			if v.ID == winner {
				found = true
				break
			}
		}
		if !found {
			return fmt.Errorf("invalid winner variant: %s", winner)
		}
	}

	now := time.Now()
	exp.Status = ExperimentStatusCompleted
	exp.EndTime = &now
	exp.Winner = winner
	exp.UpdatedAt = now

	m.logger.WithFields(logrus.Fields{
		"id":     id,
		"winner": winner,
	}).Info("Experiment completed")

	return nil
}

// Cancel cancels an experiment
func (m *InMemoryExperimentManager) Cancel(ctx context.Context, id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	exp, ok := m.experiments[id]
	if !ok {
		return fmt.Errorf("experiment not found: %s", id)
	}

	if exp.Status == ExperimentStatusCompleted || exp.Status == ExperimentStatusCancelled {
		return fmt.Errorf("experiment already finalized")
	}

	now := time.Now()
	exp.Status = ExperimentStatusCancelled
	exp.EndTime = &now
	exp.UpdatedAt = now

	m.logger.WithField("id", id).Info("Experiment cancelled")

	return nil
}

// AssignVariant assigns a variant for a request
func (m *InMemoryExperimentManager) AssignVariant(ctx context.Context, experimentID, userID string) (*Variant, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	exp, ok := m.experiments[experimentID]
	if !ok {
		return nil, fmt.Errorf("experiment not found: %s", experimentID)
	}

	if exp.Status != ExperimentStatusRunning {
		return nil, fmt.Errorf("experiment not running: %s", exp.Status)
	}

	// Check for existing assignment
	if variantID, exists := m.assignments[experimentID][userID]; exists {
		for _, v := range exp.Variants {
			if v.ID == variantID {
				return v, nil
			}
		}
	}

	// Assign based on traffic split
	variant := m.selectVariant(exp, userID)

	// Store assignment
	m.assignments[experimentID][userID] = variant.ID

	m.logger.WithFields(logrus.Fields{
		"experiment": experimentID,
		"user":       userID,
		"variant":    variant.ID,
	}).Debug("Variant assigned")

	return variant, nil
}

// RecordMetric records a metric for a variant
func (m *InMemoryExperimentManager) RecordMetric(ctx context.Context, experimentID, variantID, metric string, value float64) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, ok := m.experiments[experimentID]; !ok {
		return fmt.Errorf("experiment not found: %s", experimentID)
	}

	if _, ok := m.metrics[experimentID][variantID]; !ok {
		return fmt.Errorf("variant not found: %s", variantID)
	}

	// Store sample (could store by metric name for more granularity)
	m.metrics[experimentID][variantID] = append(m.metrics[experimentID][variantID], &metricSample{
		Value:     value,
		Timestamp: time.Now(),
	})

	return nil
}

// GetResults gets experiment results
func (m *InMemoryExperimentManager) GetResults(ctx context.Context, experimentID string) (*ExperimentResult, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	exp, ok := m.experiments[experimentID]
	if !ok {
		return nil, fmt.Errorf("experiment not found: %s", experimentID)
	}

	result := &ExperimentResult{
		ExperimentID:   experimentID,
		VariantResults: make(map[string]*VariantResult),
		StartTime:      exp.CreatedAt,
	}

	if exp.StartTime != nil {
		result.StartTime = *exp.StartTime
	}
	if exp.EndTime != nil {
		result.EndTime = *exp.EndTime
	} else {
		result.EndTime = time.Now()
	}

	// Calculate results for each variant
	var controlResult *VariantResult
	for _, v := range exp.Variants {
		samples := m.metrics[experimentID][v.ID]
		vr := m.calculateVariantResult(v.ID, samples)
		result.VariantResults[v.ID] = vr
		result.TotalSamples += vr.SampleCount

		if v.IsControl {
			controlResult = vr
		}
	}

	// Calculate improvement vs control
	if controlResult != nil {
		for _, vr := range result.VariantResults {
			if vr.VariantID != controlResult.VariantID {
				if controlResult.MetricValues["primary"] != nil && controlResult.MetricValues["primary"].Value > 0 {
					if vr.MetricValues["primary"] != nil {
						vr.Improvement = (vr.MetricValues["primary"].Value - controlResult.MetricValues["primary"].Value) /
							controlResult.MetricValues["primary"].Value * 100
					}
				}
			}
		}
	}

	// Determine statistical significance and winner
	result.Significance, result.Confidence = m.calculateSignificance(result)

	if result.Confidence >= 0.95 {
		result.Winner = m.determineWinner(result)
		result.Recommendation = fmt.Sprintf("Deploy variant %s with %.1f%% confidence", result.Winner, result.Confidence*100)
	} else {
		result.Recommendation = "Continue experiment - insufficient confidence"
	}

	return result, nil
}

func (m *InMemoryExperimentManager) validateTrafficSplit(exp *Experiment) error {
	if exp.TrafficSplit == nil || len(exp.TrafficSplit) == 0 {
		// Default: equal split
		exp.TrafficSplit = make(map[string]float64)
		split := 1.0 / float64(len(exp.Variants))
		for _, v := range exp.Variants {
			exp.TrafficSplit[v.ID] = split
		}
		return nil
	}

	// Validate sum = 1.0
	var total float64
	for _, v := range exp.TrafficSplit {
		if v < 0 {
			return fmt.Errorf("traffic split cannot be negative")
		}
		total += v
	}

	if math.Abs(total-1.0) > 0.01 {
		return fmt.Errorf("traffic split must sum to 1.0, got %.2f", total)
	}

	// Ensure all variants have split
	for _, v := range exp.Variants {
		if _, ok := exp.TrafficSplit[v.ID]; !ok {
			return fmt.Errorf("missing traffic split for variant: %s", v.ID)
		}
	}

	return nil
}

func (m *InMemoryExperimentManager) selectVariant(exp *Experiment, userID string) *Variant {
	// Deterministic assignment based on hash
	h := fnv.New32a()
	h.Write([]byte(exp.ID + userID))
	hashValue := float64(h.Sum32()) / float64(^uint32(0))

	var cumulative float64
	for _, v := range exp.Variants {
		cumulative += exp.TrafficSplit[v.ID]
		if hashValue < cumulative {
			return v
		}
	}

	// Fallback to last variant
	return exp.Variants[len(exp.Variants)-1]
}

func (m *InMemoryExperimentManager) calculateVariantResult(variantID string, samples []*metricSample) *VariantResult {
	vr := &VariantResult{
		VariantID:    variantID,
		SampleCount:  len(samples),
		MetricValues: make(map[string]*MetricValue),
	}

	if len(samples) == 0 {
		return vr
	}

	// Calculate primary metric stats
	var sum, sumSq, min, max float64
	min = samples[0].Value
	max = samples[0].Value

	for _, s := range samples {
		sum += s.Value
		sumSq += s.Value * s.Value
		if s.Value < min {
			min = s.Value
		}
		if s.Value > max {
			max = s.Value
		}
	}

	mean := sum / float64(len(samples))
	variance := (sumSq / float64(len(samples))) - (mean * mean)
	stdDev := math.Sqrt(variance)

	vr.MetricValues["primary"] = &MetricValue{
		Name:   "primary",
		Value:  mean,
		StdDev: stdDev,
		Min:    min,
		Max:    max,
		Count:  len(samples),
	}

	return vr
}

func (m *InMemoryExperimentManager) calculateSignificance(result *ExperimentResult) (significance, confidence float64) {
	// Simplified significance calculation
	// In production, use proper statistical tests (t-test, chi-squared, etc.)

	if len(result.VariantResults) < 2 {
		return 0, 0
	}

	var values [][]float64
	for _, vr := range result.VariantResults {
		if vr.MetricValues["primary"] != nil {
			values = append(values, []float64{vr.MetricValues["primary"].Value, vr.MetricValues["primary"].StdDev, float64(vr.SampleCount)})
		}
	}

	if len(values) < 2 {
		return 0, 0
	}

	// Simple z-test approximation
	mean1, std1, n1 := values[0][0], values[0][1], values[0][2]
	mean2, std2, n2 := values[1][0], values[1][1], values[1][2]

	if n1 < 30 || n2 < 30 {
		return 0, 0.5 // Insufficient sample size
	}

	pooledSE := math.Sqrt((std1*std1)/n1 + (std2*std2)/n2)
	if pooledSE == 0 {
		return 0, 0
	}

	zScore := math.Abs(mean1-mean2) / pooledSE

	// Convert z-score to confidence (approximate)
	if zScore >= 2.576 {
		confidence = 0.99
	} else if zScore >= 1.96 {
		confidence = 0.95
	} else if zScore >= 1.645 {
		confidence = 0.90
	} else {
		confidence = 0.5 + 0.4*(zScore/1.645)
	}

	return zScore, confidence
}

func (m *InMemoryExperimentManager) determineWinner(result *ExperimentResult) string {
	var bestID string
	var bestValue float64

	for id, vr := range result.VariantResults {
		if mv := vr.MetricValues["primary"]; mv != nil {
			if mv.Value > bestValue {
				bestValue = mv.Value
				bestID = id
			}
		}
	}

	return bestID
}
