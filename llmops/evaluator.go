package llmops

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/sirupsen/logrus"
)

// LLMEvaluator interface for LLM-based evaluation
type LLMEvaluator interface {
	Evaluate(ctx context.Context, prompt, response, expected string, metrics []string) (map[string]float64, error)
}

// InMemoryContinuousEvaluator implements ContinuousEvaluator
type InMemoryContinuousEvaluator struct {
	runs         map[string]*EvaluationRun
	datasets     map[string]*Dataset
	samples      map[string][]*DatasetSample // dataset ID -> samples
	schedules    map[string]*schedule
	evaluator    LLMEvaluator
	registry     PromptRegistry
	mu           sync.RWMutex
	logger       *logrus.Logger
	alertManager AlertManager
}

type schedule struct {
	run     *EvaluationRun
	cron    string
	lastRun time.Time
	stopCh  chan struct{}
}

// NewInMemoryContinuousEvaluator creates a new continuous evaluator
func NewInMemoryContinuousEvaluator(evaluator LLMEvaluator, registry PromptRegistry, alertManager AlertManager, logger *logrus.Logger) *InMemoryContinuousEvaluator {
	if logger == nil {
		logger = logrus.New()
	}
	return &InMemoryContinuousEvaluator{
		runs:         make(map[string]*EvaluationRun),
		datasets:     make(map[string]*Dataset),
		samples:      make(map[string][]*DatasetSample),
		schedules:    make(map[string]*schedule),
		evaluator:    evaluator,
		registry:     registry,
		alertManager: alertManager,
		logger:       logger,
	}
}

// CreateRun creates a new evaluation run
func (e *InMemoryContinuousEvaluator) CreateRun(ctx context.Context, run *EvaluationRun) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if run.Name == "" {
		return fmt.Errorf("run name is required")
	}
	if run.Dataset == "" {
		return fmt.Errorf("dataset is required")
	}

	// Validate dataset exists
	if _, ok := e.datasets[run.Dataset]; !ok {
		return fmt.Errorf("dataset not found: %s", run.Dataset)
	}

	if run.ID == "" {
		run.ID = uuid.New().String()
	}

	run.Status = EvaluationStatusPending
	run.CreatedAt = time.Now()

	e.runs[run.ID] = run

	e.logger.WithFields(logrus.Fields{
		"id":      run.ID,
		"name":    run.Name,
		"dataset": run.Dataset,
	}).Info("Evaluation run created")

	return nil
}

// StartRun starts an evaluation run
func (e *InMemoryContinuousEvaluator) StartRun(ctx context.Context, runID string) error {
	e.mu.Lock()
	run, ok := e.runs[runID]
	if !ok {
		e.mu.Unlock()
		return fmt.Errorf("run not found: %s", runID)
	}

	if run.Status != EvaluationStatusPending {
		e.mu.Unlock()
		return fmt.Errorf("run already started or completed")
	}

	now := time.Now()
	run.Status = EvaluationStatusRunning
	run.StartTime = &now

	samples := e.samples[run.Dataset]
	e.mu.Unlock()

	// Run evaluation asynchronously
	go e.executeRun(ctx, run, samples)

	return nil
}

func (e *InMemoryContinuousEvaluator) executeRun(ctx context.Context, run *EvaluationRun, samples []*DatasetSample) {
	results := &EvaluationResults{
		MetricScores:   make(map[string]float64),
		MetricDetails:  make(map[string]*MetricValue),
		FailureReasons: make(map[string]int),
		SampleResults:  make([]*SampleResult, 0, len(samples)),
	}

	// Get prompt template
	var promptTemplate string
	if e.registry != nil && run.PromptName != "" {
		version := run.PromptVersion
		if version == "" {
			version = "latest"
		}
		var prompt *PromptVersion
		var err error
		if version == "latest" {
			prompt, err = e.registry.GetLatest(ctx, run.PromptName)
		} else {
			prompt, err = e.registry.Get(ctx, run.PromptName, version)
		}
		if err == nil {
			promptTemplate = prompt.Content
		}
	}

	metricSums := make(map[string]float64)
	metricCounts := make(map[string]int)

	for _, sample := range samples {
		select {
		case <-ctx.Done():
			e.mu.Lock()
			run.Status = EvaluationStatusFailed
			e.mu.Unlock()
			return
		default:
		}

		result := e.evaluateSample(ctx, run, sample, promptTemplate)
		results.SampleResults = append(results.SampleResults, result)
		results.TotalSamples++

		if result.Passed {
			results.PassedSamples++
		} else {
			results.FailedSamples++
			if result.Error != "" {
				results.FailureReasons[result.Error]++
			}
		}

		// Accumulate metrics
		for metric, score := range result.Scores {
			metricSums[metric] += score
			metricCounts[metric]++
		}
	}

	// Calculate final metrics
	for metric, sum := range metricSums {
		if count := metricCounts[metric]; count > 0 {
			results.MetricScores[metric] = sum / float64(count)
		}
	}

	results.PassRate = float64(results.PassedSamples) / float64(results.TotalSamples)

	// Update run — capture log values under the lock to avoid races with
	// concurrent writers that may modify run.Results after Unlock.
	e.mu.Lock()
	now := time.Now()
	run.Status = EvaluationStatusCompleted
	run.EndTime = &now
	run.Results = results
	logRunID := run.ID
	logPassRate := results.PassRate
	logSamples := results.TotalSamples
	e.mu.Unlock()

	// Check for regressions and trigger alerts
	e.checkForRegressions(ctx, run)

	e.logger.WithFields(logrus.Fields{
		"id":        logRunID,
		"pass_rate": logPassRate,
		"samples":   logSamples,
	}).Info("Evaluation run completed")
}

func (e *InMemoryContinuousEvaluator) evaluateSample(ctx context.Context, run *EvaluationRun, sample *DatasetSample, promptTemplate string) *SampleResult {
	start := time.Now()

	result := &SampleResult{
		ID:     sample.ID,
		Input:  sample.Input,
		Scores: make(map[string]float64),
	}

	// In a real implementation, this would:
	// 1. Render the prompt with the sample input
	// 2. Call the LLM with the prompt
	// 3. Evaluate the response

	if sample.ExpectedOutput != "" {
		result.Expected = sample.ExpectedOutput
	}

	// Use LLM evaluator if available
	if e.evaluator != nil {
		// Simulated response for evaluation
		response := "simulated response"
		result.Actual = response

		scores, err := e.evaluator.Evaluate(ctx, sample.Input, response, sample.ExpectedOutput, run.Metrics)
		if err != nil {
			result.Error = err.Error()
			result.Passed = false
		} else {
			result.Scores = scores

			// Check if all metrics pass threshold (0.7)
			result.Passed = true
			for _, score := range scores {
				if score < 0.7 {
					result.Passed = false
					break
				}
			}
		}
	} else {
		// Simple heuristic evaluation
		result.Actual = "evaluated"
		result.Passed = true
		for _, metric := range run.Metrics {
			result.Scores[metric] = 0.8 // Placeholder
		}
	}

	result.Latency = time.Since(start)

	return result
}

func (e *InMemoryContinuousEvaluator) checkForRegressions(ctx context.Context, run *EvaluationRun) {
	if e.alertManager == nil {
		return
	}

	// Find previous run with same prompt/model
	var previousRun *EvaluationRun
	e.mu.RLock()
	for _, r := range e.runs {
		if r.ID != run.ID &&
			r.PromptName == run.PromptName &&
			r.ModelName == run.ModelName &&
			r.Status == EvaluationStatusCompleted &&
			r.Results != nil {
			if previousRun == nil || r.CreatedAt.After(previousRun.CreatedAt) {
				previousRun = r
			}
		}
	}
	// While still holding the RLock, take snapshots of the result values
	// we need for regression comparison to avoid races with concurrent writers.
	var (
		runPassRate      float64
		runMetricScores  map[string]float64
		prevPassRate     float64
		prevMetricScores map[string]float64
		prevRunID        string
	)
	if previousRun != nil && previousRun.Results != nil && run.Results != nil {
		runPassRate = run.Results.PassRate
		runMetricScores = make(map[string]float64, len(run.Results.MetricScores))
		for k, v := range run.Results.MetricScores {
			runMetricScores[k] = v
		}
		prevPassRate = previousRun.Results.PassRate
		prevMetricScores = make(map[string]float64, len(previousRun.Results.MetricScores))
		for k, v := range previousRun.Results.MetricScores {
			prevMetricScores[k] = v
		}
		prevRunID = previousRun.ID
	}
	e.mu.RUnlock()

	if previousRun == nil || prevRunID == "" {
		return
	}

	// Check for regressions using snapshots (no lock held, safe to use)
	passRateChange := runPassRate - prevPassRate
	if passRateChange < -0.05 { // 5% regression
		alert := &Alert{
			ID:          uuid.New().String(),
			Type:        AlertTypeRegression,
			Severity:    AlertSeverityWarning,
			Message:     fmt.Sprintf("Pass rate regression: %.1f%% -> %.1f%%", prevPassRate*100, runPassRate*100),
			Source:      "evaluation",
			SourceID:    run.ID,
			Threshold:   -0.05,
			ActualValue: passRateChange,
			CreatedAt:   time.Now(),
		}

		if passRateChange < -0.10 {
			alert.Severity = AlertSeverityCritical
		}

		_ = e.alertManager.Create(ctx, alert)
	}

	// Check individual metrics using snapshots
	for metric, score := range runMetricScores {
		if prevScore, ok := prevMetricScores[metric]; ok {
			change := score - prevScore
			if change < -0.1 {
				alert := &Alert{
					ID:          uuid.New().String(),
					Type:        AlertTypeRegression,
					Severity:    AlertSeverityWarning,
					Message:     fmt.Sprintf("Metric %s regression: %.2f -> %.2f", metric, prevScore, score),
					Source:      "evaluation",
					SourceID:    run.ID,
					Metric:      metric,
					Threshold:   -0.1,
					ActualValue: change,
					CreatedAt:   time.Now(),
				}
				_ = e.alertManager.Create(ctx, alert)
			}
		}
	}
}

// GetRun gets evaluation run status
func (e *InMemoryContinuousEvaluator) GetRun(ctx context.Context, runID string) (*EvaluationRun, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	run, ok := e.runs[runID]
	if !ok {
		return nil, fmt.Errorf("run not found: %s", runID)
	}

	// Return a shallow copy so callers do not race with concurrent writers on
	// scalar fields (Status, StartTime, EndTime) after the lock is released.
	cp := *run
	return &cp, nil
}

// ListRuns lists evaluation runs
func (e *InMemoryContinuousEvaluator) ListRuns(ctx context.Context, filter *EvaluationFilter) ([]*EvaluationRun, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	var result []*EvaluationRun
	for _, run := range e.runs {
		if e.matchesFilter(run, filter) {
			// Return a shallow copy to avoid races after the lock is released.
			cp := *run
			result = append(result, &cp)
		}
	}

	// Sort by creation time (newest first)
	sort.Slice(result, func(i, j int) bool {
		return result[i].CreatedAt.After(result[j].CreatedAt)
	})

	// Apply limit
	if filter != nil && filter.Limit > 0 && filter.Limit < len(result) {
		result = result[:filter.Limit]
	}

	return result, nil
}

func (e *InMemoryContinuousEvaluator) matchesFilter(run *EvaluationRun, filter *EvaluationFilter) bool {
	if filter == nil {
		return true
	}

	if filter.PromptName != "" && run.PromptName != filter.PromptName {
		return false
	}
	if filter.ModelName != "" && run.ModelName != filter.ModelName {
		return false
	}
	if filter.Status != "" && run.Status != filter.Status {
		return false
	}
	if filter.StartTime != nil && run.CreatedAt.Before(*filter.StartTime) {
		return false
	}
	if filter.EndTime != nil && run.CreatedAt.After(*filter.EndTime) {
		return false
	}

	return true
}

// ScheduleRun schedules a recurring evaluation
func (e *InMemoryContinuousEvaluator) ScheduleRun(ctx context.Context, run *EvaluationRun, scheduleExpr string) error {
	// Create initial run
	if err := e.CreateRun(ctx, run); err != nil {
		return err
	}

	e.mu.Lock()
	e.schedules[run.ID] = &schedule{
		run:    run,
		cron:   scheduleExpr,
		stopCh: make(chan struct{}),
	}
	e.mu.Unlock()

	// Start scheduler (simplified - in production use proper cron library)
	go e.runScheduler(run.ID)

	return nil
}

func (e *InMemoryContinuousEvaluator) runScheduler(runID string) {
	e.mu.RLock()
	sched, ok := e.schedules[runID]
	e.mu.RUnlock()

	if !ok {
		return
	}

	// Simplified: run every hour
	ticker := time.NewTicker(time.Hour)
	defer ticker.Stop()

	for {
		select {
		case <-sched.stopCh:
			return
		case <-ticker.C:
			// Create new run based on template
			newRun := &EvaluationRun{
				Name:          sched.run.Name,
				Dataset:       sched.run.Dataset,
				PromptName:    sched.run.PromptName,
				PromptVersion: sched.run.PromptVersion,
				ModelName:     sched.run.ModelName,
				Metrics:       sched.run.Metrics,
			}

			ctx := context.Background()
			if err := e.CreateRun(ctx, newRun); err != nil {
				e.logger.WithError(err).Warn("Failed to create scheduled run")
				continue
			}

			if err := e.StartRun(ctx, newRun.ID); err != nil {
				e.logger.WithError(err).Warn("Failed to start scheduled run")
			}

			sched.lastRun = time.Now()
		}
	}
}

// CompareRuns compares two evaluation runs
func (e *InMemoryContinuousEvaluator) CompareRuns(ctx context.Context, runID1, runID2 string) (*RunComparison, error) {
	e.mu.RLock()
	run1, ok1 := e.runs[runID1]
	run2, ok2 := e.runs[runID2]
	e.mu.RUnlock()

	if !ok1 {
		return nil, fmt.Errorf("run not found: %s", runID1)
	}
	if !ok2 {
		return nil, fmt.Errorf("run not found: %s", runID2)
	}

	if run1.Results == nil || run2.Results == nil {
		return nil, fmt.Errorf("both runs must be completed")
	}

	comparison := &RunComparison{
		Run1ID:         runID1,
		Run2ID:         runID2,
		MetricChanges:  make(map[string]float64),
		PassRateChange: run2.Results.PassRate - run1.Results.PassRate,
	}

	// Calculate metric changes
	for metric, score2 := range run2.Results.MetricScores {
		if score1, ok := run1.Results.MetricScores[metric]; ok {
			change := ((score2 - score1) / score1) * 100
			comparison.MetricChanges[metric] = change

			if change < -5 {
				comparison.Regressions = append(comparison.Regressions, metric)
			} else if change > 5 {
				comparison.Improvements = append(comparison.Improvements, metric)
			}
		}
	}

	// Generate summary
	if len(comparison.Regressions) > 0 {
		comparison.Summary = fmt.Sprintf("Regressions in %d metrics", len(comparison.Regressions))
	} else if len(comparison.Improvements) > 0 {
		comparison.Summary = fmt.Sprintf("Improvements in %d metrics", len(comparison.Improvements))
	} else {
		comparison.Summary = "No significant changes"
	}

	return comparison, nil
}

// Dataset management methods

// CreateDataset creates a new dataset
func (e *InMemoryContinuousEvaluator) CreateDataset(ctx context.Context, dataset *Dataset) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if dataset.Name == "" {
		return fmt.Errorf("dataset name is required")
	}

	if dataset.ID == "" {
		dataset.ID = uuid.New().String()
	}

	dataset.CreatedAt = time.Now()
	dataset.UpdatedAt = time.Now()

	e.datasets[dataset.ID] = dataset
	e.samples[dataset.ID] = make([]*DatasetSample, 0)

	return nil
}

// GetDataset retrieves a dataset
func (e *InMemoryContinuousEvaluator) GetDataset(ctx context.Context, id string) (*Dataset, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	dataset, ok := e.datasets[id]
	if !ok {
		return nil, fmt.Errorf("dataset not found: %s", id)
	}

	return dataset, nil
}

// AddSamples adds samples to a dataset
func (e *InMemoryContinuousEvaluator) AddSamples(ctx context.Context, datasetID string, samples []*DatasetSample) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	dataset, ok := e.datasets[datasetID]
	if !ok {
		return fmt.Errorf("dataset not found: %s", datasetID)
	}

	for _, sample := range samples {
		if sample.ID == "" {
			sample.ID = uuid.New().String()
		}
		e.samples[datasetID] = append(e.samples[datasetID], sample)
	}

	dataset.SampleCount = len(e.samples[datasetID])
	dataset.UpdatedAt = time.Now()

	return nil
}
