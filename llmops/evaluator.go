package llmops

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/sirupsen/logrus"

	"digital.vasic.llmops/pkg/i18n"
)

// LLMEvaluator interface for LLM-based evaluation
type LLMEvaluator interface {
	Evaluate(ctx context.Context, prompt, response, expected string, metrics []string) (map[string]float64, error)
}

// LLMResponder is the contract InMemoryContinuousEvaluator uses to obtain a
// real LLM-generated response for a dataset sample BEFORE that response is
// passed to the LLMEvaluator for scoring. Callers MUST wire a real
// implementation via SetResponder (or an equivalent dependency-injection
// path) before invoking StartRun — otherwise evaluateSample returns
// ErrLLMResponderNotConfigured rather than the previous simulated string.
//
// Returning a non-nil error from Generate causes the sample to be marked
// FAILED with the error captured in SampleResult.Error; the scoring step
// is skipped for that sample.
type LLMResponder interface {
	Generate(ctx context.Context, prompt string) (string, error)
}

// LLMResponderFunc adapts a plain function to LLMResponder for ergonomic
// injection in tests and integration wiring.
type LLMResponderFunc func(ctx context.Context, prompt string) (string, error)

// Generate satisfies LLMResponder for LLMResponderFunc.
func (f LLMResponderFunc) Generate(ctx context.Context, prompt string) (string, error) {
	return f(ctx, prompt)
}

// ErrLLMResponderNotConfigured is returned by evaluateSample when no real
// LLMResponder has been wired into the evaluator. The previous default
// fabricated a literal "simulated response" string and fed it to the
// LLMEvaluator, producing meaningless scores that nevertheless surfaced
// as PASS results — §11.4 PASS-bluff at the production-default layer
// (CRITICAL per CONST-035 / Article XI §11.9, audit round 25, 2026-05-17).
// Wire a real LLM provider via (*InMemoryContinuousEvaluator).SetResponder
// before invoking StartRun.
var ErrLLMResponderNotConfigured = errors.New(
	"llmops: LLM responder has not been wired into the evaluator — call (*InMemoryContinuousEvaluator).SetResponder with a real LLM-dispatching responder before invoking StartRun (the previous default fabricated a 'simulated response' string and evaluated against it; §11.4 PASS-bluff removed)",
)

// ErrEvaluatorNotConfigured is returned by evaluateSample when no
// LLMEvaluator has been wired into the InMemoryContinuousEvaluator.
// The previous default fabricated a PASS result with a 0.8 placeholder
// score for every requested metric, which surfaced as a successful
// evaluation run despite zero real scoring having occurred — §11.4
// PASS-bluff at the production-default layer (CRITICAL per CONST-035 /
// Article XI §11.9, audit round 25, 2026-05-17). Pass a non-nil
// LLMEvaluator to NewInMemoryContinuousEvaluator (or to NewLLMOpsSystem
// via SetDebateEvaluator) before invoking StartRun.
var ErrEvaluatorNotConfigured = errors.New(
	"llmops: LLM evaluator has not been wired into the InMemoryContinuousEvaluator — construct it with a non-nil LLMEvaluator (or call SetDebateEvaluator on LLMOpsSystem) before invoking StartRun (the previous default fabricated PASS with a 0.8 placeholder score per metric; §11.4 PASS-bluff removed)",
)

// InMemoryContinuousEvaluator implements ContinuousEvaluator
type InMemoryContinuousEvaluator struct {
	runs         map[string]*EvaluationRun
	datasets     map[string]*Dataset
	samples      map[string][]*DatasetSample // dataset ID -> samples
	schedules    map[string]*schedule
	evaluator    LLMEvaluator
	responder    LLMResponder
	registry     PromptRegistry
	mu           sync.RWMutex
	logger       *logrus.Logger
	alertManager AlertManager
	// translator resolves user-facing message keys (CONST-046). Defaults
	// to NoopTranslator (key-verbatim passthrough) so legacy assertions
	// keep working until a consuming project wires a real translator via
	// SetTranslator. Per CONST-051(B) this is decoupled injection, not a
	// parent-tree reach.
	translator i18n.Translator
}

type schedule struct {
	run     *EvaluationRun
	cron    string
	lastRun time.Time
	stopCh  chan struct{}
}

// NewInMemoryContinuousEvaluator creates a new continuous evaluator.
//
// Callers that pass a non-nil LLMEvaluator MUST also wire a real
// LLMResponder via SetResponder before invoking StartRun. Without a
// wired responder, evaluateSample returns ErrLLMResponderNotConfigured
// (previously it fabricated the literal string "simulated response" —
// §11.4 PASS-bluff removed, audit round 25, 2026-05-17).
//
// Callers that pass a nil LLMEvaluator will receive
// ErrEvaluatorNotConfigured from evaluateSample for every sample
// (previously a fake PASS with placeholder 0.8 score per metric was
// fabricated — §11.4 PASS-bluff removed in the same audit round).
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
		translator:   i18n.NoopTranslator{},
	}
}

// SetTranslator wires the i18n translator used to render user-facing
// run-comparison summary text. Passing a nil translator is a no-op (the
// evaluator continues to use the NoopTranslator key-verbatim default
// installed at construction). Safe to call concurrently with other
// methods — guarded by the same RW mutex that protects runs / datasets.
//
// Per CONST-051(B), this is configuration injection — the LLMOps
// submodule never reaches into a parent project to discover its
// catalogue; the consuming project hands it in.
func (e *InMemoryContinuousEvaluator) SetTranslator(t i18n.Translator) {
	if t == nil {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	e.translator = t
}

// tr returns the active translator, defaulting to NoopTranslator. Read
// without locking is safe because SetTranslator only ever stores a
// non-nil value and the field is assigned at construction.
func (e *InMemoryContinuousEvaluator) tr() i18n.Translator {
	if e.translator == nil {
		return i18n.NoopTranslator{}
	}
	return e.translator
}

// renderSummary resolves a user-facing summary message through the
// injected i18n.Translator. NoopTranslator returns the key verbatim;
// when that happens we substitute the legacy English fallback so
// long-standing string assertions against Summary (e.g. "No significant
// changes") keep passing. A real translator wired by the consuming
// project supplies the localised rendering and short-circuits the
// fallback. Per CONST-046, no human-readable static literal is exposed
// without going through this seam.
func (e *InMemoryContinuousEvaluator) renderSummary(key string, params map[string]any, englishFallback string) string {
	rendered := e.tr().T(key, params)
	if rendered == key {
		return englishFallback
	}
	return rendered
}

// SetResponder wires the real LLMResponder used by evaluateSample to
// produce a model response for each dataset sample. Passing a nil
// responder is a no-op (the evaluator continues to return
// ErrLLMResponderNotConfigured from evaluateSample when the LLMEvaluator
// is non-nil). Safe to call concurrently with other methods — guarded
// by the same RW mutex that protects runs / datasets.
func (e *InMemoryContinuousEvaluator) SetResponder(r LLMResponder) {
	if r == nil {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	e.responder = r
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

	if sample.ExpectedOutput != "" {
		result.Expected = sample.ExpectedOutput
	}

	// Snapshot the injected dependencies under the lock so concurrent
	// SetResponder / constructor mutations cannot race with the read.
	e.mu.RLock()
	evaluator := e.evaluator
	responder := e.responder
	e.mu.RUnlock()

	// CONST-035 / Article XI §11.9 anti-bluff: without a real LLMEvaluator
	// wired in, the function MUST NOT fabricate a PASS — previously it
	// returned Passed=true with a 0.8 placeholder score per metric, which
	// surfaced as a meaningful evaluation result despite zero real
	// scoring having occurred. Round-25 §11.4 audit (2026-05-17) removes
	// that bluff: the sample is marked FAILED with ErrEvaluatorNotConfigured
	// so the absence of coverage is loud rather than silent.
	if evaluator == nil {
		result.Error = ErrEvaluatorNotConfigured.Error()
		result.Passed = false
		result.Latency = time.Since(start)
		return result
	}

	// CONST-035 / Article XI §11.9 anti-bluff: without a real LLMResponder
	// wired in, the function MUST NOT fabricate the literal "simulated
	// response" string and feed it to the LLMEvaluator — that produced
	// meaningless scores presented as if they had assessed a real model
	// response. Round-25 §11.4 audit (2026-05-17) removes that bluff:
	// the sample is marked FAILED with ErrLLMResponderNotConfigured.
	// Render the prompt template with the sample input when available;
	// otherwise the raw sample input is what the LLM sees.
	if responder == nil {
		result.Error = ErrLLMResponderNotConfigured.Error()
		result.Passed = false
		result.Latency = time.Since(start)
		return result
	}

	// Compose the LLM input: prefer the rendered prompt template when the
	// run was configured with one, otherwise pass the raw sample input.
	llmPrompt := sample.Input
	if promptTemplate != "" {
		llmPrompt = promptTemplate + "\n\n" + sample.Input
	}

	response, err := responder.Generate(ctx, llmPrompt)
	if err != nil {
		result.Error = fmt.Errorf("llmops: LLM responder Generate failed: %w", err).Error()
		result.Passed = false
		result.Latency = time.Since(start)
		return result
	}
	result.Actual = response

	scores, err := evaluator.Evaluate(ctx, sample.Input, response, sample.ExpectedOutput, run.Metrics)
	if err != nil {
		result.Error = err.Error()
		result.Passed = false
		result.Latency = time.Since(start)
		return result
	}
	result.Scores = scores

	// All requested metrics must clear the 0.7 floor for the sample to PASS.
	result.Passed = true
	for _, score := range scores {
		if score < 0.7 {
			result.Passed = false
			break
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

	// Generate summary — CONST-046: text resolved via the injected
	// i18n.Translator. NoopTranslator (default) returns the key verbatim;
	// renderSummaryKey then falls back to the legacy English literal so
	// existing-string assertions keep passing. A real translator wired by
	// the consuming project supplies the locale-correct rendering.
	switch {
	case len(comparison.Regressions) > 0:
		comparison.Summary = e.renderSummary(
			"llmops_run_comparison_regressions",
			map[string]any{"count": len(comparison.Regressions)},
			fmt.Sprintf("Regressions in %d metrics", len(comparison.Regressions)),
		)
	case len(comparison.Improvements) > 0:
		comparison.Summary = e.renderSummary(
			"llmops_run_comparison_improvements",
			map[string]any{"count": len(comparison.Improvements)},
			fmt.Sprintf("Improvements in %d metrics", len(comparison.Improvements)),
		)
	default:
		comparison.Summary = e.renderSummary(
			"llmops_run_comparison_no_change",
			nil,
			"No significant changes",
		)
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
