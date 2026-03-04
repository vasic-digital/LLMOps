package llmops

import (
	"context"
	"time"
)

// PromptVersion represents a versioned prompt template
type PromptVersion struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Version     string                 `json:"version"` // Semantic version
	Content     string                 `json:"content"`
	Variables   []PromptVariable       `json:"variables,omitempty"`
	Tags        []string               `json:"tags,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
	Author      string                 `json:"author,omitempty"`
	Description string                 `json:"description,omitempty"`
	IsActive    bool                   `json:"is_active"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
}

// PromptVariable represents a variable in a prompt template
type PromptVariable struct {
	Name        string      `json:"name"`
	Type        string      `json:"type"` // string, int, float, bool, array
	Required    bool        `json:"required"`
	Default     interface{} `json:"default,omitempty"`
	Description string      `json:"description,omitempty"`
	Validation  string      `json:"validation,omitempty"` // Regex or validation rule
}

// PromptRegistry manages prompt versions
type PromptRegistry interface {
	// Create creates a new prompt version
	Create(ctx context.Context, prompt *PromptVersion) error

	// Get retrieves a specific version
	Get(ctx context.Context, name, version string) (*PromptVersion, error)

	// GetLatest retrieves the latest active version
	GetLatest(ctx context.Context, name string) (*PromptVersion, error)

	// List lists all versions of a prompt
	List(ctx context.Context, name string) ([]*PromptVersion, error)

	// ListAll lists all prompts
	ListAll(ctx context.Context) ([]*PromptVersion, error)

	// Activate sets a version as active
	Activate(ctx context.Context, name, version string) error

	// Delete removes a prompt version
	Delete(ctx context.Context, name, version string) error

	// Render renders a prompt with variables
	Render(ctx context.Context, name, version string, vars map[string]interface{}) (string, error)
}

// Experiment represents an A/B test experiment
type Experiment struct {
	ID           string                 `json:"id"`
	Name         string                 `json:"name"`
	Description  string                 `json:"description,omitempty"`
	Variants     []*Variant             `json:"variants"`
	TrafficSplit map[string]float64     `json:"traffic_split"` // Variant ID -> percentage
	Status       ExperimentStatus       `json:"status"`
	Metrics      []string               `json:"metrics"`       // Metrics to track
	TargetMetric string                 `json:"target_metric"` // Primary metric for decisions
	StartTime    *time.Time             `json:"start_time,omitempty"`
	EndTime      *time.Time             `json:"end_time,omitempty"`
	Winner       string                 `json:"winner,omitempty"` // Winning variant ID
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt    time.Time              `json:"created_at"`
	UpdatedAt    time.Time              `json:"updated_at"`
}

// ExperimentStatus represents experiment lifecycle
type ExperimentStatus string

const (
	ExperimentStatusDraft     ExperimentStatus = "draft"
	ExperimentStatusRunning   ExperimentStatus = "running"
	ExperimentStatusPaused    ExperimentStatus = "paused"
	ExperimentStatusCompleted ExperimentStatus = "completed"
	ExperimentStatusCancelled ExperimentStatus = "cancelled"
)

// Variant represents a variant in an experiment
type Variant struct {
	ID            string                 `json:"id"`
	Name          string                 `json:"name"`
	Description   string                 `json:"description,omitempty"`
	PromptName    string                 `json:"prompt_name,omitempty"`
	PromptVersion string                 `json:"prompt_version,omitempty"`
	ModelName     string                 `json:"model_name,omitempty"`
	Parameters    map[string]interface{} `json:"parameters,omitempty"` // Temperature, etc.
	IsControl     bool                   `json:"is_control"`
}

// ExperimentResult represents experiment analytics
type ExperimentResult struct {
	ExperimentID   string                    `json:"experiment_id"`
	VariantResults map[string]*VariantResult `json:"variant_results"`
	TotalSamples   int                       `json:"total_samples"`
	StartTime      time.Time                 `json:"start_time"`
	EndTime        time.Time                 `json:"end_time"`
	Winner         string                    `json:"winner,omitempty"`
	Significance   float64                   `json:"significance"` // Statistical significance
	Confidence     float64                   `json:"confidence"`   // Confidence level
	Recommendation string                    `json:"recommendation"`
}

// VariantResult represents results for a single variant
type VariantResult struct {
	VariantID      string                  `json:"variant_id"`
	SampleCount    int                     `json:"sample_count"`
	MetricValues   map[string]*MetricValue `json:"metric_values"`
	ConversionRate float64                 `json:"conversion_rate,omitempty"`
	Improvement    float64                 `json:"improvement,omitempty"` // vs control
}

// MetricValue represents a metric measurement
type MetricValue struct {
	Name   string  `json:"name"`
	Value  float64 `json:"value"`
	StdDev float64 `json:"std_dev"`
	Min    float64 `json:"min"`
	Max    float64 `json:"max"`
	Count  int     `json:"count"`
}

// ExperimentManager manages A/B testing
type ExperimentManager interface {
	// Create creates a new experiment
	Create(ctx context.Context, exp *Experiment) error

	// Get retrieves an experiment
	Get(ctx context.Context, id string) (*Experiment, error)

	// List lists all experiments
	List(ctx context.Context, status ExperimentStatus) ([]*Experiment, error)

	// Start starts an experiment
	Start(ctx context.Context, id string) error

	// Pause pauses an experiment
	Pause(ctx context.Context, id string) error

	// Complete completes an experiment
	Complete(ctx context.Context, id string, winner string) error

	// Cancel cancels an experiment
	Cancel(ctx context.Context, id string) error

	// AssignVariant assigns a variant for a request
	AssignVariant(ctx context.Context, experimentID, userID string) (*Variant, error)

	// RecordMetric records a metric for a variant
	RecordMetric(ctx context.Context, experimentID, variantID, metric string, value float64) error

	// GetResults gets experiment results
	GetResults(ctx context.Context, experimentID string) (*ExperimentResult, error)
}

// EvaluationRun represents a continuous evaluation run
type EvaluationRun struct {
	ID            string                 `json:"id"`
	Name          string                 `json:"name"`
	Description   string                 `json:"description,omitempty"`
	Dataset       string                 `json:"dataset"` // Dataset identifier
	PromptName    string                 `json:"prompt_name"`
	PromptVersion string                 `json:"prompt_version"`
	ModelName     string                 `json:"model_name"`
	Metrics       []string               `json:"metrics"`
	Status        EvaluationStatus       `json:"status"`
	Results       *EvaluationResults     `json:"results,omitempty"`
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
	StartTime     *time.Time             `json:"start_time,omitempty"`
	EndTime       *time.Time             `json:"end_time,omitempty"`
	CreatedAt     time.Time              `json:"created_at"`
}

// EvaluationStatus represents evaluation lifecycle
type EvaluationStatus string

const (
	EvaluationStatusPending   EvaluationStatus = "pending"
	EvaluationStatusRunning   EvaluationStatus = "running"
	EvaluationStatusCompleted EvaluationStatus = "completed"
	EvaluationStatusFailed    EvaluationStatus = "failed"
)

// EvaluationResults represents evaluation results
type EvaluationResults struct {
	TotalSamples   int                     `json:"total_samples"`
	PassedSamples  int                     `json:"passed_samples"`
	FailedSamples  int                     `json:"failed_samples"`
	PassRate       float64                 `json:"pass_rate"`
	MetricScores   map[string]float64      `json:"metric_scores"`
	MetricDetails  map[string]*MetricValue `json:"metric_details,omitempty"`
	FailureReasons map[string]int          `json:"failure_reasons,omitempty"`
	SampleResults  []*SampleResult         `json:"sample_results,omitempty"`
}

// SampleResult represents result for a single evaluation sample
type SampleResult struct {
	ID       string             `json:"id"`
	Input    string             `json:"input"`
	Expected string             `json:"expected,omitempty"`
	Actual   string             `json:"actual"`
	Passed   bool               `json:"passed"`
	Scores   map[string]float64 `json:"scores"`
	Latency  time.Duration      `json:"latency"`
	Error    string             `json:"error,omitempty"`
}

// ContinuousEvaluator manages continuous evaluation
type ContinuousEvaluator interface {
	// CreateRun creates a new evaluation run
	CreateRun(ctx context.Context, run *EvaluationRun) error

	// StartRun starts an evaluation run
	StartRun(ctx context.Context, runID string) error

	// GetRun gets evaluation run status
	GetRun(ctx context.Context, runID string) (*EvaluationRun, error)

	// ListRuns lists evaluation runs
	ListRuns(ctx context.Context, filter *EvaluationFilter) ([]*EvaluationRun, error)

	// ScheduleRun schedules a recurring evaluation
	ScheduleRun(ctx context.Context, run *EvaluationRun, schedule string) error

	// CompareRuns compares two evaluation runs
	CompareRuns(ctx context.Context, runID1, runID2 string) (*RunComparison, error)
}

// EvaluationFilter for filtering evaluation runs
type EvaluationFilter struct {
	PromptName string           `json:"prompt_name,omitempty"`
	ModelName  string           `json:"model_name,omitempty"`
	Status     EvaluationStatus `json:"status,omitempty"`
	StartTime  *time.Time       `json:"start_time,omitempty"`
	EndTime    *time.Time       `json:"end_time,omitempty"`
	Limit      int              `json:"limit,omitempty"`
}

// RunComparison represents comparison between two runs
type RunComparison struct {
	Run1ID         string             `json:"run1_id"`
	Run2ID         string             `json:"run2_id"`
	MetricChanges  map[string]float64 `json:"metric_changes"` // Percentage change
	PassRateChange float64            `json:"pass_rate_change"`
	Regressions    []string           `json:"regressions,omitempty"`
	Improvements   []string           `json:"improvements,omitempty"`
	Summary        string             `json:"summary"`
}

// DatasetManager manages evaluation datasets
type DatasetManager interface {
	// Create creates a new dataset
	Create(ctx context.Context, dataset *Dataset) error

	// Get retrieves a dataset
	Get(ctx context.Context, id string) (*Dataset, error)

	// List lists datasets
	List(ctx context.Context) ([]*Dataset, error)

	// AddSamples adds samples to a dataset
	AddSamples(ctx context.Context, datasetID string, samples []*DatasetSample) error

	// GetSamples retrieves samples from a dataset
	GetSamples(ctx context.Context, datasetID string, limit, offset int) ([]*DatasetSample, error)
}

// Dataset represents an evaluation dataset
type Dataset struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Type        DatasetType            `json:"type"`
	SampleCount int                    `json:"sample_count"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
}

// DatasetType represents dataset types
type DatasetType string

const (
	DatasetTypeGolden     DatasetType = "golden"     // Golden test set
	DatasetTypeRegression DatasetType = "regression" // Regression tests
	DatasetTypeBenchmark  DatasetType = "benchmark"  // Benchmark set
	DatasetTypeUser       DatasetType = "user"       // User-generated
)

// DatasetSample represents a sample in a dataset
type DatasetSample struct {
	ID             string                 `json:"id"`
	Input          string                 `json:"input"`
	ExpectedOutput string                 `json:"expected_output,omitempty"`
	Context        string                 `json:"context,omitempty"`
	Metadata       map[string]interface{} `json:"metadata,omitempty"`
}

// Alert represents an evaluation alert
type Alert struct {
	ID          string                 `json:"id"`
	Type        AlertType              `json:"type"`
	Severity    AlertSeverity          `json:"severity"`
	Message     string                 `json:"message"`
	Source      string                 `json:"source"` // evaluation run, experiment, etc.
	SourceID    string                 `json:"source_id"`
	Metric      string                 `json:"metric,omitempty"`
	Threshold   float64                `json:"threshold,omitempty"`
	ActualValue float64                `json:"actual_value,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt   time.Time              `json:"created_at"`
	AckedAt     *time.Time             `json:"acked_at,omitempty"`
}

// AlertType represents alert types
type AlertType string

const (
	AlertTypeRegression AlertType = "regression" // Performance regression
	AlertTypeThreshold  AlertType = "threshold"  // Threshold breach
	AlertTypeAnomaly    AlertType = "anomaly"    // Anomaly detected
	AlertTypeExperiment AlertType = "experiment" // Experiment result
)

// AlertSeverity represents alert severity
type AlertSeverity string

const (
	AlertSeverityInfo     AlertSeverity = "info"
	AlertSeverityWarning  AlertSeverity = "warning"
	AlertSeverityCritical AlertSeverity = "critical"
)

// AlertManager manages alerts
type AlertManager interface {
	// Create creates a new alert
	Create(ctx context.Context, alert *Alert) error

	// List lists alerts
	List(ctx context.Context, filter *AlertFilter) ([]*Alert, error)

	// Acknowledge acknowledges an alert
	Acknowledge(ctx context.Context, alertID string) error

	// Subscribe subscribes to alerts
	Subscribe(ctx context.Context, callback AlertCallback) error
}

// AlertFilter for filtering alerts
type AlertFilter struct {
	Types      []AlertType     `json:"types,omitempty"`
	Severities []AlertSeverity `json:"severities,omitempty"`
	Source     string          `json:"source,omitempty"`
	Unacked    bool            `json:"unacked,omitempty"`
	StartTime  *time.Time      `json:"start_time,omitempty"`
	Limit      int             `json:"limit,omitempty"`
}

// AlertCallback is called when an alert is triggered
type AlertCallback func(alert *Alert) error
