# LLMOps - API Reference

**Module:** `digital.vasic.llmops`
**Package:** `llmops`

## Constructor Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `NewInMemoryContinuousEvaluator` | `NewInMemoryContinuousEvaluator(logger *logrus.Logger) *InMemoryContinuousEvaluator` | Creates an in-memory evaluation pipeline. |
| `NewInMemoryExperimentManager` | `NewInMemoryExperimentManager(logger *logrus.Logger) *InMemoryExperimentManager` | Creates an in-memory experiment manager. |
| `NewInMemoryPromptRegistry` | `NewInMemoryPromptRegistry(logger *logrus.Logger) *InMemoryPromptRegistry` | Creates an in-memory prompt version registry. |

## Interfaces

### ContinuousEvaluator

Manages evaluation pipelines against datasets.

| Method | Signature | Description |
|--------|-----------|-------------|
| `Evaluate` | `Evaluate(ctx context.Context, dataset *Dataset, config EvalConfig) (*EvaluationRun, error)` | Runs evaluation against a dataset. |
| `GetRun` | `GetRun(ctx context.Context, runID string) (*EvaluationRun, error)` | Retrieves a specific evaluation run. |
| `ListRuns` | `ListRuns(ctx context.Context) ([]*EvaluationRun, error)` | Lists all evaluation runs. |

### ExperimentManager

Manages A/B experiments and metric tracking.

| Method | Signature | Description |
|--------|-----------|-------------|
| `Create` | `Create(ctx context.Context, experiment *Experiment) error` | Creates a new experiment. |
| `Start` | `Start(ctx context.Context, experimentID string) error` | Starts an experiment. |
| `Stop` | `Stop(ctx context.Context, experimentID string) error` | Stops a running experiment. |
| `RecordMetric` | `RecordMetric(ctx context.Context, experimentID, variantID, metric string, value float64) error` | Records a metric observation. |
| `GetStatus` | `GetStatus(ctx context.Context, experimentID string) (*ExperimentStatus, error)` | Returns current experiment status. |
| `DeclareWinner` | `DeclareWinner(ctx context.Context, experimentID, variantID string) error` | Declares a winning variant. |

### PromptRegistry

Manages prompt templates with version history.

| Method | Signature | Description |
|--------|-----------|-------------|
| `Create` | `Create(ctx context.Context, prompt *PromptVersion) error` | Creates a new prompt version. |
| `Get` | `Get(ctx context.Context, name, version string) (*PromptVersion, error)` | Retrieves a specific version. |
| `GetLatest` | `GetLatest(ctx context.Context, name string) (*PromptVersion, error)` | Gets the latest active version. |
| `List` | `List(ctx context.Context, name string) ([]*PromptVersion, error)` | Lists all versions of a prompt. |
| `ListAll` | `ListAll(ctx context.Context) ([]*PromptVersion, error)` | Lists all registered prompts. |
| `Activate` | `Activate(ctx context.Context, name, version string) error` | Sets a version as the active one. |
| `Delete` | `Delete(ctx context.Context, name, version string) error` | Removes a prompt version. |
| `Render` | `Render(ctx context.Context, name, version string, vars map[string]interface{}) (string, error)` | Renders a prompt with variable substitution. |

## Core Types

### PromptVersion

```go
type PromptVersion struct {
    ID          string                 `json:"id"`
    Name        string                 `json:"name"`
    Version     string                 `json:"version"`
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
```

### PromptVariable

```go
type PromptVariable struct {
    Name        string      `json:"name"`
    Type        string      `json:"type"`       // string, int, float, bool, array
    Required    bool        `json:"required"`
    Default     interface{} `json:"default,omitempty"`
    Description string      `json:"description,omitempty"`
    Validation  string      `json:"validation,omitempty"`
}
```

### Experiment

```go
type Experiment struct {
    ID           string                 `json:"id"`
    Name         string                 `json:"name"`
    Description  string                 `json:"description,omitempty"`
    Variants     []*Variant             `json:"variants"`
    TrafficSplit map[string]float64     `json:"traffic_split"`
    Status       ExperimentStatus       `json:"status"`
    Metrics      []string               `json:"metrics"`
    TargetMetric string                 `json:"target_metric"`
    StartTime    *time.Time             `json:"start_time,omitempty"`
    EndTime      *time.Time             `json:"end_time,omitempty"`
    Winner       string                 `json:"winner,omitempty"`
    Metadata     map[string]interface{} `json:"metadata,omitempty"`
    CreatedAt    time.Time              `json:"created_at"`
    UpdatedAt    time.Time              `json:"updated_at"`
}
```

### Variant

```go
type Variant struct {
    ID     string                 `json:"id"`
    Name   string                 `json:"name"`
    Config map[string]interface{} `json:"config"`
}
```

### Dataset / DatasetEntry

```go
type Dataset struct {
    ID      string          `json:"id"`
    Name    string          `json:"name"`
    Type    DatasetType     `json:"type"`
    Entries []*DatasetEntry `json:"entries"`
}

type DatasetEntry struct {
    Input          string `json:"input"`
    ExpectedOutput string `json:"expected_output"`
}
```

### EvaluationRun

```go
type EvaluationRun struct {
    ID           string                 `json:"id"`
    DatasetID    string                 `json:"dataset_id"`
    ProviderName string                 `json:"provider_name"`
    ModelName    string                 `json:"model_name"`
    Metrics      map[string]float64     `json:"metrics"`
    Status       string                 `json:"status"`
    StartTime    time.Time              `json:"start_time"`
    EndTime      *time.Time             `json:"end_time,omitempty"`
    Metadata     map[string]interface{} `json:"metadata,omitempty"`
}
```

## Enums

### ExperimentStatus

| Constant | Value |
|----------|-------|
| `ExperimentStatusDraft` | `"draft"` |
| `ExperimentStatusRunning` | `"running"` |
| `ExperimentStatusPaused` | `"paused"` |
| `ExperimentStatusCompleted` | `"completed"` |
| `ExperimentStatusCancelled` | `"cancelled"` |

### DatasetType

| Constant | Value | Description |
|----------|-------|-------------|
| `DatasetTypeGolden` | `"golden"` | Curated reference examples |
| `DatasetTypeSynthetic` | `"synthetic"` | Auto-generated test data |
| `DatasetTypeProduction` | `"production"` | Sampled from live traffic |
