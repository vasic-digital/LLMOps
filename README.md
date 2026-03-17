# LLMOps

`digital.vasic.llmops` -- LLM operations with continuous evaluation pipelines, A/B experiment management, dataset management, prompt versioning, and alerting.

## Overview

LLMOps is a Go module that provides a complete MLOps/LLMOps toolkit for managing the lifecycle of LLM-powered applications. It covers four key operational areas: continuous evaluation of model quality against datasets, A/B experimentation for comparing prompts and models, prompt template versioning with variable rendering, and alerting for performance regressions.

The continuous evaluator runs evaluation datasets against specified prompt/model combinations, tracks pass rates and per-metric scores, automatically detects regressions against previous runs, and triggers alerts when thresholds are breached. Evaluations can be scheduled for recurring execution.

The experiment manager implements full A/B testing with traffic splitting, deterministic user-to-variant assignment (hash-based), metric recording, and statistical significance calculation using z-test approximation. It supports prompt experiments (comparing prompt versions) and model experiments (comparing different LLM providers).

Prompt versioning provides semantic versioning, variable templates with validation, active version tracking, and a diff tool for comparing prompt versions. All components are tied together by the `LLMOpsSystem` orchestrator which wires up debate-based evaluation, verifier integration, and component initialization.

## Architecture

```
+----------------------------+
|       LLMOpsSystem         |
|    (Main Orchestrator)     |
+----+-------+-------+------+
     |       |       |
+----v--+ +--v---+ +-v---------+
|Prompt | |Exper.| |Continuous  |
|Registry| |Mgr  | |Evaluator  |
+-------+ +--+--+ +-----+-----+
              |          |
         +----v----+ +---v--------+
         | Variant | | Alert      |
         | Assign  | | Manager    |
         | Metrics | | (Regress.) |
         +---------+ +------------+

Integration Adapters:
  - DebateLLMEvaluator    (debate-based scoring)
  - VerifierIntegration   (provider selection)
```

## Package Structure

| Package | Purpose |
|---------|---------|
| `llmops` | Core module: evaluator, experiments, prompts, alerts, integration |

### Source Files

| File | Description |
|------|-------------|
| `types.go` | All type definitions, interfaces, enums: prompts, experiments, evaluations, datasets, alerts |
| `evaluator.go` | `InMemoryContinuousEvaluator` -- evaluation runs, dataset management, regression detection, scheduling |
| `experiments.go` | `InMemoryExperimentManager` -- A/B testing, traffic splitting, variant assignment, statistical significance |
| `prompts.go` | `InMemoryPromptRegistry` -- prompt versioning, variable rendering, version comparison/diffing |
| `integration.go` | `LLMOpsSystem` orchestrator, alert manager, debate adapter, verifier integration |

## API Reference

### Core Interfaces

```go
// PromptRegistry manages prompt versions
type PromptRegistry interface {
    Create(ctx context.Context, prompt *PromptVersion) error
    Get(ctx context.Context, name, version string) (*PromptVersion, error)
    GetLatest(ctx context.Context, name string) (*PromptVersion, error)
    List(ctx context.Context, name string) ([]*PromptVersion, error)
    ListAll(ctx context.Context) ([]*PromptVersion, error)
    Activate(ctx context.Context, name, version string) error
    Delete(ctx context.Context, name, version string) error
    Render(ctx context.Context, name, version string, vars map[string]interface{}) (string, error)
}

// ExperimentManager manages A/B testing
type ExperimentManager interface {
    Create(ctx context.Context, exp *Experiment) error
    Get(ctx context.Context, id string) (*Experiment, error)
    List(ctx context.Context, status ExperimentStatus) ([]*Experiment, error)
    Start(ctx context.Context, id string) error
    Pause(ctx context.Context, id string) error
    Complete(ctx context.Context, id string, winner string) error
    Cancel(ctx context.Context, id string) error
    AssignVariant(ctx context.Context, experimentID, userID string) (*Variant, error)
    RecordMetric(ctx context.Context, experimentID, variantID, metric string, value float64) error
    GetResults(ctx context.Context, experimentID string) (*ExperimentResult, error)
}

// ContinuousEvaluator manages evaluation pipelines
type ContinuousEvaluator interface {
    CreateRun(ctx context.Context, run *EvaluationRun) error
    StartRun(ctx context.Context, runID string) error
    GetRun(ctx context.Context, runID string) (*EvaluationRun, error)
    ListRuns(ctx context.Context, filter *EvaluationFilter) ([]*EvaluationRun, error)
    ScheduleRun(ctx context.Context, run *EvaluationRun, schedule string) error
    CompareRuns(ctx context.Context, runID1, runID2 string) (*RunComparison, error)
}

// AlertManager manages performance alerts
type AlertManager interface {
    Create(ctx context.Context, alert *Alert) error
    List(ctx context.Context, filter *AlertFilter) ([]*Alert, error)
    Acknowledge(ctx context.Context, alertID string) error
    Subscribe(ctx context.Context, callback AlertCallback) error
}
```

### LLMOpsSystem Methods

```go
func NewLLMOpsSystem(config *LLMOpsConfig, logger *logrus.Logger) *LLMOpsSystem
func (s *LLMOpsSystem) Initialize() error
func (s *LLMOpsSystem) SetDebateEvaluator(evaluator DebateLLMEvaluator)
func (s *LLMOpsSystem) SetVerifierIntegration(vi *VerifierIntegration)
func (s *LLMOpsSystem) GetPromptRegistry() PromptRegistry
func (s *LLMOpsSystem) GetExperimentManager() ExperimentManager
func (s *LLMOpsSystem) GetEvaluator() ContinuousEvaluator
func (s *LLMOpsSystem) GetAlertManager() AlertManager
func (s *LLMOpsSystem) CreatePromptExperiment(ctx, name, control, treatment, trafficSplit) (*Experiment, error)
func (s *LLMOpsSystem) CreateModelExperiment(ctx, name, models, parameters) (*Experiment, error)
```

## Usage Examples

### Prompt versioning and rendering

```go
registry := llmops.NewInMemoryPromptRegistry(logger)

registry.Create(ctx, &llmops.PromptVersion{
    Name:    "code-review",
    Version: "1.0.0",
    Content: "Review this {{language}} code for {{focus_area}}: {{code}}",
    Variables: []llmops.PromptVariable{
        {Name: "language", Type: "string", Required: true},
        {Name: "focus_area", Type: "string", Required: true, Default: "bugs"},
        {Name: "code", Type: "string", Required: true},
    },
})

rendered, _ := registry.Render(ctx, "code-review", "1.0.0", map[string]interface{}{
    "language":   "Go",
    "focus_area": "performance",
    "code":       "func main() { ... }",
})
```

### A/B experiment for prompts

```go
system := llmops.NewLLMOpsSystem(llmops.DefaultLLMOpsConfig(), logger)
system.Initialize()

exp, _ := system.CreatePromptExperiment(ctx,
    "code-review-v2-test",
    controlPrompt,   // *PromptVersion
    treatmentPrompt,  // *PromptVersion
    0.3,              // 30% traffic to treatment
)

mgr := system.GetExperimentManager()
mgr.Start(ctx, exp.ID)

// During request handling:
variant, _ := mgr.AssignVariant(ctx, exp.ID, userID)
// ... use variant.PromptName/PromptVersion ...
mgr.RecordMetric(ctx, exp.ID, variant.ID, "quality", 0.85)

// Check results
results, _ := mgr.GetResults(ctx, exp.ID)
// results.Confidence, results.Winner, results.Recommendation
```

### Continuous evaluation with regression detection

```go
eval := system.GetEvaluator().(*llmops.InMemoryContinuousEvaluator)

// Create dataset
eval.CreateDataset(ctx, &llmops.Dataset{
    Name: "golden-set",
    Type: llmops.DatasetTypeGolden,
})
eval.AddSamples(ctx, datasetID, samples)

// Create and run evaluation
eval.CreateRun(ctx, &llmops.EvaluationRun{
    Name:       "nightly-eval",
    Dataset:    datasetID,
    PromptName: "code-review",
    ModelName:  "claude-3-sonnet",
    Metrics:    []string{"accuracy", "helpfulness"},
})
eval.StartRun(ctx, runID)
// Automatic regression alerts if pass rate drops > 5%
```

### Alert subscription

```go
alerts := system.GetAlertManager()
alerts.Subscribe(ctx, func(alert *llmops.Alert) error {
    log.Printf("[%s] %s: %s", alert.Severity, alert.Type, alert.Message)
    return nil
})
```

## Configuration

```go
type LLMOpsConfig struct {
    EnableAutoEvaluation   bool               // Enable background evaluation (default: true)
    EvaluationInterval     time.Duration      // Evaluation frequency (default: 24h)
    MinSamplesForSignif    int                // Min samples for statistical significance (default: 100)
    AlertThresholds        map[string]float64 // Metric thresholds for alerts
    EnableDebateEvaluation bool               // Use debate for evaluation (default: true)
}
```

### Experiment Status Lifecycle

`draft` -> `running` -> `paused` -> `running` -> `completed` (with winner)
`draft` -> `running` -> `cancelled`

### Alert Types and Severities

| Type | Description |
|------|-------------|
| `regression` | Performance regression detected |
| `threshold` | Metric threshold breached |
| `anomaly` | Anomaly detected |
| `experiment` | Experiment result notification |

Severities: `info`, `warning`, `critical`

### Dataset Types

| Type | Purpose |
|------|---------|
| `golden` | Curated golden test set |
| `regression` | Regression test cases |
| `benchmark` | Benchmark evaluation set |
| `user` | User-generated examples |

## Testing

```bash
go build ./...
go test ./... -count=1 -race
```

## Integration with HelixAgent

LLMOps connects to HelixAgent through the adapter layer:

- **Debate Evaluation**: The `DebateLLMEvaluator` interface allows HelixAgent's debate service to evaluate model responses during continuous evaluation runs, providing multi-LLM consensus scoring.
- **Verifier Integration**: `VerifierIntegration` uses LLMsVerifier's provider scores and health checks to select the best provider for experiments and evaluations.
- **Prompt Management**: The prompt registry serves as the central store for all system prompts used by HelixAgent, enabling versioned rollouts and A/B testing of prompt changes.
- **Regression Alerts**: Alerts integrate with HelixAgent's notification system (SSE, WebSocket, Webhooks) to surface quality regressions to operators.
