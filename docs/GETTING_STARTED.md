# LLMOps - Getting Started

**Module:** `digital.vasic.llmops`

## Installation

```go
import "digital.vasic.llmops/llmops"
```

## Quick Start: Setting Up Experiments

### 1. Create the Core Components

```go
package main

import (
    "context"
    "fmt"

    "digital.vasic.llmops/llmops"
    "github.com/sirupsen/logrus"
)

func main() {
    logger := logrus.New()

    evaluator := llmops.NewInMemoryContinuousEvaluator(logger)
    experiments := llmops.NewInMemoryExperimentManager(logger)
    prompts := llmops.NewInMemoryPromptRegistry(logger)
```

### 2. Register and Version a Prompt

```go
    prompt := &llmops.PromptVersion{
        Name:        "summarizer",
        Version:     "1.0.0",
        Content:     "Summarize the following text in {{max_words}} words:\n\n{{text}}",
        IsActive:    true,
        Description: "Text summarization prompt",
        Variables: []llmops.PromptVariable{
            {Name: "text", Type: "string", Required: true},
            {Name: "max_words", Type: "int", Required: false, Default: 100},
        },
    }

    err := prompts.Create(context.Background(), prompt)
    if err != nil {
        panic(err)
    }

    // Render the prompt with variables
    rendered, _ := prompts.Render(context.Background(), "summarizer", "1.0.0", map[string]interface{}{
        "text":      "Long document content...",
        "max_words": 50,
    })
    fmt.Println(rendered)
```

### 3. Create an A/B Experiment

```go
    experiment := &llmops.Experiment{
        Name:         "summarizer-model-comparison",
        Description:  "Compare GPT-4 vs Claude for summarization",
        TargetMetric: "quality_score",
        Metrics:      []string{"quality_score", "latency_ms", "cost"},
        Variants: []*llmops.Variant{
            {ID: "control", Name: "GPT-4", Config: map[string]interface{}{"model": "gpt-4"}},
            {ID: "treatment", Name: "Claude", Config: map[string]interface{}{"model": "claude-3"}},
        },
        TrafficSplit: map[string]float64{
            "control":   0.50,
            "treatment": 0.50,
        },
    }

    err = experiments.Create(context.Background(), experiment)
    if err != nil {
        panic(err)
    }

    experiments.Start(context.Background(), experiment.ID)
```

### 4. Track Experiment Metrics

```go
    // Record metric observations for each variant
    experiments.RecordMetric(context.Background(), experiment.ID, "control", "quality_score", 0.85)
    experiments.RecordMetric(context.Background(), experiment.ID, "treatment", "quality_score", 0.91)

    // Check experiment status
    status, _ := experiments.GetStatus(context.Background(), experiment.ID)
    fmt.Printf("Status: %s\n", status.Status)
    fmt.Printf("Winner: %s\n", status.Winner)
}
```

## Continuous Evaluation

Set up evaluation pipelines that run on live traffic:

```go
// Create an evaluation run against a dataset
dataset := &llmops.Dataset{
    ID:   "golden-summarization",
    Name: "Golden Summarization Set",
    Type: llmops.DatasetTypeGolden,
    Entries: []*llmops.DatasetEntry{
        {Input: "Long text...", ExpectedOutput: "Summary..."},
    },
}

run, err := evaluator.Evaluate(ctx, dataset, llmops.EvalConfig{
    ProviderName: "openai",
    ModelName:    "gpt-4",
    Metrics:      []string{"accuracy", "fluency", "relevance"},
})
```

## Prompt Versioning Workflow

| Operation | Method | Description |
|-----------|--------|-------------|
| Create | `Create(ctx, prompt)` | Register a new prompt version |
| Get specific | `Get(ctx, name, version)` | Retrieve a specific version |
| Get latest | `GetLatest(ctx, name)` | Get the latest active version |
| List versions | `List(ctx, name)` | List all versions of a prompt |
| Activate | `Activate(ctx, name, version)` | Set a version as active |
| Render | `Render(ctx, name, version, vars)` | Render prompt with variables |
| Delete | `Delete(ctx, name, version)` | Remove a prompt version |

## Experiment Lifecycle

```
Created --> Started --> Running --> Completed
                          |
                     (RecordMetric)
                          |
                     Winner declared
```

| Status | Description |
|--------|-------------|
| `ExperimentStatusDraft` | Created but not started |
| `ExperimentStatusRunning` | Actively collecting metrics |
| `ExperimentStatusPaused` | Temporarily paused |
| `ExperimentStatusCompleted` | Concluded with winner |
| `ExperimentStatusCancelled` | Cancelled before completion |

## Dataset Types

| Type | Constant | Description |
|------|----------|-------------|
| Golden | `DatasetTypeGolden` | Curated reference examples |
| Synthetic | `DatasetTypeSynthetic` | Generated test data |
| Production | `DatasetTypeProduction` | Sampled from live traffic |

## Integration with HelixAgent

The LLMOps module integrates through:

- **Adapter** at `internal/adapters/llmops/adapter.go`
- Connects to the provider registry for multi-provider evaluation
- Feeds results to the debate system for quality-driven selection

## Next Steps

- See [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- See [API_REFERENCE.md](API_REFERENCE.md) for the full type catalog
