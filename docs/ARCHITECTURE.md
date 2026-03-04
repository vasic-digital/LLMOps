# LLMOps Module Architecture

**Module:** `digital.vasic.llmops`

## Overview

LLM operations platform for continuous evaluation, A/B experiment management, dataset management, and prompt versioning.

## Core Components

### Evaluator (`evaluator.go`)
Continuous evaluation of LLM outputs against quality metrics. Runs evaluation pipelines on live traffic and stored datasets.

### Experiments (`experiments.go`)
A/B experiment management for comparing model configurations, prompts, and providers. Tracks metrics per variant with statistical significance testing.

### Prompts (`prompts.go`)
Prompt versioning and management. Stores prompt templates with version history, supports rollback, and tracks performance per version.

### Integration (`integration.go`)
Connects LLMOps components with HelixAgent's provider registry and debate system.

### Types (`types.go`)
Shared type definitions: EvaluationResult, Experiment, PromptVersion, Dataset, Metric.

## Data Flow

```
Live Traffic → Evaluation Pipeline → Metrics Collection
                                      ↓
                              Experiment Analysis
                                      ↓
                              Prompt Optimization
```

## Package Structure

```
llmops/
├── evaluator.go     # Evaluation pipeline
├── experiments.go   # A/B experiment management
├── prompts.go       # Prompt versioning
├── integration.go   # HelixAgent integration
└── types.go         # Shared types
```

## Integration

Adapter: `internal/adapters/llmops/adapter.go`
