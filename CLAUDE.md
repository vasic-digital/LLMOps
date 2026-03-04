# CLAUDE.md - LLMOps Module

## Overview

`digital.vasic.llmops` is a generic, reusable Go module for LLM operations: continuous evaluation pipelines, A/B experiment management, model dataset management, and prompt versioning.

**Module**: `digital.vasic.llmops` (Go 1.24+)

## Build & Test

```bash
go build ./...
go test ./... -count=1 -race
go test ./... -short
```

## Package Structure

| Package | Purpose |
|---------|---------|
| `llmops` | Core LLMOps types: evaluator, experiments, datasets, prompts |

## Key Types

- `InMemoryContinuousEvaluator` — In-memory evaluation pipeline
- `InMemoryExperimentManager` — A/B experiment management
- `Dataset` — Evaluation dataset (golden, synthetic, production)
- `EvaluationRun` — Single evaluation run against a dataset
