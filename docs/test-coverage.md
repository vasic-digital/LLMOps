# LLMOps — Test Coverage Ledger (CONST-050(B))

**Round 290 §11.4 — symbol-to-test ledger.** Per CONST-050(B), every
exported production symbol in the LLMOps submodule MUST be covered
by at least one test type with captured runtime evidence. This
document enumerates that mapping and is regenerated whenever the
public surface changes.

Ledger format: each row lists the production symbol, the test files
that exercise it, and the closest test-type bucket per CONST-050(B).
Rows whose `evidence` column is `runtime` carry positive captured
output (test execution stdout); rows marked `structural` need
upgrading to runtime evidence and are tracked in
`docs/issues/Issues.md` per §11.4.15.

## llmops package (`digital.vasic.llmops/llmops`)

| Symbol                                  | Test file(s)                                                                 | Type            | Evidence  |
|-----------------------------------------|------------------------------------------------------------------------------|-----------------|-----------|
| `NewInMemoryPromptRegistry`             | `prompts_test.go`, `challenges/runner/main.go`                              | unit + challenge| runtime   |
| `InMemoryPromptRegistry.Create`         | `prompts_test.go`, `challenges/runner/main.go::phasePromptRegistry`         | unit + challenge| runtime   |
| `InMemoryPromptRegistry.Get`            | `prompts_test.go`                                                            | unit            | runtime   |
| `InMemoryPromptRegistry.GetLatest`      | `prompts_test.go`, `challenges/runner/main.go::phasePromptRegistry`         | unit + challenge| runtime   |
| `InMemoryPromptRegistry.List`           | `prompts_test.go`, `challenges/runner/main.go::phasePromptRegistry`         | unit + challenge| runtime   |
| `InMemoryPromptRegistry.ListAll`        | `prompts_test.go`                                                            | unit            | runtime   |
| `InMemoryPromptRegistry.Activate`       | `prompts_test.go`                                                            | unit            | runtime   |
| `InMemoryPromptRegistry.Delete`         | `prompts_test.go`                                                            | unit            | runtime   |
| `InMemoryPromptRegistry.Render`         | `prompts_test.go`, `challenges/runner/main.go::phaseRender`                 | unit + challenge| runtime   |
| `NewPromptVersionComparator`            | `prompts_test.go`, `challenges/runner/main.go::phaseCompare`                | unit + challenge| runtime   |
| `PromptVersionComparator.Compare`       | `prompts_test.go`, `challenges/runner/main.go::phaseCompare`                | unit + challenge| runtime   |
| `NewInMemoryContinuousEvaluator`        | `evaluator_test.go`                                                          | unit            | runtime   |
| `InMemoryContinuousEvaluator.CreateRun` | `evaluator_test.go`, `tests/integration/`                                   | unit + integ.   | runtime   |
| `InMemoryContinuousEvaluator.StartRun`  | `evaluator_test.go`, `tests/integration/`                                   | unit + integ.   | runtime   |
| `InMemoryContinuousEvaluator.GetRun`    | `evaluator_test.go`                                                          | unit            | runtime   |
| `InMemoryContinuousEvaluator.CompareRuns`| `evaluator_test.go`                                                         | unit            | runtime   |
| `InMemoryContinuousEvaluator.ScheduleRun`| `evaluator_test.go`                                                         | unit            | runtime   |
| `HeuristicEvaluator`                    | `heuristic_evaluator_test.go`                                                | unit            | runtime   |
| `HeuristicEvaluator.Evaluate`           | `heuristic_evaluator_test.go`                                                | unit            | runtime   |
| `LLMBackedEvaluator`                    | `llm_backed_evaluator_test.go`                                               | unit            | runtime   |
| `LLMBackedEvaluator.Evaluate`           | `llm_backed_evaluator_test.go`                                               | unit            | runtime   |
| `NewInMemoryExperimentManager`          | `experiments_test.go`, `tests/e2e/`                                          | unit + e2e      | runtime   |
| `InMemoryExperimentManager.Create`      | `experiments_test.go`                                                        | unit            | runtime   |
| `InMemoryExperimentManager.Start`       | `experiments_test.go`, `tests/e2e/`                                          | unit + e2e      | runtime   |
| `InMemoryExperimentManager.AssignVariant`| `experiments_test.go`                                                       | unit            | runtime   |
| `InMemoryExperimentManager.RecordMetric`| `experiments_test.go`                                                        | unit            | runtime   |
| `InMemoryExperimentManager.GetResults`  | `experiments_test.go`                                                        | unit            | runtime   |
| `NewInMemoryAlertManager`               | `experiments_test.go` (alert subtests)                                       | unit            | runtime   |
| `InMemoryAlertManager.Create`           | `experiments_test.go` (alert subtests)                                       | unit            | runtime   |
| `InMemoryAlertManager.Subscribe`        | `experiments_test.go` (alert subtests)                                       | unit            | runtime   |
| `HTTPResponder`                         | `responder_test.go`                                                          | unit            | runtime   |
| `VerifierIntegration`                   | `integration_test.go`                                                        | unit + integ.   | runtime   |
| `debateEvaluatorAdapter`                | `integration_test.go`                                                        | unit + integ.   | runtime   |

## i18n package (`digital.vasic.llmops/pkg/i18n`)

| Symbol                                  | Test file(s)                                                                 | Type            | Evidence  |
|-----------------------------------------|------------------------------------------------------------------------------|-----------------|-----------|
| `Translator` (interface)                | `translator_test.go`, `llmops/i18n_callsite_test.go`                        | unit            | runtime   |
| `NoopTranslator.T`                      | `translator_test.go`                                                         | unit            | runtime   |

## Challenge-runner anti-bluff harness

| Symbol                                  | Test driver                                                                  | Type            | Evidence  |
|-----------------------------------------|------------------------------------------------------------------------------|-----------------|-----------|
| `main` (runner binary)                  | `challenges/llmops_describe_challenge.sh` normal + mutated invocations      | challenge       | runtime   |
| `runner.phasePromptRegistry`            | `LLMOPS_CHALLENGE_MUTATE=registry` mutation gate                            | challenge       | runtime   |
| `runner.phaseRender`                    | normal-mode pass; mutation hook reserved                                     | challenge       | runtime   |
| `runner.phaseCompare`                   | normal-mode pass; mutation hook reserved                                     | challenge       | runtime   |
| `runner.phaseI18n`                      | 5-locale sweep (en/es/de/ja/sr)                                              | challenge       | runtime   |
| `runner.loadLocale`                     | 5-locale fixture load + fallback marker                                      | challenge       | runtime   |

## CONST-050(B) test-type ledger — per-bucket residency

- **unit** — `llmops/*_test.go`, `pkg/i18n/translator_test.go`
- **integration** — `tests/integration/`
- **e2e** — `tests/e2e/`
- **benchmark** — `tests/benchmark/`
- **security** — `tests/security/`
- **stress** — `tests/stress/`
- **chaos** — `challenges/scripts/chaos_failure_injection_challenge.sh`
- **ddos** — `challenges/scripts/ddos_health_flood_challenge.sh`
- **scaling** — `challenges/scripts/scaling_horizontal_challenge.sh`
- **stress (challenge layer)** — `challenges/scripts/stress_sustained_load_challenge.sh`
- **ui / ux** — `challenges/scripts/{ui_terminal,ux_end_to_end}_*_challenge.sh`
- **challenges (anti-bluff harness)** — `challenges/llmops_describe_challenge.sh` (round-290)
- **host-power-bans (CONST-033/036)** — `challenges/scripts/{no_suspend_calls,host_no_auto_suspend}_challenge.sh`

## Verification recipe

```bash
# Unit + benchmark suite under -race:
GOMAXPROCS=2 go test -race -count=1 ./llmops/... ./pkg/i18n/...

# Round-290 anti-bluff challenge (normal + mutation + locale sweep):
bash challenges/llmops_describe_challenge.sh

# Paired-mutation of the describe-challenge driver itself:
LLMOPS_DESCRIBE_CHALLENGE_MUTATE=1 bash challenges/llmops_describe_challenge.sh
#   expected exit code: 99
```

Captured runtime evidence per CONST-035 is committed alongside the
governance commit message and surfaces in `git log` for any future
auditor — no metadata-only PASS, no absence-of-error PASS.
