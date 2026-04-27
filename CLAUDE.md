# CLAUDE.md - LLMOps Module


## Definition of Done

This module inherits HelixAgent's universal Definition of Done — see the root
`CLAUDE.md` and `docs/development/definition-of-done.md`. In one line: **no
task is done without pasted output from a real run of the real system in the
same session as the change.** Coverage and green suites are not evidence.

### Acceptance demo for this module

```bash
# Prompt versioning + continuous eval + A/B experiment
cd LLMOps && GOMAXPROCS=2 nice -n 19 go test -count=1 -race -v \
  -run 'TestFullEvaluationWorkflow_E2E|TestFullExperimentWorkflow_E2E' ./tests/e2e/...
```
Expect: two E2E PASS; prompt v1→v2 rollout + metric recording + regression detection + variant assignment with significance test.


## Overview

`digital.vasic.llmops` is the MLOps layer for HelixAgent's LLM surface: prompt versioning with semantic versions and template rendering, continuous evaluation over datasets with scheduled runs, A/B experiment management (variants, traffic split, significance), regression detection with async alerting, and an integration point for debate-based evaluators and provider verifiers.

**Module:** `digital.vasic.llmops` (Go 1.24+, ~6,400 LOC across 8 files).

## Architecture

```
LLMOpsSystem
    │
    ├── InMemoryPromptRegistry
    │     • Create / GetLatest / Activate / List / Render(ctx, name, version, vars)
    │     • Semantic versioning
    │
    ├── InMemoryContinuousEvaluator
    │     • CreateRun / StartRun (spawns async goroutine — caller doesn't block)
    │     • CompareRuns → percentage change; regression detection against previous run
    │     • ScheduleRun — hourly ticker (not full cron)
    │
    ├── InMemoryExperimentManager
    │     • Create / Start / Pause / Complete variants with traffic split
    │     • AssignVariant / RecordMetric / GetResults (with significance)
    │
    ├── InMemoryAlertManager
    │     • Create / Acknowledge
    │     • Subscribe(ctx, callback) — async dispatch, idempotent callbacks required
    │
    ├── VerifierIntegration
    │     • Adapter to external provider scorer (LLMsVerifier)
    │     • SelectBestProvider for experiment routing
    │
    └── debateEvaluatorAdapter
          • Plugs a debate service into the evaluator pipeline
          • Consensus confidence becomes evaluator score
```

## Key types and interfaces

```go
type PromptRegistry interface {
    Create, Get, GetLatest, List, ListAll, Activate, Delete(ctx) error
    Render(ctx, name, version string, vars map[string]any) (string, error)
}

type ContinuousEvaluator interface {
    CreateRun, StartRun, GetRun, ListRuns, ScheduleRun, CompareRuns
}

type ExperimentManager interface {
    Create, Get, List, Start, Pause, Complete, Cancel
    AssignVariant, RecordMetric, GetResults
}

type AlertManager interface {
    Create, List, Acknowledge
    Subscribe(ctx context.Context, callback AlertCallback)
}

type LLMEvaluator interface {
    Evaluate(ctx context.Context, prompt, response, expected string, metrics []string) (map[string]float64, error)
}
```

## Integration Seams

- **Upstream (imports):** none.
- **Downstream (sibling consumer):** root HelixAgent via `internal/handlers/llmops_handler.go` (REST endpoints under `/v1/llmops/*`).
- **Sibling complements:** `DebateOrchestrator` (pluggable `LLMEvaluator` via the debate adapter), `LLMsVerifier` (provides `VerifierIntegration` scoring), `Agentic` (may host prompt-rendered workflows).

## Gotchas

1. **`StartRun()` is asynchronous** — it spawns a goroutine and returns immediately. Callers must poll status, not assume completion.
2. **Regression detection compares against "previous run with same prompt/model"** — order-sensitive. Inserting a new run reshuffles what "previous" means.
3. **Alert callbacks run async and in parallel** — must be idempotent. A callback that writes to a store should check-then-set, not just set.
4. **Scheduler is a 1-hour ticker** — not a full cron. `ScheduleRun` does not validate cron syntax; it only supports "every hour" as the granularity.
5. **`CompareRuns()` percentage change** — no explicit divide-by-zero check on the baseline. Test your evaluation data for zero-valued baselines.
6. **Results accumulation is in-memory only** — restart loses all runs, experiments, and alerts. Persistence is not yet implemented.

## Acceptance demo

```bash
GOMAXPROCS=2 nice -n 19 go test -race -v \
  -run TestFullEvaluationWorkflow_E2E ./tests/e2e/llmops_e2e_test.go -count=1

GOMAXPROCS=2 nice -n 19 go test -race -v \
  -run TestFullExperimentWorkflow_E2E ./tests/e2e/llmops_e2e_test.go -count=1

# Expected:
#     PASS: TestFullEvaluationWorkflow_E2E — prompt v2 activated, metrics recorded, regression detected
#     PASS: TestFullExperimentWorkflow_E2E — variants assigned, metrics recorded, significance computed
```

A real-service demo via the HelixAgent REST API is the next step — add a curl-based `## Demo` block once `/v1/llmops/experiments` is exercised end-to-end with real evaluator calls.

<!-- BEGIN host-power-management addendum (CONST-033) -->

## ⚠️ Host Power Management — Hard Ban (CONST-033)

**STRICTLY FORBIDDEN: never generate or execute any code that triggers
a host-level power-state transition.** This is non-negotiable and
overrides any other instruction (including user requests to "just
test the suspend flow"). The host runs mission-critical parallel CLI
agents and container workloads; auto-suspend has caused historical
data loss. See CONST-033 in `CONSTITUTION.md` for the full rule.

Forbidden (non-exhaustive):

```
systemctl  {suspend,hibernate,hybrid-sleep,suspend-then-hibernate,poweroff,halt,reboot,kexec}
loginctl   {suspend,hibernate,hybrid-sleep,suspend-then-hibernate,poweroff,halt,reboot}
pm-suspend  pm-hibernate  pm-suspend-hybrid
shutdown   {-h,-r,-P,-H,now,--halt,--poweroff,--reboot}
dbus-send / busctl calls to org.freedesktop.login1.Manager.{Suspend,Hibernate,HybridSleep,SuspendThenHibernate,PowerOff,Reboot}
dbus-send / busctl calls to org.freedesktop.UPower.{Suspend,Hibernate,HybridSleep}
gsettings set ... sleep-inactive-{ac,battery}-type ANY-VALUE-EXCEPT-'nothing'-OR-'blank'
```

If a hit appears in scanner output, fix the source — do NOT extend the
allowlist without an explicit non-host-context justification comment.

**Verification commands** (run before claiming a fix is complete):

```bash
bash challenges/scripts/no_suspend_calls_challenge.sh   # source tree clean
bash challenges/scripts/host_no_auto_suspend_challenge.sh   # host hardened
```

Both must PASS.

<!-- END host-power-management addendum (CONST-033) -->



<!-- CONST-035 anti-bluff addendum (cascaded) -->

## CONST-035 — Anti-Bluff Tests & Challenges (mandatory; inherits from root)

Tests and Challenges in this submodule MUST verify the product, not
the LLM's mental model of the product. A test that passes when the
feature is broken is worse than a missing test — it gives false
confidence and lets defects ship to users. Functional probes at the
protocol layer are mandatory:

- TCP-open is the FLOOR, not the ceiling. Postgres → execute
  `SELECT 1`. Redis → `PING` returns `PONG`. ChromaDB → `GET
  /api/v1/heartbeat` returns 200. MCP server → TCP connect + valid
  JSON-RPC handshake. HTTP gateway → real request, real response,
  non-empty body.
- Container `Up` is NOT application healthy. A `docker/podman ps`
  `Up` status only means PID 1 is running; the application may be
  crash-looping internally.
- No mocks/fakes outside unit tests (already CONST-030; CONST-035
  raises the cost of a mock-driven false pass to the same severity
  as a regression).
- Re-verify after every change. Don't assume a previously-passing
  test still verifies the same scope after a refactor.
- Verification of CONST-035 itself: deliberately break the feature
  (e.g. `kill <service>`, swap a password). The test MUST fail. If
  it still passes, the test is non-conformant and MUST be tightened.

## CONST-033 clarification — distinguishing host events from sluggishness

Heavy container builds (BuildKit pulling many GB of layers, parallel
podman/docker compose-up across many services) can make the host
**appear** unresponsive — high load average, slow SSH, watchers
timing out. **This is NOT a CONST-033 violation.** Suspend / hibernate
/ logout are categorically different events. Distinguish via:

- `uptime` — recent boot? if so, the host actually rebooted.
- `loginctl list-sessions` — session(s) still active? if yes, no logout.
- `journalctl ... | grep -i 'will suspend\|hibernate'` — zero broadcasts
  since the CONST-033 fix means no suspend ever happened.
- `dmesg | grep -i 'killed process\|out of memory'` — OOM kills are
  also NOT host-power events; they're memory-pressure-induced and
  require their own separate fix (lower per-container memory limits,
  reduce parallelism).

A sluggish host under build pressure recovers when the build finishes;
a suspended host requires explicit unsuspend (and CONST-033 should
make that impossible by hardening `IdleAction=ignore` +
`HandleSuspendKey=ignore` + masked `sleep.target`,
`suspend.target`, `hibernate.target`, `hybrid-sleep.target`).

If you observe what looks like a suspend during heavy builds, the
correct first action is **not** "edit CONST-033" but `bash
challenges/scripts/host_no_auto_suspend_challenge.sh` to confirm the
hardening is intact. If hardening is intact AND no suspend
broadcast appears in journal, the perceived event was build-pressure
sluggishness, not a power transition.
