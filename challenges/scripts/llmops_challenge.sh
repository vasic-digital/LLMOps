#!/usr/bin/env bash
set -euo pipefail
PASS=0; FAIL=0
pass() { PASS=$((PASS+1)); echo "PASS: $1"; }
fail() { FAIL=$((FAIL+1)); echo "FAIL: $1"; }
[ -f "$(dirname "$0")/../../go.mod" ] && pass "go.mod exists" || fail "go.mod missing"
[ -f "$(dirname "$0")/../../llmops/evaluator.go" ] && pass "evaluator.go exists" || fail "evaluator.go missing"
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
