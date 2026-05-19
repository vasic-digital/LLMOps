#!/usr/bin/env bash
#
# Round-290 §11.4 — paired-mutation Challenge driver for LLMOps.
#
# Per CONST-035 / Article XI §11.9 / round-220 paired-mutation
# pattern, the runner is exercised TWICE:
#   1. Normal mode — must exit 0 (all phases PASS).
#   2. Mutated mode (LLMOPS_CHALLENGE_MUTATE=registry) — the
#      runner's honest assertion against the active prompt version
#      must FAIL, exit 3. Any other exit code means the assertion
#      is a bluff (does not actually verify product behaviour).
#
# Additional locale sweep: invoke the normal-mode runner once per
# supported locale and assert exit 0 for each, proving the i18n
# surface stays intact across the 5-locale spread.
#
# Exit codes:
#   0  — normal PASS + mutated FAIL observed (anti-bluff verified)
#   3  — normal mode failed (real regression — not a bluff)
#   4  — mutation gate did NOT flip (assertion is a bluff)
#   5  — locale sweep failed for at least one locale
#   99 — operator override: deliberately mark the entire script
#         FAILED for external mutation-driver verification

set -euo pipefail

cd "$(dirname "$0")/.."

ROOT=$(pwd)
RUNNER_BIN="${ROOT}/bin/llmops_challenge_runner"
mkdir -p "${ROOT}/bin"

echo "== build runner =="
go build -o "${RUNNER_BIN}" ./challenges/runner

# Allow operator-driven outer mutation (paired-mutation of THIS script).
# When LLMOPS_DESCRIBE_CHALLENGE_MUTATE=1, exit 99 immediately so the
# external driver can prove THIS script's success-claim is real.
if [[ "${LLMOPS_DESCRIBE_CHALLENGE_MUTATE:-0}" == "1" ]]; then
  echo "[mutate-outer] exiting 99 as requested (paired-mutation driver)"
  exit 99
fi

NORMAL_OK=0
MUTATE_FLIPPED=0
LOCALE_OK=0
LOCALE_FAIL=0

echo "== phase 1: normal mode =="
if "${RUNNER_BIN}" -locale=en; then
  echo "[normal] exit=0 OK"
  NORMAL_OK=1
else
  rc=$?
  echo "[normal] exit=${rc} FAIL (real regression — not a bluff)"
  exit 3
fi

echo "== phase 2: mutated mode (registry) =="
set +e
"${RUNNER_BIN}" -locale=en -mutate=registry
mrc=$?
set -e
if [[ "${mrc}" -eq 3 ]]; then
  echo "[mutate=registry] exit=3 EXPECTED — assertion is real"
  MUTATE_FLIPPED=1
else
  echo "[mutate=registry] exit=${mrc} UNEXPECTED — assertion may be a bluff"
  exit 4
fi

echo "== phase 3: 5-locale sweep =="
for locale in en es de ja sr; do
  if "${RUNNER_BIN}" -locale="${locale}" >/dev/null; then
    echo "[locale=${locale}] exit=0 OK"
    LOCALE_OK=$((LOCALE_OK+1))
  else
    rc=$?
    echo "[locale=${locale}] exit=${rc} FAIL"
    LOCALE_FAIL=$((LOCALE_FAIL+1))
  fi
done

if [[ "${LOCALE_FAIL}" -gt 0 ]]; then
  echo "[locale-sweep] ${LOCALE_FAIL} failures across 5 locales"
  exit 5
fi

echo
echo "== summary =="
echo "normal mode PASS:          ${NORMAL_OK} (expect 1)"
echo "mutation flipped to FAIL:  ${MUTATE_FLIPPED} (expect 1)"
echo "locale sweep PASS:         ${LOCALE_OK}/5"
echo
echo "OK — LLMOps round-290 anti-bluff Challenge verified."
exit 0
