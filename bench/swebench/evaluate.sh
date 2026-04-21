#!/usr/bin/env bash
set -euo pipefail

# Grade a run's predictions.jsonl using the upstream SWE-bench evaluator.
# Usage:
#   bench/swebench/evaluate.sh <run-dir-or-predictions.jsonl> [-- extra run_evaluation.py args]
#
# Default dataset: SWE-bench/SWE-bench_Verified (test split). Pass
# `--dataset_name` / `--split` to override.

if [[ $# -lt 1 ]]; then
  echo "usage: bench/swebench/evaluate.sh <run-dir-or-predictions.jsonl> [-- extra run_evaluation.py args]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STATE_DIR="${REPO_ROOT}/.benchmarks/swebench"
VENDOR_DIR="${STATE_DIR}/vendor/SWE-bench"

TARGET="$1"
shift || true

if [[ ! -d "${VENDOR_DIR}" ]]; then
  echo "SWE-bench vendor repo missing; run bench/swebench/setup.sh first" >&2
  exit 1
fi

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  source "${REPO_ROOT}/.env"
  set +a
fi

if [[ -d "${TARGET}" ]]; then
  PREDICTIONS="${TARGET}/predictions.jsonl"
  RUN_ID="$(basename "${TARGET}")"
  REPORT_DIR="${TARGET}"
else
  PREDICTIONS="${TARGET}"
  RUN_ID="$(basename "$(dirname "${PREDICTIONS}")")"
  REPORT_DIR="$(dirname "${PREDICTIONS}")"
fi

if [[ ! -f "${PREDICTIONS}" ]]; then
  echo "predictions file not found: ${PREDICTIONS}" >&2
  exit 1
fi

echo "Evaluating ${PREDICTIONS}"
echo "  run_id:     ${RUN_ID}"
echo "  report_dir: ${REPORT_DIR}"

uv --project "${VENDOR_DIR}" run --no-dev python -m swebench.harness.run_evaluation \
  --dataset_name SWE-bench/SWE-bench_Verified \
  --split test \
  --predictions_path "${PREDICTIONS}" \
  --run_id "${RUN_ID}" \
  --report_dir "${REPORT_DIR}" \
  "$@"
