#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: bench/longmemeval-rlm/evaluate.sh <hypotheses.jsonl> [ref.json] [judge-model]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STATE_DIR="${REPO_ROOT}/.benchmarks/longmemeval-rlm"
VENDOR_DIR="${STATE_DIR}/vendor/LongMemEval"
VENV_DIR="${STATE_DIR}/venv"
REF_FILE="${2:-${STATE_DIR}/data/longmemeval_s_cleaned.json}"
JUDGE_MODEL="${3:-gpt-4o}"
HYPOTHESES_FILE="$(python3 - <<'PY' "$1"
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
)"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  source "${REPO_ROOT}/.env"
  set +a
fi

if [[ ! -d "${VENDOR_DIR}" || ! -d "${VENV_DIR}" ]]; then
  echo "benchmark evaluator is not set up; run bench/longmemeval-rlm/setup.sh first" >&2
  exit 1
fi

EVAL_DIR="${VENDOR_DIR}/src/evaluation"

pushd "${EVAL_DIR}" >/dev/null
"${VENV_DIR}/bin/python" evaluate_qa.py "${JUDGE_MODEL}" "${HYPOTHESES_FILE}" "${REF_FILE}"
if [[ "${JUDGE_MODEL}" == "gpt-4o" ]]; then
  "${VENV_DIR}/bin/python" print_qa_metrics.py "${HYPOTHESES_FILE}.eval-results-${JUDGE_MODEL}" "${REF_FILE}"
fi
popd >/dev/null
