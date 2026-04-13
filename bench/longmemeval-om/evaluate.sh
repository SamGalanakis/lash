#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "usage: bench/longmemeval-om/evaluate.sh <hypotheses.jsonl> [ref.json] [judge-model]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
STATE_DIR="${REPO_ROOT}/.benchmarks/longmemeval-om"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

"${SCRIPT_DIR}/setup.sh"

HYP_FILE="$1"
REF_FILE="${2:-${STATE_DIR}/data/longmemeval_s_cleaned.json}"
JUDGE_MODEL="${3:-gpt-4o}"
HYP_FILE_ABS="$(cd -- "$(dirname -- "${HYP_FILE}")" && pwd)/$(basename -- "${HYP_FILE}")"
REF_FILE_ABS="$(cd -- "$(dirname -- "${REF_FILE}")" && pwd)/$(basename -- "${REF_FILE}")"
PYTHON_BIN="${STATE_DIR}/venv/bin/python"
EVAL_DIR="${STATE_DIR}/vendor/LongMemEval/src/evaluation"
RESULT_FILE="${HYP_FILE_ABS}.eval-results-${JUDGE_MODEL}"

(
  cd "${EVAL_DIR}"
  "${PYTHON_BIN}" evaluate_qa.py "${JUDGE_MODEL}" "${HYP_FILE_ABS}" "${REF_FILE_ABS}"
)

if [[ "${JUDGE_MODEL}" == "gpt-4o" ]]; then
  (
    cd "${EVAL_DIR}"
    "${PYTHON_BIN}" print_qa_metrics.py "${RESULT_FILE}" "${REF_FILE_ABS}"
  )
else
  echo "Skipping print_qa_metrics.py because the upstream script hard-codes the gpt-4o evaluator model." >&2
fi
