#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: bench/longbench-v2/evaluate.sh <run-dir> [pred-model-name] [-- --e]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STATE_DIR="${REPO_ROOT}/.benchmarks/longbench-v2"
VENDOR_DIR="${STATE_DIR}/vendor/LongBench"
RUN_DIR="$(python3 - <<'PY' "$1"
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
)"
MODEL_NAME="${2:-}"
shift $(( $# > 1 ? 2 : 1 ))

if [[ -z "${MODEL_NAME}" ]]; then
  MANIFEST_PATH="${RUN_DIR}/manifest.json"
  if [[ ! -f "${MANIFEST_PATH}" ]]; then
    echo "manifest missing at ${MANIFEST_PATH}" >&2
    exit 1
  fi
  MODEL_NAME="$(python3 - <<'PY' "${MANIFEST_PATH}"
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as fh:
    data = json.load(fh)
print(data['pred_model_name'])
PY
)"
fi

if [[ ! -d "${VENDOR_DIR}" ]]; then
  echo "LongBench vendor repo missing; run bench/longbench-v2/setup.sh first" >&2
  exit 1
fi

PRED_DIR="${VENDOR_DIR}/pred/${MODEL_NAME}"
mkdir -p "${PRED_DIR}"
rm -f "${PRED_DIR}"/*.jsonl
cp "${RUN_DIR}/pred/${MODEL_NAME}"/*.jsonl "${PRED_DIR}/"

pushd "${VENDOR_DIR}" >/dev/null
python3 eval.py --model "${MODEL_NAME}" "$@"
popd >/dev/null
