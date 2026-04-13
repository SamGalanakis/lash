#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
STATE_DIR="${REPO_ROOT}/.benchmarks/longmemeval-om"
VENV_DIR="${STATE_DIR}/venv"
VENDOR_DIR="${STATE_DIR}/vendor/LongMemEval"
DATA_DIR="${STATE_DIR}/data"

mkdir -p "${STATE_DIR}/vendor" "${DATA_DIR}"

if [[ ! -d "${VENDOR_DIR}/.git" ]]; then
  git clone --depth 1 https://github.com/xiaowu0162/LongMemEval.git "${VENDOR_DIR}"
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip >/dev/null
"${VENV_DIR}/bin/python" -m pip install -r "${VENDOR_DIR}/requirements-lite.txt"
# Upstream pins openai==1.35.1, which is incompatible with httpx 0.28+.
"${VENV_DIR}/bin/python" -m pip install "httpx<0.28"

if [[ ! -f "${DATA_DIR}/longmemeval_s_cleaned.json" ]]; then
  curl -L https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json -o "${DATA_DIR}/longmemeval_s_cleaned.json"
fi
