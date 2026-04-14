#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STATE_DIR="${REPO_ROOT}/.benchmarks/longmemeval-rlm"
DATA_DIR="${STATE_DIR}/data"
VENDOR_DIR="${STATE_DIR}/vendor/LongMemEval"
VENV_DIR="${STATE_DIR}/venv"

mkdir -p "${DATA_DIR}" "${STATE_DIR}/vendor"

if [[ ! -d "${VENDOR_DIR}" ]]; then
  git clone --depth 1 https://github.com/xiaowu0162/LongMemEval.git "${VENDOR_DIR}"
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/pip" install -q --upgrade pip
"${VENV_DIR}/bin/pip" install -q "openai==1.35.1" "httpx<0.28" backoff tqdm numpy

if [[ ! -f "${DATA_DIR}/longmemeval_s_cleaned.json" ]]; then
  curl -L https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json -o "${DATA_DIR}/longmemeval_s_cleaned.json"
fi

if [[ ! -f "${DATA_DIR}/longmemeval_s_flash_failures_64.json" ]]; then
  curl -L https://raw.githubusercontent.com/rawwerks/longmemeval-rlm/master/data/longmemeval_s_flash_failures_64.json -o "${DATA_DIR}/longmemeval_s_flash_failures_64.json"
fi

if [[ ! -f "${DATA_DIR}/discordant_110.json" ]]; then
  curl -L https://raw.githubusercontent.com/rawwerks/longmemeval-rlm/master/data/discordant_110.json -o "${DATA_DIR}/discordant_110.json"
fi

echo "LongMemEval RLM benchmark state ready under ${STATE_DIR}"
