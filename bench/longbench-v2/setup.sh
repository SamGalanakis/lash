#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STATE_DIR="${REPO_ROOT}/.benchmarks/longbench-v2"
VENDOR_DIR="${STATE_DIR}/vendor/LongBench"
VENV_DIR="${STATE_DIR}/venv"

mkdir -p "${STATE_DIR}/vendor"

if [[ ! -d "${VENDOR_DIR}" ]]; then
  git clone --depth 1 https://github.com/THUDM/LongBench.git "${VENDOR_DIR}"
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/pip" install -q --upgrade pip
"${VENV_DIR}/bin/pip" install -q tqdm

echo "LongBench benchmark state ready under ${STATE_DIR}"
