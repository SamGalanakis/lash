#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STATE_DIR="${REPO_ROOT}/.benchmarks/continual-learning-bench"
VENDOR_DIR="${STATE_DIR}/vendor/continual-learning-bench"
TARGET_DIR="${VENDOR_DIR}/src/systems/lash"

if [[ ! -d "${VENDOR_DIR}/.git" ]]; then
  echo "error: CLBench checkout missing at ${VENDOR_DIR}; run bench/continual-learning-bench/setup.sh first" >&2
  exit 1
fi

mkdir -p "${TARGET_DIR}"
cp "${SCRIPT_DIR}/lash_system/__init__.py" "${TARGET_DIR}/__init__.py"
cp "${SCRIPT_DIR}/lash_system/system.py" "${TARGET_DIR}/system.py"

echo "Synced Lash CLBench system adapter into ${TARGET_DIR}"
