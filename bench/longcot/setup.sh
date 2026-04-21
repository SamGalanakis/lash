#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STATE_DIR="${REPO_ROOT}/.benchmarks/longcot"
VENDOR_DIR="${STATE_DIR}/vendor/longcot"
VENV_DIR="${VENDOR_DIR}/.venv"

mkdir -p "${STATE_DIR}/vendor" "${STATE_DIR}/runs"

if [[ ! -d "${VENDOR_DIR}" ]]; then
  git clone --depth 1 https://github.com/LongHorizonReasoning/longcot.git "${VENDOR_DIR}"
else
  git -C "${VENDOR_DIR}" fetch --depth 1 origin main >/dev/null 2>&1 || true
  git -C "${VENDOR_DIR}" reset --hard origin/main >/dev/null 2>&1 || true
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to set up the LongCoT evaluator (https://docs.astral.sh/uv/)." >&2
  echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

pushd "${VENDOR_DIR}" >/dev/null
uv sync --no-dev >/dev/null
popd >/dev/null

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "uv sync did not produce ${VENV_DIR}" >&2
  exit 1
fi

echo "LongCoT benchmark state ready under ${STATE_DIR}"
echo "  vendor: ${VENDOR_DIR}"
echo "  venv:   ${VENV_DIR}"
