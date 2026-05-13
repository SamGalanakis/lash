#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STATE_DIR="${REPO_ROOT}/.benchmarks/oolong"
DATA_DIR="${STATE_DIR}/data"

mkdir -p "${DATA_DIR}" "${STATE_DIR}/runs"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to prepare OOLONG slices (https://docs.astral.sh/uv/)." >&2
  echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

exec uv run --with datasets --with huggingface-hub --with pyarrow --with tqdm \
  "${SCRIPT_DIR}/prepare.py" --output-dir "${DATA_DIR}" "$@"
