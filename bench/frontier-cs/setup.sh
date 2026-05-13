#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STATE_DIR="${REPO_ROOT}/.benchmarks/frontier-cs"
SOURCE_DIR="${STATE_DIR}/source"

mkdir -p "${STATE_DIR}/runs"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required to prepare Frontier-CS." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to prepare Frontier-CS (https://docs.astral.sh/uv/)." >&2
  echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
  git clone --depth 1 https://github.com/FrontierCS/Frontier-CS.git "${SOURCE_DIR}"
else
  git -C "${SOURCE_DIR}" pull --ff-only
fi

cd "${SOURCE_DIR}"
uv sync

echo "Frontier-CS prepared at ${SOURCE_DIR}"
