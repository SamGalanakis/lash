#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STATE_DIR="${REPO_ROOT}/.benchmarks/continual-learning-bench"
VENDOR_DIR="${STATE_DIR}/vendor/continual-learning-bench"
VENV_DIR="${STATE_DIR}/venv"
CLBENCH_REPO_URL="https://github.com/pgasawa/continual-learning-bench.git"
CLBENCH_REF=""
RUN_TASK_SETUP=1

usage() {
  cat <<'EOF'
Usage: bench/continual-learning-bench/setup.sh [options]

Options:
  --ref <git-ref>        Check out a specific CLBench branch/tag/commit after cloning.
  --skip-task-setup      Do not run `clbench setup --all`.
  --help                 Show this help.

Installs Continual Learning Bench into ignored .benchmarks state and syncs the
tracked Lash system adapter into the cached upstream checkout.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ref)
      CLBENCH_REF="${2:?missing value for --ref}"
      shift 2
      ;;
    --skip-task-setup)
      RUN_TASK_SETUP=0
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "error: required command not found: $1" >&2
    exit 1
  fi
}

require_cmd git
require_cmd uv

mkdir -p "${STATE_DIR}/vendor"

if [[ ! -d "${VENDOR_DIR}/.git" ]]; then
  git clone --depth 1 "${CLBENCH_REPO_URL}" "${VENDOR_DIR}"
fi

if [[ -n "${CLBENCH_REF}" ]]; then
  git -C "${VENDOR_DIR}" fetch --depth 1 origin "${CLBENCH_REF}"
  git -C "${VENDOR_DIR}" checkout --detach FETCH_HEAD
fi

"${SCRIPT_DIR}/sync_lash_system.sh"

if [[ ! -d "${VENV_DIR}" ]]; then
  uv venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m ensurepip --upgrade >/dev/null 2>&1 || true
uv pip install --python "${VENV_DIR}/bin/python" -q --upgrade pip
uv pip install --python "${VENV_DIR}/bin/python" -q -e "${VENDOR_DIR}[all]"

if [[ "${RUN_TASK_SETUP}" -eq 1 ]]; then
  (
    cd "${VENDOR_DIR}"
    "${VENV_DIR}/bin/clbench" setup --all
  )
fi

cat <<EOF
Continual Learning Bench state ready under ${STATE_DIR}
  checkout: ${VENDOR_DIR}
  venv:     ${VENV_DIR}
EOF
