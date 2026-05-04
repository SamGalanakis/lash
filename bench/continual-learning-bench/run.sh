#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STATE_DIR="${REPO_ROOT}/.benchmarks/continual-learning-bench"
VENDOR_DIR="${STATE_DIR}/vendor/continual-learning-bench"
VENV_DIR="${STATE_DIR}/venv"

if [[ ! -x "${VENV_DIR}/bin/clbench" || ! -d "${VENDOR_DIR}/.git" ]]; then
  "${SCRIPT_DIR}/setup.sh" --skip-task-setup
fi

"${SCRIPT_DIR}/sync_lash_system.sh" >/dev/null

export LASH_CLBENCH_REPO_ROOT="${REPO_ROOT}"
export LASH_CLBENCH_STATE_ROOT="${STATE_DIR}"

ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-concurrency|--n-concurrent)
      ARGS+=(--max-workers "${2:?missing value for $1}")
      shift 2
      ;;
    --max-concurrency=*|--n-concurrent=*)
      ARGS+=(--max-workers "${1#*=}")
      shift
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

cd "${VENDOR_DIR}"
exec "${VENV_DIR}/bin/clbench" run "${ARGS[@]}"
