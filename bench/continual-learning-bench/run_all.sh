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
BACKGROUND=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --background|--bg)
      BACKGROUND=1
      shift
      ;;
    --skip-baseline|--no-baseline)
      export LASH_CLBENCH_SKIP_BASELINE=1
      shift
      ;;
    --skip-baseline=*|--no-baseline=*)
      export LASH_CLBENCH_SKIP_BASELINE="${1#*=}"
      shift
      ;;
    --max-concurrency|--n-concurrent)
      ARGS+=(--task-parallelism "${2:?missing value for $1}")
      shift 2
      ;;
    --max-concurrency=*|--n-concurrent=*)
      ARGS+=(--task-parallelism "${1#*=}")
      shift
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

cd "${VENDOR_DIR}"

if [[ "${BACKGROUND}" != "1" ]]; then
  exec "${VENV_DIR}/bin/clbench" run-all "${ARGS[@]}"
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${STATE_DIR}/logs/run-all-background"
LOG_PATH="${LOG_DIR}/${RUN_ID}.log"
PID_PATH="${STATE_DIR}/run-all-background.pid"
STATUS_PATH="${STATE_DIR}/run-all-background.env"

mkdir -p "${LOG_DIR}"
setsid "${VENV_DIR}/bin/clbench" run-all "${ARGS[@]}" </dev/null >"${LOG_PATH}" 2>&1 &
PID="$!"

cat >"${PID_PATH}" <<EOF
${PID}
EOF

cat >"${STATUS_PATH}" <<EOF
pid=${PID}
log=${LOG_PATH}
started_at=${RUN_ID}
EOF

URL=""
for _ in {1..20}; do
  if ! kill -0 "${PID}" 2>/dev/null; then
    break
  fi
  URL="$(sed -n 's/^Live dashboard: //p; s/^Aggregate viewer: //p' "${LOG_PATH}" | tail -1)"
  if [[ -n "${URL}" ]]; then
    break
  fi
  sleep 0.5
done

if [[ -n "${URL}" ]]; then
  cat >>"${STATUS_PATH}" <<EOF
dashboard_url=${URL}
EOF
fi

echo "Started CLBench run-all in background."
echo "PID: ${PID}"
echo "Log: ${LOG_PATH}"
if [[ -n "${URL}" ]]; then
  echo "Dashboard: ${URL}"
else
  echo "Dashboard URL pending; run: sed -n 's/^Live dashboard: //p' ${LOG_PATH} | tail -1"
fi
echo "Follow logs: tail -f ${LOG_PATH}"
