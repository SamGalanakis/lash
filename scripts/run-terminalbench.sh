#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Run Terminal Bench 2 with lash via Harbor.

Usage:
  scripts/run-terminalbench.sh [options] [-- <extra harbor args>]

Options:
  --dataset <name@version>      Dataset to run (default: terminal-bench-sample@2.0)
  --sample                      Shortcut for --dataset terminal-bench-sample@2.0
  --full                        Shortcut for --dataset terminal-bench@2.0
  --task <glob>                 Task include pattern (repeatable)
  --tasks <a,b,c>               Exact task names as a comma-separated list
  --task-file <path>            Exact task names from a file (one per line, # comments allowed)
  --exclude-task <glob>         Task exclude pattern (repeatable)
  --model <model>               Model passed to lash (optional)
  --variant <name>              Provider-native model variant passed to lash (optional)
  --execution-mode <mode>       Lash execution mode: repl|standard (required)
  --jobs-dir <path>             Harbor jobs output dir (default: jobs)
  --results-dir <path>          Persistent structured results dir (default: .benchmarks/terminalbench)
  --job-name <name>             Harbor job name (optional)
  --n-concurrent <int>          Concurrent trials (default: 1)
  --attempts <int>              Attempts per trial (default: 1)
  --timeout-multiplier <float>  Task timeout multiplier (default: 1.0)
  --env <name>                  Harbor environment backend (default: docker)
  --registry-url <url>          Dataset registry URL
                                (default: https://raw.githubusercontent.com/laude-institute/harbor/main/registry.json)
  --no-build                    Skip building the benchmark binary
  --debug                       Enable Harbor debug logging
  --no-debug                    Disable Harbor debug logging (default)
  --delete                      Delete benchmark environments after run
  --no-delete                   Keep benchmark environments after run
  --allow-no-config             Do not require ~/.lash/config.json
  --dry-run                     Print command and exit
  --help                        Show this help

Examples:
  scripts/run-terminalbench.sh --sample --execution-mode repl
  scripts/run-terminalbench.sh --full --execution-mode standard --task "git-*"
  scripts/run-terminalbench.sh --sample --execution-mode standard --tasks regex-log,sqlite-with-gcov
  scripts/run-terminalbench.sh --sample --execution-mode repl --task chess-best-move --model gpt-5.3-codex
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "error: required command not found: $1" >&2
    exit 1
  fi
}

DATASET="terminal-bench-sample@2.0"
JOBS_DIR="jobs"
RESULTS_DIR=".benchmarks/terminalbench"
JOB_NAME=""
MODEL=""
VARIANT=""
EXECUTION_MODE=""
N_CONCURRENT="1"
N_CONCURRENT_SET=0
ATTEMPTS="1"
TIMEOUT_MULT="1.0"
ENV_BACKEND="docker"
REGISTRY_URL="https://raw.githubusercontent.com/laude-institute/harbor/main/registry.json"
DO_BUILD=1
DELETE_AFTER_RUN=1
REQUIRE_CONFIG=1
DRY_RUN=0
DEBUG=0

TASK_PATTERNS=()
EXACT_TASKS=()
EXCLUDE_PATTERNS=()
EXTRA_ARGS=()

append_exact_tasks() {
  local raw="$1"
  local part trimmed
  IFS=',' read -r -a parts <<<"${raw}"
  for part in "${parts[@]}"; do
    trimmed="$(printf '%s' "${part}" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
    if [[ -n "${trimmed}" ]]; then
      EXACT_TASKS+=("${trimmed}")
    fi
  done
}

load_task_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "error: task file not found: ${path}" >&2
    exit 1
  fi

  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line%%#*}"
    line="$(printf '%s' "${line}" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
    if [[ -n "${line}" ]]; then
      EXACT_TASKS+=("${line}")
    fi
  done <"${path}"
}

join_by() {
  local delim="$1"
  shift
  local out=""
  local item
  for item in "$@"; do
    if [[ -n "${out}" ]]; then
      out+="${delim}"
    fi
    out+="${item}"
  done
  printf '%s' "${out}"
}

sanitize_job_fragment() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9._-]/-/g; s/--*/-/g; s/^-//; s/-$//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="${2:?missing value for --dataset}"
      shift 2
      ;;
    --sample)
      DATASET="terminal-bench-sample@2.0"
      shift
      ;;
    --full)
      DATASET="terminal-bench@2.0"
      shift
      ;;
    --task)
      TASK_PATTERNS+=("${2:?missing value for --task}")
      shift 2
      ;;
    --tasks)
      append_exact_tasks "${2:?missing value for --tasks}"
      shift 2
      ;;
    --task-file)
      load_task_file "${2:?missing value for --task-file}"
      shift 2
      ;;
    --exclude-task)
      EXCLUDE_PATTERNS+=("${2:?missing value for --exclude-task}")
      shift 2
      ;;
    --model)
      MODEL="${2:?missing value for --model}"
      shift 2
      ;;
    --variant)
      VARIANT="${2:?missing value for --variant}"
      shift 2
      ;;
    --execution-mode)
      EXECUTION_MODE="${2:?missing value for --execution-mode}"
      shift 2
      ;;
    --jobs-dir)
      JOBS_DIR="${2:?missing value for --jobs-dir}"
      shift 2
      ;;
    --results-dir)
      RESULTS_DIR="${2:?missing value for --results-dir}"
      shift 2
      ;;
    --job-name)
      JOB_NAME="${2:?missing value for --job-name}"
      shift 2
      ;;
    --n-concurrent)
      N_CONCURRENT="${2:?missing value for --n-concurrent}"
      N_CONCURRENT_SET=1
      shift 2
      ;;
    --attempts)
      ATTEMPTS="${2:?missing value for --attempts}"
      shift 2
      ;;
    --timeout-multiplier)
      TIMEOUT_MULT="${2:?missing value for --timeout-multiplier}"
      shift 2
      ;;
    --env)
      ENV_BACKEND="${2:?missing value for --env}"
      shift 2
      ;;
    --registry-url)
      REGISTRY_URL="${2:?missing value for --registry-url}"
      shift 2
      ;;
    --no-build)
      DO_BUILD=0
      shift
      ;;
    --debug)
      DEBUG=1
      shift
      ;;
    --no-debug)
      DEBUG=0
      shift
      ;;
    --delete)
      DELETE_AFTER_RUN=1
      shift
      ;;
    --no-delete)
      DELETE_AFTER_RUN=0
      shift
      ;;
    --allow-no-config)
      REQUIRE_CONFIG=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

require_cmd harbor
if [[ "${DO_BUILD}" -eq 1 || "${ENV_BACKEND}" == "docker" ]]; then
  require_cmd docker
fi

if [[ "${REQUIRE_CONFIG}" -eq 1 && ! -f "${HOME}/.lash/config.json" ]]; then
  cat >&2 <<EOF
error: ${HOME}/.lash/config.json not found.
This runner expects your local lash provider config (including OAuth tokens).
Use --allow-no-config to bypass.
EOF
  exit 1
fi

if [[ -z "${EXECUTION_MODE}" ]]; then
  echo "error: --execution-mode is required (expected repl|standard)" >&2
  exit 2
fi

if [[ "${EXECUTION_MODE}" == "native-tools" ]]; then
  EXECUTION_MODE="standard"
fi

if [[ "${EXECUTION_MODE}" != "repl" && "${EXECUTION_MODE}" != "standard" ]]; then
  echo "error: unsupported --execution-mode: ${EXECUTION_MODE} (expected repl|standard)" >&2
  exit 2
fi

if [[ ${#EXACT_TASKS[@]} -gt 0 ]]; then
  mapfile -t EXACT_TASKS < <(printf '%s\n' "${EXACT_TASKS[@]}" | awk '!seen[$0]++')
fi

if [[ ${#EXACT_TASKS[@]} -gt 0 && "${N_CONCURRENT_SET}" -eq 0 ]]; then
  N_CONCURRENT="${#EXACT_TASKS[@]}"
fi

build_benchmark_binary() {
  local target_dir="${REPO_ROOT}/target-bullseye"
  local image="rust:1-bullseye"
  mkdir -p "${target_dir}"
  echo "==> Building lash benchmark binary in rust:1-bullseye" >&2
  docker run --rm -u root \
    -v "${REPO_ROOT}:/work" \
    -w /work \
    "${image}" \
    bash -lc \
      '. /usr/local/cargo/env &&
       apt-get update >/dev/null &&
       apt-get install -y protobuf-compiler zstd python3-dev >/dev/null &&
       CARGO_TARGET_DIR=/work/target-bullseye cargo build --release --manifest-path /work/Cargo.toml --bin lash &&
       chown -R $(stat -c "%u:%g" /work) /work/target-bullseye' >/dev/null
  echo "${REPO_ROOT}/target-bullseye/release/lash"
}

BINARY_PATH="${REPO_ROOT}/target-bullseye/release/lash"
if [[ "${DO_BUILD}" -eq 1 ]]; then
  BINARY_PATH="$(build_benchmark_binary)"
fi

if [[ ! -x "${BINARY_PATH}" ]]; then
  echo "error: expected executable lash binary not found at ${BINARY_PATH}" >&2
  exit 1
fi

export LASH_BENCH_BINARY="${BINARY_PATH}"
export LASH_BENCH_EXECUTION_MODE="${EXECUTION_MODE}"
export LASH_BENCH_MODEL_VARIANT="${VARIANT}"

if [[ -z "${LASH_PROMPT_REPLACE_IDENTITY:-}" ]]; then
  export LASH_PROMPT_REPLACE_IDENTITY="$(cat <<EOF
You are running inside a benchmark harness with an enforced wall-clock time budget.
Work autonomously and prioritize passing the verifier over polish.
Do not ask the user questions; there is no interactive user in this run.
Make concrete progress continuously: inspect, edit, run checks, and converge quickly.
If you are blocked or near timeout, return the best valid result you can and call done() with a concise status + remaining risks.
EOF
)"
fi

# Always capture LLM request/response traces for benchmark debugging.
export LASH_LOG="debug"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ -z "${JOB_NAME}" ]]; then
  dataset_slug="$(sanitize_job_fragment "${DATASET%@*}")"
  mode_slug="$(sanitize_job_fragment "${EXECUTION_MODE}")"
  if [[ ${#EXACT_TASKS[@]} -gt 0 ]]; then
    task_slug="$(sanitize_job_fragment "$(join_by "-" "${EXACT_TASKS[@]}")")"
    task_slug="${task_slug:0:48}"
    JOB_NAME="${dataset_slug}-${mode_slug}-${task_slug}"
  else
    JOB_NAME="${dataset_slug}-${mode_slug}-$(date +%Y%m%d-%H%M%S)"
  fi
fi

CMD=(
  harbor run
  --agent-import-path scripts.harbor_lash_agent:LashAgent
  --dataset "${DATASET}"
  --registry-url "${REGISTRY_URL}"
  --env "${ENV_BACKEND}"
  --jobs-dir "${JOBS_DIR}"
  --n-concurrent "${N_CONCURRENT}"
  --n-attempts "${ATTEMPTS}"
  --timeout-multiplier "${TIMEOUT_MULT}"
  --job-name "${JOB_NAME}"
)

if [[ -n "${MODEL}" ]]; then
  CMD+=(--model "${MODEL}")
fi

if [[ "${DELETE_AFTER_RUN}" -eq 0 ]]; then
  CMD+=(--no-delete)
fi

if [[ "${DEBUG}" -eq 1 ]]; then
  CMD+=(--debug)
fi

for pattern in "${TASK_PATTERNS[@]}"; do
  CMD+=(--task-name "${pattern}")
done

for task_name in "${EXACT_TASKS[@]}"; do
  CMD+=(--task-name "${task_name}")
done

for pattern in "${EXCLUDE_PATTERNS[@]}"; do
  CMD+=(--exclude-task-name "${pattern}")
done

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "==> Running: ${CMD[*]}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  exit 0
fi

set +e
"${CMD[@]}"
HARBOR_RC=$?
set -e

JOB_DIR="${JOBS_DIR}/${JOB_NAME}"
if [[ -d "${JOB_DIR}" ]]; then
  EXPORT_CMD=(
    python3 "${SCRIPT_DIR}/export_terminalbench_results.py"
    "${JOB_DIR}"
    --results-dir "${RESULTS_DIR}"
    --dataset "${DATASET}"
    --execution-mode "${EXECUTION_MODE}"
    --harbor-env "${ENV_BACKEND}"
    --registry-url "${REGISTRY_URL}"
    --n-concurrent "${N_CONCURRENT}"
    --attempts "${ATTEMPTS}"
    --timeout-multiplier "${TIMEOUT_MULT}"
    --binary-path "${BINARY_PATH}"
  )

  if [[ -f "${HOME}/.lash/config.json" ]]; then
    EXPORT_CMD+=(--provider-config "${HOME}/.lash/config.json")
  fi
  if [[ -n "${MODEL}" ]]; then
    EXPORT_CMD+=(--requested-model "${MODEL}")
  fi
  if [[ -n "${VARIANT}" ]]; then
    EXPORT_CMD+=(--variant "${VARIANT}")
  fi
  if [[ "${DELETE_AFTER_RUN}" -eq 1 ]]; then
    EXPORT_CMD+=(--delete-after-run)
  fi
  if [[ "${DEBUG}" -eq 1 ]]; then
    EXPORT_CMD+=(--debug)
  fi
  for pattern in "${TASK_PATTERNS[@]}"; do
    EXPORT_CMD+=(--task-pattern "${pattern}")
  done
  for task_name in "${EXACT_TASKS[@]}"; do
    EXPORT_CMD+=(--exact-task "${task_name}")
  done
  for pattern in "${EXCLUDE_PATTERNS[@]}"; do
    EXPORT_CMD+=(--exclude-pattern "${pattern}")
  done
  for arg in "${EXTRA_ARGS[@]}"; do
    EXPORT_CMD+=(--extra-arg "${arg}")
  done

  "${EXPORT_CMD[@]}" || true
  python3 "${SCRIPT_DIR}/summarize_terminalbench.py" "${JOB_DIR}" || true
  echo
  echo "Structured results: ${RESULTS_DIR}"
  echo "Browse them with: python3 ${SCRIPT_DIR}/bench_ui.py --results-dir ${RESULTS_DIR}"
fi
exit "${HARBOR_RC}"
