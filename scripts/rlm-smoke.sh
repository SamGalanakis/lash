#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/rlm-smoke.sh [options] <scenario-dir>

Prepare an isolated RLM smoke workspace and start the real Lash TUI harness.
This script does not send the prompt, judge the run, or run the oracle. The
operator drives the harness with commands on stdin and inspects artifacts/files.

Options:
  --isolation docker|host   Isolation backend. Default: docker.
  --host                    Shortcut for --isolation host.
  --auth-home PATH          Source Lash home for provider config. Default:
                            $LASH_RLM_SMOKE_AUTH_HOME, then $LASH_HOME, then ~/.lash.
  --out-dir PATH            Artifact root. Default: target/rlm-smoke/<scenario>-<timestamp>.
  --model MODEL             Override model passed to lash.
  --timeout-secs N          Harness command timeout. Default: 240.
  --no-build                Reuse target/debug binaries.
  -h, --help                Show this help.

Scenario layout:
  prompt.md                 Prompt for the operator to send manually.
  workspace/                Fixture copied into an isolated temp workspace.
  check.sh WORKSPACE SCENARIO_DIR
                            Optional manual oracle. Run it yourself after inspection.

Harness commands once started:
  send TEXT
  idle
  screen
  screenshot NAME
  log
  quit

Docker mode runs Lash inside ubuntu:24.04 with the repo mounted read-only and
only the copied scenario workspace and copied Lash home mounted writable.
USAGE
}

die() {
  echo "error: $*" >&2
  exit 2
}

repo_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}

copy_lash_home_seed() {
  local source_home="$1"
  local dest_home="$2"
  mkdir -p "$dest_home"
  if [ ! -f "$source_home/config.json" ]; then
    die "missing provider config: $source_home/config.json"
  fi
  cp -a "$source_home/config.json" "$dest_home/config.json"
  for name in host-id cache skills docs; do
    if [ -e "$source_home/$name" ]; then
      cp -a "$source_home/$name" "$dest_home/$name"
    fi
  done
}

prompt_one_line() {
  local prompt_path="$1"
  python3 - "$prompt_path" <<'PY'
import pathlib
import sys

text = pathlib.Path(sys.argv[1]).read_text()
print(" ".join(line.strip() for line in text.splitlines() if line.strip()))
PY
}

isolation="docker"
auth_home="${LASH_RLM_SMOKE_AUTH_HOME:-${LASH_HOME:-$HOME/.lash}}"
out_dir=""
model="${LASH_RLM_SMOKE_MODEL:-}"
timeout_secs="${LASH_RLM_SMOKE_TIMEOUT_SECS:-240}"
build=1
scenario_dir=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --isolation)
      [ "$#" -ge 2 ] || die "missing --isolation value"
      isolation="$2"
      shift 2
      ;;
    --host)
      isolation="host"
      shift
      ;;
    --auth-home)
      [ "$#" -ge 2 ] || die "missing --auth-home value"
      auth_home="$2"
      shift 2
      ;;
    --out-dir)
      [ "$#" -ge 2 ] || die "missing --out-dir value"
      out_dir="$2"
      shift 2
      ;;
    --model)
      [ "$#" -ge 2 ] || die "missing --model value"
      model="$2"
      shift 2
      ;;
    --timeout-secs)
      [ "$#" -ge 2 ] || die "missing --timeout-secs value"
      timeout_secs="$2"
      shift 2
      ;;
    --no-build)
      build=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      die "unknown option: $1"
      ;;
    *)
      if [ -n "$scenario_dir" ]; then
        die "multiple scenario dirs provided: $scenario_dir and $1"
      fi
      scenario_dir="$1"
      shift
      ;;
  esac
done

[ -n "$scenario_dir" ] || die "missing scenario dir"
case "$isolation" in
  docker|host) ;;
  *) die "unknown isolation backend: $isolation" ;;
esac
case "$timeout_secs" in
  ''|*[!0-9]*) die "--timeout-secs must be a positive integer" ;;
esac

repo="$(repo_root)"
scenario_dir="$(cd "$scenario_dir" && pwd)"
[ -f "$scenario_dir/prompt.md" ] || die "missing $scenario_dir/prompt.md"
[ -d "$scenario_dir/workspace" ] || die "missing $scenario_dir/workspace"

scenario_name="$(basename "$scenario_dir")"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
if [ -z "$out_dir" ]; then
  out_dir="$repo/target/rlm-smoke/${scenario_name}-${timestamp}"
fi
mkdir -p "$out_dir"
out_dir="$(cd "$out_dir" && pwd)"

workspace="$out_dir/workspace"
lash_home="$out_dir/lash-home"
artifact_dir="$out_dir/harness"
mkdir -p "$workspace" "$artifact_dir"
cp -a "$scenario_dir/workspace/." "$workspace/"
copy_lash_home_seed "$auth_home" "$lash_home"

if [ "$build" -eq 1 ]; then
  cargo build -p lash-cli -p lash-debug-cli-harness
fi

lash_bin="$repo/target/debug/lash"
harness_bin="$repo/target/debug/lash-debug-cli-harness"
[ -x "$lash_bin" ] || die "missing executable $lash_bin"
[ -x "$harness_bin" ] || die "missing executable $harness_bin"

model_args=()
if [ -n "$model" ]; then
  model_args=(--model "$model")
fi

prompt_line="$(prompt_one_line "$scenario_dir/prompt.md")"

cat >&2 <<EOF
RLM smoke ready
  scenario:     $scenario_dir
  isolation:    $isolation
  workspace:    $workspace
  lash_home:    $lash_home
  artifacts:    $artifact_dir
  prompt:       $scenario_dir/prompt.md
  send command: send $prompt_line
  manual check: bash $scenario_dir/check.sh $workspace $scenario_dir

Drive the harness on stdin. Suggested flow:
  send ...
  idle
  screen
  screenshot after-turn
  quit
EOF

if [ "$isolation" = "docker" ]; then
  command -v docker >/dev/null 2>&1 || die "docker isolation requested but docker is not installed"
  docker info >/dev/null 2>&1 || die "docker isolation requested but docker daemon is unavailable"
  docker_extra_mounts=()
  if [ -d /etc/ssl/certs ]; then
    docker_extra_mounts+=(-v /etc/ssl/certs:/etc/ssl/certs:ro)
  fi
  exec docker run --rm -i \
    --network bridge \
    --user "$(id -u):$(id -g)" \
    --workdir /scenario-workspace \
    -e HOME=/tmp \
    "${docker_extra_mounts[@]}" \
    -v "$repo:/repo:ro" \
    -v "$workspace:/scenario-workspace:rw" \
    -v "$lash_home:/lash-home:rw" \
    -v "$artifact_dir:/artifacts:rw" \
    ubuntu:24.04 \
    /repo/target/debug/lash-debug-cli-harness \
      --repo-root /repo \
      --lash-bin /repo/target/debug/lash \
      --execution-mode rlm \
      --lash-home /lash-home \
      --working-dir /scenario-workspace \
      --output-dir /artifacts \
      --timeout-secs "$timeout_secs" \
      --no-build \
      "${model_args[@]}"
fi

echo "warning: running in host isolation; the model can execute tools outside the temp workspace" >&2
exec "$harness_bin" \
  --repo-root "$repo" \
  --lash-bin "$lash_bin" \
  --execution-mode rlm \
  --lash-home "$lash_home" \
  --working-dir "$workspace" \
  --output-dir "$artifact_dir" \
  --timeout-secs "$timeout_secs" \
  --no-build \
  "${model_args[@]}"
