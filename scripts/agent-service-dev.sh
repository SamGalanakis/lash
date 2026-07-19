#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Load repo-local env (OPENROUTER_API_KEY etc.) so nobody has to remember
# `source .env`.
if [[ -f "$repo_root/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$repo_root/.env"
  set +a
fi

state_dir="${AGENT_SERVICE_RUN_DIR:-.agent-service/run}"
mkdir -p "$state_dir"

log() {
  printf '[agent-service] %s\n' "$*" >&2
}

die() {
  printf '[agent-service] error: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'USAGE'
Usage:
  scripts/agent-service-dev.sh [up] [--port PORT | --addr HOST:PORT]
  scripts/agent-service-dev.sh foreground [--port PORT | --addr HOST:PORT]
  scripts/agent-service-dev.sh restart [--port PORT | --addr HOST:PORT]
  scripts/agent-service-dev.sh status [--port PORT | --addr HOST:PORT]
  scripts/agent-service-dev.sh logs [--port PORT | --addr HOST:PORT] [-f]
  scripts/agent-service-dev.sh down [--port PORT | --addr HOST:PORT]

Defaults:
  up is detached and idempotent; local durability (no Restate needed).
  --port PORT binds 127.0.0.1:PORT.
  Without --port/--addr, AGENT_SERVICE_ADDR is used, then 127.0.0.1:3000.
  Ports already taken are reclaimed by killing the listener
  (AGENT_SERVICE_KILL_PORT=0 to die instead).
USAGE
}

addr_host_port() {
  local addr="$1"
  local host="${addr%:*}"
  local port="${addr##*:}"
  if [[ -z "$host" || -z "$port" || "$host" = "$port" ]]; then
    die "expected address as host:port, got '$addr'"
  fi
  printf '%s %s\n' "$host" "$port"
}

tcp_ready() {
  local host="$1"
  local port="$2"
  timeout 1 bash -c "cat < /dev/null > /dev/tcp/$host/$port" >/dev/null 2>&1
}

open_browser() {
  local url="$1"
  case "${AGENT_SERVICE_OPEN:-1}" in
    0|false|False|FALSE|no|No|NO)
      return
      ;;
  esac
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$url" >/dev/null 2>&1 || true
  elif command -v open >/dev/null 2>&1; then
    open "$url" >/dev/null 2>&1 || true
  fi
}

port_owner() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"$port" -sTCP:LISTEN || true
  elif command -v ss >/dev/null 2>&1; then
    ss -ltnp 2>/dev/null | grep -E "[:.]$port[[:space:]]" || true
  fi
}

port_listener_pids() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -tnP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null || true
  elif command -v ss >/dev/null 2>&1; then
    ss -ltnp 2>/dev/null | grep -E "[:.]$port[[:space:]]" \
      | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u || true
  fi
}

kill_port_listeners() {
  local port="$1"
  case "${AGENT_SERVICE_KILL_PORT:-1}" in
    0|false|False|FALSE|no|No|NO)
      return 1
      ;;
  esac
  local pids
  pids="$(port_listener_pids "$port")"
  [[ -n "$pids" ]] || return 1
  log "port $port is taken; killing listener(s): $(echo $pids | tr '\n' ' ')"
  local pid
  for pid in $pids; do
    kill "$pid" >/dev/null 2>&1 || true
  done
  for _ in {1..20}; do
    pids="$(port_listener_pids "$port")"
    [[ -z "$pids" ]] && return 0
    sleep 0.5
  done
  for pid in $pids; do
    log "listener $pid did not exit; sending SIGKILL"
    kill -KILL "$pid" >/dev/null 2>&1 || true
  done
  for _ in {1..10}; do
    [[ -z "$(port_listener_pids "$port")" ]] && return 0
    sleep 0.5
  done
  return 1
}

pid_alive() {
  local pid="${1:-}"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" >/dev/null 2>&1
}

read_pid_file() {
  local file="$1"
  [[ -f "$file" ]] || return 1
  local pid
  pid="$(tr -d '[:space:]' < "$file")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "$pid"
}

tail_log() {
  if [[ -f "$log_file" ]]; then
    tail -n "${AGENT_SERVICE_LOG_LINES:-120}" "$log_file" >&2 || true
  else
    log "no log file yet at $log_file"
  fi
}

service_ready() {
  tcp_ready "$service_wait_host" "$service_port"
}

cleanup_stale_pid() {
  local pid=""
  pid="$(read_pid_file "$pid_file" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && pid_alive "$pid"; then
    return
  fi
  rm -f "$pid_file"
}

stop_pid_file() {
  local file="$1"
  local pid=""
  pid="$(read_pid_file "$file" 2>/dev/null || true)"
  if [[ -z "$pid" ]]; then
    rm -f "$file"
    return
  fi

  if pid_alive "$pid"; then
    log "stopping process $pid"
    kill "-$pid" >/dev/null 2>&1 || kill "$pid" >/dev/null 2>&1 || true
    for _ in {1..30}; do
      pid_alive "$pid" || break
      sleep 0.5
    done
    if pid_alive "$pid"; then
      log "process $pid did not exit; sending SIGKILL"
      kill -KILL "-$pid" >/dev/null 2>&1 || kill -KILL "$pid" >/dev/null 2>&1 || true
    fi
  fi

  rm -f "$file"
}

stop_all_known() {
  local found=0
  local file
  for file in "$state_dir"/service-*.pid; do
    [[ -e "$file" ]] || continue
    found=1
    stop_pid_file "$file"
  done
  if (( ! found )); then
    log "no managed agent-service processes found"
  fi
}

build_sha() {
  local sha
  sha="$(git -C "$repo_root" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  if ! git -C "$repo_root" diff --quiet 2>/dev/null || ! git -C "$repo_root" diff --cached --quiet 2>/dev/null; then
    sha="${sha}-dirty"
  fi
  printf '%s\n' "$sha"
}

require_api_key() {
  [[ -n "${OPENROUTER_API_KEY:-}" ]] \
    || die "OPENROUTER_API_KEY is not set (put it in $repo_root/.env)"
}

ensure_port_available() {
  cleanup_stale_pid
  local pid=""
  pid="$(read_pid_file "$pid_file" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && service_ready; then
    log "already ready: $service_url"
    return 1
  fi
  if tcp_ready "$service_wait_host" "$service_port"; then
    port_owner "$service_port" >&2
    kill_port_listeners "$service_port" \
      || die "agent-service port $service_host:$service_port is already in use (set AGENT_SERVICE_KILL_PORT=1 to reclaim)"
  fi
}

build_service() {
  log "building agent-service"
  cargo build -p agent-service
  log "built agent-service at commit $(build_sha)"
}

wait_service_ready() {
  local timeout_seconds="${1:-90}"
  local deadline=$((SECONDS + timeout_seconds))
  local pid
  pid="$(read_pid_file "$pid_file" 2>/dev/null || true)"
  until service_ready; do
    if [[ -n "$pid" ]] && ! pid_alive "$pid"; then
      tail_log
      die "agent-service process $pid exited before becoming ready"
    fi
    if (( SECONDS >= deadline )); then
      tail_log
      stop_pid_file "$pid_file"
      die "agent-service did not become ready at $service_url"
    fi
    sleep 1
  done
}

run_up() {
  require_api_key
  if ! ensure_port_available; then
    return
  fi
  build_service

  # Launch the binary cargo just built: honor CARGO_TARGET_DIR, or a stale
  # binary in the repo-local target/ boots instead of the fresh build.
  local service_bin="${CARGO_TARGET_DIR:-$repo_root/target}/debug/agent-service"
  printf '\n[%s] starting agent-service (commit %s) at %s\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$(build_sha)" "$service_url" >> "$log_file"
  if command -v setsid >/dev/null 2>&1; then
    setsid env AGENT_SERVICE_ADDR="$service_addr" \
      "$service_bin" >> "$log_file" 2>&1 < /dev/null &
  else
    nohup env AGENT_SERVICE_ADDR="$service_addr" \
      "$service_bin" >> "$log_file" 2>&1 < /dev/null &
  fi
  local pid="$!"
  printf '%s\n' "$pid" > "$pid_file"
  wait_service_ready 90
  log "ready: $service_url (pid $pid, log $log_file)"
  open_browser "$service_url"
}

run_foreground() {
  require_api_key
  if ! ensure_port_available; then
    return
  fi
  build_service
  log "starting agent-service at $service_url"
  AGENT_SERVICE_ADDR="$service_addr" cargo run -p agent-service
}

run_status() {
  cleanup_stale_pid
  local pid=""
  pid="$(read_pid_file "$pid_file" 2>/dev/null || true)"
  if service_ready; then
    if [[ -n "$pid" ]]; then
      log "running: $service_url (pid $pid, log $log_file)"
    else
      log "running: $service_url (unmanaged process)"
    fi
    return 0
  fi
  log "stopped: $service_url"
  return 1
}

run_logs() {
  [[ -f "$log_file" ]] || die "no log file found at $log_file"
  if (( follow_logs )); then
    tail -f "$log_file"
  else
    tail -n "${AGENT_SERVICE_LOG_LINES:-120}" "$log_file"
  fi
}

action="up"
if (($#)); then
  case "$1" in
    up|start|foreground|run|restart|status|logs|down|stop)
      action="$1"
      shift
      ;;
    help|-h|--help)
      usage
      exit 0
      ;;
  esac
fi

port_override=""
addr_override=""
follow_logs=0
while (($#)); do
  case "$1" in
    --port)
      [[ $# -ge 2 ]] || die "--port requires a value"
      port_override="$2"
      shift 2
      ;;
    --addr)
      [[ $# -ge 2 ]] || die "--addr requires a value"
      addr_override="$2"
      shift 2
      ;;
    -f|--follow)
      follow_logs=1
      shift
      ;;
    [0-9]*)
      port_override="$1"
      shift
      ;;
    help|-h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

if [[ -n "$addr_override" ]]; then
  service_addr="$addr_override"
elif [[ -n "$port_override" ]]; then
  service_addr="127.0.0.1:$port_override"
else
  service_addr="${AGENT_SERVICE_ADDR:-127.0.0.1:3000}"
fi

read -r service_host service_port < <(addr_host_port "$service_addr")
service_wait_host="$service_host"
if [[ "$service_wait_host" = "0.0.0.0" ]]; then
  service_wait_host="127.0.0.1"
fi
service_url="http://$service_wait_host:$service_port"

state_key="$(printf '%s' "$service_addr" | tr -c 'A-Za-z0-9_.-' '_')"
pid_file="$state_dir/service-$state_key.pid"
log_file="$state_dir/service-$state_key.log"

case "$action" in
  up|start)
    run_up
    ;;
  foreground|run)
    run_foreground
    ;;
  restart)
    stop_pid_file "$pid_file"
    run_up
    ;;
  status)
    run_status
    ;;
  logs)
    run_logs
    ;;
  down|stop)
    if [[ -n "$port_override" || -n "$addr_override" ]]; then
      stop_pid_file "$pid_file"
    else
      stop_all_known
    fi
    ;;
  *)
    die "unknown command: $action"
    ;;
esac
