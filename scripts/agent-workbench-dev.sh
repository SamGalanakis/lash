#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

log() {
  printf '[agent-workbench] %s\n' "$*" >&2
}

die() {
  printf '[agent-workbench] error: %s\n' "$*" >&2
  exit 1
}

url_host_port() {
  local url="${1#*://}"
  url="${url%%/*}"
  local host="${url%:*}"
  local port="${url##*:}"
  if [[ -z "$host" || -z "$port" || "$host" = "$port" ]]; then
    die "expected URL with explicit host and port, got '$1'"
  fi
  printf '%s %s\n' "$host" "$port"
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

wait_tcp() {
  local label="$1"
  local host="$2"
  local port="$3"
  local deadline=$((SECONDS + ${4:-60}))
  until tcp_ready "$host" "$port"; do
    if (( SECONDS >= deadline )); then
      return 1
    fi
    sleep 1
  done
}

json_string() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  printf '"%s"' "$value"
}

register_deployment() {
  local admin_url="$1"
  local endpoint_url="$2"
  local payload
  payload="$(printf '{"uri":%s,"force":true,"breaking":true}' "$(json_string "$endpoint_url")")"
  local deadline=$((SECONDS + 60))
  local last_response=""
  until last_response="$(
    curl --http2-prior-knowledge -fsS \
      -H 'content-type: application/json' \
      -X POST \
      --data "$payload" \
      "${admin_url%/}/deployments" 2>&1
  )"; do
    if (( SECONDS >= deadline )); then
      printf '%s\n' "$last_response" >&2
      die "failed to register Restate deployment $endpoint_url through $admin_url"
    fi
    sleep 1
  done
}

open_browser() {
  local url="$1"
  case "${AGENT_WORKBENCH_OPEN:-1}" in
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

workbench_addr="${AGENT_WORKBENCH_ADDR:-127.0.0.1:3030}"
restate_endpoint_addr="${AGENT_WORKBENCH_RESTATE_ADDR:-127.0.0.1:9081}"
restate_ingress_url="${RESTATE_INGRESS_URL:-http://127.0.0.1:8080}"
restate_admin_url="${RESTATE_ADMIN_URL:-http://127.0.0.1:${AGENT_WORKBENCH_RESTATE_ADMIN_PORT:-19070}}"
restate_image="${AGENT_WORKBENCH_RESTATE_IMAGE:-restatedev/restate:1.6.2}"
restate_container="${AGENT_WORKBENCH_RESTATE_CONTAINER:-lash-agent-workbench-dev-restate}"
restate_node_port="${AGENT_WORKBENCH_RESTATE_NODE_PORT:-19071}"
configured_endpoint_url="${AGENT_WORKBENCH_RESTATE_ENDPOINT_URL:-}"

read -r workbench_host workbench_port < <(addr_host_port "$workbench_addr")
read -r endpoint_host endpoint_port < <(addr_host_port "$restate_endpoint_addr")
read -r ingress_host ingress_port < <(url_host_port "$restate_ingress_url")
read -r admin_host admin_port < <(url_host_port "$restate_admin_url")
workbench_wait_host="$workbench_host"
endpoint_wait_host="$endpoint_host"
if [[ "$workbench_wait_host" = "0.0.0.0" ]]; then
  workbench_wait_host="127.0.0.1"
fi
if [[ "$endpoint_wait_host" = "0.0.0.0" ]]; then
  endpoint_wait_host="127.0.0.1"
fi

started_restate=0
workbench_pid=""

cleanup() {
  if [[ -n "$workbench_pid" ]]; then
    kill "$workbench_pid" >/dev/null 2>&1 || true
    wait "$workbench_pid" >/dev/null 2>&1 || true
  fi
  if (( started_restate )); then
    docker rm -f "$restate_container" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

if tcp_ready "$workbench_wait_host" "$workbench_port"; then
  die "workbench UI port $workbench_host:$workbench_port is already in use"
fi
if tcp_ready "$endpoint_wait_host" "$endpoint_port"; then
  die "workbench Restate endpoint port $endpoint_host:$endpoint_port is already in use"
fi

if tcp_ready "$ingress_host" "$ingress_port" && tcp_ready "$admin_host" "$admin_port"; then
  log "using existing Restate at ingress=$restate_ingress_url admin=$restate_admin_url"
else
  command -v docker >/dev/null 2>&1 || die "Restate is not running and docker is unavailable"
  log "starting Restate container $restate_container from $restate_image"
  docker rm -f "$restate_container" >/dev/null 2>&1 || true
  docker run -d --rm \
    --name "$restate_container" \
    --network host \
    -e RESTATE_INGRESS__BIND_PORT="$ingress_port" \
    -e RESTATE_ADMIN__BIND_PORT="$admin_port" \
    -e RESTATE_BIND_PORT="$restate_node_port" \
    "$restate_image" >/dev/null
  started_restate=1
fi

if ! wait_tcp "Restate ingress" "$ingress_host" "$ingress_port" 60; then
  (( started_restate )) && docker logs "$restate_container" >&2 || true
  die "Restate ingress did not become ready at $restate_ingress_url"
fi
if ! wait_tcp "Restate admin" "$admin_host" "$admin_port" 60; then
  (( started_restate )) && docker logs "$restate_container" >&2 || true
  die "Restate admin did not become ready at $restate_admin_url"
fi

if [[ -n "$configured_endpoint_url" ]]; then
  endpoint_url="$configured_endpoint_url"
elif (( started_restate )); then
  endpoint_url="http://127.0.0.1:$endpoint_port"
elif [[ "$endpoint_host" = "0.0.0.0" ]]; then
  endpoint_url="http://127.0.0.1:$endpoint_port"
else
  endpoint_url="http://$endpoint_host:$endpoint_port"
fi

log "starting workbench at http://$workbench_addr"
AGENT_WORKBENCH_ADDR="$workbench_addr" \
AGENT_WORKBENCH_RESTATE_ADDR="$restate_endpoint_addr" \
RESTATE_INGRESS_URL="$restate_ingress_url" \
cargo run -p agent-workbench &
workbench_pid="$!"

if ! wait_tcp "workbench UI" "$workbench_wait_host" "$workbench_port" 90; then
  die "workbench UI did not become ready at http://$workbench_addr"
fi
if ! wait_tcp "workbench Restate endpoint" "$endpoint_wait_host" "$endpoint_port" 90; then
  die "workbench Restate endpoint did not become ready at $restate_endpoint_addr"
fi

log "registering Restate deployment $endpoint_url"
register_deployment "$restate_admin_url" "$endpoint_url"

log "ready: http://$workbench_addr"
open_browser "http://$workbench_addr"
wait "$workbench_pid"
