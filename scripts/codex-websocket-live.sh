#!/usr/bin/env bash
set -euo pipefail

# Credentialed live probe. This is intentionally opt-in and is not part of
# normal CI; it copies local Codex auth into a temporary LASH_HOME.

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"

source_lash_home="${LASH_CODEX_SOURCE_LASH_HOME:-${LASH_HOME:-$HOME/.lash}}"
source_config="${LASH_CODEX_CONFIG:-$source_lash_home/config.json}"
model="${LASH_CODEX_LIVE_MODEL:-gpt-5.5}"
variant="${LASH_CODEX_LIVE_VARIANT:-high}"
prompt_one="${LASH_CODEX_LIVE_PROMPT_ONE:-Reply with exactly: codex websocket one}"
prompt_two="${LASH_CODEX_LIVE_PROMPT_TWO:-Reply with exactly: codex websocket two}"
transports="${LASH_CODEX_LIVE_TRANSPORTS:-websocket_cached auto}"
out_dir="${LASH_CODEX_LIVE_OUT_DIR:-$repo/target/codex-websocket-live}"
run_id="$(date -u +%Y%m%dT%H%M%SZ)"

if [[ ! -f "$source_config" ]]; then
  echo "missing Lash config: $source_config" >&2
  echo "Run lash provider setup for Codex first, then retry." >&2
  exit 2
fi

mkdir -p "$out_dir"

write_config() {
  local target_config="$1"
  local transport="$2"
  python3 - "$source_config" "$target_config" "$transport" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
transport = sys.argv[3]
config = json.loads(source.read_text())
providers = config.get("providers") or {}
codex = providers.get("codex")
if not isinstance(codex, dict):
    raise SystemExit("stored Lash config has no `codex` provider")
codex_config = codex.get("config") if isinstance(codex.get("config"), dict) else codex
if not isinstance(codex_config, dict):
    raise SystemExit("stored `codex` provider config is malformed")
for key in ("access_token", "refresh_token", "expires_at"):
    if key not in codex_config:
        raise SystemExit(f"stored `codex` provider is missing `{key}`")
config["active_provider"] = "codex"
codex_config["transport"] = transport
target.write_text(json.dumps(config, indent=2) + "\n")
PY
}

run_transport() {
  local transport="$1"
  local tmp_home
  tmp_home="$(mktemp -d)"
  local out_file="$out_dir/${run_id}-${transport}.jsonl"
  local request_file="$out_dir/${run_id}-${transport}.requests.jsonl"
  local trace_copy="$out_dir/${run_id}-${transport}.trace.jsonl"
  local summary_file="$out_dir/${run_id}-${transport}.summary.json"
  trap 'rm -rf "$tmp_home"' RETURN

  mkdir -p "$tmp_home"
  write_config "$tmp_home/config.json" "$transport"

  python3 - "$request_file" "$prompt_one" "$prompt_two" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
prompt_one = sys.argv[2]
prompt_two = sys.argv[3]
records = [
    {"id": "turn1", "method": "prompt", "params": {"prompt": prompt_one}},
    {"id": "turn2", "method": "prompt", "params": {"prompt": prompt_two}},
    {"id": "shutdown", "method": "shutdown", "params": {}},
]
path.write_text("".join(json.dumps(record) + "\n" for record in records))
PY

  local args=(
    --mode rpc
    --trace-level extended
    --execution-mode standard
    --model "$model"
  )
  if [[ -n "$variant" ]]; then
    args+=(--variant "$variant")
  fi

  echo "Running Codex websocket live multi-turn probe. Transport: $transport" >&2
  echo "Output: $out_file" >&2
  LASH_HOME="$tmp_home" \
    cargo run -q -p lash-cli -- "${args[@]}" < "$request_file" | tee "$out_file"

  local trace_file
  trace_file="$(find "$tmp_home/sessions" -maxdepth 1 -name '*.trace.jsonl' -print -quit 2>/dev/null || true)"
  if [[ -n "$trace_file" ]]; then
    cp "$trace_file" "$trace_copy"
  fi

  python3 - "$out_file" "${trace_file:-}" "$transport" "$summary_file" "$request_file" "$trace_copy" <<'PY'
import json
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
trace_arg = sys.argv[2]
transport = sys.argv[3]
summary_path = Path(sys.argv[4])
request_path = Path(sys.argv[5])
trace_copy_path = Path(sys.argv[6])

turn_finish = {}
responses = {}
errors = []
for line in out_path.read_text().splitlines():
    if not line.strip():
        continue
    record = json.loads(line)
    if record.get("type") == "turn_finish":
        turn_finish[record.get("id")] = record
    elif record.get("type") == "response":
        responses[record.get("id")] = record
    elif record.get("type") == "event":
        activity = record.get("activity") or {}
        if activity.get("type") == "error":
            errors.append(activity.get("message") or activity)

for turn_id in ("turn1", "turn2"):
    finish = turn_finish.get(turn_id)
    if finish is None:
        raise SystemExit(f"{transport}: missing turn_finish for {turn_id}")
    if not finish.get("ok"):
        raise SystemExit(f"{transport}: {turn_id} failed: {errors or finish}")
    response = responses.get(turn_id)
    if response is None or not response.get("ok"):
        raise SystemExit(f"{transport}: missing ok RPC response for {turn_id}: {response}")

if not trace_arg:
    raise SystemExit(f"{transport}: missing extended trace file")

trace_path = Path(trace_arg)
diagnostics = []
for line in trace_path.read_text().splitlines():
    if "lash.codex.websocket_request" not in line:
        continue
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        continue
    payload = record
    stack = [payload]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            if current.get("type") == "lash.codex.websocket_request":
                diagnostics.append(current)
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
if len(diagnostics) < 2:
    raise SystemExit(f"{transport}: expected at least two websocket diagnostics in {trace_path}")
required_keys = {
    "reused_connection",
    "cached_request",
    "retry_after_stale_previous_response",
    "retry_after_dead_reused_connection",
}
for index, item in enumerate(diagnostics):
    missing = sorted(required_keys.difference(item))
    if missing:
        raise SystemExit(f"{transport}: websocket diagnostic {index} missing keys: {missing}")

followups = diagnostics[1:]
if transport in {"websocket_cached", "auto"}:
    if not any(item.get("reused_connection") is True for item in followups):
        raise SystemExit(f"{transport}: second turn did not reuse the websocket connection")
    if any(item.get("cached_request") is True for item in followups):
        if not any(item.get("previous_response_id") for item in followups):
            raise SystemExit(f"{transport}: cached websocket request did not expose previous_response_id")
    else:
        if not any(item.get("continuation_available") is True for item in followups):
            raise SystemExit(f"{transport}: second turn had no continuation metadata")
        reasons = sorted({str(item.get("cache_miss_reason")) for item in followups})
        print(f"{transport}: websocket reused; delta not sent because {', '.join(reasons)}", file=sys.stderr)
    if any(item.get("retry_after_stale_previous_response") is True for item in followups):
        if not any(item.get("cached_request") is False for item in followups):
            raise SystemExit(f"{transport}: stale retry did not produce a full-context follow-up")

summary = {
    "schema": "lash.codex.websocket_live_probe.v1",
    "transport": transport,
    "files": {
        "rpc_output": str(out_path),
        "requests": str(request_path),
        "trace": str(trace_copy_path),
    },
    "turns": {
        turn_id: {
            "ok": bool(turn_finish[turn_id].get("ok")),
            "assistant_text": (responses[turn_id].get("result") or {}).get("assistant_text"),
            "usage": turn_finish[turn_id].get("usage"),
        }
        for turn_id in ("turn1", "turn2")
    },
    "diagnostics": {
        "count": len(diagnostics),
        "followup_reused_connection": any(item.get("reused_connection") is True for item in followups),
        "followup_cached_request": any(item.get("cached_request") is True for item in followups),
        "followup_cache_miss_reasons": sorted({
            str(item.get("cache_miss_reason"))
            for item in followups
            if item.get("cache_miss_reason") is not None
        }),
        "stale_retry_observed": any(
            item.get("retry_after_stale_previous_response") is True for item in followups
        ),
        "dead_reused_retry_observed": any(
            item.get("retry_after_dead_reused_connection") is True for item in followups
        ),
    },
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PY

  echo "Summary: $summary_file" >&2
  echo "Codex websocket live multi-turn probe passed for $transport." >&2
}

for transport in $transports; do
  run_transport "$transport"
done
