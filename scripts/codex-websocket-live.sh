#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"

source_lash_home="${LASH_CODEX_SOURCE_LASH_HOME:-${LASH_HOME:-$HOME/.lash}}"
source_config="${LASH_CODEX_CONFIG:-$source_lash_home/config.json}"
model="${LASH_CODEX_LIVE_MODEL:-gpt-5.5}"
variant="${LASH_CODEX_LIVE_VARIANT:-high}"
prompt="${LASH_CODEX_LIVE_PROMPT:-Reply with exactly: codex websocket ok}"
out_dir="${LASH_CODEX_LIVE_OUT_DIR:-$repo/target/codex-websocket-live}"
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
out_file="$out_dir/${run_id}.jsonl"

if [[ ! -f "$source_config" ]]; then
  echo "missing Lash config: $source_config" >&2
  echo "Run lash provider setup for Codex first, then retry." >&2
  exit 2
fi

tmp_home="$(mktemp -d)"
cleanup() {
  rm -rf "$tmp_home"
}
trap cleanup EXIT

mkdir -p "$out_dir"
cp "$source_config" "$tmp_home/config.json"

python3 - "$tmp_home/config.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
config = json.loads(path.read_text())
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
codex_config["transport"] = "websocket"
path.write_text(json.dumps(config, indent=2) + "\n")
PY

args=(
  --mode json
  --execution-mode standard
  --model "$model"
  --print "$prompt"
)
if [[ -n "$variant" ]]; then
  args+=(--variant "$variant")
fi

echo "Running Codex websocket live smoke. Output: $out_file" >&2
echo "Model: $model${variant:+ / $variant}" >&2

LASH_HOME="$tmp_home" \
  cargo run -q -p lash-cli -- "${args[@]}" | tee "$out_file"

python3 - "$out_file" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
finish = None
errors = []
for line in path.read_text().splitlines():
    if not line.strip():
        continue
    record = json.loads(line)
    if record.get("type") == "turn_finish":
        finish = record
    if record.get("type") == "event":
        activity = record.get("activity") or {}
        if activity.get("type") == "error":
            errors.append(activity.get("message") or activity)
if finish is None:
    raise SystemExit("Codex websocket live smoke did not emit turn_finish")
if not finish.get("ok"):
    raise SystemExit(f"Codex websocket live smoke failed: {errors or finish}")
PY

echo "Codex websocket live smoke passed." >&2
