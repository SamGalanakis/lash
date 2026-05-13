#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: bench/frontier-cs/evaluate.sh <run-dir-or-predictions.jsonl>" >&2
  exit 1
fi

python3 - <<'PY' "$1"
import json
import sys
from pathlib import Path

target = Path(sys.argv[1]).resolve()
if target.is_dir():
    target = target / "predictions.jsonl"
if not target.exists():
    raise SystemExit(f"missing predictions file: {target}")

total = ok = evaluated = 0
score = 0.0
by_track = {}
for line in target.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    total += 1
    evaluated += int(row.get("evaluation_status") == "success")
    value = row.get("score")
    if isinstance(value, (int, float)):
        score += float(value)
    passed = bool(row.get("successful"))
    ok += int(passed)
    bucket = by_track.setdefault(row.get("track") or "unknown", [0, 0, 0.0])
    bucket[0] += 1
    bucket[1] += int(passed)
    if isinstance(value, (int, float)):
        bucket[2] += float(value)

print(f"predictions: {target}")
print(f"successful:  {ok}/{total}")
print(f"evaluated:   {evaluated}/{total}")
print(f"avg_score:   {(score / total if total else 0.0):.4f}")
print("by_track:")
for track, (count, successes, track_score) in sorted(by_track.items()):
    print(f"  {track}: successful={successes}/{count} avg_score={(track_score / count if count else 0.0):.4f}")
PY
