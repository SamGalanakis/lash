#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: bench/oolong/evaluate.sh <run-dir-or-predictions.jsonl>" >&2
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

total = correct = failed = 0
by_task = {}
by_dataset = {}
for line in target.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    total += 1
    ok = bool(row.get("correct"))
    correct += int(ok)
    failed += int(row.get("status") != "completed")
    for table, key in [
        (by_task, row.get("task_group") or "unknown"),
        (by_dataset, row.get("dataset") or "unknown"),
    ]:
        bucket = table.setdefault(key, [0, 0])
        bucket[0] += 1
        bucket[1] += int(ok)

def emit_table(name, table):
    print(name)
    for key, (count, corr) in sorted(table.items()):
        acc = corr / count if count else 0.0
        print(f"  {key}: {corr}/{count} = {acc:.3f}")

acc = correct / total if total else 0.0
print(f"predictions: {target}")
print(f"correct:     {correct}/{total} = {acc:.3f}")
print(f"failed:      {failed}")
emit_table("by_task_group:", by_task)
emit_table("by_dataset:", by_dataset)
PY
