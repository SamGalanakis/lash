#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: bench/longcot/evaluate.sh <run-dir-or-responses.jsonl> [-- extra run_eval.py args]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STATE_DIR="${REPO_ROOT}/.benchmarks/longcot"
VENDOR_DIR="${STATE_DIR}/vendor/longcot"

TARGET="$1"
shift || true

if [[ ! -d "${VENDOR_DIR}" ]]; then
  echo "LongCoT vendor repo missing; run bench/longcot/setup.sh first" >&2
  exit 1
fi

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  source "${REPO_ROOT}/.env"
  set +a
fi

RESPONSES_FILE="$(python3 - <<'PY' "${TARGET}"
import os, sys
from pathlib import Path

target = Path(sys.argv[1]).resolve()
if target.is_file():
    print(target)
    sys.exit(0)

if target.is_dir():
    responses_dir = target / "responses"
    if responses_dir.is_dir():
        files = sorted(responses_dir.glob("*.jsonl"))
        if files:
            print(files[0] if len(files) == 1 else "")
            if len(files) > 1:
                print("multiple responses JSONLs found under", responses_dir, file=sys.stderr)
                for p in files:
                    print(f"  {p}", file=sys.stderr)
                sys.exit(2)
            sys.exit(0)
    candidates = sorted(target.glob("*.jsonl"))
    if len(candidates) == 1:
        print(candidates[0])
        sys.exit(0)

print(f"could not find a responses JSONL under {target}", file=sys.stderr)
sys.exit(3)
PY
)"

if [[ -z "${RESPONSES_FILE}" ]]; then
  exit 1
fi

pushd "${VENDOR_DIR}" >/dev/null
uv run --no-dev python run_eval.py "${RESPONSES_FILE}" "$@"
popd >/dev/null
