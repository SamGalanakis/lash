#!/usr/bin/env bash
set -euo pipefail

# Clones upstream SWE-bench (for the evaluator) and exports the SWE-bench
# Verified dataset to JSONL under `.benchmarks/swebench/`.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STATE_DIR="${REPO_ROOT}/.benchmarks/swebench"
VENDOR_DIR="${STATE_DIR}/vendor/SWE-bench"
VENV_DIR="${VENDOR_DIR}/.venv"
DATASET_JSONL="${STATE_DIR}/verified.jsonl"

mkdir -p "${STATE_DIR}/runs" "${STATE_DIR}/workspace"

if [[ ! -d "${VENDOR_DIR}" ]]; then
  git clone --depth 1 https://github.com/SWE-bench/SWE-bench.git "${VENDOR_DIR}"
else
  git -C "${VENDOR_DIR}" fetch --depth 1 origin main >/dev/null 2>&1 || true
  git -C "${VENDOR_DIR}" reset --hard origin/main >/dev/null 2>&1 || true
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to set up the SWE-bench evaluator (https://docs.astral.sh/uv/)." >&2
  echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

pushd "${VENDOR_DIR}" >/dev/null
uv sync --no-dev >/dev/null
popd >/dev/null

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "uv sync did not produce ${VENV_DIR}" >&2
  exit 1
fi

if [[ ! -f "${DATASET_JSONL}" ]]; then
  echo "Exporting SWE-bench Verified → ${DATASET_JSONL}"
  uv --project "${VENDOR_DIR}" run --no-dev python - "$DATASET_JSONL" <<'PY'
import json
import sys
from datasets import load_dataset

out_path = sys.argv[1]
ds = load_dataset("SWE-bench/SWE-bench_Verified", split="test")
with open(out_path, "w") as fh:
    for row in ds:
        fh.write(json.dumps(dict(row)) + "\n")
print(f"wrote {len(ds)} instances to {out_path}")
PY
else
  echo "Dataset already present at ${DATASET_JSONL}"
fi

echo
echo "SWE-bench benchmark state ready under ${STATE_DIR}"
echo "  vendor:   ${VENDOR_DIR}"
echo "  venv:     ${VENV_DIR}"
echo "  dataset:  ${DATASET_JSONL}"
echo "  workspace:${STATE_DIR}/workspace   (shared repo clones)"
