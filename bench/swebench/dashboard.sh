#!/usr/bin/env bash
# Serve the benchmark dashboard pointed at the SWE-bench results tree.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

exec python3 "${REPO_ROOT}/scripts/bench_ui.py" \
  --results-dir "${REPO_ROOT}/.benchmarks/swebench" \
  "$@"
