#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

cargo build --release --manifest-path "${REPO_ROOT}/Cargo.toml" --bin lash

exec python3 "${SCRIPT_DIR}/run.py" "$@"
