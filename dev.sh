#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Load .env if present
if [ -f .env ]; then
  set -a; source .env; set +a
fi

mode="${LASH_PYTHON_MODE:-system}"
features=(--features python-system)

if [ "$mode" = "bundled" ]; then
  # Bootstrap: fetch python-build-standalone if it hasn't been set up yet
  config="target/python-standalone/pyo3-config.txt"
  if [ ! -f "$config" ]; then
    echo "Python standalone not found — bootstrapping via scripts/fetch-python.sh ..."
    ./scripts/fetch-python.sh
  fi

  # Point PyO3 at the standalone Python for static linking
  if [ -z "${PYO3_CONFIG_FILE:-}" ]; then
    export PYO3_CONFIG_FILE="$PWD/$config"
  fi
  features=(--features python-bundled)
elif [ "$mode" != "system" ]; then
  echo "Invalid LASH_PYTHON_MODE=$mode (expected 'system' or 'bundled')" >&2
  exit 1
fi

# cargo run always rebuilds if sources changed — guarantees latest binary.
exec cargo run -p lash-cli --bin lash "${features[@]}" -- "$@"
