#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Load .env if present
if [ -f .env ]; then
  set -a; source .env; set +a
fi

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

# cargo run always rebuilds if sources changed — guarantees latest binary.
exec cargo run -p lash-cli --bin lash -- "$@"
