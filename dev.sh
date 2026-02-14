#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Load .env if present
if [ -f .env ]; then
  set -a; source .env; set +a
fi

# Point PyO3 at the standalone Python for static linking
if [ -z "${PYO3_CONFIG_FILE:-}" ]; then
  config="target/python-standalone/pyo3-config.txt"
  if [ -f "$config" ]; then
    export PYO3_CONFIG_FILE="$PWD/$config"
  fi
fi

# cargo run always rebuilds if sources changed — guarantees latest binary.
exec cargo run -p lash-cli --bin lash -- "$@"
