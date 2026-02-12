#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Load .env if present
if [ -f .env ]; then
  set -a; source .env; set +a
fi

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "OPENROUTER_API_KEY not set. Add it to .env or export it."
  exit 1
fi

MODEL="${1:-google/gemini-3-flash-preview}"

# cargo run always rebuilds if sources changed — guarantees latest binary.
case "${1:-}" in
  test)
    exec cargo run -p kaml-demo --bin test-session
    ;;
  *)
    exec cargo run -p kaml-demo --bin kaml-demo -- --model "$MODEL"
    ;;
esac
