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

MODEL="${1:-z-ai/glm-5}"

# cargo run always rebuilds if sources changed — guarantees latest binary.
exec cargo run -p lash-cli --bin lash -- --model "$MODEL"
