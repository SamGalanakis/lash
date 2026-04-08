#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$repo_root/.tmp"

if [ "$#" -eq 0 ]; then
  echo "rustc-wrapper.sh: missing rustc command" >&2
  exit 1
fi

exec "$@"
