#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
  exit 0
fi

for attempt in 1 2 3; do
  if sudo apt-get update && sudo apt-get install -y --fix-missing "$@"; then
    exit 0
  fi

  if [ "$attempt" -lt 3 ]; then
    sleep $((attempt * 10))
  fi
done

exit 1
