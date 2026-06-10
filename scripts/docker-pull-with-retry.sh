#!/usr/bin/env bash
set -euo pipefail

image="${1:?usage: scripts/docker-pull-with-retry.sh IMAGE [attempts] [base_delay_seconds]}"
attempts="${2:-3}"
base_delay="${3:-10}"

for ((attempt = 1; attempt <= attempts; attempt++)); do
  if docker pull "$image"; then
    exit 0
  fi

  if ((attempt == attempts)); then
    echo "Failed to pull Docker image $image after $attempts attempts" >&2
    exit 1
  fi

  delay=$((base_delay * attempt))
  echo "Docker pull failed for $image; retrying in ${delay}s ($((attempt + 1))/$attempts)" >&2
  sleep "$delay"
done
