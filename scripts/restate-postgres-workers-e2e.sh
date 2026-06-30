#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"

compose=(docker compose -f "$repo/e2e/restate-postgres-workers/docker-compose.yml")
minio_port="${LASH_E2E_MINIO_PORT:-19000}"

# Binaries are built on the host (sharing the normal cargo cache) and
# bind-mounted into the compose services; see docker-compose.yml for the
# glibc compatibility note.
cargo build --locked -p lash-restate-postgres-workers-e2e --bins
export LASH_E2E_BIN_DIR="${CARGO_TARGET_DIR:-$repo/target}/debug"

cleanup() {
  status=$?
  if [ "$status" -ne 0 ]; then
    "${compose[@]}" logs --no-color >&2 || true
  fi
  if [ "${LASH_E2E_KEEP:-0}" != "1" ]; then
    "${compose[@]}" down -v --remove-orphans >/dev/null 2>&1 || true
  fi
  exit "$status"
}
trap cleanup EXIT

"${compose[@]}" down -v --remove-orphans >/dev/null 2>&1 || true
# Several services share the host-binary runtime image. Pull it once with
# retries so a transient Docker Hub HEAD error doesn't fail compose startup.
bash scripts/docker-pull-with-retry.sh ubuntu:24.04
"${compose[@]}" up -d postgres minio minio-init restate mock-provider worker-a worker-b worker-proxy
deadline=$((SECONDS + 60))
while true; do
  minio_init_id="$("${compose[@]}" ps -a -q minio-init)"
  if [ -n "$minio_init_id" ]; then
    minio_init_status="$(docker inspect -f '{{.State.Status}}' "$minio_init_id")"
    if [ "$minio_init_status" = "exited" ]; then
      minio_init_exit="$(docker inspect -f '{{.State.ExitCode}}' "$minio_init_id")"
      if [ "$minio_init_exit" != "0" ]; then
        echo "minio-init exited with status $minio_init_exit" >&2
        exit 1
      fi
      break
    fi
  fi
  if ((SECONDS >= deadline)); then
    echo "minio-init did not complete before timeout" >&2
    exit 1
  fi
  sleep 1
done

LASH_MINIO_ENDPOINT="http://127.0.0.1:$minio_port" \
LASH_MINIO_BUCKET="lash-attachments" \
LASH_MINIO_REGION="us-east-1" \
LASH_MINIO_ACCESS_KEY="minioadmin" \
LASH_MINIO_SECRET_KEY="minioadmin" \
LASH_MINIO_PREFIX="conformance/restate-postgres-workers-$$" \
  cargo test --locked -p lash-s3-store -- --nocapture

"${compose[@]}" --profile runner run --rm runner
