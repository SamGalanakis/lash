set positional-arguments

repo := justfile_directory()

default:
  @just --list

dev *args:
  cargo build --manifest-path "{{repo}}/crates/lash-cli/Cargo.toml"
  cd "${LASH_DEV_LAUNCH_CWD:-{{invocation_directory()}}}" && exec "{{repo}}/target/debug/lash" "$@"

agent-workbench port='3030':
  ./scripts/agent-workbench-dev.sh up --port "{{port}}"

agent-workbench-up port='3030':
  ./scripts/agent-workbench-dev.sh up --port "{{port}}"

agent-workbench-restart port='3030':
  ./scripts/agent-workbench-dev.sh restart --port "{{port}}"

agent-workbench-status port='3030':
  ./scripts/agent-workbench-dev.sh status --port "{{port}}"

agent-workbench-logs port='3030':
  ./scripts/agent-workbench-dev.sh logs --port "{{port}}"

agent-workbench-logs-follow port='3030':
  ./scripts/agent-workbench-dev.sh logs --port "{{port}}" --follow

agent-workbench-down port='3030':
  ./scripts/agent-workbench-dev.sh down --port "{{port}}"

agent-workbench-foreground port='3030':
  ./scripts/agent-workbench-dev.sh foreground --port "{{port}}"

agent-service-restate-e2e:
  #!/usr/bin/env bash
  set -euo pipefail
  image="${AGENT_SERVICE_RESTATE_IMAGE:-restatedev/restate:1.6.2}"
  container="${AGENT_SERVICE_RESTATE_CONTAINER:-lash-agent-service-restate-e2e}"
  admin_port="${RESTATE_ADMIN_PORT:-19070}"
  ingress_port="${RESTATE_INGRESS_PORT:-18080}"
  node_port="${RESTATE_NODE_PORT:-15122}"
  endpoint_bind="${AGENT_SERVICE_E2E_ENDPOINT_BIND:-127.0.0.1:19080}"
  endpoint_url="${AGENT_SERVICE_E2E_ENDPOINT_URL:-http://127.0.0.1:19080}"
  admin_url="${RESTATE_ADMIN_URL:-http://127.0.0.1:$admin_port}"
  ingress_url="${RESTATE_INGRESS_URL:-http://127.0.0.1:$ingress_port}"

  cleanup() {
    docker rm -f "$container" >/dev/null 2>&1 || true
  }
  trap cleanup EXIT
  cleanup

  bash "{{repo}}/scripts/docker-pull-with-retry.sh" "$image"

  docker run -d --name "$container" --network host \
    -e RESTATE_ADMIN__BIND_PORT="$admin_port" \
    -e RESTATE_INGRESS__BIND_PORT="$ingress_port" \
    -e RESTATE_BIND_PORT="$node_port" \
    "$image" >/dev/null

  deadline=$((SECONDS + 60))
  until (echo >"/dev/tcp/127.0.0.1/$admin_port") >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      docker logs "$container" >&2 || true
      echo "Restate admin port $admin_port did not become ready" >&2
      exit 1
    fi
    sleep 1
  done
  until (echo >"/dev/tcp/127.0.0.1/$ingress_port") >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      docker logs "$container" >&2 || true
      echo "Restate ingress port $ingress_port did not become ready" >&2
      exit 1
    fi
    sleep 1
  done

  RESTATE_INGRESS_URL="$ingress_url" \
  RESTATE_ADMIN_URL="$admin_url" \
  AGENT_SERVICE_E2E_ENDPOINT_BIND="$endpoint_bind" \
  AGENT_SERVICE_E2E_ENDPOINT_URL="$endpoint_url" \
  cargo test -p agent-service --features restate \
    live_restate_ingress_runs_agent_turn_and_process_workflow_end_to_end -- --ignored --nocapture

agent-workbench-restate-e2e:
  #!/usr/bin/env bash
  set -euo pipefail
  image="${AGENT_WORKBENCH_RESTATE_IMAGE:-restatedev/restate:1.6.2}"
  container="${AGENT_WORKBENCH_RESTATE_CONTAINER:-lash-agent-workbench-restate-e2e}"
  admin_port="${AGENT_WORKBENCH_RESTATE_ADMIN_PORT:-19071}"
  ingress_port="${AGENT_WORKBENCH_RESTATE_INGRESS_PORT:-18081}"
  node_port="${AGENT_WORKBENCH_RESTATE_NODE_PORT:-15123}"
  endpoint_bind="${AGENT_WORKBENCH_E2E_ENDPOINT_BIND:-127.0.0.1:19081}"
  endpoint_url="${AGENT_WORKBENCH_E2E_ENDPOINT_URL:-http://127.0.0.1:19081}"
  admin_url="${RESTATE_ADMIN_URL:-http://127.0.0.1:$admin_port}"
  ingress_url="${RESTATE_INGRESS_URL:-http://127.0.0.1:$ingress_port}"

  cleanup() {
    docker rm -f "$container" >/dev/null 2>&1 || true
  }
  trap cleanup EXIT
  cleanup

  bash "{{repo}}/scripts/docker-pull-with-retry.sh" "$image"

  docker run -d --name "$container" --network host \
    -e RESTATE_ADMIN__BIND_PORT="$admin_port" \
    -e RESTATE_INGRESS__BIND_PORT="$ingress_port" \
    -e RESTATE_BIND_PORT="$node_port" \
    "$image" >/dev/null

  deadline=$((SECONDS + 60))
  until (echo >"/dev/tcp/127.0.0.1/$admin_port") >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      docker logs "$container" >&2 || true
      echo "Restate admin port $admin_port did not become ready" >&2
      exit 1
    fi
    sleep 1
  done
  until (echo >"/dev/tcp/127.0.0.1/$ingress_port") >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      docker logs "$container" >&2 || true
      echo "Restate ingress port $ingress_port did not become ready" >&2
      exit 1
    fi
    sleep 1
  done

  RESTATE_INGRESS_URL="$ingress_url" \
  RESTATE_ADMIN_URL="$admin_url" \
  AGENT_WORKBENCH_E2E_ENDPOINT_BIND="$endpoint_bind" \
  AGENT_WORKBENCH_E2E_ENDPOINT_URL="$endpoint_url" \
  cargo test -p agent-workbench \
    live_restate_cron_runs_trigger_and_queued_turn_end_to_end -- --ignored --nocapture

restate-postgres-workers-e2e:
  #!/usr/bin/env bash
  set -euo pipefail
  compose="docker compose -f {{repo}}/e2e/restate-postgres-workers/docker-compose.yml"
  minio_port="${LASH_E2E_MINIO_PORT:-19000}"

  # Binaries are built on the host (sharing the normal cargo cache) and
  # bind-mounted into the compose services; see docker-compose.yml for the
  # glibc compatibility note.
  cargo build --locked -p lash-restate-postgres-workers-e2e --bins
  export LASH_E2E_BIN_DIR="${CARGO_TARGET_DIR:-{{repo}}/target}/debug"

  cleanup() {
    status=$?
    if [ "$status" -ne 0 ]; then
      $compose logs --no-color >&2 || true
    fi
    if [ "${LASH_E2E_KEEP:-0}" != "1" ]; then
      $compose down -v --remove-orphans >/dev/null 2>&1 || true
    fi
    exit "$status"
  }
  trap cleanup EXIT

  $compose down -v --remove-orphans >/dev/null 2>&1 || true
  $compose up -d postgres minio minio-init restate mock-provider worker-a worker-b worker-proxy
  deadline=$((SECONDS + 60))
  while true; do
    minio_init_id="$($compose ps -a -q minio-init)"
    if [ -n "$minio_init_id" ]; then
      minio_init_status="$(docker inspect -f '{{ "{{" }}.State.Status{{ "}}" }}' "$minio_init_id")"
      if [ "$minio_init_status" = "exited" ]; then
        minio_init_exit="$(docker inspect -f '{{ "{{" }}.State.ExitCode{{ "}}" }}' "$minio_init_id")"
        if [ "$minio_init_exit" != "0" ]; then
          echo "minio-init exited with status $minio_init_exit" >&2
          exit 1
        fi
        break
      fi
    fi
    if (( SECONDS >= deadline )); then
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

  $compose --profile runner run --rm runner

stack-budget:
  bash "{{repo}}/scripts/ci-stack-budget.sh"

perf-guard:
  python3 "{{repo}}/scripts/profile_guard.py" --profile quick --release --cli-cargo-feature fff-zlob --skip-dhat --enforce --out "{{repo}}/.benchmarks/perf-guard/local.json"

release-version-test:
  python3 "{{repo}}/scripts/test_release_version.py"

release-automation-test:
  python3 "{{repo}}/scripts/test_release_version.py"
  python3 "{{repo}}/scripts/test_publish_workspace.py"

# ── crates.io publishing ─────────────────────────────────────
# Show the current publishable workspace set and version. The release
# publisher computes the real dependency order from cargo metadata.
publish-order:
  #!/usr/bin/env bash
  set -euo pipefail
  python3 - <<'PY'
  import json
  import subprocess

  metadata = json.loads(subprocess.check_output([
      "cargo",
      "metadata",
      "--format-version",
      "1",
      "--locked",
      "--no-deps",
  ], text=True))
  members = set(metadata["workspace_members"])
  publishable = sorted(
      package["name"]
      for package in metadata["packages"]
      if package["id"] in members and package.get("publish") != []
  )
  version = next(
      package["version"]
      for package in metadata["packages"]
      if package["name"] == "lash-runtime"
  )
  print(f"Workspace version: {version}")
  print()
  print("Publishable crates:")
  for index, name in enumerate(publishable, start=1):
      print(f"  {index:2}. {name}")
  PY

# Dry-run the two leaf crates (no internal deps) — quick sanity check.
# Non-leaf dry-runs only work after their deps are already on crates.io.
publish-dry-run:
  @echo "Dry-run on leaf crates (lash-sansio, lashlang)..."
  cargo publish --dry-run --locked -p lash-sansio
  cargo publish --dry-run --locked -p lashlang
  @echo "OK."

# Publish a single crate. Idempotent: returns success if the same
# version is already on crates.io.
publish-one CRATE *args:
  #!/usr/bin/env bash
  set -euo pipefail
  version=$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)
  status=$(curl -s -o /dev/null -w "%{http_code}" \
    "https://crates.io/api/v1/crates/{{CRATE}}/$version")
  if [ "$status" = "200" ]; then
    echo "  ✓ {{CRATE}}@$version already on crates.io"
    exit 0
  fi
  echo "  → publishing {{CRATE}}@$version"
  cargo publish -p "{{CRATE}}" --no-verify --locked "$@"

# Publish every publishable workspace crate in dependency order. Re-runnable:
# already-published versions are skipped; transient crates.io/Cargo registry
# failures are retried by the helper.
publish-all *args:
  python3 "{{repo}}/scripts/publish_workspace.py" "$@"

check-file-size:
  ./scripts/check-production-file-size.sh
