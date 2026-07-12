set positional-arguments

repo := justfile_directory()

default:
  @just --list

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
  image="${AGENT_SERVICE_RESTATE_IMAGE:-restatedev/restate:1.7.0}"
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
  image="${AGENT_WORKBENCH_RESTATE_IMAGE:-restatedev/restate:1.7.0}"
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
  bash "{{repo}}/scripts/restate-postgres-workers-e2e.sh"

stack-budget:
  bash "{{repo}}/scripts/ci-stack-budget.sh"

push-gate:
  bash "{{repo}}/scripts/push-gate.sh"

confidence lane='default':
  bash "{{repo}}/scripts/confidence-gate.sh" "{{lane}}"

confidence-fast:
  bash "{{repo}}/scripts/confidence-gate.sh" fast

confidence-broad:
  bash "{{repo}}/scripts/confidence-gate.sh" broad

confidence-full:
  bash "{{repo}}/scripts/confidence-gate.sh" full

perf-guard:
  python3 "{{repo}}/scripts/profile_runtime.py" --profile quick --release --enforce-budgets --out "{{repo}}/.benchmarks/perf-guard/runtime-local.json"
  python3 "{{repo}}/scripts/profile_lashlang.py" --iterations 500 --profile-iterations 500 --enforce-budgets --out "{{repo}}/.benchmarks/perf-guard/lashlang-local.json"

release-version-test:
  python3 "{{repo}}/scripts/test_release_version.py"

release-automation-test:
  python3 "{{repo}}/scripts/test_release_version.py"
  python3 "{{repo}}/scripts/test_publish_workspace.py"

# ── crates.io publishing ─────────────────────────────────────
# Show the publishable workspace set. The in-tree version is the 0.0.0-dev
# placeholder — the release publisher stamps the real version at packaging time
# and computes the dependency layers from cargo metadata
# (`python3 scripts/publish_workspace.py --plan --version X.Y.Z`).
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

# Publish a single crate at the in-tree version. Idempotent: returns success if
# the same version is already on crates.io. NOTE: the in-tree version is the
# 0.0.0-dev placeholder unless you have stamped a real version first
# (`python3 scripts/release_version.py stamp X.Y.Z`); for a real release use the
# layered publisher (`python3 scripts/publish_workspace.py --version X.Y.Z`).
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
