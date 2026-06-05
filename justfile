set positional-arguments

repo := justfile_directory()

default:
  @just --list

dev *args:
  cargo build --manifest-path "{{repo}}/crates/lash-cli/Cargo.toml"
  cd "${LASH_DEV_LAUNCH_CWD:-{{invocation_directory()}}}" && exec "{{repo}}/target/debug/lash" "$@"

agent-workbench:
  ./scripts/agent-workbench-dev.sh

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

# ── crates.io publishing ─────────────────────────────────────
# Topological order of every publishable crate. Computed once via
# `cargo metadata` then frozen here; edit by hand if a new internal
# dep edge appears. Single space-separated line so bash for-loops
# iterate cleanly when {{publish-crates}} is interpolated.
publish-crates := "lash-provider-auth lash-sansio lash-trace lashlang lash-core lash-openai-schema lash-rlm-types lash-llm-tools lash-llm-transport lash-protocol-rlm lash-protocol-standard lash-plugin-mcp lash-plugin-observational-memory lash-plugin-prompt-context lash-plugin-rolling-history lash-plugin-tool-discovery lash-turso-store lash-tool-support lash-provider-anthropic lash-provider-codex lash-provider-google lash-provider-openai lash-runtime lash-subagents lash-tool-apply-patch lash-tool-files lash-tool-search lash-tool-shell lash-tool-web lash-plugin-plan-mode lash-providers-builtin lash-standard-plugins"

# Show the publish order and current workspace version.
publish-order:
  @echo "Workspace version: $(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)"
  @echo
  @echo "Publish order:"
  @i=0; for c in {{publish-crates}}; do i=$((i+1)); printf '  %2d. %s\n' "$i" "$c"; done

# Dry-run the two leaf crates (no internal deps) — quick sanity check.
# Non-leaf dry-runs only work after their deps are already on crates.io.
publish-dry-run:
  @echo "Dry-run on leaf crates (lash-sansio, lashlang)..."
  cargo publish --dry-run -p lash-sansio
  cargo publish --dry-run -p lashlang
  @echo "OK."

# Publish a single crate. Idempotent: returns success if the same
# version is already on crates.io.
publish-one CRATE:
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
  cargo publish -p "{{CRATE}}"

# Publish every crate in dependency order. Re-runnable: each crate is
# checked against crates.io and skipped if its version is already
# there. Sleeps between publishes so the index settles before the next
# dep tries to resolve.
publish-all SLEEP="12":
  #!/usr/bin/env bash
  set -euo pipefail
  version=$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)
  echo "Publishing workspace at $version"
  echo
  for c in {{publish-crates}}; do
    status=$(curl -s -o /dev/null -w "%{http_code}" \
      "https://crates.io/api/v1/crates/$c/$version")
    if [ "$status" = "200" ]; then
      printf '  ✓ %-40s already on crates.io\n' "$c@$version"
      continue
    fi
    printf '  → %-40s publishing...\n' "$c@$version"
    cargo publish -p "$c"
    echo "    sleeping {{SLEEP}}s for index..."
    sleep {{SLEEP}}
  done
  echo
  echo "Done."
