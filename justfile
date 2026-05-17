set positional-arguments

repo := justfile_directory()

default:
  @just --list

dev *args:
  cargo build --manifest-path "{{repo}}/crates/lash-cli/Cargo.toml"
  cd "${LASH_DEV_LAUNCH_CWD:-{{invocation_directory()}}}" && exec "{{repo}}/target/debug/lash" "$@"

# ── crates.io publishing ─────────────────────────────────────
# Topological order of every publishable crate. Computed once via
# `cargo metadata` then frozen here; edit by hand if a new internal
# dep edge appears. Single space-separated line so bash for-loops
# iterate cleanly when {{publish-crates}} is interpolated.
publish-crates := "lash-provider-auth lash-sansio lash-trace lashlang lash-core lash-openai-schema lash-rlm-types lash-llm-tools lash-llm-transport lash-mode-rlm lash-mode-standard lash-plugin-mcp lash-plugin-observational-memory lash-plugin-prompt-context lash-plugin-rolling-history lash-plugin-tool-discovery lash-plugin-ui-activity lash-sqlite-store lash-tool-support lash-provider-anthropic lash-provider-codex lash-provider-google lash-provider-openai lash-runtime lash-subagents lash-tool-apply-patch lash-tool-files lash-tool-search lash-tool-shell lash-tool-web lash-plugin-plan-mode lash-providers-builtin lash-standard-plugins"

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
