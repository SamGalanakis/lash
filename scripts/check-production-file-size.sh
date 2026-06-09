#!/usr/bin/env bash
set -euo pipefail

threshold="${LASH_FILE_SIZE_LIMIT:-1000}"
root="${1:-crates}"

# Existing oversized production files are explicit debt. New entries should not
# be added casually; split the module instead, or document why the file earns an
# exception here.
allowlist=$(cat <<'EOF'
crates/lashlang/src/linker.rs
crates/lash-remote-protocol/src/lib.rs
crates/lash-remote-protocol/src/core_conversions.rs
crates/lash-cli/src/render/compositor.rs
crates/lashlang/src/runtime/compiler.rs
crates/lash-sansio/src/tool_contract.rs
crates/lash-core/src/runtime/session_manager/process_runners/mod.rs
crates/lash-tools/src/search.rs
crates/lash-cli/src/interactive/input_handling.rs
crates/lashlang/src/parser.rs
crates/lash-core/src/tool_registry.rs
crates/lash-tools/src/shell/mod.rs
crates/lash-sansio/src/sansio/mod.rs
crates/lashlang/src/runtime/vm/mod.rs
crates/lash-tools/src/apply_patch.rs
crates/lash-provider-openai/src/responses_shared.rs
crates/lash-protocol-rlm/src/executor.rs
crates/lash-core/src/runtime/turn_loop.rs
crates/lash-sansio/src/session_model/message.rs
crates/lash-standard-plugins/src/rolling_history.rs
crates/lash-provider-openai/src/codex.rs
crates/lash-cli/src/startup/onboarding.rs
crates/lashlang/src/artifact.rs
crates/lashlang/src/graph.rs
crates/lash-cli/src/markdown.rs
crates/lash-tui-extensions/src/lib.rs
crates/lash/src/control.rs
crates/lash-restate/src/lib.rs
crates/lash-tui/src/core.rs
crates/lash-autoresearch/src/ui.rs
crates/lashlang/src/runtime/ops.rs
crates/lash-plugin-tool-output-budget/src/lib.rs
crates/lash-trace/src/otel.rs
crates/lash-core/src/runtime/process/model.rs
crates/lash-cli/src/app/mod.rs
crates/lashlang/src/trigger.rs
crates/lash-sqlite-store/src/process_registry.rs
crates/lash-core/src/store.rs
crates/lash-perf/src/runtime_perf/providers.rs
crates/lash-core/src/runtime/turn_boundary.rs
crates/lash-cli/src/render/mod.rs
crates/lash-perf/src/runtime_perf/measurement.rs
crates/lash-cli/src/editor.rs
crates/lash-core/src/session_graph.rs
EOF
)

failures=()
while IFS= read -r -d '' file; do
  case "$file" in
    */tests/*|*/examples/*|*/benches/*|*/src/tests.rs|*/src/tests/*|*/src/*/tests.rs|*/src/testing/*|*/src/testing.rs) continue ;;
  esac
  lines=$(wc -l < "$file")
  if (( lines <= threshold )); then
    continue
  fi
  if grep -qxF "$file" <<<"$allowlist"; then
    continue
  fi
  failures+=("$file:$lines")
done < <(find "$root" -type f -name '*.rs' -print0)

if ((${#failures[@]})); then
  echo "Production Rust files over ${threshold} lines:" >&2
  printf '  %s\n' "${failures[@]}" >&2
  echo >&2
  echo "Split the file or add a justified exception to scripts/check-production-file-size.sh." >&2
  exit 1
fi
