#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

forbidden='static_graph_json|\bLashlangMap\b'
if rg -n "$forbidden" crates examples; then
  echo "workflow graph model check failed: retired definition-graph API still has consumers" >&2
  exit 1
fi

definition_count="$(rg -n '^pub struct WorkflowGraph \{' crates | wc -l | tr -d ' ')"
if [ "$definition_count" != "1" ]; then
  echo "workflow graph model check failed: expected one WorkflowGraph definition, found $definition_count" >&2
  exit 1
fi

if ! rg -q 'lash_lashlang_runtime::trace_lashlang_main_map\(artifact\)' \
  crates/lash-protocol-rlm/src/executor.rs; then
  echo "workflow graph model check failed: RLM no longer delegates its trace skeleton" >&2
  exit 1
fi

if ! rg -q 'workflow_graph_from_source' crates/lash-lashlang-runtime/src/process.rs; then
  echo "workflow graph model check failed: trace skeleton no longer projects WorkflowGraph" >&2
  exit 1
fi
