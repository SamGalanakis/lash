#!/usr/bin/env bash
set -euo pipefail

stack_kb="${LASH_STACK_BUDGET_KB:-2048}"
rust_min_stack="${LASH_RUST_MIN_STACK_BUDGET:-2097152}"

cargo test -p lashlang --test stack_budget --locked --no-run
cargo test -p lash-runtime stack_budget --locked --no-run
cargo test -p lash-subagents --locked --no-run

(
  ulimit -s "$stack_kb"
  export RUST_MIN_STACK="$rust_min_stack"
  cargo test -p lashlang --test stack_budget --locked -- --nocapture --test-threads=1
  cargo test -p lash-runtime stack_budget --locked -- --nocapture --test-threads=1
  cargo test -p lash-subagents rlm_spawn_seed_is_visible_to_child_executor_and_prompt --locked -- --nocapture --test-threads=1
)
