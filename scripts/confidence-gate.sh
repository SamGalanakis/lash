#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"

export PATH="$HOME/.cargo/bin:$PATH"

lane="${1:-default}"
out_root="${LASH_CONFIDENCE_OUT_DIR:-$repo/target/confidence}"
out_dir="${out_root}/${lane}"
ci_features="${LASH_CI_FEATURES:--F lash-cli/fff-zlob -F lash-cli/bench}"
critical_packages=(lash-core lashlang lash-protocol-rlm lash-protocol-standard)

mkdir -p "$out_dir"

step() {
  printf '\n==> %s\n' "$*"
}

usage() {
  cat <<'USAGE'
Usage: scripts/confidence-gate.sh [fast|default|full]

Lanes:
  fast     deterministic scenario harnesses, state-machine/property checks,
           generated DST replay/provider proof, durable fault-matrix metadata,
           and perf guard identity tests.
  default  fast + local backend conformance, coverage blind-spot artifacts,
           and cargo-mutants smoke shards for critical crates.
  full     default + Postgres backend conformance and full cargo-mutants over
           critical crates.

Tool policy:
  default/full require cargo-llvm-cov and cargo-mutants. Set
  LASH_CONFIDENCE_BOOTSTRAP=1 to install pinned versions if missing.
  Missing required tools fail the lane; skipped coverage or mutation is never
  reported as a pass.
USAGE
}

case "$lane" in
  fast|default|full) ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

bootstrap_tools() {
  if [ "${LASH_CONFIDENCE_BOOTSTRAP:-0}" != "1" ]; then
    return
  fi
  if ! command -v cargo-llvm-cov >/dev/null 2>&1; then
    step "Bootstrap cargo-llvm-cov 0.8.7"
    cargo install cargo-llvm-cov --version 0.8.7 --locked
  fi
  if ! command -v cargo-mutants >/dev/null 2>&1; then
    step "Bootstrap cargo-mutants 27.1.0"
    cargo install cargo-mutants --version 27.1.0 --locked
  fi
  if command -v rustup >/dev/null 2>&1 \
    && ! rustup component list --installed | grep -Eq '^llvm-tools-preview($|-)'; then
    step "Bootstrap rustup component llvm-tools-preview"
    rustup component add llvm-tools-preview
  fi
  if [ -z "${LLVM_COV:-}" ] && [ -z "${LLVM_PROFDATA:-}" ] \
    && ! command -v rustup >/dev/null 2>&1 \
    && command -v nix >/dev/null 2>&1; then
    bootstrap_nix_llvm_tools
  fi
}

require_tool() {
  local tool="$1"
  local crate="$2"
  local version="$3"
  if command -v "$tool" >/dev/null 2>&1; then
    return
  fi
  cat >&2 <<EOF
Required tool '$tool' is not installed for the '$lane' confidence lane.
Install with:
  cargo install ${crate} --version ${version} --locked
or rerun with:
  LASH_CONFIDENCE_BOOTSTRAP=1 scripts/confidence-gate.sh ${lane}
EOF
  exit 127
}

bootstrap_nix_llvm_tools() {
  local llvm_major llvm_attr llvm_out
  llvm_major="$(
    rustc -vV \
      | awk -F': ' '/^LLVM version:/ { split($2, version, "."); print version[1] }'
  )"
  if [ -z "$llvm_major" ]; then
    echo "Could not infer LLVM major version from rustc -vV" >&2
    exit 127
  fi

  llvm_attr="nixpkgs#llvmPackages_${llvm_major}.llvm"
  step "Bootstrap ${llvm_attr}"
  llvm_out="$(
    nix build --no-link --print-out-paths "$llvm_attr" \
      --extra-experimental-features 'nix-command flakes'
  )"
  export LLVM_COV="${llvm_out}/bin/llvm-cov"
  export LLVM_PROFDATA="${llvm_out}/bin/llvm-profdata"
  if [ ! -x "$LLVM_COV" ] || [ ! -x "$LLVM_PROFDATA" ]; then
    echo "Nix LLVM package did not provide executable llvm-cov/llvm-profdata under ${llvm_out}/bin" >&2
    exit 127
  fi
}

require_llvm_tools() {
  if [ -n "${LLVM_COV:-}" ] && [ -n "${LLVM_PROFDATA:-}" ]; then
    return
  fi
  if command -v rustup >/dev/null 2>&1 \
    && rustup component list --installed | grep -Eq '^llvm-tools-preview($|-)'; then
    return
  fi
  if [ "${LASH_CONFIDENCE_BOOTSTRAP:-0}" = "1" ] \
    && ! command -v rustup >/dev/null 2>&1 \
    && command -v nix >/dev/null 2>&1; then
    bootstrap_nix_llvm_tools
    return
  fi
  if ! command -v rustup >/dev/null 2>&1; then
    cat >&2 <<EOF
Coverage requires llvm-tools-preview, or explicit LLVM_COV and LLVM_PROFDATA paths.
This environment does not have rustup, so the gate cannot bootstrap the Rust
llvm-tools component here.

Set compatible binaries explicitly:
  LLVM_COV=/path/to/llvm-cov LLVM_PROFDATA=/path/to/llvm-profdata

Or let the gate build the matching Nix LLVM package inferred from rustc -vV:
  LASH_CONFIDENCE_BOOTSTRAP=1 scripts/confidence-gate.sh ${lane}
EOF
    exit 127
  fi
  cat >&2 <<EOF
Coverage requires llvm-tools-preview, or explicit LLVM_COV and LLVM_PROFDATA paths.
Install with:
  rustup component add llvm-tools-preview
or rerun with:
  LASH_CONFIDENCE_BOOTSTRAP=1 scripts/confidence-gate.sh ${lane}
If rustup is unavailable but Nix is installed, the bootstrap path builds
nixpkgs#llvmPackages_\${rustc_llvm_major}.llvm and exports LLVM_COV/LLVM_PROFDATA.
EOF
  exit 127
}

run_scenario_harnesses() {
  step "Runtime Scenario harness"
  cargo test -p lash-core --locked runtime_scenario

  step "Standard Protocol Scenario harness"
  cargo test -p lash-protocol-standard --locked --test protocol_scenarios

  step "RLM Protocol Scenario harness"
  cargo test -p lash-protocol-rlm --locked --test protocol_drivers

  step "Agent Scenario harness"
  cargo test -p lash-runtime --locked --features rlm,testing agent_scenarios
}

run_state_machine_and_fault_matrix() {
  step "Runtime state-machine property runner"
  cargo test -p lash-core --locked runtime_state_machine_property

  step "Lashlang property suite"
  cargo test -p lashlang --locked property

  step "Durable fault matrix metadata"
  cargo test -p lash-core --locked durable_fault_matrix

  step "Durable fault matrix executable evidence"
  # durable-fault-matrix: crash-reopen-runtime-rebuild
  cargo test -p lash-runtime --locked --features rlm,testing \
    runtime_rebuild_and_worker_recovery_with_durable_stores
  # durable-fault-matrix: provider-retry-exhaustion
  cargo test -p lash-core --locked retryable_llm_failures_exhaust_and_fail_turn
  # durable-fault-matrix: protocol-provider-failure
  cargo test -p lash-protocol-standard --locked --test protocol_scenarios \
    standard_protocol_scenario_provider_error_stops_without_checkpoint
  # durable-fault-matrix: sqlite-backend-conformance
  cargo test -p lash-sqlite-store --locked --test conformance conformance
}

run_sim_provider_scripts() {
  step "Deterministic simulation generated lane"
  local sim_profile
  case "$lane" in
    fast) sim_profile="${LASH_SIM_PROFILE:-fast-random}" ;;
    default) sim_profile="${LASH_SIM_PROFILE:-default-random}" ;;
    full) sim_profile="${LASH_SIM_PROFILE:-full-random}" ;;
  esac
  local cmd=(cargo run -p lash-sim --locked -- run --out "${out_dir}/sim" --profile "$sim_profile")
  if [ -n "${LASH_SIM_SEEDS:-}" ]; then
    cmd+=(--seeds "$LASH_SIM_SEEDS")
  fi
  if [ -n "${LASH_SIM_MAX_BOUNDARIES:-}" ]; then
    cmd+=(--max-boundaries "$LASH_SIM_MAX_BOUNDARIES")
  fi
  "${cmd[@]}"
}

run_perf_identity_checks() {
  step "Performance guard identity checks"
  python3 scripts/test_profile_guard.py
}

run_local_backend_conformance() {
  step "Sqlite backend conformance"
  cargo test -p lash-sqlite-store --locked --test conformance
}

run_postgres_conformance() {
  step "Postgres backend conformance"
  if [ -n "${LASH_POSTGRES_DATABASE_URL:-}" ]; then
    cargo test -p lash-postgres-store --locked --test conformance
    return
  fi

  if ! command -v docker >/dev/null 2>&1; then
    echo "Full confidence requires Docker or LASH_POSTGRES_DATABASE_URL for Postgres conformance." >&2
    exit 127
  fi

  local container port
  container="lash-confidence-postgres-$$"
  port="${LASH_CONFIDENCE_POSTGRES_PORT:-$((21000 + ($$ % 20000)))}"
  docker rm -f "$container" >/dev/null 2>&1 || true
  cleanup_postgres() {
    docker rm -f "$container" >/dev/null 2>&1 || true
  }
  trap cleanup_postgres RETURN

  bash scripts/docker-pull-with-retry.sh postgres:16-alpine
  docker run -d --name "$container" \
    -e POSTGRES_USER=lash \
    -e POSTGRES_PASSWORD=lash \
    -e POSTGRES_DB=lash \
    -p "127.0.0.1:${port}:5432" \
    postgres:16-alpine >/dev/null

  local deadline=$((SECONDS + 60))
  until docker exec "$container" pg_isready -U lash -d lash >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      docker logs "$container" >&2 || true
      echo "Postgres did not become ready on port ${port}" >&2
      exit 1
    fi
    sleep 1
  done

  LASH_POSTGRES_DATABASE_URL="postgres://lash:lash@127.0.0.1:${port}/lash" \
    cargo test -p lash-postgres-store --locked --test conformance
}

run_coverage_blind_spots() {
  step "Coverage blind-spot map"
  require_tool cargo-llvm-cov cargo-llvm-cov 0.8.7
  require_llvm_tools
  local coverage_dir="${out_dir}/coverage"
  mkdir -p "$coverage_dir"
  cargo llvm-cov clean --workspace
  cargo llvm-cov --locked \
    -p lash-core \
    -p lashlang \
    -p lash-protocol-rlm \
    -p lash-protocol-standard \
    --tests \
    --lcov \
    --output-path "${coverage_dir}/lcov.info"
  cargo llvm-cov report --text --show-missing-lines \
    --output-path "${coverage_dir}/missing-lines.txt"
  cargo llvm-cov report --json --summary-only \
    --output-path "${coverage_dir}/summary.json"
  awk '
    /^SF:/ {
      file = substr($0, 4)
      total = 0
      uncovered = 0
      next
    }
    /^DA:/ {
      split(substr($0, 4), fields, ",")
      total += 1
      if (fields[2] == 0) {
        uncovered += 1
      }
      next
    }
    /^end_of_record/ {
      if (file ~ /\/crates\/(lash-core|lashlang|lash-protocol-rlm|lash-protocol-standard)\// && uncovered > 0) {
        print uncovered "\t" total "\t" file
      }
    }
  ' "${coverage_dir}/lcov.info" \
    | sort -nr \
    >"${coverage_dir}/critical-uncovered-files.tsv"
  cat >"${coverage_dir}/README.md" <<EOF
# Confidence Coverage Blind Spots

Coverage is an observation artifact, not a pass/fail percentage.

- LCOV: ${coverage_dir}/lcov.info
- Missing-line text report: ${coverage_dir}/missing-lines.txt
- Per-file summary JSON: ${coverage_dir}/summary.json
- Critical uncovered file index: ${coverage_dir}/critical-uncovered-files.tsv

Use these outputs to find unexercised contracts in critical runtime,
Lashlang, RLM protocol, and Standard protocol code.
EOF
}

run_mutation_smoke() {
  step "Mutation smoke shards"
  require_tool cargo-mutants cargo-mutants 27.1.0
  local shard="${LASH_MUTATION_SMOKE_SHARD:-1/64}"
  local jobs="${LASH_MUTATION_JOBS:-2}"
  local timeout="${LASH_MUTATION_TIMEOUT_SECONDS:-180}"
  for package in "${critical_packages[@]}"; do
    cargo mutants \
      -p "$package" \
      --cargo-arg=--locked \
      --test-tool cargo \
      --shard "$shard" \
      --jobs "$jobs" \
      --timeout "$timeout" \
      --minimum-test-timeout 30 \
      --output "${out_dir}/mutants-${package}-smoke"
  done
}

run_mutation_full() {
  step "Full mutation suites"
  require_tool cargo-mutants cargo-mutants 27.1.0
  local jobs="${LASH_MUTATION_JOBS:-2}"
  local timeout="${LASH_MUTATION_TIMEOUT_SECONDS:-600}"
  for package in "${critical_packages[@]}"; do
    cargo mutants \
      -p "$package" \
      --cargo-arg=--locked \
      --test-tool cargo \
      --jobs "$jobs" \
      --timeout "$timeout" \
      --minimum-test-timeout 60 \
      --output "${out_dir}/mutants-${package}-full"
  done
}

bootstrap_tools

run_scenario_harnesses
run_state_machine_and_fault_matrix
run_sim_provider_scripts
run_perf_identity_checks

if [ "$lane" = "default" ] || [ "$lane" = "full" ]; then
  run_local_backend_conformance
  run_coverage_blind_spots
  run_mutation_smoke
fi

if [ "$lane" = "full" ]; then
  run_postgres_conformance
  run_mutation_full
fi

step "Confidence gate '${lane}' passed"
printf 'Artifacts: %s\n' "$out_dir"
