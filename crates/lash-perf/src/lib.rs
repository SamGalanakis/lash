//! Developer-only performance harness for the Lash runtime.
//!
//! This crate is never published or shipped. It owns the synthetic
//! non-inference runtime benchmark (`runtime_perf`, driven by the
//! `lash-perf` binary and `scripts/profile_runtime*.py`) plus the shared
//! measurement helpers (`perf_support`) that the `lash-cli` UI benchmark
//! (`--ui-perf-benchmark`, behind its `bench` feature) also reuses.

pub mod perf_support;
pub mod runtime_perf;

/// Allocation-counter view of the process allocator.
///
/// The `lash-perf` binary installs `stats_alloc::INSTRUMENTED_SYSTEM` as its
/// global allocator, so these counters are live there. When this library is
/// linked elsewhere (e.g. the `lash-cli` UI bench build) the counters read
/// zero because that binary owns its own allocator.
#[cfg(not(feature = "dhat-heap"))]
pub static GLOBAL_ALLOCATOR: &stats_alloc::StatsAlloc<std::alloc::System> =
    &stats_alloc::INSTRUMENTED_SYSTEM;
