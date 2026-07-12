//! Developer-only performance harness for the Lash runtime.
//!
//! This crate is never published or shipped. It owns the synthetic
//! non-inference runtime benchmark (`runtime_perf`, driven by the
//! `lash-perf` binary and `scripts/profile_runtime*.py`) plus its private
//! measurement helpers (`perf_support`). Host applications own their own UI
//! measurement support.

pub mod perf_support;
pub mod runtime_perf;

/// Allocation-counter view of the process allocator.
///
/// The `lash-perf` binary installs `stats_alloc::INSTRUMENTED_SYSTEM` as its
/// global allocator, so these counters are live there.
#[cfg(not(feature = "dhat-heap"))]
pub static GLOBAL_ALLOCATOR: &stats_alloc::StatsAlloc<std::alloc::System> =
    &stats_alloc::INSTRUMENTED_SYSTEM;
