//! Injected time source.
//!
//! lash is a replayable durable runtime, so time is an effect like any other:
//! durable timestamps must be reproducible under replay, and timeout/backoff
//! logic must be drivable deterministically in tests. Every wall-clock and
//! monotonic read in the runtime path goes through a [`Clock`]; the default
//! [`SystemClock`] reads the real OS clock, and tests inject a controllable
//! one.
//!
//! Boundary: `now()` is monotonic (measurement/elapsed only, never persisted);
//! `timestamp_ms`/`timestamp_rfc3339` are wall-clock (durable records). `sleep`
//! and `sleep_until` replace direct `tokio::time` calls so a fake clock can
//! resolve them without real wall-clock waits.

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;

/// Runtime and embedded-store time source. Cloneable as `Arc<dyn Clock>`;
/// carried on [`RuntimeHostConfig`](super::RuntimeHostConfig).
///
/// SQLite and in-memory persistence read this host-injectable clock because
/// they run in the same clock domain as their host. PostgreSQL lease decisions
/// deliberately remain database-authoritative (`transaction_timestamp()` /
/// `clock_timestamp()`) to protect fencing across skewed hosts; the
/// `postgres_clock_contract` tests pin that boundary.
#[async_trait]
pub trait Clock: Send + Sync + std::fmt::Debug {
    /// Monotonic instant for measuring elapsed time. Never persisted.
    fn now(&self) -> Instant;

    /// Wall-clock time as epoch milliseconds, for durable records.
    fn timestamp_ms(&self) -> u64;

    /// Wall-clock time as an RFC 3339 string, for durable records.
    fn timestamp_rfc3339(&self) -> String;

    /// Wall-clock time as a UTC datetime, for trace records.
    fn timestamp_datetime(&self) -> chrono::DateTime<chrono::Utc>;

    /// Sleep for `duration`. Replaces `tokio::time::sleep`.
    async fn sleep(&self, duration: Duration);

    /// Sleep until `deadline` (a value from [`now`](Clock::now)). Replaces
    /// `tokio::time::sleep_until`.
    async fn sleep_until(&self, deadline: Instant);
}

/// The real OS clock. Native behavior is identical to the direct `std`/`tokio`
/// calls it replaces.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemClock;

#[async_trait]
impl Clock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }

    fn timestamp_ms(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    fn timestamp_rfc3339(&self) -> String {
        self.timestamp_datetime().to_rfc3339()
    }

    fn timestamp_datetime(&self) -> chrono::DateTime<chrono::Utc> {
        chrono::Utc::now()
    }

    async fn sleep(&self, duration: Duration) {
        tokio::time::sleep(duration).await;
    }

    async fn sleep_until(&self, deadline: Instant) {
        tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)).await;
    }
}
