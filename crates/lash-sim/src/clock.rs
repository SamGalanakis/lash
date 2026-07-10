use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use chrono::TimeZone as _;

const SIM_EPOCH_MS: u64 = 1_700_000_000_000;
const RENEWAL_SETTLE_YIELDS: usize = 100_000;
const RENEWAL_POST_REGISTRATION_YIELDS: usize = 128;
const UNSCHEDULED_STEP_MS: u64 = 5_000;
const UNSCHEDULED_STEP_YIELDS: usize = 32;
type SleepWaiter = (bool, tokio::sync::oneshot::Sender<()>);
type SleepersByDeadline = BTreeMap<u64, Vec<SleepWaiter>>;

/// Schedule-driven clock shared by the simulated runtime and embedded stores.
///
/// Wall time and sleep completion are derived only from the delivered schedule.
/// `Instant` is retained solely as the opaque origin required by the canonical
/// `Clock::now` type; it is never read again and cannot advance simulation.
#[derive(Debug)]
pub(crate) struct SimClock {
    logical_ms: AtomicU64,
    monotonic_origin: Instant,
    sleepers: Mutex<SleepersByDeadline>,
    completed_sleeps: AtomicU64,
    renewal_registrations: AtomicU64,
}

impl SimClock {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            logical_ms: AtomicU64::new(0),
            monotonic_origin: Instant::now(),
            sleepers: Mutex::new(BTreeMap::new()),
            completed_sleeps: AtomicU64::new(0),
            renewal_registrations: AtomicU64::new(0),
        })
    }

    pub(crate) fn logical_ms(&self) -> u64 {
        self.logical_ms.load(Ordering::SeqCst)
    }

    pub(crate) fn completed_sleeps(&self) -> u64 {
        self.completed_sleeps.load(Ordering::SeqCst)
    }

    pub(crate) async fn advance_to(&self, target_ms: u64) {
        loop {
            let current = self.logical_ms();
            if current >= target_ms {
                return;
            }
            let (next, waiters) = {
                let mut sleepers = self.sleepers.lock().expect("sim clock sleepers");
                let next_deadline = sleepers
                    .range((current + 1)..=target_ms)
                    .next()
                    .map(|(deadline, _)| *deadline);
                let next = next_deadline
                    .unwrap_or_else(|| target_ms.min(current.saturating_add(UNSCHEDULED_STEP_MS)));
                self.logical_ms.store(next, Ordering::SeqCst);
                let waiters = next_deadline
                    .and_then(|deadline| sleepers.remove(&deadline))
                    .unwrap_or_default();
                (next, waiters)
            };
            let registrations_before = self.renewal_registrations.load(Ordering::SeqCst);
            let mut renewal_count = 0_u64;
            for (renewal, waiter) in waiters {
                if waiter.send(()).is_ok() && renewal {
                    renewal_count += 1;
                }
            }
            if renewal_count > 0 {
                // Let renewal tasks perform their store write and register the
                // next sleep before crossing another TTL window.
                let expected = registrations_before.saturating_add(renewal_count);
                for _ in 0..RENEWAL_SETTLE_YIELDS {
                    if self.renewal_registrations.load(Ordering::SeqCst) >= expected {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
                for _ in 0..RENEWAL_POST_REGISTRATION_YIELDS {
                    tokio::task::yield_now().await;
                }
            } else if next < target_ms {
                for _ in 0..UNSCHEDULED_STEP_YIELDS {
                    tokio::task::yield_now().await;
                }
            }
        }
    }

    pub(crate) async fn advance_by(&self, delta_ms: u64) {
        self.advance_to(self.logical_ms().saturating_add(delta_ms))
            .await;
    }

    async fn wait_until_ms(&self, deadline_ms: u64, renewal: bool) {
        let receiver = {
            let (sender, receiver) = tokio::sync::oneshot::channel();
            let mut sleepers = self.sleepers.lock().expect("sim clock sleepers");
            if self.logical_ms() >= deadline_ms {
                return;
            }
            sleepers
                .entry(deadline_ms)
                .or_default()
                .push((renewal, sender));
            if renewal {
                self.renewal_registrations.fetch_add(1, Ordering::SeqCst);
            }
            receiver
        };
        let _ = receiver.await;
    }
}

#[async_trait::async_trait]
impl lash_core::Clock for SimClock {
    fn now(&self) -> Instant {
        self.monotonic_origin + Duration::from_millis(self.logical_ms())
    }

    fn timestamp_ms(&self) -> u64 {
        SIM_EPOCH_MS.saturating_add(self.logical_ms())
    }

    fn timestamp_rfc3339(&self) -> String {
        self.timestamp_datetime().to_rfc3339()
    }

    fn timestamp_datetime(&self) -> chrono::DateTime<chrono::Utc> {
        chrono::Utc
            .timestamp_millis_opt(self.timestamp_ms() as i64)
            .single()
            .expect("sim clock timestamp is representable")
    }

    async fn sleep(&self, duration: Duration) {
        self.wait_until_ms(
            self.logical_ms()
                .saturating_add(duration.as_millis() as u64),
            duration == Duration::from_millis(10_000),
        )
        .await;
        self.completed_sleeps.fetch_add(1, Ordering::SeqCst);
    }

    async fn sleep_until(&self, deadline: Instant) {
        let deadline_ms = deadline
            .saturating_duration_since(self.monotonic_origin)
            .as_millis() as u64;
        self.wait_until_ms(deadline_ms, false).await;
        self.completed_sleeps.fetch_add(1, Ordering::SeqCst);
    }
}
