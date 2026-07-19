//! In-process await-event (Durable Wait) registry backing the inline effect
//! host and controller.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use hmac::{Hmac, Mac};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use crate::RuntimeError;

use super::executor::{
    AwaitEventKey, AwaitEventWaitIdentity, ExecutionScope, Resolution, ResolveOutcome,
};

type HmacSha256 = Hmac<sha2::Sha256>;

const COMPLETED_TURN_CONTROL_KEY_LIMIT: usize = 4_096;
const REVOKED_SESSION_LIMIT: usize = 4_096;

pub(super) fn inline_await_events() -> &'static AwaitEventRegistry {
    static REGISTRY: OnceLock<AwaitEventRegistry> = OnceLock::new();
    REGISTRY.get_or_init(AwaitEventRegistry::new)
}

#[derive(Debug)]
struct AwaitEventEntry {
    /// Session the wait's execution scope belongs to, when it has one. Used by
    /// [`AwaitEventRegistry::cancel_session`] to find a session's outstanding
    /// waits without re-deriving key hashes.
    session_id: Option<String>,
    turn_control: bool,
    terminal: Option<Resolution>,
    notify: Arc<Notify>,
}

impl AwaitEventEntry {
    fn for_key(key: &AwaitEventKey) -> Self {
        Self {
            session_id: key.scope.session_id().map(ToOwned::to_owned),
            turn_control: key.wait.is_turn_control(),
            terminal: None,
            notify: Arc::new(Notify::new()),
        }
    }
}

#[derive(Debug)]
struct AwaitEventRegistryState {
    entries: HashMap<String, AwaitEventEntry>,
    completed_turn_control: HashMap<String, CompletedTurnControlEntry>,
    completed_turn_control_order: VecDeque<String>,
    revoked_session_ids: HashSet<String>,
    revoked_session_order: VecDeque<String>,
}

#[derive(Debug)]
struct CompletedTurnControlEntry {
    session_id: Option<String>,
    terminal: Resolution,
}

#[derive(Debug)]
pub(super) struct AwaitEventRegistry {
    secret: Vec<u8>,
    state: std::sync::Mutex<AwaitEventRegistryState>,
    completed_turn_control_key_limit: usize,
    revoked_session_limit: usize,
}

impl AwaitEventRegistry {
    fn new() -> Self {
        Self::with_limits(COMPLETED_TURN_CONTROL_KEY_LIMIT, REVOKED_SESSION_LIMIT)
    }

    fn with_limits(completed_turn_control_key_limit: usize, revoked_session_limit: usize) -> Self {
        Self {
            secret: uuid::Uuid::new_v4().as_bytes().to_vec(),
            state: std::sync::Mutex::new(AwaitEventRegistryState {
                entries: HashMap::new(),
                completed_turn_control: HashMap::new(),
                completed_turn_control_order: VecDeque::new(),
                revoked_session_ids: HashSet::new(),
                revoked_session_order: VecDeque::new(),
            }),
            completed_turn_control_key_limit,
            revoked_session_limit,
        }
    }

    fn locked_state(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, AwaitEventRegistryState>, RuntimeError> {
        self.state.lock().map_err(|_| {
            RuntimeError::new(
                "await_event_registry_poisoned",
                "await-event registry lock poisoned",
            )
        })
    }

    pub(super) fn key_for(
        &self,
        scope: &ExecutionScope,
        wait: AwaitEventWaitIdentity,
    ) -> Result<AwaitEventKey, RuntimeError> {
        scope.validate()?;
        wait.validate()?;
        let key_id =
            crate::stable_hash::stable_json_sha256_hex(&(scope, &wait)).map_err(|err| {
                RuntimeError::new(
                    "await_event_key_hash",
                    format!("failed to hash await-event identity: {err}"),
                )
            })?;
        let signature = self.signature(scope, &wait, &key_id)?;
        Ok(AwaitEventKey {
            scope: scope.clone(),
            wait,
            key_id,
            signature,
        })
    }

    fn signature(
        &self,
        scope: &ExecutionScope,
        wait: &AwaitEventWaitIdentity,
        key_id: &str,
    ) -> Result<String, RuntimeError> {
        let mut mac = HmacSha256::new_from_slice(&self.secret).map_err(|err| {
            RuntimeError::new(
                "await_event_key_sign",
                format!("failed to initialize await-event key signer: {err}"),
            )
        })?;
        let canonical = serde_json::to_vec(&(scope, wait, key_id)).map_err(|err| {
            RuntimeError::new(
                "await_event_key_sign",
                format!("failed to serialize await-event key identity: {err}"),
            )
        })?;
        mac.update(&canonical);
        Ok(format!("{:x}", mac.finalize().into_bytes()))
    }

    fn verify(&self, key: &AwaitEventKey) -> Result<bool, RuntimeError> {
        let expected = self.signature(&key.scope, &key.wait, &key.key_id)?;
        Ok(expected == key.signature)
    }

    pub(super) fn resolve(
        &self,
        key: &AwaitEventKey,
        resolution: Resolution,
    ) -> Result<ResolveOutcome, RuntimeError> {
        if !self.verify(key)? {
            return Ok(ResolveOutcome::UnknownOrRevoked);
        }
        let mut state = self.locked_state()?;
        if key
            .scope
            .session_id()
            .is_some_and(|session_id| state.revoked_session_ids.contains(session_id))
        {
            return Ok(ResolveOutcome::UnknownOrRevoked);
        }
        if let Some(completed) = state.completed_turn_control.get(&key.key_id) {
            return Ok(ResolveOutcome::AlreadyResolved {
                terminal: completed.terminal.clone(),
            });
        }
        let entry = state
            .entries
            .entry(key.key_id.clone())
            .or_insert_with(|| AwaitEventEntry::for_key(key));
        if let Some(terminal) = &entry.terminal {
            return Ok(ResolveOutcome::AlreadyResolved {
                terminal: terminal.clone(),
            });
        }
        entry.terminal = Some(resolution);
        entry.notify.notify_waiters();
        if matches!(key.wait, AwaitEventWaitIdentity::TurnTerminal) {
            self.archive_turn_control(&mut state, key)?;
        }
        Ok(ResolveOutcome::Accepted)
    }

    fn archive_turn_control(
        &self,
        state: &mut AwaitEventRegistryState,
        terminal_key: &AwaitEventKey,
    ) -> Result<(), RuntimeError> {
        let gate_key = self.key_for(&terminal_key.scope, AwaitEventWaitIdentity::TurnCancelGate)?;
        for key_id in [&gate_key.key_id, &terminal_key.key_id] {
            let Some(entry) = state.entries.remove(key_id) else {
                continue;
            };
            let Some(terminal) = entry.terminal else {
                continue;
            };
            if state
                .completed_turn_control
                .insert(
                    key_id.clone(),
                    CompletedTurnControlEntry {
                        session_id: entry.session_id,
                        terminal,
                    },
                )
                .is_none()
            {
                state.completed_turn_control_order.push_back(key_id.clone());
            }
        }
        while state.completed_turn_control.len() > self.completed_turn_control_key_limit {
            let Some(key_id) = state.completed_turn_control_order.pop_front() else {
                break;
            };
            state.completed_turn_control.remove(&key_id);
        }
        Ok(())
    }

    pub(super) async fn await_resolution(
        &self,
        key: &AwaitEventKey,
        cancel: CancellationToken,
        deadline: Option<Instant>,
        clock: &dyn crate::Clock,
    ) -> Result<Resolution, RuntimeError> {
        if !self.verify(key)? {
            return Err(RuntimeError::new(
                "await_event_unknown_or_revoked",
                "await-event key is invalid or revoked",
            ));
        }
        loop {
            let notify = {
                let mut state = self.locked_state()?;
                if key
                    .scope
                    .session_id()
                    .is_some_and(|session_id| state.revoked_session_ids.contains(session_id))
                {
                    return Err(RuntimeError::new(
                        "await_event_unknown_or_revoked",
                        "await-event key is invalid or revoked",
                    ));
                }
                if let Some(completed) = state.completed_turn_control.get(&key.key_id) {
                    return Ok(completed.terminal.clone());
                }
                let entry = state
                    .entries
                    .entry(key.key_id.clone())
                    .or_insert_with(|| AwaitEventEntry::for_key(key));
                if let Some(terminal) = entry.terminal.clone() {
                    return Ok(terminal);
                }
                Arc::clone(&entry.notify)
            };
            if let Some(deadline) = deadline {
                tokio::select! {
                    _ = cancel.cancelled() => {
                        if key.wait.is_turn_control() {
                            return Err(RuntimeError::new(
                                "turn_control_wait_cancelled",
                                "turn-control waiter stopped without resolving its keyed promise",
                            ));
                        }
                        let _ = self.resolve(key, Resolution::Cancelled)?;
                    }
                    _ = clock.sleep_until(deadline) => {
                        if key.wait.is_turn_control() {
                            return Err(RuntimeError::new(
                                "turn_control_wait_timeout",
                                "turn-control waiter timed out without resolving its keyed promise",
                            ));
                        }
                        let _ = self.resolve(key, Resolution::Timeout)?;
                    }
                    _ = notify.notified() => {}
                }
            } else {
                tokio::select! {
                    _ = cancel.cancelled() => {
                        if key.wait.is_turn_control() {
                            return Err(RuntimeError::new(
                                "turn_control_wait_cancelled",
                                "turn-control waiter stopped without resolving its keyed promise",
                            ));
                        }
                        let _ = self.resolve(key, Resolution::Cancelled)?;
                    }
                    _ = notify.notified() => {}
                }
            }
        }
    }

    /// Revoke every await-event for `session_id`: in-flight waiters error with
    /// `await_event_unknown_or_revoked`, matching entries are drained, and a
    /// bounded recent-session tombstone cache rejects keys created after
    /// deletion without growing for the lifetime of the process.
    pub(super) fn revoke_session(&self, session_id: &str) -> Result<(), RuntimeError> {
        let mut state = self.locked_state()?;
        if state.revoked_session_ids.insert(session_id.to_string()) {
            state
                .revoked_session_order
                .push_back(session_id.to_string());
        }
        while state.revoked_session_ids.len() > self.revoked_session_limit {
            let Some(expired) = state.revoked_session_order.pop_front() else {
                break;
            };
            state.revoked_session_ids.remove(&expired);
        }
        state.entries.retain(|_, entry| {
            if entry.session_id.as_deref() == Some(session_id) {
                entry.notify.notify_waiters();
                false
            } else {
                true
            }
        });
        state
            .completed_turn_control
            .retain(|_, entry| entry.session_id.as_deref() != Some(session_id));
        let retained: HashSet<_> = state.completed_turn_control.keys().cloned().collect();
        state
            .completed_turn_control_order
            .retain(|key_id| retained.contains(key_id));
        Ok(())
    }

    #[cfg(test)]
    fn counts(&self) -> Result<(usize, usize, usize), RuntimeError> {
        let state = self.locked_state()?;
        Ok((
            state.entries.len(),
            state.completed_turn_control.len(),
            state.revoked_session_ids.len(),
        ))
    }

    #[cfg(test)]
    fn has_entry(&self, key: &AwaitEventKey) -> Result<bool, RuntimeError> {
        let state = self.locked_state()?;
        Ok(state.entries.contains_key(&key.key_id))
    }

    #[cfg(test)]
    fn is_completed_turn_control(&self, key: &AwaitEventKey) -> Result<bool, RuntimeError> {
        let state = self.locked_state()?;
        Ok(state.completed_turn_control.contains_key(&key.key_id))
    }

    /// Resolve every *outstanding* wait for `session_id` with
    /// [`Resolution::Cancelled`], leaving the session usable: already-terminal
    /// waits keep their terminal, and waits registered afterwards behave
    /// normally. This is the standalone host lever, in contrast to the
    /// tombstoning [`revoke_session`](Self::revoke_session).
    pub(super) fn cancel_session(&self, session_id: &str) -> Result<(), RuntimeError> {
        let mut state = self.locked_state()?;
        for entry in state.entries.values_mut() {
            if entry.session_id.as_deref() != Some(session_id)
                || entry.turn_control
                || entry.terminal.is_some()
            {
                continue;
            }
            entry.terminal = Some(Resolution::Cancelled);
            entry.notify.notify_waiters();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn_scope(session_id: &str, turn_id: &str) -> ExecutionScope {
        ExecutionScope::turn(session_id, turn_id)
    }

    #[tokio::test]
    async fn completed_turn_control_entries_leave_the_live_registry_and_are_bounded() {
        let registry = AwaitEventRegistry::with_limits(2, 2);
        let scope = turn_scope("bounded-turn-control", "turn-1");
        let gate = registry
            .key_for(&scope, AwaitEventWaitIdentity::TurnCancelGate)
            .expect("gate key");
        let terminal = registry
            .key_for(&scope, AwaitEventWaitIdentity::TurnTerminal)
            .expect("terminal key");

        registry
            .resolve(
                &gate,
                Resolution::Ok(serde_json::json!({ "gate": "sealed" })),
            )
            .expect("resolve gate");
        registry
            .resolve(
                &terminal,
                Resolution::Ok(serde_json::json!({ "terminal": "done" })),
            )
            .expect("resolve terminal");

        assert!(!registry.has_entry(&gate).expect("gate entry lookup"));
        assert!(
            !registry
                .has_entry(&terminal)
                .expect("terminal entry lookup")
        );
        assert!(
            registry
                .is_completed_turn_control(&gate)
                .expect("completed gate lookup")
        );
        assert!(
            registry
                .is_completed_turn_control(&terminal)
                .expect("completed terminal lookup")
        );
        assert!(matches!(
            registry
                .await_resolution(
                    &terminal,
                    CancellationToken::new(),
                    None,
                    &crate::SystemClock,
                )
                .await
                .expect("reattach completed terminal"),
            Resolution::Ok(_)
        ));

        for ordinal in 2..=3 {
            let scope = turn_scope("bounded-turn-control", &format!("turn-{ordinal}"));
            let gate = registry
                .key_for(&scope, AwaitEventWaitIdentity::TurnCancelGate)
                .expect("next gate key");
            let terminal = registry
                .key_for(&scope, AwaitEventWaitIdentity::TurnTerminal)
                .expect("next terminal key");
            registry
                .resolve(&gate, Resolution::Ok(serde_json::json!("sealed")))
                .expect("resolve next gate");
            registry
                .resolve(&terminal, Resolution::Ok(serde_json::json!("done")))
                .expect("resolve next terminal");
        }
        assert_eq!(registry.counts().expect("registry counts"), (0, 2, 0));
    }

    #[tokio::test]
    async fn turn_control_waiter_cancellation_never_resolves_the_gate() {
        let registry = AwaitEventRegistry::with_limits(2, 2);
        let gate = registry
            .key_for(
                &turn_scope("waiter-cancel", "turn"),
                AwaitEventWaitIdentity::TurnCancelGate,
            )
            .expect("gate key");
        let cancel = CancellationToken::new();
        cancel.cancel();
        let error = registry
            .await_resolution(&gate, cancel, None, &crate::SystemClock)
            .await
            .expect_err("cancelled waiter must stop without resolving");
        assert_eq!(error.code.as_str(), "turn_control_wait_cancelled");
        assert!(matches!(
            registry
                .resolve(&gate, Resolution::Ok(serde_json::json!("real-writer")))
                .expect("real gate writer"),
            ResolveOutcome::Accepted
        ));
    }

    #[tokio::test]
    async fn session_revoke_drains_entries_and_bounds_tombstones() {
        let registry = AwaitEventRegistry::with_limits(2, 2);
        let key = registry
            .key_for(
                &turn_scope("revoke-1", "turn"),
                AwaitEventWaitIdentity::TurnTerminal,
            )
            .expect("terminal key");
        let waiter_cancel = CancellationToken::new();
        let wait = registry.await_resolution(&key, waiter_cancel, None, &crate::SystemClock);
        tokio::pin!(wait);
        tokio::select! {
            result = &mut wait => panic!("wait unexpectedly completed: {result:?}"),
            _ = tokio::task::yield_now() => {}
        }
        registry.revoke_session("revoke-1").expect("revoke session");
        assert!(wait.await.is_err());
        assert_eq!(registry.counts().expect("drained counts").0, 0);

        registry.revoke_session("revoke-2").expect("second revoke");
        registry.revoke_session("revoke-3").expect("third revoke");
        assert_eq!(registry.counts().expect("bounded revokes").2, 2);
    }
}
