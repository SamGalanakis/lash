//! In-process await-event (Durable Wait) registry backing the inline effect
//! host and controller.

use std::collections::{HashMap, HashSet};
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
    terminal: Option<Resolution>,
    notify: Arc<Notify>,
}

impl AwaitEventEntry {
    fn for_key(key: &AwaitEventKey) -> Self {
        Self {
            session_id: key.scope.session_id().map(ToOwned::to_owned),
            terminal: None,
            notify: Arc::new(Notify::new()),
        }
    }
}

#[derive(Debug)]
struct AwaitEventRegistryState {
    entries: HashMap<String, AwaitEventEntry>,
    revoked_key_ids: HashSet<String>,
    revoked_session_ids: HashSet<String>,
}

#[derive(Debug)]
pub(super) struct AwaitEventRegistry {
    secret: Vec<u8>,
    state: std::sync::Mutex<AwaitEventRegistryState>,
}

impl AwaitEventRegistry {
    fn new() -> Self {
        Self {
            secret: uuid::Uuid::new_v4().as_bytes().to_vec(),
            state: std::sync::Mutex::new(AwaitEventRegistryState {
                entries: HashMap::new(),
                revoked_key_ids: HashSet::new(),
                revoked_session_ids: HashSet::new(),
            }),
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
        if state.revoked_key_ids.contains(&key.key_id)
            || key
                .scope
                .session_id()
                .is_some_and(|session_id| state.revoked_session_ids.contains(session_id))
        {
            return Ok(ResolveOutcome::UnknownOrRevoked);
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
        Ok(ResolveOutcome::Accepted)
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
                if state.revoked_key_ids.contains(&key.key_id)
                    || key
                        .scope
                        .session_id()
                        .is_some_and(|session_id| state.revoked_session_ids.contains(session_id))
                {
                    return Err(RuntimeError::new(
                        "await_event_unknown_or_revoked",
                        "await-event key is invalid or revoked",
                    ));
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
                        let _ = self.resolve(key, Resolution::Cancelled)?;
                    }
                    _ = clock.sleep_until(deadline) => {
                        let _ = self.resolve(key, Resolution::Timeout)?;
                    }
                    _ = notify.notified() => {}
                }
            } else {
                tokio::select! {
                    _ = cancel.cancelled() => {
                        let _ = self.resolve(key, Resolution::Cancelled)?;
                    }
                    _ = notify.notified() => {}
                }
            }
        }
    }

    /// Tombstone every await-event for `session_id`: in-flight waiters error
    /// with `await_event_unknown_or_revoked` and future keys for the session
    /// are permanently rejected. This is the session-deletion path.
    pub(super) fn revoke_session(&self, session_id: &str) -> Result<(), RuntimeError> {
        let mut state = self.locked_state()?;
        state.revoked_session_ids.insert(session_id.to_string());
        for entry in state.entries.values() {
            entry.notify.notify_waiters();
        }
        Ok(())
    }

    /// Resolve every *outstanding* wait for `session_id` with
    /// [`Resolution::Cancelled`], leaving the session usable: already-terminal
    /// waits keep their terminal, and waits registered afterwards behave
    /// normally. This is the standalone host lever, in contrast to the
    /// tombstoning [`revoke_session`](Self::revoke_session).
    pub(super) fn cancel_session(&self, session_id: &str) -> Result<(), RuntimeError> {
        let mut state = self.locked_state()?;
        for entry in state.entries.values_mut() {
            if entry.session_id.as_deref() != Some(session_id) || entry.terminal.is_some() {
                continue;
            }
            entry.terminal = Some(Resolution::Cancelled);
            entry.notify.notify_waiters();
        }
        Ok(())
    }
}
