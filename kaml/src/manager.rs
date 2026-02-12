use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use uuid::Uuid;

use crate::Session;

struct ManagedSession {
    session: Session,
    idle_handle: Option<JoinHandle<()>>,
}

/// Manages Session instances per conversation with idle timeout.
pub struct SessionManager {
    sessions: Arc<Mutex<HashMap<Uuid, ManagedSession>>>,
    idle_timeout_secs: u64,
}

impl SessionManager {
    pub fn new(idle_timeout_secs: u64) -> Self {
        Self {
            sessions: Arc::new(Mutex::new(HashMap::new())),
            idle_timeout_secs,
        }
    }

    /// Take session from pool. Returns None if no live session.
    pub async fn take(&self, id: Uuid) -> Option<Session> {
        let mut map = self.sessions.lock().await;
        if let Some(mut m) = map.remove(&id) {
            if let Some(h) = m.idle_handle.take() {
                h.abort();
            }
            Some(m.session)
        } else {
            None
        }
    }

    /// Return session to pool. Starts idle timer.
    pub async fn put(&self, id: Uuid, session: Session) {
        let sessions = Arc::clone(&self.sessions);
        let secs = self.idle_timeout_secs;
        let idle_handle = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(secs)).await;
            sessions.lock().await.remove(&id);
        });
        self.sessions.lock().await.insert(
            id,
            ManagedSession {
                session,
                idle_handle: Some(idle_handle),
            },
        );
    }

    /// Force-close a session.
    pub async fn destroy(&self, id: Uuid) {
        let mut map = self.sessions.lock().await;
        if let Some(mut m) = map.remove(&id)
            && let Some(h) = m.idle_handle.take()
        {
            h.abort();
        }
    }
}
