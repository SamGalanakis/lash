//! In-tree test fixtures shared across the lash crate's test modules.
//!
//! Cuts down on per-test-module `MockSessionManager` boilerplate by
//! providing a configurable mock implementation plus a couple of small
//! builders for common policy / turn fixtures.

use std::sync::Mutex;

use crate::plugin::{
    PluginError, SessionCreateRequest, SessionHandle, SessionManager, SessionSnapshot,
    SessionTurnHandle,
};
use crate::{
    AssembledTurn, AssistantOutput, DoneReason, ExecutionMode, ExecutionSummary, OutputState,
    Provider, ProviderOptions, SessionPolicy, SessionStateEnvelope, TokenUsage, TurnInput,
    TurnStatus,
};

/// Build a `SessionPolicy` populated with the canonical mock provider
/// + model used by lash's in-tree tests.
pub fn mock_session_policy() -> SessionPolicy {
    SessionPolicy {
        provider: Provider::OpenAiGeneric {
            api_key: String::new(),
            base_url: "https://example.invalid/v1".to_string(),
            options: ProviderOptions::default(),
        },
        model: "mock-model".to_string(),
        execution_mode: ExecutionMode::Standard,
        ..Default::default()
    }
}

/// Build an empty `AssembledTurn` whose assistant text is `summary`.
pub fn mock_assembled_turn(session_id: &str, summary: &str) -> AssembledTurn {
    AssembledTurn {
        state: SessionStateEnvelope {
            session_id: session_id.to_string(),
            policy: SessionPolicy {
                execution_mode: ExecutionMode::Standard,
                ..Default::default()
            },
            ..Default::default()
        },
        status: TurnStatus::Completed,
        assistant_output: AssistantOutput {
            safe_text: summary.to_string(),
            raw_text: summary.to_string(),
            state: OutputState::Usable,
        },
        has_plugin_visible_output: false,
        done_reason: DoneReason::ModelStop,
        execution: ExecutionSummary {
            mode: ExecutionMode::Standard,
            had_tool_calls: false,
            had_code_execution: false,
        },
        token_usage: TokenUsage::default(),
        tool_calls: Vec::new(),
        errors: Vec::new(),
        typed_finish: None,
    }
}

/// Configurable mock for the [`SessionManager`] trait. Tests override
/// the snapshot, tool catalog, and turn outcome via the builder
/// methods; mutations (`create_session`, `cancel_turn`, `close_session`)
/// are recorded so tests can assert against them.
pub struct MockSessionManager {
    pub snapshot: SessionSnapshot,
    pub tool_catalog: Vec<serde_json::Value>,
    pub turn: AssembledTurn,
    pub created: Mutex<Vec<SessionCreateRequest>>,
    pub cancelled: Mutex<Vec<String>>,
    pub closed: Mutex<Vec<String>>,
}

impl Default for MockSessionManager {
    fn default() -> Self {
        Self {
            snapshot: SessionStateEnvelope::default(),
            tool_catalog: Vec::new(),
            turn: mock_assembled_turn("root", ""),
            created: Mutex::new(Vec::new()),
            cancelled: Mutex::new(Vec::new()),
            closed: Mutex::new(Vec::new()),
        }
    }
}

impl MockSessionManager {
    pub fn with_snapshot(mut self, snapshot: SessionSnapshot) -> Self {
        self.snapshot = snapshot;
        self
    }

    pub fn with_tool_catalog(mut self, catalog: Vec<serde_json::Value>) -> Self {
        self.tool_catalog = catalog;
        self
    }

    pub fn with_turn(mut self, turn: AssembledTurn) -> Self {
        self.turn = turn;
        self
    }

    /// Snapshot of the requests captured by `create_session`. Panics if
    /// the lock is poisoned (a panic from another test thread).
    pub fn created_snapshot(&self) -> Vec<SessionCreateRequest> {
        self.created.lock().expect("created lock").clone()
    }
}

#[async_trait::async_trait]
impl SessionManager for MockSessionManager {
    async fn snapshot_current(&self) -> Result<SessionSnapshot, PluginError> {
        Ok(self.snapshot.clone())
    }

    async fn snapshot_session(&self, _session_id: &str) -> Result<SessionSnapshot, PluginError> {
        Ok(self.snapshot.clone())
    }

    async fn tool_catalog(&self, _session_id: &str) -> Result<Vec<serde_json::Value>, PluginError> {
        Ok(self.tool_catalog.clone())
    }

    async fn create_session(
        &self,
        request: SessionCreateRequest,
    ) -> Result<SessionHandle, PluginError> {
        self.created
            .lock()
            .expect("created lock")
            .push(request.clone());
        Ok(SessionHandle {
            session_id: request
                .session_id
                .clone()
                .unwrap_or_else(|| "child".to_string()),
            parent_session_id: request.parent_session_id.clone(),
            policy: request.policy.unwrap_or_else(mock_session_policy),
        })
    }

    async fn close_session(&self, session_id: &str) -> Result<(), PluginError> {
        self.closed
            .lock()
            .expect("closed lock")
            .push(session_id.to_string());
        Ok(())
    }

    async fn start_turn_stream(
        &self,
        session_id: &str,
        _input: TurnInput,
    ) -> Result<SessionTurnHandle, PluginError> {
        let (_tx, rx) = tokio::sync::mpsc::channel(1);
        Ok(SessionTurnHandle {
            turn_id: format!("{session_id}-turn"),
            session_id: session_id.to_string(),
            policy: mock_session_policy(),
            events: rx,
        })
    }

    async fn await_turn(&self, _turn_id: &str) -> Result<AssembledTurn, PluginError> {
        Ok(self.turn.clone())
    }

    async fn cancel_turn(&self, turn_id: &str) -> Result<(), PluginError> {
        self.cancelled
            .lock()
            .expect("cancelled lock")
            .push(turn_id.to_string());
        Ok(())
    }
}
