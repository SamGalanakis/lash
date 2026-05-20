use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::SystemTime;

use futures_util::stream::{self, BoxStream, StreamExt};
use lash_trace::{JsonlTraceSink, TraceContext, TraceLevel, TraceSink};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, broadcast};

use crate::plugin::PluginError;

use super::{
    InlineRuntimeEffectController, RuntimeEffectController, SessionStoreFactory, TerminationPolicy,
};

pub type ProcessId = String;

/// Lifecycle state projected from process events.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessState {
    Pending,
    Scheduled,
    Running,
    Waiting,
    Completed,
    Failed,
    CancelRequested,
    Cancelled,
}

impl ProcessState {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

/// Durable executable input for a process.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProcessInput {
    ToolCall {
        call: crate::PreparedToolCall,
    },
    SessionTurn {
        create_request: Box<crate::SessionCreateRequest>,
        turn_input: Box<crate::TurnInput>,
        output_contract: crate::ToolOutputContract,
    },
    Command {
        command: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cwd: Option<String>,
        #[serde(default)]
        env: BTreeMap<String, String>,
        timeout_ms: u64,
        persistent: bool,
    },
    External {
        #[serde(default)]
        metadata: serde_json::Value,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProcessEventType {
    pub name: String,
    pub payload_schema: crate::LashSchema,
    pub semantics: ProcessEventSemanticsSpec,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ProcessEventSemanticsSpec {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<ProcessStateSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal: Option<ProcessTerminalSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wake: Option<ProcessWakeSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transfer: Option<ProcessTransferSpec>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProcessStateSpec {
    pub state: ProcessState,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProcessTerminalSpec {
    pub state: ProcessTerminalState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub await_output: Option<ProcessValueSelector>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProcessWakeSpec {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub when: Option<ProcessValueSelector>,
    pub input: ProcessValueSelector,
    #[serde(default)]
    pub dedupe_key: ProcessWakeDedupeKey,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessWakeDedupeKey {
    #[default]
    EventIdentity,
    Selector(ProcessValueSelector),
    Const(String),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProcessTransferSpec {
    pub session_id: ProcessValueSelector,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessValueSelector {
    Payload,
    Pointer(String),
    Const(serde_json::Value),
    Template {
        template: String,
        #[serde(default)]
        fields: BTreeMap<String, ProcessValueSelector>,
    },
    Present(String),
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ProcessEventSemantics {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<ProcessState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal: Option<ProcessTerminalSemantics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wake: Option<ProcessWake>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transfer: Option<ProcessTransfer>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessTerminalState {
    Completed,
    Failed,
    Cancelled,
}

impl ProcessTerminalState {
    pub fn process_state(self) -> ProcessState {
        match self {
            Self::Completed => ProcessState::Completed,
            Self::Failed => ProcessState::Failed,
            Self::Cancelled => ProcessState::Cancelled,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProcessTerminalSemantics {
    pub state: ProcessTerminalState,
    pub await_output: ProcessAwaitOutput,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProcessAwaitOutput {
    Success {
        value: serde_json::Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        control: Option<crate::ToolControl>,
    },
    Failure {
        class: crate::ToolFailureClass,
        code: String,
        message: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        raw: Option<serde_json::Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        control: Option<crate::ToolControl>,
    },
    Cancelled {
        message: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        raw: Option<serde_json::Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        control: Option<crate::ToolControl>,
    },
}

impl ProcessAwaitOutput {
    pub fn terminal_state(&self) -> ProcessTerminalState {
        match self {
            Self::Success { .. } => ProcessTerminalState::Completed,
            Self::Failure { .. } => ProcessTerminalState::Failed,
            Self::Cancelled { .. } => ProcessTerminalState::Cancelled,
        }
    }

    pub fn from_tool_output(output: crate::ToolCallOutput) -> Self {
        let control = output.control;
        match output.outcome {
            crate::ToolCallOutcome::Success(value) => Self::Success {
                value: value.to_json_value(),
                control,
            },
            crate::ToolCallOutcome::Failure(failure) => Self::Failure {
                class: failure.class,
                code: failure.code,
                message: failure.message,
                raw: failure.raw.map(|value| value.to_json_value()),
                control,
            },
            crate::ToolCallOutcome::Cancelled(cancellation) => Self::Cancelled {
                message: cancellation.message,
                raw: cancellation.raw.map(|value| value.to_json_value()),
                control,
            },
        }
    }

    pub fn into_tool_output(self) -> crate::ToolCallOutput {
        match self {
            Self::Success { value, control } => {
                let mut output = crate::ToolCallOutput::success(value);
                output.control = control;
                output
            }
            Self::Failure {
                class,
                code,
                message,
                raw,
                control,
            } => {
                let mut failure = crate::ToolFailure::tool(class, code, message);
                failure.raw = raw.map(crate::ToolValue::from);
                let mut output = crate::ToolCallOutput::failure(failure);
                output.control = control;
                output
            }
            Self::Cancelled {
                message,
                raw,
                control,
            } => {
                let mut cancellation = crate::ToolCancellation::runtime(message);
                cancellation.raw = raw.map(crate::ToolValue::from);
                let mut output = crate::ToolCallOutput::cancelled(cancellation);
                output.control = control;
                output
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProcessWake {
    pub input: String,
    pub dedupe_key: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessTransfer {
    pub session_id: String,
}

/// Serializable process spec used to start or recover a runtime process.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcessRegistration {
    pub id: ProcessId,
    pub producer: String,
    pub scope: ProcessScope,
    pub child_session_id: Option<String>,
    pub parent_process_id: Option<ProcessId>,
    pub input: ProcessInput,
    #[serde(default)]
    pub event_types: Vec<ProcessEventType>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub metadata: serde_json::Value,
    #[serde(default)]
    pub handle_visible: bool,
    pub attempt: ProcessAttempt,
    pub cancel_policy: ProcessCancelPolicy,
    pub close_policy: ProcessClosePolicy,
}

impl ProcessRegistration {
    pub fn new(
        id: impl Into<ProcessId>,
        producer: impl Into<String>,
        scope: ProcessScope,
        input: ProcessInput,
    ) -> Self {
        Self {
            id: id.into(),
            producer: producer.into(),
            scope,
            child_session_id: None,
            parent_process_id: None,
            input,
            event_types: default_process_event_types(),
            tags: Vec::new(),
            metadata: serde_json::Value::Null,
            handle_visible: false,
            attempt: ProcessAttempt::default(),
            cancel_policy: ProcessCancelPolicy::Cooperative,
            close_policy: ProcessClosePolicy::Keep,
        }
    }

    pub fn with_child_session_id(mut self, child_session_id: impl Into<String>) -> Self {
        self.child_session_id = Some(child_session_id.into());
        self
    }

    pub fn with_parent_process_id(mut self, parent_process_id: impl Into<ProcessId>) -> Self {
        self.parent_process_id = Some(parent_process_id.into());
        self
    }

    pub fn with_optional_parent_process_id(
        mut self,
        parent_process_id: Option<impl Into<ProcessId>>,
    ) -> Self {
        self.parent_process_id = parent_process_id.map(Into::into);
        self
    }

    pub fn with_event_types(
        mut self,
        event_types: impl IntoIterator<Item = ProcessEventType>,
    ) -> Self {
        self.event_types = event_types.into_iter().collect();
        self
    }

    pub fn with_extra_event_types(
        mut self,
        event_types: impl IntoIterator<Item = ProcessEventType>,
    ) -> Self {
        self.event_types.extend(event_types);
        self
    }

    pub fn with_tags(mut self, tags: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tags = tags.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }

    pub fn with_handle_visible(mut self, visible: bool) -> Self {
        self.handle_visible = visible;
        self
    }

    pub fn with_attempt(mut self, attempt: ProcessAttempt) -> Self {
        self.attempt = attempt;
        self
    }

    pub fn with_cancel_policy(mut self, cancel_policy: ProcessCancelPolicy) -> Self {
        self.cancel_policy = cancel_policy;
        self
    }

    pub fn with_close_policy(mut self, close_policy: ProcessClosePolicy) -> Self {
        self.close_policy = close_policy;
        self
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ProcessScope {
    pub session_id: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessCancelPolicy {
    #[default]
    Cooperative,
    External,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessClosePolicy {
    #[default]
    Keep,
    Cancel,
    Transfer,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ProcessAttempt {
    pub attempt: u32,
    pub max_attempts: Option<u32>,
    pub idempotency_key: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ProcessOutcome {
    pub summary: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub await_output: Option<ProcessAwaitOutput>,
}

/// Event-sourced process record projected from registration plus events.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcessRecord {
    pub id: ProcessId,
    pub producer: String,
    pub scope: ProcessScope,
    pub parent_process_id: Option<ProcessId>,
    pub child_session_id: Option<String>,
    pub input: ProcessInput,
    #[serde(default)]
    pub event_types: Vec<ProcessEventType>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub metadata: serde_json::Value,
    #[serde(default)]
    pub handle_visible: bool,
    pub state: ProcessState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal: Option<ProcessTerminalSemantics>,
    pub cancel_policy: ProcessCancelPolicy,
    pub close_policy: ProcessClosePolicy,
    pub attempt: ProcessAttempt,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_ref: Option<ProcessExternalRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub progress: Option<String>,
    pub result: Option<ProcessOutcome>,
    pub failure: Option<ProcessOutcome>,
    #[serde(default)]
    pub event_count: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latest_event: Option<ProcessEvent>,
    pub created_at: SystemTime,
    pub updated_at: SystemTime,
    pub completed_at: Option<SystemTime>,
}

impl ProcessRecord {
    pub fn local_session(
        session_id: impl Into<String>,
        id: impl Into<ProcessId>,
        producer: impl Into<String>,
        state: ProcessState,
    ) -> Self {
        let now = SystemTime::now();
        Self {
            id: id.into(),
            producer: producer.into(),
            scope: ProcessScope {
                session_id: session_id.into(),
            },
            parent_process_id: None,
            child_session_id: None,
            input: ProcessInput::External {
                metadata: serde_json::Value::Null,
            },
            event_types: default_process_event_types(),
            tags: Vec::new(),
            metadata: serde_json::Value::Null,
            handle_visible: false,
            state,
            terminal: None,
            cancel_policy: ProcessCancelPolicy::Cooperative,
            close_policy: ProcessClosePolicy::Keep,
            attempt: ProcessAttempt::default(),
            external_ref: None,
            progress: None,
            result: None,
            failure: None,
            event_count: 0,
            latest_event: None,
            created_at: now,
            updated_at: now,
            completed_at: state.is_terminal().then_some(now),
        }
    }
}

/// Durable backend reference for background work accepted outside the local process.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ProcessExternalRef {
    pub backend: String,
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

/// Serializable receipt returned by a durable process scheduler.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ProcessStartReceipt {
    pub process_id: ProcessId,
    pub state: ProcessState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_ref: Option<ProcessExternalRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcessEvent {
    pub process_id: ProcessId,
    pub sequence: u64,
    pub event_type: String,
    pub payload: serde_json::Value,
    pub semantics: ProcessEventSemantics,
    pub occurred_at: SystemTime,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ProcessFilter {
    pub session_id: Option<String>,
    pub producer: Option<String>,
    pub tags: Vec<String>,
    pub handle_visible: Option<bool>,
    pub include_terminal: bool,
}

/// Durability-neutral process registry.
#[async_trait::async_trait]
pub trait ProcessRegistry: Send + Sync {
    async fn register(
        &self,
        registration: ProcessRegistration,
    ) -> Result<ProcessRecord, PluginError>;

    async fn append_event(
        &self,
        process_id: &str,
        event_type: String,
        payload: serde_json::Value,
    ) -> Result<ProcessEvent, PluginError>;

    async fn events_after(
        &self,
        process_id: &str,
        after_sequence: u64,
    ) -> Result<Vec<ProcessEvent>, PluginError>;

    async fn wake_events_after(
        &self,
        process_id: &str,
        after_sequence: u64,
    ) -> Result<Vec<ProcessEvent>, PluginError>;

    async fn await_process(&self, process_id: &str) -> Result<ProcessAwaitOutput, PluginError>;

    async fn mark_running(&self, process_id: &str) -> Result<ProcessRecord, PluginError> {
        self.append_event(
            process_id,
            "process.running".to_string(),
            serde_json::json!({}),
        )
        .await?;
        self.get(process_id).await.ok_or_else(|| {
            PluginError::Session(format!(
                "unknown process `{process_id}` after running event"
            ))
        })
    }

    async fn mark_scheduled(
        &self,
        receipt: ProcessStartReceipt,
    ) -> Result<ProcessRecord, PluginError>;

    async fn complete(
        &self,
        process_id: &str,
        await_output: ProcessAwaitOutput,
    ) -> Result<ProcessRecord, PluginError>;

    async fn request_cancel(
        &self,
        process_id: &str,
        reason: Option<String>,
    ) -> Result<ProcessRecord, PluginError>;

    async fn get(&self, process_id: &str) -> Option<ProcessRecord>;

    async fn list(&self, filter: ProcessFilter) -> Vec<ProcessRecord>;

    async fn transfer(
        &self,
        process_id: &str,
        new_scope: ProcessScope,
    ) -> Result<ProcessRecord, PluginError>;

    async fn ack_wake(&self, process_id: &str, sequence: u64) -> Result<(), PluginError>;

    fn subscribe(&self, filter: ProcessFilter) -> BoxStream<'static, ProcessEvent>;
}

/// In-memory process registry shared across runtime sessions.
pub struct LocalProcessRegistry {
    managed: Arc<Mutex<ManagedProcessMap>>,
    events: broadcast::Sender<ProcessEvent>,
}

impl Default for LocalProcessRegistry {
    fn default() -> Self {
        let (events, _) = broadcast::channel(256);
        Self {
            managed: Arc::new(Mutex::new(HashMap::new())),
            events,
        }
    }
}

type ManagedProcessMap = HashMap<String, HashMap<String, ManagedProcessRecord>>;

struct ManagedProcessRecord {
    registration: ProcessRegistration,
    events: Vec<ProcessEvent>,
    acked_wakes: HashSet<u64>,
    notify: Arc<tokio::sync::Notify>,
}

fn event_matches_filter(event: &ProcessEvent, filter: &ProcessFilter) -> bool {
    let _ = filter;
    let _ = event;
    true
}

fn record_matches_filter(record: &ProcessRecord, filter: &ProcessFilter) -> bool {
    if filter
        .session_id
        .as_ref()
        .is_some_and(|session_id| &record.scope.session_id != session_id)
    {
        return false;
    }
    if filter
        .producer
        .as_ref()
        .is_some_and(|producer| &record.producer != producer)
    {
        return false;
    }
    if !filter.tags.iter().all(|tag| record.tags.contains(tag)) {
        return false;
    }
    if filter
        .handle_visible
        .is_some_and(|visible| record.handle_visible != visible)
    {
        return false;
    }
    filter.include_terminal || !record.state.is_terminal()
}

impl LocalProcessRegistry {
    fn publish(&self, event: ProcessEvent) {
        let _ = self.events.send(event);
    }

    async fn insert_process(
        &self,
        mut registration: ProcessRegistration,
    ) -> Result<ProcessRecord, PluginError> {
        ensure_core_event_types(&mut registration);
        validate_process_registration(&registration)?;
        let mut managed = self.managed.lock().await;
        let processes = managed
            .entry(registration.scope.session_id.clone())
            .or_default();
        if processes
            .get(&registration.id)
            .is_some_and(|record| !project_process_record(record).state.is_terminal())
        {
            return Err(PluginError::Session(format!(
                "process `{}` is already registered",
                registration.id
            )));
        }
        let id = registration.id.clone();
        processes.insert(
            id.clone(),
            ManagedProcessRecord {
                registration,
                events: Vec::new(),
                acked_wakes: HashSet::new(),
                notify: Arc::new(tokio::sync::Notify::new()),
            },
        );
        drop(managed);
        self.append_event(&id, "process.registered".to_string(), serde_json::json!({}))
            .await?;
        let record = self.get(&id).await.ok_or_else(|| {
            PluginError::Session(format!("unknown process `{id}` after registration"))
        })?;
        Ok(record)
    }
}

#[async_trait::async_trait]
impl ProcessRegistry for LocalProcessRegistry {
    async fn register(
        &self,
        registration: ProcessRegistration,
    ) -> Result<ProcessRecord, PluginError> {
        self.insert_process(registration).await
    }

    async fn append_event(
        &self,
        process_id: &str,
        event_type: String,
        payload: serde_json::Value,
    ) -> Result<ProcessEvent, PluginError> {
        let mut managed = self.managed.lock().await;
        let Some(record) = managed
            .values_mut()
            .find_map(|processes| processes.get_mut(process_id))
        else {
            return Err(PluginError::Session(format!(
                "unknown process `{process_id}`"
            )));
        };
        let declared = record
            .registration
            .event_types
            .iter()
            .find(|declared| declared.name == event_type)
            .ok_or_else(|| {
                PluginError::Session(format!(
                    "process `{process_id}` emitted undeclared event type `{event_type}`"
                ))
            })?;
        declared.payload_schema.validate(&payload).map_err(|err| {
            PluginError::Session(format!("invalid `{event_type}` payload: {err}"))
        })?;
        let sequence = record.events.len() as u64 + 1;
        let semantics =
            materialize_event_semantics(process_id, sequence, &payload, &declared.semantics)?;
        let event = ProcessEvent {
            process_id: process_id.to_string(),
            sequence,
            event_type,
            payload,
            semantics,
            occurred_at: SystemTime::now(),
        };
        let terminal = event.semantics.terminal.is_some();
        record.events.push(event.clone());
        if terminal {
            record.notify.notify_waiters();
        }
        drop(managed);
        self.publish(event.clone());
        Ok(event)
    }

    async fn events_after(
        &self,
        process_id: &str,
        after_sequence: u64,
    ) -> Result<Vec<ProcessEvent>, PluginError> {
        let managed = self.managed.lock().await;
        let Some(record) = managed
            .values()
            .find_map(|processes| processes.get(process_id))
        else {
            return Err(PluginError::Session(format!(
                "unknown process `{process_id}`"
            )));
        };
        Ok(record
            .events
            .iter()
            .filter(|event| event.sequence > after_sequence)
            .cloned()
            .collect())
    }

    async fn wake_events_after(
        &self,
        process_id: &str,
        after_sequence: u64,
    ) -> Result<Vec<ProcessEvent>, PluginError> {
        let managed = self.managed.lock().await;
        let Some(record) = managed
            .values()
            .find_map(|processes| processes.get(process_id))
        else {
            return Err(PluginError::Session(format!(
                "unknown process `{process_id}`"
            )));
        };
        Ok(record
            .events
            .iter()
            .filter(|event| event.sequence > after_sequence)
            .filter(|event| event.semantics.wake.is_some())
            .filter(|event| !record.acked_wakes.contains(&event.sequence))
            .cloned()
            .collect())
    }

    async fn await_process(&self, process_id: &str) -> Result<ProcessAwaitOutput, PluginError> {
        loop {
            let notify = {
                let managed = self.managed.lock().await;
                let Some(record) = managed
                    .values()
                    .find_map(|processes| processes.get(process_id))
                else {
                    return Err(PluginError::Session(format!(
                        "unknown process `{process_id}`"
                    )));
                };
                if let Some(terminal) = record
                    .events
                    .iter()
                    .find_map(|event| event.semantics.terminal.clone())
                {
                    return Ok(terminal.await_output);
                }
                Arc::clone(&record.notify)
            };
            notify.notified().await;
        }
    }

    async fn mark_scheduled(
        &self,
        receipt: ProcessStartReceipt,
    ) -> Result<ProcessRecord, PluginError> {
        let state = if receipt.state == ProcessState::Pending {
            ProcessState::Scheduled
        } else {
            receipt.state
        };
        self.append_event(
            &receipt.process_id,
            match state {
                ProcessState::Running => "process.running",
                ProcessState::Waiting => "process.waiting",
                _ => "process.scheduled",
            }
            .to_string(),
            serde_json::json!({
                "external_ref": receipt.external_ref,
                "message": receipt.message,
            }),
        )
        .await?;
        self.get(&receipt.process_id).await.ok_or_else(|| {
            PluginError::Session(format!("unknown process `{}`", receipt.process_id))
        })
    }

    async fn complete(
        &self,
        process_id: &str,
        await_output: ProcessAwaitOutput,
    ) -> Result<ProcessRecord, PluginError> {
        let event_type = match await_output.terminal_state() {
            ProcessTerminalState::Completed => "process.completed",
            ProcessTerminalState::Failed => "process.failed",
            ProcessTerminalState::Cancelled => "process.cancelled",
        };
        self.append_event(
            process_id,
            event_type.to_string(),
            serde_json::json!({ "await_output": await_output }),
        )
        .await?;
        self.get(process_id).await.ok_or_else(|| {
            PluginError::Session(format!(
                "unknown process `{process_id}` after terminal event"
            ))
        })
    }

    async fn request_cancel(
        &self,
        process_id: &str,
        reason: Option<String>,
    ) -> Result<ProcessRecord, PluginError> {
        self.append_event(
            process_id,
            "process.cancel_requested".to_string(),
            serde_json::json!({ "reason": reason }),
        )
        .await?;
        self.get(process_id).await.ok_or_else(|| {
            PluginError::Session(format!(
                "unknown process `{process_id}` after cancel request"
            ))
        })
    }

    async fn get(&self, process_id: &str) -> Option<ProcessRecord> {
        let managed = self.managed.lock().await;
        managed
            .values()
            .find_map(|processes| processes.get(process_id).map(project_process_record))
    }

    async fn list(&self, filter: ProcessFilter) -> Vec<ProcessRecord> {
        let managed = self.managed.lock().await;
        let mut out = managed
            .values()
            .flat_map(|processes| processes.values())
            .map(project_process_record)
            .filter(|record| record_matches_filter(record, &filter))
            .collect::<Vec<_>>();
        out.sort_by_key(|record| record.created_at);
        out
    }

    async fn transfer(
        &self,
        process_id: &str,
        new_scope: ProcessScope,
    ) -> Result<ProcessRecord, PluginError> {
        self.append_event(
            process_id,
            "process.transferred".to_string(),
            serde_json::json!({ "session_id": new_scope.session_id }),
        )
        .await?;
        self.get(process_id).await.ok_or_else(|| {
            PluginError::Session(format!("unknown process `{process_id}` after transfer"))
        })
    }

    async fn ack_wake(&self, process_id: &str, sequence: u64) -> Result<(), PluginError> {
        let mut managed = self.managed.lock().await;
        let Some(record) = managed
            .values_mut()
            .find_map(|processes| processes.get_mut(process_id))
        else {
            return Err(PluginError::Session(format!(
                "unknown process `{process_id}`"
            )));
        };
        record.acked_wakes.insert(sequence);
        Ok(())
    }

    fn subscribe(&self, filter: ProcessFilter) -> BoxStream<'static, ProcessEvent> {
        let rx = self.events.subscribe();
        stream::unfold((rx, filter), |(mut rx, filter)| async move {
            loop {
                match rx.recv().await {
                    Ok(event) if event_matches_filter(&event, &filter) => {
                        return Some((event, (rx, filter)));
                    }
                    Ok(_) => continue,
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => return None,
                }
            }
        })
        .boxed()
    }
}

fn default_process_event_types() -> Vec<ProcessEventType> {
    vec![
        state_event_type("process.registered", ProcessState::Pending),
        state_event_type("process.scheduled", ProcessState::Scheduled),
        state_event_type("process.running", ProcessState::Running),
        state_event_type("process.waiting", ProcessState::Waiting),
        state_event_type("process.cancel_requested", ProcessState::CancelRequested),
        terminal_event_type("process.completed", ProcessTerminalState::Completed),
        terminal_event_type("process.failed", ProcessTerminalState::Failed),
        terminal_event_type("process.cancelled", ProcessTerminalState::Cancelled),
        ProcessEventType {
            name: "process.transferred".to_string(),
            payload_schema: crate::LashSchema::any(),
            semantics: ProcessEventSemanticsSpec {
                transfer: Some(ProcessTransferSpec {
                    session_id: ProcessValueSelector::Pointer("/session_id".to_string()),
                }),
                ..ProcessEventSemanticsSpec::default()
            },
        },
    ]
}

fn state_event_type(name: &str, state: ProcessState) -> ProcessEventType {
    ProcessEventType {
        name: name.to_string(),
        payload_schema: crate::LashSchema::any(),
        semantics: ProcessEventSemanticsSpec {
            state: Some(ProcessStateSpec { state }),
            ..ProcessEventSemanticsSpec::default()
        },
    }
}

fn terminal_event_type(name: &str, state: ProcessTerminalState) -> ProcessEventType {
    ProcessEventType {
        name: name.to_string(),
        payload_schema: crate::LashSchema::any(),
        semantics: ProcessEventSemanticsSpec {
            terminal: Some(ProcessTerminalSpec {
                state,
                await_output: Some(ProcessValueSelector::Pointer("/await_output".to_string())),
            }),
            ..ProcessEventSemanticsSpec::default()
        },
    }
}

fn ensure_core_event_types(registration: &mut ProcessRegistration) {
    let mut existing = registration
        .event_types
        .iter()
        .map(|event_type| event_type.name.clone())
        .collect::<HashSet<_>>();
    for event_type in default_process_event_types() {
        if existing.insert(event_type.name.clone()) {
            registration.event_types.push(event_type);
        }
    }
}

fn validate_process_registration(registration: &ProcessRegistration) -> Result<(), PluginError> {
    if registration.id.trim().is_empty() {
        return Err(PluginError::Session(
            "process id must be a non-empty string".to_string(),
        ));
    }
    let mut names = HashSet::new();
    for event_type in &registration.event_types {
        if event_type.name.trim().is_empty() {
            return Err(PluginError::Session(format!(
                "process `{}` declares an empty event type",
                registration.id
            )));
        }
        if !names.insert(event_type.name.as_str()) {
            return Err(PluginError::Session(format!(
                "process `{}` declares duplicate event type `{}`",
                registration.id, event_type.name
            )));
        }
        if let Some(terminal) = &event_type.semantics.terminal
            && terminal.state != ProcessTerminalState::Completed
            && terminal.await_output.is_none()
        {
            return Err(PluginError::Session(format!(
                "terminal event `{}` for process `{}` must declare await output",
                event_type.name, registration.id
            )));
        }
    }
    Ok(())
}

fn materialize_event_semantics(
    process_id: &str,
    sequence: u64,
    payload: &serde_json::Value,
    spec: &ProcessEventSemanticsSpec,
) -> Result<ProcessEventSemantics, PluginError> {
    let state = spec.state.as_ref().map(|state| state.state);
    let terminal = spec
        .terminal
        .as_ref()
        .map(|terminal| materialize_terminal_semantics(payload, terminal))
        .transpose()?;
    let wake = spec
        .wake
        .as_ref()
        .map(|wake| materialize_wake(process_id, sequence, payload, wake))
        .transpose()?
        .flatten();
    let transfer = spec
        .transfer
        .as_ref()
        .map(|transfer| {
            let session_id = select_value(payload, &transfer.session_id)?;
            Ok::<ProcessTransfer, PluginError>(ProcessTransfer {
                session_id: selector_value_to_string(&session_id),
            })
        })
        .transpose()?;
    Ok(ProcessEventSemantics {
        state,
        terminal,
        wake,
        transfer,
    })
}

fn materialize_terminal_semantics(
    payload: &serde_json::Value,
    terminal: &ProcessTerminalSpec,
) -> Result<ProcessTerminalSemantics, PluginError> {
    let await_output = match &terminal.await_output {
        Some(selector) => {
            let selected = select_value(payload, selector)?;
            serde_json::from_value::<ProcessAwaitOutput>(selected.clone())
                .unwrap_or_else(|_| selected_value_to_await_output(terminal.state, selected))
        }
        None if terminal.state == ProcessTerminalState::Completed => ProcessAwaitOutput::Success {
            value: payload.clone(),
            control: None,
        },
        None => {
            return Err(PluginError::Session(
                "failed or cancelled terminal events must declare await output".to_string(),
            ));
        }
    };
    Ok(ProcessTerminalSemantics {
        state: terminal.state,
        await_output,
    })
}

fn selected_value_to_await_output(
    state: ProcessTerminalState,
    value: serde_json::Value,
) -> ProcessAwaitOutput {
    match state {
        ProcessTerminalState::Completed => ProcessAwaitOutput::Success {
            value,
            control: None,
        },
        ProcessTerminalState::Failed => ProcessAwaitOutput::Failure {
            class: crate::ToolFailureClass::Execution,
            code: "process_failed".to_string(),
            message: selector_value_to_string(&value),
            raw: Some(value),
            control: None,
        },
        ProcessTerminalState::Cancelled => ProcessAwaitOutput::Cancelled {
            message: selector_value_to_string(&value),
            raw: Some(value),
            control: None,
        },
    }
}

fn materialize_wake(
    process_id: &str,
    sequence: u64,
    payload: &serde_json::Value,
    wake: &ProcessWakeSpec,
) -> Result<Option<ProcessWake>, PluginError> {
    if let Some(when) = &wake.when {
        let selected = select_value(payload, when)?;
        if !selector_value_is_truthy(&selected) {
            return Ok(None);
        }
    }
    let input = selector_value_to_string(&select_value(payload, &wake.input)?);
    let dedupe_key = match &wake.dedupe_key {
        ProcessWakeDedupeKey::EventIdentity => format!("{process_id}:{sequence}"),
        ProcessWakeDedupeKey::Selector(selector) => {
            selector_value_to_string(&select_value(payload, selector)?)
        }
        ProcessWakeDedupeKey::Const(value) => value.clone(),
    };
    Ok(Some(ProcessWake { input, dedupe_key }))
}

fn select_value(
    payload: &serde_json::Value,
    selector: &ProcessValueSelector,
) -> Result<serde_json::Value, PluginError> {
    match selector {
        ProcessValueSelector::Payload => Ok(payload.clone()),
        ProcessValueSelector::Pointer(pointer) => {
            payload.pointer(pointer).cloned().ok_or_else(|| {
                PluginError::Session(format!("payload pointer `{pointer}` did not match"))
            })
        }
        ProcessValueSelector::Const(value) => Ok(value.clone()),
        ProcessValueSelector::Template { template, fields } => {
            let mut rendered = template.clone();
            for (name, selector) in fields {
                let value = select_value(payload, selector)?;
                rendered =
                    rendered.replace(&format!("{{{name}}}"), &selector_value_to_string(&value));
            }
            Ok(serde_json::Value::String(rendered))
        }
        ProcessValueSelector::Present(pointer) => {
            Ok(serde_json::Value::Bool(payload.pointer(pointer).is_some()))
        }
    }
}

fn selector_value_to_string(value: &serde_json::Value) -> String {
    value
        .as_str()
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| value.to_string())
}

fn selector_value_is_truthy(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Null => false,
        serde_json::Value::Bool(value) => *value,
        serde_json::Value::String(value) => !value.is_empty(),
        serde_json::Value::Array(value) => !value.is_empty(),
        serde_json::Value::Object(value) => !value.is_empty(),
        serde_json::Value::Number(_) => true,
    }
}

fn project_process_record(record: &ManagedProcessRecord) -> ProcessRecord {
    let now = SystemTime::now();
    let mut projected = ProcessRecord::local_session(
        record.registration.scope.session_id.clone(),
        record.registration.id.clone(),
        record.registration.producer.clone(),
        ProcessState::Pending,
    );
    projected.parent_process_id = record.registration.parent_process_id.clone();
    projected.child_session_id = record.registration.child_session_id.clone();
    projected.input = record.registration.input.clone();
    projected.event_types = record.registration.event_types.clone();
    projected.tags = record.registration.tags.clone();
    projected.metadata = record.registration.metadata.clone();
    projected.handle_visible = record.registration.handle_visible;
    projected.cancel_policy = record.registration.cancel_policy.clone();
    projected.close_policy = record.registration.close_policy.clone();
    projected.attempt = record.registration.attempt.clone();
    projected.event_count = record.events.len() as u64;
    projected.latest_event = record.events.last().cloned();
    projected.created_at = record
        .events
        .first()
        .map(|event| event.occurred_at)
        .unwrap_or(now);
    projected.updated_at = record
        .events
        .last()
        .map(|event| event.occurred_at)
        .unwrap_or(projected.created_at);

    for event in &record.events {
        if let Some(state) = event.semantics.state {
            projected.state = state;
        }
        if let Some(transfer) = &event.semantics.transfer {
            projected.scope.session_id = transfer.session_id.clone();
        }
        if let Some(message) = event
            .payload
            .get("message")
            .and_then(serde_json::Value::as_str)
            .filter(|message| !message.is_empty())
        {
            projected.progress = Some(message.to_string());
        }
        if let Some(external_ref) = event
            .payload
            .get("external_ref")
            .and_then(|value| serde_json::from_value::<ProcessExternalRef>(value.clone()).ok())
        {
            projected.external_ref = Some(external_ref);
        }
        if let Some(terminal) = event.semantics.terminal.clone() {
            projected.state = terminal.state.process_state();
            projected.terminal = Some(terminal.clone());
            projected.completed_at = Some(event.occurred_at);
            let outcome = ProcessOutcome {
                summary: event
                    .payload
                    .get("summary")
                    .and_then(serde_json::Value::as_str)
                    .map(ToOwned::to_owned),
                await_output: Some(terminal.await_output),
            };
            if projected.state == ProcessState::Failed {
                projected.failure = Some(outcome);
            } else {
                projected.result = Some(outcome);
            }
        }
    }
    projected
}

/// Required host configuration for all runtimes.
#[derive(Clone)]
pub struct RuntimeCoreConfig {
    pub attachment_store: Arc<dyn crate::AttachmentStore>,
    pub prompt: crate::PromptLayer,
    pub trace_sink: Option<Arc<dyn TraceSink>>,
    pub trace_level: TraceLevel,
    pub trace_context: TraceContext,
    pub termination: TerminationPolicy,
    pub effect_controller: Arc<dyn RuntimeEffectController>,
}

impl Default for RuntimeCoreConfig {
    fn default() -> Self {
        Self {
            attachment_store: Arc::new(crate::InMemoryAttachmentStore::new()),
            prompt: crate::PromptLayer::new(),
            trace_sink: None,
            trace_level: TraceLevel::Standard,
            trace_context: TraceContext::default(),
            termination: TerminationPolicy::default(),
            effect_controller: Arc::new(InlineRuntimeEffectController::default()),
        }
    }
}

impl RuntimeCoreConfig {
    pub fn with_attachment_store(
        mut self,
        attachment_store: Arc<dyn crate::AttachmentStore>,
    ) -> Self {
        self.attachment_store = attachment_store;
        self
    }

    pub fn with_prompt_template(mut self, prompt_template: crate::PromptTemplate) -> Self {
        self.prompt.template = Some(prompt_template);
        self
    }

    pub fn with_prompt_contribution(mut self, contribution: crate::PromptContribution) -> Self {
        self.prompt.add_contribution(contribution);
        self
    }

    pub fn with_replaced_prompt_slot(
        mut self,
        slot: crate::PromptSlot,
        contributions: impl IntoIterator<Item = crate::PromptContribution>,
    ) -> Self {
        self.prompt.replace_slot(slot, contributions);
        self
    }

    pub fn with_cleared_prompt_slot(mut self, slot: crate::PromptSlot) -> Self {
        self.prompt.clear_slot(slot);
        self
    }

    pub fn with_prompt_layer(mut self, prompt: crate::PromptLayer) -> Self {
        self.prompt = prompt;
        self
    }

    pub fn with_trace_jsonl_path(mut self, trace_path: Option<PathBuf>) -> Self {
        self.trace_sink =
            trace_path.map(|path| Arc::new(JsonlTraceSink::new(path)) as Arc<dyn TraceSink>);
        self
    }

    pub fn with_trace_sink(mut self, sink: Option<Arc<dyn TraceSink>>) -> Self {
        self.trace_sink = sink;
        self
    }

    pub fn with_trace_level(mut self, level: TraceLevel) -> Self {
        self.trace_level = level;
        self
    }

    pub fn with_trace_context(mut self, context: TraceContext) -> Self {
        self.trace_context = context;
        self
    }

    pub fn with_termination(mut self, termination: TerminationPolicy) -> Self {
        self.termination = termination;
        self
    }

    pub fn with_effect_controller(
        mut self,
        effect_controller: Arc<dyn RuntimeEffectController>,
    ) -> Self {
        self.effect_controller = effect_controller;
        self
    }
}

/// Base host shape for embedded runtimes.
#[derive(Clone)]
pub struct EmbeddedRuntimeHost {
    pub core: RuntimeCoreConfig,
    pub session_store_factory: Option<Arc<dyn SessionStoreFactory>>,
}

impl EmbeddedRuntimeHost {
    pub fn new(core: RuntimeCoreConfig) -> Self {
        Self {
            core,
            session_store_factory: None,
        }
    }

    pub fn with_session_store_factory(
        mut self,
        session_store_factory: Arc<dyn SessionStoreFactory>,
    ) -> Self {
        self.session_store_factory = Some(session_store_factory);
        self
    }
}

/// Host shape for runtimes that support background plugin work.
#[derive(Clone)]
pub struct ProcessRuntimeHost {
    pub embedded: EmbeddedRuntimeHost,
    pub process_registry: Arc<dyn ProcessRegistry>,
}

impl ProcessRuntimeHost {
    pub fn new(embedded: EmbeddedRuntimeHost, process_registry: Arc<dyn ProcessRegistry>) -> Self {
        Self {
            embedded,
            process_registry,
        }
    }
}

#[derive(Clone)]
pub(crate) struct RuntimeHost {
    pub core: RuntimeCoreConfig,
    pub session_store_factory: Option<Arc<dyn SessionStoreFactory>>,
    pub process_registry: Option<Arc<dyn ProcessRegistry>>,
}

impl From<EmbeddedRuntimeHost> for RuntimeHost {
    fn from(value: EmbeddedRuntimeHost) -> Self {
        Self {
            core: value.core,
            session_store_factory: value.session_store_factory,
            process_registry: Some(Arc::new(LocalProcessRegistry::default())),
        }
    }
}

impl From<ProcessRuntimeHost> for RuntimeHost {
    fn from(value: ProcessRuntimeHost) -> Self {
        Self {
            core: value.embedded.core,
            session_store_factory: value.embedded.session_store_factory,
            process_registry: Some(value.process_registry),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registration(id: &str) -> ProcessRegistration {
        ProcessRegistration::new(
            id,
            "test",
            ProcessScope {
                session_id: "s1".to_string(),
            },
            ProcessInput::External {
                metadata: serde_json::Value::Null,
            },
        )
        .with_tags(["demo"])
        .with_handle_visible(true)
    }

    #[test]
    fn selector_extracts_payload_pointer_const_template_and_present() {
        let payload = serde_json::json!({
            "line": "done",
            "wake_input": "wake me"
        });

        assert_eq!(
            select_value(&payload, &ProcessValueSelector::Payload).unwrap(),
            payload
        );
        assert_eq!(
            select_value(
                &payload,
                &ProcessValueSelector::Pointer("/line".to_string())
            )
            .unwrap(),
            serde_json::json!("done")
        );
        assert_eq!(
            select_value(
                &payload,
                &ProcessValueSelector::Const(serde_json::json!({"ok": true}))
            )
            .unwrap(),
            serde_json::json!({"ok": true})
        );
        assert_eq!(
            select_value(
                &payload,
                &ProcessValueSelector::Template {
                    template: "event: {line}".to_string(),
                    fields: BTreeMap::from([(
                        "line".to_string(),
                        ProcessValueSelector::Pointer("/line".to_string())
                    )]),
                },
            )
            .unwrap(),
            serde_json::json!("event: done")
        );
        assert_eq!(
            select_value(
                &payload,
                &ProcessValueSelector::Present("/wake_input".to_string())
            )
            .unwrap(),
            serde_json::json!(true)
        );
    }

    #[tokio::test]
    async fn process_registry_validates_custom_events_and_materializes_wakes() {
        let registry = LocalProcessRegistry::default();
        let mut properties = serde_json::Map::new();
        properties.insert("line".to_string(), serde_json::json!({ "type": "string" }));
        properties.insert(
            "wake_input".to_string(),
            serde_json::json!({ "type": "string" }),
        );
        let event_type = ProcessEventType {
            name: "monitor.line".to_string(),
            payload_schema: crate::LashSchema::object(properties, vec!["line".to_string()]),
            semantics: ProcessEventSemanticsSpec {
                wake: Some(ProcessWakeSpec {
                    when: Some(ProcessValueSelector::Present("/wake_input".to_string())),
                    input: ProcessValueSelector::Pointer("/wake_input".to_string()),
                    dedupe_key: ProcessWakeDedupeKey::EventIdentity,
                }),
                ..ProcessEventSemanticsSpec::default()
            },
        };
        registry
            .register(registration("proc-1").with_extra_event_types([event_type]))
            .await
            .expect("register");

        let event = registry
            .append_event(
                "proc-1",
                "monitor.line".to_string(),
                serde_json::json!({
                    "line": "deploy failed",
                    "wake_input": "Monitor event: deploy failed"
                }),
            )
            .await
            .expect("append");

        assert_eq!(event.sequence, 2);
        assert_eq!(
            event
                .semantics
                .wake
                .as_ref()
                .map(|wake| wake.input.as_str()),
            Some("Monitor event: deploy failed")
        );
        assert_eq!(
            registry
                .wake_events_after("proc-1", 0)
                .await
                .expect("wake events")
                .len(),
            1
        );
        registry
            .ack_wake("proc-1", event.sequence)
            .await
            .expect("ack wake");
        assert!(
            registry
                .wake_events_after("proc-1", 0)
                .await
                .expect("wake events")
                .is_empty()
        );
        assert!(
            registry
                .append_event(
                    "proc-1",
                    "monitor.line".to_string(),
                    serde_json::json!({ "wake_input": "missing required line" }),
                )
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn await_process_reads_terminal_event_materialized_output() {
        let registry = LocalProcessRegistry::default();
        registry
            .register(registration("proc-2"))
            .await
            .expect("register");
        registry.mark_running("proc-2").await.expect("running");
        registry
            .complete(
                "proc-2",
                ProcessAwaitOutput::Success {
                    value: serde_json::json!({ "ok": true }),
                    control: None,
                },
            )
            .await
            .expect("complete");

        assert_eq!(
            registry.await_process("proc-2").await.expect("await"),
            ProcessAwaitOutput::Success {
                value: serde_json::json!({ "ok": true }),
                control: None,
            }
        );
        assert!(
            registry
                .list(ProcessFilter {
                    session_id: Some("s1".to_string()),
                    producer: Some("test".to_string()),
                    tags: vec!["demo".to_string()],
                    handle_visible: Some(true),
                    include_terminal: false,
                })
                .await
                .is_empty()
        );
    }

    #[tokio::test]
    async fn transfer_scope_is_event_sourced() {
        let registry = LocalProcessRegistry::default();
        registry
            .register(registration("proc-3"))
            .await
            .expect("register");
        registry
            .transfer(
                "proc-3",
                ProcessScope {
                    session_id: "s2".to_string(),
                },
            )
            .await
            .expect("transfer");

        assert_eq!(
            registry
                .get("proc-3")
                .await
                .expect("record")
                .scope
                .session_id,
            "s2"
        );
    }
}
