use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::SystemTime;

use lash_trace::{JsonlTraceSink, TraceContext, TraceLevel, TraceSink};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::plugin::PluginError;

use super::{
    InlineRuntimeEffectController, RuntimeEffectController, SessionStoreFactory, TerminationPolicy,
};

pub type ProcessId = String;

/// Durable executable input for a process.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProcessInput {
    ToolCall {
        call: crate::PreparedToolCall,
    },
    LashlangBlock {
        program: serde_json::Value,
        #[serde(default)]
        input: serde_json::Map<String, serde_json::Value>,
        #[serde(default)]
        tool_bindings: Vec<LashlangProcessToolBinding>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        timeout_ms: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        display_name: Option<String>,
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
        #[serde(default, skip_serializing_if = "Option::is_none")]
        line_event: Option<ProcessCommandLineEventSpec>,
    },
    External {
        #[serde(default)]
        metadata: serde_json::Value,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LashlangProcessToolBinding {
    pub name: String,
    pub tool_id: crate::ToolId,
}

/// Optional line-event projection for command-backed processes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessCommandLineEventSpec {
    pub event_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wake_input_template: Option<String>,
}

/// Execution-local context for a process start effect. This is not part of the
/// durable process row.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ProcessExecutionContext {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_effect_metadata: Option<crate::EffectInvocationMetadata>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wake_session_id: Option<String>,
}

impl ProcessExecutionContext {
    pub fn with_tool_effect_metadata(
        mut self,
        metadata: Option<crate::EffectInvocationMetadata>,
    ) -> Self {
        self.tool_effect_metadata = metadata;
        self
    }

    pub fn with_wake_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.wake_session_id = Some(session_id.into());
        self
    }

    pub fn is_empty(&self) -> bool {
        self.tool_effect_metadata.is_none() && self.wake_session_id.is_none()
    }
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
    pub terminal: Option<ProcessTerminalSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wake: Option<ProcessWakeSpec>,
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
    pub terminal: Option<ProcessTerminalSemantics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wake: Option<ProcessWake>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessTerminalState {
    Completed,
    Failed,
    Cancelled,
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

/// Serializable process spec used to start or recover a runtime process.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcessRegistration {
    pub id: ProcessId,
    pub input: ProcessInput,
    #[serde(default)]
    pub event_types: Vec<ProcessEventType>,
}

impl ProcessRegistration {
    pub fn new(id: impl Into<ProcessId>, input: ProcessInput) -> Self {
        Self {
            id: id.into(),
            input,
            event_types: default_process_event_types(),
        }
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
}

pub fn lashlang_process_event_types() -> Vec<ProcessEventType> {
    vec![
        ProcessEventType {
            name: "process.yield".to_string(),
            payload_schema: crate::LashSchema::any(),
            semantics: ProcessEventSemanticsSpec::default(),
        },
        ProcessEventType {
            name: "process.wake".to_string(),
            payload_schema: crate::LashSchema::any(),
            semantics: ProcessEventSemanticsSpec {
                wake: Some(ProcessWakeSpec {
                    when: None,
                    input: ProcessValueSelector::Pointer("/text".to_string()),
                    dedupe_key: ProcessWakeDedupeKey::EventIdentity,
                }),
                ..ProcessEventSemanticsSpec::default()
            },
        },
    ]
}

/// Durable process row. Session-visible addressability lives in
/// [`ProcessHandleGrant`], not in the process record.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcessRecord {
    pub id: ProcessId,
    pub input: ProcessInput,
    #[serde(default)]
    pub event_types: Vec<ProcessEventType>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_ref: Option<ProcessExternalRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal: Option<ProcessTerminalSemantics>,
}

impl ProcessRecord {
    pub fn from_registration(registration: ProcessRegistration) -> Self {
        Self {
            id: registration.id,
            input: registration.input,
            event_types: registration.event_types,
            external_ref: None,
            terminal: None,
        }
    }

    pub fn is_terminal(&self) -> bool {
        self.terminal.is_some()
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcessEvent {
    pub process_id: ProcessId,
    pub sequence: u64,
    pub event_type: String,
    pub payload: serde_json::Value,
    pub semantics: ProcessEventSemantics,
    pub occurred_at: SystemTime,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessHandleDescriptor {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

impl ProcessHandleDescriptor {
    pub fn new(kind: Option<impl Into<String>>, label: Option<impl Into<String>>) -> Self {
        Self {
            kind: kind.map(Into::into),
            label: label.map(Into::into),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessHandleGrant {
    pub session_id: String,
    pub process_id: ProcessId,
    pub descriptor: ProcessHandleDescriptor,
}

pub type ProcessHandleGrantEntry = (ProcessHandleGrant, ProcessRecord);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessStartGrant {
    pub session_id: String,
    pub descriptor: ProcessHandleDescriptor,
}

/// Durability-neutral process registry.
#[async_trait::async_trait]
pub trait ProcessRegistry: Send + Sync {
    async fn register_process(
        &self,
        registration: ProcessRegistration,
    ) -> Result<ProcessRecord, PluginError>;

    async fn set_external_ref(
        &self,
        process_id: &str,
        external_ref: ProcessExternalRef,
    ) -> Result<ProcessRecord, PluginError>;

    async fn grant_handle(
        &self,
        session_id: &str,
        process_id: &str,
        descriptor: ProcessHandleDescriptor,
    ) -> Result<ProcessHandleGrant, PluginError>;

    async fn revoke_handle(&self, session_id: &str, process_id: &str) -> Result<(), PluginError>;

    async fn transfer_handle_grants(
        &self,
        from_session_id: &str,
        to_session_id: &str,
        process_ids: &[String],
    ) -> Result<(), PluginError>;

    async fn list_handle_grants(
        &self,
        session_id: &str,
    ) -> Result<Vec<ProcessHandleGrantEntry>, PluginError>;

    async fn handle_grants_for_process(
        &self,
        process_id: &str,
    ) -> Result<Vec<ProcessHandleGrant>, PluginError>;

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

    async fn complete_process(
        &self,
        process_id: &str,
        await_output: ProcessAwaitOutput,
    ) -> Result<ProcessRecord, PluginError>;

    async fn get_process(&self, process_id: &str) -> Option<ProcessRecord>;

    async fn ack_wake(&self, process_id: &str, sequence: u64) -> Result<(), PluginError>;
}

/// In-memory process registry shared across runtime sessions.
pub struct LocalProcessRegistry {
    managed: Arc<Mutex<ManagedProcessMap>>,
    grants: Arc<Mutex<ManagedGrantMap>>,
}

impl Default for LocalProcessRegistry {
    fn default() -> Self {
        Self {
            managed: Arc::new(Mutex::new(HashMap::new())),
            grants: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

type ManagedProcessMap = HashMap<String, ManagedProcessRecord>;
type ManagedGrantMap = HashMap<String, HashMap<String, ProcessHandleGrant>>;

struct ManagedProcessRecord {
    record: ProcessRecord,
    events: Vec<ProcessEvent>,
    acked_wakes: HashSet<u64>,
    notify: Arc<tokio::sync::Notify>,
}

impl LocalProcessRegistry {
    async fn insert_process(
        &self,
        mut registration: ProcessRegistration,
    ) -> Result<ProcessRecord, PluginError> {
        ensure_core_event_types(&mut registration);
        validate_process_registration(&registration)?;
        let mut managed = self.managed.lock().await;
        if managed.contains_key(&registration.id) {
            return Err(PluginError::Session(format!(
                "process `{}` is already registered",
                registration.id
            )));
        }
        let id = registration.id.clone();
        let record = ProcessRecord::from_registration(registration);
        managed.insert(
            id.clone(),
            ManagedProcessRecord {
                record: record.clone(),
                events: Vec::new(),
                acked_wakes: HashSet::new(),
                notify: Arc::new(tokio::sync::Notify::new()),
            },
        );
        Ok(record)
    }
}

#[async_trait::async_trait]
impl ProcessRegistry for LocalProcessRegistry {
    async fn register_process(
        &self,
        registration: ProcessRegistration,
    ) -> Result<ProcessRecord, PluginError> {
        self.insert_process(registration).await
    }

    async fn set_external_ref(
        &self,
        process_id: &str,
        external_ref: ProcessExternalRef,
    ) -> Result<ProcessRecord, PluginError> {
        let mut managed = self.managed.lock().await;
        let Some(record) = managed.get_mut(process_id) else {
            return Err(PluginError::Session(format!(
                "unknown process `{process_id}`"
            )));
        };
        record.record.external_ref = Some(external_ref);
        Ok(record.record.clone())
    }

    async fn grant_handle(
        &self,
        session_id: &str,
        process_id: &str,
        descriptor: ProcessHandleDescriptor,
    ) -> Result<ProcessHandleGrant, PluginError> {
        if self.get_process(process_id).await.is_none() {
            return Err(PluginError::Session(format!(
                "unknown process `{process_id}`"
            )));
        }
        let grant = ProcessHandleGrant {
            session_id: session_id.to_string(),
            process_id: process_id.to_string(),
            descriptor,
        };
        self.grants
            .lock()
            .await
            .entry(session_id.to_string())
            .or_default()
            .insert(process_id.to_string(), grant.clone());
        Ok(grant)
    }

    async fn revoke_handle(&self, session_id: &str, process_id: &str) -> Result<(), PluginError> {
        if let Some(session_grants) = self.grants.lock().await.get_mut(session_id) {
            session_grants.remove(process_id);
        }
        Ok(())
    }

    async fn transfer_handle_grants(
        &self,
        from_session_id: &str,
        to_session_id: &str,
        process_ids: &[String],
    ) -> Result<(), PluginError> {
        let mut grants = self.grants.lock().await;
        for process_id in process_ids {
            let grant = grants
                .get_mut(from_session_id)
                .and_then(|session_grants| session_grants.remove(process_id))
                .ok_or_else(|| {
                    PluginError::Session(format!(
                        "process handle `{process_id}` is not granted to session `{from_session_id}`"
                    ))
                })?;
            grants.entry(to_session_id.to_string()).or_default().insert(
                process_id.clone(),
                ProcessHandleGrant {
                    session_id: to_session_id.to_string(),
                    process_id: process_id.clone(),
                    descriptor: grant.descriptor,
                },
            );
        }
        Ok(())
    }

    async fn list_handle_grants(
        &self,
        session_id: &str,
    ) -> Result<Vec<ProcessHandleGrantEntry>, PluginError> {
        let grants = self
            .grants
            .lock()
            .await
            .get(session_id)
            .cloned()
            .unwrap_or_default();
        let managed = self.managed.lock().await;
        let mut entries = grants
            .into_values()
            .filter_map(|grant| {
                managed
                    .get(&grant.process_id)
                    .map(|record| (grant, record.record.clone()))
            })
            .collect::<Vec<_>>();
        entries.sort_by(|(left, _), (right, _)| left.process_id.cmp(&right.process_id));
        Ok(entries)
    }

    async fn handle_grants_for_process(
        &self,
        process_id: &str,
    ) -> Result<Vec<ProcessHandleGrant>, PluginError> {
        if self.get_process(process_id).await.is_none() {
            return Err(PluginError::Session(format!(
                "unknown process `{process_id}`"
            )));
        }
        let grants = self.grants.lock().await;
        let mut entries = grants
            .values()
            .filter_map(|session_grants| session_grants.get(process_id).cloned())
            .collect::<Vec<_>>();
        entries.sort_by(|left, right| left.session_id.cmp(&right.session_id));
        Ok(entries)
    }

    async fn append_event(
        &self,
        process_id: &str,
        event_type: String,
        payload: serde_json::Value,
    ) -> Result<ProcessEvent, PluginError> {
        let mut managed = self.managed.lock().await;
        let Some(record) = managed.get_mut(process_id) else {
            return Err(PluginError::Session(format!(
                "unknown process `{process_id}`"
            )));
        };
        let declared = record
            .record
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
        if let Some(terminal) = event.semantics.terminal.clone() {
            record.record.terminal = Some(terminal);
        }
        record.events.push(event.clone());
        if terminal {
            record.notify.notify_waiters();
        }
        Ok(event)
    }

    async fn events_after(
        &self,
        process_id: &str,
        after_sequence: u64,
    ) -> Result<Vec<ProcessEvent>, PluginError> {
        let managed = self.managed.lock().await;
        let Some(record) = managed.get(process_id) else {
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
        let Some(record) = managed.get(process_id) else {
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
                let Some(record) = managed.get(process_id) else {
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

    async fn complete_process(
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
        self.get_process(process_id).await.ok_or_else(|| {
            PluginError::Session(format!(
                "unknown process `{process_id}` after terminal event"
            ))
        })
    }

    async fn get_process(&self, process_id: &str) -> Option<ProcessRecord> {
        let managed = self.managed.lock().await;
        managed.get(process_id).map(|record| record.record.clone())
    }

    async fn ack_wake(&self, process_id: &str, sequence: u64) -> Result<(), PluginError> {
        let mut managed = self.managed.lock().await;
        let Some(record) = managed.get_mut(process_id) else {
            return Err(PluginError::Session(format!(
                "unknown process `{process_id}`"
            )));
        };
        record.acked_wakes.insert(sequence);
        Ok(())
    }
}

fn default_process_event_types() -> Vec<ProcessEventType> {
    vec![
        terminal_event_type("process.completed", ProcessTerminalState::Completed),
        terminal_event_type("process.failed", ProcessTerminalState::Failed),
        terminal_event_type("process.cancelled", ProcessTerminalState::Cancelled),
    ]
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
    Ok(ProcessEventSemantics { terminal, wake })
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
            ProcessInput::External {
                metadata: serde_json::Value::Null,
            },
        )
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
            .register_process(registration("proc-1").with_extra_event_types([event_type]))
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

        assert_eq!(event.sequence, 1);
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
            .register_process(registration("proc-2"))
            .await
            .expect("register");
        registry
            .complete_process(
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
                .get_process("proc-2")
                .await
                .expect("record")
                .is_terminal()
        );
    }

    #[tokio::test]
    async fn transfer_handle_grants_moves_addressability_without_process_events() {
        let registry = LocalProcessRegistry::default();
        registry
            .register_process(registration("proc-3"))
            .await
            .expect("register");
        registry
            .grant_handle(
                "s1",
                "proc-3",
                ProcessHandleDescriptor::new(Some("tool"), Some("demo")),
            )
            .await
            .expect("grant");
        registry
            .transfer_handle_grants("s1", "s2", &["proc-3".to_string()])
            .await
            .expect("transfer");

        assert_eq!(
            registry
                .list_handle_grants("s1")
                .await
                .expect("grants")
                .len(),
            0
        );
        assert_eq!(
            registry
                .list_handle_grants("s2")
                .await
                .expect("grants")
                .len(),
            1
        );
        assert!(
            registry
                .events_after("proc-3", 0)
                .await
                .expect("events")
                .is_empty()
        );
    }

    #[tokio::test]
    async fn multiple_sessions_can_hold_grants_to_one_process() {
        let registry = LocalProcessRegistry::default();
        registry
            .register_process(registration("proc-5"))
            .await
            .expect("register");
        registry
            .grant_handle(
                "s1",
                "proc-5",
                ProcessHandleDescriptor::new(Some("tool"), Some("demo")),
            )
            .await
            .expect("grant s1");
        registry
            .grant_handle(
                "s2",
                "proc-5",
                ProcessHandleDescriptor::new(Some("monitor"), Some("demo")),
            )
            .await
            .expect("grant s2");

        let grant_sessions = registry
            .handle_grants_for_process("proc-5")
            .await
            .expect("process grants")
            .into_iter()
            .map(|grant| grant.session_id)
            .collect::<Vec<_>>();
        assert_eq!(grant_sessions, vec!["s1".to_string(), "s2".to_string()]);

        registry
            .transfer_handle_grants("s1", "s3", &["proc-5".to_string()])
            .await
            .expect("transfer s1");
        let grant_sessions = registry
            .handle_grants_for_process("proc-5")
            .await
            .expect("process grants")
            .into_iter()
            .map(|grant| grant.session_id)
            .collect::<Vec<_>>();
        assert_eq!(grant_sessions, vec!["s2".to_string(), "s3".to_string()]);
        assert!(
            registry
                .events_after("proc-5", 0)
                .await
                .expect("events")
                .is_empty()
        );
    }

    #[tokio::test]
    async fn processes_can_exist_with_zero_grants() {
        let registry = LocalProcessRegistry::default();
        registry
            .register_process(registration("proc-4"))
            .await
            .expect("register");
        assert!(
            registry
                .list_handle_grants("s1")
                .await
                .expect("grants")
                .is_empty()
        );
    }
}
