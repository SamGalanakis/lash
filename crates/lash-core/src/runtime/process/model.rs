use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::events::{
    ProcessAwaitOutput, ProcessEventType, ProcessTerminalSemantics, ProcessTerminalState,
    default_process_event_types,
};
use super::time::current_epoch_ms;
use super::validation::{
    ensure_core_event_types, process_registration_hash, validate_process_registration,
};

pub type ProcessId = String;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ProcessScopeId(String);

impl ProcessScopeId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ProcessScopeId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl From<String> for ProcessScopeId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<&str> for ProcessScopeId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

/// Durable executable input for a process.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProcessInput {
    ToolCall {
        call: crate::PreparedToolCall,
    },
    LashlangProcess {
        module_ref: lashlang::ModuleRef,
        process_ref: lashlang::ProcessRef,
        required_surface_ref: lashlang::RequiredSurfaceRef,
        process_name: String,
        #[serde(default)]
        args: serde_json::Map<String, serde_json::Value>,
    },
    SessionTurn {
        create_request: Box<crate::SessionCreateRequest>,
        turn_input: Box<crate::TurnInput>,
        output_contract: crate::ToolOutputContract,
    },
    External {
        #[serde(default)]
        metadata: serde_json::Value,
    },
}

impl Clone for ProcessInput {
    fn clone(&self) -> Self {
        match self {
            Self::ToolCall { call } => Self::ToolCall { call: call.clone() },
            Self::LashlangProcess {
                module_ref,
                process_ref,
                required_surface_ref,
                process_name,
                args,
            } => Self::LashlangProcess {
                module_ref: module_ref.clone(),
                process_ref: process_ref.clone(),
                required_surface_ref: required_surface_ref.clone(),
                process_name: process_name.clone(),
                args: args.clone(),
            },
            Self::SessionTurn {
                create_request,
                turn_input,
                output_contract,
            } => Self::SessionTurn {
                create_request: create_request.clone(),
                turn_input: turn_input.clone(),
                output_contract: output_contract.clone(),
            },
            Self::External { metadata } => Self::External {
                metadata: metadata.clone(),
            },
        }
    }
}

/// Execution-local context for a process start effect. This is not part of the
/// durable process row.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ProcessExecutionContext {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub causal_invocation: Option<crate::RuntimeInvocation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wake_target_scope: Option<ProcessScope>,
}

impl ProcessExecutionContext {
    pub fn with_causal_invocation(mut self, invocation: Option<crate::RuntimeInvocation>) -> Self {
        self.causal_invocation = invocation;
        self
    }

    pub fn with_wake_target_scope(mut self, scope: ProcessScope) -> Self {
        self.wake_target_scope = Some(scope);
        self
    }

    pub fn is_empty(&self) -> bool {
        self.causal_invocation.is_none() && self.wake_target_scope.is_none()
    }
}

#[derive(Clone, Default)]
pub struct ProcessOpScope<'scope> {
    pub parent_invocation: Option<crate::RuntimeInvocation>,
    pub effect_controller: Option<&'scope dyn crate::RuntimeEffectController>,
    pub turn_lease: Option<crate::RuntimeTurnLease>,
    pub agent_frame_id: Option<crate::AgentFrameId>,
    pub target_agent_frame_id: Option<crate::AgentFrameId>,
}

impl<'scope> ProcessOpScope<'scope> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_parent_invocation(
        mut self,
        parent_invocation: Option<crate::RuntimeInvocation>,
    ) -> Self {
        self.parent_invocation = parent_invocation;
        self
    }

    pub fn with_effect_controller(
        mut self,
        effect_controller: &'scope dyn crate::RuntimeEffectController,
    ) -> Self {
        self.effect_controller = Some(effect_controller);
        self
    }

    pub fn with_turn_lease(mut self, turn_lease: Option<crate::RuntimeTurnLease>) -> Self {
        self.turn_lease = turn_lease;
        self
    }

    pub fn with_agent_frame_id(mut self, agent_frame_id: Option<crate::AgentFrameId>) -> Self {
        self.agent_frame_id = agent_frame_id;
        self
    }

    pub fn with_target_agent_frame_id(
        mut self,
        agent_frame_id: Option<crate::AgentFrameId>,
    ) -> Self {
        self.target_agent_frame_id = agent_frame_id;
        self
    }
}

#[derive(Clone, Debug, Default)]
pub struct ProcessStartOptions {
    pub descriptor: Option<ProcessHandleDescriptor>,
}

impl ProcessStartOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_descriptor(mut self, descriptor: ProcessHandleDescriptor) -> Self {
        self.descriptor = Some(descriptor);
        self
    }

    pub fn with_optional_descriptor(mut self, descriptor: Option<ProcessHandleDescriptor>) -> Self {
        self.descriptor = descriptor;
        self
    }

    pub fn execution_context(&self, scope: &ProcessOpScope<'_>) -> ProcessExecutionContext {
        ProcessExecutionContext {
            causal_invocation: scope.parent_invocation.clone(),
            wake_target_scope: None,
        }
    }
}

/// Public host-facing request for starting a visible process handle.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcessStartRequest {
    pub id: ProcessId,
    pub input: ProcessInput,
    pub descriptor: ProcessHandleDescriptor,
    #[serde(default)]
    pub event_types: Vec<ProcessEventType>,
}

impl ProcessStartRequest {
    pub fn new(
        id: impl Into<ProcessId>,
        input: ProcessInput,
        descriptor: ProcessHandleDescriptor,
    ) -> Self {
        Self {
            id: id.into(),
            input,
            descriptor,
            event_types: default_process_event_types(),
        }
    }

    pub fn external(
        id: impl Into<ProcessId>,
        descriptor: ProcessHandleDescriptor,
        metadata: serde_json::Value,
    ) -> Self {
        Self::new(id, ProcessInput::External { metadata }, descriptor)
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

    pub(crate) fn into_registration_and_options(
        self,
    ) -> (
        ProcessRegistration,
        ProcessStartOptions,
        ProcessHandleDescriptor,
    ) {
        let descriptor = self.descriptor;
        let registration =
            ProcessRegistration::new(self.id, self.input).with_event_types(self.event_types);
        let options = ProcessStartOptions::new().with_descriptor(descriptor.clone());
        (registration, options, descriptor)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessScope {
    pub session_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_frame_id: Option<crate::AgentFrameId>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessProvenance {
    pub owner_scope: ProcessScope,
    pub host_profile_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub caused_by: Option<crate::CausalRef>,
}

impl ProcessProvenance {
    pub fn new(owner_scope: ProcessScope, host_profile_id: impl Into<String>) -> Self {
        Self {
            owner_scope,
            host_profile_id: host_profile_id.into(),
            caused_by: None,
        }
    }

    pub fn with_caused_by(mut self, caused_by: Option<crate::CausalRef>) -> Self {
        self.caused_by = caused_by;
        self
    }
}

impl ProcessScope {
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            agent_frame_id: None,
        }
    }

    pub fn for_agent_frame(
        session_id: impl Into<String>,
        agent_frame_id: impl Into<crate::AgentFrameId>,
    ) -> Self {
        Self {
            session_id: session_id.into(),
            agent_frame_id: Some(agent_frame_id.into()),
        }
    }

    pub fn id(&self) -> ProcessScopeId {
        match self.agent_frame_id.as_deref() {
            Some(frame_id) if !frame_id.is_empty() => {
                ProcessScopeId::new(format!("session:{}/frame:{frame_id}", self.session_id))
            }
            _ => ProcessScopeId::new(format!("session:{}", self.session_id)),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.session_id.is_empty()
    }
}

/// Serializable process spec used to start or recover a runtime process.
#[derive(Debug, Serialize, Deserialize)]
pub struct ProcessRegistration {
    pub id: ProcessId,
    pub input: Arc<ProcessInput>,
    #[serde(default)]
    pub event_types: Vec<ProcessEventType>,
    pub provenance: ProcessProvenance,
}

impl Clone for ProcessRegistration {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            input: Arc::clone(&self.input),
            event_types: self.event_types.clone(),
            provenance: self.provenance.clone(),
        }
    }
}

impl ProcessRegistration {
    pub fn new(id: impl Into<ProcessId>, input: ProcessInput) -> Self {
        Self {
            id: id.into(),
            input: Arc::new(input),
            event_types: default_process_event_types(),
            provenance: ProcessProvenance::new(ProcessScope::new("root"), "default"),
        }
    }

    pub fn with_process_provenance(mut self, provenance: ProcessProvenance) -> Self {
        self.provenance = provenance;
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
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ProcessStatus {
    #[default]
    Running,
    Completed {
        await_output: ProcessAwaitOutput,
    },
    Failed {
        await_output: ProcessAwaitOutput,
    },
    Cancelled {
        await_output: ProcessAwaitOutput,
    },
}

impl ProcessStatus {
    pub fn from_terminal(terminal: ProcessTerminalSemantics) -> Self {
        match terminal.state {
            ProcessTerminalState::Completed => Self::Completed {
                await_output: terminal.await_output,
            },
            ProcessTerminalState::Failed => Self::Failed {
                await_output: terminal.await_output,
            },
            ProcessTerminalState::Cancelled => Self::Cancelled {
                await_output: terminal.await_output,
            },
        }
    }

    pub fn is_terminal(&self) -> bool {
        !matches!(self, Self::Running)
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Completed { .. } => "completed",
            Self::Failed { .. } => "failed",
            Self::Cancelled { .. } => "cancelled",
        }
    }

    pub fn terminal_state(&self) -> Option<ProcessTerminalState> {
        match self {
            Self::Running => None,
            Self::Completed { .. } => Some(ProcessTerminalState::Completed),
            Self::Failed { .. } => Some(ProcessTerminalState::Failed),
            Self::Cancelled { .. } => Some(ProcessTerminalState::Cancelled),
        }
    }

    pub fn await_output(&self) -> Option<&ProcessAwaitOutput> {
        match self {
            Self::Running => None,
            Self::Completed { await_output }
            | Self::Failed { await_output }
            | Self::Cancelled { await_output } => Some(await_output),
        }
    }

    pub fn terminal_semantics(&self) -> Option<ProcessTerminalSemantics> {
        Some(ProcessTerminalSemantics {
            state: self.terminal_state()?,
            await_output: self.await_output()?.clone(),
        })
    }
}

/// Durable process row. Session-visible addressability lives in
/// [`ProcessHandleGrant`], not in the process record.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcessRecord {
    pub id: ProcessId,
    pub registration_hash: String,
    pub input: Arc<ProcessInput>,
    #[serde(default)]
    pub event_types: Vec<ProcessEventType>,
    pub provenance: ProcessProvenance,
    #[serde(default)]
    pub created_at_ms: u64,
    #[serde(default)]
    pub updated_at_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_ref: Option<ProcessExternalRef>,
    #[serde(default)]
    pub status: ProcessStatus,
}

impl ProcessRecord {
    pub fn from_registration(mut registration: ProcessRegistration) -> Self {
        ensure_core_event_types(&mut registration);
        validate_process_registration(&registration)
            .expect("process registration should be valid before record construction");
        let registration_hash = process_registration_hash(&registration)
            .expect("process registration should hash before record construction");
        Self::from_prepared_registration(registration, registration_hash, current_epoch_ms())
    }

    pub fn from_prepared_registration(
        registration: ProcessRegistration,
        registration_hash: String,
        now_ms: u64,
    ) -> Self {
        Self {
            id: registration.id,
            registration_hash,
            input: registration.input,
            event_types: registration.event_types,
            provenance: registration.provenance,
            created_at_ms: now_ms,
            updated_at_ms: now_ms,
            external_ref: None,
            status: ProcessStatus::Running,
        }
    }

    pub fn is_terminal(&self) -> bool {
        self.status.is_terminal()
    }

    pub fn owner_scope_id(&self) -> ProcessScopeId {
        self.provenance.owner_scope.id()
    }

    pub fn host_profile_id(&self) -> &str {
        &self.provenance.host_profile_id
    }
}

/// Wire-format version stamped on every persisted [`ProcessLease`].
///
/// Bump when the on-wire shape of `ProcessLease` changes in a way that older
/// code cannot safely deserialize. Mirrors
/// [`RUNTIME_TURN_LEASE_SCHEMA_VERSION`](crate::RUNTIME_TURN_LEASE_SCHEMA_VERSION)
/// and follows the same upgrade semantics.
pub const PROCESS_LEASE_SCHEMA_VERSION: u32 = 1;

/// Durable lease over a non-terminal background process.
///
/// This is the process-domain analogue of
/// [`RuntimeTurnLease`](crate::RuntimeTurnLease): the lease pair
/// `(owner_id, lease_token)` plus `fencing_token` are how lash guarantees that
/// one non-terminal process is re-executed by exactly one worker at a time —
/// even after a crash, even across two workers that both sweep the same
/// registry for recoverable work. The durable backend
/// (`lash-sqlite-store`) uses these to serialize concurrent claims on the same
/// `process_id`; future distributed durable backends use the *same* fields to
/// coordinate workers that don't share a file system.
///
/// **This is not single-process theatre.** The owner / fencing-token /
/// lease-token triple is the public contract that lets any backend detect and
/// reject stale writers. Treat it as load-bearing, not defensive.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcessLease {
    pub schema_version: u32,
    pub process_id: ProcessId,
    pub owner_id: String,
    pub lease_token: String,
    pub fencing_token: u64,
    pub claimed_at_epoch_ms: u64,
    pub expires_at_epoch_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcessLeaseCompletion {
    pub process_id: ProcessId,
    pub lease_token: String,
}

impl ProcessLeaseCompletion {
    pub fn from_lease(lease: &ProcessLease) -> Self {
        Self {
            process_id: lease.process_id.clone(),
            lease_token: lease.lease_token.clone(),
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessLifecycleStatus {
    #[default]
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl ProcessLifecycleStatus {
    pub fn label(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    pub fn is_terminal(self) -> bool {
        !matches!(self, Self::Running)
    }

    pub fn terminal_state(self) -> Option<ProcessTerminalState> {
        match self {
            Self::Running => None,
            Self::Completed => Some(ProcessTerminalState::Completed),
            Self::Failed => Some(ProcessTerminalState::Failed),
            Self::Cancelled => Some(ProcessTerminalState::Cancelled),
        }
    }
}

impl From<&ProcessStatus> for ProcessLifecycleStatus {
    fn from(status: &ProcessStatus) -> Self {
        match status {
            ProcessStatus::Running => Self::Running,
            ProcessStatus::Completed { .. } => Self::Completed,
            ProcessStatus::Failed { .. } => Self::Failed,
            ProcessStatus::Cancelled { .. } => Self::Cancelled,
        }
    }
}

impl From<ProcessStatus> for ProcessLifecycleStatus {
    fn from(status: ProcessStatus) -> Self {
        Self::from(&status)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessHandleSummary {
    #[serde(rename = "__handle__")]
    pub handle_type: String,
    pub id: ProcessId,
    pub process_id: ProcessId,
    pub descriptor: ProcessHandleDescriptor,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub definition: Option<ProcessDefinitionSummary>,
    pub status: ProcessLifecycleStatus,
}

impl ProcessHandleSummary {
    pub fn new(
        process_id: impl Into<ProcessId>,
        descriptor: ProcessHandleDescriptor,
        status: ProcessLifecycleStatus,
    ) -> Self {
        let process_id = process_id.into();
        Self {
            handle_type: "process".to_string(),
            id: process_id.clone(),
            process_id,
            descriptor,
            definition: None,
            status,
        }
    }

    pub fn with_definition(mut self, definition: Option<ProcessDefinitionSummary>) -> Self {
        self.definition = definition;
        self
    }

    pub fn from_grant_record(grant: ProcessHandleGrant, record: ProcessRecord) -> Self {
        let definition = ProcessDefinitionSummary::from_input(record.input.as_ref());
        Self::new(
            record.id,
            grant.descriptor,
            ProcessLifecycleStatus::from(record.status),
        )
        .with_definition(definition)
    }
}

impl From<ProcessHandleGrantEntry> for ProcessHandleSummary {
    fn from((grant, record): ProcessHandleGrantEntry) -> Self {
        Self::from_grant_record(grant, record)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessCancelSummary {
    pub process_id: ProcessId,
    pub status: ProcessLifecycleStatus,
}

impl ProcessCancelSummary {
    pub fn from_record(record: ProcessRecord) -> Self {
        Self {
            process_id: record.id,
            status: ProcessLifecycleStatus::from(record.status),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessDefinitionSummary {
    pub name: String,
}

impl ProcessDefinitionSummary {
    pub fn from_input(input: &ProcessInput) -> Option<Self> {
        match input {
            ProcessInput::LashlangProcess { process_name, .. } => Some(Self {
                name: process_name.clone(),
            }),
            ProcessInput::ToolCall { .. }
            | ProcessInput::SessionTurn { .. }
            | ProcessInput::External { .. } => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProcessDefinitionSelector {
    module_ref: lashlang::ModuleRef,
    required_surface_ref: lashlang::RequiredSurfaceRef,
    process_ref: lashlang::ProcessRef,
    process_name: String,
}

impl ProcessDefinitionSelector {
    pub fn decode(value: &serde_json::Value) -> Result<Self, String> {
        if value
            .get(lashlang::LASH_PROCESS_VALUE_KEY)
            .and_then(serde_json::Value::as_bool)
            != Some(true)
        {
            return Err("definition must be a process definition value".to_string());
        }
        Ok(Self {
            module_ref: decode_process_definition_field(
                value,
                lashlang::LASH_MODULE_REF_KEY,
                "definition",
            )?,
            required_surface_ref: decode_process_definition_field(
                value,
                lashlang::LASH_REQUIRED_SURFACE_REF_KEY,
                "definition",
            )?,
            process_ref: decode_process_definition_field(
                value,
                lashlang::LASH_PROCESS_REF_KEY,
                "definition",
            )?,
            process_name: value
                .get(lashlang::LASH_PROCESS_NAME_KEY)
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| "definition is missing its process name".to_string())?
                .to_string(),
        })
    }

    pub fn matches_input(&self, input: &ProcessInput) -> bool {
        match input {
            ProcessInput::LashlangProcess {
                module_ref,
                process_ref,
                required_surface_ref,
                process_name,
                ..
            } => {
                self.module_ref == *module_ref
                    && self.required_surface_ref == *required_surface_ref
                    && self.process_ref == *process_ref
                    && self.process_name == *process_name
            }
            ProcessInput::ToolCall { .. }
            | ProcessInput::SessionTurn { .. }
            | ProcessInput::External { .. } => false,
        }
    }
}

fn decode_process_definition_field<T: serde::de::DeserializeOwned>(
    value: &serde_json::Value,
    field: &'static str,
    label: &'static str,
) -> Result<T, String> {
    serde_json::from_value(
        value
            .get(field)
            .cloned()
            .ok_or_else(|| format!("{label} is missing {field}"))?,
    )
    .map_err(|err| format!("{label} has invalid {field}: {err}"))
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ProcessStatusFilter {
    #[default]
    Running,
    Completed,
    Failed,
    Cancelled,
    Any,
}

impl ProcessStatusFilter {
    pub fn decode(value: Option<&str>) -> Result<Self, String> {
        match value.unwrap_or("running") {
            "running" => Ok(Self::Running),
            "completed" => Ok(Self::Completed),
            "failed" => Ok(Self::Failed),
            "cancelled" => Ok(Self::Cancelled),
            "any" => Ok(Self::Any),
            other => Err(format!(
                "processes.list status must be `running`, `completed`, `failed`, `cancelled`, or `any`, got `{other}`"
            )),
        }
    }

    pub fn list_mode(self) -> ProcessListMode {
        match self {
            Self::Running => ProcessListMode::Live,
            Self::Completed | Self::Failed | Self::Cancelled | Self::Any => ProcessListMode::All,
        }
    }

    pub fn matches(self, status: ProcessLifecycleStatus) -> bool {
        match self {
            Self::Running => status == ProcessLifecycleStatus::Running,
            Self::Completed => status == ProcessLifecycleStatus::Completed,
            Self::Failed => status == ProcessLifecycleStatus::Failed,
            Self::Cancelled => status == ProcessLifecycleStatus::Cancelled,
            Self::Any => true,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProcessListFilter {
    pub definition: Option<ProcessDefinitionSelector>,
    pub status: ProcessStatusFilter,
}

impl ProcessListFilter {
    pub fn decode(args: &serde_json::Value) -> Result<Self, String> {
        let map = args
            .as_object()
            .ok_or_else(|| "processes.list expects a record of process filters".to_string())?;
        for key in map.keys() {
            match key.as_str() {
                "definition" | "status" => {}
                _ => return Err(format!("processes.list unknown filter `{key}`")),
            }
        }
        let definition = args
            .get("definition")
            .map(ProcessDefinitionSelector::decode)
            .transpose()?;
        let status =
            ProcessStatusFilter::decode(args.get("status").and_then(serde_json::Value::as_str))?;
        Ok(Self { definition, status })
    }

    pub fn list_mode(&self) -> ProcessListMode {
        self.status.list_mode()
    }

    pub fn matches_entry(&self, entry: &ProcessHandleGrantEntry) -> bool {
        let (_grant, record) = entry;
        let status = ProcessLifecycleStatus::from(&record.status);
        self.status.matches(status)
            && self
                .definition
                .as_ref()
                .is_none_or(|definition| definition.matches_input(record.input.as_ref()))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessListMode {
    #[default]
    Live,
    All,
}

impl ProcessListMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Live => "live",
            Self::All => "all",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessStartGrant {
    pub owner_scope: ProcessScope,
    pub descriptor: ProcessHandleDescriptor,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessSessionDeleteReport {
    pub session_id: String,
    pub revoked_handle_count: usize,
    pub deleted_wake_count: usize,
    pub cancel_process_ids: Vec<String>,
    pub preserved_process_ids: Vec<String>,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn process_ref(component: &str, pos: usize) -> lashlang::ProcessRef {
        lashlang::ProcessRef {
            component: lashlang::ContentHash::new(component),
            pos: pos as u32,
        }
    }

    fn process_value(
        module_ref: &lashlang::ModuleRef,
        surface_ref: &lashlang::RequiredSurfaceRef,
        process_ref: &lashlang::ProcessRef,
        name: &str,
    ) -> serde_json::Value {
        let mut value = serde_json::Map::new();
        value.insert(lashlang::LASH_PROCESS_VALUE_KEY.to_string(), json!(true));
        value.insert(lashlang::LASH_MODULE_REF_KEY.to_string(), json!(module_ref));
        value.insert(
            lashlang::LASH_REQUIRED_SURFACE_REF_KEY.to_string(),
            json!(surface_ref),
        );
        value.insert(
            lashlang::LASH_PROCESS_REF_KEY.to_string(),
            json!(process_ref),
        );
        value.insert(lashlang::LASH_PROCESS_NAME_KEY.to_string(), json!(name));
        serde_json::Value::Object(value)
    }

    fn lashlang_entry(
        process_id: &str,
        module_ref: lashlang::ModuleRef,
        surface_ref: lashlang::RequiredSurfaceRef,
        process_ref: lashlang::ProcessRef,
        process_name: &str,
        status: ProcessStatus,
    ) -> ProcessHandleGrantEntry {
        let mut record = ProcessRecord::from_registration(ProcessRegistration::new(
            process_id,
            ProcessInput::LashlangProcess {
                module_ref,
                process_ref,
                required_surface_ref: surface_ref,
                process_name: process_name.to_string(),
                args: serde_json::Map::new(),
            },
        ));
        record.status = status;
        (
            ProcessHandleGrant {
                session_id: "session".to_string(),
                process_id: process_id.to_string(),
                descriptor: ProcessHandleDescriptor::new(Some("lashlang"), Some(process_name)),
            },
            record,
        )
    }

    #[test]
    fn process_list_filter_matches_definition_and_status() {
        let module_ref = lashlang::ModuleRef::new(&lashlang::ContentHash::new("module"));
        let surface_ref = lashlang::RequiredSurfaceRef::new(&lashlang::ContentHash::new("surface"));
        let target_ref = process_ref("target", 0);
        let other_ref = process_ref("other", 1);
        let filter = ProcessListFilter::decode(&json!({
            "definition": process_value(&module_ref, &surface_ref, &target_ref, "target"),
            "status": "completed"
        }))
        .expect("decode filter");

        let matching = lashlang_entry(
            "matching",
            module_ref.clone(),
            surface_ref.clone(),
            target_ref,
            "target",
            ProcessStatus::Completed {
                await_output: ProcessAwaitOutput::from_tool_output(crate::ToolCallOutput::success(
                    json!(true),
                )),
            },
        );
        let wrong_definition = lashlang_entry(
            "wrong-definition",
            module_ref,
            surface_ref,
            other_ref,
            "other",
            ProcessStatus::Completed {
                await_output: ProcessAwaitOutput::from_tool_output(crate::ToolCallOutput::success(
                    json!(true),
                )),
            },
        );

        assert_eq!(filter.list_mode(), ProcessListMode::All);
        assert!(filter.matches_entry(&matching));
        assert!(!filter.matches_entry(&wrong_definition));
    }
}
