use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::events::{ProcessEventType, ProcessTerminalSemantics, default_process_event_types};
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
}

#[derive(Clone, Debug, Default)]
pub struct ProcessStartOptions {
    pub descriptor: Option<ProcessHandleDescriptor>,
    pub wake_session_id: Option<String>,
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

    pub fn with_wake_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.wake_session_id = Some(session_id.into());
        self
    }

    pub fn execution_context(&self, scope: &ProcessOpScope<'_>) -> ProcessExecutionContext {
        ProcessExecutionContext {
            causal_invocation: scope.parent_invocation.clone(),
            wake_target_scope: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessScope {
    pub session_id: String,
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
        }
    }

    pub fn id(&self) -> ProcessScopeId {
        ProcessScopeId::new(format!("session:{}", self.session_id))
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal: Option<ProcessTerminalSemantics>,
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
            terminal: None,
        }
    }

    pub fn is_terminal(&self) -> bool {
        self.terminal.is_some()
    }

    pub fn owner_scope_id(&self) -> ProcessScopeId {
        self.provenance.owner_scope.id()
    }

    pub fn host_profile_id(&self) -> &str {
        &self.provenance.host_profile_id
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
