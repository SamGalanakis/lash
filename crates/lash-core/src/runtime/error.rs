/// Durable substrate that a durable execution path requires but the host wired
/// as ephemeral.
///
/// Names the failing facet so a [`RuntimeErrorCode::DurableSubstrateRequired`]
/// can be matched and serialized losslessly per facet, while still subsuming the
/// older attachment-only signal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DurableSubstrateFacet {
    AttachmentStore,
    ArtifactStore,
    SessionStore,
}

impl DurableSubstrateFacet {
    /// Stable per-facet error-code string (the full
    /// `durable_substrate_required:*` code surfaced in traces and host errors).
    fn as_code(self) -> &'static str {
        match self {
            Self::AttachmentStore => "durable_substrate_required:attachment_store",
            Self::ArtifactStore => "durable_substrate_required:artifact_store",
            Self::SessionStore => "durable_substrate_required:session_store",
        }
    }
}

/// Stable runtime error code.
///
/// Codes serialize as the same snake_case strings exposed in traces and host
/// errors, but callers should match this type instead of parsing display text.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RuntimeErrorCode {
    MissingDurableTurnScopeTurnId,
    DurableTurnScopeTurnIdMismatch,
    /// A process (re-)execution was handed an empty/non-persisted process id.
    /// Process execution identity is the persisted `process_id`; a retry that
    /// cannot present that stable id has lost its idempotency anchor.
    MissingProcessExecutionId,
    /// A durable execution path was wired against an ephemeral store for the
    /// named facet. Subsumes the former `DurableAttachmentStoreRequired`.
    DurableSubstrateRequired {
        facet: DurableSubstrateFacet,
    },
    RuntimeTurnResumeStoreRequired,
    RuntimeTurnCheckpointLoad,
    RuntimeTurnCheckpointMissing,
    RuntimeTurnResumeProviderMismatch,
    RuntimeTurnLease,
    RuntimeTurnCheckpointHash,
    RuntimeTurnCheckpointHashMismatch,
    RuntimeTurnRestoreTurnDriverPreamble,
    StoreCommitFailed,
    PluginSessionManager,
    PluginFinalizeTurn,
    PluginCheckpoint,
    PluginPrepareTurn,
    ContextPrepareTurn,
    ProtocolTurnExtension,
    ProtocolBeforeLlmCall,
    TurnStreamJoin,
    EmptyAgentFrameRun,
    DurableTurnLiveProtocolExtension,
    DurableTurnLivePluginInput,
    Other(String),
}

impl RuntimeErrorCode {
    pub fn as_str(&self) -> &str {
        match self {
            Self::MissingDurableTurnScopeTurnId => "missing_durable_turn_scope_turn_id",
            Self::DurableTurnScopeTurnIdMismatch => "durable_turn_scope_turn_id_mismatch",
            Self::MissingProcessExecutionId => "missing_process_execution_id",
            Self::DurableSubstrateRequired { facet } => facet.as_code(),
            Self::RuntimeTurnResumeStoreRequired => "runtime_turn_resume_store_required",
            Self::RuntimeTurnCheckpointLoad => "runtime_turn_checkpoint_load",
            Self::RuntimeTurnCheckpointMissing => "runtime_turn_checkpoint_missing",
            Self::RuntimeTurnResumeProviderMismatch => "runtime_turn_resume_provider_mismatch",
            Self::RuntimeTurnLease => "runtime_turn_lease",
            Self::RuntimeTurnCheckpointHash => "runtime_turn_checkpoint_hash",
            Self::RuntimeTurnCheckpointHashMismatch => "runtime_turn_checkpoint_hash_mismatch",
            Self::RuntimeTurnRestoreTurnDriverPreamble => {
                "runtime_turn_restore_turn_driver_preamble"
            }
            Self::StoreCommitFailed => "store_commit_failed",
            Self::PluginSessionManager => "plugin_session_manager",
            Self::PluginFinalizeTurn => "plugin_finalize_turn",
            Self::PluginCheckpoint => "plugin_checkpoint",
            Self::PluginPrepareTurn => "plugin_prepare_turn",
            Self::ContextPrepareTurn => "context_prepare_turn",
            Self::ProtocolTurnExtension => "protocol_turn_extension",
            Self::ProtocolBeforeLlmCall => "protocol_before_llm_call",
            Self::TurnStreamJoin => "turn_stream_join",
            Self::EmptyAgentFrameRun => "empty_agent_frame_run",
            Self::DurableTurnLiveProtocolExtension => "durable_turn_live_protocol_extension",
            Self::DurableTurnLivePluginInput => "durable_turn_live_plugin_input",
            Self::Other(code) => code.as_str(),
        }
    }
}

impl std::fmt::Display for RuntimeErrorCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<&str> for RuntimeErrorCode {
    fn from(code: &str) -> Self {
        match code {
            "missing_durable_turn_scope_turn_id" => Self::MissingDurableTurnScopeTurnId,
            "durable_turn_scope_turn_id_mismatch" => Self::DurableTurnScopeTurnIdMismatch,
            "missing_process_execution_id" => Self::MissingProcessExecutionId,
            "durable_substrate_required:attachment_store" => Self::DurableSubstrateRequired {
                facet: DurableSubstrateFacet::AttachmentStore,
            },
            "durable_substrate_required:artifact_store" => Self::DurableSubstrateRequired {
                facet: DurableSubstrateFacet::ArtifactStore,
            },
            "durable_substrate_required:session_store" => Self::DurableSubstrateRequired {
                facet: DurableSubstrateFacet::SessionStore,
            },
            "runtime_turn_resume_store_required" => Self::RuntimeTurnResumeStoreRequired,
            "runtime_turn_checkpoint_load" => Self::RuntimeTurnCheckpointLoad,
            "runtime_turn_checkpoint_missing" => Self::RuntimeTurnCheckpointMissing,
            "runtime_turn_resume_provider_mismatch" => Self::RuntimeTurnResumeProviderMismatch,
            "runtime_turn_lease" => Self::RuntimeTurnLease,
            "runtime_turn_checkpoint_hash" => Self::RuntimeTurnCheckpointHash,
            "runtime_turn_checkpoint_hash_mismatch" => Self::RuntimeTurnCheckpointHashMismatch,
            "runtime_turn_restore_turn_driver_preamble" => {
                Self::RuntimeTurnRestoreTurnDriverPreamble
            }
            "store_commit_failed" => Self::StoreCommitFailed,
            "plugin_session_manager" => Self::PluginSessionManager,
            "plugin_finalize_turn" => Self::PluginFinalizeTurn,
            "plugin_checkpoint" => Self::PluginCheckpoint,
            "plugin_prepare_turn" => Self::PluginPrepareTurn,
            "context_prepare_turn" => Self::ContextPrepareTurn,
            "protocol_turn_extension" => Self::ProtocolTurnExtension,
            "protocol_before_llm_call" => Self::ProtocolBeforeLlmCall,
            "turn_stream_join" => Self::TurnStreamJoin,
            "empty_agent_frame_run" => Self::EmptyAgentFrameRun,
            "durable_turn_live_protocol_extension" => Self::DurableTurnLiveProtocolExtension,
            "durable_turn_live_plugin_input" => Self::DurableTurnLivePluginInput,
            other => Self::Other(other.to_string()),
        }
    }
}

impl From<String> for RuntimeErrorCode {
    fn from(code: String) -> Self {
        Self::from(code.as_str())
    }
}

impl serde::Serialize for RuntimeErrorCode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> serde::Deserialize<'de> for RuntimeErrorCode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let code = <String as serde::Deserialize>::deserialize(deserializer)?;
        Ok(Self::from(code))
    }
}

/// Runtime error for unexpected failures.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RuntimeError {
    pub code: RuntimeErrorCode,
    pub message: String,
}

impl RuntimeError {
    pub fn new(code: impl Into<RuntimeErrorCode>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }

    pub fn is_code(&self, code: RuntimeErrorCode) -> bool {
        self.code == code
    }

    /// Build the loud error raised when a durable execution path was wired
    /// against an ephemeral store for `facet`.
    pub fn durable_substrate_required(facet: DurableSubstrateFacet) -> Self {
        let facet_label = match facet {
            DurableSubstrateFacet::AttachmentStore => "attachment store",
            DurableSubstrateFacet::ArtifactStore => "lashlang artifact store",
            DurableSubstrateFacet::SessionStore => "session store",
        };
        Self::new(
            RuntimeErrorCode::DurableSubstrateRequired { facet },
            format!("durable effect controllers require a durable {facet_label}"),
        )
    }

    /// Build the loud error raised when a process (re-)execution is handed an
    /// empty/non-persisted id.
    ///
    /// Process execution identity is the persisted `process_id`, so a retry
    /// must present that stable id — mirroring how
    /// [`DurableTurnScope::new`](crate::DurableTurnScope)
    /// rejects an empty turn id.
    pub fn missing_process_execution_id() -> Self {
        Self::new(
            RuntimeErrorCode::MissingProcessExecutionId,
            "process execution requires a non-empty persisted process id",
        )
    }
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for RuntimeError {}

#[cfg(test)]
mod tests {
    use super::{DurableSubstrateFacet, RuntimeError, RuntimeErrorCode};

    #[test]
    fn durable_substrate_required_round_trips_per_facet() {
        for facet in [
            DurableSubstrateFacet::AttachmentStore,
            DurableSubstrateFacet::ArtifactStore,
            DurableSubstrateFacet::SessionStore,
        ] {
            let err = RuntimeError::durable_substrate_required(facet);
            let json = serde_json::to_value(&err).expect("serialize runtime error");
            let decoded: RuntimeError = serde_json::from_value(json).expect("decode runtime error");
            assert_eq!(
                decoded.code,
                RuntimeErrorCode::DurableSubstrateRequired { facet }
            );
        }
    }

    #[test]
    fn missing_process_execution_id_round_trips() {
        let err = RuntimeError::missing_process_execution_id();
        assert_eq!(err.code, RuntimeErrorCode::MissingProcessExecutionId);
        let json = serde_json::to_value(&err).expect("serialize runtime error");
        assert_eq!(json["code"], "missing_process_execution_id");
        let decoded: RuntimeError = serde_json::from_value(json).expect("decode runtime error");
        assert_eq!(decoded.code, RuntimeErrorCode::MissingProcessExecutionId);
    }

    #[test]
    fn runtime_error_code_serializes_as_stable_string() {
        let err = RuntimeError::new(
            RuntimeErrorCode::RuntimeTurnCheckpointMissing,
            "missing checkpoint",
        );

        let json = serde_json::to_value(&err).expect("serialize runtime error");
        assert_eq!(json["code"], "runtime_turn_checkpoint_missing");

        let decoded: RuntimeError = serde_json::from_value(json).expect("decode runtime error");
        assert_eq!(decoded.code, RuntimeErrorCode::RuntimeTurnCheckpointMissing);
    }

    #[test]
    fn unknown_runtime_error_code_round_trips() {
        let decoded: RuntimeError = serde_json::from_value(serde_json::json!({
            "code": "plugin_defined_abort",
            "message": "stopped by plugin"
        }))
        .expect("decode plugin runtime error");

        assert_eq!(
            decoded.code,
            RuntimeErrorCode::Other("plugin_defined_abort".to_string())
        );
        assert_eq!(decoded.code.as_str(), "plugin_defined_abort");
    }
}
