use crate::support::*;

#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    #[error("no mode presets installed; call install_mode(ModePreset::...) first")]
    NoModesInstalled,
    #[error("default mode `{mode}` is not installed on this LashCore")]
    DefaultModeNotInstalled { mode: ModeId },
    #[error("mode `{mode}` is not installed on this LashCore")]
    ModeNotInstalled { mode: ModeId },
    #[error("model spec is required; hosts must supply explicit model metadata")]
    MissingModelSpec,
    #[error(
        "effect controller is required; call .effect_controller(...) (e.g. InlineRuntimeEffectController) or .in_memory_stores()"
    )]
    MissingEffectController,
    #[error(
        "lashlang artifact store is required; call .lashlang_artifact_store(...) or .in_memory_stores()"
    )]
    MissingLashlangArtifactStore,
    #[error("attachment store is required; call .attachment_store(...) or .in_memory_stores()")]
    MissingAttachmentStore,
    #[error("failed to create store for session `{session_id}`: {message}")]
    StoreFactory { session_id: String, message: String },
    #[error("store is bound to session `{loaded}` but builder requested `{requested}`")]
    StoreSessionMismatch { loaded: String, requested: String },
    #[error("durable process worker requires a LashCore store factory")]
    MissingProcessWorkerStoreFactory,
    #[error(
        "durable session store requires a durable {facet}; an ephemeral {facet} cannot back a durable session store"
    )]
    DurableStorePeerRequired { facet: &'static str },
    #[error(
        "durable process registry requires a durable session store factory; call .store_factory(...) with a durable store"
    )]
    DurableProcessRegistryRequiresStoreFactory,
    #[error(
        "a process registry is configured but no process work runner is available; the default inline runner is disabled and no runner was supplied via .with_process_work_runner(...). Non-terminal processes would never execute. Enable the default runner or register an explicit one."
    )]
    ProcessRegistryWithoutWorkRunner,
    #[error(
        "a process registry is configured but no session store factory is wired; the default process work runner rebuilds a session runtime per process and cannot do so without one, so processes would never execute. Wire .store_factory(...) — InMemorySessionStoreFactory::new() for ephemeral process execution, or a durable factory — or .disable_default_process_work_runner() if processes are inspected but never run."
    )]
    ProcessRegistryRequiresStoreFactory,
    #[error("session deletion requires a LashCore store factory")]
    MissingSessionStoreFactory,
    #[error("failed to delete process state for session `{session_id}`: {message}")]
    SessionDeleteProcess { session_id: String, message: String },
    #[error("missing required turn input for plugin `{plugin_id}`")]
    MissingPluginTurnInput { plugin_id: &'static str },
    #[error("runtime session error: {0}")]
    Session(#[from] SessionError),
    #[error("runtime turn error: {0}")]
    Runtime(#[from] lash_core::RuntimeError),
    #[error("runtime plugin/control error: {0}")]
    Plugin(#[from] lash_core::PluginError),
    #[error("failed to encode protocol turn options: {0}")]
    ProtocolTurnOptions(#[from] serde_json::Error),
    #[error("runtime control unavailable: {0}")]
    Control(#[from] lash_core::PluginActionInvokeError),
    #[error("queued image `{id}` is missing its image blob")]
    MissingQueuedImageBlob { id: String },
}

pub type Result<T> = std::result::Result<T, EmbedError>;
