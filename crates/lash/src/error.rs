use crate::support::*;

#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    #[error(
        "protocol plugin is required; call .protocol_plugin(...) or use StandardCore::builder()/RlmCore::builder()"
    )]
    MissingProtocolPlugin,
    #[error("model spec is required; hosts must supply explicit model metadata")]
    MissingModelSpec,
    #[error("effect host is required; provide an explicit effect host with .effect_host(...)")]
    MissingEffectHost,
    #[error(
        "lashlang artifact store is required; provide an explicit Lashlang artifact store with .lashlang_artifact_store(...)"
    )]
    MissingLashlangArtifactStore,
    #[error(
        "attachment store is required; provide an explicit attachment store with .attachment_store(...)"
    )]
    MissingAttachmentStore,
    #[error(
        "process execution environment store is required; provide an explicit process env store with .process_env_store(...)"
    )]
    MissingProcessEnvStore,
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
        "a process registry is configured for the default inline process work runner but no session store factory is wired; the runner rebuilds a session runtime per process and cannot do so without one. Wire .store_factory(...) - InMemorySessionStoreFactory::new() for ephemeral process execution, or a durable factory - or use .process_work_driver(...) for an externally driven durable runner."
    )]
    ProcessRegistryRequiresStoreFactory,
    #[error("durable process worker config requires a LashCore process registry")]
    MissingProcessRegistry,
    #[error("session deletion requires a LashCore store factory")]
    MissingSessionStoreFactory,
    #[error("failed to delete process state for session `{session_id}`: {message}")]
    SessionDeleteProcess { session_id: String, message: String },
    #[error("missing required turn input for plugin `{plugin_id}`")]
    MissingPluginTurnInput { plugin_id: &'static str },
    #[error(
        "configured effect host for {operation} is durable and requires a handler context; use .effects(&controller) and provide .turn_id(...) for replayable foreground requests"
    )]
    DurableEffectHostRequiresHandlerContext { operation: &'static str },
    #[error(
        "pull-style turn streams require an effect host that can create a static scoped controller; use stream_to(...) inside the handler context"
    )]
    StaticTurnStreamRequiresStaticEffectHost,
    #[error("runtime session error: {0}")]
    Session(#[from] SessionError),
    #[error("runtime turn error: {0}")]
    Runtime(#[from] lash_core::RuntimeError),
    #[error("runtime plugin/control error: {0}")]
    Plugin(#[from] lash_core::PluginError),
    #[error("failed to encode protocol turn options: {0}")]
    ProtocolTurnOptions(#[from] serde_json::Error),
    #[error("runtime control unavailable: {0}")]
    Control(#[from] lash_core::PluginOperationInvokeError),
}

pub type Result<T> = std::result::Result<T, EmbedError>;
