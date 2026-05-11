use crate::support::*;

#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    #[error("no mode presets installed; call install_mode(ModePreset::...) first")]
    NoModesInstalled,
    #[error("default mode `{mode}` is not installed on this LashCore")]
    DefaultModeNotInstalled { mode: ModeId },
    #[error("mode `{mode}` is not installed on this LashCore")]
    ModeNotInstalled { mode: ModeId },
    #[error("max_context_tokens is required; hosts must supply explicit model metadata")]
    MissingMaxContextTokens,
    #[error("failed to create store for session `{session_id}`: {message}")]
    StoreFactory { session_id: String, message: String },
    #[error("store is bound to session `{loaded}` but builder requested `{requested}`")]
    StoreSessionMismatch { loaded: String, requested: String },
    #[error("plugin binding `{plugin_id}` is not registered on this LashCore")]
    PluginNotRegistered { plugin_id: &'static str },
    #[error("missing required turn context for plugin `{plugin_id}`")]
    MissingPluginTurnContext { plugin_id: &'static str },
    #[error("plugin binding `{plugin_id}` config error: {message}")]
    PluginConfig {
        plugin_id: &'static str,
        message: String,
    },
    #[error("runtime session error: {0}")]
    Session(#[from] SessionError),
    #[error("runtime turn error: {0}")]
    Runtime(#[from] lash::RuntimeError),
    #[error("runtime plugin/control error: {0}")]
    Plugin(#[from] lash::PluginError),
    #[error("failed to encode mode turn options: {0}")]
    ModeTurnOptions(#[from] serde_json::Error),
    #[error("runtime control unavailable: {0}")]
    Control(#[from] lash::PluginActionInvokeError),
    #[error("queued image `{id}` is missing its image blob")]
    MissingQueuedImageBlob { id: String },
}

pub type Result<T> = std::result::Result<T, EmbedError>;
