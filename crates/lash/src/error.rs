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
    #[error("missing required turn input for plugin `{plugin_id}`")]
    MissingPluginTurnInput { plugin_id: &'static str },
    #[error("runtime session error: {0}")]
    Session(#[from] SessionError),
    #[error("runtime turn error: {0}")]
    Runtime(#[from] lash_core::RuntimeError),
    #[error("runtime plugin/control error: {0}")]
    Plugin(#[from] lash_core::PluginError),
    #[error("failed to encode mode turn options: {0}")]
    ModeTurnOptions(#[from] serde_json::Error),
    #[error("runtime control unavailable: {0}")]
    Control(#[from] lash_core::PluginActionInvokeError),
    #[error("queued image `{id}` is missing its image blob")]
    MissingQueuedImageBlob { id: String },
}

pub type Result<T> = std::result::Result<T, EmbedError>;
