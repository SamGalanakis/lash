#[derive(Debug, thiserror::Error, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "message", rename_all = "snake_case")]
pub enum PluginError {
    #[error("plugin registration error: {0}")]
    Registration(String),
    #[error("plugin snapshot error: {0}")]
    Snapshot(String),
    #[error("plugin invoke error: {0}")]
    Invoke(String),
    #[error("plugin session error: {0}")]
    Session(String),
}
