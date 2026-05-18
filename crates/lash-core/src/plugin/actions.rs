use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use super::*;

pub type PluginActionInvokeFuture = Pin<Box<dyn Future<Output = ToolResult> + Send>>;
pub type PluginActionHandler =
    Arc<dyn Fn(PluginActionContext, serde_json::Value) -> PluginActionInvokeFuture + Send + Sync>;
pub type PluginActionFuture<T> =
    Pin<Box<dyn Future<Output = Result<T, PluginActionFailure>> + Send>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionParam {
    Required,
    Optional,
    Forbidden,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PluginActionKind {
    Query,
    Command,
    Task,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PluginActionDef {
    pub name: String,
    pub description: String,
    pub kind: PluginActionKind,
    pub session_param: SessionParam,
    #[serde(default)]
    pub input_schema: serde_json::Value,
    #[serde(default)]
    pub output_schema: serde_json::Value,
}

pub trait PluginAction: Send + Sync + 'static {
    const NAME: &'static str;
    const DESCRIPTION: &'static str;
    const KIND: PluginActionKind;
    const SESSION_PARAM: SessionParam;
    type Args: Serialize + DeserializeOwned + JsonSchema + Send + 'static;
    type Output: Serialize + DeserializeOwned + JsonSchema + Send + 'static;
}

#[derive(Clone, Debug, thiserror::Error)]
#[error("{message}")]
pub struct PluginActionFailure {
    message: String,
}

impl PluginActionFailure {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl From<String> for PluginActionFailure {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<&str> for PluginActionFailure {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<PluginError> for PluginActionFailure {
    fn from(value: PluginError) -> Self {
        Self::new(value.to_string())
    }
}

pub fn plugin_action_def<Op: PluginAction>() -> PluginActionDef {
    PluginActionDef {
        name: Op::NAME.to_string(),
        description: Op::DESCRIPTION.to_string(),
        kind: Op::KIND,
        session_param: Op::SESSION_PARAM,
        input_schema: serde_json::to_value(schemars::schema_for!(Op::Args))
            .unwrap_or_else(|_| serde_json::json!({})),
        output_schema: serde_json::to_value(schemars::schema_for!(Op::Output))
            .unwrap_or_else(|_| serde_json::json!({})),
    }
}

#[derive(Clone)]
pub struct PluginActionContext {
    pub session_id: Option<String>,
    pub host: Arc<dyn RuntimeSessionHost>,
}

#[derive(Clone)]
pub(crate) struct RegisteredPluginAction {
    pub(crate) def: PluginActionDef,
    pub(crate) handler: PluginActionHandler,
}
