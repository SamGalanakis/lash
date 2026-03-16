pub mod agent;
pub mod dynamic;
pub mod embedded;
pub mod instructions;
pub mod llm;
pub mod model_info;
pub mod model_variant;
pub mod oauth;
pub mod plugin;
pub mod provider;
pub mod runtime;
pub mod sansio;
#[cfg(feature = "sqlite-store")]
mod search;
pub mod session;
pub mod skill_catalog;
#[cfg(feature = "sqlite-store")]
pub mod store;
pub mod text;
mod tool_dispatch;
pub mod tools;

use std::path::PathBuf;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Return the root data directory for lash.
///
/// Checks `LASH_HOME` env var first, falling back to `~/.lash/`.
pub fn lash_home() -> PathBuf {
    if let Ok(dir) = std::env::var("LASH_HOME") {
        PathBuf::from(dir)
    } else {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".lash")
    }
}

/// Return the cache directory for lash.
///
/// When `LASH_HOME` is set: `$LASH_HOME/cache`.
/// Otherwise: `~/.cache/lash/` (via `dirs::cache_dir`).
pub fn lash_cache_dir() -> PathBuf {
    if std::env::var("LASH_HOME").is_ok() {
        lash_home().join("cache")
    } else {
        dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from(".cache"))
            .join("lash")
    }
}

/// Return the preferred repo-local directory for lash artifacts.
pub fn repo_local_lash_dir() -> PathBuf {
    PathBuf::from(".agents").join("lash")
}

/// Return the legacy repo-local directory for lash artifacts.
pub fn legacy_repo_local_lash_dir() -> PathBuf {
    PathBuf::from(".lash")
}

/// Return skill search directories in override order from lowest to highest priority.
pub fn default_skill_dirs() -> Vec<PathBuf> {
    vec![
        lash_home().join("skills"),
        legacy_repo_local_lash_dir().join("skills"),
        repo_local_lash_dir().join("skills"),
    ]
}

// Re-exports
pub use agent::message::MessageOrigin;
pub use agent::{
    AgentConfig, AgentEvent, DefaultPromptRenderer, Message, MessageRole, Part, PartKind,
    PromptOverrideMode, PromptRenderer, PromptSectionName, PromptSectionOverride, PruneState,
    TokenUsage, default_prompt_renderer,
};
pub use dynamic::{
    DynamicStateSnapshot, DynamicToolProvider, DynamicToolSpec, InProcessToolExecutionAdapter,
    InProcessToolFuture, InProcessToolHandler, ReconfigureError, ToolExecutionAdapter,
};
pub use instructions::InstructionLoaderConfig;
pub use instructions::{FsInstructionSource, InstructionLoader, InstructionSource};
pub use model_info::{
    CachedModelCatalog, FileModelCatalogStore, MemoryModelCatalogStore, ModelCatalog,
    ModelCatalogSource, ModelCatalogStore, ModelInfo, ModelsDevHttpSource, ResolvedModelSpec,
};
pub use model_variant::VariantRequestConfig;
pub use plugin::{
    AssistantResponseHookContext, AssistantResponseTransform, AssistantStreamHookContext,
    AssistantStreamTransform, BuiltinToolResultProjectionPluginFactory, CheckpointHookContext,
    CheckpointKind, ExternalInvokeContext, ExternalInvokeError, ExternalOpDef, ExternalOpKind,
    MessageMutatorContext, MessageMutatorHook, PluginDirective, PluginError, PluginFactory,
    PluginHost, PluginMessage, PluginOwned, PluginRegistrar, PluginSession, PluginSessionContext,
    PluginSessionSnapshot, PluginSnapshotArtifact, PluginSnapshotEntry, PluginSnapshotMeta,
    PluginSpec, PluginSpecFactory, PluginSurfaceEvent, PromptContribution, PromptHookContext,
    RuntimeServices, SessionConfigOverrides, SessionConfigSnapshot, SessionCreateRequest,
    SessionHandle, SessionManager, SessionParam, SessionPlugin, SessionSnapshot, SessionStartPoint,
    SessionTurnHandle, SnapshotReader, SnapshotWriter, ToolResultProjectionContext,
    ToolResultProjectionHook, ToolResultProjectionMode, ToolResultProjectionPluginConfig,
    ToolResultProjector, ToolSurfaceContribution, TurnHookContext, TurnResultHookContext,
};
#[cfg(feature = "sqlite-store")]
pub use plugin::{
    BuiltinHistoryPluginFactory, BuiltinPlanModePluginFactory, BuiltinPlanTrackerPluginFactory,
    BuiltinPromptContextPluginFactory, PromptContextPluginConfig,
};
pub use provider::{LashConfig, Provider, ProviderOptions, RequestTimeout};
pub use runtime::{
    AgentStateEnvelope, AssembledTurn, AssistantOutput, CodeOutputRecord, DoneReason, EventSink,
    ExecutionSummary, HostProfile, InputItem, LashRuntime, NoopEventSink, OutputState,
    PathResolver, PromptUsage, RunMode, RuntimeConfig, RuntimeError, SanitizerPolicy,
    TerminationPolicy, TurnInput, TurnIssue, TurnStatus,
};
pub use sansio::{Effect, EffectId, LlmCallError, Response, TurnMachine, TurnMachineConfig};
pub use session::{
    ExecResponse, PromptBridge, Session, SessionError, TurnInjectionBridge, UserPrompt,
};
pub use skill_catalog::{LoadedSkill, SkillCatalog};
#[cfg(feature = "sqlite-store")]
pub use store::{AgentState, Store};
pub use text::strip_repl_fragments;
pub use tools::{AgentCallPluginFactory, DefaultToolPluginDeps, default_tool_plugin_factories};

/// Execution backend for agent turns.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMode {
    #[default]
    Repl,
    Standard,
}

pub fn execution_mode_supported(mode: ExecutionMode) -> bool {
    match mode {
        ExecutionMode::Repl | ExecutionMode::Standard => true,
    }
}

pub fn default_execution_mode() -> ExecutionMode {
    ExecutionMode::default()
}

/// Watermark policy for folding old context out of the active prompt window.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ContextFoldingConfig {
    #[serde(default = "ContextFoldingConfig::default_soft_limit_pct")]
    pub soft_limit_pct: u8,
    #[serde(default = "ContextFoldingConfig::default_hard_limit_pct")]
    pub hard_limit_pct: u8,
}

impl ContextFoldingConfig {
    pub const fn default_soft_limit_pct() -> u8 {
        50
    }

    pub const fn default_hard_limit_pct() -> u8 {
        60
    }

    pub fn validate(self) -> Result<Self, String> {
        if self.soft_limit_pct == 0 || self.hard_limit_pct == 0 {
            return Err("context folding percentages must be greater than 0".to_string());
        }
        if self.soft_limit_pct >= self.hard_limit_pct {
            return Err("context folding soft limit must be less than hard limit".to_string());
        }
        if self.hard_limit_pct >= 100 {
            return Err("context folding hard limit must be less than 100".to_string());
        }
        Ok(self)
    }

    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

impl Default for ContextFoldingConfig {
    fn default() -> Self {
        Self {
            soft_limit_pct: Self::default_soft_limit_pct(),
            hard_limit_pct: Self::default_hard_limit_pct(),
        }
    }
}

/// A message sent from the sandbox to the host during execution.
#[derive(Clone, Debug)]
pub struct SandboxMessage {
    pub text: String,
    /// "final" or "tool_output"
    pub kind: String,
}

/// Sender for streaming progress messages from tools (e.g. live bash output).
pub type ProgressSender = tokio::sync::mpsc::UnboundedSender<SandboxMessage>;

/// A typed parameter for a tool definition.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct ToolParam {
    pub name: String,
    /// REPL type: "str", "int", "float", "bool", "list", "dict", "any"
    #[serde(default = "ToolParam::default_type")]
    pub r#type: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    #[serde(default = "ToolParam::default_required")]
    pub required: bool,
}

impl ToolParam {
    fn default_type() -> String {
        "any".into()
    }
    fn default_required() -> bool {
        true
    }

    pub fn typed(name: &str, ty: &str) -> Self {
        Self {
            name: name.into(),
            r#type: ty.into(),
            description: String::new(),
            required: true,
        }
    }
    pub fn optional(name: &str, ty: &str) -> Self {
        Self {
            name: name.into(),
            r#type: ty.into(),
            description: String::new(),
            required: false,
        }
    }
}

/// A tool definition exposed to the `repl` executor.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub params: Vec<ToolParam>,
    /// REPL return type: "str", "int", "float", "bool", "list", "dict", "any", "None"
    #[serde(
        default = "ToolDefinition::default_returns",
        skip_serializing_if = "String::is_empty"
    )]
    pub returns: String,
    /// Short usage examples for discovery UIs / REPL browsing.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub examples: Vec<String>,
    /// Whether this tool is available to the agent at all.
    #[serde(default)]
    pub enabled: bool,
    /// Whether this tool should be injected into the LLM system prompt/docs.
    #[serde(default)]
    pub injected: bool,
}

impl ToolDefinition {
    fn default_returns() -> String {
        "any".into()
    }

    /// Format as a typed Python signature: `name(param: type, ...) -> ret`
    pub fn signature(&self) -> String {
        let params: Vec<String> = self
            .params
            .iter()
            .map(|p| {
                let mut s = format!("{}: {}", p.name, p.r#type);
                if !p.required {
                    s.push_str(" = None");
                }
                s
            })
            .collect();
        let ret = if self.returns.is_empty() {
            "any"
        } else {
            &self.returns
        };
        format!("{}({}) -> {}", self.name, params.join(", "), ret)
    }

    /// Format all tools as a documentation block for LLM prompts.
    pub fn format_tool_docs(tools: &[ToolDefinition]) -> String {
        tools
            .iter()
            .map(|t| {
                let mut lines = format!("- `{}`", t.signature());
                let summary = t.prompt_summary();
                if !summary.is_empty() {
                    lines.push_str(&format!(" — {}", summary));
                }
                lines
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn prompt_summary(&self) -> String {
        let first_block = self
            .description
            .split("\n\n")
            .next()
            .unwrap_or("")
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        if first_block.chars().count() <= 220 {
            return first_block;
        }
        let compact = first_block.chars().take(217).collect::<String>();
        format!("{compact}...")
    }

    /// Convert the tool signature into a basic JSON Schema object for structured tool-calling APIs.
    pub fn input_schema(&self) -> serde_json::Value {
        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();

        for param in &self.params {
            let schema = match param.r#type.as_str() {
                "str" | "string" => serde_json::json!({ "type": "string" }),
                "int" => serde_json::json!({ "type": "integer" }),
                "float" => serde_json::json!({ "type": "number" }),
                "bool" => serde_json::json!({ "type": "boolean" }),
                "list" => serde_json::json!({ "type": "array", "items": {} }),
                "dict" => serde_json::json!({ "type": "object", "additionalProperties": true }),
                _ => serde_json::json!({}),
            };
            let mut schema_obj = match schema {
                serde_json::Value::Object(obj) => obj,
                _ => serde_json::Map::new(),
            };
            if !param.description.is_empty() {
                schema_obj.insert(
                    "description".to_string(),
                    serde_json::Value::String(param.description.clone()),
                );
            }
            properties.insert(param.name.clone(), serde_json::Value::Object(schema_obj));
            if param.required {
                required.push(param.name.clone());
            }
        }

        serde_json::json!({
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": false,
        })
    }
}

/// An image returned by a tool (e.g. read_file on a PNG).
#[derive(Clone, Debug)]
pub struct ToolImage {
    pub mime: String,
    pub data: Vec<u8>,
    pub label: String,
}

/// Result of executing a tool.
#[derive(Clone, Debug)]
pub struct ToolResult {
    pub success: bool,
    pub result: serde_json::Value,
    pub images: Vec<ToolImage>,
}

impl ToolResult {
    pub fn ok(result: serde_json::Value) -> Self {
        Self {
            success: true,
            result,
            images: vec![],
        }
    }
    pub fn err(result: serde_json::Value) -> Self {
        Self {
            success: false,
            result,
            images: vec![],
        }
    }
    pub fn err_fmt(msg: impl std::fmt::Display) -> Self {
        Self::err(serde_json::json!(msg.to_string()))
    }
    pub fn with_images(success: bool, result: serde_json::Value, images: Vec<ToolImage>) -> Self {
        Self {
            success,
            result,
            images,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ToolParam ──

    #[test]
    fn tool_param_typed() {
        let p = ToolParam::typed("name", "str");
        assert_eq!(p.name, "name");
        assert_eq!(p.r#type, "str");
        assert!(p.required);
    }

    #[test]
    fn tool_param_optional() {
        let p = ToolParam::optional("name", "int");
        assert_eq!(p.name, "name");
        assert_eq!(p.r#type, "int");
        assert!(!p.required);
    }

    // ── ToolDefinition::signature ──

    #[test]
    fn signature_required_params() {
        let td = ToolDefinition {
            name: "foo".into(),
            description: String::new(),
            params: vec![ToolParam::typed("x", "int"), ToolParam::typed("y", "str")],
            returns: "bool".into(),
            examples: vec![],
            enabled: true,
            injected: true,
        };
        assert_eq!(td.signature(), "foo(x: int, y: str) -> bool");
    }

    #[test]
    fn signature_optional_params() {
        let td = ToolDefinition {
            name: "bar".into(),
            description: String::new(),
            params: vec![ToolParam::optional("limit", "int")],
            returns: "list".into(),
            examples: vec![],
            enabled: true,
            injected: true,
        };
        assert_eq!(td.signature(), "bar(limit: int = None) -> list");
    }

    #[test]
    fn signature_empty_params() {
        let td = ToolDefinition {
            name: "noop".into(),
            description: String::new(),
            params: vec![],
            returns: "None".into(),
            examples: vec![],
            enabled: true,
            injected: true,
        };
        assert_eq!(td.signature(), "noop() -> None");
    }

    #[test]
    fn signature_empty_returns_defaults_to_any() {
        let td = ToolDefinition {
            name: "f".into(),
            description: String::new(),
            params: vec![],
            returns: String::new(),
            examples: vec![],
            enabled: true,
            injected: true,
        };
        assert_eq!(td.signature(), "f() -> any");
    }

    // ── format_tool_docs ──

    #[test]
    fn format_tool_docs_with_descriptions() {
        let tools = vec![ToolDefinition {
            name: "read".into(),
            description: "Read a file".into(),
            params: vec![ToolParam {
                name: "path".into(),
                r#type: "str".into(),
                description: "File path".into(),
                required: true,
            }],
            returns: "str".into(),
            examples: vec![],
            enabled: true,
            injected: true,
        }];
        let docs = ToolDefinition::format_tool_docs(&tools);
        assert!(docs.contains("- `read(path: str) -> str`"));
        assert!(docs.contains("— Read a file"));
        assert!(!docs.contains("File path"));
    }

    #[test]
    fn format_tool_docs_empty() {
        let docs = ToolDefinition::format_tool_docs(&[]);
        assert!(docs.is_empty());
    }

    #[test]
    fn context_folding_defaults_and_validation() {
        let cfg = ContextFoldingConfig::default();
        assert_eq!(cfg.soft_limit_pct, 50);
        assert_eq!(cfg.hard_limit_pct, 60);
        assert!(cfg.validate().is_ok());
        assert!(
            ContextFoldingConfig {
                soft_limit_pct: 60,
                hard_limit_pct: 50,
            }
            .validate()
            .is_err()
        );
        assert!(
            ContextFoldingConfig {
                soft_limit_pct: 50,
                hard_limit_pct: 100,
            }
            .validate()
            .is_err()
        );
    }
}

/// Record of a tool call (for context/logging).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ToolCallRecord {
    pub tool: String,
    pub args: serde_json::Value,
    pub result: serde_json::Value,
    pub success: bool,
    pub duration_ms: u64,
}

#[derive(Clone, Debug, Default)]
pub struct PromptContext {
    pub mode: ExecutionMode,
    pub tool_list: String,
    pub tool_names: Vec<String>,
    pub omitted_tool_count: usize,
    pub is_subagent: bool,
    pub can_write: bool,
    pub include_soul: bool,
    pub contributions: Vec<crate::plugin::PromptContribution>,
}

impl PromptContext {
    pub fn has_tool(&self, tool_name: &str) -> bool {
        self.tool_names.iter().any(|name| name == tool_name)
    }
}

#[derive(Clone)]
pub struct ToolExecutionContext {
    pub session_id: String,
    pub host: std::sync::Arc<dyn crate::plugin::SessionManager>,
}

/// Trait for providing tools to the sandbox. Implement this per-project.
#[async_trait::async_trait]
pub trait ToolProvider: Send + Sync + 'static {
    fn definitions(&self) -> Vec<ToolDefinition>;
    fn dynamic_snapshot(&self) -> Option<crate::dynamic::DynamicStateSnapshot> {
        None
    }
    fn fork_dynamic_with_snapshot(
        &self,
        _snapshot: crate::dynamic::DynamicStateSnapshot,
    ) -> Option<std::sync::Arc<dyn ToolProvider>> {
        None
    }
    fn dynamic_generation(&self) -> Option<u64> {
        None
    }
    async fn execute(&self, name: &str, args: &serde_json::Value) -> ToolResult;

    async fn execute_with_context(
        &self,
        name: &str,
        args: &serde_json::Value,
        _context: &ToolExecutionContext,
    ) -> ToolResult {
        self.execute(name, args).await
    }

    /// Execute with progress streaming. Default: delegates to execute().
    async fn execute_streaming(
        &self,
        name: &str,
        args: &serde_json::Value,
        _progress: Option<&ProgressSender>,
    ) -> ToolResult {
        self.execute(name, args).await
    }

    /// Execute with progress streaming and session context. Default: delegates to
    /// `execute_streaming()`.
    async fn execute_streaming_with_context(
        &self,
        name: &str,
        args: &serde_json::Value,
        _context: &ToolExecutionContext,
        progress: Option<&ProgressSender>,
    ) -> ToolResult {
        self.execute_streaming(name, args, progress).await
    }
}
