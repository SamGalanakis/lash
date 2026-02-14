pub mod agent;
#[allow(non_snake_case, unused_imports, non_camel_case_types, dead_code)]
pub(crate) mod baml_client;
pub mod embedded;
pub mod instructions;
pub mod model_info;
pub mod manager;
pub mod oauth;
pub mod provider;
pub mod python_home;
pub mod session;
pub mod store;
pub mod tools;

// Re-exports
pub use agent::{Agent, AgentConfig, AgentEvent, Message, MessageRole, Part, PartKind, PruneState, TokenUsage};
pub use instructions::InstructionLoader;
pub use manager::SessionManager;
pub use store::{AgentState, Store};
pub use provider::{LashConfig, Provider};
pub use session::{ExecResponse, Session, SessionConfig, SessionError, UserPrompt};

/// A message sent from the sandbox to the host during execution.
#[derive(Clone, Debug)]
pub struct SandboxMessage {
    pub text: String,
    /// "progress", "final", or "tool_output"
    pub kind: String,
}

/// Sender for streaming progress messages from tools (e.g. live bash output).
pub type ProgressSender = tokio::sync::mpsc::UnboundedSender<SandboxMessage>;

/// A typed parameter for a tool definition.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ToolParam {
    pub name: String,
    /// Python type: "str", "int", "float", "bool", "list", "dict", "any"
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

/// A tool definition exposed to the Python REPL.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub params: Vec<ToolParam>,
    /// Python return type: "str", "int", "float", "bool", "list", "dict", "any", "None"
    #[serde(
        default = "ToolDefinition::default_returns",
        skip_serializing_if = "String::is_empty"
    )]
    pub returns: String,
    /// Hidden tools are callable via `_call()` but don't appear in the REPL namespace or agent prompt.
    #[serde(default)]
    pub hidden: bool,
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
                if !t.description.is_empty() {
                    lines.push_str(&format!(" — {}", t.description));
                }
                // Include parameter descriptions if any have them
                for p in &t.params {
                    if !p.description.is_empty() {
                        lines.push_str(&format!(
                            "\n    - `{}`: {}",
                            p.name, p.description
                        ));
                    }
                }
                lines
            })
            .collect::<Vec<_>>()
            .join("\n")
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
        Self { success: true, result, images: vec![] }
    }
    pub fn err(result: serde_json::Value) -> Self {
        Self { success: false, result, images: vec![] }
    }
    pub fn with_images(success: bool, result: serde_json::Value, images: Vec<ToolImage>) -> Self {
        Self { success, result, images }
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

/// Trait for providing tools to the sandbox. Implement this per-project.
#[async_trait::async_trait]
pub trait ToolProvider: Send + Sync + 'static {
    fn definitions(&self) -> Vec<ToolDefinition>;
    async fn execute(&self, name: &str, args: &serde_json::Value) -> ToolResult;

    /// Execute with progress streaming. Default: delegates to execute().
    async fn execute_streaming(
        &self,
        name: &str,
        args: &serde_json::Value,
        _progress: Option<&ProgressSender>,
    ) -> ToolResult {
        self.execute(name, args).await
    }
}
