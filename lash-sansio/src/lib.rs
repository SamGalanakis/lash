pub mod llm;
pub mod plugin;
pub mod sansio;
pub mod session;
pub mod session_model;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub use plugin::{CheckpointKind, PluginMessage, PluginSurfaceEvent, PromptContribution};
pub use sansio::{Effect, EffectId, LlmCallError, Response, TurnMachine, TurnMachineConfig};
pub use session::ExecResponse;
pub use session_model::message::MessageOrigin;
pub use session_model::{
    DefaultPromptRenderer, DurableTurnSnapshot, ErrorEnvelope, Message, MessageRole, Part,
    PartKind, PromptOverrideMode, PromptPanel, PromptRenderer, PromptRequest, PromptResponse,
    PromptSectionName, PromptSectionOverride, PromptSelectionMode, PruneState, SessionEvent,
    TokenUsage, default_prompt_renderer, messages_are_live_resume_safe,
};

/// Execution backend for session turns.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMode {
    Repl,
    #[default]
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

/// Strategy for selecting and rendering session context into the next turn.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContextStrategy {
    #[default]
    RollingContext,
}

impl ContextStrategy {
    pub fn validate(self) -> Result<Self, String> {
        Ok(self)
    }
}

pub fn default_context_strategy() -> ContextStrategy {
    ContextStrategy::default()
}

/// A typed parameter for a tool definition.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct ToolParam {
    pub name: String,
    /// REPL type: "str", "int", "float", "bool", "list", "dict", "any"
    #[serde(default = "ToolParam::default_type")]
    pub r#type: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_value: Option<serde_json::Value>,
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
            default_value: None,
            required: true,
        }
    }

    pub fn optional(name: &str, ty: &str) -> Self {
        Self {
            name: name.into(),
            r#type: ty.into(),
            description: String::new(),
            default_value: None,
            required: false,
        }
    }
}

/// A tool definition exposed to the runtime.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub params: Vec<ToolParam>,
    #[serde(
        default = "ToolDefinition::default_returns",
        skip_serializing_if = "String::is_empty"
    )]
    pub returns: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub examples: Vec<String>,
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub injected: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_schema_override: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_schema_override: Option<serde_json::Value>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelTool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    pub output_schema: serde_json::Value,
}

impl ToolDefinition {
    fn default_returns() -> String {
        "any".into()
    }

    pub fn signature(&self) -> String {
        let params: Vec<String> = self
            .params
            .iter()
            .map(|p| {
                let mut s = if p.default_value.is_none() && !p.required {
                    format!("{}?: {}", p.name, display_prompt_type(&p.r#type))
                } else {
                    format!("{}: {}", p.name, display_prompt_type(&p.r#type))
                };
                if let Some(default) = &p.default_value {
                    s.push_str(" = ");
                    s.push_str(&display_default_value(default));
                }
                s
            })
            .collect();
        let ret = if self.returns.is_empty() {
            "any".to_string()
        } else {
            display_prompt_type(&self.returns)
        };
        format!("{}({}) -> {}", self.name, params.join(", "), ret)
    }

    pub fn model_tool(&self) -> ModelTool {
        ModelTool {
            name: self.name.clone(),
            description: self.description.clone(),
            input_schema: self.input_schema(),
            output_schema: self.output_schema(),
        }
    }

    pub fn format_tool_docs(tools: &[ToolDefinition]) -> String {
        tools
            .iter()
            .map(|tool| {
                let mut sections = vec![format!("### `{}`", tool.signature())];
                if !tool.description.trim().is_empty() {
                    sections.push(tool.description.trim().to_string());
                }
                sections.join("\n")
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    pub fn input_schema(&self) -> serde_json::Value {
        if let Some(schema) = &self.input_schema_override {
            return schema.clone();
        }

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
            if let Some(default) = &param.default_value {
                schema_obj.insert("default".to_string(), default.clone());
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

    pub fn output_schema(&self) -> serde_json::Value {
        if let Some(schema) = &self.output_schema_override {
            return schema.clone();
        }

        let ty = self.returns.trim();
        match ty {
            "" | "any" => serde_json::json!({}),
            "str" | "string" => serde_json::json!({ "type": "string" }),
            "int" | "integer" => serde_json::json!({ "type": "integer" }),
            "float" | "number" => serde_json::json!({ "type": "number" }),
            "bool" | "boolean" => serde_json::json!({ "type": "boolean" }),
            "dict" | "json" => {
                serde_json::json!({ "type": "object", "additionalProperties": true })
            }
            "None" | "null" => serde_json::json!({ "type": "null" }),
            _ if ty.starts_with("list") => serde_json::json!({ "type": "array", "items": {} }),
            _ => serde_json::json!({}),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_definition_uses_schema_overrides_for_model_tools() {
        let tool = ToolDefinition {
            name: "mcp__demo__search".to_string(),
            description: "Search demo server".to_string(),
            params: vec![ToolParam::typed("query", "str")],
            returns: "str".to_string(),
            examples: vec![],
            enabled: true,
            injected: true,
            input_schema_override: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "limit": { "type": "integer" }
                },
                "required": ["query"],
                "additionalProperties": false
            })),
            output_schema_override: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "hits": { "type": "array", "items": { "type": "string" } }
                },
                "required": ["hits"],
                "additionalProperties": false
            })),
        };

        let model_tool = tool.model_tool();
        assert_eq!(
            model_tool.input_schema["properties"]["limit"]["type"],
            serde_json::json!("integer")
        );
        assert_eq!(
            model_tool.output_schema["properties"]["hits"]["type"],
            serde_json::json!("array")
        );
    }
}

fn display_prompt_type(ty: &str) -> String {
    let trimmed = ty.trim();
    if let Some(inner) = trimmed
        .strip_prefix("list[")
        .and_then(|rest| rest.strip_suffix(']'))
    {
        return format!("list[{}]", display_prompt_type(inner));
    }
    match trimmed {
        "string" => "str".to_string(),
        "integer" => "int".to_string(),
        "number" => "float".to_string(),
        "boolean" => "bool".to_string(),
        "dict" | "json" => "record".to_string(),
        "None" | "null" => "null".to_string(),
        other => other.to_string(),
    }
}

fn display_default_value(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Bool(v) => v.to_string(),
        serde_json::Value::Number(v) => v.to_string(),
        serde_json::Value::String(v) => format!("{v:?}"),
        _ => serde_json::to_string(value).unwrap_or_else(|_| "null".to_string()),
    }
}

#[derive(Clone, Debug)]
pub struct ToolImage {
    pub mime: String,
    pub data: Vec<u8>,
    pub label: String,
}

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

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ToolCallRecord {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
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
    pub contributions: Vec<PromptContribution>,
}

impl PromptContext {
    pub fn has_tool(&self, tool_name: &str) -> bool {
        self.tool_names.iter().any(|name| name == tool_name)
    }
}
