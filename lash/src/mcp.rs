use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;
use tokio::time::timeout;

use crate::{
    DynamicToolProvider, ReconfigureError, ToolDefinition, ToolExecutionAdapter, ToolExecutionMode,
    ToolImage, ToolParam, ToolResult,
};

const DEFAULT_STARTUP_TIMEOUT_MS: u64 = 10_000;
const DEFAULT_CALL_TIMEOUT_MS: u64 = 60_000;
const SUPPORTED_PROTOCOL_VERSIONS: &[&str] = &["2025-06-18", "2025-03-26", "2024-11-05"];

fn default_startup_timeout_ms() -> u64 {
    DEFAULT_STARTUP_TIMEOUT_MS
}

fn default_call_timeout_ms() -> u64 {
    DEFAULT_CALL_TIMEOUT_MS
}

fn is_default_startup_timeout_ms(value: &u64) -> bool {
    *value == DEFAULT_STARTUP_TIMEOUT_MS
}

fn is_default_call_timeout_ms(value: &u64) -> bool {
    *value == DEFAULT_CALL_TIMEOUT_MS
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "transport", rename_all = "snake_case")]
pub enum McpServerConfig {
    Stdio {
        command: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        args: Vec<String>,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        env: BTreeMap<String, String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cwd: Option<PathBuf>,
        #[serde(
            default = "default_startup_timeout_ms",
            skip_serializing_if = "is_default_startup_timeout_ms"
        )]
        startup_timeout_ms: u64,
        #[serde(
            default = "default_call_timeout_ms",
            skip_serializing_if = "is_default_call_timeout_ms"
        )]
        call_timeout_ms: u64,
    },
}

impl McpServerConfig {
    pub fn stdio(command: impl Into<String>, args: Vec<String>) -> Self {
        Self::Stdio {
            command: command.into(),
            args,
            env: BTreeMap::new(),
            cwd: None,
            startup_timeout_ms: default_startup_timeout_ms(),
            call_timeout_ms: default_call_timeout_ms(),
        }
    }

    pub fn startup_timeout(&self) -> Duration {
        Duration::from_millis(match self {
            Self::Stdio {
                startup_timeout_ms, ..
            } => *startup_timeout_ms,
        })
    }

    pub fn call_timeout(&self) -> Duration {
        Duration::from_millis(match self {
            Self::Stdio {
                call_timeout_ms, ..
            } => *call_timeout_ms,
        })
    }

    fn command_display(&self) -> String {
        match self {
            Self::Stdio { command, args, .. } => {
                let mut parts = vec![command.clone()];
                parts.extend(args.iter().cloned());
                parts.join(" ")
            }
        }
    }

    fn validate(&self, server_name: &str) -> Result<(), McpError> {
        if server_name.trim().is_empty() {
            return Err(McpError::Config(
                "MCP server name cannot be empty".to_string(),
            ));
        }
        if server_name.contains("__") {
            return Err(McpError::Config(format!(
                "MCP server `{server_name}` cannot contain `__`"
            )));
        }
        match self {
            Self::Stdio { command, .. } if command.trim().is_empty() => Err(McpError::Config(
                format!("MCP server `{server_name}` command cannot be empty"),
            )),
            _ => Ok(()),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum McpError {
    #[error("{0}")]
    Config(String),
    #[error("MCP transport I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid MCP JSON message: {0}")]
    Json(#[from] serde_json::Error),
    #[error("MCP protocol error: {0}")]
    Protocol(String),
    #[error("MCP startup timed out for `{server}` after {timeout_ms}ms")]
    StartupTimeout { server: String, timeout_ms: u64 },
    #[error("MCP tool call timed out for `{server}` after {timeout_ms}ms")]
    CallTimeout { server: String, timeout_ms: u64 },
    #[error("failed to decode MCP image payload: {0}")]
    Decode(#[from] base64::DecodeError),
    #[error("dynamic tool registration failed: {0}")]
    Reconfigure(#[from] ReconfigureError),
}

#[derive(Clone)]
struct ImportedTool {
    original_name: String,
    definition: ToolDefinition,
}

struct McpSession {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
}

impl McpSession {
    async fn send_notification(&mut self, method: &str, params: Value) -> Result<(), McpError> {
        let message = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        self.write_message(&message).await
    }

    async fn request(&mut self, method: &str, params: Value) -> Result<Value, McpError> {
        let id = self.next_id;
        self.next_id += 1;
        let message = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        self.write_message(&message).await?;
        self.read_response(id).await
    }

    async fn write_message(&mut self, value: &Value) -> Result<(), McpError> {
        let line = serde_json::to_string(value)?;
        self.stdin.write_all(line.as_bytes()).await?;
        self.stdin.write_all(b"\n").await?;
        self.stdin.flush().await?;
        Ok(())
    }

    async fn read_response(&mut self, request_id: u64) -> Result<Value, McpError> {
        loop {
            let mut line = String::new();
            let read = self.stdout.read_line(&mut line).await?;
            if read == 0 {
                return Err(McpError::Protocol(format!(
                    "server closed stdout while waiting for response to request {request_id}"
                )));
            }
            if line.trim().is_empty() {
                continue;
            }

            let message: Value = serde_json::from_str(line.trim())?;
            let Some(id) = message.get("id") else {
                continue;
            };
            if id.as_u64() != Some(request_id) {
                continue;
            }
            if let Some(error) = message.get("error") {
                return Err(McpError::Protocol(render_protocol_error(error)));
            }
            return message
                .get("result")
                .cloned()
                .ok_or_else(|| McpError::Protocol("response missing `result` field".to_string()));
        }
    }

    fn start_kill(&mut self) {
        let _ = self.child.start_kill();
    }
}

pub struct McpToolExecutionAdapter {
    id: String,
    server_name: String,
    config: McpServerConfig,
    imported_tools: BTreeMap<String, ImportedTool>,
    session: Mutex<Option<McpSession>>,
}

impl McpToolExecutionAdapter {
    pub async fn from_named_config(
        server_name: impl Into<String>,
        config: McpServerConfig,
    ) -> Result<Self, McpError> {
        let server_name = server_name.into();
        config.validate(&server_name)?;
        let connected = connect_with_fallback(&server_name, &config).await?;
        Ok(Self {
            id: format!("mcp:{server_name}"),
            server_name,
            config,
            imported_tools: connected.imported_tools,
            session: Mutex::new(Some(connected.session)),
        })
    }
}

impl Drop for McpToolExecutionAdapter {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.session.try_lock()
            && let Some(session) = guard.as_mut()
        {
            session.start_kill();
        }
    }
}

#[async_trait::async_trait]
impl ToolExecutionAdapter for McpToolExecutionAdapter {
    fn id(&self) -> &str {
        &self.id
    }

    fn advertised_tools(&self) -> Vec<ToolDefinition> {
        self.imported_tools
            .values()
            .map(|tool| tool.definition.clone())
            .collect()
    }

    async fn execute(
        &self,
        tool: &str,
        args: &serde_json::Value,
        _context: Option<crate::ToolExecutionContext>,
        _progress: Option<&crate::ProgressSender>,
    ) -> ToolResult {
        let Some(imported) = self.imported_tools.get(tool) else {
            return ToolResult::err_fmt(format!("Unknown MCP tool: {tool}"));
        };

        let call_timeout = self.config.call_timeout();
        let server_name = self.server_name.clone();
        let raw_name = imported.original_name.clone();
        let args = args.clone();

        let response = timeout(call_timeout, async {
            let mut session_guard = self.session.lock().await;
            let Some(session) = session_guard.as_mut() else {
                return Err(McpError::Protocol(format!(
                    "MCP server `{server_name}` is not connected"
                )));
            };
            session
                .request(
                    "tools/call",
                    json!({
                        "name": raw_name,
                        "arguments": args,
                    }),
                )
                .await
        })
        .await;

        match response {
            Ok(Ok(value)) => match tool_result_from_value(value) {
                Ok(result) => result,
                Err(err) => ToolResult::err_fmt(err),
            },
            Ok(Err(err)) => ToolResult::err_fmt(err),
            Err(_) => ToolResult::err_fmt(McpError::CallTimeout {
                server: server_name,
                timeout_ms: call_timeout.as_millis() as u64,
            }),
        }
    }
}

pub async fn attach_mcp_servers(
    dynamic_tools: &DynamicToolProvider,
    servers: &BTreeMap<String, McpServerConfig>,
) -> Result<(), McpError> {
    for (server_name, config) in servers {
        let adapter = Arc::new(
            McpToolExecutionAdapter::from_named_config(server_name.clone(), config.clone()).await?,
        ) as Arc<dyn ToolExecutionAdapter>;
        dynamic_tools.upsert_adapter(adapter)?;
    }
    Ok(())
}

struct ConnectedSession {
    session: McpSession,
    imported_tools: BTreeMap<String, ImportedTool>,
}

async fn connect_with_fallback(
    server_name: &str,
    config: &McpServerConfig,
) -> Result<ConnectedSession, McpError> {
    let mut last_error: Option<McpError> = None;
    for protocol_version in SUPPORTED_PROTOCOL_VERSIONS {
        match timeout(
            config.startup_timeout(),
            connect_once(server_name, config, protocol_version),
        )
        .await
        {
            Ok(Ok(connected)) => return Ok(connected),
            Ok(Err(err)) => last_error = Some(err),
            Err(_) => {
                last_error = Some(McpError::StartupTimeout {
                    server: server_name.to_string(),
                    timeout_ms: config.startup_timeout().as_millis() as u64,
                })
            }
        }
    }

    Err(last_error.unwrap_or_else(|| {
        McpError::Protocol(format!(
            "failed to negotiate protocol with MCP server `{server_name}`"
        ))
    }))
}

async fn connect_once(
    server_name: &str,
    config: &McpServerConfig,
    protocol_version: &str,
) -> Result<ConnectedSession, McpError> {
    let mut session = spawn_session(config).await?;
    match initialize_session(&mut session, protocol_version).await {
        Ok(()) => {}
        Err(err) => {
            session.start_kill();
            return Err(err);
        }
    }

    let tool_entries = match list_all_tools(&mut session).await {
        Ok(entries) => entries,
        Err(err) => {
            session.start_kill();
            return Err(err);
        }
    };

    let imported_tools = import_tools(server_name, tool_entries)?;
    Ok(ConnectedSession {
        session,
        imported_tools,
    })
}

async fn spawn_session(config: &McpServerConfig) -> Result<McpSession, McpError> {
    match config {
        McpServerConfig::Stdio {
            command,
            args,
            env,
            cwd,
            ..
        } => {
            let mut cmd = Command::new(command);
            cmd.args(args)
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::inherit());
            if let Some(cwd) = cwd {
                cmd.current_dir(cwd);
            }
            for (key, value) in env {
                cmd.env(key, value);
            }

            let mut child = cmd.spawn().map_err(|err| {
                McpError::Protocol(format!(
                    "failed to spawn `{}`: {err}",
                    config.command_display()
                ))
            })?;
            let stdin = child.stdin.take().ok_or_else(|| {
                McpError::Protocol("spawned MCP process has no stdin".to_string())
            })?;
            let stdout = child.stdout.take().ok_or_else(|| {
                McpError::Protocol("spawned MCP process has no stdout".to_string())
            })?;
            Ok(McpSession {
                child,
                stdin,
                stdout: BufReader::new(stdout),
                next_id: 1,
            })
        }
    }
}

async fn initialize_session(
    session: &mut McpSession,
    protocol_version: &str,
) -> Result<(), McpError> {
    session
        .request(
            "initialize",
            json!({
                "protocolVersion": protocol_version,
                "capabilities": {
                    "roots": { "listChanged": false },
                    "sampling": {}
                },
                "clientInfo": {
                    "name": "lash",
                    "version": crate::VERSION,
                }
            }),
        )
        .await?;
    session
        .send_notification("notifications/initialized", json!({}))
        .await?;
    Ok(())
}

async fn list_all_tools(session: &mut McpSession) -> Result<Vec<Value>, McpError> {
    let mut tools = Vec::new();
    let mut cursor: Option<String> = None;

    loop {
        let params = match cursor.clone() {
            Some(cursor) => json!({ "cursor": cursor }),
            None => json!({}),
        };
        let result = session.request("tools/list", params).await?;
        let page = result
            .get("tools")
            .and_then(Value::as_array)
            .ok_or_else(|| McpError::Protocol("tools/list response missing `tools`".to_string()))?;
        tools.extend(page.iter().cloned());
        cursor = result
            .get("nextCursor")
            .or_else(|| result.get("next_cursor"))
            .and_then(Value::as_str)
            .map(str::to_string);
        if cursor.is_none() {
            break;
        }
    }

    Ok(tools)
}

fn import_tools(
    server_name: &str,
    tool_entries: Vec<Value>,
) -> Result<BTreeMap<String, ImportedTool>, McpError> {
    let mut used_names = BTreeSet::new();
    let server_prefix = normalize_identifier(server_name);
    let mut imported = BTreeMap::new();

    for tool in tool_entries {
        let original_name = tool
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| !name.trim().is_empty())
            .ok_or_else(|| McpError::Protocol("tool entry missing `name`".to_string()))?
            .to_string();
        let description = tool
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_string();
        let input_schema = tool
            .get("inputSchema")
            .or_else(|| tool.get("input_schema"))
            .cloned()
            .unwrap_or_else(|| {
                json!({
                    "type": "object",
                    "properties": {},
                    "additionalProperties": true
                })
            });
        let normalized_tool = normalize_identifier(&original_name);
        let prefixed_name = unique_prefixed_name(
            &format!("mcp__{server_prefix}__{normalized_tool}"),
            &mut used_names,
        );
        let params = schema_to_params(&input_schema);

        imported.insert(
            prefixed_name.clone(),
            ImportedTool {
                original_name,
                definition: ToolDefinition {
                    name: prefixed_name,
                    description: if description.is_empty() {
                        format!("MCP tool from server `{server_name}`")
                    } else {
                        format!("[MCP {server_name}] {description}")
                    },
                    params,
                    returns: "any".to_string(),
                    examples: vec![],
                    enabled: true,
                    injected: true,
                    input_schema_override: Some(input_schema),
                    output_schema_override: None,
                    execution_mode: ToolExecutionMode::Parallel,
                },
            },
        );
    }

    Ok(imported)
}

fn schema_to_params(schema: &Value) -> Vec<ToolParam> {
    let required_names: BTreeSet<String> = schema
        .get("required")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_string)
        .collect();

    let mut params = schema
        .get("properties")
        .and_then(Value::as_object)
        .map(|properties| {
            let mut items = properties
                .iter()
                .map(|(name, prop)| ToolParam {
                    name: name.clone(),
                    r#type: lash_type_for_schema(prop),
                    description: prop
                        .get("description")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    default_value: prop.get("default").cloned(),
                    required: required_names.contains(name),
                })
                .collect::<Vec<_>>();
            items.sort_by(|a, b| a.name.cmp(&b.name));
            items
        })
        .unwrap_or_default();

    if params.is_empty()
        && schema
            .get("type")
            .and_then(Value::as_str)
            .map(|ty| ty == "object")
            .unwrap_or(false)
    {
        params.shrink_to_fit();
    }

    params
}

fn lash_type_for_schema(schema: &Value) -> String {
    let resolved = schema
        .get("type")
        .map(resolve_json_schema_type)
        .unwrap_or_else(|| "any".to_string());
    if resolved == "array" {
        if let Some(items) = schema.get("items") {
            let item_ty = match lash_type_for_schema(items).as_str() {
                "str" | "int" | "float" | "bool" | "dict" | "any" => lash_type_for_schema(items),
                _ => "any".to_string(),
            };
            return format!("list[{item_ty}]");
        }
        return "list".to_string();
    }
    match resolved.as_str() {
        "string" => "str".to_string(),
        "integer" => "int".to_string(),
        "number" => "float".to_string(),
        "boolean" => "bool".to_string(),
        "object" => "dict".to_string(),
        "array" => "list".to_string(),
        _ => "any".to_string(),
    }
}

fn resolve_json_schema_type(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Array(items) => items
            .iter()
            .filter_map(Value::as_str)
            .find(|ty| *ty != "null")
            .unwrap_or("any")
            .to_string(),
        _ => "any".to_string(),
    }
}

fn unique_prefixed_name(base: &str, used_names: &mut BTreeSet<String>) -> String {
    if used_names.insert(base.to_string()) {
        return base.to_string();
    }
    for idx in 2.. {
        let candidate = format!("{base}_{idx}");
        if used_names.insert(candidate.clone()) {
            return candidate;
        }
    }
    unreachable!("integer range exhausted while uniquifying tool name")
}

fn normalize_identifier(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut last_underscore = false;
    for ch in raw.chars() {
        let normalized = if ch.is_ascii_alphanumeric() { ch } else { '_' };
        if normalized == '_' {
            if !last_underscore && !out.is_empty() {
                out.push('_');
            }
            last_underscore = true;
        } else {
            out.push(normalized.to_ascii_lowercase());
            last_underscore = false;
        }
    }
    while out.ends_with('_') {
        out.pop();
    }
    if out.is_empty() {
        "tool".to_string()
    } else {
        out
    }
}

fn render_protocol_error(error: &Value) -> String {
    let code = error.get("code").and_then(Value::as_i64);
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("unknown MCP error");
    match code {
        Some(code) => format!("{message} (code {code})"),
        None => message.to_string(),
    }
}

fn tool_result_from_value(value: Value) -> Result<ToolResult, McpError> {
    let is_error = value
        .get("isError")
        .or_else(|| value.get("is_error"))
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let structured = value
        .get("structuredContent")
        .or_else(|| value.get("structured_content"))
        .cloned();
    let content = value
        .get("content")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    let mut text_parts = Vec::new();
    let mut passthrough_items = Vec::new();
    let mut images = Vec::new();

    for item in content {
        match item.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(text) = item.get("text").and_then(Value::as_str) {
                    text_parts.push(text.to_string());
                } else {
                    passthrough_items.push(item);
                }
            }
            Some("image") => {
                let Some(data) = item.get("data").and_then(Value::as_str) else {
                    passthrough_items.push(item);
                    continue;
                };
                let mime = item
                    .get("mimeType")
                    .or_else(|| item.get("mime_type"))
                    .and_then(Value::as_str)
                    .unwrap_or("application/octet-stream");
                let label = item
                    .get("name")
                    .or_else(|| item.get("label"))
                    .and_then(Value::as_str)
                    .unwrap_or("MCP image");
                images.push(ToolImage {
                    mime: mime.to_string(),
                    data: base64::engine::general_purpose::STANDARD.decode(data)?,
                    label: label.to_string(),
                });
            }
            _ => passthrough_items.push(item),
        }
    }

    let result = if let Some(structured) = structured {
        if text_parts.is_empty() && passthrough_items.is_empty() {
            structured
        } else {
            let mut object = serde_json::Map::new();
            object.insert("structured_content".to_string(), structured);
            if !text_parts.is_empty() {
                object.insert("text".to_string(), json!(text_parts.join("\n\n")));
            }
            if !passthrough_items.is_empty() {
                object.insert("content".to_string(), Value::Array(passthrough_items));
            }
            Value::Object(object)
        }
    } else if passthrough_items.is_empty() {
        match text_parts.len() {
            0 => Value::Null,
            1 => Value::String(text_parts.into_iter().next().unwrap_or_default()),
            _ => Value::String(text_parts.join("\n\n")),
        }
    } else if text_parts.is_empty() {
        Value::Array(passthrough_items)
    } else {
        json!({
            "text": text_parts.join("\n\n"),
            "content": passthrough_items,
        })
    };

    Ok(ToolResult::with_images(!is_error, result, images))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_to_params_extracts_top_level_fields() {
        let params = schema_to_params(&json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "filters": {
                    "type": "array",
                    "items": { "type": "string" }
                },
                "strict": {
                    "type": ["boolean", "null"],
                    "default": false
                }
            },
            "required": ["query", "filters"]
        }));

        assert_eq!(params.len(), 3);
        assert_eq!(params[0].name, "filters");
        assert_eq!(params[0].r#type, "list[str]");
        assert_eq!(params[1].name, "query");
        assert_eq!(params[1].r#type, "str");
        assert!(params[1].required);
        assert_eq!(params[2].name, "strict");
        assert_eq!(params[2].r#type, "bool");
        assert_eq!(params[2].default_value, Some(json!(false)));
    }

    #[tokio::test]
    async fn adapter_imports_and_executes_stdio_tools() {
        let initialize = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "demo", "version": "1.0.0" }
            }
        });
        let list = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "result": {
                "tools": [{
                    "name": "search-docs",
                    "description": "Search docs",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": { "type": "string" }
                        },
                        "required": ["query"],
                        "additionalProperties": false
                    }
                }]
            }
        });
        let call = json!({
            "jsonrpc": "2.0",
            "id": 3,
            "result": {
                "content": [{
                    "type": "text",
                    "text": "matched"
                }]
            }
        });

        let script = format!(
            "printf '%s\\n' '{}' '{}' '{}'; cat >/dev/null",
            shell_single_quote(&initialize.to_string()),
            shell_single_quote(&list.to_string()),
            shell_single_quote(&call.to_string()),
        );
        let config = McpServerConfig::stdio("sh", vec!["-lc".to_string(), script]);
        let adapter = McpToolExecutionAdapter::from_named_config("docs", config)
            .await
            .expect("adapter created");
        let defs = adapter.advertised_tools();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "mcp__docs__search_docs");
        assert_eq!(
            defs[0]
                .input_schema_override
                .as_ref()
                .and_then(|schema| schema.get("properties"))
                .and_then(Value::as_object)
                .and_then(|props| props.get("query"))
                .and_then(|query| query.get("type"))
                .cloned(),
            Some(json!("string"))
        );

        let result = adapter
            .execute(
                "mcp__docs__search_docs",
                &json!({ "query": "lash" }),
                None,
                None,
            )
            .await;
        assert!(result.success, "{result:?}");
        assert_eq!(result.result, json!("matched"));
    }

    fn shell_single_quote(text: &str) -> String {
        text.replace('\'', "'\"'\"'")
    }
}
