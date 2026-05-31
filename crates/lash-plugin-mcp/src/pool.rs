//! Per-core MCP connection pool.
//!
//! [`McpConnectionPool`] holds one client per configured server and is shared
//! across every session built from the same [`lash_core::LashCore`]. Connections
//! are established eagerly when the pool is constructed and held for the
//! lifetime of the pool.
//!
//! The wire-level transport is provided by the official [`rmcp`] SDK.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;
use std::sync::RwLock;

use base64::Engine;
use http::{HeaderName, HeaderValue};
use rmcp::model::{CallToolRequestParams, ClientInfo, Content, Implementation, RawContent};
use rmcp::service::{RoleClient, RunningService, ServiceExt};
use rmcp::transport::child_process::TokioChildProcess;
use rmcp::transport::streamable_http_client::{
    StreamableHttpClientTransport, StreamableHttpClientTransportConfig,
};
use serde_json::{Value, json};
use tokio::process::Command;
use tokio::time::timeout;

use lash_core::{
    AttachmentCreateMeta, MediaType, ToolCallOutput, ToolContext, ToolDefinition, ToolFailure,
    ToolFailureClass, ToolFailureSource, ToolResult, ToolRetryDisposition, ToolScheduling,
    ToolValue,
};

use crate::config::McpServerConfig;
use crate::error::McpError;
use crate::naming;

/// Shared, per-core connection pool. Wrapped in `Arc` and cloned into each
/// session plugin instance.
pub struct McpConnectionPool {
    entries: RwLock<BTreeMap<String, Arc<McpEntry>>>,
}

struct McpEntry {
    server_name: String,
    config: McpServerConfig,
    /// `None` until [`McpEntry::connect`] succeeds. Once connected we keep
    /// the running service handle alive until the pool drops; the transport
    /// owns its own process internally.
    service: tokio::sync::Mutex<Option<RunningService<RoleClient, ClientInfo>>>,
    /// Cached, prefixed tool definitions for this server, computed once after
    /// the `tools/list` handshake. Keys are the prefixed names
    /// (`mcp__<server>__<tool>`).
    imported_tools: BTreeMap<String, ImportedTool>,
}

#[derive(Clone)]
struct ImportedTool {
    /// The native MCP tool name as advertised by the server (before
    /// prefixing/normalisation).
    original_name: String,
    definition: ToolDefinition,
}

impl McpConnectionPool {
    /// Construct an empty pool.
    pub fn empty() -> Self {
        Self {
            entries: RwLock::new(BTreeMap::new()),
        }
    }

    /// Build a pool, connecting to every configured server. Servers are
    /// connected sequentially; the first failure aborts.
    pub async fn connect(
        servers: BTreeMap<String, McpServerConfig>,
    ) -> Result<Arc<Self>, McpError> {
        let pool = Arc::new(Self::empty());
        for (name, config) in servers {
            pool.attach(name, config).await?;
        }
        Ok(pool)
    }

    /// Add (or replace) one server in the pool. Connects synchronously and
    /// caches the server's advertised tools.
    pub async fn attach(
        self: &Arc<Self>,
        server_name: String,
        config: McpServerConfig,
    ) -> Result<(), McpError> {
        config.validate(&server_name)?;
        let entry = McpEntry::connect(server_name.clone(), config).await?;
        let mut entries = self
            .entries
            .write()
            .expect("MCP pool entries lock poisoned");
        entries.insert(server_name, Arc::new(entry));
        Ok(())
    }

    /// Remove and shut down one server.
    pub async fn detach(self: &Arc<Self>, server_name: &str) -> Result<(), McpError> {
        let removed = {
            let mut entries = self
                .entries
                .write()
                .expect("MCP pool entries lock poisoned");
            entries.remove(server_name)
        };
        if let Some(entry) = removed {
            entry.shutdown().await;
        }
        Ok(())
    }

    /// All advertised tools across every server, with `mcp__<server>__<tool>`
    /// prefixed names. Cheap — these are precomputed `ToolDefinition` clones.
    pub fn advertised_tools_blocking(&self) -> Vec<ToolDefinition> {
        let guard = self.entries.read().expect("MCP pool entries lock poisoned");
        guard
            .values()
            .flat_map(|entry| {
                entry
                    .imported_tools
                    .values()
                    .map(|tool| tool.definition.clone())
            })
            .collect()
    }

    /// Same as [`advertised_tools_blocking`] but for use from async contexts.
    pub async fn advertised_tools(&self) -> Vec<ToolDefinition> {
        let guard = self.entries.read().expect("MCP pool entries lock poisoned");
        guard
            .values()
            .flat_map(|entry| {
                entry
                    .imported_tools
                    .values()
                    .map(|tool| tool.definition.clone())
            })
            .collect()
    }

    /// Route a prefixed tool call (`mcp__<server>__<tool>`) to the appropriate
    /// server and translate its result back to `ToolResult`.
    pub async fn call_tool(
        &self,
        prefixed_name: &str,
        args: &Value,
        context: &ToolContext<'_>,
    ) -> ToolResult {
        let (entry, original_name) = match self.lookup(prefixed_name).await {
            Some(found) => found,
            None => {
                return ToolResult::err_fmt(format!("Unknown MCP tool: {prefixed_name}"));
            }
        };

        let call_timeout = entry.config.call_timeout();
        let server_name = entry.server_name.clone();
        let arguments = match args {
            Value::Object(map) => Some(map.clone()),
            Value::Null => None,
            other => {
                return ToolResult::err_fmt(format!(
                    "MCP tool `{prefixed_name}` expected an object argument, got {}",
                    other
                ));
            }
        };

        // Clone the peer handle while briefly holding the lock, then release it
        // before issuing the request. `rmcp::Peer` is a cheap, cloneable handle
        // (an mpsc sender plus an internal request-id provider) that supports
        // concurrent in-flight requests, so holding the mutex across the network
        // await would needlessly serialize parallel tool calls to the same
        // server (these tools advertise `ToolScheduling::Parallel`) and risk a
        // guard held across `.await`.
        let peer = {
            let service_guard = entry.service.lock().await;
            match service_guard.as_ref() {
                Some(service) => service.peer().clone(),
                None => {
                    return ToolResult::err_fmt(McpError::Protocol(format!(
                        "MCP server `{server_name}` is not connected"
                    )));
                }
            }
        };

        let response = timeout(call_timeout, async {
            let mut params = CallToolRequestParams::default();
            params.name = original_name.clone().into();
            params.arguments = arguments;
            peer.call_tool(params)
                .await
                .map_err(|err| McpError::Protocol(err.to_string()))
        })
        .await;

        match response {
            Ok(Ok(result)) => tool_result_from_rmcp(result, context),
            Ok(Err(err)) => ToolResult::err_fmt(err),
            Err(_) => ToolResult::err_fmt(McpError::CallTimeout {
                server: server_name,
                timeout_ms: call_timeout.as_millis() as u64,
            }),
        }
    }

    async fn lookup(&self, prefixed_name: &str) -> Option<(Arc<McpEntry>, String)> {
        let guard = self.entries.read().expect("MCP pool entries lock poisoned");
        for entry in guard.values() {
            if let Some(tool) = entry.imported_tools.get(prefixed_name) {
                return Some((Arc::clone(entry), tool.original_name.clone()));
            }
        }
        None
    }

    /// Tear down all connections. Call this before dropping the pool for a
    /// graceful shutdown; `Drop` itself cannot await.
    pub async fn shutdown_all(&self) {
        let entries: Vec<Arc<McpEntry>> = {
            let mut guard = self
                .entries
                .write()
                .expect("MCP pool entries lock poisoned");
            std::mem::take(&mut *guard).into_values().collect()
        };
        for entry in entries {
            entry.shutdown().await;
        }
    }
}

impl McpEntry {
    async fn connect(server_name: String, config: McpServerConfig) -> Result<Self, McpError> {
        let service = timeout(
            config.startup_timeout(),
            connect_service(&server_name, &config),
        )
        .await
        .map_err(|_| McpError::StartupTimeout {
            server: server_name.clone(),
            timeout_ms: config.startup_timeout().as_millis() as u64,
        })??;

        // Bound the discovery call so a server that completes the handshake but
        // then hangs on `tools/list` surfaces a timeout instead of blocking
        // `connect` (and the whole pool construction) indefinitely. Discovery
        // happens during startup, so the startup budget is the natural bound.
        let discovery_timeout = config.startup_timeout();
        let tools = timeout(discovery_timeout, service.peer().list_all_tools())
            .await
            .map_err(|_| McpError::StartupTimeout {
                server: server_name.clone(),
                timeout_ms: discovery_timeout.as_millis() as u64,
            })?
            .map_err(|err| McpError::Protocol(format!("list_tools failed: {err}")))?;
        let imported_tools = import_tools(&server_name, tools);

        Ok(Self {
            server_name,
            config,
            service: tokio::sync::Mutex::new(Some(service)),
            imported_tools,
        })
    }

    async fn shutdown(&self) {
        let mut guard = self.service.lock().await;
        if let Some(service) = guard.take() {
            // `cancel` consumes the service and waits for the transport
            // task to finish. Errors only surface if the transport already
            // shut down; ignore them.
            let _ = service.cancel().await;
        }
    }
}

async fn connect_service(
    server_name: &str,
    config: &McpServerConfig,
) -> Result<RunningService<RoleClient, ClientInfo>, McpError> {
    let mut implementation = Implementation::default();
    implementation.name = "lash".to_string();
    implementation.version = lash_core::VERSION.to_string();
    let mut client_info = ClientInfo::default();
    client_info.client_info = implementation;

    match config {
        McpServerConfig::Stdio {
            command,
            args,
            env,
            cwd,
            ..
        } => {
            let mut cmd = Command::new(command);
            cmd.args(args);
            if let Some(cwd) = cwd {
                cmd.current_dir(cwd);
            }
            for (key, value) in env {
                cmd.env(key, value);
            }
            let transport = TokioChildProcess::new(cmd).map_err(|err| {
                McpError::Protocol(format!(
                    "failed to spawn `{command}` for `{server_name}`: {err}"
                ))
            })?;
            client_info.serve(transport).await.map_err(|err| {
                McpError::Protocol(format!("MCP handshake with `{server_name}`: {err}"))
            })
        }
        McpServerConfig::StreamableHttp { url, headers, .. } => {
            let custom_headers = build_http_headers(server_name, headers)?;
            let config = StreamableHttpClientTransportConfig::with_uri(url.as_str())
                .custom_headers(custom_headers);
            let transport = StreamableHttpClientTransport::from_config(config);
            client_info.serve(transport).await.map_err(|err| {
                McpError::Protocol(format!("MCP handshake with `{server_name}`: {err}"))
            })
        }
        // The legacy HTTP+SSE client transport (deprecated in the MCP spec in
        // favour of streamable HTTP) is not provided by the `rmcp` SDK we
        // depend on, so there is nothing to wire up here. Fail with a clear,
        // actionable error rather than panicking, and point operators at the
        // supported transport — modern SSE-capable servers are reachable via
        // `streamable_http`, which itself negotiates SSE responses.
        McpServerConfig::Sse { .. } => Err(McpError::Config(format!(
            "MCP server `{server_name}` uses the legacy `sse` transport, which is not supported \
             by this build. Use the `streamable_http` transport instead (it speaks the current \
             MCP HTTP transport and handles SSE responses)."
        ))),
    }
}

/// Translate a config `headers` map into the `http` header types `rmcp`'s
/// streamable-HTTP transport expects, failing with a clear config error on a
/// malformed name or value. Header names are case-insensitive per HTTP, so a
/// configured `Authorization` reaches the server as `authorization`.
fn build_http_headers(
    server_name: &str,
    headers: &BTreeMap<String, String>,
) -> Result<HashMap<HeaderName, HeaderValue>, McpError> {
    let mut out = HashMap::with_capacity(headers.len());
    for (name, value) in headers {
        let header_name = HeaderName::try_from(name.as_str()).map_err(|err| {
            McpError::Config(format!(
                "MCP server `{server_name}` has invalid HTTP header name `{name}`: {err}"
            ))
        })?;
        let header_value = HeaderValue::try_from(value.as_str()).map_err(|err| {
            McpError::Config(format!(
                "MCP server `{server_name}` has invalid value for HTTP header `{name}`: {err}"
            ))
        })?;
        out.insert(header_name, header_value);
    }
    Ok(out)
}

fn import_tools(
    server_name: &str,
    tools: Vec<rmcp::model::Tool>,
) -> BTreeMap<String, ImportedTool> {
    let mut used_names = BTreeSet::new();
    let mut imported = BTreeMap::new();
    for tool in tools {
        let original_name = tool.name.to_string();
        let description = tool
            .description
            .as_deref()
            .map(str::trim)
            .unwrap_or_default();
        let input_schema = Value::Object((*tool.input_schema).clone());
        let output_schema = tool
            .output_schema
            .as_ref()
            .map(|s| Value::Object((**s).clone()))
            .unwrap_or_else(|| json!({}));
        let (prefixed, agent_surface) =
            naming::build_prefixed_name(server_name, &original_name, &mut used_names);

        let description = if description.is_empty() {
            format!("MCP tool from server `{server_name}`")
        } else {
            format!("[MCP {server_name}] {description}")
        };

        imported.insert(
            prefixed.clone(),
            ImportedTool {
                original_name,
                definition: ToolDefinition::raw(
                    format!("mcp:{server_name}/{prefixed}"),
                    prefixed,
                    description,
                    input_schema,
                    output_schema,
                )
                .with_agent_surface(agent_surface)
                .with_scheduling(ToolScheduling::Parallel),
            },
        );
    }
    imported
}

fn tool_result_from_rmcp(
    result: rmcp::model::CallToolResult,
    context: &ToolContext<'_>,
) -> ToolResult {
    let is_error = result.is_error.unwrap_or(false);

    let mut text_parts = Vec::new();
    let mut content_items: Vec<ToolValue> = Vec::new();
    let mut has_attachments = false;

    for Content { raw, .. } in result.content {
        match raw {
            RawContent::Text(text) => {
                text_parts.push(text.text.clone());
                content_items.push(ToolValue::String(text.text));
            }
            RawContent::Image(image) => {
                let data = match base64::engine::general_purpose::STANDARD.decode(image.data) {
                    Ok(bytes) => bytes,
                    Err(err) => {
                        return ToolResult::err_fmt(McpError::Decode(err));
                    }
                };
                let Some(media_type) = MediaType::from_mime(&image.mime_type) else {
                    return ToolResult::err_fmt(format!(
                        "Unsupported MCP image MIME type: {}",
                        image.mime_type
                    ));
                };
                let reference = match context.attachments().put(
                    data,
                    AttachmentCreateMeta::new(media_type, None, None, Some("MCP image".into())),
                ) {
                    Ok(reference) => reference,
                    Err(err) => {
                        return ToolResult::err_fmt(format!(
                            "Failed to store MCP image attachment: {err}"
                        ));
                    }
                };
                has_attachments = true;
                content_items.push(ToolValue::Attachment(reference));
            }
            other => {
                if let Ok(value) = serde_json::to_value(&other) {
                    content_items.push(ToolValue::from(value));
                }
            }
        }
    }

    let value = if let Some(structured) = result.structured_content {
        if !has_attachments {
            ToolValue::from(structured)
        } else {
            ToolValue::Object(
                [
                    ("structured".to_string(), ToolValue::from(structured)),
                    ("content".to_string(), ToolValue::Array(content_items)),
                ]
                .into_iter()
                .collect(),
            )
        }
    } else if content_items.is_empty() {
        ToolValue::Null
    } else if content_items.len() == 1 {
        content_items.into_iter().next().unwrap_or(ToolValue::Null)
    } else {
        ToolValue::Array(content_items)
    };
    if is_error {
        ToolResult::from_output(ToolCallOutput::failure(ToolFailure {
            class: ToolFailureClass::Execution,
            code: "mcp_tool_error".into(),
            message: if text_parts.is_empty() {
                "MCP tool returned an error".into()
            } else {
                text_parts.join("\n\n")
            },
            source: ToolFailureSource::Tool,
            retry: ToolRetryDisposition::Never,
            raw: Some(value),
        }))
    } else {
        ToolResult::from_output(ToolCallOutput::success(value))
    }
}

impl Drop for McpConnectionPool {
    fn drop(&mut self) {
        // We can't .await in Drop. The RunningService values inside each
        // entry will cancel their processes when they're dropped
        // (rmcp drops the transport, which kills the child process or
        // closes the HTTP connection). For a graceful shutdown, callers
        // should call `shutdown_all` themselves.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression for the header-drop bug: custom/auth headers configured for
    /// an HTTP MCP server must be translated into the `http` header types the
    /// transport actually sends. Before the fix, `connect_service` called
    /// `from_uri` and dropped the configured `headers` map entirely, so an
    /// `Authorization` header never reached the server.
    #[test]
    fn build_http_headers_carries_configured_headers() {
        let mut headers = BTreeMap::new();
        headers.insert(
            "Authorization".to_string(),
            "Bearer secret-token".to_string(),
        );
        headers.insert("X-Tenant".to_string(), "acme".to_string());

        let built = build_http_headers("api", &headers).expect("valid headers convert");

        assert_eq!(
            built
                .get(&HeaderName::from_static("authorization"))
                .map(|v| v.to_str().unwrap()),
            Some("Bearer secret-token"),
            "configured Authorization header must be carried through to the transport"
        );
        assert_eq!(
            built
                .get(&HeaderName::from_static("x-tenant"))
                .map(|v| v.to_str().unwrap()),
            Some("acme")
        );
        assert_eq!(built.len(), 2);
    }

    #[test]
    fn build_http_headers_empty_map_is_empty() {
        let built = build_http_headers("api", &BTreeMap::new()).expect("empty converts");
        assert!(built.is_empty());
    }

    #[test]
    fn build_http_headers_rejects_malformed_name() {
        let mut headers = BTreeMap::new();
        headers.insert("Bad Header Name".to_string(), "x".to_string());
        let err = build_http_headers("api", &headers).expect_err("malformed name rejected");
        assert!(
            matches!(err, McpError::Config(_)),
            "expected a config error, got {err:?}"
        );
    }

    #[test]
    fn build_http_headers_rejects_malformed_value() {
        let mut headers = BTreeMap::new();
        // A newline is not a legal header value byte.
        headers.insert("X-Bad".to_string(), "line1\nline2".to_string());
        let err = build_http_headers("api", &headers).expect_err("malformed value rejected");
        assert!(
            matches!(err, McpError::Config(_)),
            "expected a config error, got {err:?}"
        );
    }

    /// The legacy `sse` transport is unsupported by the rmcp build we depend
    /// on. It must surface a clear, non-panicking config error (not a `todo!()`
    /// or silent success) so operators know to switch to `streamable_http`.
    #[tokio::test]
    async fn sse_transport_reports_clear_unsupported_error() {
        let err = connect_service("legacy", &McpServerConfig::sse("http://localhost:9/sse"))
            .await
            .expect_err("sse transport must error, not connect");
        match err {
            McpError::Config(msg) => {
                assert!(
                    msg.contains("streamable_http"),
                    "error should point operators at the supported transport: {msg}"
                );
            }
            other => panic!("expected a config error for sse, got {other:?}"),
        }
    }

    /// Regression for the missing discovery timeout: a server that completes
    /// the handshake but then hangs on `tools/list` must surface a
    /// `StartupTimeout` rather than blocking `connect` forever.
    #[tokio::test]
    async fn discovery_hang_surfaces_startup_timeout() {
        let initialize = json!({
            "jsonrpc": "2.0",
            "id": 0,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "demo", "version": "1.0.0" }
            }
        });

        // Respond to `initialize`, swallow `notifications/initialized`, read the
        // `tools/list` request line, then hang (never respond) by blocking on
        // stdin. The short startup timeout must trip.
        let script = "\
            read -r _; printf '%s\\n' \"$RESP1\"; \
            read -r _; \
            read -r _; \
            cat >/dev/null"
            .to_string();

        let mut env = BTreeMap::new();
        env.insert("RESP1".to_string(), initialize.to_string());

        let config = McpServerConfig::Stdio {
            command: "sh".to_string(),
            args: vec!["-c".to_string(), script],
            env,
            cwd: None,
            startup_timeout_ms: 750,
            call_timeout_ms: 10_000,
        };

        match McpEntry::connect("hangs".to_string(), config).await {
            Err(McpError::StartupTimeout { .. }) => {}
            Err(other) => panic!("expected StartupTimeout from a hung tools/list, got {other:?}"),
            Ok(_) => panic!("a hung tools/list must not connect"),
        }
    }

    /// Regression for the service mutex held across the network await: two
    /// concurrent `tools/call` requests to the same server must be able to be
    /// in flight at once. The mock refuses to answer the first call until it
    /// has read the second request line, so a serializing implementation (lock
    /// held across `.await`) would deadlock and time out, while the concurrent
    /// implementation completes both calls.
    #[tokio::test]
    async fn concurrent_calls_are_not_serialized_by_the_service_mutex() {
        let initialize = json!({
            "jsonrpc": "2.0",
            "id": 0,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "demo", "version": "1.0.0" }
            }
        });
        let list = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "tools": [{
                    "name": "ping",
                    "description": "Ping",
                    "inputSchema": { "type": "object", "properties": {} }
                }]
            }
        });
        // rmcp assigns request ids 2 and 3 to the two concurrent calls. The
        // mock reads BOTH request lines before emitting EITHER response, which
        // is only possible if both requests are in flight concurrently.
        let call2 = json!({ "jsonrpc": "2.0", "id": 2, "result": { "content": [{ "type": "text", "text": "pong" }] } });
        let call3 = json!({ "jsonrpc": "2.0", "id": 3, "result": { "content": [{ "type": "text", "text": "pong" }] } });

        let script = "\
            read -r _; printf '%s\\n' \"$RESP1\"; \
            read -r _; \
            read -r _; printf '%s\\n' \"$RESP2\"; \
            read -r _; \
            read -r _; \
            printf '%s\\n' \"$RESP3\"; \
            printf '%s\\n' \"$RESP4\"; \
            cat >/dev/null"
            .to_string();

        let mut env = BTreeMap::new();
        env.insert("RESP1".to_string(), initialize.to_string());
        env.insert("RESP2".to_string(), list.to_string());
        env.insert("RESP3".to_string(), call2.to_string());
        env.insert("RESP4".to_string(), call3.to_string());

        let mut servers = BTreeMap::new();
        servers.insert(
            "svc".to_string(),
            McpServerConfig::Stdio {
                command: "sh".to_string(),
                args: vec!["-c".to_string(), script],
                env,
                cwd: None,
                startup_timeout_ms: 10_000,
                call_timeout_ms: 5_000,
            },
        );

        let pool = McpConnectionPool::connect(servers)
            .await
            .expect("connects to concurrency mock");

        let ctx = lash_core::testing::mock_tool_context();
        let args = json!({});
        let (a, b) = tokio::join!(
            pool.call_tool("mcp__svc__ping", &args, &ctx),
            pool.call_tool("mcp__svc__ping", &args, &ctx),
        );
        assert!(a.is_success(), "first concurrent call failed: {a:?}");
        assert!(b.is_success(), "second concurrent call failed: {b:?}");

        pool.shutdown_all().await;
    }
}
