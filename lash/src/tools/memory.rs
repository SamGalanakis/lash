use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use serde_json::json;

use crate::agent::format_tool_result_content;
use crate::plugin::{
    PromptContribution, SessionContextSurface, SessionCreateRequest, SessionPluginMode,
};
use crate::search::{SearchDoc, SearchMode, limit_from_args, rank_docs, truncate_preview};
use crate::tools::run_blocking;
use crate::{
    ContextStrategy, InputItem, Message, PartKind, SessionPolicy, ToolDefinition,
    ToolExecutionContext, ToolParam, ToolProvider, ToolResult, TurnInput,
};

use super::{Glob, Ls, ReadFile, require_str};

const SEARCH_PREVIEW_CHARS: usize = 700;
const READ_MEMORY_MAX_CHARS: usize = 16_000;
const RECALL_MAX_ITEMS: usize = 8;

#[derive(Clone)]
pub struct FilteredToolProvider {
    inner: Arc<dyn ToolProvider>,
    allowed: HashSet<String>,
}

impl FilteredToolProvider {
    pub fn new(
        inner: Arc<dyn ToolProvider>,
        allowed: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            inner,
            allowed: allowed.into_iter().map(Into::into).collect(),
        }
    }

    fn allowed(&self, name: &str) -> bool {
        self.allowed.contains(name)
    }
}

#[async_trait::async_trait]
impl ToolProvider for FilteredToolProvider {
    fn definitions(&self) -> Vec<ToolDefinition> {
        self.inner
            .definitions()
            .into_iter()
            .filter(|def| self.allowed(&def.name))
            .collect()
    }

    async fn execute(&self, name: &str, args: &serde_json::Value) -> ToolResult {
        if !self.allowed(name) {
            return ToolResult::err_fmt(format_args!("Unknown tool: {name}"));
        }
        self.inner.execute(name, args).await
    }

    async fn execute_with_context(
        &self,
        name: &str,
        args: &serde_json::Value,
        context: &ToolExecutionContext,
    ) -> ToolResult {
        if !self.allowed(name) {
            return ToolResult::err_fmt(format_args!("Unknown tool: {name}"));
        }
        self.inner.execute_with_context(name, args, context).await
    }

    async fn execute_streaming(
        &self,
        name: &str,
        args: &serde_json::Value,
        progress: Option<&crate::ProgressSender>,
    ) -> ToolResult {
        if !self.allowed(name) {
            return ToolResult::err_fmt(format_args!("Unknown tool: {name}"));
        }
        self.inner.execute_streaming(name, args, progress).await
    }

    async fn execute_streaming_with_context(
        &self,
        name: &str,
        args: &serde_json::Value,
        context: &ToolExecutionContext,
        progress: Option<&crate::ProgressSender>,
    ) -> ToolResult {
        if !self.allowed(name) {
            return ToolResult::err_fmt(format_args!("Unknown tool: {name}"));
        }
        self.inner
            .execute_streaming_with_context(name, args, context, progress)
            .await
    }
}

#[derive(Clone, Default)]
pub struct SessionHistoryTools;

#[derive(Clone, Default)]
pub struct RecallAgentTools;

#[derive(Clone)]
struct RecallSubmitTool {
    submission: Arc<Mutex<Option<serde_json::Value>>>,
}

impl RecallSubmitTool {
    fn new() -> Self {
        Self {
            submission: Arc::new(Mutex::new(None)),
        }
    }

    fn take(&self) -> Option<serde_json::Value> {
        self.submission
            .lock()
            .ok()
            .and_then(|mut guard| guard.take())
    }
}

#[derive(Clone)]
enum MemorySource {
    Message {
        id: String,
        role: String,
        text: String,
    },
    Tool {
        id: String,
        tool: String,
        args: String,
        result: String,
    },
}

impl MemorySource {
    fn item_type(&self) -> &'static str {
        match self {
            Self::Message { .. } => "message_excerpt",
            Self::Tool { .. } => "tool_result_excerpt",
        }
    }

    fn id(&self) -> &str {
        match self {
            Self::Message { id, .. } | Self::Tool { id, .. } => id,
        }
    }

    fn full_text(&self) -> String {
        match self {
            Self::Message { role, text, .. } => format!("{role}: {text}"),
            Self::Tool {
                tool, args, result, ..
            } => {
                if args.is_empty() {
                    format!("tool {tool}\n\n{result}")
                } else {
                    format!("tool {tool}\nargs: {args}\n\n{result}")
                }
            }
        }
    }

    fn fields(&self) -> HashMap<&'static str, String> {
        let mut fields = HashMap::new();
        match self {
            Self::Message { role, text, .. } => {
                fields.insert("role", role.clone());
                fields.insert("content", text.clone());
            }
            Self::Tool {
                tool, args, result, ..
            } => {
                fields.insert("tool", tool.clone());
                fields.insert("args", args.clone());
                fields.insert("result", result.clone());
            }
        }
        fields
    }
}

fn message_text(message: &Message) -> String {
    message
        .parts
        .iter()
        .filter_map(|part| match part.kind {
            PartKind::Text | PartKind::Prose | PartKind::Code => Some(part.content.as_str()),
            _ => None,
        })
        .filter(|text| !text.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn truncate_text(text: &str, limit: usize) -> (String, bool) {
    if text.chars().count() <= limit {
        return (text.to_string(), false);
    }
    let truncated = text.chars().take(limit).collect::<String>();
    (format!("{truncated}..."), true)
}

fn history_sources(snapshot: &crate::AgentStateEnvelope) -> Vec<MemorySource> {
    let mut out = snapshot
        .messages
        .iter()
        .filter_map(|message| {
            let text = message_text(message);
            if text.trim().is_empty() {
                return None;
            }
            Some(MemorySource::Message {
                id: format!("msg:{}", message.id),
                role: format!("{:?}", message.role).to_ascii_lowercase(),
                text,
            })
        })
        .collect::<Vec<_>>();
    out.extend(snapshot.tool_calls.iter().enumerate().map(|(idx, record)| {
        let raw_id = record
            .call_id
            .clone()
            .unwrap_or_else(|| format!("idx:{idx}"));
        MemorySource::Tool {
            id: format!("tool:{raw_id}"),
            tool: record.tool.clone(),
            args: serde_json::to_string(&record.args).unwrap_or_default(),
            result: format_tool_result_content(record.success, &record.result),
        }
    }));
    out
}

fn search_snapshot(snapshot: &crate::AgentStateEnvelope, args: &serde_json::Value) -> ToolResult {
    let query = args
        .get("query")
        .and_then(|value| value.as_str())
        .unwrap_or_default()
        .trim()
        .to_string();
    let limit = limit_from_args(args).min(RECALL_MAX_ITEMS.max(1));
    let sources = history_sources(snapshot);
    if query.is_empty() {
        let items = sources
            .into_iter()
            .rev()
            .take(limit)
            .map(memory_preview_item)
            .collect::<Vec<_>>();
        return ToolResult::ok(json!(items));
    }

    let docs = sources
        .iter()
        .map(|source| SearchDoc {
            fields: source.fields(),
        })
        .collect::<Vec<_>>();
    let ranked = rank_docs(
        &docs,
        &query,
        SearchMode::parse(args.get("mode").and_then(|value| value.as_str())),
        args.get("regex").and_then(|value| value.as_str()),
        &[
            ("content", 3.0),
            ("result", 2.0),
            ("tool", 1.0),
            ("args", 1.0),
            ("role", 0.5),
        ],
    );
    let items = ranked
        .into_iter()
        .take(limit)
        .map(|(idx, _, _)| memory_preview_item(sources[idx].clone()))
        .collect::<Vec<_>>();
    ToolResult::ok(json!(items))
}

fn read_snapshot_item(snapshot: &crate::AgentStateEnvelope, id: &str) -> ToolResult {
    if let Some(message_id) = id.strip_prefix("msg:") {
        if let Some(message) = snapshot
            .messages
            .iter()
            .find(|message| message.id == message_id)
        {
            let text = message_text(message);
            let (content, truncated) = truncate_text(&text, READ_MEMORY_MAX_CHARS);
            return ToolResult::ok(json!({
                "id": id,
                "type": "message_excerpt",
                "content": content,
                "truncated": truncated,
            }));
        }
        return ToolResult::err_fmt(format_args!("Unknown memory item: {id}"));
    }

    if let Some(tool_id) = id.strip_prefix("tool:") {
        let record = if let Some(record) = snapshot
            .tool_calls
            .iter()
            .find(|record| record.call_id.as_deref() == Some(tool_id))
        {
            Some(record)
        } else if let Some(index) = tool_id
            .strip_prefix("idx:")
            .and_then(|value| value.parse::<usize>().ok())
        {
            snapshot.tool_calls.get(index)
        } else {
            None
        };
        if let Some(record) = record {
            let text = format_tool_result_content(record.success, &record.result);
            let (content, truncated) = truncate_text(&text, READ_MEMORY_MAX_CHARS);
            return ToolResult::ok(json!({
                "id": id,
                "type": "tool_result_excerpt",
                "content": content,
                "truncated": truncated,
            }));
        }
        return ToolResult::err_fmt(format_args!("Unknown memory item: {id}"));
    }

    if let Some(path) = id.strip_prefix("file:") {
        match std::fs::read_to_string(path) {
            Ok(text) => {
                let (content, truncated) = truncate_text(&text, READ_MEMORY_MAX_CHARS);
                return ToolResult::ok(json!({
                    "id": id,
                    "type": "file_excerpt",
                    "content": content,
                    "truncated": truncated,
                }));
            }
            Err(err) => {
                return ToolResult::err_fmt(format_args!("Failed to read {path}: {err}"));
            }
        }
    }

    ToolResult::err_fmt(format_args!("Unsupported memory item id: {id}"))
}

fn memory_preview_item(source: MemorySource) -> serde_json::Value {
    let full = source.full_text();
    let content = truncate_preview(&full, SEARCH_PREVIEW_CHARS);
    let truncated = content != full;
    json!({
        "id": source.id(),
        "type": source.item_type(),
        "content": content,
        "truncated": truncated,
    })
}

fn parse_queries(args: &serde_json::Value) -> Result<Vec<String>, ToolResult> {
    let queries = args
        .get("queries")
        .and_then(|value| value.as_array())
        .ok_or_else(|| ToolResult::err_fmt("Missing required parameter: queries"))?;
    let queries = queries
        .iter()
        .filter_map(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    if queries.is_empty() {
        return Err(ToolResult::err_fmt(
            "queries must contain at least one non-empty string",
        ));
    }
    Ok(queries)
}

fn recall_prompt(queries: &[String]) -> String {
    let rendered_queries = queries
        .iter()
        .map(|query| format!("- {query}"))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "You are a session recall-agent worker. Search the current session history and current workspace only as needed.\n\nUse `search_session` first. Use `read_memory` to expand promising hits. Use `ls`, `glob`, or `read_file` only when history points to relevant files.\n\nWhen done, call `submit_memory_recall` exactly once.\n- `summary`: short freeform text for the parent agent.\n- `items`: a flat list of high-signal memory items.\n\nEach item must be an object with exactly these fields:\n- `type`: short string like `message_excerpt`, `tool_result_excerpt`, or `file_excerpt`\n- `content`: concise rendered content useful to the parent agent\n- `id`: stable backing identifier such as `msg:...`, `tool:...`, or `file:...`\n- `truncated`: boolean\n\nRules:\n- Return a flat list, not grouped by query.\n- Prefer source-backed excerpts over paraphrase when possible.\n- Keep the list short and high-signal.\n- Reuse exact IDs returned by tools when you cite a message or tool result.\n- If you cite a file, use `file:<path>`.\n\nUser recall requests:\n{rendered_queries}"
    )
}

fn validate_recall_submission(
    summary: Option<&str>,
    items: &serde_json::Value,
) -> Result<serde_json::Value, ToolResult> {
    let Some(items) = items.as_array() else {
        return Err(ToolResult::err_fmt(
            "submit_memory_recall.items must be an array",
        ));
    };
    let mut validated_items = Vec::with_capacity(items.len());
    for item in items {
        let Some(obj) = item.as_object() else {
            return Err(ToolResult::err_fmt(
                "submit_memory_recall.items must contain only objects",
            ));
        };
        let Some(item_type) = obj.get("type").and_then(|value| value.as_str()) else {
            return Err(ToolResult::err_fmt(
                "submit_memory_recall.items omitted `type`",
            ));
        };
        let Some(content) = obj.get("content").and_then(|value| value.as_str()) else {
            return Err(ToolResult::err_fmt(
                "submit_memory_recall.items omitted `content`",
            ));
        };
        let Some(id) = obj.get("id").and_then(|value| value.as_str()) else {
            return Err(ToolResult::err_fmt(
                "submit_memory_recall.items omitted `id`",
            ));
        };
        let truncated = obj
            .get("truncated")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        validated_items.push(json!({
            "type": item_type,
            "content": content,
            "id": id,
            "truncated": truncated,
        }));
    }
    Ok(json!({
        "summary": summary.unwrap_or_default(),
        "items": validated_items,
    }))
}

fn recall_child_policy(base: &SessionPolicy) -> SessionPolicy {
    let (model, model_variant) = if let Some(model) = base.recall_agent_model.as_ref() {
        (
            model.clone(),
            preferred_recall_variant(&base.provider, model.as_str()),
        )
    } else if matches!(base.provider, crate::Provider::Codex { .. }) {
        (
            "gpt-5.4-mini".to_string(),
            preferred_recall_variant(&base.provider, "gpt-5.4-mini"),
        )
    } else if let Some((model, variant)) = base.provider.default_agent_model("low") {
        (model.to_string(), variant.map(str::to_string))
    } else {
        (
            base.model.clone(),
            base.provider
                .default_model_variant(&base.model)
                .map(str::to_string)
                .or_else(|| base.model_variant.clone()),
        )
    };
    SessionPolicy {
        model,
        model_variant,
        provider: base.provider.clone(),
        max_context_tokens: base.max_context_tokens,
        sub_agent: true,
        recall_agent_model: base.recall_agent_model.clone(),
        session_id: base.session_id.clone(),
        max_turns: None,
        include_soul: false,
        execution_mode: crate::ExecutionMode::Standard,
        context_strategy: ContextStrategy::RollingContext,
    }
}

fn preferred_recall_variant(provider: &crate::Provider, model: &str) -> Option<String> {
    if provider.supported_variants(model).contains(&"low") {
        return Some("low".to_string());
    }
    provider.default_model_variant(model).map(str::to_string)
}

impl RecallAgentTools {
    pub fn prompt_contributions(trimmed: bool) -> Vec<PromptContribution> {
        let guidance = if trimmed {
            "### Recall Agent\nOlder session history has been trimmed from the prompt. Use `recall_memory` to recover high-signal older context when it matters. The result includes a short summary plus a flat list of source-backed memory items. Use `see_full_memory` to expand a returned item by id when needed."
        } else {
            "### Recall Agent\nUse `recall_memory` when older session history may matter, even if it is no longer convenient to inspect directly in the prompt. The result includes a short summary plus a flat list of source-backed memory items. Use `see_full_memory` to expand a returned item by id when needed."
        };
        vec![PromptContribution::guidance(guidance)]
    }
}

#[async_trait::async_trait]
impl ToolProvider for RecallSubmitTool {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "submit_memory_recall".into(),
            description: "Submit the structured recall result for the parent agent.".into(),
            params: vec![
                ToolParam::optional("summary", "str"),
                ToolParam::typed("items", "list"),
            ],
            returns: "dict".into(),
            examples: vec![],
            enabled: true,
            injected: true,
            input_schema_override: None,
            output_schema_override: None,
        }]
    }

    async fn execute(&self, name: &str, args: &serde_json::Value) -> ToolResult {
        if name != "submit_memory_recall" {
            return ToolResult::err_fmt(format_args!("Unknown tool: {name}"));
        }
        let summary = args.get("summary").and_then(|value| value.as_str());
        let Some(items) = args.get("items") else {
            return ToolResult::err_fmt("Missing required parameter: items");
        };
        let submission = match validate_recall_submission(summary, items) {
            Ok(value) => value,
            Err(err) => return err,
        };
        match self.submission.lock() {
            Ok(mut guard) => {
                *guard = Some(submission.clone());
                ToolResult::ok(
                    json!({"submitted": true, "item_count": submission["items"].as_array().map(|items| items.len()).unwrap_or(0)}),
                )
            }
            Err(_) => ToolResult::err_fmt("submit_memory_recall storage is unavailable"),
        }
    }
}

#[async_trait::async_trait]
impl ToolProvider for SessionHistoryTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![
            ToolDefinition {
                name: "search_session".into(),
                description: "Search raw current-session messages and tool-call history, returning a flat list of candidate memory items.".into(),
                params: vec![
                    ToolParam::optional("query", "str"),
                    ToolParam::optional("mode", "str"),
                    ToolParam::optional("regex", "str"),
                    ToolParam::optional("limit", "int"),
                ],
                returns: "list".into(),
                examples: vec![],
                enabled: true,
                injected: true,
                input_schema_override: None,
                output_schema_override: None,
            },
            ToolDefinition {
                name: "read_memory".into(),
                description: "Read a full current-session memory item by id.".into(),
                params: vec![ToolParam::typed("id", "str")],
                returns: "dict".into(),
                examples: vec![],
                enabled: true,
                injected: true,
                input_schema_override: None,
                output_schema_override: None,
            },
        ]
    }

    async fn execute(&self, name: &str, _args: &serde_json::Value) -> ToolResult {
        ToolResult::err_fmt(format_args!("{name} requires session context"))
    }

    async fn execute_with_context(
        &self,
        name: &str,
        args: &serde_json::Value,
        context: &ToolExecutionContext,
    ) -> ToolResult {
        let snapshot = match context.host.snapshot_session(&context.session_id).await {
            Ok(snapshot) => snapshot,
            Err(err) => {
                return ToolResult::err_fmt(format_args!("Failed to snapshot session: {err}"));
            }
        };
        let name = name.to_string();
        let args = args.clone();
        run_blocking(move || match name.as_str() {
            "search_session" => search_snapshot(&snapshot, &args),
            "read_memory" => match require_str(&args, "id") {
                Ok(id) => read_snapshot_item(&snapshot, id),
                Err(err) => err,
            },
            _ => ToolResult::err_fmt(format_args!("Unknown tool: {name}")),
        })
        .await
    }
}

#[async_trait::async_trait]
impl ToolProvider for RecallAgentTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![
            ToolDefinition {
                name: "recall_memory".into(),
                description: "Recall older session context with a low-cost recall agent and return a short summary plus a flat list of memory items.".into(),
                params: vec![ToolParam::typed("queries", "list")],
                returns: "dict".into(),
                examples: vec![],
                enabled: true,
                injected: true,
                input_schema_override: None,
                output_schema_override: None,
            },
            ToolDefinition {
                name: "see_full_memory".into(),
                description: "Expand a returned session memory item by id.".into(),
                params: vec![ToolParam::typed("id", "str")],
                returns: "dict".into(),
                examples: vec![],
                enabled: true,
                injected: true,
                input_schema_override: None,
                output_schema_override: None,
            },
        ]
    }

    async fn execute(&self, name: &str, _args: &serde_json::Value) -> ToolResult {
        ToolResult::err_fmt(format_args!("{name} requires session context"))
    }

    async fn execute_with_context(
        &self,
        name: &str,
        args: &serde_json::Value,
        context: &ToolExecutionContext,
    ) -> ToolResult {
        match name {
            "see_full_memory" => {
                let id = match require_str(args, "id") {
                    Ok(id) => id,
                    Err(err) => return err,
                };
                let snapshot = match context.host.snapshot_session(&context.session_id).await {
                    Ok(snapshot) => snapshot,
                    Err(err) => {
                        return ToolResult::err_fmt(format_args!(
                            "Failed to snapshot session: {err}"
                        ));
                    }
                };
                return read_snapshot_item(&snapshot, id);
            }
            "recall_memory" => {}
            _ => return ToolResult::err_fmt(format_args!("Unknown tool: {name}")),
        }

        let queries = match parse_queries(args) {
            Ok(queries) => queries,
            Err(err) => return err,
        };
        let snapshot = match context.host.snapshot_session(&context.session_id).await {
            Ok(snapshot) => snapshot,
            Err(err) => {
                return ToolResult::err_fmt(format_args!("Failed to snapshot session: {err}"));
            }
        };
        let agent_id = format!(
            "{}-recall-agent-{}",
            context.session_id,
            uuid::Uuid::new_v4()
        );
        let policy = recall_child_policy(&snapshot.policy);
        let submit_tool = RecallSubmitTool::new();
        let request = SessionCreateRequest {
            agent_id: Some(agent_id.clone()),
            start: crate::SessionStartPoint::Snapshot {
                snapshot: Box::new(snapshot),
            },
            policy: Some(policy),
            plugin_mode: SessionPluginMode::Fresh,
            initial_messages: Vec::new(),
            context_surface: SessionContextSurface {
                include_base_tools: false,
                tool_providers: vec![
                    Arc::new(SessionHistoryTools),
                    Arc::new(Ls),
                    Arc::new(Glob),
                    Arc::new(ReadFile::new()),
                    Arc::new(submit_tool.clone()),
                ],
                prompt_contributions: vec![PromptContribution::guidance(
                    "### Recall Agent Worker\nSearch session history first, then expand only the most relevant items. Use filesystem reads only when session history points to a relevant file. Finish by calling `submit_memory_recall` exactly once.",
                )],
            },
        };
        let handle = match context.host.create_session(request).await {
            Ok(handle) => handle,
            Err(err) => {
                return ToolResult::err_fmt(format_args!(
                    "Failed to create recall-agent session: {err}"
                ));
            }
        };
        let turn = context
            .host
            .start_turn(
                &handle.session_id,
                TurnInput {
                    items: vec![InputItem::Text {
                        text: recall_prompt(&queries),
                    }],
                    image_blobs: HashMap::new(),
                    mode: None,
                },
            )
            .await;
        let _ = context.host.close_session(&handle.session_id).await;
        let turn = match turn {
            Ok(turn) => turn,
            Err(err) => {
                return ToolResult::err_fmt(format_args!("Recall-agent session failed: {err}"));
            }
        };
        if let Some(submission) = submit_tool.take() {
            return ToolResult::ok(submission);
        }
        if turn.assistant_output.safe_text.trim().is_empty() {
            return ToolResult::ok(json!({"summary": "", "items": []}));
        }
        ToolResult::err_fmt("recall-agent session did not call submit_memory_recall")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AgentStateEnvelope, MessageRole, Part, Provider, PruneState, ToolCallRecord};

    fn text_message(id: &str, role: MessageRole, content: &str) -> Message {
        Message {
            id: id.to_string(),
            role,
            parts: vec![Part {
                id: format!("{id}.p0"),
                kind: PartKind::Text,
                content: content.to_string(),
                tool_call_id: None,
                tool_name: None,
                prune_state: PruneState::Intact,
            }],
            origin: None,
        }
    }

    fn snapshot() -> AgentStateEnvelope {
        AgentStateEnvelope {
            messages: vec![
                text_message("m1", MessageRole::User, "Investigate auth retry loop"),
                text_message(
                    "m2",
                    MessageRole::Assistant,
                    "We moved refresh handling into middleware.",
                ),
            ],
            tool_calls: vec![ToolCallRecord {
                call_id: Some("call_1".to_string()),
                tool: "exec_command".to_string(),
                args: json!({"cmd":"cargo test"}),
                result: json!("auth.rs: retry loop detected"),
                success: true,
                duration_ms: 10,
            }],
            ..Default::default()
        }
    }

    #[test]
    fn search_snapshot_returns_flat_memory_items() {
        let result = search_snapshot(&snapshot(), &json!({"query":"middleware","limit":5}));
        assert!(result.success);
        let items = result.result.as_array().expect("array");
        assert!(!items.is_empty());
        let first = items[0].as_object().expect("object");
        assert!(first.contains_key("id"));
        assert!(first.contains_key("type"));
        assert!(first.contains_key("content"));
        assert!(first.contains_key("truncated"));
    }

    #[test]
    fn read_snapshot_item_supports_messages_and_tools() {
        let snapshot = snapshot();
        let message = read_snapshot_item(&snapshot, "msg:m2");
        assert!(message.success);
        assert_eq!(
            message.result.get("type").and_then(|value| value.as_str()),
            Some("message_excerpt")
        );

        let tool = read_snapshot_item(&snapshot, "tool:call_1");
        assert!(tool.success);
        assert_eq!(
            tool.result.get("type").and_then(|value| value.as_str()),
            Some("tool_result_excerpt")
        );
    }

    #[test]
    fn validate_recall_submission_wraps_summary_and_items() {
        let submission = validate_recall_submission(
            Some("Short recall summary"),
            &json!([
                {
                    "type": "message_excerpt",
                    "id": "msg:m2",
                    "content": "We moved refresh handling into middleware.",
                    "truncated": false
                }
            ]),
        )
        .expect("valid submission");

        assert_eq!(
            submission.get("summary").and_then(|value| value.as_str()),
            Some("Short recall summary")
        );
        assert_eq!(
            submission["items"][0]
                .get("id")
                .and_then(|value| value.as_str()),
            Some("msg:m2")
        );
    }

    #[test]
    fn recall_child_policy_prefers_explicit_recall_agent_model() {
        let provider = Provider::Codex {
            access_token: "token".into(),
            refresh_token: "refresh".into(),
            expires_at: 0,
            account_id: None,
            options: crate::provider::ProviderOptions::default(),
        };
        let policy = SessionPolicy {
            model: "gpt-5".into(),
            provider,
            recall_agent_model: Some("gpt-5.4-mini".into()),
            ..SessionPolicy::default()
        };

        let child = recall_child_policy(&policy);
        assert_eq!(child.model, "gpt-5.4-mini");
        assert_eq!(child.model_variant.as_deref(), Some("low"));
        assert_eq!(child.recall_agent_model.as_deref(), Some("gpt-5.4-mini"));
        assert!(child.max_turns.is_none());
    }

    #[test]
    fn recall_child_policy_uses_codex_low_tier_default_when_unset() {
        let policy = SessionPolicy {
            model: "gpt-5.4".into(),
            provider: Provider::Codex {
                access_token: "token".into(),
                refresh_token: "refresh".into(),
                expires_at: 0,
                account_id: None,
                options: crate::provider::ProviderOptions::default(),
            },
            ..SessionPolicy::default()
        };

        let child = recall_child_policy(&policy);
        assert_eq!(child.model, "gpt-5.4-mini");
        assert_eq!(child.model_variant.as_deref(), Some("low"));
    }

    #[test]
    fn recall_child_policy_forces_standard_execution_mode() {
        let policy = SessionPolicy {
            execution_mode: crate::ExecutionMode::Repl,
            ..SessionPolicy::default()
        };

        let child = recall_child_policy(&policy);
        assert_eq!(child.execution_mode, crate::ExecutionMode::Standard);
    }
}
