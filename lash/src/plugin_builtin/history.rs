use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use serde_json::json;

use crate::plugin::{
    MessageMutatorHook, PluginError, PluginFactory, PluginRegistrar, PluginSessionContext,
    PluginSnapshotMeta, SnapshotReader, SnapshotWriter, ToolSurfaceContribution,
    ToolSurfaceOverride,
};
use crate::search::{SearchDoc, SearchMode, limit_from_args, rank_docs, truncate_preview};
use crate::store::{HistoryTurnRecord, Store};
use crate::tools::run_blocking;
use crate::{
    AssembledTurn, ContextFoldingConfig, Message, MessageRole, PartKind, PromptContribution,
    SessionPlugin, ToolDefinition, ToolParam, ToolProvider, ToolResult,
};

const MIN_RECENT_USER_TURNS: usize = 3;

#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
struct HistoryPluginState {
    has_archived_history: bool,
}

fn agent_id(args: &serde_json::Value) -> String {
    args.get("__agent_id__")
        .and_then(|v| v.as_str())
        .filter(|value| !value.is_empty())
        .unwrap_or("root")
        .to_string()
}

fn leading_system_prefix_len(msgs: &[Message]) -> usize {
    msgs.iter()
        .take_while(|msg| msg.role == MessageRole::System)
        .count()
}

fn keep_from_for_recent_turns(msgs: &[Message], prefix_len: usize) -> usize {
    let mut user_turns = 0usize;
    for i in (prefix_len..msgs.len()).rev() {
        if msgs[i].role == MessageRole::User {
            user_turns += 1;
            if user_turns >= MIN_RECENT_USER_TURNS {
                return i;
            }
        }
    }
    prefix_len
}

fn compact_history_messages(
    msgs: &mut Vec<Message>,
    last_context_budget_tokens: usize,
    max_context: usize,
    policy: ContextFoldingConfig,
) -> bool {
    if last_context_budget_tokens == 0 || msgs.is_empty() {
        return false;
    }

    let hard_budget = max_context * usize::from(policy.hard_limit_pct) / 100;
    if last_context_budget_tokens < hard_budget {
        return false;
    }

    let soft_budget = max_context * usize::from(policy.soft_limit_pct) / 100;
    let prefix_len = leading_system_prefix_len(msgs);
    let total_chars: usize = msgs.iter().map(Message::char_count).sum();
    let target_chars = total_chars.saturating_mul(soft_budget) / last_context_budget_tokens.max(1);

    let mut keep_from = msgs.len();
    let mut tail_chars = 0usize;
    for i in (prefix_len..msgs.len()).rev() {
        let cost = msgs[i].char_count();
        if tail_chars + cost > target_chars {
            break;
        }
        tail_chars += cost;
        keep_from = i;
    }

    keep_from = keep_from.min(keep_from_for_recent_turns(msgs, prefix_len));
    if keep_from <= prefix_len {
        return false;
    }

    msgs.drain(prefix_len..keep_from);
    true
}

fn history_message_text(msg: &Message) -> String {
    msg.parts
        .iter()
        .filter_map(|part| match part.kind {
            PartKind::Text | PartKind::Prose | PartKind::Code => Some(part.content.as_str()),
            _ => None,
        })
        .filter(|text| !text.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn latest_turn_history_payload(turn: &AssembledTurn) -> serde_json::Value {
    let messages = &turn.state.messages;
    let turn_index = messages
        .iter()
        .filter(|msg| matches!(msg.role, MessageRole::User))
        .count() as i64;
    let last_user_idx = messages
        .iter()
        .rposition(|msg| matches!(msg.role, MessageRole::User));

    let user_message = last_user_idx
        .and_then(|idx| messages.get(idx))
        .map(history_message_text)
        .unwrap_or_default();

    let mut prose_parts = Vec::new();
    let mut code_parts = Vec::new();
    if let Some(idx) = last_user_idx {
        for msg in messages.iter().skip(idx + 1) {
            if !matches!(msg.role, MessageRole::Assistant) {
                continue;
            }
            for part in &msg.parts {
                match part.kind {
                    PartKind::Text | PartKind::Prose => {
                        if !part.content.trim().is_empty() {
                            prose_parts.push(part.content.clone());
                        }
                    }
                    PartKind::Code => {
                        if !part.content.trim().is_empty() {
                            code_parts.push(part.content.clone());
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    let prose = if prose_parts.is_empty() {
        turn.assistant_output.raw_text.clone()
    } else {
        prose_parts.join("\n\n")
    };
    let code = code_parts.join("\n\n");
    let output = turn
        .code_outputs
        .iter()
        .map(|record| match (&record.output, &record.error) {
            (output, Some(error)) if !output.is_empty() && !error.is_empty() => {
                format!("{output}\n{error}")
            }
            (output, _) if !output.is_empty() => output.clone(),
            (_, Some(error)) => error.clone(),
            _ => String::new(),
        })
        .filter(|chunk| !chunk.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    let error = turn.errors.first().map(|issue| issue.message.clone());

    serde_json::json!({
        "index": turn_index,
        "user_message": user_message,
        "prose": prose,
        "code": code,
        "output": output,
        "error": error,
        "tool_calls": turn.tool_calls,
    })
}

#[derive(Clone)]
pub(crate) struct HistoryTools {
    store: Arc<Store>,
}

impl HistoryTools {
    pub(crate) fn new(store: Arc<Store>) -> Self {
        Self { store }
    }

    fn history_item(turn: &HistoryTurnRecord) -> serde_json::Value {
        let preview_source = if !turn.user_message.trim().is_empty() {
            &turn.user_message
        } else if !turn.prose.trim().is_empty() {
            &turn.prose
        } else {
            &turn.output
        };
        json!({
            "turn": turn.index,
            "preview": truncate_preview(preview_source, 220),
            "tool_calls": turn.tool_calls,
            "files_read": turn.files_read,
            "files_written": turn.files_written,
        })
    }

    pub(crate) fn search_history(&self, args: &serde_json::Value) -> ToolResult {
        let agent_id = agent_id(args);
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let browse_all = query.trim().is_empty();
        let mode = SearchMode::parse(args.get("mode").and_then(|v| v.as_str()));
        let regex = args.get("regex").and_then(|v| v.as_str());
        let limit = if browse_all && args.get("limit").is_none() {
            usize::MAX
        } else {
            limit_from_args(args)
        };
        let since_turn = args.get("since_turn").and_then(|v| v.as_i64());

        let selected_fields: HashSet<String> = args
            .get("fields")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(str::to_string)
                    .collect::<HashSet<_>>()
            })
            .unwrap_or_else(|| {
                [
                    "user_message".to_string(),
                    "code".to_string(),
                    "prose".to_string(),
                    "output".to_string(),
                    "tool_calls".to_string(),
                ]
                .into_iter()
                .collect()
            });

        let turns: Vec<HistoryTurnRecord> = self
            .store
            .history_export(&agent_id)
            .into_iter()
            .filter(|turn| since_turn.is_none_or(|min_turn| turn.index >= min_turn))
            .collect();

        if browse_all {
            return ToolResult::ok(json!(
                turns
                    .iter()
                    .take(limit)
                    .map(Self::history_item)
                    .collect::<Vec<_>>()
            ));
        }

        let docs: Vec<SearchDoc> = turns
            .iter()
            .map(|turn| {
                let mut fields = HashMap::new();
                if selected_fields.contains("user_message") {
                    fields.insert("user_message", turn.user_message.clone());
                }
                if selected_fields.contains("code") {
                    fields.insert("code", turn.code.clone());
                }
                if selected_fields.contains("prose") {
                    fields.insert("prose", turn.prose.clone());
                }
                if selected_fields.contains("output") {
                    fields.insert("output", turn.output.clone());
                }
                if selected_fields.contains("tool_calls") {
                    let tool_calls = serde_json::to_string(&turn.tool_calls).unwrap_or_default();
                    fields.insert("tool_calls", tool_calls);
                }
                SearchDoc { fields }
            })
            .collect();

        let ranked = rank_docs(
            &docs,
            &query,
            mode,
            regex,
            &[
                ("user_message", 3.0),
                ("prose", 2.0),
                ("code", 2.0),
                ("output", 1.0),
                ("tool_calls", 1.0),
            ],
        );

        let items: Vec<serde_json::Value> = ranked
            .into_iter()
            .take(limit)
            .map(|(idx, score, hits)| {
                let turn = &turns[idx];
                let mut item = Self::history_item(turn);
                if let Some(obj) = item.as_object_mut() {
                    obj.insert("score".to_string(), json!(score));
                    obj.insert("field_hits".to_string(), json!(hits));
                }
                item
            })
            .collect();
        ToolResult::ok(json!(items))
    }
}

#[async_trait::async_trait]
impl ToolProvider for HistoryTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "search_history".into(),
            description: "Search turn history using hybrid/literal/regex matching. With no `query`, returns the full turn list in stable turn order.".into(),
            params: vec![
                ToolParam::optional("query", "str"),
                ToolParam::optional("mode", "str"),
                ToolParam::optional("regex", "str"),
                ToolParam::optional("limit", "int"),
                ToolParam::optional("fields", "list"),
                ToolParam::optional("since_turn", "int"),
            ],
            returns: "list".into(),
            examples: vec![],
            enabled: false,
            injected: false,
        }]
    }

    async fn execute(&self, name: &str, args: &serde_json::Value) -> ToolResult {
        let tools = self.clone();
        let name = name.to_string();
        let args = args.clone();
        run_blocking(move || match name.as_str() {
            "search_history" => tools.search_history(&args),
            _ => ToolResult::err_fmt(format_args!("Unknown tool: {name}")),
        })
        .await
    }
}

fn history_prompt_contributions(context: &crate::PromptContext) -> Vec<crate::PromptContribution> {
    if !context.has_tool("search_history") {
        return Vec::new();
    }

    vec![crate::PromptContribution::guidance(
        "### History\nUse `search_history` only when older context actually matters. With no query, it lists archived turns in stable turn order; add a focused query when you need retrieval instead of browsing.",
    )]
}

pub struct HistoryPluginFactory {
    store: Arc<Store>,
    tools: Arc<HistoryTools>,
}

impl HistoryPluginFactory {
    pub fn new(store: Arc<Store>) -> Self {
        Self {
            tools: Arc::new(HistoryTools::new(Arc::clone(&store))),
            store,
        }
    }
}

impl PluginFactory for HistoryPluginFactory {
    fn id(&self) -> &'static str {
        "history"
    }

    fn build(&self, ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(HistoryPlugin {
            store: Arc::clone(&self.store),
            tools: Arc::clone(&self.tools),
            agent_id: ctx.agent_id.clone(),
            state: Arc::new(Mutex::new(HistoryPluginState::default())),
        }))
    }
}

struct HistoryPlugin {
    store: Arc<Store>,
    tools: Arc<HistoryTools>,
    agent_id: String,
    state: Arc<Mutex<HistoryPluginState>>,
}

impl SessionPlugin for HistoryPlugin {
    fn id(&self) -> &'static str {
        "history"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        reg.tools()
            .provider(Arc::clone(&self.tools) as Arc<dyn ToolProvider>)?;
        reg.prompt().contribute(Arc::new(move |ctx| {
            Box::pin(async move { Ok(history_prompt_contributions(&ctx.prompt)) })
        }));
        let state = Arc::clone(&self.state);
        reg.messages().mutator(
            MessageMutatorHook::AfterTokenCount,
            Arc::new(move |ctx, mut messages| {
                let state = Arc::clone(&state);
                Box::pin(async move {
                    let Some(prompt_usage) = ctx.prompt_usage else {
                        return Ok(messages);
                    };
                    let Some(max_context) = ctx.max_context_tokens else {
                        return Ok(messages);
                    };
                    let policy = ctx.context_folding.unwrap_or_default();
                    if compact_history_messages(
                        &mut messages,
                        prompt_usage.context_budget_tokens,
                        max_context,
                        policy,
                    ) {
                        state
                            .lock()
                            .map_err(|_| {
                                PluginError::Session("history plugin state poisoned".to_string())
                            })?
                            .has_archived_history = true;
                    }
                    Ok(messages)
                })
            }),
        )?;
        let state = Arc::clone(&self.state);
        reg.surface().contribute(Arc::new(move |ctx| {
            let has_archived_history = state
                .lock()
                .map_err(|_| PluginError::Session("history plugin state poisoned".to_string()))?
                .has_archived_history;
            if !ctx.tools.iter().any(|tool| tool.name == "search_history") {
                return Ok(ToolSurfaceContribution::default());
            }
            Ok(ToolSurfaceContribution {
                overrides: vec![ToolSurfaceOverride {
                    tool_name: "search_history".to_string(),
                    enabled: Some(has_archived_history),
                    injected: Some(has_archived_history),
                }],
                tool_list_notes: Vec::new(),
            })
        }));
        let state = Arc::clone(&self.state);
        reg.prompt().contribute(Arc::new(move |ctx| {
            let state = Arc::clone(&state);
            Box::pin(async move {
                let has_archived_history = state
                    .lock()
                    .map_err(|_| PluginError::Session("history plugin state poisoned".to_string()))?
                    .has_archived_history;
                let mut contributions = history_prompt_contributions(&ctx.prompt);
                if has_archived_history {
                    contributions.push(PromptContribution::guidance(
                        "Older turns were archived outside the active context. Use `search_history(...)` only when older context actually matters.",
                    ));
                }
                Ok(contributions)
            })
        }));

        let store = Arc::clone(&self.store);
        let agent_id = self.agent_id.clone();
        reg.turn().committed(Arc::new(move |turn| {
            let store = Arc::clone(&store);
            let agent_id = agent_id.clone();
            Box::pin(async move {
                store.history_add_turn(&agent_id, &latest_turn_history_payload(&turn));
                Ok(())
            })
        }));

        let plugin_state = Arc::clone(&self.state);
        reg.session()
            .config_mutator(Arc::new(move |ctx, mut state| {
                let plugin_state = Arc::clone(&plugin_state);
                Box::pin(async move {
                    let Some(max_context) = ctx.current.context_window else {
                        return Ok(state);
                    };
                    let Some(prompt_usage) = state.last_prompt_usage.clone() else {
                        return Ok(state);
                    };
                    if ctx.previous.context_window == Some(max_context)
                        && ctx.previous.model == ctx.current.model
                        && ctx.previous.provider_kind == ctx.current.provider_kind
                    {
                        return Ok(state);
                    }
                    if compact_history_messages(
                        &mut state.messages,
                        prompt_usage.context_budget_tokens,
                        max_context as usize,
                        state.context_folding,
                    ) {
                        plugin_state
                            .lock()
                            .map_err(|_| {
                                PluginError::Session("history plugin state poisoned".to_string())
                            })?
                            .has_archived_history = true;
                    }
                    Ok(state)
                })
            }));

        let store = Arc::clone(&self.store);
        reg.external().op(
            crate::ExternalOpDef {
                name: "history.summary".to_string(),
                description: "Summarize the recent turn history for a session.".to_string(),
                kind: crate::ExternalOpKind::Query,
                session_param: crate::SessionParam::Required,
                input_schema: json!({
                    "type":"object",
                    "properties":{"limit":{"type":"integer","minimum":1}},
                    "additionalProperties": false
                }),
                output_schema: json!({
                    "type":"object",
                    "properties":{
                        "session_id":{"type":"string"},
                        "turn_count":{"type":"integer"},
                        "latest_user_message":{"type":["string","null"]},
                        "recent_turns":{"type":"array"}
                    },
                    "required":["session_id","turn_count","recent_turns"],
                    "additionalProperties": false
                }),
            },
            Arc::new(move |ctx, args| {
                let store = Arc::clone(&store);
                Box::pin(async move {
                    let Some(session_id) = ctx.session_id else {
                        return ToolResult::err(json!("history.summary requires session_id"));
                    };
                    let limit = args
                        .get("limit")
                        .and_then(|v| v.as_u64())
                        .and_then(|v| usize::try_from(v).ok())
                        .unwrap_or(5)
                        .clamp(1, 20);
                    let turns = store.history_export(&session_id);
                    let latest_user_message = turns.iter().rev().find_map(|turn| {
                        (!turn.user_message.trim().is_empty()).then_some(turn.user_message.clone())
                    });
                    let recent_turns = turns
                        .iter()
                        .rev()
                        .take(limit)
                        .map(|turn| {
                            let preview_source = if !turn.user_message.trim().is_empty() {
                                &turn.user_message
                            } else if !turn.prose.trim().is_empty() {
                                &turn.prose
                            } else {
                                &turn.output
                            };
                            json!({
                                "turn": turn.index,
                                "preview": truncate_preview(preview_source, 180),
                                "tool_calls": turn.tool_calls.len(),
                            })
                        })
                        .collect::<Vec<_>>();
                    ToolResult::ok(json!({
                        "session_id": session_id,
                        "turn_count": turns.len(),
                        "latest_user_message": latest_user_message,
                        "recent_turns": recent_turns,
                    }))
                })
            }),
        )?;
        Ok(())
    }

    fn snapshot(
        &self,
        _writer: &mut dyn SnapshotWriter,
    ) -> Result<PluginSnapshotMeta, PluginError> {
        let turns = self
            .store
            .history_export(&self.agent_id)
            .into_iter()
            .map(|turn| {
                json!({
                    "index": turn.index,
                    "user_message": turn.user_message,
                    "prose": turn.prose,
                    "code": turn.code,
                    "output": turn.output,
                    "error": turn.error,
                    "tool_calls": turn.tool_calls,
                    "files_read": turn.files_read,
                    "files_written": turn.files_written,
                })
            })
            .collect::<Vec<_>>();
        let state = self
            .state
            .lock()
            .map_err(|_| PluginError::Snapshot("history plugin state poisoned".to_string()))?;
        Ok(PluginSnapshotMeta {
            plugin_id: self.id().to_string(),
            plugin_version: self.version().to_string(),
            state: Some(json!({
                "turns": turns,
                "has_archived_history": state.has_archived_history,
            })),
        })
    }

    fn restore(
        &self,
        meta: &PluginSnapshotMeta,
        _reader: &dyn SnapshotReader,
    ) -> Result<(), PluginError> {
        let turns = meta
            .state
            .as_ref()
            .and_then(|state| state.get("turns"))
            .and_then(|value| value.as_array())
            .cloned()
            .unwrap_or_default();
        self.store.history_load(&self.agent_id, &turns);
        let has_archived_history = meta
            .state
            .as_ref()
            .and_then(|state| state.get("has_archived_history"))
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        self.state
            .lock()
            .map_err(|_| PluginError::Snapshot("history plugin state poisoned".to_string()))?
            .has_archived_history = has_archived_history;
        Ok(())
    }
}
