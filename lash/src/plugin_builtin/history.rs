use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use serde_json::json;

use crate::search::{SearchDoc, SearchMode, limit_from_args, rank_docs, truncate_preview};
use crate::store::{HistoryTurnRecord, Store};
use crate::tools::run_blocking;
use crate::{
    AssembledTurn, Message, MessageRole, PartKind, PromptContribution, ToolDefinition, ToolParam,
    ToolProvider, ToolResult,
};

fn agent_id(args: &serde_json::Value) -> String {
    args.get("__agent_id__")
        .and_then(|v| v.as_str())
        .filter(|value| !value.is_empty())
        .unwrap_or("root")
        .to_string()
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

pub(crate) fn final_history_record(turn: &AssembledTurn) -> HistoryTurnRecord {
    let payload = latest_turn_history_payload(turn);
    HistoryTurnRecord {
        index: payload.get("index").and_then(|v| v.as_i64()).unwrap_or(0),
        user_message: payload
            .get("user_message")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string(),
        prose: payload
            .get("prose")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string(),
        code: payload
            .get("code")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string(),
        output: payload
            .get("output")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string(),
        error: payload
            .get("error")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        tool_calls: payload
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default(),
        files_read: Vec::new(),
        files_written: Vec::new(),
    }
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
            description: "Search turn history using hybrid, literal, or regex matching. With no `query`, returns the full turn list in stable turn order.".into(),
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
            enabled: true,
            injected: true,
            input_schema_override: None,
            output_schema_override: None,
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

pub(crate) fn history_prompt_contributions(
    context: &crate::PromptContext,
) -> Vec<PromptContribution> {
    if !context.has_tool("search_history") {
        return Vec::new();
    }

    vec![PromptContribution::guidance(
        "### History\nUse `search_history` only when older context actually matters. With no query, it lists recorded turns in stable turn order; add a focused query when you need retrieval instead of browsing.",
    )]
}

pub(crate) fn history_summary(store: &Store, session_id: &str, limit: usize) -> ToolResult {
    let turns = store.history_export(session_id);
    let latest_user_message = turns.iter().rev().find_map(|turn| {
        (!turn.user_message.trim().is_empty()).then_some(turn.user_message.clone())
    });
    let recent_turns = turns
        .iter()
        .rev()
        .take(limit.clamp(1, 20))
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
}
