use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use sha2::{Digest, Sha256};

use crate::agent::format_tool_result_content;
#[cfg(feature = "sqlite-store")]
use crate::plugin::history::{HistoryTools, history_prompt_contributions};
use crate::plugin::{
    PluginError, PromptContribution, SessionContextSurface, SessionCreateRequest, SessionManager,
    SessionPluginMode, SessionStartPoint,
};
#[cfg(feature = "sqlite-store")]
use crate::store::Store;
use crate::tools::RecallAgentTools;
use crate::{
    AgentStateEnvelope, ContextStrategy, ExecutionMode, InputItem, Message, MessageOrigin,
    MessageRole, Part, PartKind, PromptUsage, ToolCallRecord, ToolProvider, TurnInput,
    lash_cache_dir,
};

const MIN_RECENT_USER_TURNS: usize = 3;
const TOOL_RESULT_MAX_LINES: usize = 2_000;
const TOOL_RESULT_MAX_BYTES: usize = 50 * 1024;
const PRUNE_MINIMUM_TOKENS: usize = 20_000;
const PRUNE_PROTECT_TOKENS: usize = 40_000;
const PRUNE_RECENT_USER_TURNS: usize = 2;
const COMPACTION_BUFFER_TOKENS: usize = 20_000;
const COMPACTION_PLUGIN_ID: &str = "context_strategy";
const COMPACTION_SUMMARY_TITLE: &str = "Compaction summary:";
const COMPACTION_PROMPT: &str = "Provide a detailed summary of the conversation above so another agent can continue the work without the full history.\n\nUse this template:\n---\n## Goal\n[What is the user trying to accomplish?]\n\n## Instructions\n- [Relevant instructions or constraints]\n\n## Discoveries\n[Important findings, failures, or decisions]\n\n## Accomplished\n[What is done, what is in progress, what remains]\n\n## Relevant files / directories\n[List important files or directories]\n---";
const PRUNE_PROTECTED_TOOLS: &[&str] = &["skill"];

#[derive(Clone)]
pub struct ContextBuildRequest {
    pub session_id: String,
    pub state: AgentStateEnvelope,
    pub messages: Vec<Message>,
    pub prompt_usage: Option<PromptUsage>,
    pub max_context_tokens: Option<usize>,
    pub host: Arc<dyn SessionManager>,
    #[cfg(feature = "sqlite-store")]
    pub store: Option<Arc<Store>>,
}

#[derive(Clone)]
pub struct PreparedContext {
    pub messages: Vec<Message>,
    pub prompt_contributions: Vec<PromptContribution>,
    pub tool_providers: Vec<Arc<dyn ToolProvider>>,
    pub include_base_tools: bool,
}

impl Default for PreparedContext {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            prompt_contributions: Vec::new(),
            tool_providers: Vec::new(),
            include_base_tools: true,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ContextBuildError {
    #[error("{0}")]
    Session(String),
}

impl From<PluginError> for ContextBuildError {
    fn from(value: PluginError) -> Self {
        Self::Session(value.to_string())
    }
}

#[async_trait]
trait ContextBuilder: Send + Sync {
    async fn build(
        &self,
        request: ContextBuildRequest,
    ) -> Result<PreparedContext, ContextBuildError>;
}

pub async fn build_context(
    request: ContextBuildRequest,
) -> Result<PreparedContext, ContextBuildError> {
    match request.state.policy.context_strategy {
        ContextStrategy::RollingContext => RollingContextBuilder.build(request).await,
        ContextStrategy::RecallAgent { .. } => RecallAgentBuilder.build(request).await,
    }
}

struct RollingContextBuilder;
struct RecallAgentBuilder;

#[async_trait]
impl ContextBuilder for RollingContextBuilder {
    async fn build(
        &self,
        request: ContextBuildRequest,
    ) -> Result<PreparedContext, ContextBuildError> {
        let ContextBuildRequest {
            session_id,
            state,
            mut messages,
            prompt_usage,
            max_context_tokens,
            host,
            #[cfg(feature = "sqlite-store")]
            store,
        } = request;

        let tool_calls = tool_record_map(&state.tool_calls);
        hydrate_tool_result_parts(&session_id, &mut messages, &tool_calls);
        prune_old_tool_results(&mut messages, &tool_calls);

        let mut prepared = PreparedContext {
            messages: messages.clone(),
            ..Default::default()
        };
        #[cfg(feature = "sqlite-store")]
        if let Some(store) = store {
            let tools = Arc::new(HistoryTools::new(store)) as Arc<dyn ToolProvider>;
            prepared
                .prompt_contributions
                .extend(history_prompt_contributions(&crate::PromptContext {
                    tool_names: vec!["search_history".to_string()],
                    ..Default::default()
                }));
            prepared.tool_providers.push(tools);
        }

        if !compaction_needed(prompt_usage, max_context_tokens) {
            prepared.messages = messages;
            return Ok(prepared);
        }

        let prefix_len = leading_system_prefix_len(&messages);
        let Some(last_user_idx) = latest_user_index(&messages) else {
            prepared.messages = messages;
            return Ok(prepared);
        };
        if last_user_idx <= prefix_len {
            prepared.messages = messages;
            return Ok(prepared);
        }

        let prefix_messages = messages[prefix_len..last_user_idx].to_vec();
        let Some(summary) =
            summarize_compaction_prefix(&session_id, &state, prefix_messages, host).await?
        else {
            prepared.messages = messages;
            return Ok(prepared);
        };

        prepared.messages = apply_compaction_summary(&messages, &summary);
        Ok(prepared)
    }
}

#[async_trait]
impl ContextBuilder for RecallAgentBuilder {
    async fn build(
        &self,
        request: ContextBuildRequest,
    ) -> Result<PreparedContext, ContextBuildError> {
        let ContextBuildRequest {
            session_id,
            state,
            mut messages,
            prompt_usage,
            max_context_tokens,
            ..
        } = request;
        let ContextStrategy::RecallAgent { keep_recent_pct } = state.policy.context_strategy else {
            unreachable!();
        };

        let tool_calls = tool_record_map(&state.tool_calls);
        hydrate_tool_result_parts(&session_id, &mut messages, &tool_calls);
        let mut recall_available = false;

        if let (Some(prompt_usage), Some(max_context)) = (prompt_usage, max_context_tokens) {
            let target_budget = max_context * usize::from(keep_recent_pct) / 100;
            if prompt_usage.context_budget_tokens > target_budget {
                let prefix_len = leading_system_prefix_len(&messages);
                let total_chars: usize = messages.iter().map(Message::char_count).sum();
                let target_chars = total_chars.saturating_mul(target_budget)
                    / prompt_usage.context_budget_tokens.max(1);

                let mut keep_from = messages.len();
                let mut tail_chars = 0usize;
                for i in (prefix_len..messages.len()).rev() {
                    let cost = messages[i].char_count();
                    if tail_chars + cost > target_chars {
                        break;
                    }
                    tail_chars += cost;
                    keep_from = i;
                }

                keep_from = keep_from.min(keep_from_for_recent_turns(&messages, prefix_len));
                if keep_from > prefix_len {
                    messages.drain(prefix_len..keep_from);
                    recall_available = true;
                }
            }
        }

        Ok(PreparedContext {
            messages,
            prompt_contributions: if recall_available {
                RecallAgentTools::prompt_contributions()
            } else {
                Vec::new()
            },
            tool_providers: if recall_available {
                vec![Arc::new(RecallAgentTools) as Arc<dyn ToolProvider>]
            } else {
                Vec::new()
            },
            include_base_tools: true,
        })
    }
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

fn approx_token_count(text: &str) -> usize {
    text.len().div_ceil(4)
}

fn strip_ansi_escapes(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut idx = 0usize;
    while idx < bytes.len() {
        if bytes[idx] == 0x1b {
            idx += 1;
            if idx < bytes.len() && bytes[idx] == b'[' {
                idx += 1;
                while idx < bytes.len() {
                    let byte = bytes[idx];
                    idx += 1;
                    if (byte as char).is_ascii_alphabetic() {
                        break;
                    }
                }
                continue;
            }
            continue;
        }
        if let Some(ch) = text[idx..].chars().next() {
            out.push(ch);
            idx += ch.len_utf8();
        } else {
            break;
        }
    }
    out
}

#[derive(Clone, Copy)]
enum TruncationDirection {
    Head,
    Tail,
}

fn tool_result_truncation_direction(tool_name: &str) -> TruncationDirection {
    match tool_name {
        "exec_command" | "write_stdin" | "batch" => TruncationDirection::Tail,
        _ => TruncationDirection::Head,
    }
}

fn truncate_tool_result_preview(
    text: &str,
    direction: TruncationDirection,
    output_path: Option<&Path>,
) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let total_bytes = text.len();
    if lines.len() <= TOOL_RESULT_MAX_LINES && total_bytes <= TOOL_RESULT_MAX_BYTES {
        return text.to_string();
    }

    let mut out = Vec::new();
    let mut bytes = 0usize;
    let mut hit_bytes = false;

    match direction {
        TruncationDirection::Head => {
            for (idx, line) in lines.iter().enumerate().take(TOOL_RESULT_MAX_LINES) {
                let size = line.len() + usize::from(idx > 0);
                if bytes + size > TOOL_RESULT_MAX_BYTES {
                    hit_bytes = true;
                    break;
                }
                out.push(*line);
                bytes += size;
            }
        }
        TruncationDirection::Tail => {
            for (idx, line) in lines.iter().rev().take(TOOL_RESULT_MAX_LINES).enumerate() {
                let size = line.len() + usize::from(idx > 0);
                if bytes + size > TOOL_RESULT_MAX_BYTES {
                    hit_bytes = true;
                    break;
                }
                out.push(*line);
                bytes += size;
            }
            out.reverse();
        }
    }

    let preview = out.join("\n");
    let removed = if hit_bytes {
        total_bytes.saturating_sub(preview.len())
    } else {
        lines.len().saturating_sub(out.len())
    };
    let unit = if hit_bytes { "bytes" } else { "lines" };
    let retained_hint = match output_path {
        Some(path) => format!(
            "The tool call succeeded but the output was truncated. Full output saved to: {}\nUse `read_file` with `offset`/`limit` or `grep` to inspect specific sections instead of reading the whole file at once.",
            path.display()
        ),
        None => "Full tool result retained in session state.".to_string(),
    };
    match direction {
        TruncationDirection::Head => {
            format!("{preview}\n\n...{removed} {unit} truncated...\n\n{retained_hint}")
        }
        TruncationDirection::Tail => {
            format!("...{removed} {unit} truncated...\n\n{retained_hint}\n\n{preview}")
        }
    }
}

fn tool_result_needs_truncation(text: &str) -> bool {
    text.lines().count() > TOOL_RESULT_MAX_LINES || text.len() > TOOL_RESULT_MAX_BYTES
}

fn normalize_tool_result_content(record: &ToolCallRecord) -> String {
    let rendered = format_tool_result_content(record.success, &record.result);
    match record.tool.as_str() {
        "exec_command" | "write_stdin" | "batch" => strip_ansi_escapes(&rendered),
        _ => rendered,
    }
}

fn tool_output_file_name(record: &ToolCallRecord) -> String {
    let mut hasher = Sha256::new();
    if let Some(call_id) = &record.call_id {
        hasher.update(call_id.as_bytes());
    } else {
        hasher.update(record.tool.as_bytes());
        hasher.update(record.args.to_string().as_bytes());
        hasher.update(record.result.to_string().as_bytes());
    }
    let digest = format!("{:x}", hasher.finalize());
    let stem = record
        .call_id
        .as_deref()
        .unwrap_or(&record.tool)
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-') {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    format!("{stem}-{}.txt", &digest[..12])
}

fn spill_tool_output_to_dir(
    base_dir: &Path,
    session_id: &str,
    record: &ToolCallRecord,
    full_output: &str,
) -> Option<PathBuf> {
    let dir = base_dir.join("tool-output").join(session_id);
    if fs::create_dir_all(&dir).is_err() {
        return None;
    }
    let path = dir.join(tool_output_file_name(record));
    let needs_write = match fs::read_to_string(&path) {
        Ok(existing) => existing != full_output,
        Err(_) => true,
    };
    if needs_write && fs::write(&path, full_output).is_err() {
        return None;
    }
    Some(path)
}

fn render_tool_result_preview(record: &ToolCallRecord) -> String {
    let normalized = normalize_tool_result_content(record);
    truncate_tool_result_preview(
        &normalized,
        tool_result_truncation_direction(&record.tool),
        None,
    )
}

fn render_tool_result_preview_for_session(session_id: &str, record: &ToolCallRecord) -> String {
    let normalized = normalize_tool_result_content(record);
    let output_path = if tool_result_needs_truncation(&normalized) {
        spill_tool_output_to_dir(&lash_cache_dir(), session_id, record, &normalized)
    } else {
        None
    };
    truncate_tool_result_preview(
        &normalized,
        tool_result_truncation_direction(&record.tool),
        output_path.as_deref(),
    )
}

fn tool_record_map(tool_calls: &[ToolCallRecord]) -> HashMap<String, ToolCallRecord> {
    tool_calls
        .iter()
        .filter_map(|record| {
            record
                .call_id
                .as_ref()
                .map(|call_id| (call_id.clone(), record.clone()))
        })
        .collect()
}

fn hydrate_tool_result_parts(
    session_id: &str,
    messages: &mut [Message],
    tool_calls: &HashMap<String, ToolCallRecord>,
) {
    for message in messages {
        for part in &mut message.parts {
            if !matches!(part.kind, PartKind::ToolResult) {
                continue;
            }
            if matches!(part.prune_state, crate::PruneState::Cleared) {
                continue;
            }
            if !matches!(part.prune_state, crate::PruneState::Intact) {
                continue;
            }
            let Some(call_id) = part.tool_call_id.as_deref() else {
                continue;
            };
            let Some(record) = tool_calls.get(call_id) else {
                continue;
            };
            part.content = render_tool_result_preview_for_session(session_id, record);
        }
    }
}

fn prune_old_tool_results(
    messages: &mut [Message],
    tool_calls: &HashMap<String, ToolCallRecord>,
) -> bool {
    let mut total = 0usize;
    let mut pruned = 0usize;
    let mut to_prune = Vec::new();
    let mut recent_user_turns = 0usize;

    'scan: for msg_idx in (0..messages.len()).rev() {
        if is_compaction_summary_message(&messages[msg_idx]) {
            break;
        }
        if messages[msg_idx].role == MessageRole::User {
            recent_user_turns += 1;
        }
        if recent_user_turns < PRUNE_RECENT_USER_TURNS {
            continue;
        }
        for part_idx in (0..messages[msg_idx].parts.len()).rev() {
            let part = &messages[msg_idx].parts[part_idx];
            if !matches!(part.kind, PartKind::ToolResult) {
                continue;
            }
            if matches!(part.prune_state, crate::PruneState::Cleared) {
                break 'scan;
            }
            if !matches!(part.prune_state, crate::PruneState::Intact) {
                continue;
            }
            let Some(call_id) = part.tool_call_id.as_deref() else {
                continue;
            };
            let Some(record) = tool_calls.get(call_id) else {
                continue;
            };
            if PRUNE_PROTECTED_TOOLS.contains(&record.tool.as_str()) {
                continue;
            }
            let estimate = approx_token_count(&render_tool_result_preview(record));
            total += estimate;
            if total > PRUNE_PROTECT_TOKENS {
                pruned += estimate;
                to_prune.push((msg_idx, part_idx));
            }
        }
    }

    if pruned <= PRUNE_MINIMUM_TOKENS {
        return false;
    }

    for (msg_idx, part_idx) in to_prune {
        messages[msg_idx].parts[part_idx].prune_state = crate::PruneState::Cleared;
        messages[msg_idx].parts[part_idx].content.clear();
    }
    true
}

fn is_compaction_summary_message(message: &Message) -> bool {
    matches!(
        message.origin,
        Some(MessageOrigin::Plugin { ref plugin_id }) if plugin_id == COMPACTION_PLUGIN_ID
    )
}

fn latest_user_index(messages: &[Message]) -> Option<usize> {
    messages
        .iter()
        .rposition(|message| matches!(message.role, MessageRole::User))
}

fn compaction_needed(prompt_usage: Option<PromptUsage>, max_context_tokens: Option<usize>) -> bool {
    let Some(usage) = prompt_usage else {
        return false;
    };
    let Some(max_context) = max_context_tokens else {
        return false;
    };
    let usable = max_context.saturating_sub(COMPACTION_BUFFER_TOKENS.min(max_context));
    usage.context_budget_tokens >= usable
}

fn referenced_tool_call_ids(messages: &[Message]) -> HashSet<String> {
    messages
        .iter()
        .flat_map(|message| message.parts.iter())
        .filter_map(|part| part.tool_call_id.clone())
        .collect()
}

async fn summarize_compaction_prefix(
    session_id: &str,
    state: &AgentStateEnvelope,
    prefix_messages: Vec<Message>,
    host: Arc<dyn SessionManager>,
) -> Result<Option<String>, ContextBuildError> {
    if prefix_messages.is_empty() {
        return Ok(None);
    }

    let mut snapshot = state.clone();
    snapshot.messages = prefix_messages;
    snapshot.policy.execution_mode = ExecutionMode::Standard;
    snapshot.policy.context_strategy = ContextStrategy::RollingContext;
    snapshot.policy.max_turns = Some(1);
    snapshot.plugin_snapshot = None;
    snapshot.repl_snapshot = None;
    let referenced = referenced_tool_call_ids(&snapshot.messages);
    snapshot.tool_calls.retain(|record| {
        record
            .call_id
            .as_ref()
            .is_some_and(|call_id| referenced.contains(call_id))
    });

    let compaction_session_id = format!("{session_id}-compaction");
    let mut policy = snapshot.policy.clone();
    policy.execution_mode = ExecutionMode::Standard;
    policy.context_strategy = ContextStrategy::RollingContext;
    policy.max_turns = Some(1);
    let handle = host
        .create_session(SessionCreateRequest {
            agent_id: Some(compaction_session_id),
            start: SessionStartPoint::Snapshot {
                snapshot: Box::new(snapshot),
            },
            policy: Some(policy),
            plugin_mode: SessionPluginMode::Fresh,
            initial_messages: Vec::new(),
            context_surface: SessionContextSurface {
                include_base_tools: false,
                tool_providers: Vec::new(),
                prompt_contributions: Vec::new(),
            },
        })
        .await?;

    let turn = host
        .start_turn(
            &handle.session_id,
            TurnInput {
                items: vec![InputItem::Text {
                    text: COMPACTION_PROMPT.to_string(),
                }],
                image_blobs: HashMap::new(),
                mode: None,
            },
        )
        .await;
    let _ = host.close_session(&handle.session_id).await;
    let turn = turn?;
    let summary = turn.assistant_output.safe_text.trim().to_string();
    if summary.is_empty() {
        return Ok(None);
    }
    Ok(Some(summary))
}

fn apply_compaction_summary(messages: &[Message], summary: &str) -> Vec<Message> {
    let prefix_len = leading_system_prefix_len(messages);
    let Some(last_user_idx) = latest_user_index(messages) else {
        return messages.to_vec();
    };
    if last_user_idx <= prefix_len {
        return messages.to_vec();
    }

    let mut out = Vec::new();
    out.extend_from_slice(&messages[..prefix_len]);
    out.push(Message {
        id: "compaction-summary".to_string(),
        role: MessageRole::Assistant,
        parts: vec![Part {
            id: "compaction-summary.p0".to_string(),
            kind: PartKind::Prose,
            content: format!("{COMPACTION_SUMMARY_TITLE}\n{summary}"),
            tool_call_id: None,
            tool_name: None,
            prune_state: crate::PruneState::Intact,
        }],
        origin: Some(MessageOrigin::Plugin {
            plugin_id: COMPACTION_PLUGIN_ID.to_string(),
        }),
    });
    out.extend_from_slice(&messages[last_user_idx..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::{SessionHandle, SessionTurnHandle};
    use crate::{
        AssistantOutput, CodeOutputRecord, DoneReason, ExecutionSummary, OutputState, TurnStatus,
    };
    use serde_json::json;
    use tempfile::tempdir;
    use tokio::sync::Mutex;

    use crate::SessionPolicy;

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
                prune_state: crate::PruneState::Intact,
            }],
            origin: None,
        }
    }

    #[test]
    fn spilled_tool_output_is_written_and_referenced() {
        let temp = tempdir().expect("tempdir");
        let record = ToolCallRecord {
            call_id: Some("call-123".to_string()),
            tool: "exec_command".to_string(),
            args: json!({"cmd":"cat giant.log"}),
            result: json!(format!("{}\nend", "line\n".repeat(2_500))),
            success: true,
            duration_ms: 5,
        };

        let normalized = normalize_tool_result_content(&record);
        let path = spill_tool_output_to_dir(temp.path(), "root", &record, &normalized)
            .expect("spill path");
        let preview = truncate_tool_result_preview(
            &normalized,
            tool_result_truncation_direction(&record.tool),
            Some(&path),
        );

        assert!(path.exists());
        assert_eq!(
            std::fs::read_to_string(&path).expect("written output"),
            normalized
        );
        assert!(preview.contains(path.to_string_lossy().as_ref()));
        assert!(preview.contains("Full output saved to:"));
    }

    fn empty_turn(session_id: &str, summary: &str) -> crate::AssembledTurn {
        crate::AssembledTurn {
            state: AgentStateEnvelope {
                agent_id: session_id.to_string(),
                policy: SessionPolicy {
                    execution_mode: ExecutionMode::Standard,
                    context_strategy: ContextStrategy::RollingContext,
                    ..Default::default()
                },
                ..Default::default()
            },
            status: TurnStatus::Completed,
            assistant_output: AssistantOutput {
                safe_text: summary.to_string(),
                raw_text: summary.to_string(),
                state: OutputState::Usable,
            },
            done_reason: DoneReason::ModelStop,
            execution: ExecutionSummary {
                mode: ExecutionMode::Standard,
                had_tool_calls: false,
                had_code_execution: false,
            },
            token_usage: crate::TokenUsage::default(),
            tool_calls: Vec::new(),
            code_outputs: Vec::<CodeOutputRecord>::new(),
            errors: Vec::new(),
        }
    }

    #[derive(Default)]
    struct MockSessionManager {
        created: Mutex<Vec<SessionCreateRequest>>,
    }

    #[async_trait::async_trait]
    impl SessionManager for MockSessionManager {
        async fn snapshot_current(&self) -> Result<crate::plugin::SessionSnapshot, PluginError> {
            Ok(AgentStateEnvelope::default())
        }

        async fn snapshot_session(
            &self,
            _session_id: &str,
        ) -> Result<crate::plugin::SessionSnapshot, PluginError> {
            Ok(AgentStateEnvelope::default())
        }

        async fn tool_catalog(
            &self,
            _session_id: &str,
        ) -> Result<Vec<serde_json::Value>, PluginError> {
            Ok(vec![
                json!({"name":"exec_command"}),
                json!({"name":"read_file"}),
            ])
        }

        async fn create_session(
            &self,
            request: SessionCreateRequest,
        ) -> Result<SessionHandle, PluginError> {
            self.created.lock().await.push(request.clone());
            Ok(SessionHandle {
                session_id: request.agent_id.unwrap_or_else(|| "child".to_string()),
                policy: SessionPolicy {
                    model: "mock-model".to_string(),
                    execution_mode: ExecutionMode::Standard,
                    context_strategy: ContextStrategy::RollingContext,
                    ..Default::default()
                },
            })
        }

        async fn close_session(&self, _session_id: &str) -> Result<(), PluginError> {
            Ok(())
        }

        async fn start_turn_stream(
            &self,
            session_id: &str,
            _input: TurnInput,
        ) -> Result<SessionTurnHandle, PluginError> {
            let (_tx, rx) = tokio::sync::mpsc::channel(1);
            Ok(SessionTurnHandle {
                turn_id: format!("{session_id}-turn"),
                session_id: session_id.to_string(),
                policy: SessionPolicy {
                    model: "mock-model".to_string(),
                    execution_mode: ExecutionMode::Standard,
                    context_strategy: ContextStrategy::RollingContext,
                    ..Default::default()
                },
                events: rx,
            })
        }

        async fn await_turn(&self, _turn_id: &str) -> Result<crate::AssembledTurn, PluginError> {
            Ok(empty_turn("root", "Compacted work summary"))
        }

        async fn cancel_turn(&self, _turn_id: &str) -> Result<(), PluginError> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn rolling_context_builder_clears_old_tool_outputs() {
        let tool_calls = (0..12)
            .map(|idx| ToolCallRecord {
                call_id: Some(format!("call-{idx}")),
                tool: "exec_command".to_string(),
                args: json!({"cmd": format!("echo {idx}")}),
                result: json!(format!("{}\n{}", "line".repeat(12_000), idx)),
                success: true,
                duration_ms: 1,
            })
            .collect::<Vec<_>>();
        let mut messages = vec![text_message("u0", MessageRole::User, "older")];
        messages.push(Message {
            id: "a1".to_string(),
            role: MessageRole::Assistant,
            parts: tool_calls
                .iter()
                .enumerate()
                .map(|(idx, record)| Part {
                    id: format!("a1.p{idx}"),
                    kind: PartKind::ToolResult,
                    content: String::new(),
                    tool_call_id: record.call_id.clone(),
                    tool_name: Some(record.tool.clone()),
                    prune_state: crate::PruneState::Intact,
                })
                .collect(),
            origin: None,
        });
        messages.push(text_message("u2", MessageRole::User, "recent"));
        messages.push(text_message("u3", MessageRole::User, "latest"));

        let built = build_context(ContextBuildRequest {
            session_id: "root".to_string(),
            state: AgentStateEnvelope {
                policy: SessionPolicy {
                    context_strategy: ContextStrategy::RollingContext,
                    ..Default::default()
                },
                tool_calls,
                ..Default::default()
            },
            messages,
            prompt_usage: None,
            max_context_tokens: None,
            host: Arc::new(MockSessionManager::default()),
            #[cfg(feature = "sqlite-store")]
            store: None,
        })
        .await
        .expect("context")
        .messages;

        let cleared = built
            .iter()
            .flat_map(|message| message.parts.iter())
            .filter(|part| matches!(part.prune_state, crate::PruneState::Cleared))
            .count();
        assert!(cleared > 0);
    }

    #[tokio::test]
    async fn rolling_context_builder_replaces_prefix_with_summary() {
        let manager = Arc::new(MockSessionManager::default());
        let built = build_context(ContextBuildRequest {
            session_id: "root".to_string(),
            state: AgentStateEnvelope {
                agent_id: "root".to_string(),
                policy: SessionPolicy {
                    execution_mode: ExecutionMode::Standard,
                    context_strategy: ContextStrategy::RollingContext,
                    ..Default::default()
                },
                ..Default::default()
            },
            messages: vec![
                text_message("u1", MessageRole::User, "old work"),
                text_message("a1", MessageRole::Assistant, "assistant old"),
                text_message("u2", MessageRole::User, "latest request"),
            ],
            prompt_usage: Some(PromptUsage {
                prompt_context_tokens: 90_000,
                input_tokens: 90_000,
                cached_input_tokens: 0,
                context_budget_tokens: 90_000,
            }),
            max_context_tokens: Some(100_000),
            host: manager.clone(),
            #[cfg(feature = "sqlite-store")]
            store: None,
        })
        .await
        .expect("context")
        .messages;

        assert!(built.iter().any(|message| {
            message
                .parts
                .iter()
                .any(|part| part.content.contains("Compacted work summary"))
        }));
        assert!(built.iter().any(|message| {
            message
                .parts
                .iter()
                .any(|part| part.content.contains("latest request"))
        }));
        assert!(!built.iter().any(|message| {
            message
                .parts
                .iter()
                .any(|part| part.content.contains("old work"))
        }));

        let created = manager.created.lock().await;
        assert_eq!(created.len(), 1);
        assert_eq!(
            created[0]
                .policy
                .as_ref()
                .map(|policy| policy.context_strategy),
            Some(ContextStrategy::RollingContext)
        );
    }

    #[tokio::test]
    async fn recall_agent_builder_only_injects_tools_after_trimming() {
        let prepared = build_context(ContextBuildRequest {
            session_id: "root".to_string(),
            state: AgentStateEnvelope {
                policy: SessionPolicy {
                    context_strategy: ContextStrategy::recall_agent_default(),
                    ..Default::default()
                },
                ..Default::default()
            },
            messages: vec![
                text_message("u1", MessageRole::User, "short request"),
                text_message("a1", MessageRole::Assistant, "short reply"),
            ],
            prompt_usage: Some(PromptUsage {
                prompt_context_tokens: 5,
                input_tokens: 5,
                cached_input_tokens: 0,
                context_budget_tokens: 5,
            }),
            max_context_tokens: Some(1_000),
            host: Arc::new(MockSessionManager::default()),
            #[cfg(feature = "sqlite-store")]
            store: None,
        })
        .await
        .expect("context");

        assert!(prepared.prompt_contributions.is_empty());
        assert!(prepared.tool_providers.is_empty());
    }
}
