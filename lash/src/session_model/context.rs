use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use sha2::{Digest, Sha256};

use crate::plugin::{
    DEFAULT_TOOL_RESULT_PROJECTION_LIMIT_BYTES, DEFAULT_TOOL_RESULT_PROJECTION_MAX_LINES,
    PluginError, PromptContribution, SessionContextSurface, SessionCreateRequest, SessionManager,
    SessionPluginMode, SessionStartPoint,
};
use crate::session_model::format_tool_result_content;
use crate::{
    ContextStrategy, ExecutionMode, InputItem, Message, MessageOrigin, MessageRole, Part, PartKind,
    PromptUsage, SessionStateEnvelope, ToolCallRecord, ToolProvider, TurnInput, lash_cache_dir,
};

const TOOL_RESULT_MAX_LINES: usize = DEFAULT_TOOL_RESULT_PROJECTION_MAX_LINES;
const TOOL_RESULT_MAX_BYTES: usize = DEFAULT_TOOL_RESULT_PROJECTION_LIMIT_BYTES;
const PRUNE_MINIMUM_TOKENS: usize = 20_000;
const PRUNE_PROTECT_TOKENS: usize = 40_000;
const PRUNE_RECENT_USER_TURNS: usize = 2;
const COMPACTION_BUFFER_TOKENS: usize = 20_000;
const COMPACTION_KEEP_RECENT_TOKENS: usize = 20_000;
const PRUNE_CONTEXT_THRESHOLD: f64 = 0.6;
const COMPACTION_PLUGIN_ID: &str = "context_strategy";
const COMPACTION_SUMMARY_TITLE: &str = "Compaction summary:";
const COMPACTION_PROMPT: &str = "Provide a detailed summary of the conversation above so a later session can continue the work without the full history.\n\nUse this template:\n---\n## Goal\n[What is the user trying to accomplish?]\n\n## Instructions\n- [Relevant instructions or constraints]\n\n## Discoveries\n[Important findings, failures, or decisions]\n\n## Accomplished\n[What is done, what is in progress, what remains]\n\n## Relevant files / directories\n[List important files or directories]\n---";
const PRUNE_PROTECTED_TOOLS: &[&str] = &["skill"];
const PRUNED_IMAGE_PLACEHOLDER: &str = "[Image omitted from older context]";
const COMPACTED_IMAGE_PLACEHOLDER: &str = "[Image omitted during compaction]";

fn compaction_update_prompt(previous_summary: &str) -> String {
    format!(
        "A previous compaction summary exists (shown below). Update it with information from the conversation above.\n\n\
         Rules:\n\
         - PRESERVE all existing information from the previous summary\n\
         - ADD new progress, decisions, and context from the new messages\n\
         - Move items from in-progress to done where applicable\n\
         - PRESERVE exact file paths, function names, and error messages\n\n\
         Previous summary:\n{previous_summary}\n\n\
         Use this template:\n---\n\
         ## Goal\n[What is the user trying to accomplish?]\n\n\
         ## Instructions\n- [Relevant instructions or constraints]\n\n\
         ## Discoveries\n[Important findings, failures, or decisions]\n\n\
         ## Accomplished\n[What is done, what is in progress, what remains]\n\n\
         ## Relevant files / directories\n[List important files or directories]\n---"
    )
}

#[derive(Clone)]
pub struct ContextBuildRequest {
    pub session_id: String,
    pub state: SessionStateEnvelope,
    pub messages: Vec<Message>,
    pub prompt_usage: Option<PromptUsage>,
    pub max_context_tokens: Option<usize>,
    pub host: Arc<dyn SessionManager>,
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
    debug_assert_eq!(
        request.state.policy.context_strategy,
        ContextStrategy::RollingContext
    );
    RollingContextBuilder.build(request).await
}

struct RollingContextBuilder;

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
        } = request;

        let tool_calls = tool_record_map(&state.tool_calls);
        hydrate_tool_result_parts(&session_id, &mut messages, &tool_calls);

        // Only prune when context usage exceeds threshold — defer otherwise.
        if pruning_needed(prompt_usage.as_ref(), max_context_tokens) {
            prune_old_tool_results(&mut messages, &tool_calls);
            prune_old_images(&mut messages);
        }

        let mut prepared = PreparedContext {
            messages: messages.clone(),
            ..Default::default()
        };

        if !compaction_needed(prompt_usage.as_ref(), max_context_tokens) {
            prepared.messages = messages;
            return Ok(prepared);
        }

        let prefix_len = leading_system_prefix_len(&messages);
        let cut_point = find_compaction_cut_point(&messages, prefix_len);
        if cut_point <= prefix_len {
            prepared.messages = messages;
            return Ok(prepared);
        }

        let prefix_messages = messages[prefix_len..cut_point].to_vec();
        let Some(summary) =
            summarize_compaction_prefix(&session_id, &state, prefix_messages, host).await?
        else {
            prepared.messages = messages;
            return Ok(prepared);
        };

        prepared.messages = apply_compaction_summary(&messages, &summary, cut_point);
        Ok(prepared)
    }
}

fn leading_system_prefix_len(msgs: &[Message]) -> usize {
    msgs.iter()
        .take_while(|msg| msg.role == MessageRole::System)
        .count()
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
        None => "The tool output was truncated. No separate full-output file was written for this result.".to_string(),
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

fn strip_image_attachment(part: &mut Part, placeholder: &str) -> bool {
    if !matches!(part.kind, PartKind::Image) || part.attachment.is_none() {
        return false;
    }
    part.attachment = None;
    part.content = placeholder.to_string();
    true
}

fn prune_old_images(messages: &mut [Message]) -> bool {
    let mut changed = false;
    let mut recent_user_turns = 0usize;

    'scan: for msg_idx in (0..messages.len()).rev() {
        if is_compaction_summary_message(&messages[msg_idx]) {
            break 'scan;
        }
        if messages[msg_idx].role == MessageRole::User {
            recent_user_turns += 1;
        }
        if recent_user_turns < PRUNE_RECENT_USER_TURNS {
            continue;
        }
        for part in &mut messages[msg_idx].parts {
            changed |= strip_image_attachment(part, PRUNED_IMAGE_PLACEHOLDER);
        }
    }

    changed
}

fn strip_all_image_attachments(messages: &mut [Message], placeholder: &str) -> bool {
    let mut changed = false;
    for message in messages {
        for part in &mut message.parts {
            changed |= strip_image_attachment(part, placeholder);
        }
    }
    changed
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

/// Walk backwards from the end keeping ~`COMPACTION_KEEP_RECENT_TOKENS` worth of messages.
/// Returns the index of the first message in the "keep" region — everything before it gets
/// summarized.  The cut always lands on a user-message boundary so we never split a turn.
fn find_compaction_cut_point(messages: &[Message], prefix_len: usize) -> usize {
    // The earliest possible cut is right after the last compaction summary (if any),
    // or right after the system prefix.
    let start = messages[prefix_len..]
        .iter()
        .rposition(is_compaction_summary_message)
        .map(|i| prefix_len + i + 1)
        .unwrap_or(prefix_len);

    let mut accumulated = 0usize;
    for idx in (start..messages.len()).rev() {
        for part in &messages[idx].parts {
            accumulated += approx_token_count(&part.content);
            if part.attachment.is_some() {
                accumulated += 1200; // approximate image token cost
            }
        }
        if accumulated >= COMPACTION_KEEP_RECENT_TOKENS && messages[idx].role == MessageRole::User {
            return idx;
        }
    }
    // Couldn't accumulate enough — fall back to latest user message.
    latest_user_index(messages).unwrap_or(messages.len())
}

/// Only prune old tool results / images when context usage is above a threshold.
fn pruning_needed(prompt_usage: Option<&PromptUsage>, max_context_tokens: Option<usize>) -> bool {
    let Some(usage) = prompt_usage else {
        return false;
    };
    let Some(max_context) = max_context_tokens else {
        return false;
    };
    if max_context == 0 {
        return false;
    }
    (usage.context_budget_tokens as f64 / max_context as f64) >= PRUNE_CONTEXT_THRESHOLD
}

/// Extract the text of a previous compaction summary from a message slice, if present.
fn extract_previous_summary(messages: &[Message]) -> Option<String> {
    messages.iter().rev().find_map(|m| {
        if !is_compaction_summary_message(m) {
            return None;
        }
        m.parts.first().map(|p| {
            p.content
                .strip_prefix(COMPACTION_SUMMARY_TITLE)
                .unwrap_or(&p.content)
                .trim()
                .to_string()
        })
    })
}

/// Run compaction eagerly on current messages when the context window shrinks.
pub async fn compact_messages_if_needed(
    session_id: &str,
    state: &SessionStateEnvelope,
    messages: &[Message],
    prompt_usage: Option<PromptUsage>,
    max_context_tokens: Option<usize>,
    host: Arc<dyn SessionManager>,
) -> Result<Option<Vec<Message>>, ContextBuildError> {
    if !compaction_needed(prompt_usage.as_ref(), max_context_tokens) {
        return Ok(None);
    }
    compact_messages_core(session_id, state, messages, host).await
}

/// Force-compact messages regardless of threshold. Used for overflow recovery.
pub async fn force_compact_messages(
    session_id: &str,
    state: &SessionStateEnvelope,
    messages: &mut Vec<Message>,
    max_context_tokens: Option<usize>,
    host: Arc<dyn SessionManager>,
) -> Result<(), ContextBuildError> {
    strip_all_image_attachments(messages, COMPACTED_IMAGE_PLACEHOLDER);
    let tool_calls = tool_record_map(&state.tool_calls);
    prune_old_tool_results(messages, &tool_calls);
    if let Some(compacted) = compact_messages_core(session_id, state, messages, host).await? {
        *messages = compacted;
    }
    if let Some(max) = max_context_tokens {
        let total: usize = messages
            .iter()
            .flat_map(|m| m.parts.iter())
            .map(|p| approx_token_count(&p.content))
            .sum();
        if total > max.saturating_sub(COMPACTION_BUFFER_TOKENS) {
            for msg_idx in (0..messages.len()).rev() {
                let recent_user_turns = messages[msg_idx + 1..]
                    .iter()
                    .filter(|msg| msg.role == MessageRole::User)
                    .count();
                if recent_user_turns < PRUNE_RECENT_USER_TURNS {
                    continue;
                }
                for part in &mut messages[msg_idx].parts {
                    if matches!(part.kind, PartKind::ToolResult)
                        && matches!(part.prune_state, crate::PruneState::Intact)
                    {
                        part.prune_state = crate::PruneState::Cleared;
                        part.content.clear();
                    }
                }
            }
        }
    }
    Ok(())
}

async fn compact_messages_core(
    session_id: &str,
    state: &SessionStateEnvelope,
    messages: &[Message],
    host: Arc<dyn SessionManager>,
) -> Result<Option<Vec<Message>>, ContextBuildError> {
    let prefix_len = leading_system_prefix_len(messages);
    let cut_point = find_compaction_cut_point(messages, prefix_len);
    if cut_point <= prefix_len {
        return Ok(None);
    }
    let prefix_messages = messages[prefix_len..cut_point].to_vec();
    let Some(summary) =
        summarize_compaction_prefix(session_id, state, prefix_messages, host).await?
    else {
        return Ok(None);
    };
    Ok(Some(apply_compaction_summary(
        messages, &summary, cut_point,
    )))
}

fn compaction_needed(
    prompt_usage: Option<&PromptUsage>,
    max_context_tokens: Option<usize>,
) -> bool {
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
    state: &SessionStateEnvelope,
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
    strip_all_image_attachments(&mut snapshot.messages, COMPACTED_IMAGE_PLACEHOLDER);
    snapshot.plugin_snapshot = None;
    snapshot.repl_snapshot = None;
    snapshot.last_prompt_usage = None;
    let previous_summary = extract_previous_summary(&snapshot.messages);
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
            session_id: Some(compaction_session_id),
            parent_session_id: Some(session_id.to_string()),
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

    // Use update prompt when a previous summary exists; fresh prompt otherwise.
    let prompt_text = match previous_summary {
        Some(prev) => compaction_update_prompt(&prev),
        None => COMPACTION_PROMPT.to_string(),
    };

    let turn = host
        .start_turn(
            &handle.session_id,
            TurnInput {
                items: vec![InputItem::Text { text: prompt_text }],
                image_blobs: HashMap::new(),
                user_input: None,
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

fn apply_compaction_summary(messages: &[Message], summary: &str, cut_point: usize) -> Vec<Message> {
    let prefix_len = leading_system_prefix_len(messages);
    if cut_point <= prefix_len || cut_point > messages.len() {
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
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            prune_state: crate::PruneState::Intact,
        }],
        user_input: None,
        origin: Some(MessageOrigin::Plugin {
            plugin_id: COMPACTION_PLUGIN_ID.to_string(),
        }),
    });
    out.extend_from_slice(&messages[cut_point..]);
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
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                prune_state: crate::PruneState::Intact,
            }],
            user_input: None,
            origin: None,
        }
    }

    fn image_message(id: &str, role: MessageRole, bytes: &[u8]) -> Message {
        Message {
            id: id.to_string(),
            role,
            parts: vec![Part {
                id: format!("{id}.p0"),
                kind: PartKind::Image,
                content: String::new(),
                attachment: Some(crate::session_model::message::PartAttachment {
                    mime: "image/png".to_string(),
                    url: crate::session_model::message::data_url_for_bytes("image/png", bytes),
                    filename: None,
                }),
                tool_call_id: None,
                tool_name: None,
                prune_state: crate::PruneState::Intact,
            }],
            user_input: None,
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
            state: SessionStateEnvelope {
                session_id: session_id.to_string(),
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
            has_plugin_visible_output: false,
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
            Ok(SessionStateEnvelope::default())
        }

        async fn snapshot_session(
            &self,
            _session_id: &str,
        ) -> Result<crate::plugin::SessionSnapshot, PluginError> {
            Ok(SessionStateEnvelope::default())
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
                session_id: request.session_id.unwrap_or_else(|| "child".to_string()),
                parent_session_id: request.parent_session_id,
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
        let tool_calls = (0..18)
            .map(|idx| ToolCallRecord {
                call_id: Some(format!("call-{idx}")),
                tool: "exec_command".to_string(),
                args: json!({"cmd": format!("echo {idx}")}),
                result: json!(format!(
                    "{}\n{}",
                    (0..600)
                        .map(|_| "line".repeat(64))
                        .collect::<Vec<_>>()
                        .join("\n"),
                    idx
                )),
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
                    attachment: None,
                    tool_call_id: record.call_id.clone(),
                    tool_name: Some(record.tool.clone()),
                    prune_state: crate::PruneState::Intact,
                })
                .collect(),
            user_input: None,
            origin: None,
        });
        messages.push(text_message("u2", MessageRole::User, "recent"));
        messages.push(text_message("u3", MessageRole::User, "latest"));

        // Provide usage above PRUNE_CONTEXT_THRESHOLD (60%) to trigger pruning.
        let built = build_context(ContextBuildRequest {
            session_id: "root".to_string(),
            state: SessionStateEnvelope {
                policy: SessionPolicy {
                    context_strategy: ContextStrategy::RollingContext,
                    ..Default::default()
                },
                tool_calls,
                ..Default::default()
            },
            messages,
            prompt_usage: Some(PromptUsage {
                prompt_context_tokens: 130_000,
                input_tokens: 130_000,
                cached_input_tokens: 0,
                context_budget_tokens: 130_000,
            }),
            max_context_tokens: Some(200_000),
            host: Arc::new(MockSessionManager::default()),
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
    async fn rolling_context_builder_strips_old_image_attachments() {
        let messages = vec![
            image_message("u0", MessageRole::User, &[1, 2, 3]),
            text_message("u1", MessageRole::User, "recent"),
            text_message("u2", MessageRole::User, "latest"),
        ];

        // Provide usage above PRUNE_CONTEXT_THRESHOLD (60%) to trigger pruning.
        let built = build_context(ContextBuildRequest {
            session_id: "root".to_string(),
            state: SessionStateEnvelope {
                policy: SessionPolicy {
                    context_strategy: ContextStrategy::RollingContext,
                    ..Default::default()
                },
                ..Default::default()
            },
            messages,
            prompt_usage: Some(PromptUsage {
                prompt_context_tokens: 130_000,
                input_tokens: 130_000,
                cached_input_tokens: 0,
                context_budget_tokens: 130_000,
            }),
            max_context_tokens: Some(200_000),
            host: Arc::new(MockSessionManager::default()),
        })
        .await
        .expect("context")
        .messages;

        let image_part = built[0].parts.first().expect("image part");
        assert!(matches!(image_part.kind, PartKind::Image));
        assert!(image_part.attachment.is_none());
        assert_eq!(image_part.content, PRUNED_IMAGE_PLACEHOLDER);
    }

    #[tokio::test]
    async fn rolling_context_builder_replaces_prefix_with_summary() {
        let manager = Arc::new(MockSessionManager::default());
        let built = build_context(ContextBuildRequest {
            session_id: "root".to_string(),
            state: SessionStateEnvelope {
                session_id: "root".to_string(),
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
}
