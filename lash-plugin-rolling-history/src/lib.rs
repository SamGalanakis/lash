//! Default rolling-history plugin.
//!
//! Owns the compaction strategy that used to live in
//! `lash/src/session_model/context.rs`: per-turn pruning (tool results,
//! old images) and summary insertion, plus persistent history rewrites
//! for manual compaction, overflow recovery, and window-shrink events.
//!
//! Registered as a default plugin by
//! the first-party default tool bundles from `lash-default-tools`,
//! so standard lash sessions pick it up automatically.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use sha2::{Digest, Sha256};

use lash::plugin::{
    DEFAULT_TOOL_RESULT_PROJECTION_LIMIT_BYTES, DEFAULT_TOOL_RESULT_PROJECTION_MAX_LINES,
    HistoryError, HistoryRewriter, HistoryState, ModeExtras, PluginError, PluginFactory,
    PluginRegistrar, PluginSessionContext, RewriteContext, RewriteTrigger, SessionContextSurface,
    SessionCreateRequest, SessionPlugin, SessionPluginMode, SessionStartPoint,
    TurnContextTransform, TurnTransformContext,
};
use lash::session_model::context::PreparedContext;
use lash::session_model::format_tool_result_content;
use lash::{
    ExecutionMode, InputItem, Message, MessageOrigin, MessageRole, Part, PartKind, PromptUsage,
    RollingHistoryConfig, SessionStateEnvelope, StandardContextApproach, ToolCallRecord, TurnInput,
};

fn tool_spill_dir() -> std::path::PathBuf {
    std::env::temp_dir().join("lash-tool-output")
}

const TOOL_RESULT_MAX_LINES: usize = DEFAULT_TOOL_RESULT_PROJECTION_MAX_LINES;
const TOOL_RESULT_MAX_BYTES: usize = DEFAULT_TOOL_RESULT_PROJECTION_LIMIT_BYTES;
const PRUNE_MINIMUM_TOKENS: usize = 20_000;
const PRUNE_PROTECT_TOKENS: usize = 40_000;
const PRUNE_RECENT_USER_TURNS: usize = 2;
const COMPACTION_BUFFER_TOKENS: usize = 20_000;
const COMPACTION_KEEP_RECENT_TOKENS: usize = 20_000;
const PRUNE_CONTEXT_THRESHOLD: f64 = 0.6;
/// Marker `plugin_id` stamped on compaction summary messages so the
/// history pipeline can recognize them on subsequent turns.
pub(crate) const ROLLING_HISTORY_PLUGIN_ID: &str = "rolling_history";
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

fn with_instructions(base: &str, instructions: Option<&str>) -> String {
    match instructions {
        Some(text) if !text.trim().is_empty() => {
            format!("{base}\n\nAdditional focus:\n{}\n", text.trim())
        }
        _ => base.to_string(),
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
        Some(path) => retained_output_hint(path),
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

fn retained_output_hint(path: &Path) -> String {
    format!(
        "Full output saved to: {}\nUse `read_file` with `offset`/`limit` or `grep` to inspect specific sections instead of reading the whole file at once.",
        path.display()
    )
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
    record: &ToolCallRecord,
    full_output: &str,
) -> Option<PathBuf> {
    let dir = base_dir.join("tool-output");
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

fn existing_tool_output_path(record: &ToolCallRecord) -> Option<PathBuf> {
    record
        .result
        .get("full_output_path")
        .and_then(|value| value.as_str())
        .filter(|value| !value.trim().is_empty())
        .map(PathBuf::from)
}

fn render_tool_result_preview(record: &ToolCallRecord) -> String {
    let normalized = normalize_tool_result_content(record);
    truncate_tool_result_preview(
        &normalized,
        tool_result_truncation_direction(&record.tool),
        None,
    )
}

fn render_tool_result_preview_for_session(record: &ToolCallRecord) -> String {
    let normalized = normalize_tool_result_content(record);
    let existing_output_path = existing_tool_output_path(record);
    if tool_result_needs_truncation(&normalized) {
        let output_path = existing_output_path
            .or_else(|| spill_tool_output_to_dir(&tool_spill_dir(), record, &normalized));
        return truncate_tool_result_preview(
            &normalized,
            tool_result_truncation_direction(&record.tool),
            output_path.as_deref(),
        );
    }

    match existing_output_path {
        Some(path) => format!("{normalized}\n\n{}", retained_output_hint(&path)),
        None => normalized,
    }
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
    messages: &mut [Message],
    tool_calls: &HashMap<String, ToolCallRecord>,
) {
    for message in messages {
        for part in std::sync::Arc::make_mut(&mut message.parts).iter_mut() {
            if !matches!(part.kind, PartKind::ToolResult) {
                continue;
            }
            if matches!(part.prune_state, lash::PruneState::Cleared) {
                continue;
            }
            if !matches!(part.prune_state, lash::PruneState::Intact) {
                continue;
            }
            let Some(call_id) = part.tool_call_id.as_deref() else {
                continue;
            };
            let Some(record) = tool_calls.get(call_id) else {
                continue;
            };
            part.content = render_tool_result_preview_for_session(record);
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
            if matches!(part.prune_state, lash::PruneState::Cleared) {
                break 'scan;
            }
            if !matches!(part.prune_state, lash::PruneState::Intact) {
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
        let parts = std::sync::Arc::make_mut(&mut messages[msg_idx].parts);
        parts[part_idx].prune_state = lash::PruneState::Cleared;
        parts[part_idx].content.clear();
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
        for part in std::sync::Arc::make_mut(&mut messages[msg_idx].parts).iter_mut() {
            changed |= strip_image_attachment(part, PRUNED_IMAGE_PLACEHOLDER);
        }
    }

    changed
}

fn strip_all_image_attachments(messages: &mut [Message], placeholder: &str) -> bool {
    let mut changed = false;
    for message in messages {
        for part in std::sync::Arc::make_mut(&mut message.parts).iter_mut() {
            changed |= strip_image_attachment(part, placeholder);
        }
    }
    changed
}

fn is_compaction_summary_message(message: &Message) -> bool {
    matches!(
        message.origin,
        Some(MessageOrigin::Plugin { ref plugin_id, .. }) if plugin_id == ROLLING_HISTORY_PLUGIN_ID
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
    let start = messages[prefix_len..]
        .iter()
        .rposition(is_compaction_summary_message)
        .map(|i| prefix_len + i + 1)
        .unwrap_or(prefix_len);

    let mut accumulated = 0usize;
    for idx in (start..messages.len()).rev() {
        for part in messages[idx].parts.iter() {
            accumulated += approx_token_count(&part.content);
            if part.attachment.is_some() {
                accumulated += 1200; // approximate image token cost
            }
        }
        if accumulated >= COMPACTION_KEEP_RECENT_TOKENS && messages[idx].role == MessageRole::User {
            return idx;
        }
    }
    latest_user_index(messages).unwrap_or(messages.len())
}

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
    instructions: Option<&str>,
    host: Arc<dyn lash::HistoryHost>,
) -> Result<Option<String>, HistoryError> {
    if prefix_messages.is_empty() {
        return Ok(None);
    }

    let mut snapshot = lash::PersistedSessionState::from_state(state.clone());
    snapshot.policy.execution_mode = ExecutionMode::standard();
    snapshot.policy.max_turns = Some(1);
    let mut messages = prefix_messages;
    strip_all_image_attachments(&mut messages, COMPACTED_IMAGE_PLACEHOLDER);
    snapshot.execution_state_snapshot = None;
    snapshot.last_prompt_usage = None;
    let previous_summary = extract_previous_summary(&messages);
    let referenced = referenced_tool_call_ids(&messages);
    let read_view = state.read_view();
    let tool_calls = read_view
        .tool_calls()
        .iter()
        .filter(|record| {
            record
                .call_id
                .as_ref()
                .is_some_and(|call_id| referenced.contains(call_id))
        })
        .cloned()
        .collect::<Vec<_>>();
    snapshot.replace_active_read_state(&messages, &tool_calls);

    let compaction_session_id = format!("{session_id}-compaction");
    let mut policy = snapshot.policy.clone();
    policy.execution_mode = ExecutionMode::standard();
    policy.max_turns = Some(1);
    let handle = host
        .create_session(SessionCreateRequest {
            session_id: Some(compaction_session_id),
            relation: lash::SessionRelation::Child {
                parent_session_id: session_id.to_string(),
            },
            start: SessionStartPoint::Snapshot {
                snapshot: Box::new(snapshot),
            },
            policy: Some(policy),
            plugin_mode: SessionPluginMode::Fresh,
            initial_nodes: Vec::new(),
            first_turn_input: None,
            tool_access: lash::SessionToolAccess::default(),
            subagent: None,
            context_surface: SessionContextSurface {
                include_base_tools: false,
                tool_providers: Vec::new(),
                prompt_contributions: Vec::new(),
            },
            mode_extras: ModeExtras::default(),
            usage_source: Some("compaction".to_string()),
        })
        .await
        .map_err(HistoryError::from)?;

    let base_prompt = match previous_summary {
        Some(prev) => compaction_update_prompt(&prev),
        None => COMPACTION_PROMPT.to_string(),
    };
    let prompt_text = with_instructions(&base_prompt, instructions);

    let turn = host
        .start_turn(
            &handle.session_id,
            TurnInput {
                items: vec![InputItem::Text { text: prompt_text }],
                image_blobs: HashMap::new(),
                user_input: None,
                mode: None,
                mode_turn_options: None,
                trace_turn_id: None,
                mode_extension: None,
                turn_context: lash::TurnContext::default(),
            },
        )
        .await;
    let _ = host.close_session(&handle.session_id).await;
    let turn = turn.map_err(HistoryError::from)?;
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
        parts: lash::shared_parts(vec![Part {
            id: "compaction-summary.p0".to_string(),
            kind: PartKind::Prose,
            content: format!("{COMPACTION_SUMMARY_TITLE}\n{summary}"),
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            tool_item_id: None,
            tool_signature: None,
            prune_state: lash::PruneState::Intact,
            reasoning_meta: None,
            response_meta: None,
        }]),
        user_input: None,
        origin: Some(MessageOrigin::Plugin {
            plugin_id: ROLLING_HISTORY_PLUGIN_ID.to_string(),
            transient: false,
        }),
    });
    out.extend_from_slice(&messages[cut_point..]);
    out
}

async fn compact_messages_core(
    session_id: &str,
    state: &SessionStateEnvelope,
    messages: &[Message],
    instructions: Option<&str>,
    host: Arc<dyn lash::HistoryHost>,
) -> Result<Option<Vec<Message>>, HistoryError> {
    let prefix_len = leading_system_prefix_len(messages);
    let cut_point = find_compaction_cut_point(messages, prefix_len);
    if cut_point <= prefix_len {
        return Ok(None);
    }
    let prefix_messages = messages[prefix_len..cut_point].to_vec();
    let Some(summary) =
        summarize_compaction_prefix(session_id, state, prefix_messages, instructions, host).await?
    else {
        return Ok(None);
    };
    Ok(Some(apply_compaction_summary(
        messages, &summary, cut_point,
    )))
}

pub struct RollingHistoryPluginFactory {
    config: RollingHistoryConfig,
}

impl RollingHistoryPluginFactory {
    pub fn new(config: RollingHistoryConfig) -> Self {
        Self { config }
    }
}

impl Default for RollingHistoryPluginFactory {
    fn default() -> Self {
        Self::new(RollingHistoryConfig)
    }
}

impl PluginFactory for RollingHistoryPluginFactory {
    fn id(&self) -> &'static str {
        ROLLING_HISTORY_PLUGIN_ID
    }

    fn supported_standard_context_approaches(
        &self,
    ) -> &'static [lash::StandardContextApproachKind] {
        &[lash::StandardContextApproachKind::RollingHistory]
    }

    fn build(&self, ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        if ctx.execution_mode != ExecutionMode::standard()
            || !matches!(
                ctx.standard_context_approach,
                Some(StandardContextApproach::RollingHistory(_))
            )
        {
            return Ok(Arc::new(DisabledRollingHistoryPlugin));
        }
        Ok(Arc::new(RollingHistoryPlugin {
            config: self.config.clone(),
        }))
    }
}

struct DisabledRollingHistoryPlugin;

impl SessionPlugin for DisabledRollingHistoryPlugin {
    fn id(&self) -> &'static str {
        ROLLING_HISTORY_PLUGIN_ID
    }

    fn register(&self, _reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        Ok(())
    }
}

struct RollingHistoryPlugin {
    config: RollingHistoryConfig,
}

impl SessionPlugin for RollingHistoryPlugin {
    fn id(&self) -> &'static str {
        ROLLING_HISTORY_PLUGIN_ID
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        let config = self.config.clone();
        reg.history()
            .prepare_turn(100, Arc::new(RollingTurnTransform::new(config.clone())));
        reg.history()
            .rewrite(100, Arc::new(RollingHistoryRewriter::new(config)));
        Ok(())
    }
}

struct RollingTurnTransform;

impl RollingTurnTransform {
    fn new(_config: RollingHistoryConfig) -> Self {
        Self
    }
}

#[async_trait]
impl TurnContextTransform for RollingTurnTransform {
    fn id(&self) -> &'static str {
        "rolling_history.prepare_turn"
    }

    async fn transform(
        &self,
        ctx: &TurnTransformContext,
        mut input: PreparedContext,
    ) -> Result<PreparedContext, HistoryError> {
        let state = &ctx.state;
        let prompt_usage = ctx.prompt_usage.as_ref();
        let max_context_tokens = ctx.max_context_tokens;
        let host = Arc::clone(&ctx.host);

        let tool_calls = tool_record_map(state.tool_calls());
        let needs_pruning = pruning_needed(prompt_usage, max_context_tokens);
        let needs_compaction = compaction_needed(prompt_usage, max_context_tokens);
        if tool_calls.is_empty() && !needs_pruning && !needs_compaction {
            return Ok(input);
        }

        let messages = input.messages.make_mut();
        hydrate_tool_result_parts(messages, &tool_calls);

        if needs_pruning {
            prune_old_tool_results(messages, &tool_calls);
            prune_old_images(messages);
        }

        if !needs_compaction {
            return Ok(input);
        }

        let messages = input.messages.as_slice();
        let prefix_len = leading_system_prefix_len(messages);
        let cut_point = find_compaction_cut_point(messages, prefix_len);
        if cut_point <= prefix_len {
            return Ok(input);
        }

        let prefix_messages = messages[prefix_len..cut_point].to_vec();
        let Some(summary) = summarize_compaction_prefix(
            &ctx.session_id,
            &state.to_owned_state(),
            prefix_messages,
            None,
            host,
        )
        .await?
        else {
            return Ok(input);
        };

        input
            .messages
            .replace(apply_compaction_summary(messages, &summary, cut_point));
        Ok(input)
    }
}

struct RollingHistoryRewriter;

impl RollingHistoryRewriter {
    fn new(_config: RollingHistoryConfig) -> Self {
        Self
    }
}

#[async_trait]
impl HistoryRewriter for RollingHistoryRewriter {
    fn id(&self) -> &'static str {
        "rolling_history.rewrite"
    }

    async fn rewrite(
        &self,
        ctx: &RewriteContext,
        mut input: HistoryState,
    ) -> Result<HistoryState, HistoryError> {
        let session_id = ctx.session_id.clone();
        let host = Arc::clone(&ctx.host);

        match &ctx.trigger {
            RewriteTrigger::Manual { instructions } => {
                if let Some(compacted) = compact_messages_core(
                    &session_id,
                    &ctx.state.to_owned_state(),
                    &input.messages,
                    instructions.as_deref(),
                    host,
                )
                .await?
                {
                    input.metadata.produced_summary = true;
                    input.metadata.pruned_message_count =
                        input.messages.len().saturating_sub(compacted.len()) as u32;
                    input.messages = compacted;
                }
                Ok(input)
            }
            RewriteTrigger::OverflowRecovery => {
                strip_all_image_attachments(&mut input.messages, COMPACTED_IMAGE_PLACEHOLDER);
                let tool_calls = tool_record_map(&input.tool_calls);
                prune_old_tool_results(&mut input.messages, &tool_calls);
                if let Some(compacted) = compact_messages_core(
                    &session_id,
                    &ctx.state.to_owned_state(),
                    &input.messages,
                    None,
                    host,
                )
                .await?
                {
                    input.metadata.produced_summary = true;
                    input.messages = compacted;
                }
                if let Some(max) = ctx.state.policy().max_context_tokens {
                    let total: usize = input
                        .messages
                        .iter()
                        .flat_map(|m| m.parts.iter())
                        .map(|p| approx_token_count(&p.content))
                        .sum();
                    if total > max.saturating_sub(COMPACTION_BUFFER_TOKENS) {
                        for msg_idx in (0..input.messages.len()).rev() {
                            let recent_user_turns = input.messages[msg_idx + 1..]
                                .iter()
                                .filter(|msg| msg.role == MessageRole::User)
                                .count();
                            if recent_user_turns < PRUNE_RECENT_USER_TURNS {
                                continue;
                            }
                            for part in std::sync::Arc::make_mut(&mut input.messages[msg_idx].parts)
                                .iter_mut()
                            {
                                if matches!(part.kind, PartKind::ToolResult)
                                    && matches!(part.prune_state, lash::PruneState::Intact)
                                {
                                    part.prune_state = lash::PruneState::Cleared;
                                    part.content.clear();
                                }
                            }
                        }
                    }
                }
                Ok(input)
            }
            RewriteTrigger::WindowShrink { .. } => {
                if !compaction_needed(
                    ctx.state.last_prompt_usage(),
                    ctx.state.policy().max_context_tokens,
                ) {
                    return Ok(input);
                }
                if let Some(compacted) = compact_messages_core(
                    &session_id,
                    &ctx.state.to_owned_state(),
                    &input.messages,
                    None,
                    host,
                )
                .await?
                {
                    input.metadata.produced_summary = true;
                    input.messages = compacted;
                }
                Ok(input)
            }
            RewriteTrigger::Periodic => Ok(input),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lash::SessionPolicy;
    use serde_json::json;
    use tempfile::tempdir;

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
                tool_item_id: None,
                tool_signature: None,
                prune_state: lash::PruneState::Intact,
                reasoning_meta: None,
                response_meta: None,
            }]
            .into(),
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
                attachment: Some(lash::session_model::message::PartAttachment {
                    reference: lash::AttachmentRef {
                        id: lash::AttachmentId::new(format!("{id}-att")),
                        media_type: lash::MediaType::Image(lash::ImageMediaType::Png),
                        byte_len: bytes.len() as u64,
                        width: None,
                        height: None,
                        label: None,
                    },
                }),
                tool_call_id: None,
                tool_name: None,
                tool_item_id: None,
                tool_signature: None,
                prune_state: lash::PruneState::Intact,
                reasoning_meta: None,
                response_meta: None,
            }]
            .into(),
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
            control: None,
        };

        let normalized = normalize_tool_result_content(&record);
        let path = spill_tool_output_to_dir(temp.path(), &record, &normalized).expect("spill path");
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

    #[test]
    fn render_tool_result_preview_for_session_reuses_existing_output_path() {
        let record = ToolCallRecord {
            call_id: Some("call-123".to_string()),
            tool: "exec_command".to_string(),
            args: json!({"cmd":"cat giant.log"}),
            result: json!({
                "output": format!("{}\nend", "line\n".repeat(2_500)),
                "full_output_path": "/tmp/existing-shell-output.log",
            }),
            success: true,
            duration_ms: 5,
            control: None,
        };

        let preview = render_tool_result_preview_for_session(&record);
        assert!(preview.contains("Full output saved to: /tmp/existing-shell-output.log"));
    }

    use lash::testing::{MockSessionManager, mock_assembled_turn as empty_turn};

    fn mock_manager() -> MockSessionManager {
        MockSessionManager::default()
            .with_tool_catalog(vec![
                json!({"name":"exec_command"}),
                json!({"name":"read_file"}),
            ])
            .with_turn(empty_turn("root", "Compacted work summary"))
    }

    fn build_turn_ctx(
        session_id: &str,
        state: SessionStateEnvelope,
        prompt_usage: Option<PromptUsage>,
        max_context_tokens: Option<usize>,
        host: Arc<dyn lash::HistoryHost>,
    ) -> TurnTransformContext {
        TurnTransformContext {
            session_id: session_id.to_string(),
            state: state.read_view(),
            prompt_usage,
            max_context_tokens,
            host,
        }
    }

    #[tokio::test]
    async fn rolling_turn_transform_clears_old_tool_outputs() {
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
                control: None,
            })
            .collect::<Vec<_>>();
        let mut messages = vec![text_message("u0", MessageRole::User, "older")];
        messages.push(Message {
            id: "a1".to_string(),
            role: MessageRole::Assistant,
            parts: Arc::new(
                tool_calls
                    .iter()
                    .enumerate()
                    .map(|(idx, record)| Part {
                        id: format!("a1.p{idx}"),
                        kind: PartKind::ToolResult,
                        content: String::new(),
                        attachment: None,
                        tool_call_id: record.call_id.clone(),
                        tool_name: Some(record.tool.clone()),
                        tool_item_id: None,
                        tool_signature: None,
                        prune_state: lash::PruneState::Intact,
                        reasoning_meta: None,
                        response_meta: None,
                    })
                    .collect(),
            ),
            user_input: None,
            origin: None,
        });
        messages.push(text_message("u2", MessageRole::User, "recent"));
        messages.push(text_message("u3", MessageRole::User, "latest"));

        let mut state = SessionStateEnvelope {
            policy: SessionPolicy::default(),
            ..Default::default()
        };
        state.replace_active_read_state(&messages, &tool_calls);
        let transform = RollingTurnTransform::new(RollingHistoryConfig);
        let host: Arc<dyn lash::HistoryHost> = Arc::new(mock_manager());
        let ctx = build_turn_ctx(
            "root",
            state,
            Some(PromptUsage {
                prompt_context_tokens: 130_000,
                input_tokens: 130_000,
                cached_input_tokens: 0,
                context_budget_tokens: 130_000,
            }),
            Some(200_000),
            host,
        );
        let prepared = PreparedContext {
            messages: messages.into(),
            ..Default::default()
        };
        let built = transform
            .transform(&ctx, prepared)
            .await
            .expect("transform")
            .messages;

        let cleared = built
            .iter()
            .flat_map(|message| message.parts.iter())
            .filter(|part| matches!(part.prune_state, lash::PruneState::Cleared))
            .count();
        assert!(cleared > 0);
    }

    #[tokio::test]
    async fn rolling_turn_transform_strips_old_image_attachments() {
        let messages = vec![
            image_message("u0", MessageRole::User, &[1, 2, 3]),
            text_message("u1", MessageRole::User, "recent"),
            text_message("u2", MessageRole::User, "latest"),
        ];

        let state = SessionStateEnvelope::default();
        let host: Arc<dyn lash::HistoryHost> = Arc::new(mock_manager());
        let transform = RollingTurnTransform::new(RollingHistoryConfig);
        let ctx = build_turn_ctx(
            "root",
            state,
            Some(PromptUsage {
                prompt_context_tokens: 130_000,
                input_tokens: 130_000,
                cached_input_tokens: 0,
                context_budget_tokens: 130_000,
            }),
            Some(200_000),
            host,
        );
        let prepared = PreparedContext {
            messages: messages.into(),
            ..Default::default()
        };
        let built = transform
            .transform(&ctx, prepared)
            .await
            .expect("transform")
            .messages;

        let image_part = built[0].parts.first().expect("image part");
        assert!(matches!(image_part.kind, PartKind::Image));
        assert!(image_part.attachment.is_none());
        assert_eq!(image_part.content, PRUNED_IMAGE_PLACEHOLDER);
    }

    #[tokio::test]
    async fn rolling_turn_transform_replaces_prefix_with_summary() {
        let manager = Arc::new(mock_manager());
        let host: Arc<dyn lash::HistoryHost> = manager.clone();
        let transform = RollingTurnTransform::new(RollingHistoryConfig);
        let state = SessionStateEnvelope {
            session_id: "root".to_string(),
            policy: SessionPolicy {
                execution_mode: ExecutionMode::standard(),
                ..Default::default()
            },
            ..Default::default()
        };
        let ctx = build_turn_ctx(
            "root",
            state,
            Some(PromptUsage {
                prompt_context_tokens: 90_000,
                input_tokens: 90_000,
                cached_input_tokens: 0,
                context_budget_tokens: 90_000,
            }),
            Some(100_000),
            host,
        );
        let prepared = PreparedContext {
            messages: vec![
                text_message("u1", MessageRole::User, "old work"),
                text_message("a1", MessageRole::Assistant, "assistant old"),
                text_message("u2", MessageRole::User, "latest request"),
            ]
            .into(),
            ..Default::default()
        };
        let built = transform
            .transform(&ctx, prepared)
            .await
            .expect("transform")
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

        let created = manager.created_snapshot();
        assert_eq!(created.len(), 1);
    }
}
