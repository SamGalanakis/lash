use super::*;
use crate::assistant_text::{merge_assistant_reasoning_text, normalize_assistant_text};
use std::collections::{HashMap, HashSet};

const TEXT_PREVIEW_MAX_HEAD_LINES: usize = 8;
const TEXT_PREVIEW_MAX_TAIL_LINES: usize = 3;
const TEXT_PREVIEW_LINE_CHAR_LIMIT: usize = 240;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct UiTimeline {
    items: Vec<UiTimelineItem>,
}

impl UiTimeline {
    fn new() -> Self {
        Self::default()
    }

    pub(crate) fn push(&mut self, item: UiTimelineItem) {
        self.items.push(item);
    }

    pub(crate) fn extend(&mut self, items: impl IntoIterator<Item = UiTimelineItem>) {
        self.items.extend(items);
    }

    pub(crate) fn last(&self) -> Option<&UiTimelineItem> {
        self.items.last()
    }

    pub(crate) fn last_mut(&mut self) -> Option<&mut UiTimelineItem> {
        self.items.last_mut()
    }

    pub(crate) fn items(&self) -> &[UiTimelineItem] {
        self.items.as_slice()
    }

    pub(crate) fn truncate(&mut self, len: usize) {
        self.items.truncate(len);
    }

    pub(crate) fn retain(&mut self, f: impl FnMut(&UiTimelineItem) -> bool) {
        self.items.retain(f);
    }

    pub(crate) fn iter_mut(&mut self) -> std::slice::IterMut<'_, UiTimelineItem> {
        self.items.iter_mut()
    }

    pub(crate) fn push_system_message_if_new(&mut self, message: String) {
        if matches!(
            self.last(),
            Some(UiTimelineItem::SystemMessage(existing)) if existing == &message
        ) {
            return;
        }
        self.push(UiTimelineItem::SystemMessage(message));
    }

    pub(crate) fn push_user_turn_start(&mut self) {
        let show_separator = match self.last() {
            None => false,
            Some(UiTimelineItem::Splash) => false,
            Some(UiTimelineItem::TurnStart(_)) => false,
            Some(_) => true,
        };
        self.push(UiTimelineItem::TurnStart(Turn::user(show_separator)));
    }
}

impl From<Vec<UiTimelineItem>> for UiTimeline {
    fn from(items: Vec<UiTimelineItem>) -> Self {
        Self { items }
    }
}

impl std::ops::Deref for UiTimeline {
    type Target = [UiTimelineItem];

    fn deref(&self) -> &Self::Target {
        self.items()
    }
}

/// A renderable item in the scrollable history timeline.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) enum UiTimelineItem {
    /// Turn boundary. Carries the Turn metadata and is the source of
    /// truth for the separator rule.
    TurnStart(Turn),
    UserInput(String),
    /// Model reasoning summary ("thinking"). Rendered above the
    /// assistant's final text in a muted/italic style so the user can see
    /// the model's scratch thoughts without confusing them with the
    /// actual reply. Display-only; never persisted back into prompts.
    AssistantReasoning(String),
    AssistantText(String),
    Activity(Box<ActivityBlock>),
    ShellOutput {
        command: String,
        output: String,
        error: Option<String>,
    },
    Error(String),
    /// Informational message from the system (e.g. /help output).
    SystemMessage(String),
    PluginPanel(PluginPanelBlock),
    /// The fenced `lashlang` source from an RLM turn, captured so the
    /// transcript can reveal the code that produced the subsequent tool
    /// activities. Hidden by default; shown at full expansion.
    LashlangCode(String),
    Splash,
}

pub(crate) fn timeline_from_read_view(
    read_view: &lash_core::SessionReadView,
    ui_state: &UiProjectionState,
) -> UiTimeline {
    let projection = read_view.chronological_projection();
    let mut timeline = timeline_from_chronological(&projection);
    append_live_projection_items(&mut timeline, ui_state);
    timeline.extend(
        ui_state
            .plugin_panels
            .iter()
            .cloned()
            .map(UiTimelineItem::PluginPanel),
    );
    timeline
}

pub(crate) fn interrupted_blocks_from_read_view(
    read_view: &lash_core::SessionReadView,
    ui_state: &UiProjectionState,
    status_message: impl Into<String>,
) -> UiTimeline {
    let mut timeline = timeline_from_read_view(read_view, ui_state);
    timeline.push_system_message_if_new(status_message.into());
    timeline
}

fn timeline_from_chronological(projection: &lash_core::ChronologicalProjection) -> UiTimeline {
    let mut timeline = UiTimeline::new();
    let mut activity_state = ActivityState::default();
    let rlm_owned_tool_call_ids = rlm_owned_tool_call_ids(projection);
    let tool_call_map = projection
        .entries()
        .iter()
        .filter_map(|entry| match &entry.payload {
            lash_core::ChronologicalPayload::ToolCall(record) => record
                .call_id
                .as_deref()
                .map(|call_id| (call_id, record.clone())),
            lash_core::ChronologicalPayload::Message(_)
            | lash_core::ChronologicalPayload::ProtocolEvent(_) => None,
        })
        .collect::<HashMap<_, _>>();

    for entry in projection.entries() {
        match &entry.payload {
            lash_core::ChronologicalPayload::Message(message) => {
                append_transcript_items(
                    &mut timeline,
                    message,
                    &tool_call_map,
                    &mut activity_state,
                    false,
                );
            }
            lash_core::ChronologicalPayload::ToolCall(record) => {
                if !record
                    .call_id
                    .as_deref()
                    .is_some_and(|call_id| rlm_owned_tool_call_ids.contains(call_id))
                {
                    append_tool_call_record_items(&mut timeline, record, &mut activity_state);
                }
            }
            lash_core::ChronologicalPayload::ProtocolEvent(event) => {
                if let Some(lash_rlm_types::RlmProtocolEvent::RlmTrajectoryEntry(entry)) =
                    lash_protocol_rlm::decode_rlm_protocol_event(event)
                {
                    append_rlm_trajectory_items(
                        &mut timeline,
                        &entry,
                        &tool_call_map,
                        &mut activity_state,
                    );
                }
            }
        }
    }

    timeline
}

fn rlm_owned_tool_call_ids(projection: &lash_core::ChronologicalProjection) -> HashSet<String> {
    projection
        .entries()
        .iter()
        .filter_map(|entry| match &entry.payload {
            lash_core::ChronologicalPayload::ProtocolEvent(event) => {
                lash_protocol_rlm::decode_rlm_protocol_event(event)
            }
            lash_core::ChronologicalPayload::Message(_)
            | lash_core::ChronologicalPayload::ToolCall(_) => None,
        })
        .flat_map(|event| match event {
            lash_rlm_types::RlmProtocolEvent::RlmTrajectoryEntry(entry) => entry.tool_call_ids,
            lash_rlm_types::RlmProtocolEvent::RlmGlobalsPatch(_)
            | lash_rlm_types::RlmProtocolEvent::RlmSeed(_)
            | lash_rlm_types::RlmProtocolEvent::RlmDiagnostic(_) => Vec::new(),
        })
        .collect()
}

fn append_rlm_trajectory_items(
    timeline: &mut UiTimeline,
    entry: &lash_rlm_types::RlmTrajectoryEntry,
    tool_calls: &HashMap<&str, ToolCallRecord>,
    activity_state: &mut ActivityState,
) {
    if let Some(reasoning) = rlm_reasoning_display_text(&entry.reasoning) {
        let _ = push_assistant_reasoning_item(timeline, &reasoning);
    }
    timeline.push(UiTimelineItem::LashlangCode(entry.code.clone()));
    for call_id in &entry.tool_call_ids {
        if let Some(record) = tool_calls.get(call_id.as_str()) {
            append_tool_call_record_items(timeline, record, activity_state);
        }
    }
    if let Some(final_output) = &entry.final_output {
        let _ = push_assistant_text_item(timeline, &render_submitted_value(final_output));
    }
}

fn rlm_reasoning_display_text(text: &str) -> Option<String> {
    let cleaned = strip_first_lashlang_fence(text).trim().to_string();
    (!cleaned.is_empty()).then_some(cleaned)
}

fn strip_first_lashlang_fence(text: &str) -> String {
    let Some(open_rel) = text.find("```") else {
        return text.to_string();
    };
    let after_open = open_rel + 3;
    let rest = &text[after_open..];
    let Some(lang_end_rel) = rest.find('\n') else {
        return text[..open_rel].to_string();
    };
    let lang = rest[..lang_end_rel].trim();
    if !matches!(lang, "lashlang" | "rlm" | "lash") {
        return text.to_string();
    }
    let body_start = after_open + lang_end_rel + 1;
    let close = text[body_start..]
        .find("```")
        .map(|rel| body_start + rel)
        .unwrap_or(text.len());
    let after_close = (close + 3).min(text.len());
    let mut out = String::new();
    out.push_str(text[..open_rel].trim_end());
    let tail = text[after_close..].trim_start();
    if !tail.is_empty() {
        if !out.is_empty() {
            out.push_str("\n\n");
        }
        out.push_str(tail);
    }
    out
}

pub(crate) fn interrupted_assistant_tail(blocks: &[UiTimelineItem], text: &str) -> Option<String> {
    let cleaned = normalize_assistant_text(text);
    if cleaned.is_empty() {
        return None;
    }
    // Providers like codex emit prose between tool calls and commit each
    // chunk as its own assistant message, so the transcript already
    // projected into multiple AssistantText blocks for the current turn.
    // The runtime hands us the ENTIRE accumulated `text_deltas` as
    // uncommitted assistant text on abort — pushing it whole would
    // duplicate every prose block already on screen. Walk the current
    // turn's AssistantText blocks in order and peel each from the front
    // of `cleaned`; any leftover is the uncommitted mid-stream tail.
    let scan_start = blocks
        .iter()
        .rposition(|block| {
            matches!(
                block,
                UiTimelineItem::TurnStart(turn) if turn.role == TurnRole::User
            )
        })
        .unwrap_or(0);
    let mut peeled_count = 0usize;
    let mut peel_failed = false;
    let mut remaining = cleaned.as_str();
    for block in &blocks[scan_start..] {
        let UiTimelineItem::AssistantText(existing) = block else {
            continue;
        };
        let normalized = normalize_assistant_text(existing);
        if normalized.is_empty() {
            continue;
        }
        match remaining.strip_prefix(normalized.as_str()) {
            Some(rest) => {
                remaining = rest.trim_start_matches('\n');
                peeled_count += 1;
            }
            None => {
                peel_failed = true;
                break;
            }
        }
    }
    if peeled_count > 0 && !peel_failed {
        let trailing = remaining.trim();
        return (!trailing.is_empty()).then(|| trailing.to_string());
    }

    if let Some(UiTimelineItem::AssistantText(existing)) = blocks.last() {
        let normalized = normalize_assistant_text(existing);
        if cleaned.starts_with(normalized.as_str()) {
            let trailing = cleaned[normalized.len()..].trim_start_matches('\n').trim();
            return (!trailing.is_empty()).then(|| trailing.to_string());
        }
        if normalized.starts_with(cleaned.as_str()) {
            return None;
        }
    }

    Some(cleaned)
}

fn append_live_projection_items(timeline: &mut UiTimeline, ui_state: &UiProjectionState) {
    if let Some(text) = ui_state.live_reasoning_text.as_deref() {
        let _ = push_assistant_reasoning_item(timeline, text);
    }
    if let Some(text) = ui_state.live_assistant_text.as_deref()
        && let Some(tail) = interrupted_assistant_tail(timeline.items(), text)
    {
        let _ = push_assistant_text_item(timeline, &tail);
    }
}

pub(crate) fn preview_text_lines(text: &str) -> Vec<String> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() <= TEXT_PREVIEW_MAX_HEAD_LINES + TEXT_PREVIEW_MAX_TAIL_LINES + 1 {
        return lines
            .into_iter()
            .map(|line| smart_truncate_preview_line(line, TEXT_PREVIEW_LINE_CHAR_LIMIT))
            .collect();
    }

    let hidden = lines
        .len()
        .saturating_sub(TEXT_PREVIEW_MAX_HEAD_LINES + TEXT_PREVIEW_MAX_TAIL_LINES);
    let mut out = Vec::with_capacity(TEXT_PREVIEW_MAX_HEAD_LINES + TEXT_PREVIEW_MAX_TAIL_LINES + 1);
    out.extend(
        lines
            .iter()
            .take(TEXT_PREVIEW_MAX_HEAD_LINES)
            .map(|line| smart_truncate_preview_line(line, TEXT_PREVIEW_LINE_CHAR_LIMIT)),
    );
    out.push(format!("… {hidden} lines hidden …"));
    out.extend(
        lines
            .iter()
            .skip(lines.len().saturating_sub(TEXT_PREVIEW_MAX_TAIL_LINES))
            .map(|line| smart_truncate_preview_line(line, TEXT_PREVIEW_LINE_CHAR_LIMIT)),
    );
    out
}

pub(crate) fn smart_truncate_preview_line(text: &str, max_chars: usize) -> String {
    let char_count = text.chars().count();
    if max_chars == 0 || char_count <= max_chars {
        return text.to_string();
    }

    let marker = format!(" … {} chars hidden … ", char_count - max_chars);
    let marker_chars = marker.chars().count();
    if marker_chars >= max_chars {
        return text.chars().take(max_chars).collect();
    }

    let left_chars = (max_chars - marker_chars) / 2;
    let right_chars = max_chars - marker_chars - left_chars;
    let prefix: String = text.chars().take(left_chars).collect();
    let suffix: String = text
        .chars()
        .skip(char_count.saturating_sub(right_chars))
        .collect();
    format!("{prefix}{marker}{suffix}")
}

#[cfg(test)]
pub(crate) fn strip_ansi_escape_sequences(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut idx = 0usize;
    while idx < bytes.len() {
        if bytes[idx] == 0x1b {
            idx += 1;
            if idx >= bytes.len() {
                break;
            }
            match bytes[idx] {
                b'[' => {
                    idx += 1;
                    while idx < bytes.len() {
                        let byte = bytes[idx];
                        idx += 1;
                        if (0x40..=0x7e).contains(&byte) {
                            break;
                        }
                    }
                }
                b']' => {
                    idx += 1;
                    while idx < bytes.len() {
                        match bytes[idx] {
                            0x07 => {
                                idx += 1;
                                break;
                            }
                            0x1b if idx + 1 < bytes.len() && bytes[idx + 1] == b'\\' => {
                                idx += 2;
                                break;
                            }
                            _ => idx += 1,
                        }
                    }
                }
                b'P' | b'X' | b'^' | b'_' => {
                    idx += 1;
                    while idx < bytes.len() {
                        if bytes[idx] == 0x1b && idx + 1 < bytes.len() && bytes[idx + 1] == b'\\' {
                            idx += 2;
                            break;
                        }
                        idx += 1;
                    }
                }
                _ => {
                    idx += 1;
                }
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

fn append_transcript_items(
    timeline: &mut UiTimeline,
    message: &Message,
    tool_calls: &HashMap<&str, ToolCallRecord>,
    activity_state: &mut ActivityState,
    render_tool_results: bool,
) {
    if is_internal_rlm_message(message) {
        return;
    }
    match message.role {
        MessageRole::User => {
            if message.parts.iter().any(|part| {
                part_is_rlm_exec_result(part) || matches!(part.kind, PartKind::ToolResult)
            }) {
                if render_tool_results {
                    for part in message.parts.iter() {
                        append_tool_result_items(timeline, part, tool_calls, activity_state);
                    }
                }
            } else {
                let text = rendered_message_text(message);
                if !text.is_empty() {
                    timeline.push_user_turn_start();
                    timeline.push(UiTimelineItem::UserInput(text));
                }
            }
        }
        MessageRole::Assistant => {
            for part in message.parts.iter() {
                if !matches!(part.kind, PartKind::Reasoning) {
                    continue;
                }
                let trimmed = part.content.trim();
                if !trimmed.is_empty() {
                    let _ = push_assistant_reasoning_item(timeline, trimmed);
                }
            }

            let mut prose = Vec::new();
            for part in message.parts.iter() {
                if matches!(part.kind, PartKind::Reasoning) {
                    continue;
                }
                let Some(text) = rendered_part_text(&part.kind, &part.content) else {
                    continue;
                };
                match part.kind {
                    PartKind::Text | PartKind::Prose | PartKind::Image => prose.push(text),
                    PartKind::ToolCall => {
                        flush_assistant_prose_items(timeline, &mut prose);
                    }
                    PartKind::Code | PartKind::Output => {}
                    PartKind::Error => {
                        flush_assistant_prose_items(timeline, &mut prose);
                        timeline.push(UiTimelineItem::Error(text));
                    }
                    PartKind::ToolResult => {}
                    PartKind::Reasoning => {}
                }
            }
            flush_assistant_prose_items(timeline, &mut prose);
        }
        MessageRole::System => {
            let text = rendered_message_text(message);
            if !text.is_empty() {
                timeline.push(UiTimelineItem::SystemMessage(text));
            }
        }
        MessageRole::Event => {
            let text = rendered_message_text(message);
            if !text.is_empty() {
                timeline.push(UiTimelineItem::SystemMessage(format!("Event\n{text}")));
            }
        }
    }
}

fn is_internal_rlm_message(message: &Message) -> bool {
    let Some(lash_core::MessageOrigin::Plugin { plugin_id, .. }) = &message.origin else {
        return false;
    };
    if plugin_id != "rlm_protocol" {
        return false;
    }
    match message.role {
        // RLM assistant conversation nodes are loop machinery. The visible
        // answer is projected from the trajectory final_output instead.
        MessageRole::Assistant => true,
        MessageRole::System => {
            rendered_message_text(message).contains("Plain text outside a fence is not delivered.")
        }
        MessageRole::User | MessageRole::Event => false,
    }
}

fn append_tool_call_record_items(
    timeline: &mut UiTimeline,
    record: &ToolCallRecord,
    activity_state: &mut ActivityState,
) {
    if record.tool == "execute_lashlang" {
        return;
    }
    activity_state.append_tool_call_to_timeline(
        timeline,
        &record.tool,
        record.args.clone(),
        record.output.clone(),
        record.duration_ms,
    );
}

/// True when a legacy user-message part carries an RLM exec result.
/// Current RLM history uses trajectory events instead; these old
/// messages should stay off rather than render as fake tool calls.
fn part_is_rlm_exec_result(part: &lash_core::Part) -> bool {
    matches!(part.kind, PartKind::Text)
        && part.tool_call_id.is_some()
        && part.tool_name.as_deref() == Some("execute_lashlang")
}

fn append_tool_result_items(
    timeline: &mut UiTimeline,
    part: &lash_core::Part,
    tool_calls: &HashMap<&str, ToolCallRecord>,
    activity_state: &mut ActivityState,
) {
    if part_is_rlm_exec_result(part) {
        return;
    }
    if !matches!(part.kind, PartKind::ToolResult) {
        return;
    }

    let Some(call_id) = part.tool_call_id.as_deref() else {
        return;
    };
    let Some(record) = tool_calls.get(call_id) else {
        return;
    };

    activity_state.append_tool_call_to_timeline(
        timeline,
        &record.tool,
        record.args.clone(),
        record.output.clone(),
        record.duration_ms,
    );
}

fn flush_assistant_prose_items(timeline: &mut UiTimeline, prose: &mut Vec<String>) {
    if prose.is_empty() {
        return;
    }
    let text = prose.join("\n\n");
    let _ = push_assistant_text_item(timeline, &text);
    prose.clear();
}

fn push_assistant_text_item(timeline: &mut UiTimeline, text: &str) -> bool {
    let cleaned = normalize_assistant_text(text);
    if cleaned.is_empty() {
        return false;
    }
    timeline.push(UiTimelineItem::AssistantText(cleaned));
    true
}

pub(crate) fn render_submitted_value(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => String::new(),
        serde_json::Value::String(text) => text.clone(),
        other => serde_json::to_string_pretty(other).unwrap_or_else(|_| other.to_string()),
    }
}

fn push_assistant_reasoning_item(timeline: &mut UiTimeline, text: &str) -> bool {
    let cleaned = normalize_assistant_text(text);
    if cleaned.is_empty() {
        return false;
    }
    if let Some(UiTimelineItem::AssistantReasoning(existing)) = timeline.last_mut() {
        return merge_assistant_reasoning_text(existing, &cleaned);
    }
    timeline.push(UiTimelineItem::AssistantReasoning(cleaned));
    true
}

pub(crate) fn rendered_message_text(message: &Message) -> String {
    message
        .parts
        .iter()
        .filter_map(|part| rendered_part_text(&part.kind, &part.content))
        .collect::<Vec<_>>()
        .join("\n\n")
        .trim()
        .to_string()
}

fn rendered_part_text(kind: &PartKind, content: &str) -> Option<String> {
    match kind {
        PartKind::Reasoning | PartKind::ToolCall | PartKind::ToolResult => None,
        PartKind::Image => Some("[Image attached]".to_string()),
        _ => (!content.trim().is_empty()).then(|| content.to_string()),
    }
}
