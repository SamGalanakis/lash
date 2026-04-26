use super::*;
use std::collections::HashSet;

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

    fn push(&mut self, item: UiTimelineItem) {
        self.items.push(item);
    }

    fn extend(&mut self, items: impl IntoIterator<Item = UiTimelineItem>) {
        self.items.extend(items);
    }

    fn last(&self) -> Option<&UiTimelineItem> {
        self.items.last()
    }

    fn last_mut(&mut self) -> Option<&mut UiTimelineItem> {
        self.items.last_mut()
    }

    #[cfg(test)]
    pub(crate) fn items(&self) -> &[UiTimelineItem] {
        self.items.as_slice()
    }

    pub(crate) fn to_display_blocks(&self) -> Vec<DisplayBlock> {
        self.items.iter().cloned().map(Into::into).collect()
    }

    fn into_display_blocks(self) -> Vec<DisplayBlock> {
        self.items.into_iter().map(Into::into).collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum UiTimelineItem {
    TurnStart(Turn),
    UserInput(String),
    AssistantReasoning(String),
    AssistantText(String),
    Activity(Box<ActivityBlock>),
    ShellOutput {
        command: String,
        output: String,
        error: Option<String>,
    },
    Error(String),
    SystemMessage(String),
    PluginPanel(PluginPanelBlock),
    LashlangCode(String),
    Splash,
}

impl From<UiTimelineItem> for DisplayBlock {
    fn from(item: UiTimelineItem) -> Self {
        match item {
            UiTimelineItem::TurnStart(turn) => DisplayBlock::TurnStart(turn),
            UiTimelineItem::UserInput(text) => DisplayBlock::UserInput(text),
            UiTimelineItem::AssistantReasoning(text) => DisplayBlock::AssistantReasoning(text),
            UiTimelineItem::AssistantText(text) => DisplayBlock::AssistantText(text),
            UiTimelineItem::Activity(activity) => DisplayBlock::Activity(activity),
            UiTimelineItem::ShellOutput {
                command,
                output,
                error,
            } => DisplayBlock::ShellOutput {
                command,
                output,
                error,
            },
            UiTimelineItem::Error(text) => DisplayBlock::Error(text),
            UiTimelineItem::SystemMessage(text) => DisplayBlock::SystemMessage(text),
            UiTimelineItem::PluginPanel(panel) => DisplayBlock::PluginPanel(panel),
            UiTimelineItem::LashlangCode(code) => DisplayBlock::LashlangCode(code),
            UiTimelineItem::Splash => DisplayBlock::Splash,
        }
    }
}

impl From<DisplayBlock> for UiTimelineItem {
    fn from(block: DisplayBlock) -> Self {
        match block {
            DisplayBlock::TurnStart(turn) => UiTimelineItem::TurnStart(turn),
            DisplayBlock::UserInput(text) => UiTimelineItem::UserInput(text),
            DisplayBlock::AssistantReasoning(text) => UiTimelineItem::AssistantReasoning(text),
            DisplayBlock::AssistantText(text) => UiTimelineItem::AssistantText(text),
            DisplayBlock::Activity(activity) => UiTimelineItem::Activity(activity),
            DisplayBlock::ShellOutput {
                command,
                output,
                error,
            } => UiTimelineItem::ShellOutput {
                command,
                output,
                error,
            },
            DisplayBlock::Error(text) => UiTimelineItem::Error(text),
            DisplayBlock::SystemMessage(text) => UiTimelineItem::SystemMessage(text),
            DisplayBlock::PluginPanel(panel) => UiTimelineItem::PluginPanel(panel),
            DisplayBlock::LashlangCode(code) => UiTimelineItem::LashlangCode(code),
            DisplayBlock::Splash => UiTimelineItem::Splash,
        }
    }
}

pub(crate) fn projected_blocks_from_state(
    events: &[lash::SessionEventRecord],
    messages: &[Message],
    tool_calls: &[ToolCallRecord],
    ui_state: &UiProjectionState,
) -> Vec<DisplayBlock> {
    projected_timeline_from_state(events, messages, tool_calls, ui_state).into_display_blocks()
}

pub(crate) fn projected_timeline_from_state(
    events: &[lash::SessionEventRecord],
    messages: &[Message],
    tool_calls: &[ToolCallRecord],
    ui_state: &UiProjectionState,
) -> UiTimeline {
    let mut timeline = timeline_from_events(events, messages, tool_calls);
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

pub(crate) fn project_interrupted_blocks(
    events: &[lash::SessionEventRecord],
    messages: &[Message],
    tool_calls: &[ToolCallRecord],
    ui_state: &UiProjectionState,
    status_message: impl Into<String>,
) -> Vec<DisplayBlock> {
    let mut timeline = projected_timeline_from_state(events, messages, tool_calls, ui_state);
    push_system_message_item_if_new(&mut timeline, status_message.into());
    timeline.into_display_blocks()
}

fn timeline_from_events(
    events: &[lash::SessionEventRecord],
    messages: &[Message],
    tool_calls: &[ToolCallRecord],
) -> UiTimeline {
    if events.is_empty() {
        return timeline_from_transcript(messages, tool_calls);
    }

    let mut timeline = UiTimeline::new();
    let mut activity_state = ActivityState::default();
    let tool_call_map = tool_calls
        .iter()
        .filter_map(|record| {
            record
                .call_id
                .as_deref()
                .map(|call_id| (call_id, record.clone()))
        })
        .collect::<HashMap<_, _>>();
    let mut seen_messages = HashSet::new();

    for event in events {
        match event {
            lash::SessionEventRecord::Conversation(record) => {
                seen_messages.insert(record.id.clone());
                let message = record.to_message();
                append_transcript_items(
                    &mut timeline,
                    &message,
                    &tool_call_map,
                    &mut activity_state,
                );
            }
            lash::SessionEventRecord::Mode(event) => {
                if let Some(lash_rlm_types::RlmModeEvent::RlmTrajectoryEntry(entry)) =
                    event.rlm_event()
                {
                    append_rlm_trajectory_items(&mut timeline, &entry, &mut activity_state);
                }
            }
            _ => {}
        }
    }

    for message in messages
        .iter()
        .filter(|message| !seen_messages.contains(&message.id))
    {
        append_transcript_items(&mut timeline, message, &tool_call_map, &mut activity_state);
    }

    timeline
}

fn append_rlm_trajectory_items(
    timeline: &mut UiTimeline,
    entry: &lash_rlm_types::RlmTrajectoryEntry,
    activity_state: &mut ActivityState,
) {
    if let Some(reasoning) = rlm_reasoning_display_text(&entry.reasoning) {
        let _ = push_assistant_reasoning_item(timeline, &reasoning);
    }
    timeline.push(UiTimelineItem::LashlangCode(entry.code.clone()));
    for record in &entry.tool_calls {
        for activity in activity_state.blocks_for_tool_call(
            &record.tool,
            record.args.clone(),
            record.result.clone(),
            record.success,
            record.duration_ms,
        ) {
            append_activity_item(timeline, activity);
        }
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

pub(crate) fn interrupted_assistant_tail(blocks: &[DisplayBlock], text: &str) -> Option<String> {
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
                DisplayBlock::TurnStart(turn) if turn.role == TurnRole::User
            )
        })
        .unwrap_or(0);
    let mut peeled_count = 0usize;
    let mut peel_failed = false;
    let mut remaining = cleaned.as_str();
    for block in &blocks[scan_start..] {
        let DisplayBlock::AssistantText(existing) = block else {
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

    if let Some(DisplayBlock::AssistantText(existing)) = blocks.last() {
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
        && let Some(tail) = interrupted_assistant_tail(&timeline.to_display_blocks(), text)
    {
        let _ = push_assistant_text_item(timeline, &tail);
    }
}

pub(crate) fn push_system_message_block_if_new(blocks: &mut Vec<DisplayBlock>, message: String) {
    if matches!(
        blocks.last(),
        Some(DisplayBlock::SystemMessage(existing)) if existing == &message
    ) {
        return;
    }
    blocks.push(DisplayBlock::SystemMessage(message));
}

/// Emit a `TurnStart` marker before a `UserInput` block. The marker is the
/// data-driven source of truth for the horizontal rule between turns. The
/// first turn in the stream (nothing above it, or only a `Splash` above
/// it) does not show a separator — the rule is a *between* signal, not a
/// *leading* ornament.
pub(crate) fn push_user_turn_start(blocks: &mut Vec<DisplayBlock>) {
    let show_separator = match blocks.last() {
        None => false,
        Some(DisplayBlock::Splash) => false,
        Some(DisplayBlock::TurnStart(_)) => false,
        Some(_) => true,
    };
    blocks.push(DisplayBlock::TurnStart(Turn::user(show_separator)));
}

fn timeline_from_transcript(messages: &[Message], tool_calls: &[ToolCallRecord]) -> UiTimeline {
    let mut timeline = UiTimeline::new();
    let mut activity_state = ActivityState::default();
    let tool_call_map = tool_calls
        .iter()
        .filter_map(|record| {
            record
                .call_id
                .as_deref()
                .map(|call_id| (call_id, record.clone()))
        })
        .collect::<HashMap<_, _>>();
    for message in messages {
        append_transcript_items(&mut timeline, message, &tool_call_map, &mut activity_state);
    }
    timeline
}

pub(crate) fn append_activity_block(blocks: &mut Vec<DisplayBlock>, activity: ActivityBlock) {
    if let Some(DisplayBlock::Activity(existing)) = blocks.last_mut()
        && existing.call.kind == ActivityKind::Exploration
        && activity.call.kind == ActivityKind::Exploration
        && existing.result.status == ActivityStatus::Completed
        && activity.result.status == ActivityStatus::Completed
        && merge_exploration_activity(existing, activity.clone())
    {
        return;
    }
    if let Some(DisplayBlock::Activity(existing)) = blocks.last_mut()
        && existing.call.kind == ActivityKind::Edit
        && activity.call.kind == ActivityKind::Edit
        && existing.result.status == ActivityStatus::Completed
        && activity.result.status == ActivityStatus::Completed
        && merge_edit_activity(existing, activity.clone())
    {
        return;
    }
    blocks.push(DisplayBlock::Activity(Box::new(activity)));
}

fn append_activity_item(timeline: &mut UiTimeline, activity: ActivityBlock) {
    if let Some(UiTimelineItem::Activity(existing)) = timeline.last_mut()
        && existing.call.kind == ActivityKind::Exploration
        && activity.call.kind == ActivityKind::Exploration
        && existing.result.status == ActivityStatus::Completed
        && activity.result.status == ActivityStatus::Completed
        && merge_exploration_activity(existing, activity.clone())
    {
        return;
    }
    if let Some(UiTimelineItem::Activity(existing)) = timeline.last_mut()
        && existing.call.kind == ActivityKind::Edit
        && activity.call.kind == ActivityKind::Edit
        && existing.result.status == ActivityStatus::Completed
        && activity.result.status == ActivityStatus::Completed
        && merge_edit_activity(existing, activity.clone())
    {
        return;
    }
    timeline.push(UiTimelineItem::Activity(Box::new(activity)));
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
) {
    match message.role {
        MessageRole::User => {
            if message.parts.iter().any(|part| {
                part_is_rlm_exec_result(part) || matches!(part.kind, PartKind::ToolResult)
            }) {
                for part in &message.parts {
                    append_tool_result_items(timeline, part, tool_calls, activity_state);
                }
            } else {
                let text = message
                    .user_input
                    .as_ref()
                    .map(|user_input| user_input.display_text.clone())
                    .unwrap_or_else(|| rendered_message_text(message));
                if !text.is_empty() {
                    push_user_turn_start_item(timeline);
                    timeline.push(UiTimelineItem::UserInput(text));
                }
            }
        }
        MessageRole::Assistant => {
            for part in &message.parts {
                if !matches!(part.kind, PartKind::Reasoning) {
                    continue;
                }
                let trimmed = part.content.trim();
                if !trimmed.is_empty() {
                    let _ = push_assistant_reasoning_item(timeline, trimmed);
                }
            }

            let mut prose = Vec::new();
            for part in &message.parts {
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
    }
}

/// True when a legacy user-message part carries an RLM exec result.
/// Current RLM history uses trajectory events instead; these old
/// messages should stay hidden rather than render as fake tool calls.
fn part_is_rlm_exec_result(part: &lash::Part) -> bool {
    matches!(part.kind, PartKind::Text)
        && part.tool_call_id.is_some()
        && part.tool_name.as_deref() == Some("execute_lashlang")
}

fn append_tool_result_items(
    timeline: &mut UiTimeline,
    part: &lash::Part,
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

    for activity in activity_state.blocks_for_tool_call(
        &record.tool,
        record.args.clone(),
        record.result.clone(),
        record.success,
        record.duration_ms,
    ) {
        append_activity_item(timeline, activity);
    }
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

fn push_system_message_item_if_new(timeline: &mut UiTimeline, message: String) {
    if matches!(
        timeline.last(),
        Some(UiTimelineItem::SystemMessage(existing)) if existing == &message
    ) {
        return;
    }
    timeline.push(UiTimelineItem::SystemMessage(message));
}

fn push_user_turn_start_item(timeline: &mut UiTimeline) {
    let show_separator = match timeline.last() {
        None => false,
        Some(UiTimelineItem::Splash) => false,
        Some(UiTimelineItem::TurnStart(_)) => false,
        Some(_) => true,
    };
    timeline.push(UiTimelineItem::TurnStart(Turn::user(show_separator)));
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
