use super::*;

const TEXT_PREVIEW_MAX_HEAD_LINES: usize = 8;
const TEXT_PREVIEW_MAX_TAIL_LINES: usize = 3;
const TEXT_PREVIEW_LINE_CHAR_LIMIT: usize = 240;

pub(crate) fn projected_blocks_from_state(
    messages: &[Message],
    tool_calls: &[ToolCallRecord],
    ui_state: &UiResumeState,
) -> Vec<DisplayBlock> {
    let mut blocks = blocks_from_transcript(messages, tool_calls);
    blocks.extend(
        ui_state
            .plugin_panels
            .iter()
            .cloned()
            .map(DisplayBlock::PluginPanel),
    );
    blocks
}

pub(crate) fn project_interrupted_blocks(
    messages: &[Message],
    tool_calls: &[ToolCallRecord],
    ui_state: &UiResumeState,
    status_message: impl Into<String>,
) -> Vec<DisplayBlock> {
    let mut blocks = projected_blocks_from_state(messages, tool_calls, ui_state);
    push_system_message_block_if_new(&mut blocks, status_message.into());
    blocks
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

pub(crate) fn blocks_from_transcript(
    messages: &[Message],
    tool_calls: &[ToolCallRecord],
) -> Vec<DisplayBlock> {
    let mut blocks = Vec::new();
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
        append_transcript_blocks(&mut blocks, message, &tool_call_map, &mut activity_state);
    }
    blocks
}

pub(crate) fn latest_assistant_text_from_messages(messages: &[Message]) -> Option<String> {
    messages
        .iter()
        .rev()
        .filter(|message| message.role == MessageRole::Assistant)
        .find_map(|message| {
            let text = rendered_message_text(message);
            (!text.is_empty()).then_some(text)
        })
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

fn append_transcript_blocks(
    blocks: &mut Vec<DisplayBlock>,
    message: &Message,
    tool_calls: &HashMap<&str, ToolCallRecord>,
    activity_state: &mut ActivityState,
) {
    match message.role {
        MessageRole::User => {
            if message
                .parts
                .iter()
                .any(|part| matches!(part.kind, PartKind::ToolResult))
            {
                for part in &message.parts {
                    append_tool_result_blocks(blocks, part, tool_calls, activity_state);
                }
            } else {
                let text = message
                    .user_input
                    .as_ref()
                    .map(|user_input| user_input.display_text.clone())
                    .unwrap_or_else(|| rendered_message_text(message));
                if !text.is_empty() {
                    push_user_turn_start(blocks);
                    blocks.push(DisplayBlock::UserInput(text));
                }
            }
        }
        MessageRole::Assistant => {
            let mut prose = Vec::new();
            for part in &message.parts {
                let Some(text) = rendered_part_text(&part.kind, &part.content) else {
                    continue;
                };
                match part.kind {
                    PartKind::Text | PartKind::Prose | PartKind::Image => prose.push(text),
                    PartKind::ToolCall => {
                        flush_assistant_prose(blocks, &mut prose);
                    }
                    PartKind::Code | PartKind::Output => {}
                    PartKind::Error => {
                        flush_assistant_prose(blocks, &mut prose);
                        blocks.push(DisplayBlock::Error(text));
                    }
                    PartKind::ToolResult => {}
                }
            }
            flush_assistant_prose(blocks, &mut prose);
        }
        MessageRole::System => {
            let text = rendered_message_text(message);
            if !text.is_empty() {
                blocks.push(DisplayBlock::SystemMessage(text));
            }
        }
    }
}

fn append_tool_result_blocks(
    blocks: &mut Vec<DisplayBlock>,
    part: &lash::Part,
    tool_calls: &HashMap<&str, ToolCallRecord>,
    activity_state: &mut ActivityState,
) {
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
        append_activity_block(blocks, activity);
    }
}

fn flush_assistant_prose(blocks: &mut Vec<DisplayBlock>, prose: &mut Vec<String>) {
    if prose.is_empty() {
        return;
    }
    let text = prose.join("\n\n");
    let _ = push_assistant_text_block(blocks, &text);
    prose.clear();
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
        PartKind::ToolCall | PartKind::ToolResult => None,
        PartKind::Image => Some("[Image attached]".to_string()),
        _ => (!content.trim().is_empty()).then(|| content.to_string()),
    }
}
