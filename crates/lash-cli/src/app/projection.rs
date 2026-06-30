use super::*;
use crate::assistant_text::normalize_assistant_text;
use crate::skill_prompt::strip_appended_skill_blocks;
use lash_export::transcript::{
    TranscriptEntryKind, final_value_display_text, projection_transcript_entries,
};
use std::collections::HashMap;

use crate::activity::{ActivityCall, ActivityExtra, ActivityResult};

const TEXT_PREVIEW_MAX_HEAD_LINES: usize = 8;
const TEXT_PREVIEW_MAX_TAIL_LINES: usize = 3;
const TEXT_PREVIEW_LINE_CHAR_LIMIT: usize = 240;
const UI_ACTIVITY_SUMMARY_CHAR_LIMIT: usize = 512;
const UI_ACTIVITY_DETAIL_LINE_LIMIT: usize = 24;
const UI_ACTIVITY_DETAIL_CHAR_LIMIT: usize = 320;
const UI_ACTIVITY_ARTIFACT_TEXT_CHAR_LIMIT: usize = 16 * 1024;
const UI_ACTIVITY_ARTIFACT_ITEM_LIMIT: usize = 64;
const UI_ACTIVITY_CHILD_LIMIT: usize = 32;

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
    /// The source from a paired `<lashlang>` RLM block, captured so the
    /// transcript can reveal the code that produced the subsequent tool
    /// activities. Hidden by default; shown at full expansion.
    LashlangCode(String),
    Splash,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct UiActivityJournal {
    #[serde(default)]
    entries: Vec<UiActivityJournalEntry>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct UiActivityRecord {
    pub turn_ordinal: usize,
    pub lashlang_block_ordinal: usize,
    pub activity: UiReplayActivity,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct UiReplayActivity {
    pub call: UiReplayActivityCall,
    pub result: UiReplayActivityResult,
    pub duration_ms: u64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<UiReplayActivity>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct UiReplayActivityCall {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    pub kind: ActivityKind,
    pub tool_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tag: Option<String>,
    pub summary: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra: Option<ActivityExtra>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct UiReplayActivityResult {
    pub status: ActivityStatus,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub detail_lines: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact: Option<ActivityArtifact>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct UiActivityJournalEntry {
    pub turn_ordinal: usize,
    pub lashlang_block_ordinal: usize,
    pub items: Vec<UiTimelineItem>,
}

impl UiActivityJournal {
    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub(crate) fn entries(&self) -> &[UiActivityJournalEntry] {
        &self.entries
    }

    pub(crate) fn apply_record(&mut self, record: UiActivityRecord) {
        let entry = match self.entries.iter_mut().find(|entry| {
            entry.turn_ordinal == record.turn_ordinal
                && entry.lashlang_block_ordinal == record.lashlang_block_ordinal
        }) {
            Some(entry) => entry,
            None => {
                self.entries.push(UiActivityJournalEntry {
                    turn_ordinal: record.turn_ordinal,
                    lashlang_block_ordinal: record.lashlang_block_ordinal,
                    items: Vec::new(),
                });
                self.entries
                    .last_mut()
                    .expect("entry was just pushed into journal")
            }
        };
        let mut timeline = UiTimeline::from(std::mem::take(&mut entry.items));
        ActivityState::append_projected_activity_to_timeline(
            &mut timeline,
            record.activity.into_activity_block(),
        );
        entry.items = timeline.items;
    }
}

impl UiActivityRecord {
    pub(crate) fn new(
        turn_ordinal: usize,
        lashlang_block_ordinal: usize,
        activity: ActivityBlock,
    ) -> Self {
        Self {
            turn_ordinal,
            lashlang_block_ordinal,
            activity: UiReplayActivity::from_activity(compact_activity_for_replay(activity)),
        }
    }
}

impl UiReplayActivity {
    fn from_activity(activity: ActivityBlock) -> Self {
        Self {
            call: UiReplayActivityCall {
                call_id: activity.call.call_id,
                kind: activity.call.kind,
                tool_name: activity.call.tool_name,
                tag: activity.call.tag,
                summary: activity.call.summary,
                extra: activity.call.extra,
            },
            result: UiReplayActivityResult {
                status: activity.result.status,
                detail_lines: activity.result.detail_lines,
                artifact: activity.result.artifact,
            },
            duration_ms: activity.duration_ms,
            children: activity
                .children
                .into_iter()
                .map(UiReplayActivity::from_activity)
                .collect(),
        }
    }

    fn into_activity_block(self) -> ActivityBlock {
        ActivityBlock {
            call: ActivityCall {
                call_id: self.call.call_id,
                kind: self.call.kind,
                tool_name: self.call.tool_name,
                args: serde_json::Value::Null,
                tag: self.call.tag,
                summary: self.call.summary,
                extra: self.call.extra,
            },
            result: ActivityResult {
                status: self.result.status,
                raw: serde_json::Value::Null,
                detail_lines: self.result.detail_lines,
                artifact: self.result.artifact,
            },
            duration_ms: self.duration_ms,
            children: self
                .children
                .into_iter()
                .map(UiReplayActivity::into_activity_block)
                .collect(),
        }
    }
}

fn compact_activity_for_replay(mut activity: ActivityBlock) -> ActivityBlock {
    activity.call.args = serde_json::Value::Null;
    activity.call.tool_name =
        truncate_chars(activity.call.tool_name, UI_ACTIVITY_SUMMARY_CHAR_LIMIT);
    activity.call.summary = truncate_chars(activity.call.summary, UI_ACTIVITY_SUMMARY_CHAR_LIMIT);
    activity.call.tag = activity
        .call
        .tag
        .map(|tag| truncate_chars(tag, UI_ACTIVITY_SUMMARY_CHAR_LIMIT));
    if let Some(ActivityExtra::Exploration(ops)) = activity.call.extra.as_mut() {
        ops.truncate(UI_ACTIVITY_ARTIFACT_ITEM_LIMIT);
        for op in ops {
            op.subject = truncate_chars(
                std::mem::take(&mut op.subject),
                UI_ACTIVITY_SUMMARY_CHAR_LIMIT,
            );
        }
    }
    activity.result.raw = serde_json::Value::Null;
    let detail_lines = std::mem::take(&mut activity.result.detail_lines);
    activity.result.detail_lines = compact_strings(
        detail_lines,
        UI_ACTIVITY_DETAIL_LINE_LIMIT,
        UI_ACTIVITY_DETAIL_CHAR_LIMIT,
    );
    activity.result.artifact = activity.result.artifact.map(compact_activity_artifact);
    activity.children.truncate(UI_ACTIVITY_CHILD_LIMIT);
    activity.children = activity
        .children
        .into_iter()
        .map(compact_activity_for_replay)
        .collect();
    activity
}

fn compact_activity_artifact(artifact: ActivityArtifact) -> ActivityArtifact {
    match artifact {
        ActivityArtifact::QuestionPanel(mut panel) => {
            panel.prompt_lines = compact_strings(
                panel.prompt_lines,
                UI_ACTIVITY_DETAIL_LINE_LIMIT,
                UI_ACTIVITY_DETAIL_CHAR_LIMIT,
            );
            panel.options.truncate(UI_ACTIVITY_ARTIFACT_ITEM_LIMIT);
            for option in &mut panel.options {
                option.label = truncate_chars(
                    std::mem::take(&mut option.label),
                    UI_ACTIVITY_DETAIL_CHAR_LIMIT,
                );
            }
            panel.answer = panel
                .answer
                .map(|answer| truncate_chars(answer, UI_ACTIVITY_DETAIL_CHAR_LIMIT));
            panel.note = panel
                .note
                .map(|note| truncate_chars(note, UI_ACTIVITY_DETAIL_CHAR_LIMIT));
            ActivityArtifact::QuestionPanel(panel)
        }
        ActivityArtifact::DiffPreview { title, diff } => ActivityArtifact::DiffPreview {
            title: truncate_chars(title, UI_ACTIVITY_SUMMARY_CHAR_LIMIT),
            diff: truncate_chars(diff, UI_ACTIVITY_ARTIFACT_TEXT_CHAR_LIMIT),
        },
        ActivityArtifact::PatchPreview {
            mut files,
            total_added,
            total_removed,
        } => {
            files.truncate(UI_ACTIVITY_ARTIFACT_ITEM_LIMIT);
            for file in &mut files {
                file.path = truncate_chars(
                    std::mem::take(&mut file.path),
                    UI_ACTIVITY_SUMMARY_CHAR_LIMIT,
                );
                file.from_path = file
                    .from_path
                    .take()
                    .map(|path| truncate_chars(path, UI_ACTIVITY_SUMMARY_CHAR_LIMIT));
                file.status = truncate_chars(
                    std::mem::take(&mut file.status),
                    UI_ACTIVITY_SUMMARY_CHAR_LIMIT,
                );
                file.diff = truncate_chars(
                    std::mem::take(&mut file.diff),
                    UI_ACTIVITY_ARTIFACT_TEXT_CHAR_LIMIT,
                );
            }
            ActivityArtifact::PatchPreview {
                files,
                total_added,
                total_removed,
            }
        }
        ActivityArtifact::TextPreview { title, text } => ActivityArtifact::TextPreview {
            title: title.map(|title| truncate_chars(title, UI_ACTIVITY_SUMMARY_CHAR_LIMIT)),
            text: truncate_chars(text, UI_ACTIVITY_ARTIFACT_TEXT_CHAR_LIMIT),
        },
        ActivityArtifact::SourceList { title, items } => ActivityArtifact::SourceList {
            title: truncate_chars(title, UI_ACTIVITY_SUMMARY_CHAR_LIMIT),
            items: compact_strings(
                items,
                UI_ACTIVITY_ARTIFACT_ITEM_LIMIT,
                UI_ACTIVITY_DETAIL_CHAR_LIMIT,
            ),
        },
        ActivityArtifact::SnippetPreview(mut snippet) => {
            snippet.title = snippet
                .title
                .map(|title| truncate_chars(title, UI_ACTIVITY_SUMMARY_CHAR_LIMIT));
            snippet.path = truncate_chars(snippet.path, UI_ACTIVITY_SUMMARY_CHAR_LIMIT);
            snippet.content = truncate_chars(snippet.content, UI_ACTIVITY_ARTIFACT_TEXT_CHAR_LIMIT);
            snippet.language = snippet
                .language
                .map(|language| truncate_chars(language, UI_ACTIVITY_SUMMARY_CHAR_LIMIT));
            ActivityArtifact::SnippetPreview(snippet)
        }
    }
}

fn compact_strings(values: Vec<String>, item_limit: usize, char_limit: usize) -> Vec<String> {
    values
        .into_iter()
        .take(item_limit)
        .map(|value| truncate_chars(value, char_limit))
        .collect()
}

fn truncate_chars(value: String, limit: usize) -> String {
    let cut_at = {
        let mut chars = value.char_indices();
        for _ in 0..limit {
            if chars.next().is_none() {
                return value;
            }
        }
        chars.next().map(|(idx, _)| idx)
    };
    match cut_at {
        Some(idx) => value[..idx].to_string(),
        None => value,
    }
}

pub(crate) fn timeline_from_read_view(
    read_view: &lash_core::SessionReadView,
    ui_state: &UiProjectionState,
) -> UiTimeline {
    let projection = read_view.chronological_projection();
    let mut timeline = timeline_from_chronological(&projection);
    apply_activity_journal(&mut timeline, &ui_state.activity_journal);
    timeline.extend(
        ui_state
            .plugin_panels
            .iter()
            .cloned()
            .map(UiTimelineItem::PluginPanel),
    );
    timeline
}

fn apply_activity_journal(timeline: &mut UiTimeline, journal: &UiActivityJournal) {
    if journal.is_empty() {
        return;
    }
    let mut insertions = journal
        .entries()
        .iter()
        .filter(|entry| !entry.items.is_empty())
        .filter_map(|entry| {
            activity_journal_insert_position(timeline.items(), entry)
                .map(|position| (position, entry.items.clone()))
        })
        .collect::<Vec<_>>();
    insertions.sort_by_key(|(position, _)| *position);
    let mut offset = 0usize;
    for (position, items) in insertions {
        let position = position + offset;
        let item_count = items.len();
        timeline.items.splice(position..position, items);
        offset += item_count;
    }
}

fn activity_journal_insert_position(
    items: &[UiTimelineItem],
    entry: &UiActivityJournalEntry,
) -> Option<usize> {
    let turn_starts = user_turn_start_indices(items);
    let (turn_start, turn_end) = if turn_starts.is_empty() {
        if entry.turn_ordinal == 0 {
            (0, items.len())
        } else {
            return None;
        }
    } else {
        let turn_start = *turn_starts.get(entry.turn_ordinal)?;
        let turn_end = turn_starts
            .get(entry.turn_ordinal + 1)
            .copied()
            .unwrap_or(items.len());
        (turn_start, turn_end)
    };

    let mut seen_lashlang_blocks = 0usize;
    for (idx, item) in items[turn_start..turn_end].iter().enumerate() {
        if !matches!(item, UiTimelineItem::LashlangCode(_)) {
            continue;
        }
        if seen_lashlang_blocks == entry.lashlang_block_ordinal {
            return Some(turn_start + idx + 1);
        }
        seen_lashlang_blocks += 1;
    }

    Some(turn_end)
}

#[cfg(test)]
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
    let tool_call_map = HashMap::new();

    for entry in projection_transcript_entries(projection) {
        match entry.kind {
            TranscriptEntryKind::Message(message) => {
                append_transcript_items(
                    &mut timeline,
                    message,
                    &tool_call_map,
                    &mut activity_state,
                    false,
                );
            }
            TranscriptEntryKind::AssistantReasoning(text) => {
                let _ = push_assistant_reasoning_item(&mut timeline, &text);
            }
            TranscriptEntryKind::AssistantText(text) => {
                let _ = push_assistant_text_item(&mut timeline, &text);
            }
            TranscriptEntryKind::LashlangStep(step) => {
                append_lashlang_step_items(&mut timeline, &step);
            }
        }
    }

    timeline
}

fn append_lashlang_step_items(
    timeline: &mut UiTimeline,
    step: &lash_export::transcript::LashlangTranscriptStep,
) {
    timeline.push(UiTimelineItem::LashlangCode(step.code.clone()));
    // Mirror the live path (`CodeBlockCompleted`): a failed block keeps its
    // code on screen with the error rendered after it, so scrollback shows the
    // same thing the turn did.
    if let Some(error) = &step.error {
        timeline.push(UiTimelineItem::Error(error.clone()));
    }
    if let Some(final_output) = &step.final_output {
        let _ = push_assistant_text_item(timeline, &final_value_display_text(final_output));
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
        // RLM system nodes are protocol-loop prompts, such as finish reminders
        // injected after prose-only responses. They are instructions to the
        // model, not user-visible transcript content.
        MessageRole::System => true,
        MessageRole::User | MessageRole::Event => false,
    }
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

fn push_assistant_reasoning_item(timeline: &mut UiTimeline, text: &str) -> bool {
    let cleaned = normalize_assistant_text(text);
    if cleaned.is_empty() {
        return false;
    }
    timeline.push(UiTimelineItem::AssistantReasoning(cleaned));
    true
}

pub(crate) fn rendered_message_text(message: &Message) -> String {
    let text = message
        .parts
        .iter()
        .filter_map(|part| rendered_part_text(&part.kind, &part.content))
        .collect::<Vec<_>>()
        .join("\n\n")
        .trim()
        .to_string();
    if matches!(message.role, MessageRole::User) {
        strip_appended_skill_blocks(&text)
    } else {
        text
    }
}

fn rendered_part_text(kind: &PartKind, content: &str) -> Option<String> {
    match kind {
        PartKind::Reasoning | PartKind::ToolCall | PartKind::ToolResult => None,
        PartKind::Image => Some("[Image attached]".to_string()),
        _ => (!content.trim().is_empty()).then(|| content.to_string()),
    }
}
