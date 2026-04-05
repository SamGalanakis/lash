use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use lash::{
    AgentEvent, Message, MessageRole, PartKind, PluginMessage, PromptUsage, SkillCatalog,
    TokenUsage, ToolCallRecord, append_skill_blocks, collect_skill_mentions,
    plugin_surface_event_renders_visible_output,
};

use ratatui::layout::Rect;

use crate::activity::{
    ActivityBlock, ActivityKind, ActivityState, ActivityStatus, merge_edit_activity,
    merge_exploration_activity,
};
use crate::assistant_text::{normalize_assistant_text, push_assistant_text_block};
use crate::editor::EditorState;
use crate::overlay::{OverlayState, PickerState};
use crate::plugin_surface;
use crate::repo_status::RepoStatus;
use crate::ui;
use crate::util::{is_cancelled_error, manual_interrupt_message};

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PluginPanelBlock {
    pub plugin_id: String,
    pub key: String,
    pub title: String,
    pub content: String,
}

pub use crate::editor::{LargePaste, PendingImage, SuggestionKind};
pub use crate::overlay::PromptState;

pub struct LiveTurnState {
    pub status_text: String,
    pub status_detail: Option<String>,
    pub phase_started_at: std::time::Instant,
    pub turn_started_at: std::time::Instant,
    pub assistant_block_idx: Option<usize>,
    pub has_visible_output: bool,
    pub transient_until: Option<std::time::Instant>,
}

impl LiveTurnState {
    fn new(status_text: impl Into<String>, status_detail: Option<String>) -> Self {
        let now = std::time::Instant::now();
        Self {
            status_text: status_text.into(),
            status_detail,
            phase_started_at: now,
            turn_started_at: now,
            assistant_block_idx: None,
            has_visible_output: false,
            transient_until: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FollowOutputMode {
    Paused,
    Bottom,
    Contextual,
}

fn expand_large_paste_placeholders(text: &str, large_pastes: &[LargePaste]) -> String {
    let mut expanded = text.to_string();
    for paste in large_pastes {
        if expanded.contains(&paste.placeholder) {
            expanded = expanded.replace(&paste.placeholder, &paste.content);
        }
    }
    expanded
}

fn annotate_large_paste_placeholders(text: &str, large_pastes: &[LargePaste]) -> String {
    let mut annotated = text.to_string();
    for paste in large_pastes {
        if annotated.contains(&paste.placeholder) {
            let char_count = paste.content.chars().count();
            annotated =
                annotated.replace(&paste.placeholder, &format!("[pasted {char_count} chars]"));
        }
    }
    annotated
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedTurn {
    pub draft_id: String,
    pub raw_text: String,
    pub display_text: String,
    pub effective_text: String,
    pub images: Vec<PendingImage>,
    pub large_pastes: Vec<LargePaste>,
    pub transform_labels: Vec<String>,
}

impl PreparedTurn {
    #[cfg(test)]
    pub fn new(text: String, images: Vec<Vec<u8>>) -> Self {
        Self {
            draft_id: uuid::Uuid::new_v4().to_string(),
            raw_text: text.clone(),
            display_text: text.clone(),
            effective_text: text,
            images: images
                .into_iter()
                .enumerate()
                .map(|(idx, png_bytes)| PendingImage {
                    id: idx + 1,
                    png_bytes,
                })
                .collect(),
            large_pastes: Vec::new(),
            transform_labels: Vec::new(),
        }
    }

    pub fn prepare(text: String, images: Vec<PendingImage>, skills: &SkillCatalog) -> Self {
        Self::prepare_with_large_pastes(text, images, skills, Vec::new())
    }

    pub fn prepare_with_large_pastes(
        text: String,
        images: Vec<PendingImage>,
        skills: &SkillCatalog,
        large_pastes: Vec<LargePaste>,
    ) -> Self {
        let mut labels = Vec::new();
        let mut seen = HashSet::new();
        for name in collect_skill_mentions(&text) {
            if seen.insert(name.clone()) && skills.get(&name).is_some() {
                labels.push(name);
            }
        }
        let large_pastes = large_pastes
            .into_iter()
            .filter(|paste| text.contains(&paste.placeholder))
            .collect::<Vec<_>>();
        let expanded_text = expand_large_paste_placeholders(&text, &large_pastes);
        let effective_text = append_skill_blocks(&expanded_text, skills);
        Self {
            draft_id: uuid::Uuid::new_v4().to_string(),
            raw_text: text.clone(),
            display_text: text,
            effective_text,
            images,
            large_pastes,
            transform_labels: labels,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.display_text.is_empty() && self.images.is_empty()
    }

    pub fn preview(&self) -> String {
        let collapsed = self.display_text.replace('\n', " ").trim().to_string();
        if !collapsed.is_empty() {
            return collapsed;
        }
        match self.images.len() {
            0 => String::new(),
            1 => "[1 image]".to_string(),
            n => format!("[{} images]", n),
        }
    }

    pub fn history_text(&self) -> String {
        let annotated = annotate_large_paste_placeholders(&self.raw_text, &self.large_pastes);
        if annotated.is_empty() {
            String::new()
        } else {
            preview_text_lines(&annotated).join("\n")
        }
    }
}

/// A renderable block in the scrollable history.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DisplayBlock {
    UserInput(String),
    AssistantText(String),
    CodeBlock {
        code: String,
        /// If true, this block is a continuation of a prior code-block group.
        continuation: bool,
    },
    Activity(ActivityBlock),
    CodeOutput {
        output: String,
        error: Option<String>,
    },
    ShellOutput {
        command: String,
        output: String,
        error: Option<String>,
    },
    Error(String),
    /// Informational message from the system (e.g. /help output).
    SystemMessage(String),
    /// Rendered plan content from update_plan (bordered markdown).
    PlanContent(String),
    PluginPanel(PluginPanelBlock),
    Splash,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct UiResumeState {
    #[serde(default)]
    pub last_response_usage: TokenUsage,
    #[serde(default)]
    pub plugin_mode_indicators: BTreeMap<String, String>,
    #[serde(default)]
    pub plugin_panels: Vec<PluginPanelBlock>,
    #[serde(default)]
    pub streaming_output: Vec<String>,
    #[serde(default)]
    pub streaming_output_hidden: usize,
    #[serde(default)]
    pub streaming_output_partial: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub live_assistant_text: Option<String>,
}

impl UiResumeState {
    pub fn from_app(app: &App) -> Self {
        Self {
            last_response_usage: app.last_response_usage.clone(),
            plugin_mode_indicators: app.plugin_mode_indicators.clone(),
            plugin_panels: app
                .blocks
                .iter()
                .filter_map(|block| match block {
                    DisplayBlock::PluginPanel(panel) => Some(panel.clone()),
                    _ => None,
                })
                .collect(),
            streaming_output: app.streaming_output.clone(),
            streaming_output_hidden: app.streaming_output_hidden,
            streaming_output_partial: app.streaming_output_partial.clone(),
            live_assistant_text: app
                .live_turn
                .as_ref()
                .and_then(|turn| turn.assistant_block_idx)
                .and_then(|idx| match app.blocks.get(idx) {
                    Some(DisplayBlock::AssistantText(text)) if !text.is_empty() => {
                        Some(text.clone())
                    }
                    _ => None,
                }),
        }
    }
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

pub(crate) fn apply_ui_resume_state_to_blocks(
    blocks: &mut Vec<DisplayBlock>,
    ui_state: &UiResumeState,
) {
    if let Some(text) = ui_state.live_assistant_text.as_deref() {
        let cleaned = normalize_assistant_text(text);
        if !cleaned.is_empty() {
            if let Some(DisplayBlock::AssistantText(existing)) = blocks.last_mut() {
                if cleaned.starts_with(existing.as_str()) {
                    *existing = cleaned;
                } else {
                    let _ = push_assistant_text_block(blocks, &cleaned);
                }
            } else {
                let _ = push_assistant_text_block(blocks, &cleaned);
            }
        }
    }

    blocks.extend(
        ui_state
            .plugin_panels
            .iter()
            .cloned()
            .map(DisplayBlock::PluginPanel),
    );
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
                let text = rendered_message_text(message);
                if !text.is_empty() {
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
                    PartKind::Code => {
                        flush_assistant_prose(blocks, &mut prose);
                        blocks.push(DisplayBlock::CodeBlock {
                            code: text,
                            continuation: false,
                        });
                    }
                    PartKind::Output => {
                        flush_assistant_prose(blocks, &mut prose);
                        blocks.push(DisplayBlock::CodeOutput {
                            output: text,
                            error: None,
                        });
                    }
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

    let plan_content = if record.success && record.tool == "update_plan" {
        render_plan_content_from_args(&record.args)
    } else {
        None
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

    if let Some(content) = plan_content {
        blocks.push(DisplayBlock::PlanContent(content));
    }
}

fn append_activity_block(blocks: &mut Vec<DisplayBlock>, activity: ActivityBlock) {
    if let Some(DisplayBlock::Activity(existing)) = blocks.last_mut()
        && existing.kind == ActivityKind::Exploration
        && activity.kind == ActivityKind::Exploration
        && existing.status == ActivityStatus::Completed
        && activity.status == ActivityStatus::Completed
        && merge_exploration_activity(existing, activity.clone())
    {
        return;
    }
    if let Some(DisplayBlock::Activity(existing)) = blocks.last_mut()
        && existing.kind == ActivityKind::Edit
        && activity.kind == ActivityKind::Edit
        && existing.status == ActivityStatus::Completed
        && activity.status == ActivityStatus::Completed
        && merge_edit_activity(existing, activity.clone())
    {
        return;
    }
    blocks.push(DisplayBlock::Activity(activity));
}

fn flush_assistant_prose(blocks: &mut Vec<DisplayBlock>, prose: &mut Vec<String>) {
    if prose.is_empty() {
        return;
    }
    let text = prose.join("\n\n");
    let _ = push_assistant_text_block(blocks, &text);
    prose.clear();
}

fn rendered_message_text(message: &Message) -> String {
    message
        .parts
        .iter()
        .filter_map(|part| rendered_part_text(&part.kind, &part.content))
        .collect::<Vec<_>>()
        .join("\n\n")
        .trim()
        .to_string()
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

fn rendered_part_text(kind: &PartKind, content: &str) -> Option<String> {
    match kind {
        PartKind::ToolCall | PartKind::ToolResult => None,
        PartKind::Image => Some("[Image attached]".to_string()),
        _ => (!content.trim().is_empty()).then(|| content.to_string()),
    }
}

pub(crate) const SPLASH_CONTENT_HEIGHT: usize = 8;
pub(crate) const SPLASH_SCROLLBACK_HEIGHT: usize = SPLASH_CONTENT_HEIGHT + 2;

const TEXT_PREVIEW_MAX_HEAD_LINES: usize = 8;
const TEXT_PREVIEW_MAX_TAIL_LINES: usize = 3;
const TEXT_PREVIEW_LINE_CHAR_LIMIT: usize = 240;
const STREAMING_OUTPUT_MAX_LINES: usize = 48;
const STREAMING_OUTPUT_LINE_CHAR_LIMIT: usize = 240;
const FOLLOW_OUTPUT_CONTEXT_LINES: usize = 2;

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

fn smart_truncate_preview_line(text: &str, max_chars: usize) -> String {
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

/// Fast, coarse token estimate used only for live UI counters while streaming.
fn estimate_tokens_from_char_count(chars: i64) -> i64 {
    if chars <= 0 { 0 } else { (chars + 3) / 4 }
}

#[cfg(test)]
impl DisplayBlock {
    pub fn height(&self, expand_level: u8, width: usize, viewport_height: usize) -> usize {
        let blocks = [self.clone()];
        ui::rendered_block_height(&blocks, 0, expand_level, width, viewport_height)
    }
}

/// Mouse text selection state for the history viewport.
///
/// Coordinates are in **content space**: (column, virtual_row) where
/// `virtual_row = screen_row - history_area.y + scroll_offset`.
/// This makes the selection stable across scrolling.
#[derive(Clone, Debug, Default)]
pub struct TextSelection {
    /// Anchor point (where the drag started) in content-space (col, virtual_row).
    pub anchor: (u16, usize),
    /// Current end point of the selection in content-space (col, virtual_row).
    pub end: (u16, usize),
    /// Whether a drag is in progress.
    pub active: bool,
    /// Whether a completed selection is being displayed.
    pub visible: bool,
}

pub struct App {
    pub blocks: Vec<DisplayBlock>,
    pub scroll_offset: usize,
    pub expand_level: u8,
    pub running: bool,
    pub model: String,
    pub iteration: usize,
    /// Spinner frame counter
    pub tick: usize,
    /// Active live turn state for the bottom status strip.
    pub live_turn: Option<LiveTurnState>,
    /// Raw TextDelta buffer for the active streamed assistant segment.
    pub pending_text: String,
    /// Ignore stray late TextDelta events once the latest assistant block has
    /// been reconciled to authoritative final text.
    assistant_text_finalized: bool,
    /// Whether the UI needs a redraw.
    pub dirty: bool,
    /// Output-following mode for the history viewport.
    pub follow_mode: FollowOutputMode,
    /// Cumulative height prefix sums for each block (used for O(1) total height and O(log n) lookup).
    height_cache: Vec<usize>,
    /// Terminal width the height cache was computed for.
    height_cache_width: usize,
    /// Viewport height the height cache was computed for (needed for Splash centering).
    height_cache_vh: usize,
    /// Owned editor/input state.
    pub editor: EditorState,
    /// Live streaming output lines from tool execution (e.g. bash).
    pub streaming_output: Vec<String>,
    pub streaming_output_hidden: usize,
    pub streaming_output_partial: String,
    /// Whether to render live `tool_output` chunks in history.
    /// Default: on (can be disabled via `LASH_SHOW_TOOL_OUTPUT=0`).
    pub show_live_tool_output: bool,
    /// Loaded skills registry.
    pub skills: SkillCatalog,
    /// Priority follow-ups entered with Enter while a turn is running.
    pub pending_steers: VecDeque<PreparedTurn>,
    /// FIFO drafts explicitly queued for later turns.
    pub queued_turns: VecDeque<PreparedTurn>,
    /// Active overlay/picker/dialog state.
    pub overlay: Option<OverlayState>,
    /// Whether the terminal window is currently focused.
    pub focused: bool,
    /// Cumulative token usage for the current session.
    pub token_usage: TokenUsage,
    /// Context window size for the current model (from models.dev).
    pub context_window: Option<u64>,
    /// Whether provider-reported input tokens exclude cached prompt tokens.
    pub context_usage_excludes_cached_input: bool,
    /// Active provider-native variant for the current model, if any.
    pub model_variant: Option<String>,
    /// Latest completed model usage for context accounting.
    pub last_response_usage: TokenUsage,
    /// Latest normalized prompt-budget usage for context accounting and folding.
    pub last_prompt_usage: Option<PromptUsage>,
    /// Estimated output character count from live streaming chunks.
    pub live_output_chars_estimate: i64,
    /// Estimated output tokens from live streamed chunks before final usage arrives.
    pub live_output_tokens_estimate: i64,
    /// Unique session name (e.g. "alpine-canyon").
    pub session_name: String,
    /// Repo/branch/worktree metadata for the current cwd, when available.
    pub repo_status: Option<RepoStatus>,
    /// Active plugin-owned mode indicators rendered in the input chrome.
    pub plugin_mode_indicators: BTreeMap<String, String>,
    /// Current working directory with ~ substitution.
    pub cwd: String,
    /// Active delegate child session: (name, task description, started_at).
    pub active_delegate: Option<(String, String, std::time::Instant)>,
    /// Handle state used to derive semantic activity rows from raw tool calls.
    pub activity_state: ActivityState,
    /// Set only when this local UI requested cancellation via Esc.
    manual_interrupt_requested: bool,
    /// Current text selection state.
    pub selection: TextSelection,
    /// Cached history area rect from the last draw, used to map mouse coords.
    pub history_area: Rect,
}

impl App {
    pub fn ui_resume_state(&self) -> UiResumeState {
        UiResumeState::from_app(self)
    }

    pub fn finish_turn_for_resume_with_output(
        &mut self,
        final_assistant_text: Option<&str>,
    ) -> UiResumeState {
        if let Some(text) = final_assistant_text {
            self.commit_final_assistant_text(text);
        }
        let persisted = self.ui_resume_state();
        self.stop_turn();
        persisted
    }

    fn ensure_live_turn(&mut self) -> &mut LiveTurnState {
        self.live_turn
            .get_or_insert_with(|| LiveTurnState::new("starting", None))
    }

    pub fn start_turn(&mut self) {
        self.running = true;
        self.manual_interrupt_requested = false;
        self.iteration = 0;
        self.pending_text.clear();
        self.assistant_text_finalized = false;
        self.clear_streaming_output();
        self.active_delegate = None;
        self.live_output_chars_estimate = 0;
        self.live_output_tokens_estimate = 0;
        self.live_turn = Some(LiveTurnState::new("starting", None));
        self.follow_mode = FollowOutputMode::Contextual;
    }

    pub fn stop_turn(&mut self) {
        self.running = false;
        self.manual_interrupt_requested = false;
        self.pending_text.clear();
        self.clear_streaming_output();
        self.active_delegate = None;
        self.live_output_chars_estimate = 0;
        self.live_output_tokens_estimate = 0;
        if self.follow_mode == FollowOutputMode::Contextual {
            self.follow_mode = FollowOutputMode::Bottom;
        }
        if self
            .live_turn
            .as_ref()
            .and_then(|turn| turn.transient_until)
            .is_none_or(|until| until <= std::time::Instant::now())
        {
            self.live_turn = None;
        }
    }

    fn set_status(
        &mut self,
        header: impl Into<String>,
        details: Option<String>,
        reset_timer: bool,
    ) {
        let header = header.into();
        let turn = self.ensure_live_turn();
        let changed = turn.status_text != header || turn.status_detail != details;
        turn.status_text = header;
        turn.status_detail = details;
        turn.transient_until = None;
        if changed || reset_timer {
            turn.phase_started_at = std::time::Instant::now();
        }
    }

    fn set_transient_status(
        &mut self,
        header: impl Into<String>,
        details: Option<String>,
        duration: std::time::Duration,
    ) {
        let header = header.into();
        let now = std::time::Instant::now();
        let turn = self.ensure_live_turn();
        turn.status_text = header;
        turn.status_detail = details;
        turn.phase_started_at = now;
        turn.transient_until = Some(now + duration);
    }

    fn clear_status(&mut self) {
        self.manual_interrupt_requested = false;
        self.live_turn = None;
    }

    pub fn note_manual_interrupt_requested(&mut self) {
        self.manual_interrupt_requested = true;
    }

    pub fn on_tick(&mut self) {
        if self.running {
            self.tick += 1;
            self.dirty = true;
        }

        if self.live_turn.as_ref().is_some_and(|turn| {
            turn.transient_until
                .is_some_and(|until| until <= std::time::Instant::now())
        }) {
            self.live_turn = None;
            self.dirty = true;
            return;
        }

        if self
            .live_turn
            .as_ref()
            .is_some_and(|turn| turn.transient_until.is_some())
        {
            self.dirty = true;
        }
    }

    fn mark_first_token_arrived(&mut self) {
        if self.live_turn.as_ref().is_some_and(|turn| {
            turn.status_text == "thinking"
                && turn.status_detail.as_deref() == Some("waiting for first token")
        }) {
            self.set_status("responding", None, true);
        }
    }

    fn mark_visible_output(&mut self) {
        if let Some(turn) = self.live_turn.as_mut() {
            turn.has_visible_output = true;
        }
    }

    fn push_activity_block(&mut self, activity: ActivityBlock) {
        append_activity_block(&mut self.blocks, activity);
        self.invalidate_height_cache();
    }

    fn push_plan_content(&mut self, content: String) {
        self.blocks.push(DisplayBlock::PlanContent(content));
        self.invalidate_height_cache();
    }

    pub fn new(model: String, session_name: String) -> Self {
        let cwd = {
            let home = std::env::var("HOME").unwrap_or_default();
            let dir = std::env::current_dir()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| ".".into());
            if !home.is_empty() && dir.starts_with(&home) {
                format!("~{}", &dir[home.len()..])
            } else {
                dir
            }
        };
        Self {
            blocks: vec![DisplayBlock::Splash],
            scroll_offset: 0,
            expand_level: 1,
            running: false,
            model,
            iteration: 0,
            tick: 0,
            live_turn: None,
            pending_text: String::new(),
            assistant_text_finalized: false,
            dirty: true,
            follow_mode: FollowOutputMode::Bottom,
            height_cache: Vec::new(),
            height_cache_width: 0,
            height_cache_vh: 0,
            editor: EditorState::default(),
            streaming_output: Vec::new(),
            streaming_output_hidden: 0,
            streaming_output_partial: String::new(),
            show_live_tool_output: !matches!(
                std::env::var("LASH_SHOW_TOOL_OUTPUT")
                    .unwrap_or_default()
                    .to_ascii_lowercase()
                    .as_str(),
                "0" | "false" | "no" | "off"
            ),
            skills: SkillCatalog::load(),
            pending_steers: VecDeque::new(),
            queued_turns: VecDeque::new(),
            overlay: None,
            focused: true,
            token_usage: TokenUsage::default(),
            context_window: None,
            context_usage_excludes_cached_input: false,
            model_variant: None,
            last_response_usage: TokenUsage::default(),
            last_prompt_usage: None,
            live_output_chars_estimate: 0,
            live_output_tokens_estimate: 0,
            session_name,
            repo_status: std::env::current_dir()
                .ok()
                .and_then(|cwd| crate::repo_status::detect_repo_status(&cwd)),
            plugin_mode_indicators: BTreeMap::new(),
            cwd,
            active_delegate: None,
            activity_state: ActivityState::default(),
            manual_interrupt_requested: false,
            selection: TextSelection::default(),
            history_area: Rect::default(),
        }
    }

    /// Clear any active text selection.
    pub fn clear_selection(&mut self) {
        self.selection = TextSelection::default();
    }

    /// Get the current input text and reset input state.
    pub fn take_input(&mut self) -> String {
        let text = self.editor.take_input();
        self.follow_mode = FollowOutputMode::Bottom;
        text
    }

    pub fn take_prepared_turn(&mut self) -> PreparedTurn {
        let input = self.take_input();
        let images = self.take_pending_images();
        let large_pastes = self.take_large_pastes();
        PreparedTurn::prepare_with_large_pastes(input, images, &self.skills, large_pastes)
    }

    /// Take pending images, preserving their stable inline ids for marker parsing.
    pub fn take_pending_images(&mut self) -> Vec<PendingImage> {
        self.editor.take_pending_images()
    }

    /// Take pending large-paste payloads, clearing them from the draft.
    pub fn take_large_pastes(&mut self) -> Vec<LargePaste> {
        self.editor.take_large_pastes()
    }

    /// Mark the height cache as stale so it will be recomputed on next access.
    pub fn invalidate_height_cache(&mut self) {
        self.height_cache.clear();
    }

    /// Remove the empty-state splash once real conversation content is present.
    #[cfg(test)]
    pub fn dismiss_splash(&mut self) {
        if !matches!(self.blocks.first(), Some(DisplayBlock::Splash)) {
            return;
        }
        self.blocks
            .retain(|block| !matches!(block, DisplayBlock::Splash));
        self.invalidate_height_cache();
    }

    /// Check whether a new CodeBlock belongs to an existing code-block group.
    /// Returns `true` if there is a prior CodeBlock with only Activity / CodeOutput
    /// blocks between it and the end (no user-facing boundary like AssistantText,
    /// Error, UserInput, etc.).
    fn is_code_continuation(&self) -> bool {
        for block in self.blocks.iter().rev() {
            match block {
                DisplayBlock::CodeBlock { .. } => return true,
                DisplayBlock::Activity(_) | DisplayBlock::CodeOutput { .. } => continue,
                _ => return false,
            }
        }
        false
    }

    fn sync_pending_text_block(&mut self) {
        let cleaned = normalize_assistant_text(&self.pending_text);
        if cleaned.is_empty() {
            return;
        }

        let active_idx = self
            .live_turn
            .as_ref()
            .and_then(|turn| turn.assistant_block_idx);
        if let Some(idx) = active_idx
            && let Some(DisplayBlock::AssistantText(text)) = self.blocks.get_mut(idx)
        {
            if *text != cleaned {
                *text = cleaned;
                self.invalidate_height_cache();
            }
            self.mark_visible_output();
            return;
        }

        self.blocks.push(DisplayBlock::AssistantText(cleaned));
        if let Some(turn) = self.live_turn.as_mut() {
            turn.assistant_block_idx = Some(self.blocks.len() - 1);
        }
        self.mark_visible_output();
        self.invalidate_height_cache();
    }

    fn merge_into_trailing_assistant_block(&mut self, text: &str) -> bool {
        let Some(DisplayBlock::AssistantText(existing)) = self.blocks.last_mut() else {
            return false;
        };
        let combined = if text.starts_with(existing.as_str()) {
            text.to_string()
        } else {
            format!("{existing}{text}")
        };
        let cleaned = normalize_assistant_text(&combined);
        if cleaned.is_empty() {
            return false;
        }
        if *existing != cleaned {
            *existing = cleaned;
            self.invalidate_height_cache();
        }
        true
    }

    fn reconcile_trailing_assistant_block(&mut self, text: &str) -> bool {
        let Some(DisplayBlock::AssistantText(existing)) = self.blocks.last_mut() else {
            return false;
        };
        if text.starts_with(existing.as_str()) {
            if *existing != text {
                *existing = text.to_string();
                self.invalidate_height_cache();
            }
            return true;
        }
        if existing.is_empty() || existing.starts_with(text) {
            return true;
        }
        false
    }

    fn close_pending_text(&mut self) {
        if let Some(turn) = self.live_turn.as_mut() {
            turn.assistant_block_idx = None;
        }
        self.pending_text.clear();
    }

    fn commit_final_assistant_text(&mut self, text: &str) {
        let cleaned = normalize_assistant_text(text);
        if cleaned.is_empty() {
            self.close_pending_text();
            return;
        }

        let active_idx = self
            .live_turn
            .as_ref()
            .and_then(|turn| turn.assistant_block_idx);
        if let Some(idx) = active_idx
            && let Some(DisplayBlock::AssistantText(existing)) = self.blocks.get_mut(idx)
        {
            if cleaned.starts_with(existing.as_str()) {
                if *existing != cleaned {
                    *existing = cleaned;
                    self.invalidate_height_cache();
                }
            } else if existing.is_empty() {
                *existing = cleaned;
                self.invalidate_height_cache();
            }
            self.assistant_text_finalized = true;
            self.mark_visible_output();
            self.close_pending_text();
            return;
        }

        if self.reconcile_trailing_assistant_block(&cleaned) {
            self.assistant_text_finalized = true;
            self.mark_visible_output();
            self.close_pending_text();
            return;
        }

        if push_assistant_text_block(&mut self.blocks, &cleaned) {
            self.invalidate_height_cache();
            self.mark_visible_output();
        }
        self.assistant_text_finalized = true;
        self.close_pending_text();
    }

    fn commit_injected_messages(&mut self, messages: &[PluginMessage]) {
        self.close_pending_text();
        let mut committed_user_message = false;
        for message in messages {
            match message.role {
                MessageRole::User => {
                    committed_user_message = true;
                    if let Some(turn) = self.pending_steers.pop_front() {
                        self.push_prepared_user_input(&turn);
                    } else if !self.commit_pending_user_preview(&message.content) {
                        self.blocks
                            .push(DisplayBlock::UserInput(message.content.clone()));
                    }
                }
                MessageRole::System => {
                    self.blocks
                        .push(DisplayBlock::SystemMessage(message.content.clone()));
                }
                _ => continue,
            }
        }
        if !messages.is_empty() {
            self.invalidate_height_cache();
            if committed_user_message {
                self.keep_latest_user_block_visible();
            } else {
                self.scroll_to_bottom();
            }
        }
    }

    fn queue_plan_exit_follow_up(&mut self, result: &serde_json::Value, success: bool) {
        if !success {
            return;
        }
        let Some(next_turn_input) = result
            .get("next_turn_input")
            .and_then(|value| value.as_str())
            .filter(|value| !value.trim().is_empty())
        else {
            return;
        };
        self.queue_turn(PreparedTurn::prepare(
            next_turn_input.to_string(),
            Vec::new(),
            &self.skills,
        ));
    }

    pub fn push_prepared_user_input(&mut self, turn: &PreparedTurn) {
        let history_text = turn.history_text();
        if history_text.is_empty() {
            return;
        }
        self.blocks.push(DisplayBlock::UserInput(history_text));
        self.invalidate_height_cache();
    }

    /// Process an agent event, updating display blocks.
    pub fn handle_agent_event(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::TextDelta { content } => {
                if !self.running && self.live_turn.is_none() && self.pending_text.is_empty() {
                    if self.assistant_text_finalized {
                        return;
                    }
                    if self.merge_into_trailing_assistant_block(&content) {
                        self.scroll_to_bottom();
                        return;
                    }
                }
                self.mark_first_token_arrived();
                self.live_output_chars_estimate += content.chars().count() as i64;
                self.live_output_tokens_estimate =
                    estimate_tokens_from_char_count(self.live_output_chars_estimate);
                self.pending_text.push_str(&content);
                self.sync_pending_text_block();
                self.scroll_to_bottom();
            }
            AgentEvent::CodeBlock { code } => {
                self.set_status("writing code", None, true);
                self.close_pending_text();
                let trimmed = code.trim_matches('\n');
                if !trimmed.is_empty() {
                    let continuation = self.is_code_continuation();
                    self.blocks.push(DisplayBlock::CodeBlock {
                        code: trimmed.to_string(),
                        continuation,
                    });
                    self.invalidate_height_cache();
                    self.mark_visible_output();
                }
                self.scroll_to_bottom();
            }
            AgentEvent::ToolCall {
                name,
                args,
                result,
                success,
                duration_ms,
                ..
            } => {
                self.close_pending_text();
                self.clear_streaming_output();
                if name == "plan_exit" {
                    self.queue_plan_exit_follow_up(&result, success);
                }
                if matches!(name.as_str(), "agent_result" | "agent_kill") {
                    self.active_delegate = None;
                }
                let plan_content = if success && name == "update_plan" {
                    render_plan_content_from_args(&args)
                } else {
                    None
                };
                let activities = self.activity_state.blocks_for_tool_call(
                    &name,
                    args,
                    result,
                    success,
                    duration_ms,
                );
                if let Some(activity) = activities.last() {
                    let detail = activity.detail_lines.first().cloned();
                    self.set_status(activity.summary.clone(), detail, true);
                } else {
                    self.set_status(name.clone(), None, true);
                }
                for activity in activities {
                    self.push_activity_block(activity);
                }
                if let Some(content) = plan_content {
                    self.push_plan_content(content);
                }
                if !matches!(self.blocks.last(), Some(DisplayBlock::Splash)) {
                    self.mark_visible_output();
                }
                self.scroll_to_bottom();
            }
            AgentEvent::CodeOutput { output, error } => {
                let error = error.filter(|value| !value.trim().is_empty());
                if error.is_some() {
                    self.set_status("execution failed", None, true);
                }
                if error.is_some() || !output.is_empty() {
                    self.blocks.push(DisplayBlock::CodeOutput { output, error });
                    self.invalidate_height_cache();
                    self.mark_visible_output();
                }
                self.scroll_to_bottom();
            }
            AgentEvent::Message { text, kind } => {
                if kind == "delegate_start" {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
                        let model = v
                            .get("model")
                            .and_then(|value| value.as_str())
                            .unwrap_or_default();
                        let variant = v
                            .get("model_variant")
                            .and_then(|value| value.as_str())
                            .unwrap_or_default();
                        let name = v
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("delegate")
                            .to_string();
                        let task = v
                            .get("task")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        self.active_delegate = Some((name, task, std::time::Instant::now()));
                        let model_detail = if model.is_empty() {
                            None
                        } else if variant.is_empty() {
                            Some(model.to_string())
                        } else {
                            Some(format!("{model} ({variant})"))
                        };
                        let details = match (
                            self.active_delegate
                                .as_ref()
                                .map(|(_, task, _)| task.clone())
                                .filter(|value| !value.is_empty()),
                            model_detail,
                        ) {
                            (Some(task), Some(model)) => Some(format!("{task} · {model}")),
                            (Some(task), None) => Some(task),
                            (None, Some(model)) => Some(model),
                            (None, None) => Some("delegate".to_string()),
                        };
                        self.set_status("delegating", details, true);
                    }
                    self.scroll_to_bottom();
                } else if kind == "tool_output" {
                    // Explicit policy:
                    // - live tool output can be disabled via env var
                    // - shell + delegate result streams can render text to the TUI
                    let current_status = self
                        .live_turn
                        .as_ref()
                        .map(|turn| turn.status_text.as_str());
                    let is_delegate_stream =
                        matches!(current_status, Some("delegating") | Some("delegate"));
                    if is_delegate_stream {
                        self.live_output_chars_estimate += text.chars().count() as i64;
                        self.live_output_tokens_estimate =
                            estimate_tokens_from_char_count(self.live_output_chars_estimate);
                    }
                    let stream_active = self.running
                        || self.active_delegate.is_some()
                        || current_status.is_some_and(|status| status.contains("shell"));
                    if self.show_live_tool_output && stream_active {
                        self.push_streaming_output_text(&text);
                        self.mark_visible_output();
                        self.scroll_to_bottom();
                    }
                } else if kind == "final" {
                    self.commit_final_assistant_text(&text);
                    self.scroll_to_bottom();
                } else {
                    // Unknown message kinds are intentionally dropped.
                }
            }
            AgentEvent::LlmRequest { iteration, .. } => {
                self.close_pending_text();
                self.iteration = iteration + 1;
                self.set_status("thinking", Some("waiting for first token".into()), true);
                self.live_output_chars_estimate = 0;
                self.live_output_tokens_estimate = 0;
                self.keep_latest_user_block_visible();
            }
            AgentEvent::RetryStatus {
                wait_seconds,
                attempt,
                max_attempts,
                reason,
                envelope: _,
            } => {
                let mut reason_short: String = reason.chars().take(60).collect();
                if reason.chars().count() > 60 {
                    reason_short.push_str("...");
                }
                self.set_status(
                    "retrying",
                    Some(format!(
                        "in {}s · attempt {}/{} · {}",
                        wait_seconds, attempt, max_attempts, reason_short
                    )),
                    true,
                );
                self.scroll_to_bottom();
            }
            AgentEvent::Done => {
                self.close_pending_text();
                self.stop_turn();
                self.scroll_to_bottom();
            }
            AgentEvent::Error { message, envelope } => {
                self.close_pending_text();
                let code = envelope.as_ref().and_then(|err| err.code.as_deref());
                if self.manual_interrupt_requested && is_cancelled_error(&message, code) {
                    self.manual_interrupt_requested = false;
                    self.blocks.push(DisplayBlock::SystemMessage(
                        manual_interrupt_message().to_string(),
                    ));
                } else {
                    self.manual_interrupt_requested = false;
                    self.set_transient_status(
                        "error",
                        Some(message.chars().take(96).collect()),
                        std::time::Duration::from_secs(8),
                    );
                    self.blocks.push(DisplayBlock::Error(message));
                }
                self.invalidate_height_cache();
                self.mark_visible_output();
                self.scroll_to_bottom();
            }
            AgentEvent::TokenUsage {
                usage, cumulative, ..
            } => {
                let should_clear_live_estimate = usage.output_tokens > 0
                    || usage.reasoning_tokens > 0
                    || usage.cached_input_tokens > 0;
                self.last_response_usage = usage.clone();
                self.token_usage = cumulative;
                if should_clear_live_estimate {
                    self.live_output_chars_estimate = 0;
                    self.live_output_tokens_estimate = 0;
                }
            }
            AgentEvent::PluginEvent { plugin_id, event } => {
                let renders_visible_output = plugin_surface_event_renders_visible_output(&event);
                let mutation = plugin_surface::apply_surface_event(
                    &mut self.blocks,
                    &mut self.plugin_mode_indicators,
                    &plugin_id,
                    event,
                );
                if mutation.blocks_changed {
                    self.invalidate_height_cache();
                    if renders_visible_output {
                        self.mark_visible_output();
                    }
                    self.scroll_to_bottom();
                }
                if mutation.indicators_changed {
                    self.dirty = true;
                }
            }
            AgentEvent::InjectedMessagesCommitted { messages, .. } => {
                self.commit_injected_messages(&messages);
            }
            AgentEvent::DurableSnapshot { .. } => {}
            AgentEvent::LlmResponse { .. } => {}
            AgentEvent::Prompt { .. } => {
                // Handled by the main event loop, not here
            }
        }
    }

    pub fn queue_pending_steer(&mut self, turn: PreparedTurn) {
        if turn.is_empty() {
            return;
        }
        self.pending_steers.push_back(turn);
    }

    pub fn queue_turn(&mut self, turn: PreparedTurn) {
        if turn.is_empty() {
            return;
        }
        self.queued_turns.push_back(turn);
    }

    pub fn requeue_front(&mut self, turn: PreparedTurn, pending: bool) {
        if turn.is_empty() {
            return;
        }
        if pending {
            self.pending_steers.push_front(turn);
        } else {
            self.queued_turns.push_front(turn);
        }
    }

    pub fn take_next_queued_turn(&mut self) -> Option<(PreparedTurn, bool)> {
        self.queued_turns.pop_front().map(|turn| (turn, false))
    }

    pub fn take_last_queued_turn(&mut self) -> Option<(PreparedTurn, bool)> {
        self.queued_turns.pop_back().map(|turn| (turn, false))
    }

    pub fn has_queued_messages(&self) -> bool {
        !self.pending_steers.is_empty() || !self.queued_turns.is_empty()
    }

    pub fn preview_queued_turn(&mut self, turn: &PreparedTurn, inject_at_checkpoint: bool) {
        let _ = inject_at_checkpoint;
        let _ = turn;
        // Queued turns already render in the dedicated queue-preview panel. Duplicating
        // them into history makes the same text appear twice before it is actually sent.
    }

    pub fn commit_pending_user_preview(&mut self, text: &str) -> bool {
        let _ = text;
        false
    }

    pub fn set_plan_mode_enabled(&mut self, enabled: bool) {
        let key = plugin_surface::surface_key("plan_mode", "mode");
        if enabled {
            self.plugin_mode_indicators.insert(key, "plan".to_string());
        } else {
            self.plugin_mode_indicators.remove(&key);
        }
        self.dirty = true;
    }

    pub fn set_model_variant(&mut self, model_variant: Option<String>) {
        self.model_variant = model_variant;
        self.dirty = true;
    }

    /// Reset conversation to initial splash screen.
    pub fn clear(&mut self) {
        self.blocks = vec![DisplayBlock::Splash];
        self.scroll_offset = 0;
        self.follow_mode = FollowOutputMode::Bottom;
        self.pending_text.clear();
        self.assistant_text_finalized = false;
        self.clear_status();
        self.editor.pending_images.clear();
        self.editor.pending_large_pastes.clear();
        self.clear_streaming_output();
        self.pending_steers.clear();
        self.queued_turns.clear();
        self.active_delegate = None;
        self.activity_state.reset();
        self.token_usage = TokenUsage::default();
        self.last_response_usage = TokenUsage::default();
        self.last_prompt_usage = None;
        self.live_output_chars_estimate = 0;
        self.live_output_tokens_estimate = 0;
        self.model_variant = None;
        self.plugin_mode_indicators.clear();
        self.invalidate_height_cache();
    }

    pub fn restore_prepared_turn(&mut self, turn: PreparedTurn) {
        self.editor
            .restore_turn(turn.display_text, turn.images, turn.large_pastes);
    }

    pub fn next_image_marker_id(&self) -> usize {
        self.editor.next_image_marker_id()
    }

    #[allow(dead_code)]
    pub fn add_pending_image(&mut self, png_bytes: Vec<u8>) -> usize {
        let id = self.next_image_marker_id();
        self.editor
            .pending_images
            .push(PendingImage { id, png_bytes });
        id
    }

    pub fn begin_pending_image(&mut self, id: usize) {
        self.editor.begin_pending_image(id);
    }

    pub fn has_pending_image_jobs(&self) -> bool {
        self.editor.has_pending_image_jobs()
    }

    pub fn complete_pending_image(&mut self, id: usize, png_bytes: Vec<u8>) -> bool {
        self.editor.complete_pending_image(id, png_bytes)
    }

    pub fn fail_pending_image(&mut self, id: usize) -> bool {
        self.editor.fail_pending_image(id)
    }

    /// Toggle expand level 0↔1 (ghost fold ↔ compact plus) with scroll anchoring.
    pub fn cycle_expand(&mut self) {
        let new_level = if self.expand_level == 0 { 1 } else { 0 };
        self.set_expand_level(new_level);
    }

    /// Toggle to/from level 2 (full). If not at 2, set 2; if at 2, set 1.
    pub fn toggle_full_expand(&mut self) {
        let new_level = if self.expand_level != 2 { 2 } else { 1 };
        self.set_expand_level(new_level);
    }

    /// Set the expand level with scroll anchoring.
    ///
    /// When the user is scrolled to a specific position, we anchor to the block
    /// at the top of the viewport so the same content stays visible after
    /// blocks change height.
    pub fn set_expand_level(&mut self, level: u8) {
        // When following output, keep the active follow mode's anchor.
        if self.follows_output() {
            self.expand_level = level;
            self.invalidate_height_cache();
            if self.height_cache_width > 0 && self.height_cache_vh > 0 {
                self.ensure_height_cache(self.height_cache_width, self.height_cache_vh);
                self.scroll_offset =
                    self.follow_output_anchor_offset(self.height_cache_width, self.height_cache_vh);
            } else {
                self.scroll_offset = usize::MAX;
            }
            return;
        }

        // Ensure the height cache is fresh so we can compute the anchor
        if self.height_cache_width > 0 {
            let w = self.height_cache_width;
            let vh = self.height_cache_vh;
            self.ensure_height_cache(w, vh);
        }

        // Determine anchor: which block is at the top of the viewport?
        let anchor = if self.height_cache_width == 0 || self.height_cache.is_empty() {
            None
        } else {
            let cache = &self.height_cache;
            let idx = cache.partition_point(|&cum| cum <= self.scroll_offset);
            if idx >= self.blocks.len() {
                None
            } else {
                let block_start = if idx == 0 { 0 } else { cache[idx - 1] };
                let skip = self.scroll_offset - block_start;
                Some((idx, skip))
            }
        };

        // Set new level
        self.expand_level = level;
        self.invalidate_height_cache();

        // Restore scroll to keep the anchor block visible
        if let Some((anchor_idx, anchor_skip)) = anchor {
            let w = self.height_cache_width;
            let vh = self.height_cache_vh;
            self.ensure_height_cache(w, vh);

            let new_block_start = if anchor_idx == 0 {
                0
            } else {
                self.height_cache[anchor_idx - 1]
            };
            let new_block_height = self.height_cache[anchor_idx] - new_block_start;
            let clamped_skip = anchor_skip.min(new_block_height.saturating_sub(1));
            self.scroll_offset = new_block_start + clamped_skip;
        }
    }

    pub fn scroll_up(&mut self, amount: usize) {
        if self.follows_output() {
            if self.height_cache_width > 0 && self.height_cache_vh > 0 {
                self.scroll_offset =
                    self.follow_output_anchor_offset(self.height_cache_width, self.height_cache_vh);
            } else {
                self.scroll_offset = 0;
            }
            self.follow_mode = FollowOutputMode::Paused;
        }
        self.scroll_offset = self.scroll_offset.saturating_sub(amount);
        self.follow_mode = FollowOutputMode::Paused;
    }

    pub fn scroll_down(&mut self, amount: usize, viewport_height: usize, viewport_width: usize) {
        let total = self.total_content_height(viewport_width, viewport_height);
        let max_scroll = total.saturating_sub(viewport_height);
        self.scroll_offset = self.scroll_offset.saturating_add(amount).min(max_scroll);
        // Re-enable follow at the tail if we've reached the bottom manually.
        if self.scroll_offset >= max_scroll {
            self.follow_mode = FollowOutputMode::Bottom;
        }
    }

    pub fn scroll_to_bottom(&mut self) {
        if !self.follows_output() {
            return;
        }
        // We'll clamp this in rendering when we know viewport height
        self.scroll_offset = usize::MAX;
    }

    /// Explicitly resume following new output from the bottom.
    pub fn resume_follow_output(&mut self) {
        self.follow_mode = FollowOutputMode::Bottom;
        self.scroll_offset = usize::MAX;
    }

    /// Explicitly follow the current running turn from its visible start.
    pub fn resume_contextual_follow_output(&mut self) {
        self.follow_mode = FollowOutputMode::Contextual;
        self.scroll_offset = usize::MAX;
    }

    /// Recompute the follow-output anchor for the current viewport dimensions.
    pub fn refresh_follow_output_anchor(&mut self, viewport_width: usize, viewport_height: usize) {
        if !self.follows_output() {
            return;
        }

        self.scroll_offset = self.follow_output_anchor_offset(viewport_width, viewport_height);
    }

    /// Keep the latest user-authored block visible while following output.
    ///
    /// Before the first visible assistant output arrives, prefer the start of a
    /// tall prompt so the user can still read what they just sent while the
    /// live status row renders beneath the history pane.
    pub fn keep_latest_user_block_visible(&mut self) {
        if self.follow_mode != FollowOutputMode::Contextual {
            return;
        }

        let Some(last_idx) = self
            .blocks
            .iter()
            .rposition(|block| matches!(block, DisplayBlock::UserInput(_)))
        else {
            self.scroll_to_bottom();
            return;
        };

        if self.height_cache_width == 0 || self.height_cache_vh == 0 {
            self.scroll_to_bottom();
            return;
        }

        let width = self.height_cache_width;
        let viewport_height = self.height_cache_vh;
        self.ensure_height_cache(width, viewport_height);

        let total_height = self.total_content_height(width, viewport_height);
        let max_scroll = total_height.saturating_sub(viewport_height);
        let block_start = self.block_start_offset(last_idx);
        let block_end = self.height_cache[last_idx];
        let block_height = block_end.saturating_sub(block_start);
        let block_content_start = self.block_content_start_offset(last_idx);
        let has_splash_before = self.blocks[..last_idx]
            .iter()
            .any(|block| matches!(block, DisplayBlock::Splash));

        let awaiting_first_visible_output = self
            .live_turn
            .as_ref()
            .is_some_and(|turn| !turn.has_visible_output);

        self.scroll_offset = if awaiting_first_visible_output
            && (has_splash_before || block_height >= viewport_height)
        {
            self.contextual_follow_offset(block_content_start, max_scroll)
        } else {
            block_end.saturating_sub(viewport_height).min(max_scroll)
        };
    }

    fn follow_output_anchor_offset(
        &mut self,
        viewport_width: usize,
        viewport_height: usize,
    ) -> usize {
        let total_height = self.total_content_height(viewport_width, viewport_height);
        let max_scroll = total_height.saturating_sub(viewport_height);

        match self.follow_mode {
            FollowOutputMode::Paused => return self.scroll_offset.min(max_scroll),
            FollowOutputMode::Bottom => return max_scroll,
            FollowOutputMode::Contextual => {}
        }

        if !self.running {
            return max_scroll;
        }

        let awaiting_first_visible_output = self
            .live_turn
            .as_ref()
            .is_some_and(|turn| !turn.has_visible_output);

        if awaiting_first_visible_output {
            return self.latest_user_block_anchor_offset(max_scroll);
        }

        let Some(last_idx) = self.latest_turn_output_start_index() else {
            if self.streaming_output_height() > 0 {
                let streaming_start = self.height_cache.last().copied().unwrap_or(0);
                return self.contextual_follow_offset(streaming_start, max_scroll);
            }
            return max_scroll;
        };

        self.contextual_follow_offset(self.block_content_start_offset(last_idx), max_scroll)
    }

    fn latest_turn_output_start_index(&self) -> Option<usize> {
        let search_start = self
            .blocks
            .iter()
            .rposition(|block| matches!(block, DisplayBlock::UserInput(_)))
            .map(|idx| idx + 1)
            .unwrap_or(0);

        self.blocks[search_start..]
            .iter()
            .position(Self::is_turn_visible_output_block)
            .map(|offset| search_start + offset)
    }

    fn is_turn_visible_output_block(block: &DisplayBlock) -> bool {
        matches!(
            block,
            DisplayBlock::AssistantText(_)
                | DisplayBlock::CodeBlock { .. }
                | DisplayBlock::Activity(_)
                | DisplayBlock::CodeOutput { .. }
                | DisplayBlock::ShellOutput { .. }
                | DisplayBlock::Error(_)
                | DisplayBlock::PlanContent(_)
                | DisplayBlock::PluginPanel(_)
        )
    }

    fn latest_user_block_anchor_offset(&self, max_scroll: usize) -> usize {
        let Some(last_idx) = self
            .blocks
            .iter()
            .rposition(|block| matches!(block, DisplayBlock::UserInput(_)))
        else {
            return max_scroll;
        };

        self.contextual_follow_offset(self.block_content_start_offset(last_idx), max_scroll)
    }

    fn contextual_follow_offset(&self, content_start: usize, max_scroll: usize) -> usize {
        content_start
            .saturating_sub(FOLLOW_OUTPUT_CONTEXT_LINES)
            .min(max_scroll)
    }

    fn follows_output(&self) -> bool {
        self.follow_mode != FollowOutputMode::Paused
    }

    fn block_start_offset(&self, idx: usize) -> usize {
        if idx == 0 {
            0
        } else {
            self.height_cache[idx - 1]
        }
    }

    fn block_content_start_offset(&self, idx: usize) -> usize {
        self.block_start_offset(idx) + self.block_leading_padding(idx)
    }

    fn block_leading_padding(&self, idx: usize) -> usize {
        if idx == 0 {
            return 0;
        }

        match self.blocks.get(idx) {
            Some(DisplayBlock::UserInput(_)) => {
                usize::from(!matches!(self.blocks[idx - 1], DisplayBlock::Splash))
            }
            Some(DisplayBlock::AssistantText(_)) => usize::from(!matches!(
                self.blocks[idx - 1],
                DisplayBlock::AssistantText(_) | DisplayBlock::Splash
            )),
            _ => 0,
        }
    }

    /// Public accessor to pre-warm the height cache before an immutable borrow (e.g. draw).
    pub fn ensure_height_cache_pub(&mut self, width: usize, viewport_height: usize) {
        self.ensure_height_cache(width, viewport_height);
    }

    /// Read-only view of the height cache (must be pre-warmed).
    pub fn height_cache_snapshot(&self) -> &[usize] {
        &self.height_cache
    }

    /// Ensure the height cache prefix sums are up to date for the given dimensions.
    fn ensure_height_cache(&mut self, width: usize, viewport_height: usize) {
        if !self.height_cache.is_empty()
            && self.height_cache_width == width
            && self.height_cache_vh == viewport_height
        {
            return;
        }
        self.height_cache_width = width;
        self.height_cache_vh = viewport_height;
        self.height_cache.clear();
        self.height_cache.reserve(self.blocks.len());
        let mut cumulative: usize = 0;
        for (i, block) in self.blocks.iter().enumerate() {
            let _ = block;
            cumulative += ui::rendered_block_height(
                &self.blocks,
                i,
                self.expand_level,
                width,
                viewport_height,
            );
            self.height_cache.push(cumulative);
        }
    }

    pub fn total_content_height(&mut self, width: usize, viewport_height: usize) -> usize {
        self.ensure_height_cache(width, viewport_height);
        let block_height = self.height_cache.last().copied().unwrap_or(0);
        let streaming_height = self.streaming_output_height();
        block_height + streaming_height
    }

    pub fn streaming_output_height(&self) -> usize {
        usize::from(self.streaming_output_hidden > 0)
            + self.streaming_output.len()
            + usize::from(!self.streaming_output_partial.is_empty())
    }

    fn clear_streaming_output(&mut self) {
        self.streaming_output.clear();
        self.streaming_output_hidden = 0;
        self.streaming_output_partial.clear();
    }

    fn push_streaming_output_text(&mut self, text: &str) {
        for segment in text.split_inclusive('\n') {
            let line = segment.trim_end_matches('\n').trim_end_matches('\r');
            self.streaming_output_partial.push_str(line);
            if segment.ends_with('\n') {
                let completed = std::mem::take(&mut self.streaming_output_partial);
                self.push_streaming_output_line(completed);
            }
        }
        if self.streaming_output_partial.chars().count() > STREAMING_OUTPUT_LINE_CHAR_LIMIT {
            self.streaming_output_partial = smart_truncate_preview_line(
                &self.streaming_output_partial,
                STREAMING_OUTPUT_LINE_CHAR_LIMIT,
            );
        }
    }

    fn push_streaming_output_line(&mut self, line: String) {
        if self.streaming_output.len() == STREAMING_OUTPUT_MAX_LINES {
            self.streaming_output.remove(0);
            self.streaming_output_hidden += 1;
        }
        self.streaming_output.push(smart_truncate_preview_line(
            &line,
            STREAMING_OUTPUT_LINE_CHAR_LIMIT,
        ));
    }

    /// Navigate input history with up arrow.
    /// In multi-line mode, if cursor is not on the first line, moves cursor up instead.
    pub fn history_up(&mut self) {
        if self.editor.input.is_empty()
            && self.editor.input_history_idx.is_none()
            && let Some((turn, _was_pending)) = self.take_last_queued_turn()
        {
            self.restore_prepared_turn(turn);
            return;
        }
        self.editor.history_up();
    }

    /// Navigate input history with down arrow.
    /// In multi-line mode, if cursor is not on the last line, moves cursor down instead.
    pub fn history_down(&mut self) {
        self.editor.history_down();
    }

    /// Insert a character at cursor position.
    pub fn insert_char(&mut self, c: char) {
        self.editor.insert_char(c);
    }

    /// Insert literal text at the cursor position.
    pub fn insert_text(&mut self, text: &str) {
        self.editor.insert_text(text);
    }

    pub fn insert_pasted_text(&mut self, text: &str) {
        self.editor.insert_pasted_text(text);
    }

    /// Delete character before cursor.
    pub fn backspace(&mut self) {
        self.editor.backspace();
    }

    /// Delete character at cursor.
    pub fn delete(&mut self) {
        self.editor.delete();
    }

    pub fn move_cursor_left(&mut self) {
        self.editor.move_cursor_left();
    }

    pub fn move_cursor_right(&mut self) {
        self.editor.move_cursor_right();
    }

    pub fn move_cursor_home(&mut self) {
        self.editor.move_cursor_home();
    }

    pub fn move_cursor_end(&mut self) {
        self.editor.move_cursor_end();
    }

    /// Load input history from $LASH_HOME/history.
    pub fn load_history(&mut self) {
        self.editor.load_history();
    }

    /// Save input history to $LASH_HOME/history (last 500 entries).
    pub fn save_history(&self) {
        self.editor.save_history();
    }

    /// Update the suggestion list based on current input.
    pub fn update_suggestions(&mut self) {
        self.editor.update_suggestions(&self.skills);
    }

    /// Whether the suggestion popup is active.
    pub fn has_suggestions(&self) -> bool {
        self.editor.has_suggestions()
    }

    /// Move suggestion selection up.
    pub fn suggestion_up(&mut self) {
        self.editor.suggestion_up();
    }

    /// Move suggestion selection down.
    pub fn suggestion_down(&mut self) {
        self.editor.suggestion_down();
    }

    /// Accept the selected suggestion.
    pub fn complete_suggestion(&mut self) {
        self.editor.complete_suggestion(&self.skills);
    }

    /// Whether the session picker is active.
    pub fn has_session_picker(&self) -> bool {
        matches!(&self.overlay, Some(OverlayState::SessionPicker(state)) if !state.items.is_empty())
    }

    /// Move session picker selection up.
    pub fn session_picker_up(&mut self) {
        if let Some(OverlayState::SessionPicker(state)) = &mut self.overlay {
            state.up();
        }
    }

    /// Move session picker selection down.
    pub fn session_picker_down(&mut self) {
        if let Some(OverlayState::SessionPicker(state)) = &mut self.overlay {
            state.down();
        }
    }

    /// Get the selected session filename, clearing the picker.
    pub fn take_session_pick(&mut self) -> Option<String> {
        match self.overlay.take() {
            Some(OverlayState::SessionPicker(mut state)) => {
                state.take_selected().map(|s| s.filename)
            }
            other => {
                self.overlay = other;
                None
            }
        }
    }

    /// Dismiss the session picker without selecting.
    pub fn dismiss_session_picker(&mut self) {
        if matches!(self.overlay, Some(OverlayState::SessionPicker(_))) {
            self.overlay = None;
        }
    }

    /// Whether the skill picker is active.
    pub fn has_skill_picker(&self) -> bool {
        matches!(&self.overlay, Some(OverlayState::SkillPicker(state)) if !state.items.is_empty())
    }

    /// Move skill picker selection up.
    pub fn skill_picker_up(&mut self) {
        if let Some(OverlayState::SkillPicker(state)) = &mut self.overlay {
            state.up();
        }
    }

    /// Move skill picker selection down.
    pub fn skill_picker_down(&mut self) {
        if let Some(OverlayState::SkillPicker(state)) = &mut self.overlay {
            state.down();
        }
    }

    /// Get the selected skill name, clearing the picker.
    pub fn take_skill_pick(&mut self) -> Option<String> {
        match self.overlay.take() {
            Some(OverlayState::SkillPicker(mut state)) => {
                state.take_selected().map(|(name, _)| name)
            }
            other => {
                self.overlay = other;
                None
            }
        }
    }

    /// Dismiss the skill picker without selecting.
    pub fn dismiss_skill_picker(&mut self) {
        if matches!(self.overlay, Some(OverlayState::SkillPicker(_))) {
            self.overlay = None;
        }
    }

    // ── Prompt (ask dialog) methods ──

    /// Whether the prompt dialog is active.
    pub fn has_prompt(&self) -> bool {
        matches!(self.overlay, Some(OverlayState::Prompt(_)))
    }

    /// Whether the prompt is currently focused on reply editing.
    pub fn is_prompt_editing_reply(&self) -> bool {
        match &self.overlay {
            Some(OverlayState::Prompt(prompt)) => prompt.is_editing_reply(),
            _ => false,
        }
    }

    /// Whether the prompt is freeform-only (no options).
    pub fn is_prompt_freeform(&self) -> bool {
        match &self.overlay {
            Some(OverlayState::Prompt(prompt)) => prompt.is_freeform(),
            _ => false,
        }
    }

    /// Move prompt selection up.
    pub fn prompt_up(&mut self) {
        if let Some(OverlayState::Prompt(p)) = &mut self.overlay {
            p.move_up();
        }
    }

    /// Move prompt selection down.
    pub fn prompt_down(&mut self) {
        if let Some(OverlayState::Prompt(p)) = &mut self.overlay {
            p.move_down();
        }
    }

    /// Toggle extra text editing for prompts that also have discrete options.
    pub fn prompt_toggle_extra(&mut self) {
        if let Some(OverlayState::Prompt(p)) = &mut self.overlay {
            p.toggle_focus();
        }
    }

    /// Insert a character into the prompt extra text (or freeform input).
    pub fn prompt_insert_char(&mut self, c: char) {
        if let Some(OverlayState::Prompt(p)) = &mut self.overlay {
            p.insert_char(c);
        }
    }

    /// Insert literal text into the prompt extra text (or freeform input).
    pub fn prompt_insert_text(&mut self, text: &str) {
        if let Some(OverlayState::Prompt(p)) = &mut self.overlay {
            p.insert_text(text);
        }
    }

    /// Delete character before cursor in prompt extra text.
    pub fn prompt_backspace(&mut self) {
        if let Some(OverlayState::Prompt(p)) = &mut self.overlay {
            p.backspace();
        }
    }

    /// Submit the prompt response, render it as user input, and dismiss the dialog.
    pub fn take_prompt_response(&mut self) -> Option<String> {
        if let Some(OverlayState::Prompt(p)) = self.overlay.take() {
            let response = p.submitted_response();
            let _ = p.response_tx.send(response.clone());
            self.invalidate_height_cache();
            self.scroll_to_bottom();
            self.dirty = true;
            if !response.trim().is_empty() {
                self.blocks.push(DisplayBlock::UserInput(response.clone()));
                self.invalidate_height_cache();
                self.keep_latest_user_block_visible();
                return Some(response);
            }
        }
        None
    }

    /// Dismiss the prompt without selecting (Esc) — sends empty string to unblock the REPL runtime.
    pub fn dismiss_prompt(&mut self) {
        if let Some(OverlayState::Prompt(p)) = self.overlay.take() {
            let _ = p.response_tx.send(String::new());
            self.invalidate_height_cache();
            self.scroll_to_bottom();
            self.dirty = true;
        }
    }

    pub fn show_session_picker(&mut self, items: Vec<crate::session_log::SessionInfo>) {
        self.overlay = Some(OverlayState::SessionPicker(PickerState::new(items)));
    }

    pub fn session_picker_state(&self) -> Option<&PickerState<crate::session_log::SessionInfo>> {
        match &self.overlay {
            Some(OverlayState::SessionPicker(state)) => Some(state),
            _ => None,
        }
    }

    pub fn show_skill_picker(&mut self, items: Vec<(String, String)>) {
        self.overlay = Some(OverlayState::SkillPicker(PickerState::new(items)));
    }

    pub fn skill_picker_state(&self) -> Option<&PickerState<(String, String)>> {
        match &self.overlay {
            Some(OverlayState::SkillPicker(state)) => Some(state),
            _ => None,
        }
    }

    pub fn show_prompt(&mut self, prompt: PromptState) {
        self.overlay = Some(OverlayState::Prompt(prompt));
    }

    pub fn prompt_state(&self) -> Option<&PromptState> {
        match &self.overlay {
            Some(OverlayState::Prompt(prompt)) => Some(prompt),
            _ => None,
        }
    }

    pub fn input(&self) -> &str {
        &self.editor.input
    }

    pub fn set_input(&mut self, input: String) {
        self.editor.input = input;
        self.editor.cursor_pos = self.editor.input.len();
    }

    pub fn cursor_pos(&self) -> usize {
        self.editor.cursor_pos
    }

    pub fn suggestions(&self) -> &[(String, String)] {
        &self.editor.suggestions
    }

    pub fn suggestion_kind(&self) -> SuggestionKind {
        self.editor.suggestion_kind
    }

    pub fn suggestion_idx(&self) -> usize {
        self.editor.suggestion_idx
    }
}

/// Compute the visual height of pending streaming text.
/// Format a token count for display: 1234 → "1.2k", 567 → "567", 12345 → "12.3k"
pub fn format_tokens(n: i64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        format!("{}", n)
    }
}

pub(crate) fn render_plan_content_from_args(args: &serde_json::Value) -> Option<String> {
    let items = args.get("plan").and_then(|value| value.as_array())?;
    if items.is_empty() {
        return None;
    }

    let mut lines = Vec::new();
    if let Some(explanation) = args
        .get("explanation")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        lines.push(explanation.to_string());
        lines.push(String::new());
    }

    for item in items {
        let step = item.get("step").and_then(|value| value.as_str())?;
        let status = item.get("status").and_then(|value| value.as_str())?;
        let marker = match status {
            "completed" => "\u{2713}",
            "in_progress" => "\u{25b8}",
            _ => "\u{25cb}",
        };
        lines.push(format!("{marker} {step}"));
    }

    Some(lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::LARGE_PASTE_CHAR_THRESHOLD;
    use crate::overlay::{PromptFocus, PromptSelection};

    #[test]
    fn renders_plan_content_from_update_plan_args() {
        let content = render_plan_content_from_args(&serde_json::json!({
            "explanation": "Found the renderer.",
            "plan": [
                {"step":"Inspect UI", "status":"completed"},
                {"step":"Patch layout", "status":"in_progress"}
            ]
        }))
        .expect("plan content");
        assert!(content.contains("Found the renderer."));
        assert!(content.contains("\u{2713} Inspect UI"));
        assert!(content.contains("\u{25b8} Patch layout"));
        assert!(!content.contains("1."));
    }

    #[test]
    fn display_block_code_block_height() {
        let block = DisplayBlock::CodeBlock {
            code: "print('hi')".into(),
            continuation: false,
        };
        // Level 0: first in group = 1 (ghost fold summary)
        assert_eq!(block.height(0, 80, 0), 1);
        // Level 1: compact view hides code blocks
        assert_eq!(block.height(1, 80, 0), 0);
        // Level 2: full code view
        assert_eq!(block.height(2, 80, 0), 1);

        let cont = DisplayBlock::CodeBlock {
            code: "print('hi')".into(),
            continuation: true,
        };
        assert_eq!(cont.height(0, 80, 0), 0); // absorbed into ghost fold
        assert_eq!(cont.height(1, 80, 0), 0); // enabled at level 1
        assert_eq!(cont.height(2, 80, 0), 1); // visible at level 2
    }

    #[test]
    fn text_delta_accumulates_raw() {
        let mut app = App::new("test-model".into(), "test".into());
        app.handle_agent_event(AgentEvent::TextDelta {
            content: "\n\nfirst\n".into(),
        });
        // Raw accumulation — no normalization until flush
        assert_eq!(app.pending_text, "\n\nfirst\n");

        app.handle_agent_event(AgentEvent::TextDelta {
            content: "\n\n\nsecond\n".into(),
        });
        assert_eq!(app.pending_text, "\n\nfirst\n\n\n\nsecond\n");
    }

    #[test]
    fn text_delta_code_fence_preserved() {
        let mut app = App::new("test-model".into(), "test".into());
        app.handle_agent_event(AgentEvent::TextDelta {
            content: "text\n\n```python\n".into(),
        });
        app.handle_agent_event(AgentEvent::TextDelta {
            content: "# comment\n".into(),
        });
        // The newline between ```python and # comment must be preserved
        assert!(app.pending_text.contains("```python\n# comment"));
    }

    #[test]
    fn text_delta_renders_into_a_durable_assistant_block() {
        let mut app = App::new("test-model".into(), "test".into());
        app.start_turn();

        app.handle_agent_event(AgentEvent::TextDelta {
            content: "Draft answer".into(),
        });

        assert!(matches!(
            app.blocks.last(),
            Some(DisplayBlock::AssistantText(text)) if text == "Draft answer"
        ));
        assert_eq!(
            app.live_turn
                .as_ref()
                .and_then(|turn| turn.assistant_block_idx),
            Some(app.blocks.len() - 1)
        );
    }

    #[test]
    fn final_message_never_replaces_visible_streamed_text_with_shorter_text() {
        let mut app = App::new("test-model".into(), "test".into());
        app.start_turn();
        app.handle_agent_event(AgentEvent::TextDelta {
            content: "Visible streamed text".into(),
        });

        app.handle_agent_event(AgentEvent::Message {
            text: "Visible".into(),
            kind: "final".into(),
        });

        assert!(matches!(
            app.blocks.last(),
            Some(DisplayBlock::AssistantText(text)) if text == "Visible streamed text"
        ));
    }

    #[test]
    fn text_delta_updates_live_token_estimate() {
        let mut app = App::new("test-model".into(), "test".into());
        app.handle_agent_event(AgentEvent::LlmRequest {
            iteration: 0,
            message_count: 0,
            tool_list: String::new(),
        });
        app.handle_agent_event(AgentEvent::TextDelta {
            content: "abcd".into(),
        });
        assert_eq!(app.live_output_tokens_estimate, 1);
        app.handle_agent_event(AgentEvent::TextDelta {
            content: "efgh".into(),
        });
        assert_eq!(app.live_output_tokens_estimate, 2);
    }

    #[test]
    fn late_text_deltas_after_stop_turn_extend_last_assistant_block() {
        let mut app = App::new("test-model".into(), "test".into());
        app.start_turn();
        app.handle_agent_event(AgentEvent::TextDelta {
            content: "I".into(),
        });
        app.handle_agent_event(AgentEvent::TextDelta {
            content: "’m".into(),
        });

        app.stop_turn();

        app.handle_agent_event(AgentEvent::TextDelta {
            content: " an".into(),
        });
        app.handle_agent_event(AgentEvent::TextDelta {
            content: " AI".into(),
        });

        let assistant_blocks: Vec<&str> = app
            .blocks
            .iter()
            .filter_map(|block| match block {
                DisplayBlock::AssistantText(text) => Some(text.as_str()),
                _ => None,
            })
            .collect();

        assert_eq!(assistant_blocks, vec!["I’m an AI"]);
    }

    #[test]
    fn first_text_delta_clears_waiting_for_first_token_status() {
        let mut app = App::new("test-model".into(), "test".into());
        app.handle_agent_event(AgentEvent::LlmRequest {
            iteration: 0,
            message_count: 0,
            tool_list: String::new(),
        });
        app.handle_agent_event(AgentEvent::TextDelta {
            content: "hello".into(),
        });
        assert_eq!(
            app.live_turn.as_ref().map(|turn| turn.status_text.as_str()),
            Some("responding")
        );
        assert_eq!(
            app.live_turn
                .as_ref()
                .and_then(|turn| turn.status_detail.as_deref()),
            None
        );
    }

    #[test]
    fn llm_request_sets_waiting_for_first_token_status() {
        let mut app = App::new("test-model".into(), "test".into());
        app.handle_agent_event(AgentEvent::LlmRequest {
            iteration: 0,
            message_count: 0,
            tool_list: String::new(),
        });
        assert_eq!(
            app.live_turn.as_ref().map(|turn| turn.status_text.as_str()),
            Some("thinking")
        );
        assert_eq!(
            app.live_turn
                .as_ref()
                .and_then(|turn| turn.status_detail.as_deref()),
            Some("waiting for first token")
        );
    }

    #[test]
    fn llm_request_flushes_intermediate_stream_text() {
        let mut app = App::new("test-model".into(), "test".into());
        app.handle_agent_event(AgentEvent::TextDelta {
            content: "Let me continue testing.".into(),
        });
        app.handle_agent_event(AgentEvent::ToolCall {
            call_id: Some("tc1".into()),
            name: "read_file".into(),
            args: serde_json::json!({"path":"src/main.rs"}),
            result: serde_json::json!("ok"),
            success: true,
            duration_ms: 1,
        });
        app.handle_agent_event(AgentEvent::LlmRequest {
            iteration: 1,
            message_count: 0,
            tool_list: String::new(),
        });

        assert!(app.pending_text.is_empty());
        assert!(app.blocks.iter().any(|block| {
            matches!(block, DisplayBlock::AssistantText(text) if text == "Let me continue testing.")
        }));
    }

    #[test]
    fn tool_call_flushes_intermediate_stream_text_immediately() {
        let mut app = App::new("test-model".into(), "test".into());
        app.blocks.clear();

        app.handle_agent_event(AgentEvent::TextDelta {
            content: "I’m checking the rendering path first.".into(),
        });
        app.handle_agent_event(AgentEvent::ToolCall {
            call_id: Some("tc2".into()),
            name: "read_file".into(),
            args: serde_json::json!({"path":"lash-cli/src/app.rs"}),
            result: serde_json::json!("ok"),
            success: true,
            duration_ms: 1,
        });

        assert!(app.pending_text.is_empty());
        assert!(matches!(
            app.blocks.first(),
            Some(DisplayBlock::AssistantText(text))
                if text == "I’m checking the rendering path first."
        ));
        assert!(matches!(app.blocks.get(1), Some(DisplayBlock::Activity(_))));
    }

    #[test]
    fn token_usage_resets_live_token_estimate() {
        let mut app = App::new("test-model".into(), "test".into());
        app.handle_agent_event(AgentEvent::TextDelta {
            content: "abcdefgh".into(),
        });
        assert!(app.live_output_tokens_estimate > 0);
        app.handle_agent_event(AgentEvent::TokenUsage {
            iteration: 0,
            usage: TokenUsage {
                input_tokens: 10,
                output_tokens: 5,
                cached_input_tokens: 0,
                reasoning_tokens: 2,
            },
            cumulative: TokenUsage {
                input_tokens: 10,
                output_tokens: 5,
                cached_input_tokens: 0,
                reasoning_tokens: 2,
            },
        });
        assert_eq!(app.live_output_tokens_estimate, 0);
        assert_eq!(app.last_response_usage.input_tokens, 10);
        assert_eq!(app.last_response_usage.reasoning_tokens, 2);
    }

    #[test]
    fn input_only_streamed_usage_keeps_live_output_estimate() {
        let mut app = App::new("test-model".into(), "test".into());
        app.handle_agent_event(AgentEvent::TextDelta {
            content: "abcdefgh".into(),
        });
        let live_estimate = app.live_output_tokens_estimate;
        assert!(live_estimate > 0);
        app.handle_agent_event(AgentEvent::TokenUsage {
            iteration: 0,
            usage: TokenUsage {
                input_tokens: 10,
                output_tokens: 0,
                cached_input_tokens: 0,
                reasoning_tokens: 0,
            },
            cumulative: TokenUsage {
                input_tokens: 10,
                output_tokens: 0,
                cached_input_tokens: 0,
                reasoning_tokens: 0,
            },
        });
        assert_eq!(app.live_output_tokens_estimate, live_estimate);
        assert_eq!(app.token_usage.input_tokens, 10);
        assert_eq!(app.last_response_usage.input_tokens, 10);
    }

    #[test]
    fn final_message_event_is_rendered() {
        let mut app = App::new("test-model".into(), "test".into());
        app.handle_agent_event(AgentEvent::Message {
            text: "final output".into(),
            kind: "final".into(),
        });
        assert!(matches!(
            app.blocks.last(),
            Some(DisplayBlock::AssistantText(text)) if text == "final output"
        ));
    }

    #[test]
    fn finish_turn_for_resume_reconciles_authoritative_assistant_text() {
        let mut app = App::new("test-model".into(), "test".into());
        app.start_turn();
        app.handle_agent_event(AgentEvent::TextDelta {
            content: "I looked at the actual librarian prompt".into(),
        });
        app.stop_turn();

        let persisted = app.finish_turn_for_resume_with_output(Some(
            "I looked at the actual librarian prompt, the graph tool constraints.\n\n## What exists now",
        ));

        let last_block = app
            .blocks
            .iter()
            .rev()
            .find_map(|block| match block {
                DisplayBlock::AssistantText(text) => Some(text.as_str()),
                _ => None,
            })
            .expect("assistant block");
        assert_eq!(
            last_block,
            "I looked at the actual librarian prompt, the graph tool constraints.\n\n## What exists now"
        );
        assert_eq!(persisted.live_assistant_text, None);
    }

    #[test]
    fn finish_turn_for_resume_does_not_append_shorter_authoritative_text() {
        let mut app = App::new("test-model".into(), "test".into());
        app.start_turn();
        app.handle_agent_event(AgentEvent::TextDelta {
            content: "Visible streamed text".into(),
        });
        app.stop_turn();

        let persisted = app.finish_turn_for_resume_with_output(Some("Visible"));

        let last_block = app
            .blocks
            .iter()
            .rev()
            .find_map(|block| match block {
                DisplayBlock::AssistantText(text) => Some(text.as_str()),
                _ => None,
            })
            .expect("assistant block");
        assert_eq!(last_block, "Visible streamed text");
        assert_eq!(persisted.live_assistant_text, None);
    }

    #[test]
    fn late_text_deltas_after_authoritative_final_output_are_ignored() {
        let mut app = App::new("test-model".into(), "test".into());
        let final_text = "Use this minimal set:\n\n- `code`\n- `feature`\n- `issue`\n- `decision`\n\nThat’s probably the sweet spot.";
        app.start_turn();
        app.handle_agent_event(AgentEvent::TextDelta {
            content: "Use this minimal set:\n\n- `code`\n- `feature`\n".into(),
        });

        let _persisted = app.finish_turn_for_resume_with_output(Some(final_text));

        app.handle_agent_event(AgentEvent::TextDelta {
            content: "Yeah — **`feature` is nicer than `topic`** if you want the graph to stay product-shaped.\n\nMy take:\n\n- **`topic` is safer**".into(),
        });

        let assistant_blocks: Vec<&str> = app
            .blocks
            .iter()
            .filter_map(|block| match block {
                DisplayBlock::AssistantText(text) => Some(text.as_str()),
                _ => None,
            })
            .collect();

        assert_eq!(assistant_blocks, vec![final_text]);
    }

    #[test]
    fn delegate_start_enables_streaming_output_until_agent_result_arrives() {
        let mut app = App::new("test-model".into(), "test".into());
        app.handle_agent_event(AgentEvent::Message {
            text: serde_json::json!({
                "name":"delegate",
                "task":"inspect queue rendering",
                "model":"gpt-5.4",
                "model_variant":"high"
            })
            .to_string(),
            kind: "delegate_start".into(),
        });
        assert!(app.active_delegate.is_some());
        assert_eq!(
            app.live_turn.as_ref().map(|turn| turn.status_text.as_str()),
            Some("delegating")
        );
        assert_eq!(
            app.live_turn
                .as_ref()
                .and_then(|turn| turn.status_detail.as_deref()),
            Some("inspect queue rendering · gpt-5.4 (high)")
        );

        app.handle_agent_event(AgentEvent::Message {
            text: "delegate stream\n".into(),
            kind: "tool_output".into(),
        });
        assert_eq!(app.streaming_output, vec!["delegate stream".to_string()]);

        app.handle_agent_event(AgentEvent::ToolCall {
            call_id: Some("tc-delegate".into()),
            name: "agent_result".into(),
            args: serde_json::json!({"id":"child-1"}),
            result: serde_json::json!({
                "result":"done",
                "status":"completed",
                "session":{"task":"inspect queue rendering"}
            }),
            success: true,
            duration_ms: 5,
        });
        assert!(app.active_delegate.is_none());
        assert!(app.streaming_output.is_empty());
    }

    #[test]
    fn tool_output_renders_during_generic_running_turn() {
        let mut app = App::new("test-model".into(), "test".into());
        app.start_turn();
        app.handle_agent_event(AgentEvent::Message {
            text: "started git status --short\n".into(),
            kind: "tool_output".into(),
        });

        assert_eq!(
            app.streaming_output,
            vec!["started git status --short".to_string()]
        );
    }

    #[test]
    fn finish_turn_for_resume_preserves_streaming_output_snapshot() {
        let mut app = App::new("test-model".into(), "test".into());
        app.start_turn();
        app.handle_agent_event(AgentEvent::Message {
            text: "started git status --short\n".into(),
            kind: "tool_output".into(),
        });

        let persisted = app.finish_turn_for_resume_with_output(None);

        assert!(!app.running);
        assert!(app.streaming_output.is_empty());
        assert_eq!(
            persisted.streaming_output,
            vec!["started git status --short".to_string()]
        );
    }

    #[test]
    fn plugin_panel_events_upsert_and_clear_blocks() {
        let mut app = App::new("test-model".into(), "test".into());
        app.start_turn();
        app.handle_agent_event(AgentEvent::PluginEvent {
            plugin_id: "demo".into(),
            event: lash::PluginSurfaceEvent::PanelUpsert {
                key: "panel:1".into(),
                title: "TASK BOARD".into(),
                content: "1. Inspect\n2. Patch".into(),
            },
        });
        assert!(matches!(
            app.blocks.last(),
            Some(DisplayBlock::PluginPanel(panel)) if panel.title == "TASK BOARD"
        ));
        assert!(
            app.live_turn
                .as_ref()
                .is_some_and(|turn| turn.has_visible_output)
        );

        app.handle_agent_event(AgentEvent::PluginEvent {
            plugin_id: "demo".into(),
            event: lash::PluginSurfaceEvent::PanelClear {
                key: "panel:1".into(),
            },
        });
        assert!(
            !app.blocks
                .iter()
                .any(|block| matches!(block, DisplayBlock::PluginPanel(_)))
        );
    }

    #[test]
    fn plan_exit_tool_queues_fresh_follow_up_turn() {
        let mut app = App::new("test-model".into(), "test".into());
        app.handle_agent_event(AgentEvent::ToolCall {
            call_id: Some("tc-plan-exit".into()),
            name: "plan_exit".into(),
            args: serde_json::json!({}),
            result: serde_json::json!({
                "approved": true,
                "plan_path": ".lash/plans/session.md",
                "next_turn_input": "The plan at `.lash/plans/session.md` is approved. Execute that plan."
            }),
            success: true,
            duration_ms: 5,
        });

        let (queued, was_pending) = app.take_next_queued_turn().expect("queued turn");
        assert!(!was_pending);
        assert_eq!(
            queued.display_text,
            "The plan at `.lash/plans/session.md` is approved. Execute that plan."
        );
    }

    #[test]
    fn cancelled_error_renders_as_system_message() {
        let mut app = App::new("test-model".into(), "test".into());
        app.note_manual_interrupt_requested();
        app.handle_agent_event(AgentEvent::Error {
            message: "LLM error: cancelled".into(),
            envelope: Some(lash::agent::ErrorEnvelope {
                kind: "llm_provider".into(),
                code: Some("cancelled".into()),
                user_message: "LLM error: cancelled".into(),
                raw: None,
            }),
        });

        assert!(matches!(
            app.blocks.last(),
            Some(DisplayBlock::SystemMessage(msg)) if msg == "Manually interrupted."
        ));
    }

    #[test]
    fn cancelled_error_without_manual_request_renders_as_error() {
        let mut app = App::new("test-model".into(), "test".into());
        app.handle_agent_event(AgentEvent::Error {
            message: "LLM error: cancelled".into(),
            envelope: Some(lash::agent::ErrorEnvelope {
                kind: "llm_provider".into(),
                code: Some("cancelled".into()),
                user_message: "LLM error: cancelled".into(),
                raw: None,
            }),
        });

        assert!(matches!(
            app.blocks.last(),
            Some(DisplayBlock::Error(msg)) if msg == "LLM error: cancelled"
        ));
    }

    #[test]
    fn non_manual_error_sets_transient_status() {
        let mut app = App::new("test-model".into(), "test".into());
        app.handle_agent_event(AgentEvent::LlmRequest {
            iteration: 0,
            message_count: 0,
            tool_list: String::new(),
        });
        app.handle_agent_event(AgentEvent::Error {
            message: "LLM error: Claude request failed with 500".into(),
            envelope: Some(lash::agent::ErrorEnvelope {
                kind: "llm_provider".into(),
                code: Some("http_500".into()),
                user_message: "LLM error: Claude request failed with 500".into(),
                raw: None,
            }),
        });
        app.handle_agent_event(AgentEvent::Done);

        assert_eq!(
            app.live_turn.as_ref().map(|turn| turn.status_text.as_str()),
            Some("error")
        );
        assert_eq!(
            app.live_turn
                .as_ref()
                .and_then(|turn| turn.status_detail.as_deref()),
            Some("LLM error: Claude request failed with 500")
        );
    }

    #[test]
    fn transient_status_expires_on_tick() {
        let mut app = App::new("test-model".into(), "test".into());
        app.handle_agent_event(AgentEvent::LlmRequest {
            iteration: 0,
            message_count: 0,
            tool_list: String::new(),
        });
        app.handle_agent_event(AgentEvent::Error {
            message: "runtime error".into(),
            envelope: None,
        });
        app.handle_agent_event(AgentEvent::Done);

        if let Some(turn) = app.live_turn.as_mut() {
            turn.transient_until =
                Some(std::time::Instant::now() - std::time::Duration::from_secs(1));
        }
        app.on_tick();
        assert!(app.live_turn.is_none());
    }

    #[test]
    fn ui_resume_state_omits_transient_live_turn() {
        let mut app = App::new("test-model".into(), "test".into());
        app.start_turn();
        app.set_status("retrying", Some("in 5s".into()), true);
        if let Some(turn) = app.live_turn.as_mut() {
            turn.has_visible_output = true;
        }

        let persisted = serde_json::to_value(app.ui_resume_state()).expect("serialize ui");
        assert!(persisted.get("live_turn").is_none());
    }

    #[test]
    fn queued_turns_are_fifo_and_skip_pending_injections() {
        let mut app = App::new("test-model".into(), "test".into());
        app.queue_turn(PreparedTurn::new("queued-1".into(), Vec::new()));
        app.queue_turn(PreparedTurn::new("queued-2".into(), Vec::new()));
        app.queue_pending_steer(PreparedTurn::new("next-1".into(), Vec::new()));
        app.queue_pending_steer(PreparedTurn::new("next-2".into(), Vec::new()));

        let order: Vec<(String, bool)> = std::iter::from_fn(|| app.take_next_queued_turn())
            .map(|(turn, was_pending)| (turn.display_text, was_pending))
            .collect();

        assert_eq!(
            order,
            vec![("queued-1".into(), false), ("queued-2".into(), false),]
        );
        assert_eq!(app.pending_steers.len(), 2);
    }

    #[test]
    fn take_last_queued_turn_restores_explicit_queue_only() {
        let mut app = App::new("test-model".into(), "test".into());
        app.queue_pending_steer(PreparedTurn::new("next".into(), Vec::new()));
        app.queue_turn(PreparedTurn::new("queued".into(), vec![vec![1, 2, 3]]));

        let (turn, was_pending) = app.take_last_queued_turn().expect("queued turn");
        assert_eq!(turn.display_text, "queued");
        assert_eq!(turn.images.len(), 1);
        assert_eq!(turn.images[0].id, 1);
        assert_eq!(turn.images[0].png_bytes, vec![1, 2, 3]);
        assert!(!was_pending);

        assert!(app.take_last_queued_turn().is_none());
        assert_eq!(app.pending_steers.len(), 1);
    }

    #[test]
    fn injected_messages_commit_render_user_blocks_and_clear_pending_preview() {
        let mut app = App::new("test-model".into(), "test".into());
        let turn = PreparedTurn::new("follow up".into(), Vec::new());
        app.queue_pending_steer(turn.clone());
        app.preview_queued_turn(&turn, true);

        app.handle_agent_event(AgentEvent::InjectedMessagesCommitted {
            messages: vec![PluginMessage::text(MessageRole::User, "follow up")],
            checkpoint: lash::CheckpointKind::AfterWork,
        });

        assert!(app.pending_steers.is_empty());
        assert!(matches!(
            app.blocks.last(),
            Some(DisplayBlock::UserInput(text)) if text == "follow up"
        ));
    }

    #[test]
    fn injected_messages_clear_pending_queue_even_when_runtime_content_differs() {
        let mut app = App::new("test-model".into(), "test".into());
        let mut turn = PreparedTurn::new("/localref lash for context if needed".into(), Vec::new());
        turn.transform_labels = vec!["localref".into()];
        app.queue_pending_steer(turn.clone());
        app.preview_queued_turn(&turn, true);

        app.handle_agent_event(AgentEvent::InjectedMessagesCommitted {
            messages: vec![PluginMessage::text(
                MessageRole::User,
                "/localref lash for context if needed\n\n<skill>\n<name>localref</name>\nbody\n</skill>",
            )],
            checkpoint: lash::CheckpointKind::AfterWork,
        });

        assert!(app.pending_steers.is_empty());
        assert!(
            matches!(app.blocks.last(), Some(DisplayBlock::UserInput(text)) if text == "/localref lash for context if needed")
        );
    }

    #[test]
    fn queued_injection_preview_stays_out_of_history() {
        let mut app = App::new("test-model".into(), "test".into());
        let turn = PreparedTurn::new("follow up now".into(), Vec::new());

        app.queue_pending_steer(turn.clone());
        app.preview_queued_turn(&turn, true);

        assert!(!app.commit_pending_user_preview("follow up now"));
        assert!(!matches!(
            app.blocks.last(),
            Some(DisplayBlock::UserInput(_))
        ));
    }

    #[test]
    fn regular_queued_turn_preview_stays_out_of_history() {
        let mut app = App::new("test-model".into(), "test".into());
        let turn = PreparedTurn::new("queued text".into(), Vec::new());
        app.preview_queued_turn(&turn, false);

        assert!(!app.commit_pending_user_preview("queued text"));
        assert!(!matches!(
            app.blocks.last(),
            Some(DisplayBlock::UserInput(_))
        ));
    }

    #[test]
    fn history_up_restores_last_queued_turn_before_history() {
        let mut app = App::new("test-model".into(), "test".into());
        app.editor.input_history = vec!["older turn".into()];
        app.queue_turn(PreparedTurn::new("queued text".into(), vec![vec![1, 2, 3]]));

        app.history_up();

        assert_eq!(app.input(), "queued text");
        assert_eq!(app.editor.pending_images.len(), 1);
        assert_eq!(app.editor.pending_images[0].id, 1);
        assert_eq!(app.editor.pending_images[0].png_bytes, vec![1, 2, 3]);
        assert!(app.queued_turns.is_empty());
        assert_eq!(app.editor.input_history_idx, None);
    }

    #[test]
    fn restore_prepared_turn_clears_history_selection() {
        let mut app = App::new("test-model".into(), "test".into());
        app.editor.input_history = vec!["older turn".into()];
        app.editor.input_history_idx = Some(0);

        app.restore_prepared_turn(PreparedTurn::new("queued text".into(), Vec::new()));

        assert_eq!(app.editor.input_history_idx, None);
    }

    #[test]
    fn backspace_deletes_image_marker_atomically() {
        let mut app = App::new("test-model".into(), "test".into());
        app.set_input("hello [Image #2] world".into());
        app.editor.cursor_pos = "hello [Image #2]".len();
        app.editor.pending_images = vec![PendingImage {
            id: 2,
            png_bytes: vec![1, 2, 3],
        }];

        app.backspace();

        assert_eq!(app.input(), "hello  world");
        assert!(app.editor.pending_images.is_empty());
        assert_eq!(app.cursor_pos(), "hello ".len());
    }

    #[test]
    fn next_image_marker_id_tracks_highest_visible_marker() {
        let mut app = App::new("test-model".into(), "test".into());
        app.set_input("[Image #2] [Image #5]".into());
        app.editor.pending_images = vec![PendingImage {
            id: 2,
            png_bytes: vec![1, 2, 3],
        }];

        assert_eq!(app.next_image_marker_id(), 6);
    }

    #[test]
    fn add_pending_image_uses_highest_marker_plus_one() {
        let mut app = App::new("test-model".into(), "test".into());
        app.set_input("before [Image #4] after".into());
        app.editor.pending_images = vec![PendingImage {
            id: 2,
            png_bytes: vec![9],
        }];

        let id = app.add_pending_image(vec![1, 2, 3]);

        assert_eq!(id, 5);
        assert_eq!(app.editor.pending_images.last().map(|img| img.id), Some(5));
    }

    #[test]
    fn complete_pending_image_only_attaches_when_marker_still_exists() {
        let mut app = App::new("test-model".into(), "test".into());
        app.set_input("before [Image #3] after".into());
        app.begin_pending_image(3);

        assert!(app.complete_pending_image(3, vec![1, 2, 3]));
        assert_eq!(app.editor.pending_images.len(), 1);
        assert_eq!(app.editor.pending_images[0].id, 3);

        app.editor.input.clear();
        app.begin_pending_image(4);
        assert!(!app.complete_pending_image(4, vec![9]));
        assert!(app.editor.pending_images.iter().all(|image| image.id != 4));
    }

    #[test]
    fn fail_pending_image_removes_marker_and_inflight_state() {
        let mut app = App::new("test-model".into(), "test".into());
        app.set_input("before [Image #7] after".into());
        app.editor.cursor_pos = app.input().len();
        app.begin_pending_image(7);

        assert!(app.fail_pending_image(7));
        assert_eq!(app.input(), "before  after");
        assert!(!app.editor.inflight_image_ids.contains(&7));
    }

    #[test]
    fn pending_image_jobs_only_count_visible_markers() {
        let mut app = App::new("test-model".into(), "test".into());
        app.begin_pending_image(2);
        assert!(!app.has_pending_image_jobs());

        app.set_input("[Image #2]".into());
        app.editor.cursor_pos = app.input().len();
        assert!(app.has_pending_image_jobs());

        app.backspace();
        assert!(!app.has_pending_image_jobs());
    }

    #[test]
    fn take_prompt_response_renders_visible_user_block() {
        let mut app = App::new("test-model".into(), "test".into());
        let (tx, rx) = std::sync::mpsc::channel();
        app.show_prompt(PromptState {
            question: "Pick one".into(),
            options: vec!["red".into(), "blue".into()],
            selection: PromptSelection::Option(0),
            focus: PromptFocus::Selection,
            reply_text: String::new(),
            reply_cursor: 0,
            response_tx: tx,
        });

        let response = app.take_prompt_response();

        assert_eq!(response.as_deref(), Some("1. red"));
        assert_eq!(rx.recv().expect("response"), "1. red");
        assert!(app.prompt_state().is_none());
        assert!(app.dirty);
        assert!(matches!(
            app.blocks.last(),
            Some(DisplayBlock::UserInput(text)) if text == "1. red"
        ));
    }

    #[test]
    fn dismiss_prompt_marks_ui_dirty() {
        let mut app = App::new("test-model".into(), "test".into());
        let (tx, rx) = std::sync::mpsc::channel();
        app.show_prompt(PromptState {
            question: "Pick one".into(),
            options: vec!["red".into()],
            selection: PromptSelection::Option(0),
            focus: PromptFocus::Selection,
            reply_text: String::new(),
            reply_cursor: 0,
            response_tx: tx,
        });

        app.dismiss_prompt();

        assert_eq!(rx.recv().expect("response"), "");
        assert!(app.prompt_state().is_none());
        assert!(app.dirty);
    }

    #[test]
    fn keep_latest_user_block_visible_shows_prompt_start_before_first_token() {
        let mut app = App::new("test-model".into(), "test".into());
        app.blocks.clear();
        for idx in 0..6 {
            app.blocks
                .push(DisplayBlock::AssistantText(format!("history {idx}")));
        }
        app.blocks.push(DisplayBlock::UserInput(
            [
                "first line",
                "second line",
                "third line",
                "fourth line",
                "fifth line",
                "sixth line",
            ]
            .join("\n"),
        ));
        app.start_turn();
        app.follow_mode = FollowOutputMode::Contextual;

        let width = 32usize;
        let viewport_height = 4usize;
        app.ensure_height_cache_pub(width, viewport_height);
        app.scroll_offset = usize::MAX;

        app.keep_latest_user_block_visible();

        let last_idx = app.blocks.len() - 1;
        let max_scroll = app
            .total_content_height(width, viewport_height)
            .saturating_sub(viewport_height);
        let expected =
            app.contextual_follow_offset(app.block_content_start_offset(last_idx), max_scroll);
        assert_eq!(app.scroll_offset, expected);
    }

    #[test]
    fn keep_latest_user_block_visible_keeps_short_prompt_bottom_aligned() {
        let mut app = App::new("test-model".into(), "test".into());
        app.blocks.clear();
        for idx in 0..6 {
            app.blocks
                .push(DisplayBlock::AssistantText(format!("history {idx}")));
        }
        app.blocks
            .push(DisplayBlock::UserInput("short prompt".into()));
        app.running = true;
        app.follow_mode = FollowOutputMode::Contextual;

        let width = 32usize;
        let viewport_height = 4usize;
        app.ensure_height_cache_pub(width, viewport_height);
        app.scroll_offset = usize::MAX;

        app.keep_latest_user_block_visible();

        let cache = app.height_cache_snapshot().to_vec();
        let last_idx = app.blocks.len() - 1;
        let block_end = cache[last_idx];
        assert_eq!(app.scroll_offset, block_end.saturating_sub(viewport_height));
    }

    #[test]
    fn splash_collapses_to_compact_scrollback_height_once_history_exists() {
        let mut app = App::new("test-model".into(), "test".into());
        app.blocks
            .push(DisplayBlock::UserInput("short prompt".into()));

        let width = 32usize;
        let viewport_height = 12usize;
        app.ensure_height_cache_pub(width, viewport_height);

        let cache = app.height_cache_snapshot().to_vec();
        assert_eq!(cache[0], SPLASH_SCROLLBACK_HEIGHT);
    }

    #[test]
    fn dismiss_splash_removes_empty_state_before_history_content() {
        let mut app = App::new("test-model".into(), "test".into());
        app.dismiss_splash();

        assert!(app.blocks.is_empty());

        app.blocks.push(DisplayBlock::UserInput("hello".into()));
        app.dismiss_splash();

        assert_eq!(app.blocks.len(), 1);
        assert!(matches!(app.blocks[0], DisplayBlock::UserInput(_)));
    }

    #[test]
    fn toggle_code_expand_preserves_scroll_position() {
        let mut app = App::new("test-model".into(), "test".into());
        // Remove the Splash block
        app.blocks.clear();

        // Add a mix of blocks: prose, code, prose, code, prose
        app.blocks.push(DisplayBlock::UserInput("Message 1".into()));
        app.blocks.push(DisplayBlock::AssistantText(
            "Response 1 - a short reply".into(),
        ));
        app.blocks.push(DisplayBlock::CodeBlock {
            code: "let x = 1;\nlet y = 2;\nlet z = 3;\nprintln!();\nlet a = 4;\nlet b = 5;".into(),
            continuation: false,
        });
        app.blocks.push(DisplayBlock::Activity(dummy_activity()));
        app.blocks.push(DisplayBlock::AssistantText(
            "Response 2 - another reply that is a bit longer".into(),
        ));
        app.blocks.push(DisplayBlock::UserInput("Message 2".into()));
        app.blocks
            .push(DisplayBlock::AssistantText("Response 3".into()));
        app.blocks.push(DisplayBlock::CodeBlock {
            code: "fn main() {\n    println!(\"hello\");\n}".into(),
            continuation: false,
        });
        app.blocks.push(DisplayBlock::AssistantText(
            "Final response with some text".into(),
        ));

        let width = 80;
        let vh = 24;
        app.expand_level = 0;

        // Build the height cache (level 0 = ghost fold)
        app.ensure_height_cache_pub(width, vh);

        // Verify we have blocks and cache
        assert!(
            !app.height_cache.is_empty(),
            "height cache should be populated"
        );

        // Simulate scrolling up near AssistantText "Response 2" and ensure the same
        // anchored block stays visible across expand/collapse.
        let cache = app.height_cache_snapshot().to_vec();
        let block4_start = cache[3]; // cumulative height after first 4 blocks (0-indexed)
        let anchor_idx = cache.partition_point(|&cum| cum <= block4_start);
        app.scroll_offset = block4_start;
        app.follow_mode = FollowOutputMode::Paused;

        // Cycle to level 1 (compact plus)
        app.cycle_expand();
        assert_eq!(app.expand_level, 1);

        let new_cache = app.height_cache_snapshot().to_vec();
        let new_anchor_idx = new_cache.partition_point(|&cum| cum <= app.scroll_offset);

        assert_eq!(
            new_anchor_idx, anchor_idx,
            "scroll should keep the same anchored block after expanding to level 1"
        );

        // Cycle back to level 0
        app.cycle_expand();
        assert_eq!(app.expand_level, 0);

        let final_cache = app.height_cache_snapshot().to_vec();
        let final_anchor_idx = final_cache.partition_point(|&cum| cum <= app.scroll_offset);

        assert_eq!(
            final_anchor_idx, anchor_idx,
            "scroll should keep the same anchored block after collapsing back to level 0"
        );
    }

    #[test]
    fn toggle_code_expand_preserves_scroll_with_splash() {
        let mut app = App::new("test-model".into(), "test".into());
        // App starts with Splash at index 0 - keep it!

        // Add blocks after Splash
        app.blocks.push(DisplayBlock::UserInput("Message 1".into()));
        app.blocks.push(DisplayBlock::AssistantText(
            "Response 1 - a short reply".into(),
        ));
        app.blocks.push(DisplayBlock::CodeBlock {
            code: "let x = 1;\nlet y = 2;\nlet z = 3;\nprintln!();\nlet a = 4;\nlet b = 5;".into(),
            continuation: false,
        });
        app.blocks.push(DisplayBlock::Activity(dummy_activity()));
        app.blocks.push(DisplayBlock::AssistantText(
            "Response 2 - another reply".into(),
        ));
        app.blocks.push(DisplayBlock::UserInput("Message 2".into()));
        app.blocks
            .push(DisplayBlock::AssistantText("Response 3".into()));

        let width = 80;
        let vh = 24;

        // Build the height cache
        app.ensure_height_cache_pub(width, vh);
        let cache = app.height_cache_snapshot().to_vec();

        // Scroll to block 5 (AssistantText "Response 2", index 5 with Splash at 0)
        let block5_start = cache[4]; // after Splash + UserInput + AssistantText + CodeBlock + Activity
        app.scroll_offset = block5_start;
        app.follow_mode = FollowOutputMode::Paused;

        // Cycle to level 1
        app.cycle_expand();

        let new_cache = app.height_cache_snapshot().to_vec();
        let new_block5_start = new_cache[4];

        assert_eq!(
            app.scroll_offset, new_block5_start,
            "scroll should track block 5 start after expanding (with Splash)"
        );

        // Cycle back to level 0
        app.cycle_expand();
        let final_cache = app.height_cache_snapshot().to_vec();
        let final_block5_start = final_cache[4];

        assert_eq!(
            app.scroll_offset, final_block5_start,
            "scroll should track block 5 start after collapsing (with Splash)"
        );
    }

    #[test]
    fn toggle_code_expand_follow_output_stays_at_bottom() {
        let mut app = App::new("test-model".into(), "test".into());
        app.blocks.clear();

        app.blocks.push(DisplayBlock::UserInput("Message 1".into()));
        app.blocks
            .push(DisplayBlock::AssistantText("Response 1".into()));
        app.blocks.push(DisplayBlock::CodeBlock {
            code: "line1\nline2\nline3\nline4\nline5".into(),
            continuation: false,
        });
        app.blocks
            .push(DisplayBlock::AssistantText("Final response".into()));

        let width = 80;
        let vh = 24;

        // Simulate following from the bottom.
        app.follow_mode = FollowOutputMode::Bottom;
        app.scroll_offset = usize::MAX;
        app.ensure_height_cache_pub(width, vh);

        // Cycle to level 1
        app.cycle_expand();

        let expanded_bottom = app.total_content_height(width, vh).saturating_sub(vh);
        assert_eq!(
            app.scroll_offset, expanded_bottom,
            "should stay at the bottom after expanding"
        );
        assert_eq!(app.follow_mode, FollowOutputMode::Bottom);

        // Cycle back to level 0
        app.cycle_expand();
        let collapsed_bottom = app.total_content_height(width, vh).saturating_sub(vh);
        assert_eq!(
            app.scroll_offset, collapsed_bottom,
            "should stay at the bottom after collapsing"
        );
        assert_eq!(app.follow_mode, FollowOutputMode::Bottom);
    }

    #[test]
    fn refresh_follow_output_anchor_tracks_bottom_when_idle() {
        let mut app = App::new("test-model".into(), "test".into());
        app.blocks.clear();

        app.blocks.push(DisplayBlock::UserInput("Message 1".into()));
        app.blocks.push(DisplayBlock::AssistantText(
            "Line 1\nLine 2\nLine 3\nLine 4\nLine 5".into(),
        ));

        let width = 80;
        app.follow_mode = FollowOutputMode::Bottom;

        app.refresh_follow_output_anchor(width, 3);
        let small_bottom = app.total_content_height(width, 3).saturating_sub(3);
        assert_eq!(app.scroll_offset, small_bottom);

        app.refresh_follow_output_anchor(width, 6);
        let large_bottom = app.total_content_height(width, 6).saturating_sub(6);
        assert_eq!(app.scroll_offset, large_bottom);
    }

    #[test]
    fn refresh_follow_output_anchor_tracks_latest_output_start_while_running() {
        let mut app = App::new("test-model".into(), "test".into());
        app.blocks.clear();

        app.blocks.push(DisplayBlock::UserInput("Message 1".into()));
        app.blocks.push(DisplayBlock::AssistantText(
            "Line 1\nLine 2\nLine 3\nLine 4\nLine 5".into(),
        ));
        app.start_turn();
        app.follow_mode = FollowOutputMode::Contextual;
        if let Some(turn) = app.live_turn.as_mut() {
            turn.has_visible_output = true;
        }

        let width = 80;

        app.refresh_follow_output_anchor(width, 3);
        let small_anchor = app.follow_output_anchor_offset(width, 3);
        assert_eq!(app.scroll_offset, small_anchor);

        app.refresh_follow_output_anchor(width, 6);
        let large_anchor = app.follow_output_anchor_offset(width, 6);
        assert_eq!(app.scroll_offset, large_anchor);
    }

    #[test]
    fn refresh_follow_output_anchor_uses_first_visible_output_block_of_turn() {
        let mut app = App::new("test-model".into(), "test".into());
        app.blocks.clear();

        app.blocks
            .push(DisplayBlock::AssistantText("older history".into()));
        app.blocks.push(DisplayBlock::UserInput("prompt".into()));
        app.blocks
            .push(DisplayBlock::AssistantText("response start".into()));
        app.blocks.push(DisplayBlock::CodeBlock {
            code: "let value = 1;".into(),
            continuation: false,
        });
        app.blocks.push(DisplayBlock::CodeOutput {
            output: "ok".into(),
            error: None,
        });
        app.blocks
            .push(DisplayBlock::AssistantText("response tail".into()));
        app.start_turn();
        app.follow_mode = FollowOutputMode::Contextual;
        if let Some(turn) = app.live_turn.as_mut() {
            turn.has_visible_output = true;
        }

        let width = 80;
        let viewport_height = 6;

        app.refresh_follow_output_anchor(width, viewport_height);

        let max_scroll = app
            .total_content_height(width, viewport_height)
            .saturating_sub(viewport_height);
        let expected = app.contextual_follow_offset(app.block_content_start_offset(2), max_scroll);

        assert_eq!(app.scroll_offset, expected);
    }

    #[test]
    fn resume_follow_output_reenables_bottom_following() {
        let mut app = App::new("test-model".into(), "test".into());
        app.blocks.clear();
        app.blocks.push(DisplayBlock::UserInput("hello".into()));
        app.blocks.push(DisplayBlock::AssistantText("world".into()));
        app.follow_mode = FollowOutputMode::Paused;
        app.scroll_offset = 3;

        app.resume_follow_output();

        assert_eq!(app.follow_mode, FollowOutputMode::Bottom);
        assert_eq!(app.scroll_offset, usize::MAX);
    }

    #[test]
    fn scroll_up_from_follow_output_detaches_from_bottom_anchor() {
        let mut app = App::new("test-model".into(), "test".into());
        app.blocks.clear();
        app.blocks.push(DisplayBlock::UserInput("hello".into()));
        app.blocks.push(DisplayBlock::AssistantText(
            (0..20)
                .map(|idx| format!("line {idx}"))
                .collect::<Vec<_>>()
                .join("\n"),
        ));
        let width = 24;
        let viewport_height = 5;
        app.follow_mode = FollowOutputMode::Bottom;
        app.ensure_height_cache_pub(width, viewport_height);
        app.refresh_follow_output_anchor(width, viewport_height);

        let bottom = app.scroll_offset;
        app.scroll_up(2);

        assert_eq!(app.follow_mode, FollowOutputMode::Paused);
        assert_eq!(app.scroll_offset, bottom.saturating_sub(2));
    }

    #[test]
    fn scroll_down_to_bottom_reenables_tail_follow_instead_of_contextual_anchor() {
        let mut app = App::new("test-model".into(), "test".into());
        app.blocks.clear();
        app.blocks
            .push(DisplayBlock::AssistantText("older history".into()));
        app.blocks.push(DisplayBlock::UserInput("prompt".into()));
        app.start_turn();
        app.follow_mode = FollowOutputMode::Contextual;

        app.handle_agent_event(AgentEvent::TextDelta {
            content: (0..20)
                .map(|idx| format!("line {idx}"))
                .collect::<Vec<_>>()
                .join("\n"),
        });

        let width = 24;
        let viewport_height = 5;
        app.refresh_follow_output_anchor(width, viewport_height);
        let contextual_anchor = app.scroll_offset;

        app.scroll_up(2);
        assert_eq!(app.follow_mode, FollowOutputMode::Paused);

        app.scroll_down(usize::MAX / 2, viewport_height, width);
        assert_eq!(app.follow_mode, FollowOutputMode::Bottom);

        let max_scroll = app
            .total_content_height(width, viewport_height)
            .saturating_sub(viewport_height);
        assert!(
            contextual_anchor < max_scroll,
            "test requires contextual anchor above the tail"
        );

        app.refresh_follow_output_anchor(width, viewport_height);

        assert_eq!(app.scroll_offset, max_scroll);
    }

    #[test]
    fn text_delta_does_not_force_scroll_when_follow_output_is_paused() {
        let mut app = App::new("test-model".into(), "test".into());
        app.blocks.clear();
        app.blocks.push(DisplayBlock::UserInput("prompt".into()));
        app.start_turn();
        app.follow_mode = FollowOutputMode::Paused;
        app.scroll_offset = 3;

        app.handle_agent_event(AgentEvent::TextDelta {
            content: "streamed output".into(),
        });

        assert_eq!(app.scroll_offset, 3);
        assert_eq!(app.follow_mode, FollowOutputMode::Paused);
    }

    #[test]
    fn text_delta_follow_output_anchors_to_message_start_instead_of_tail() {
        let mut app = App::new("test-model".into(), "test".into());
        app.blocks.clear();
        app.blocks.push(DisplayBlock::UserInput("prompt".into()));
        app.start_turn();
        app.follow_mode = FollowOutputMode::Contextual;

        app.handle_agent_event(AgentEvent::TextDelta {
            content: (0..20)
                .map(|idx| format!("line {idx}"))
                .collect::<Vec<_>>()
                .join("\n"),
        });

        let width = 24;
        let viewport_height = 5;
        app.refresh_follow_output_anchor(width, viewport_height);

        let max_scroll = app
            .total_content_height(width, viewport_height)
            .saturating_sub(viewport_height);
        let expected = app.follow_output_anchor_offset(width, viewport_height);

        assert_eq!(app.scroll_offset, expected);
        assert!(
            app.scroll_offset < max_scroll,
            "follow mode should anchor above the tail for tall streamed output"
        );
    }

    #[test]
    fn refresh_follow_output_anchor_repositions_waiting_prompt_on_resize() {
        let mut app = App::new("test-model".into(), "test".into());
        app.blocks.push(DisplayBlock::UserInput(
            "A long prompt that should stay visible while we are waiting for first token output"
                .into(),
        ));
        app.running = true;
        app.follow_mode = FollowOutputMode::Contextual;

        let width = 24;
        app.refresh_follow_output_anchor(width, 3);
        let initial_offset = app.scroll_offset;

        app.refresh_follow_output_anchor(width, 6);
        assert!(
            app.scroll_offset <= initial_offset,
            "larger viewport should not push the waiting prompt further down"
        );
    }

    #[test]
    fn toggle_code_expand_stale_cache() {
        let mut app = App::new("test-model".into(), "test".into());
        app.blocks.clear();

        app.blocks.push(DisplayBlock::UserInput("Message 1".into()));
        app.blocks
            .push(DisplayBlock::AssistantText("Response 1".into()));
        app.blocks.push(DisplayBlock::CodeBlock {
            code: "line1\nline2\nline3".into(),
            continuation: false,
        });
        app.blocks
            .push(DisplayBlock::AssistantText("Response 2".into()));

        let width = 80;
        let vh = 24;

        // Build cache, then scroll to block 3 (Response 2)
        app.ensure_height_cache_pub(width, vh);
        let cache = app.height_cache_snapshot().to_vec();
        let block3_start = cache[2]; // after UserInput + AssistantText + CodeBlock
        app.scroll_offset = block3_start;
        app.follow_mode = FollowOutputMode::Paused;

        // Invalidate cache (simulates agent event arriving between frames)
        app.invalidate_height_cache();
        assert!(
            app.height_cache_snapshot().is_empty(),
            "cache should be empty"
        );

        // Cycle - should still anchor correctly despite stale cache
        app.cycle_expand();

        // Rebuild cache to check where block 3 is now
        app.ensure_height_cache_pub(width, vh);
        let new_cache = app.height_cache_snapshot().to_vec();
        let new_block3_start = new_cache[2];

        assert_eq!(
            app.scroll_offset, new_block3_start,
            "scroll should track block 3 even with stale cache"
        );
    }

    #[test]
    fn handle_tool_call_merges_contiguous_exploration_activity() {
        let mut app = App::new("test-model".into(), "test".into());
        app.blocks.clear();

        app.handle_agent_event(AgentEvent::ToolCall {
            call_id: Some("tc3".into()),
            name: "grep".into(),
            args: serde_json::json!({"pattern": "ctx", "path": "lash-cli/src"}),
            result: serde_json::json!("match"),
            success: true,
            duration_ms: 10,
        });
        app.handle_agent_event(AgentEvent::ToolCall {
            call_id: Some("tc4".into()),
            name: "read_file".into(),
            args: serde_json::json!({"path": "lash-cli/src/ui.rs"}),
            result: serde_json::json!("==> lash-cli/src/ui.rs <==\nline"),
            success: true,
            duration_ms: 5,
        });

        assert_eq!(app.blocks.len(), 1);
        match &app.blocks[0] {
            DisplayBlock::Activity(activity) => {
                assert_eq!(activity.kind, ActivityKind::Exploration);
                assert_eq!(activity.summary, "EXPLORE");
                assert_eq!(activity.children.len(), 1);
                assert!(
                    activity
                        .detail_lines
                        .iter()
                        .any(|line| line.contains("Search"))
                );
                assert!(
                    activity
                        .detail_lines
                        .iter()
                        .any(|line| line.contains("Read"))
                );
            }
            other => panic!(
                "expected activity block, got {:?}",
                other_variant_name(other)
            ),
        }
    }

    #[test]
    fn handle_tool_call_merges_contiguous_edit_activity() {
        let mut app = App::new("test-model".into(), "test".into());
        app.blocks.clear();

        app.handle_agent_event(AgentEvent::ToolCall {
            call_id: Some("tc5".into()),
            name: "apply_patch".into(),
            args: serde_json::json!({}),
            result: serde_json::json!({
                "summary": "Applied patch to 1 file",
                "added": 1,
                "removed": 1,
                "files": [{
                    "path": "a.rs",
                    "status": "modified",
                    "added": 1,
                    "removed": 1,
                    "diff": "--- a/a.rs\n+++ b/a.rs\n@@ -1,1 +1,1 @@\n-old\n+new"
                }]
            }),
            success: true,
            duration_ms: 7,
        });
        app.handle_agent_event(AgentEvent::ToolCall {
            call_id: Some("tc6".into()),
            name: "apply_patch".into(),
            args: serde_json::json!({}),
            result: serde_json::json!({
                "summary": "Applied patch to 1 file",
                "added": 2,
                "removed": 0,
                "files": [{
                    "path": "b.rs",
                    "status": "added",
                    "added": 2,
                    "removed": 0,
                    "diff": "--- a/b.rs\n+++ b/b.rs\n@@ -0,0 +1,2 @@\n+fn one() {}\n+fn two() {}"
                }]
            }),
            success: true,
            duration_ms: 5,
        });

        assert_eq!(app.blocks.len(), 1);
        match &app.blocks[0] {
            DisplayBlock::Activity(activity) => {
                assert_eq!(activity.kind, ActivityKind::Edit);
                assert_eq!(activity.summary, "Edited 2 files (+3 -1)");
                assert_eq!(activity.duration_ms, 12);
                assert_eq!(activity.children.len(), 1);
            }
            other => panic!(
                "expected activity block, got {:?}",
                other_variant_name(other)
            ),
        }
    }

    #[test]
    fn insert_text_inserts_literal_payload_at_cursor() {
        let mut app = App::new("test-model".into(), "test".into());
        app.set_input("startend".into());
        app.editor.cursor_pos = "start".len();

        app.insert_text("\nplain pasted text\n");

        assert_eq!(app.input(), "start\nplain pasted text\nend");
        assert_eq!(app.cursor_pos(), "start\nplain pasted text\n".len());
    }

    #[test]
    fn insert_pasted_text_large_uses_placeholder_and_prepare_expands() {
        let mut app = App::new("test-model".into(), "test".into());
        let large = "x".repeat(LARGE_PASTE_CHAR_THRESHOLD + 5);

        app.insert_pasted_text(&large);

        let placeholder = format!("[Pasted Content {} chars]", large.chars().count());
        assert_eq!(app.input(), placeholder);
        assert_eq!(app.editor.pending_large_pastes.len(), 1);

        let prepared = app.take_prepared_turn();
        assert_eq!(prepared.display_text, placeholder);
        assert_eq!(prepared.effective_text, large);
        assert_eq!(prepared.large_pastes.len(), 1);
        assert!(app.input().is_empty());
        assert!(app.editor.pending_large_pastes.is_empty());
    }

    #[test]
    fn backspace_deletes_large_paste_placeholder_atomically() {
        let mut app = App::new("test-model".into(), "test".into());
        let large = "x".repeat(LARGE_PASTE_CHAR_THRESHOLD + 2);

        app.insert_pasted_text(&large);
        let placeholder = app.input().to_string();
        app.editor.cursor_pos = placeholder.len();

        app.backspace();

        assert!(app.input().is_empty());
        assert!(app.editor.pending_large_pastes.is_empty());
    }

    #[test]
    fn repeated_same_size_large_pastes_get_numbered_placeholders() {
        let mut app = App::new("test-model".into(), "test".into());
        let large = "x".repeat(LARGE_PASTE_CHAR_THRESHOLD + 4);
        let base = format!("[Pasted Content {} chars]", large.chars().count());

        app.insert_pasted_text(&large);
        app.insert_pasted_text(&large);

        assert_eq!(app.input(), format!("{base}{base} #2"));
        assert_eq!(app.editor.pending_large_pastes.len(), 2);
        assert_eq!(app.editor.pending_large_pastes[0].placeholder, base);
        assert_eq!(
            app.editor.pending_large_pastes[1].placeholder,
            format!("{base} #2")
        );
    }

    #[test]
    fn prepared_turn_history_text_annotates_only_pasted_content_inline() {
        let large = format!(
            "alpha {}\nomega",
            "x".repeat(TEXT_PREVIEW_LINE_CHAR_LIMIT * 2)
        );
        let char_count = large.chars().count();
        let placeholder = format!("[Pasted Content {char_count} chars]");
        let turn = PreparedTurn::prepare_with_large_pastes(
            format!("before {placeholder} after"),
            Vec::new(),
            &SkillCatalog::default(),
            vec![LargePaste {
                placeholder,
                content: large,
            }],
        );

        let history = turn.history_text();
        assert!(history.starts_with("before "));
        assert!(history.ends_with(" after"));
        assert!(history.contains(&format!("[pasted {char_count} chars]")));
        assert!(!history.contains("[Pasted Content"));
        assert!(!history.contains('\n'));
    }

    #[test]
    fn prompt_insert_text_inserts_literal_payload_at_cursor() {
        let mut app = App::new("test-model".into(), "test".into());
        app.show_prompt(PromptState {
            question: "Question?".into(),
            options: Vec::new(),
            selection: PromptSelection::Option(0),
            focus: PromptFocus::ReplyEditor,
            reply_text: "startend".into(),
            reply_cursor: "start".len(),
            response_tx: std::sync::mpsc::channel().0,
        });

        app.prompt_insert_text("\nplain pasted text\n");

        let prompt = app.prompt_state().expect("prompt");
        assert_eq!(prompt.reply_text, "start\nplain pasted text\nend");
        assert_eq!(prompt.reply_cursor, "start\nplain pasted text\n".len());
    }

    #[test]
    fn prompt_toggle_extra_ignores_freeform_prompts() {
        let mut app = App::new("test-model".into(), "test".into());
        app.show_prompt(PromptState {
            question: "Question?".into(),
            options: Vec::new(),
            selection: PromptSelection::Option(0),
            focus: PromptFocus::ReplyEditor,
            reply_text: String::new(),
            reply_cursor: 0,
            response_tx: std::sync::mpsc::channel().0,
        });

        app.prompt_toggle_extra();

        assert!(
            app.prompt_state().expect("prompt").is_editing_reply(),
            "freeform prompts should stay in text-entry mode"
        );
    }

    #[test]
    fn live_tool_output_keeps_tail_and_counts_hidden_lines() {
        let mut app = App::new("test-model".into(), "test".into());
        app.running = true;
        app.live_turn = Some(LiveTurnState::new("shell", None));

        let payload = (0..60)
            .map(|idx| format!("line-{idx}\n"))
            .collect::<String>();
        app.handle_agent_event(AgentEvent::Message {
            text: payload,
            kind: "tool_output".into(),
        });

        assert_eq!(app.streaming_output_hidden, 12);
        assert_eq!(
            app.streaming_output.first().map(String::as_str),
            Some("line-12")
        );
        assert_eq!(
            app.streaming_output.last().map(String::as_str),
            Some("line-59")
        );
        assert_eq!(app.streaming_output_height(), 49);
    }

    fn other_variant_name(block: &DisplayBlock) -> &'static str {
        match block {
            DisplayBlock::UserInput(_) => "UserInput",
            DisplayBlock::AssistantText(_) => "AssistantText",
            DisplayBlock::CodeBlock { .. } => "CodeBlock",
            DisplayBlock::Activity(_) => "Activity",
            DisplayBlock::CodeOutput { .. } => "CodeOutput",
            DisplayBlock::ShellOutput { .. } => "ShellOutput",
            DisplayBlock::Error(_) => "Error",
            DisplayBlock::SystemMessage(_) => "SystemMessage",
            DisplayBlock::PlanContent(_) => "PlanContent",
            DisplayBlock::PluginPanel(_) => "PluginPanel",
            DisplayBlock::Splash => "Splash",
        }
    }

    fn dummy_activity() -> ActivityBlock {
        ActivityBlock {
            kind: ActivityKind::Exploration,
            status: ActivityStatus::Completed,
            tool_name: "read_file".into(),
            summary: "explored".into(),
            detail_lines: vec!["read lash-cli/src/app.rs".into()],
            duration_ms: 50,
            args: serde_json::json!({}),
            result: serde_json::Value::Null,
            artifact: None,
            children: Vec::new(),
            extra: None,
        }
    }
}
