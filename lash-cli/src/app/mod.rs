mod input;
mod projection;
mod view;

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::Arc;

use lash::{
    Message, MessageRole, PartKind, PluginMessage, PromptUsage, SessionEvent, SkillCatalog,
    TokenUsage, ToolCallRecord, UserInputProvenance, UserInputTransform, append_skill_blocks,
    collect_skill_mentions, plugin_surface_event_renders_visible_output,
};
use lash_tui::{Line, Rect};
use lash_ui::UiExtensions;

use crate::activity::{
    ActivityArtifact, ActivityBlock, ActivityKind, ActivityState, ActivityStatus,
    is_batch_tool_name, merge_edit_activity, merge_exploration_activity,
};
use crate::assistant_text::{
    normalize_assistant_text, push_assistant_text_block, render_live_assistant_text_block,
};
use crate::editor::EditorState;
use crate::overlay::{OverlayState, PickerState};
use crate::plugin_surface;
use crate::render;
use crate::repo_status::RepoStatus;
use crate::util::{is_cancelled_error, manual_interrupt_message};

use self::projection::{append_activity_block, push_system_message_block_if_new};

pub(crate) use self::projection::{
    interrupted_assistant_tail, latest_assistant_text_from_messages, preview_text_lines,
    project_interrupted_blocks, projected_blocks_from_state, smart_truncate_preview_line,
    strip_ansi_escape_sequences,
};

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PluginPanelBlock {
    pub plugin_id: String,
    pub key: String,
    pub title: String,
    pub content: String,
}

/// One row in the sticky plan dock. `status` drives the glyph + color:
/// `✓` lichen for `Done`, `■` sodium for `Active` (at most one), `□`
/// chalk-dim for `Pending`.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PlanDockItem {
    pub text: String,
    pub status: PlanDockItemStatus,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanDockItemStatus {
    Done,
    Active,
    Pending,
}

/// The persistent plan companion rendered at the bottom of the TUI
/// frame. Populated by the `plan_mode` plugin's panel events and
/// cleared when the plan is dismissed. See `docs/design-language.html`.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PlanDockState {
    pub title: String,
    /// Optional meta line shown alongside the title (e.g. `3m 3s · ↓ 1.7k tokens · thinking`).
    /// Rendered in ash-text next to the title.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<String>,
    pub items: Vec<PlanDockItem>,
}

impl PlanDockState {
    pub fn is_empty(&self) -> bool {
        self.items.is_empty() && self.title.trim().is_empty()
    }
}

pub use crate::editor::{LargePaste, PendingImage, SuggestionKind};
pub use crate::overlay::PromptState;

pub struct LiveTurnState {
    pub status_text: String,
    pub status_detail: Option<String>,
    pub phase_started_at: std::time::Instant,
    pub turn_started_at: std::time::Instant,
    pub has_visible_output: bool,
    pub output_start_anchor_pending: bool,
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
            has_visible_output: false,
            output_start_anchor_pending: false,
            transient_until: None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct LiveAssistantView {
    raw_text: String,
    display_text: String,
    rendered_lines: Vec<Line<'static>>,
    render_width: usize,
    dirty: bool,
    has_visible_text: bool,
}

impl LiveAssistantView {
    fn append(&mut self, content: &str) {
        self.raw_text.push_str(content);
        self.has_visible_text |= content.chars().any(|ch| !ch.is_whitespace());
        self.dirty = true;
    }

    fn normalized_text(&self) -> String {
        if self.dirty {
            normalize_assistant_text(&self.raw_text)
        } else {
            self.display_text.clone()
        }
    }

    fn ensure_rendered(&mut self, viewport_width: usize) {
        if !self.dirty && self.render_width == viewport_width {
            return;
        }

        self.display_text = normalize_assistant_text(&self.raw_text);
        self.rendered_lines = render_live_assistant_text_block(&self.display_text, viewport_width);
        self.render_width = viewport_width;
        self.dirty = false;
    }

    fn has_renderable_output(&self) -> bool {
        self.has_visible_text
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedTurn {
    pub draft_id: String,
    pub display_text: String,
    pub effective_text: String,
    pub images: Vec<PendingImage>,
    pub large_pastes: Vec<LargePaste>,
    pub input_provenance: UserInputProvenance,
}

impl PreparedTurn {
    #[cfg(test)]
    pub fn new(text: String, images: Vec<Vec<u8>>) -> Self {
        let display_text = text.clone();
        let effective_text = text.clone();
        Self {
            draft_id: uuid::Uuid::new_v4().to_string(),
            display_text: display_text.clone(),
            effective_text: effective_text.clone(),
            images: images
                .into_iter()
                .enumerate()
                .map(|(idx, png_bytes)| PendingImage {
                    id: idx + 1,
                    png_bytes,
                })
                .collect(),
            large_pastes: Vec::new(),
            input_provenance: UserInputProvenance {
                display_text,
                effective_text,
                transforms: Vec::new(),
            },
        }
    }

    pub fn prepare(text: String, images: Vec<PendingImage>, skills: &SkillCatalog) -> Self {
        Self::prepare_with_large_pastes(text, images, skills, Vec::new())
    }

    pub fn prepare_with_effective_text(
        display_text: String,
        effective_text: String,
        images: Vec<PendingImage>,
    ) -> Self {
        Self {
            draft_id: uuid::Uuid::new_v4().to_string(),
            display_text: display_text.clone(),
            effective_text: effective_text.clone(),
            images,
            large_pastes: Vec::new(),
            input_provenance: UserInputProvenance {
                display_text,
                effective_text,
                transforms: Vec::new(),
            },
        }
    }

    pub fn prepare_with_large_pastes(
        text: String,
        images: Vec<PendingImage>,
        skills: &SkillCatalog,
        large_pastes: Vec<LargePaste>,
    ) -> Self {
        let mut transforms = Vec::new();
        let mut seen = HashSet::new();
        for name in collect_skill_mentions(&text) {
            if seen.insert(name.clone()) && skills.get(&name).is_some() {
                let skill = skills.get(&name).expect("skill checked above");
                transforms.push(UserInputTransform::SkillBlockAppend {
                    skill_name: name,
                    skill_path: skill.path_to_skill_md.display().to_string(),
                });
            }
        }
        let large_pastes = large_pastes
            .into_iter()
            .filter(|paste| text.contains(&paste.placeholder))
            .collect::<Vec<_>>();
        for paste in &large_pastes {
            transforms.push(UserInputTransform::LargePasteExpand {
                placeholder: paste.placeholder.clone(),
                expanded_char_count: paste.content.chars().count(),
                display_replacement: format!("[pasted {} chars]", paste.content.chars().count()),
            });
        }
        let expanded_text = expand_large_paste_placeholders(&text, &large_pastes);
        let effective_text = append_skill_blocks(&expanded_text, skills);
        let display_text = text.clone();
        Self {
            draft_id: uuid::Uuid::new_v4().to_string(),
            display_text: display_text.clone(),
            effective_text: effective_text.clone(),
            images,
            large_pastes,
            input_provenance: UserInputProvenance {
                display_text,
                effective_text,
                transforms,
            },
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
        self.display_text.trim().to_string()
    }
}

/// Who owns this turn. Used by the renderer and later by any feature that
/// wants to address turns directly (fold, jump, collapse).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TurnRole {
    /// Initiated by user input (typed message, image paste, slash command).
    User,
    /// Standalone system chatter — /help output, resume notices, etc. —
    /// that isn't part of a user-initiated exchange.
    System,
}

/// A first-class turn marker carried inline in the block stream. The
/// renderer reads this to decide whether to draw a separator above the
/// turn's content. Turns are not nested containers yet (the flat storage
/// is preserved so scroll, height-cache, and push sites don't churn) —
/// but the Turn struct owns all the turn-level metadata, so future
/// features can address them by scanning between `TurnStart` markers.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Turn {
    pub role: TurnRole,
    /// Whether this turn wants a horizontal rule drawn above it. The
    /// first real turn after startup is `false`; every subsequent user
    /// turn is `true`; mid-session system turns are `false`.
    pub show_separator: bool,
}

impl Turn {
    pub fn user(show_separator: bool) -> Self {
        Self {
            role: TurnRole::User,
            show_separator,
        }
    }
}

/// A renderable block in the scrollable history.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DisplayBlock {
    /// Turn boundary. Carries the Turn metadata and is the source of
    /// truth for the separator rule (which used to be a peephole in
    /// the renderer that checked "is the next block a UserInput?").
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
    /// activities. Hidden by default; shown when the user presses
    /// `Alt+O` (expand_level 2) so the full executed program is visible.
    LashlangCode(String),
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
    #[serde(default)]
    pub interrupted_assistant_text: Option<String>,
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
            interrupted_assistant_text: None,
        }
    }
}

pub(crate) const SPLASH_CONTENT_HEIGHT: usize = 8;
pub(crate) const SPLASH_SCROLLBACK_HEIGHT: usize = SPLASH_CONTENT_HEIGHT + 2;
const STREAMING_OUTPUT_MAX_LINES: usize = 48;
const STREAMING_OUTPUT_LINE_CHAR_LIMIT: usize = 240;
const FOLLOW_OUTPUT_CONTEXT_LINES: usize = 2;

/// Fast, coarse token estimate used only for live UI counters while streaming.
fn estimate_tokens_from_char_count(chars: i64) -> i64 {
    if chars <= 0 { 0 } else { (chars + 3) / 4 }
}

#[cfg(test)]
mod tests;

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

#[derive(Clone, Debug)]
struct BlockRenderCacheEntry {
    width: usize,
    viewport_height: usize,
    expand_level: u8,
    lines: Vec<Line<'static>>,
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
    /// Transient assistant prose for the active streamed turn.
    live_assistant: Option<LiveAssistantView>,
    /// Ignore stray late TextDelta events once the latest assistant block has
    /// been reconciled to authoritative final text.
    assistant_text_finalized: bool,
    /// Whether the UI needs a redraw.
    pub dirty: bool,
    /// Output-following mode for the history viewport.
    pub follow_mode: FollowOutputMode,
    /// Cumulative height prefix sums for each block (used for O(1) total height and O(log n) lookup).
    height_cache: Vec<usize>,
    /// Earliest block index whose cached height is stale.
    height_cache_dirty_from: usize,
    /// Terminal width the height cache was computed for.
    height_cache_width: usize,
    /// Viewport height the height cache was computed for (needed for Splash centering).
    height_cache_vh: usize,
    /// Cached rendered lines for committed history blocks.
    block_render_cache: Vec<Option<BlockRenderCacheEntry>>,
    /// Owned editor/input state.
    pub editor: EditorState,
    /// Live streaming output lines from tool execution (e.g. bash).
    pub streaming_output: Vec<String>,
    pub streaming_output_hidden: usize,
    pub streaming_output_partial: String,
    /// Loaded skills registry.
    pub skills: SkillCatalog,
    /// Slash commands contributed by plugins in the current session.
    pub plugin_commands: Vec<lash::CommandDef>,
    /// Priority follow-ups entered with Enter while a turn is running.
    pub pending_steers: VecDeque<PreparedTurn>,
    /// FIFO drafts explicitly queued for later turns.
    pub queued_turns: VecDeque<PreparedTurn>,
    /// Hidden monitor-origin wake requests awaiting injection or dispatch.
    pending_monitor_wakes: VecDeque<String>,
    /// Hidden monitor wake requests already handed to the runtime bridge and
    /// awaiting checkpoint acceptance.
    in_flight_monitor_wakes: VecDeque<String>,
    /// Most recent selection-style prompt response, held briefly so the next
    /// tool result can render it inline if it exposes a question-panel artifact.
    pending_option_prompt_response: Option<String>,
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
    /// Active plan surfaced by the `plan_mode` plugin. When present it
    /// renders as a sticky dock between the history viewport and the
    /// input row instead of as an inline panel in the scroll.
    pub plan_dock: Option<PlanDockState>,
    /// UI extension registry used for slash-command completion and host actions.
    ui_extensions: Arc<UiExtensions>,
    /// Current working directory with ~ substitution.
    pub cwd: String,
    /// Handle state used to derive semantic activity rows from raw tool calls.
    pub activity_state: ActivityState,
    /// Set only when this local UI requested cancellation via Esc.
    manual_interrupt_requested: bool,
    /// Retry details to keep visible once the retry request is in flight.
    pending_retry_status: Option<String>,
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
        let persisted = UiResumeState::from_app(self);
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
        self.pending_retry_status = None;
        self.iteration = 0;
        self.live_assistant = None;
        self.assistant_text_finalized = false;
        self.clear_live_tool_output();
        self.live_output_chars_estimate = 0;
        self.live_output_tokens_estimate = 0;
        self.live_turn = Some(LiveTurnState::new("starting", None));
        self.follow_mode = FollowOutputMode::Contextual;
    }

    pub fn stop_turn(&mut self) {
        self.running = false;
        self.manual_interrupt_requested = false;
        self.pending_retry_status = None;
        self.commit_live_assistant_block();
        self.clear_live_tool_output();
        self.live_output_chars_estimate = 0;
        self.live_output_tokens_estimate = 0;
        if self.follow_mode == FollowOutputMode::Contextual {
            self.follow_mode = FollowOutputMode::Bottom;
        }
        if let Some(display) = self.pending_option_prompt_response.take() {
            self.push_prompt_response_user_block(display);
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
        self.pending_retry_status = None;
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
        if self
            .live_turn
            .as_ref()
            .is_some_and(|turn| turn.status_text == "thinking")
        {
            self.set_status("responding", None, true);
        }
    }

    fn mark_visible_output(&mut self) {
        if let Some(turn) = self.live_turn.as_mut()
            && !turn.has_visible_output
        {
            turn.has_visible_output = true;
            turn.output_start_anchor_pending =
                matches!(self.follow_mode, FollowOutputMode::Contextual);
        }
    }

    fn push_activity_block(&mut self, activity: ActivityBlock) {
        append_activity_block(&mut self.blocks, activity);
        if !self.blocks.is_empty() {
            self.invalidate_height_cache_from(self.blocks.len() - 1);
        }
    }

    fn activity_renders_prompt_response_inline(activity: &ActivityBlock) -> bool {
        matches!(
            activity.result.artifact,
            Some(ActivityArtifact::QuestionPanel(_))
        )
    }

    fn push_prompt_response_user_block(&mut self, display: String) {
        if display.trim().is_empty() {
            return;
        }
        self.blocks.push(DisplayBlock::UserInput(display));
        self.invalidate_height_cache();
        self.keep_latest_user_block_visible();
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
            live_assistant: None,
            assistant_text_finalized: false,
            dirty: true,
            follow_mode: FollowOutputMode::Bottom,
            height_cache: Vec::new(),
            height_cache_dirty_from: 0,
            height_cache_width: 0,
            height_cache_vh: 0,
            block_render_cache: Vec::new(),
            editor: EditorState::default(),
            streaming_output: Vec::new(),
            streaming_output_hidden: 0,
            streaming_output_partial: String::new(),
            skills: SkillCatalog::from_dirs(&crate::paths::default_skill_dirs()),
            plugin_commands: Vec::new(),
            pending_steers: VecDeque::new(),
            queued_turns: VecDeque::new(),
            pending_monitor_wakes: VecDeque::new(),
            in_flight_monitor_wakes: VecDeque::new(),
            pending_option_prompt_response: None,
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
            plan_dock: None,
            ui_extensions: Arc::new(UiExtensions::default()),
            cwd,
            activity_state: ActivityState::default(),
            manual_interrupt_requested: false,
            pending_retry_status: None,
            selection: TextSelection::default(),
            history_area: Rect::default(),
        }
    }

    /// Clear any active history text selection.
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
        self.height_cache_dirty_from = 0;
        self.block_render_cache.clear();
    }

    fn invalidate_height_cache_from(&mut self, idx: usize) {
        if self.blocks.is_empty() {
            self.height_cache.clear();
            self.height_cache_dirty_from = 0;
            self.block_render_cache.clear();
            return;
        }
        if self.height_cache.is_empty() {
            self.height_cache_dirty_from = 0;
        }
        if idx < self.block_render_cache.len() {
            self.block_render_cache.truncate(idx);
        }
        self.height_cache_dirty_from = self
            .height_cache_dirty_from
            .min(idx.min(self.blocks.len().saturating_sub(1)));
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
    fn live_assistant_mut(&mut self) -> &mut LiveAssistantView {
        self.live_assistant
            .get_or_insert_with(LiveAssistantView::default)
    }

    fn live_assistant_normalized_text(&self) -> Option<String> {
        self.live_assistant
            .as_ref()
            .filter(|view| view.has_renderable_output())
            .map(LiveAssistantView::normalized_text)
            .filter(|text| !text.is_empty())
    }

    fn ensure_live_assistant_rendered(&mut self, viewport_width: usize) {
        if let Some(view) = self.live_assistant.as_mut() {
            view.ensure_rendered(viewport_width);
        }
    }

    pub fn live_assistant_lines_snapshot(&self) -> Option<&[Line<'static>]> {
        self.live_assistant
            .as_ref()
            .filter(|view| !view.rendered_lines.is_empty())
            .map(|view| view.rendered_lines.as_slice())
    }

    pub(crate) fn rendered_block_lines_cached(
        &mut self,
        idx: usize,
        viewport_width: usize,
        viewport_height: usize,
    ) -> &[Line<'static>] {
        if self.block_render_cache.len() <= idx {
            self.block_render_cache.resize_with(idx + 1, || None);
        }

        let needs_refresh = self.block_render_cache[idx].as_ref().is_none_or(|entry| {
            entry.width != viewport_width
                || entry.viewport_height != viewport_height
                || entry.expand_level != self.expand_level
        });

        if needs_refresh {
            let lines = render::render_block_lines(self, idx, viewport_width, viewport_height);
            self.block_render_cache[idx] = Some(BlockRenderCacheEntry {
                width: viewport_width,
                viewport_height,
                expand_level: self.expand_level,
                lines,
            });
        }

        self.block_render_cache[idx]
            .as_ref()
            .map(|entry| entry.lines.as_slice())
            .unwrap_or(&[])
    }

    pub(crate) fn rendered_block_height_cached(
        &mut self,
        idx: usize,
        viewport_width: usize,
        viewport_height: usize,
    ) -> usize {
        self.rendered_block_lines_cached(idx, viewport_width, viewport_height)
            .len()
    }

    pub(crate) fn live_assistant_leading_padding(&self) -> usize {
        if self
            .live_assistant
            .as_ref()
            .is_none_or(|view| !view.has_renderable_output())
        {
            return 0;
        }

        match self.blocks.last() {
            Some(
                DisplayBlock::AssistantText(_)
                | DisplayBlock::AssistantReasoning(_)
                | DisplayBlock::Splash
                | DisplayBlock::TurnStart(_),
            )
            | None => 0,
            _ => 1,
        }
    }

    fn live_assistant_height(&self) -> usize {
        let Some(view) = self.live_assistant.as_ref() else {
            return 0;
        };
        if view.rendered_lines.is_empty() {
            return 0;
        }
        self.live_assistant_leading_padding() + view.rendered_lines.len()
    }

    pub(crate) fn live_tool_output_anchor_block_index(&self) -> Option<usize> {
        if self.streaming_output_height() == 0 {
            return None;
        }
        self.blocks
            .iter()
            .enumerate()
            .rev()
            .find_map(|(idx, block)| match block {
                DisplayBlock::Activity(activity)
                    if matches!(
                        activity.call.kind,
                        ActivityKind::ShellCommand
                            | ActivityKind::ShellInteraction
                            | ActivityKind::Subagent
                    ) =>
                {
                    Some(idx)
                }
                _ => None,
            })
    }

    fn invalidate_live_tool_output_cache(&mut self) {
        if let Some(idx) = self.live_tool_output_anchor_block_index() {
            self.invalidate_height_cache_from(idx);
        }
    }

    fn clear_live_tool_output(&mut self) {
        let had_output = self.streaming_output_height() > 0;
        let anchor_idx = self.live_tool_output_anchor_block_index();
        self.clear_streaming_output();
        if had_output && let Some(idx) = anchor_idx {
            self.invalidate_height_cache_from(idx);
        }
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
            let idx = self.blocks.len().saturating_sub(1);
            self.invalidate_height_cache_from(idx);
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
                let idx = self.blocks.len().saturating_sub(1);
                self.invalidate_height_cache_from(idx);
            }
            return true;
        }
        if existing.is_empty() || existing.starts_with(text) {
            return true;
        }
        false
    }

    fn commit_live_assistant_block(&mut self) {
        let Some(cleaned) = self.live_assistant_normalized_text() else {
            self.live_assistant = None;
            return;
        };

        if self.reconcile_trailing_assistant_block(&cleaned) {
            self.mark_visible_output();
            self.live_assistant = None;
            return;
        }

        if push_assistant_text_block(&mut self.blocks, &cleaned) {
            self.invalidate_height_cache_from(self.blocks.len() - 1);
            self.mark_visible_output();
        }
        self.live_assistant = None;
    }

    fn append_to_trailing_reasoning_block(&mut self, delta: &str) {
        if delta.is_empty() {
            return;
        }
        match self.blocks.last_mut() {
            Some(DisplayBlock::AssistantReasoning(existing)) => {
                existing.push_str(delta);
                let idx = self.blocks.len() - 1;
                self.invalidate_height_cache_from(idx);
            }
            _ => {
                self.blocks
                    .push(DisplayBlock::AssistantReasoning(delta.to_string()));
                self.invalidate_height_cache_from(self.blocks.len() - 1);
            }
        }
    }

    fn finalize_live_assistant(&mut self) {
        self.commit_live_assistant_block();
    }

    fn commit_final_assistant_text(&mut self, text: &str) {
        let cleaned = normalize_assistant_text(text);
        if cleaned.is_empty() {
            self.live_assistant = None;
            return;
        }

        let final_text = match self.live_assistant_normalized_text() {
            Some(existing) if !existing.is_empty() && !cleaned.starts_with(existing.as_str()) => {
                existing
            }
            _ => cleaned,
        };

        self.live_assistant = None;
        if self.reconcile_trailing_assistant_block(&final_text) {
            self.assistant_text_finalized = true;
            self.mark_visible_output();
            return;
        }

        if push_assistant_text_block(&mut self.blocks, &final_text) {
            self.invalidate_height_cache_from(self.blocks.len() - 1);
            self.mark_visible_output();
        }
        self.assistant_text_finalized = true;
    }

    fn accept_injected_turn_input(&mut self, messages: &[PluginMessage]) {
        self.finalize_live_assistant();
        let mut accepted_user_message = false;
        for message in messages {
            if !matches!(message.role, MessageRole::User) {
                continue;
            }
            accepted_user_message = true;
            if let Some(turn) = self.take_matching_pending_steer(&message.content) {
                self.push_prepared_user_input(&turn);
            } else {
                let history_text = message
                    .user_input
                    .as_ref()
                    .map(|input| input.display_text.clone())
                    .unwrap_or_else(|| message.content.clone());
                self.push_user_input_history_text(history_text);
            }
        }
        if accepted_user_message {
            self.keep_latest_user_block_visible();
        }
    }

    fn commit_injected_messages(&mut self, messages: &[PluginMessage]) {
        self.finalize_live_assistant();
        let mut committed_user_message = false;
        for message in messages {
            match message.role {
                MessageRole::User => {
                    committed_user_message = true;
                    if let Some(turn) = self.take_matching_pending_steer(&message.content) {
                        self.push_prepared_user_input(&turn);
                    } else {
                        let history_text = message
                            .user_input
                            .as_ref()
                            .map(|input| input.display_text.clone())
                            .unwrap_or_else(|| message.content.clone());
                        self.push_user_input_history_text(history_text);
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

    pub fn push_prepared_user_input(&mut self, turn: &PreparedTurn) {
        self.push_user_input_history_text(turn.history_text());
    }

    fn push_user_input_history_text(&mut self, history_text: String) {
        let history_text = history_text.trim().to_string();
        if history_text.is_empty() {
            return;
        }
        // The original UserInput is pushed immediately when the user sends
        // a steered/queued message. By the time the runtime fires the
        // checkpoint commit (e.g. codex `InjectedMessagesCommitted` after
        // post-tool checkpoint), the assistant may have already pushed
        // AssistantText / Activity blocks on top, so checking only the
        // last block lets a duplicate slip through. Scan everything since
        // the most recent user TurnStart instead.
        let scan_start = self
            .blocks
            .iter()
            .rposition(|block| {
                matches!(
                    block,
                    DisplayBlock::TurnStart(turn) if turn.role == TurnRole::User
                )
            })
            .unwrap_or(0);
        if self.blocks[scan_start..].iter().any(
            |block| matches!(block, DisplayBlock::UserInput(text) if text.trim() == history_text),
        ) {
            return;
        }
        self.blocks.push(DisplayBlock::UserInput(history_text));
        self.invalidate_height_cache_from(self.blocks.len() - 1);
    }

    /// Process a session event, updating display blocks.
    pub fn handle_session_event(&mut self, event: SessionEvent) {
        match event {
            SessionEvent::ReasoningDelta { content } => {
                if content.is_empty() {
                    return;
                }
                self.mark_first_token_arrived();
                // Render reasoning live in scrollback by extending the
                // trailing `AssistantReasoning` block. We stream directly
                // to a display block instead of buffering so the user
                // sees the thinking trace appear immediately, matching
                // pi's experience.
                self.append_to_trailing_reasoning_block(&content);
                self.mark_visible_output();
                self.scroll_to_bottom();
            }
            SessionEvent::TextDelta { content } => {
                if !self.running && self.live_turn.is_none() && self.live_assistant.is_none() {
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
                let has_renderable_output_before = self
                    .live_assistant
                    .as_ref()
                    .is_some_and(|view| view.has_renderable_output());
                self.live_assistant_mut().append(&content);
                if !has_renderable_output_before
                    && self
                        .live_assistant
                        .as_ref()
                        .is_some_and(|view| view.has_renderable_output())
                {
                    self.mark_visible_output();
                }
                self.scroll_to_bottom();
            }
            SessionEvent::ToolCall {
                name,
                args,
                result,
                success,
                duration_ms,
                ..
            } => {
                self.finalize_live_assistant();
                self.clear_live_tool_output();
                let activities = self.activity_state.blocks_for_tool_call(
                    &name,
                    args,
                    result,
                    success,
                    duration_ms,
                );
                let renders_prompt_response_inline = activities
                    .iter()
                    .any(Self::activity_renders_prompt_response_inline);
                if renders_prompt_response_inline || name == "plan_exit" {
                    self.pending_option_prompt_response = None;
                } else if let Some(display) = self.pending_option_prompt_response.take() {
                    self.push_prompt_response_user_block(display);
                }
                if let Some(activity) = activities.last() {
                    let detail = activity.result.detail_lines.first().cloned();
                    self.set_status(activity.call.summary.clone(), detail, true);
                } else if !is_batch_tool_name(&name) {
                    self.set_status(name.clone(), None, true);
                }
                for activity in activities {
                    self.push_activity_block(activity);
                }
                if !matches!(self.blocks.last(), Some(DisplayBlock::Splash)) {
                    self.mark_visible_output();
                }
                self.scroll_to_bottom();
            }
            SessionEvent::Message { text, kind } => {
                if kind == "lashlang_code" {
                    // Captured source of the fenced lashlang block the
                    // model just emitted. Pushed as its own DisplayBlock
                    // so the renderer can hide it by default and reveal
                    // it when the user hits Alt+O (full expansion).
                    self.finalize_live_assistant();
                    self.blocks.push(DisplayBlock::LashlangCode(text));
                    self.invalidate_live_tool_output_cache();
                    self.scroll_to_bottom();
                } else if kind == "tool_output" {
                    // Explicit policy:
                    // - live tool output can be disabled via env var
                    // - shell result streams can render text to the TUI
                    let current_status = self
                        .live_turn
                        .as_ref()
                        .map(|turn| turn.status_text.as_str());
                    let stream_active = self.running
                        || current_status.is_some_and(|status| status.contains("shell"));
                    if stream_active {
                        self.push_streaming_output_text(&text);
                        self.invalidate_live_tool_output_cache();
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
            SessionEvent::LlmRequest { iteration, .. } => {
                self.finalize_live_assistant();
                self.iteration = iteration + 1;
                if let Some(detail) = self.pending_retry_status.take() {
                    self.set_status("retrying", Some(detail), true);
                } else {
                    self.set_status("thinking", None, true);
                }
                self.live_output_chars_estimate = 0;
                self.live_output_tokens_estimate = 0;
                self.keep_latest_user_block_visible();
            }
            SessionEvent::RetryStatus {
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
                let retry_detail =
                    format!("attempt {}/{} · {}", attempt, max_attempts, reason_short);
                self.pending_retry_status = Some(retry_detail.clone());
                self.set_status(
                    "retrying",
                    Some(format!("in {}s · {}", wait_seconds, retry_detail)),
                    true,
                );
                self.scroll_to_bottom();
            }
            SessionEvent::Done => {
                self.finalize_live_assistant();
                self.stop_turn();
                self.scroll_to_bottom();
            }
            SessionEvent::Error { message, envelope } => {
                self.finalize_live_assistant();
                let code = envelope.as_ref().and_then(|err| err.code.as_deref());
                if is_cancelled_error(&message, code) {
                    let manual_interrupt_requested = self.manual_interrupt_requested;
                    self.stop_turn();
                    push_system_message_block_if_new(
                        &mut self.blocks,
                        if manual_interrupt_requested {
                            manual_interrupt_message().to_string()
                        } else {
                            "Cancelled.".to_string()
                        },
                    );
                } else {
                    self.manual_interrupt_requested = false;
                    self.set_transient_status(
                        "error",
                        Some(message.chars().take(96).collect()),
                        std::time::Duration::from_secs(8),
                    );
                    self.blocks.push(DisplayBlock::Error(message));
                    self.invalidate_height_cache_from(self.blocks.len() - 1);
                }
                self.mark_visible_output();
                self.scroll_to_bottom();
            }
            SessionEvent::TokenUsage {
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
            SessionEvent::ChildTokenUsage { .. } => {}
            SessionEvent::PluginEvent { plugin_id, event } => {
                let renders_visible_output = plugin_surface_event_renders_visible_output(&event);
                let mutation = plugin_surface::apply_surface_event(
                    &mut self.blocks,
                    &mut self.plugin_mode_indicators,
                    &self.plan_dock,
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
                if let Some(next_dock) = mutation.plan_dock_change {
                    // The dock steals height from the history viewport,
                    // so invalidating the cache keeps wrapping/fold
                    // positions correct.
                    self.plan_dock = next_dock.filter(|state| !state.is_empty());
                    self.invalidate_height_cache();
                    self.dirty = true;
                }
            }
            SessionEvent::InjectedTurnInputAccepted { messages, .. } => {
                self.accept_injected_turn_input(&messages);
            }
            SessionEvent::InjectedMessagesCommitted { messages, .. } => {
                self.commit_injected_messages(&messages);
            }
            SessionEvent::TypedFinish { .. } => {}
            SessionEvent::LlmResponse { .. } => {}
            SessionEvent::Prompt { .. } => {
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

    pub fn queue_monitor_wake(&mut self, input: String) {
        if input.trim().is_empty() {
            return;
        }
        self.pending_monitor_wakes.push_back(input);
    }

    pub fn has_pending_monitor_wakes(&self) -> bool {
        !self.pending_monitor_wakes.is_empty()
    }

    pub fn take_pending_monitor_wakes(&mut self) -> Vec<String> {
        self.pending_monitor_wakes.drain(..).collect::<Vec<_>>()
    }

    pub fn mark_monitor_wakes_in_flight(&mut self, wakes: &[String]) {
        self.in_flight_monitor_wakes.extend(wakes.iter().cloned());
    }

    pub fn acknowledge_monitor_wakes(&mut self, messages: &[PluginMessage]) {
        for message in messages {
            if !matches!(message.role, MessageRole::System) {
                continue;
            }
            if let Some(idx) = self
                .in_flight_monitor_wakes
                .iter()
                .position(|candidate| candidate == &message.content)
            {
                let _ = self.in_flight_monitor_wakes.remove(idx);
            }
        }
    }

    pub fn recycle_unaccepted_monitor_wakes(&mut self) {
        while let Some(wake) = self.in_flight_monitor_wakes.pop_back() {
            self.pending_monitor_wakes.push_front(wake);
        }
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

    fn take_matching_pending_steer(&mut self, content: &str) -> Option<PreparedTurn> {
        let idx = self.pending_steers.iter().position(|turn| {
            turn.display_text == content
                || turn.effective_text == content
                || (!turn.input_provenance.transforms.is_empty()
                    && content.starts_with(&turn.display_text))
        })?;
        self.pending_steers.remove(idx)
    }

    pub fn set_ui_extensions(&mut self, ui_extensions: Arc<UiExtensions>) {
        self.ui_extensions = ui_extensions;
    }

    pub fn upsert_mode_indicator(&mut self, key: impl Into<String>, label: impl Into<String>) {
        self.plugin_mode_indicators.insert(key.into(), label.into());
        self.dirty = true;
    }

    pub fn clear_mode_indicator(&mut self, key: &str) {
        if self.plugin_mode_indicators.remove(key).is_some() {
            self.dirty = true;
        }
    }

    pub fn clear_mode_indicators(&mut self) {
        if self.plugin_mode_indicators.is_empty() {
            return;
        }
        self.plugin_mode_indicators.clear();
        self.dirty = true;
    }

    pub fn set_model_variant(&mut self, model_variant: Option<String>) {
        self.model_variant = model_variant;
        self.dirty = true;
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
