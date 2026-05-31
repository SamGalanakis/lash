mod cache;
mod input;
mod live;
mod projection;
mod queues;
mod runtime_events;
mod view;

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

use lash_core::{
    Message, MessageRole, PartKind, PluginMessage, ProcessHandleGrantEntry, ProcessRecord,
    ProcessStatus, PromptUsage, TokenUsage, ToolCallRecord,
};
use lash_tui::{Line, Rect};
use lash_tui_extensions::TuiExtensions;

use crate::SkillCatalog;
use crate::activity::{
    ActivityArtifact, ActivityBlock, ActivityKind, ActivityState, ActivityStatus,
    is_batch_tool_name,
};
use crate::editor::EditorState;
use crate::overlay::{OverlayState, PickerState};
use crate::repo_status::RepoStatus;
use crate::skill_prompt::append_skill_blocks;
use crate::stream_markdown::LiveMarkdown;

use self::cache::BlockRenderCacheEntry;
pub(super) use self::live::{LiveToolOutput, LiveTurnState};
pub(crate) use self::projection::{
    UiTimeline, UiTimelineItem, interrupted_assistant_tail, interrupted_blocks_from_read_view,
    preview_text_lines, timeline_from_read_view,
};
#[cfg(test)]
pub(crate) use self::projection::{smart_truncate_preview_line, strip_ansi_escape_sequences};

fn user_turn_start_indices(blocks: &[UiTimelineItem]) -> Vec<usize> {
    blocks
        .iter()
        .enumerate()
        .filter_map(|(idx, block)| {
            matches!(
                block,
                UiTimelineItem::TurnStart(turn) if turn.role == TurnRole::User
            )
            .then_some(idx)
        })
        .collect()
}

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

#[derive(Clone, Debug, PartialEq)]
pub struct ProcessView {
    pub process_id: String,
    pub kind: String,
    pub label: String,
    pub status: ProcessStatus,
    pub first_seen: std::time::Instant,
    pub status_duration: Option<std::time::Duration>,
    pub transient_until: Option<std::time::Instant>,
}

impl ProcessView {
    fn from_record(
        grant: lash_core::ProcessHandleGrant,
        record: ProcessRecord,
        first_seen: std::time::Instant,
        status_duration: Option<std::time::Duration>,
        transient_until: Option<std::time::Instant>,
    ) -> Self {
        let kind = grant
            .descriptor
            .kind
            .unwrap_or_else(|| "process".to_string());
        let label = grant.descriptor.label.unwrap_or_else(|| record.id.clone());
        Self {
            process_id: record.id.clone(),
            kind,
            label,
            status: record.status,
            first_seen,
            status_duration,
            transient_until,
        }
    }

    pub fn is_visible(&self) -> bool {
        !self.status.is_terminal()
            || self
                .transient_until
                .is_some_and(|until| until > std::time::Instant::now())
    }
}

pub use crate::editor::{LargePaste, PendingImage, SuggestionKind};
pub use crate::overlay::PromptState;

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
pub enum PreparedInputTransform {
    LargePasteExpand {
        placeholder: String,
        expanded_char_count: usize,
        display_replacement: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedInputMetadata {
    pub display_text: String,
    pub effective_text: String,
    pub transforms: Vec<PreparedInputTransform>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedTurn {
    pub draft_id: String,
    pub display_text: String,
    pub effective_text: String,
    pub images: Vec<PendingImage>,
    pub large_pastes: Vec<LargePaste>,
    pub input_metadata: PreparedInputMetadata,
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
            input_metadata: PreparedInputMetadata {
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
            input_metadata: PreparedInputMetadata {
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
        let large_pastes = large_pastes
            .into_iter()
            .filter(|paste| text.contains(&paste.placeholder))
            .collect::<Vec<_>>();
        for paste in &large_pastes {
            transforms.push(PreparedInputTransform::LargePasteExpand {
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
            input_metadata: PreparedInputMetadata {
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

    pub fn matches_content(&self, content: &str) -> bool {
        self.display_text == content
            || self.effective_text == content
            || (!self.input_metadata.transforms.is_empty()
                && content.starts_with(&self.display_text))
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

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct UiProjectionState {
    #[serde(default)]
    pub last_response_usage: TokenUsage,
    #[serde(default)]
    pub plugin_mode_indicators: BTreeMap<String, String>,
    #[serde(default)]
    pub plugin_panels: Vec<PluginPanelBlock>,
    #[serde(default)]
    pub live_tool_output: LiveToolOutput,
    #[serde(default)]
    pub live_assistant_text: Option<String>,
    #[serde(default)]
    pub live_reasoning_text: Option<String>,
}

impl UiProjectionState {
    pub fn from_app(app: &App) -> Self {
        Self {
            last_response_usage: app.usage.last_response_usage.clone(),
            plugin_mode_indicators: app.plugin_mode_indicators.clone(),
            plugin_panels: app
                .timeline
                .iter()
                .filter_map(|block| match block {
                    UiTimelineItem::PluginPanel(panel) => Some(panel.clone()),
                    _ => None,
                })
                .collect(),
            live_tool_output: app.live.tool_output.clone(),
            live_assistant_text: app.live_assistant_normalized_text(),
            live_reasoning_text: app.live_reasoning_normalized_text(),
        }
    }
}

pub(crate) const SPLASH_CONTENT_HEIGHT: usize = 8;
pub(crate) const SPLASH_SCROLLBACK_HEIGHT: usize = SPLASH_CONTENT_HEIGHT + 2;
pub(super) const FOLLOW_OUTPUT_CONTEXT_LINES: usize = 2;

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

/// Incrementally maintained render caches for the history viewport.
///
/// Grouped out of [`App`] so render code touches one cohesive bag of
/// height/layout caches instead of five sibling fields scattered through
/// the god-struct.
#[derive(Default)]
pub struct RenderCache {
    /// Cumulative height prefix sums for each block (used for O(1) total
    /// height and O(log n) lookup).
    pub heights: Vec<usize>,
    /// Earliest block index whose cached height is stale.
    pub dirty_from: usize,
    /// Terminal width the height cache was computed for.
    pub width: usize,
    /// Viewport height the height cache was computed for (needed for Splash
    /// centering).
    pub viewport_height: usize,
    /// Cached rendered lines for committed history blocks.
    pub(crate) blocks: Vec<Option<BlockRenderCacheEntry>>,
}

/// Token/usage accounting for the active session.
///
/// Bundles the context-window accounting and live streaming estimates that
/// runtime-event code and the status strip read, so they no longer reach
/// into a flat field bag on [`App`].
#[derive(Default)]
pub struct UsageState {
    /// Cumulative token usage for the current session: parent's own LLM
    /// tokens plus the latest cumulative reported by each child session.
    /// Recomputed on every [`lash::TurnEvent::Usage`] and
    /// [`lash::TurnEvent::ChildUsage`].
    pub token_usage: TokenUsage,
    /// Parent's own LLM cumulative, kept separately so we can recompute
    /// `token_usage` when child sessions update.
    pub parent_session_cumulative: TokenUsage,
    /// Latest cumulative reported per child session (subagents, compaction,
    /// observers). Keyed by child session id.
    pub child_session_cumulatives: HashMap<String, TokenUsage>,
    /// Context window size for the current model (from models.dev).
    pub context_window: Option<u64>,
    /// Whether provider-reported input tokens exclude cached prompt tokens.
    pub context_usage_excludes_cached_input: bool,
    /// Latest completed model usage for context accounting.
    pub last_response_usage: TokenUsage,
    /// Latest normalized prompt-budget usage for context accounting and
    /// folding.
    pub last_prompt_usage: Option<PromptUsage>,
    /// Estimated output character count from live streaming chunks.
    pub live_output_chars_estimate: i64,
    /// Estimated output tokens from live streamed chunks before final usage
    /// arrives.
    pub live_output_tokens_estimate: i64,
}

/// Streaming buffers for the in-flight turn.
///
/// Holds the live status state plus the incremental markdown/tool-output
/// streams the active turn renders, separated from committed history.
pub struct LiveTurnView {
    /// Active live turn state for the bottom status strip.
    pub turn: Option<LiveTurnState>,
    /// Incremental markdown stream for assistant prose in the active turn.
    pub assistant: LiveMarkdown,
    /// Incremental markdown stream for reasoning in the active turn.
    pub reasoning: LiveMarkdown,
    /// Live tool output preview anchored to the active tool activity block.
    pub tool_output: LiveToolOutput,
}

impl Default for LiveTurnView {
    fn default() -> Self {
        Self {
            turn: None,
            assistant: LiveMarkdown::new(crate::assistant_text::MarkdownLane::Assistant),
            reasoning: LiveMarkdown::new(crate::assistant_text::MarkdownLane::Reasoning),
            tool_output: LiveToolOutput::default(),
        }
    }
}

/// Queued and steering turns waiting to dispatch.
#[derive(Default)]
pub struct Queues {
    /// Priority follow-ups entered with Enter while a turn is running.
    pub pending_steers: std::collections::VecDeque<PreparedTurn>,
    /// FIFO drafts explicitly queued for later turns.
    pub queued_turns: std::collections::VecDeque<PreparedTurn>,
    /// Most recent selection-style prompt response, held briefly so the next
    /// tool result can render it inline if it exposes a question-panel
    /// artifact.
    pub pending_option_prompt_response: Option<String>,
}

pub struct App {
    pub timeline: UiTimeline,
    pub scroll_offset: usize,
    pub expand_level: u8,
    pub running: bool,
    pub model: String,
    pub iteration: usize,
    /// Spinner frame counter
    pub tick: usize,
    /// Live-turn streaming buffers (status strip, assistant/reasoning
    /// markdown streams, and the anchored tool-output preview).
    pub live: LiveTurnView,
    /// Whether the UI needs a redraw.
    pub dirty: bool,
    /// Output-following mode for the history viewport.
    pub follow_mode: FollowOutputMode,
    /// Incrementally maintained render/height caches for the viewport.
    render_cache: RenderCache,
    /// Owned editor/input state.
    pub editor: EditorState,
    /// Loaded skills registry.
    pub skills: SkillCatalog,
    /// Queued and steering turns waiting to dispatch.
    pub queues: Queues,
    /// Active overlay/picker/dialog state.
    pub overlay: Option<OverlayState>,
    /// Whether the terminal window is currently focused.
    pub focused: bool,
    /// Token/usage accounting for the current session.
    pub usage: UsageState,
    /// Active provider-native variant for the current model, if any.
    pub model_variant: Option<String>,
    /// Unique session name (e.g. "alpine-canyon").
    pub session_name: String,
    /// Live session id (UUID) for the active runtime. Updated on resume
    /// and fork so UI sync calls target the real session.
    pub session_id: String,
    /// Repo/branch/worktree metadata for the current cwd, when available.
    pub repo_status: Option<RepoStatus>,
    /// Active plugin-owned mode indicators rendered in the input chrome.
    pub plugin_mode_indicators: BTreeMap<String, String>,
    /// Active plan surfaced by the `plan_mode` plugin. When present it
    /// renders as a sticky dock between the history viewport and the
    /// input row instead of as an inline panel in the scroll.
    pub plan_dock: Option<PlanDockState>,
    /// Snapshot of processes registered for this session.
    pub processes: Vec<ProcessView>,
    /// UI extension registry used for slash-command completion and host actions.
    ui_extensions: Arc<TuiExtensions>,
    /// Shared state for the lash-cli chrome UI extension. The scratch-tui
    /// draw loop pushes live-turn snapshots through this so the
    /// `chrome_ui` extension's `turn_status` surface participates in the
    /// regular footer layout instead of being hand-placed.
    pub chrome_state: Arc<Mutex<crate::chrome_ui::ChromeUiState>>,
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
    /// Background-walked index of the project root used for fuzzy `@`-path
    /// completion. Installed once after `App::new` by the interactive loop or
    /// tests. None disables `@`-completion entirely.
    file_index: Option<lash_file_index::FileIndex>,
}

impl App {
    pub fn ui_projection_state(&self) -> UiProjectionState {
        UiProjectionState::from_app(self)
    }

    pub fn finish_turn_from_read_view(&mut self, read_view: &lash_core::SessionReadView) {
        let current_turn_starts = user_turn_start_indices(self.timeline.items());
        let current_turn_start = current_turn_starts.last().copied();

        self.stop_turn();
        let ui_state = UiProjectionState::from_app(self);
        let projected_timeline = timeline_from_read_view(read_view, &ui_state);
        let projected_turn_starts = user_turn_start_indices(projected_timeline.items());
        let projected_turn_start = current_turn_start
            .and_then(|_| {
                current_turn_starts
                    .len()
                    .checked_sub(1)
                    .and_then(|ordinal| projected_turn_starts.get(ordinal).copied())
            })
            .or_else(|| projected_turn_starts.last().copied());

        match (current_turn_start, projected_turn_start) {
            (Some(current_start), Some(projected_start)) => {
                self.timeline.truncate(current_start);
                self.timeline.extend(
                    projected_timeline.items()[projected_start..]
                        .iter()
                        .cloned(),
                );
            }
            _ => {
                self.timeline = projected_timeline;
            }
        }
        self.invalidate_height_cache();
        self.scroll_to_bottom();
    }

    pub fn on_tick(&mut self) {
        if self.running {
            self.tick += 1;
            self.dirty = true;
        }

        let before_tasks = self.processes.len();
        self.processes.retain(ProcessView::is_visible);
        if self.processes.len() != before_tasks {
            self.invalidate_height_cache();
            self.dirty = true;
        }
        // The background dock renders a live `created_at.elapsed()` per
        // task. Without this nudge, the dock freezes between session
        // events: the tick task wakes us up but `dirty` never flips, so
        // the second-floor `Mm Ss` reading sticks until the next event
        // (often tens of seconds for slow subagents). Re-mark dirty
        // whenever we still have at least one visible process.
        if !self.processes.is_empty() {
            self.dirty = true;
        }

        if self.live.turn.as_ref().is_some_and(|turn| {
            turn.transient_until
                .is_some_and(|until| until <= std::time::Instant::now())
        }) {
            self.live.turn = None;
            self.dirty = true;
            return;
        }

        if self
            .live
            .turn
            .as_ref()
            .is_some_and(|turn| turn.transient_until.is_some())
        {
            self.dirty = true;
        }
    }

    pub fn new(model: String, session_name: String, session_id: String) -> Self {
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
            timeline: vec![UiTimelineItem::Splash].into(),
            scroll_offset: 0,
            expand_level: 1,
            running: false,
            model,
            iteration: 0,
            tick: 0,
            live: LiveTurnView::default(),
            dirty: true,
            follow_mode: FollowOutputMode::Bottom,
            render_cache: RenderCache::default(),
            editor: EditorState::default(),
            skills: SkillCatalog::from_dirs(&crate::paths::default_skill_dirs()),
            queues: Queues::default(),
            overlay: None,
            focused: true,
            usage: UsageState::default(),
            model_variant: None,
            session_name,
            session_id,
            repo_status: std::env::current_dir()
                .ok()
                .and_then(|cwd| crate::repo_status::detect_repo_status(&cwd)),
            plugin_mode_indicators: BTreeMap::new(),
            plan_dock: None,
            processes: Vec::new(),
            ui_extensions: Arc::new(TuiExtensions::default()),
            chrome_state: Arc::new(Mutex::new(crate::chrome_ui::ChromeUiState::default())),
            cwd,
            activity_state: ActivityState::default(),
            manual_interrupt_requested: false,
            pending_retry_status: None,
            selection: TextSelection::default(),
            history_area: Rect::default(),
            file_index: None,
        }
    }

    /// Install the background-walked file index used for `@`-completion.
    /// Called once after `App::new` by the interactive loop (with an
    /// `on_ready` callback that sends `AppEvent::FileIndexReady`) or by tests
    /// that build a deterministic index via `FileIndex::for_root_blocking`.
    pub fn install_file_index(&mut self, index: lash_file_index::FileIndex) {
        self.file_index = Some(index);
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

    pub fn try_take_prepared_turn(&mut self) -> Option<PreparedTurn> {
        if self.has_pending_image_jobs() {
            return None;
        }
        Some(self.take_prepared_turn())
    }

    /// Take pending images, preserving their stable inline ids for marker parsing.
    pub fn take_pending_images(&mut self) -> Vec<PendingImage> {
        self.editor.take_pending_images()
    }

    /// Take pending large-paste payloads, clearing them from the draft.
    pub fn take_large_pastes(&mut self) -> Vec<LargePaste> {
        self.editor.take_large_pastes()
    }

    /// Recompute the session-wide cumulative token usage from the parent's
    /// cumulative plus the latest cumulative reported by each child session.
    fn recompute_session_token_usage(&mut self) {
        let mut total = self.usage.parent_session_cumulative.clone();
        for child in self.usage.child_session_cumulatives.values() {
            total.add(child);
        }
        self.usage.token_usage = total;
    }

    /// Remove the empty-state splash once real conversation content is present.
    #[cfg(test)]
    pub fn dismiss_splash(&mut self) {
        if !matches!(self.timeline.first(), Some(UiTimelineItem::Splash)) {
            return;
        }
        self.timeline
            .retain(|block| !matches!(block, UiTimelineItem::Splash));
        self.invalidate_height_cache();
    }

    pub fn set_ui_extensions(&mut self, ui_extensions: Arc<TuiExtensions>) {
        self.ui_extensions = ui_extensions;
    }

    pub fn set_chrome_state(&mut self, state: Arc<Mutex<crate::chrome_ui::ChromeUiState>>) {
        self.chrome_state = state;
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

    pub fn update_processes(&mut self, tasks: Vec<ProcessHandleGrantEntry>) {
        let now = std::time::Instant::now();
        let previous: HashMap<String, ProcessView> = self
            .processes
            .iter()
            .cloned()
            .map(|task| (task.process_id.clone(), task))
            .collect();
        let mut next = Vec::new();
        for (grant, task) in tasks {
            let old_status = previous
                .get(&task.id)
                .and_then(|item| item.status.terminal_state());
            let status = task.status.terminal_state();
            let first_seen = previous
                .get(&task.id)
                .map(|item| item.first_seen)
                .unwrap_or(now);
            let status_duration = if status.is_some() {
                previous
                    .get(&task.id)
                    .and_then(|item| item.status_duration)
                    .or_else(|| Some(first_seen.elapsed()))
            } else {
                None
            };
            let transient_until = if status.is_some() && old_status != status {
                Some(now + std::time::Duration::from_secs(10))
            } else {
                previous.get(&task.id).and_then(|item| item.transient_until)
            };
            next.push(ProcessView::from_record(
                grant,
                task,
                first_seen,
                status_duration,
                transient_until,
            ));
        }
        next.retain(ProcessView::is_visible);
        if self.processes != next {
            self.processes = next;
            self.invalidate_height_cache();
            self.dirty = true;
        }
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
