mod cache;
mod input;
mod live;
mod projection;
mod queues;
mod runtime_events;
mod view;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, Mutex};

use lash_core::runtime::PendingTurnInput;
use lash_core::{
    Message, MessageRole, PartKind, PluginMessage, ProcessHandleSummary, ProcessLifecycleStatus,
    PromptUsage, SessionProcessEventKind, TokenUsage, ToolCallRecord,
};
use lash_tui::{Line, Rect};
use lash_tui_extensions::TuiExtensions;

use crate::SkillCatalog;
use crate::activity::{
    ActivityArtifact, ActivityBlock, ActivityKind, ActivityState, ActivityStatus,
    is_batch_tool_name,
};
use crate::editor::EditorState;
use crate::overlay::{OverlayState, PickerState, ProcessOverviewState};
use crate::repo_status::RepoStatus;
use crate::skill_prompt::append_skill_blocks;
use crate::stream_markdown::LiveMarkdown;

use self::cache::BlockRenderCacheEntry;
pub(super) use self::live::{LiveToolOutput, LiveTurnState};
pub(crate) use self::projection::{
    UiActivityJournal, UiActivityRecord, UiTimeline, UiTimelineItem, preview_text_lines,
    smart_truncate_preview_line, timeline_from_read_view,
};
#[cfg(test)]
pub(crate) use self::projection::{interrupted_blocks_from_read_view, strip_ansi_escape_sequences};

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToastKind {
    Info,
    Error,
}

#[derive(Clone, Debug)]
pub struct ToastState {
    pub message: String,
    pub kind: ToastKind,
    pub expires_at: std::time::Instant,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProcessView {
    pub process_id: String,
    pub kind: String,
    pub label: String,
    pub definition: Option<String>,
    pub status: ProcessLifecycleStatus,
    pub first_seen: std::time::Instant,
    pub status_duration: Option<std::time::Duration>,
    pub transient_until: Option<std::time::Instant>,
}

impl ProcessView {
    fn from_summary(
        summary: ProcessHandleSummary,
        first_seen: std::time::Instant,
        status_duration: Option<std::time::Duration>,
        transient_until: Option<std::time::Instant>,
    ) -> Self {
        let kind = summary
            .descriptor
            .kind
            .unwrap_or_else(|| "process".to_string());
        let label = summary
            .descriptor
            .label
            .unwrap_or_else(|| summary.process_id.clone());
        let definition = summary.definition.map(|definition| {
            definition
                .get("process_name")
                .and_then(|value| value.as_str())
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| definition.to_string())
        });
        Self {
            process_id: summary.process_id,
            kind,
            label,
            definition,
            status: summary.status,
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
    PinnedBottom,
    PinnedTurnStart,
    Manual,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CliRunState {
    #[default]
    Idle,
    Working,
    Thinking,
    Responding,
    RunningTool,
    Waiting,
    Error,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TurnSubmissionRoute {
    SendNow,
    InjectActiveTurn,
    QueueNextFullTurn,
    BlockedSessionSwitch,
}

impl CliRunState {
    pub fn is_runtime_active(self) -> bool {
        matches!(
            self,
            Self::Working | Self::Thinking | Self::Responding | Self::RunningTool | Self::Waiting
        )
    }

    pub fn is_injectable_runtime_phase(self) -> bool {
        self.is_runtime_active()
    }

    pub fn from_status_label(label: &str) -> Self {
        match label {
            "error" => Self::Error,
            "thinking" => Self::Thinking,
            "responding" => Self::Responding,
            "retrying" => Self::Waiting,
            status if status.contains("wait") => Self::Waiting,
            _ => Self::RunningTool,
        }
    }
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
    pub activity_journal: UiActivityJournal,
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
            activity_journal: app.ui_activity_journal.clone(),
        }
    }
}

pub(crate) const SPLASH_CONTENT_HEIGHT: usize = 8;
pub(crate) const SPLASH_SCROLLBACK_HEIGHT: usize = SPLASH_CONTENT_HEIGHT + 2;
pub(super) const FOLLOW_OUTPUT_CONTEXT_LINES: usize = 2;

#[cfg(test)]
mod tests;

/// Mouse text selection state for selectable output surfaces.
///
/// Coordinates are in **content space**: (column, virtual_row) where the
/// virtual row is relative to the owning selectable surface.
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

impl TextSelection {
    pub fn has_range(&self) -> bool {
        (self.active || self.visible) && self.anchor != self.end
    }

    pub fn has_visible_range(&self) -> bool {
        self.visible && self.anchor != self.end
    }
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

/// Pending user inputs and active-turn steers waiting to dispatch.
#[derive(Default)]
pub struct Queues {
    /// Presentation-only metadata for durable pending turn inputs. Runtime
    /// admission owns ordering and lifecycle; this cache only preserves draft
    /// text, images, and paste placeholders the runtime cannot reconstruct.
    pub draft_presentations: HashMap<String, PreparedTurn>,
    /// Latest durable pending-turn-input snapshot loaded from `LashSession`.
    pub pending_turn_input_snapshot: Vec<PendingTurnInput>,
    /// Input ids that have already been claimed for dispatch locally but may
    /// still appear in pending-input snapshots until the runtime commits them.
    pub suppressed_preview_input_ids: HashSet<String>,
    /// Most recent selection-style prompt response, held briefly so the next
    /// tool result can render it inline if it exposes a question-panel
    /// artifact.
    pub pending_option_prompt_response: Option<String>,
}

pub struct App {
    pub timeline: UiTimeline,
    pub scroll_offset: usize,
    pub expand_level: u8,
    foreground_turn_active: bool,
    pub run_state: CliRunState,
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
    /// Blank rows inserted above the transcript on the last frame so a
    /// short conversation hugs the input box instead of stranding it
    /// below an empty gap. Recomputed every `draw_history`; consumed by
    /// selection/mouse mapping so screen rows still resolve to the right
    /// transcript line. Zero whenever the content overflows the viewport.
    pub history_top_pad: usize,
    /// Incrementally maintained render/height caches for the viewport.
    render_cache: RenderCache,
    /// Owned editor/input state.
    pub editor: EditorState,
    /// Loaded skills registry.
    pub skills: SkillCatalog,
    /// Durable queue snapshot and draft presentation metadata.
    pub queues: Queues,
    /// Active overlay/picker/dialog state.
    pub overlay: Option<OverlayState>,
    /// Whether the terminal window is currently focused.
    pub focused: bool,
    /// Token/usage accounting for the current session.
    pub usage: UsageState,
    /// Active model-native variant for the current model, if any.
    pub model_variant: Option<String>,
    /// Display-facing execution mode for the active runtime.
    pub execution_mode_label: String,
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
    /// Focused background process row in the trailing process dock.
    pub selected_process_id: Option<String>,
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
    /// CLI-owned replay of UI rows that came from live Lashlang tool activity.
    pub ui_activity_journal: UiActivityJournal,
    pending_ui_activity_records: Vec<UiActivityRecord>,
    active_ui_turn_ordinal: Option<usize>,
    active_lashlang_block_ordinal: Option<usize>,
    next_lashlang_block_ordinal: usize,
    lashlang_tool_call_anchors: HashMap<String, (usize, usize)>,
    /// Set only when this local UI requested cancellation via Esc.
    manual_interrupt_requested: bool,
    /// Retry details to keep visible once the retry request is in flight.
    pending_retry_status: Option<String>,
    /// Current text selection state.
    pub selection: TextSelection,
    /// Transient local UI feedback, rendered as a top-right toast.
    pub toast: Option<ToastState>,
    /// Set after a visible selection is cleared by a mouse press on
    /// non-selectable UI. The corresponding mouse-up must be swallowed so the
    /// press does not also activate a button or plugin surface.
    suppress_mouse_up_after_selection_clear: bool,
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

    pub(crate) fn replace_ui_activity_journal(&mut self, journal: UiActivityJournal) {
        self.ui_activity_journal = journal;
    }

    pub(crate) fn pending_ui_activity_records(&self) -> &[UiActivityRecord] {
        &self.pending_ui_activity_records
    }

    pub(crate) fn clear_pending_ui_activity_records(&mut self) {
        self.pending_ui_activity_records.clear();
    }

    fn latest_ui_turn_ordinal(&self) -> usize {
        user_turn_start_indices(self.timeline.items())
            .len()
            .saturating_sub(1)
    }

    fn current_ui_turn_ordinal(&mut self) -> usize {
        match self.active_ui_turn_ordinal {
            Some(ordinal) => ordinal,
            None => {
                let ordinal = self.latest_ui_turn_ordinal();
                self.active_ui_turn_ordinal = Some(ordinal);
                ordinal
            }
        }
    }

    fn active_lashlang_activity_anchor(&mut self) -> Option<(usize, usize)> {
        let lashlang_block_ordinal = self.active_lashlang_block_ordinal?;
        Some((self.current_ui_turn_ordinal(), lashlang_block_ordinal))
    }

    fn remember_lashlang_tool_anchor(&mut self, call_id: &Option<String>, anchor: (usize, usize)) {
        if let Some(call_id) = call_id.as_ref() {
            self.lashlang_tool_call_anchors
                .insert(call_id.clone(), anchor);
        }
    }

    fn journal_lashlang_activity(
        &mut self,
        anchor: Option<(usize, usize)>,
        activity: &ActivityBlock,
    ) {
        let Some((turn_ordinal, lashlang_block_ordinal)) = anchor else {
            return;
        };
        let record = UiActivityRecord::new(turn_ordinal, lashlang_block_ordinal, activity.clone());
        self.ui_activity_journal.apply_record(record.clone());
        self.pending_ui_activity_records.push(record);
    }

    fn local_system_messages(&self) -> Vec<String> {
        self.timeline
            .iter()
            .filter_map(|block| match block {
                UiTimelineItem::SystemMessage(message) => Some(message.clone()),
                _ => None,
            })
            .collect()
    }

    fn append_missing_system_messages(timeline: &mut UiTimeline, messages: Vec<String>) {
        for message in messages {
            if timeline.iter().any(|block| {
                matches!(block, UiTimelineItem::SystemMessage(existing) if existing == &message)
            }) {
                continue;
            }
            timeline.push(UiTimelineItem::SystemMessage(message));
        }
    }

    pub fn finish_turn_from_read_view(&mut self, read_view: &lash_core::SessionReadView) {
        let current_turn_starts = user_turn_start_indices(self.timeline.items());
        let current_turn_start = current_turn_starts.last().copied();
        let local_system_messages = self.local_system_messages();

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
        Self::append_missing_system_messages(&mut self.timeline, local_system_messages);
        self.invalidate_height_cache();
        self.scroll_to_bottom();
    }

    pub fn finish_interrupted_turn_from_read_view(
        &mut self,
        read_view: &lash_core::SessionReadView,
        ui_state: &UiProjectionState,
        status_message: impl Into<String>,
    ) {
        let local_system_messages = self.local_system_messages();
        self.stop_turn();
        let mut projected_timeline = timeline_from_read_view(read_view, ui_state);
        Self::append_missing_system_messages(&mut projected_timeline, local_system_messages);
        projected_timeline.push_system_message_if_new(status_message.into());
        self.timeline = projected_timeline;
        self.invalidate_height_cache();
        self.scroll_to_bottom();
    }

    pub fn on_tick(&mut self) {
        if self.turn_active() {
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

        if self
            .toast
            .as_ref()
            .is_some_and(|toast| toast.expires_at <= std::time::Instant::now())
        {
            self.toast = None;
            self.dirty = true;
        } else if self.toast.is_some() {
            self.dirty = true;
        }

        if self.live.turn.as_ref().is_some_and(|turn| {
            turn.transient_until
                .is_some_and(|until| until <= std::time::Instant::now())
        }) {
            self.live.turn = None;
            if !self.foreground_turn_active {
                self.run_state = CliRunState::Idle;
            }
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
            timeline: Vec::new().into(),
            scroll_offset: 0,
            expand_level: 1,
            foreground_turn_active: false,
            run_state: CliRunState::Idle,
            model,
            iteration: 0,
            tick: 0,
            live: LiveTurnView::default(),
            dirty: true,
            follow_mode: FollowOutputMode::PinnedBottom,
            history_top_pad: 0,
            render_cache: RenderCache::default(),
            editor: EditorState::default(),
            skills: SkillCatalog::from_dirs(&crate::paths::default_skill_dirs()),
            queues: Queues::default(),
            overlay: None,
            focused: true,
            usage: UsageState::default(),
            model_variant: None,
            execution_mode_label: "standard".to_string(),
            session_name,
            session_id,
            repo_status: std::env::current_dir()
                .ok()
                .and_then(|cwd| crate::repo_status::detect_repo_status(&cwd)),
            plugin_mode_indicators: BTreeMap::new(),
            plan_dock: None,
            processes: Vec::new(),
            selected_process_id: None,
            ui_extensions: Arc::new(TuiExtensions::default()),
            chrome_state: Arc::new(Mutex::new(crate::chrome_ui::ChromeUiState::default())),
            cwd,
            activity_state: ActivityState::default(),
            ui_activity_journal: UiActivityJournal::default(),
            pending_ui_activity_records: Vec::new(),
            active_ui_turn_ordinal: None,
            active_lashlang_block_ordinal: None,
            next_lashlang_block_ordinal: 0,
            lashlang_tool_call_anchors: HashMap::new(),
            manual_interrupt_requested: false,
            pending_retry_status: None,
            selection: TextSelection::default(),
            toast: None,
            suppress_mouse_up_after_selection_clear: false,
            history_area: Rect::default(),
            file_index: None,
        }
    }

    pub fn show_toast(&mut self, message: impl Into<String>, kind: ToastKind) {
        self.toast = Some(ToastState {
            message: message.into(),
            kind,
            expires_at: std::time::Instant::now() + std::time::Duration::from_secs(5),
        });
        self.dirty = true;
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

    pub fn has_output_selection(&self) -> bool {
        self.selection.has_range()
    }

    pub fn has_visible_output_selection(&self) -> bool {
        self.selection.has_visible_range()
    }

    pub fn has_text_selection(&self) -> bool {
        self.has_output_selection() || self.has_input_selection()
    }

    pub fn has_visible_text_selection(&self) -> bool {
        self.has_visible_output_selection() || self.has_input_selection()
    }

    pub fn clear_text_selection(&mut self) {
        let had_selection =
            self.has_text_selection() || self.selection.active || self.input_selection_active();
        self.clear_selection();
        self.clear_input_selection();
        if had_selection {
            self.dirty = true;
        }
    }

    pub fn suppress_mouse_up_after_selection_clear(&mut self) {
        self.suppress_mouse_up_after_selection_clear = true;
    }

    pub fn take_suppressed_mouse_up_after_selection_clear(&mut self) -> bool {
        std::mem::take(&mut self.suppress_mouse_up_after_selection_clear)
    }

    /// Get the current input text and reset input state.
    pub fn take_input(&mut self) -> String {
        let text = self.editor.take_input();
        self.clear_process_selection();
        self.follow_mode = FollowOutputMode::PinnedBottom;
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
        self.ui_extensions.mount_surface(
            crate::chrome_ui::CHROME_UI_ID,
            crate::chrome_ui::turn_status_surface_spec(),
        );
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

    pub fn update_processes(&mut self, tasks: Vec<ProcessHandleSummary>) {
        let now = std::time::Instant::now();
        let previous: HashMap<String, ProcessView> = self
            .processes
            .iter()
            .cloned()
            .map(|task| (task.process_id.clone(), task))
            .collect();
        let mut next = Vec::new();
        for task in tasks {
            let old_status = previous
                .get(&task.process_id)
                .and_then(|item| item.status.terminal_state());
            let status = task.status.terminal_state();
            let first_seen = previous
                .get(&task.process_id)
                .map(|item| item.first_seen)
                .unwrap_or(now);
            let status_duration = if status.is_some() {
                previous
                    .get(&task.process_id)
                    .and_then(|item| item.status_duration)
                    .or_else(|| Some(first_seen.elapsed()))
            } else {
                None
            };
            let transient_until = if status.is_some() && old_status != status {
                Some(now + std::time::Duration::from_secs(10))
            } else {
                previous
                    .get(&task.process_id)
                    .and_then(|item| item.transient_until)
            };
            next.push(ProcessView::from_summary(
                task,
                first_seen,
                status_duration,
                transient_until,
            ));
        }
        next.retain(ProcessView::is_visible);
        if let Some(selected) = self.selected_process_id.as_deref()
            && !next.iter().any(|process| process.process_id == selected)
        {
            self.selected_process_id = None;
            self.dirty = true;
        }
        if self.processes != next {
            self.processes = next;
            self.invalidate_height_cache();
            self.dirty = true;
        }
    }

    pub fn apply_process_changed(&mut self, kind: SessionProcessEventKind, process_ids: &[String]) {
        if process_ids.is_empty() {
            return;
        }
        if kind != SessionProcessEventKind::Cancelled {
            return;
        }

        let process_ids: HashSet<&str> = process_ids.iter().map(String::as_str).collect();
        let now = std::time::Instant::now();
        let mut changed = false;
        for process in &mut self.processes {
            if !process_ids.contains(process.process_id.as_str()) || process.status.is_terminal() {
                continue;
            }
            process.status = ProcessLifecycleStatus::Cancelled;
            process
                .status_duration
                .get_or_insert_with(|| process.first_seen.elapsed());
            process.transient_until = Some(now + std::time::Duration::from_secs(10));
            changed = true;
        }

        if changed {
            self.invalidate_height_cache();
            self.dirty = true;
        }
    }

    pub fn selected_process(&self) -> Option<&ProcessView> {
        let selected = self.selected_process_id.as_deref()?;
        self.processes
            .iter()
            .find(|process| process.process_id == selected)
    }

    pub fn clear_process_selection(&mut self) {
        if self.selected_process_id.take().is_some() {
            self.dirty = true;
        }
    }

    pub fn select_next_process(&mut self) -> bool {
        self.select_process_with_delta(1)
    }

    pub fn select_previous_process(&mut self) -> bool {
        self.select_process_with_delta(-1)
    }

    fn select_process_with_delta(&mut self, delta: isize) -> bool {
        if self.processes.is_empty() {
            self.clear_process_selection();
            return false;
        }
        let current = self.selected_process_id.as_deref().and_then(|selected| {
            self.processes
                .iter()
                .position(|process| process.process_id == selected)
        });
        let len = self.processes.len() as isize;
        let next = match current {
            Some(index) => (index as isize + delta).rem_euclid(len) as usize,
            None if delta < 0 => self.processes.len() - 1,
            None => 0,
        };
        self.selected_process_id = Some(self.processes[next].process_id.clone());
        self.dirty = true;
        true
    }

    pub fn selected_process_overview_state(&self) -> Option<ProcessOverviewState> {
        let process = self.selected_process()?;
        let elapsed_duration = process
            .status_duration
            .unwrap_or_else(|| process.first_seen.elapsed());
        let elapsed =
            crate::util::format_duration_ms_if_visible(elapsed_duration.as_millis() as u64)
                .unwrap_or_else(|| "0:00".to_string());
        let mut rows = vec![
            ("id".to_string(), process.process_id.clone()),
            ("kind".to_string(), process.kind.clone()),
            ("status".to_string(), process.status.label().to_string()),
            ("elapsed".to_string(), elapsed),
        ];
        if let Some(definition) = process.definition.as_deref() {
            rows.insert(2, ("definition".to_string(), definition.to_string()));
        }
        Some(ProcessOverviewState {
            title: format!("Process {}", process.label),
            rows,
        })
    }

    pub fn set_model_variant(&mut self, model_variant: Option<String>) {
        self.model_variant = model_variant;
        self.dirty = true;
    }

    pub fn set_execution_mode_label(&mut self, mode: &crate::execution_settings::ExecutionMode) {
        self.execution_mode_label =
            crate::execution_settings::execution_mode_label(mode).to_string();
        self.dirty = true;
    }
}

pub use lash_export::transcript::format_tokens;
