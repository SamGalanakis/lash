mod activity;
mod artifact;
pub(crate) use artifact::highlight_code_snippet;
pub(crate) mod compositor;
mod prompt;
mod queue;
#[cfg(test)]
mod tests;

use crate::SkillCatalog;
use crate::skill_prompt::collect_skill_mentions_with_ranges;
use crate::text_layout::selection_ordered;
use lash_tui::{Line, Modifier, Rect, Span, Style};
use lash_tui_extensions::TuiSurfaceSlot;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::activity::{
    ActivityArtifact, ActivityBlock, ActivityKind, ActivityStatus, PatchFilePreview,
    QuestionPanelArtifact, QuestionPanelSelectionMode, SnippetPreviewArtifact, SnippetRenderMode,
    patch_file_subject, patch_status_title,
};
use crate::app::{
    App, PromptState, SPLASH_CONTENT_HEIGHT, SPLASH_SCROLLBACK_HEIGHT, UiTimelineItem,
    preview_text_lines,
};
use crate::assistant_text;
use crate::diff::render_inline_diff;
use crate::editor::EditorState;
use crate::input_items::image_marker_ranges;
use crate::markdown;
use crate::overlay::{DocumentRow, DocumentState};
use crate::text_display;
use crate::text_layout;
use crate::theme;

use self::activity::render_activity_block;
use self::artifact::{render_question_panel_artifact, render_section_panel_block};
use self::prompt::prompt_height;

pub(crate) use self::prompt::prompt_max_scroll;
pub(crate) use self::prompt::prompt_render_snapshot;
pub(crate) use self::queue::queue_preview_lines_snapshot;

const INPUT_HORIZONTAL_PADDING: u16 = 1;
const PROMPT_HORIZONTAL_PADDING: u16 = 1;
const MIN_HISTORY_HEIGHT: u16 = 3;
const MAX_INPUT_HEIGHT: u16 = 10;
const COMPACT_ACTIVITY_FEED_MAX_ITEMS: usize = 10;
const COMPACT_ACTIVITY_FEED_MAX_ROWS_PER_ITEM: usize = 2;
const COMPACT_PATCH_PREVIEW_MAX_FILES: usize = 5;
const STREAMING_OUTPUT_INLINE_MAX_ROWS: usize = 4;
const SCROLL_INDICATOR_MIN_HEIGHT: usize = 2;
const QUEUE_SECTION_ITEM_LIMIT: usize = 2;
const QUEUE_SECTION_WRAP_LIMIT: usize = 2;

pub(crate) struct InputRenderSnapshot {
    pub lines: Vec<Line<'static>>,
    pub cursor: (u16, u16),
    pub scroll_offset: usize,
    pub badge: Option<Line<'static>>,
}

include!("sections/layout.rs");
include!("sections/docks.rs");
include!("sections/history.rs");
include!("sections/input.rs");
include!("sections/blocks.rs");
include!("sections/splash_streaming.rs");
