use lash_tui::{Frame, Line, Modifier, Rect, Span, Style, TermCapabilities};
use lash_tui_extensions::{TuiRenderContext, TuiSurfaceScene, TuiSurfaceSlot};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::app::{App, CliRunState, format_tokens};
use crate::chrome_ui::TurnStatusLabel;
#[cfg(test)]
use crate::chrome_ui::animated_lash_word;
use crate::editor::SuggestionKind;
use crate::text_layout::{centered_rect, display_width, selection_ordered};
use crate::{render, text_display, theme};

const INPUT_HORIZONTAL_PADDING: u16 = 1;
const PROMPT_HORIZONTAL_PADDING: u16 = 1;

include!("compositor/entry.rs");
include!("compositor/status.rs");
include!("compositor/history.rs");
include!("compositor/input.rs");
include!("compositor/overlays.rs");
include!("compositor/helpers.rs");
include!("compositor/tests.rs");
