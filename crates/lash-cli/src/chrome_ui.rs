//! UI extension for lash-cli's own chrome surfaces.
//!
//! Lets the bottom status indicator participate in the regular surface
//! layout instead of carving out a bespoke row in `chrome_layout`.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use lash_tui::{Frame, Line, Span};
use lash_tui_extensions::{
    TuiExtension, TuiExtensionContext, TuiHostEffect, TuiRenderContext, TuiSurfaceSize,
    TuiSurfaceSlot, TuiSurfaceSpec,
};

use crate::theme;
use crate::util::format_duration_ms_if_visible;

pub const CHROME_UI_ID: &str = "chrome_ui";
pub const TURN_STATUS_KEY: &str = "turn_status";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TurnStatusLabel {
    Idle,
    RunningTool,
    Thinking,
    Responding,
    Waiting,
    Error,
}

impl TurnStatusLabel {
    fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "Idle",
            Self::RunningTool => "Running tool",
            Self::Thinking => "Thinking",
            Self::Responding => "Responding",
            Self::Waiting => "Waiting",
            Self::Error => "Error",
        }
    }
}

#[derive(Clone, Debug)]
pub struct TurnStatusSnapshot {
    pub label: TurnStatusLabel,
    pub turn_started_at: Option<Instant>,
    pub detail: Option<String>,
}

#[derive(Default)]
pub struct ChromeUiState {
    turn: Option<TurnStatusSnapshot>,
}

pub struct ChromeTuiExtension {
    state: Arc<Mutex<ChromeUiState>>,
}

impl ChromeTuiExtension {
    pub fn new() -> (Arc<Self>, Arc<Mutex<ChromeUiState>>) {
        let state = Arc::new(Mutex::new(ChromeUiState::default()));
        let ext = Arc::new(Self {
            state: Arc::clone(&state),
        });
        (ext, state)
    }
}

pub fn turn_status_surface_spec() -> TuiSurfaceSpec {
    TuiSurfaceSpec {
        key: TURN_STATUS_KEY.to_string(),
        slot: TuiSurfaceSlot::Footer,
        size: TuiSurfaceSize::Lines(1),
        // Negative order so the indicator floats above any plugin-mounted
        // footer surfaces.
        order: -1_000,
        focusable: false,
        visible: true,
        modal: false,
    }
}

pub fn set_turn_status(state: &Mutex<ChromeUiState>, snapshot: Option<TurnStatusSnapshot>) {
    if let Ok(mut guard) = state.lock() {
        guard.turn = snapshot;
    }
}

#[async_trait]
impl TuiExtension for ChromeTuiExtension {
    fn id(&self) -> &'static str {
        CHROME_UI_ID
    }

    async fn invoke_action(
        &self,
        action: &str,
        _arg: Option<&str>,
        _ctx: TuiExtensionContext<'_>,
    ) -> Result<Vec<TuiHostEffect>, String> {
        Err(format!("unknown chrome UI action `{action}`"))
    }

    fn render_surface(&self, surface_key: &str, _ctx: TuiRenderContext<'_>, frame: &mut Frame<'_>) {
        if surface_key != TURN_STATUS_KEY {
            return;
        }
        let Ok(state) = self.state.lock() else {
            return;
        };
        let Some(turn) = state.turn.as_ref() else {
            return;
        };
        let area = frame.area();
        if area.width == 0 || area.height == 0 {
            return;
        }

        let elapsed_dur = turn
            .turn_started_at
            .map(|started| started.elapsed())
            .unwrap_or_default();
        let elapsed = turn
            .turn_started_at
            .and_then(|_| format_duration_ms_if_visible(elapsed_dur.as_millis() as u64))
            .unwrap_or_default();
        let mut spans = animated_lash_word(elapsed_dur);
        spans.push(Span::raw("  "));
        spans.push(Span::styled(
            turn.label.as_str().to_string(),
            theme::turn_status_state(),
        ));
        if let Some(detail) = turn.detail.as_deref().filter(|detail| !detail.is_empty()) {
            spans.push(Span::styled(" · ", theme::text_faint_style()));
            spans.push(Span::styled(detail.to_string(), theme::text_subtle_style()));
        }
        if !elapsed.is_empty() {
            spans.push(Span::raw("  "));
            spans.push(Span::styled(elapsed, theme::turn_status_elapsed()));
        }
        let line = Line::from(spans);
        let line_width = line.width() as u16;
        let x = area.width.saturating_sub(line_width) / 2;
        frame.write_line(x, 0, &line, area.width.saturating_sub(x));
    }
}

pub(crate) fn animated_lash_word(elapsed: std::time::Duration) -> Vec<Span<'static>> {
    let frame = ((elapsed.as_millis() / 180) % 5) as usize;
    let glyphs = match frame {
        0 => vec!['/', 'L', 'A', 'S', 'H'],
        1 => vec!['L', '/', 'A', 'S', 'H'],
        2 => vec!['L', 'A', '/', 'S', 'H'],
        3 => vec!['L', 'A', 'S', '/', 'H'],
        _ => vec!['L', 'A', 'S', 'H', '/'],
    };

    glyphs
        .into_iter()
        .map(|glyph| {
            if glyph == '/' {
                Span::styled(glyph.to_string(), theme::turn_status_slash())
            } else {
                Span::styled(glyph.to_string(), theme::turn_status_brand())
            }
        })
        .collect()
}
