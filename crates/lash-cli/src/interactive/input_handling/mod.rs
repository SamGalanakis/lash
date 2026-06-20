//! Keyboard/mouse input routing for the interactive TUI.
//!
//! - [`shortcuts`]: global shortcuts that fire regardless of mode
//! - [`modals`]: focused modal overlays (document, pickers, tree, prompt)
//! - [`mouse`]: mouse events, selection, and viewport scrolling
//! - [`turns`]: the default input composer — sending, queueing, and editing
//!   turns plus slash-command/shell-escape handling

mod modals;
mod mouse;
mod shortcuts;
mod turns;

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use crossterm::event::{Event as TermEvent, KeyEvent, KeyEventKind};
use lash::CancellationToken;
use lash::{LashSession, provider::ProviderHandle};
use lash_core::ToolState;
use lash_core::session_model::Message;
use lash_tui::{InputEvent as TuiInputEvent, Terminal, normalize_event};
use lash_tui_extensions::{TuiExtensionContext, TuiExtensions, TuiInputOutcome};

use crate::Args;
use crate::app::App;
use crate::execution_settings::ExecutionMode;
use crate::model_catalog::CachedModelCatalog;
use crate::render;
use crate::session_log::SessionLogger;
use crate::turn_runner::RuntimeRunResult;
use crate::ui_effects::{apply_ui_host_effects, push_system_message};
use crate::ui_trace::UiTraceRecorder;

use super::commands::SlashCommandCtx;
use super::helpers::{TurnReplayPayload, key_chord_from_event};

pub(super) use mouse::handle_mouse_event;
#[cfg(test)]
pub(super) use turns::selected_slash_command_suggestion;
#[cfg(test)]
pub(super) use turns::slash_command_blocked_while_working_message;

/// Bundle of the long-lived interactive-loop state every key handler
/// needs. The run loop borrows its locals into this once per key event
/// and hands `&mut SessionCtx` to [`dispatch_key_event`]; the per-mode
/// handlers reach through `ctx.field` directly rather than re-exploding
/// the bag into locals.
pub(super) struct SessionCtx<'a> {
    pub app: &'a mut App,
    pub terminal: &'a mut Terminal,
    pub ui_trace: &'a mut Option<UiTraceRecorder>,
    pub logger: &'a mut SessionLogger,
    pub args: &'a Args,
    pub paused: &'a Arc<AtomicBool>,
    pub ui_extensions: &'a TuiExtensions,
    pub runtime_factory: &'a crate::startup::session::CliSessionOpener,
    pub lash_config: &'a crate::config::LashConfig,
    pub runtime: &'a mut Option<LashSession>,
    pub history: &'a mut Vec<Message>,
    pub turn_counter: &'a mut usize,
    pub last_turn: &'a mut Option<TurnReplayPayload>,
    pub runtime_return_rx: &'a mut Option<tokio::sync::oneshot::Receiver<RuntimeRunResult>>,
    pub cancel_token: &'a mut Option<CancellationToken>,
    pub active_stream_id: &'a mut u64,
    pub provider: &'a mut ProviderHandle,
    pub current_model_variant: &'a mut Option<String>,
    pub current_execution_mode: &'a mut ExecutionMode,
    pub active_tool_state: &'a mut ToolState,
    pub model_catalog: &'a CachedModelCatalog,
    pub toolset_hash: &'a mut String,
    pub app_tx: &'a crate::event::AppEventTx,
    pub pending_clear_after_return: &'a mut bool,
}

impl SessionCtx<'_> {
    /// Reborrow this context as the slash-command context. The single place
    /// the 20+-field [`SlashCommandCtx`] is assembled.
    fn slash_ctx(&mut self) -> SlashCommandCtx<'_> {
        SlashCommandCtx {
            terminal: &mut *self.terminal,
            app: &mut *self.app,
            logger: &mut *self.logger,
            args: self.args,
            paused: self.paused,
            ui_extensions: self.ui_extensions,
            runtime_factory: self.runtime_factory,
            lash_config: self.lash_config,
            runtime: &mut *self.runtime,
            history: &mut *self.history,
            turn_counter: &mut *self.turn_counter,
            last_turn: &mut *self.last_turn,
            runtime_return_rx: &mut *self.runtime_return_rx,
            cancel_token: &mut *self.cancel_token,
            active_stream_id: &mut *self.active_stream_id,
            provider: &mut *self.provider,
            current_model_variant: &mut *self.current_model_variant,
            current_execution_mode: &mut *self.current_execution_mode,
            active_tool_state: &mut *self.active_tool_state,
            model_catalog: self.model_catalog,
            toolset_hash: &mut *self.toolset_hash,
            app_tx: self.app_tx,
            pending_clear_after_return: &mut *self.pending_clear_after_return,
        }
    }
}

/// The modal overlay that currently owns keyboard focus, derived once from
/// `app.overlay` so modal precedence lives in data instead of a duplicated
/// `if app.has_X()` ladder. Empty pickers/trees collapse to `None` so they
/// fall through to the input composer, matching the old `has_*` guards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ActiveModal {
    None,
    Document,
    SkillPicker,
    SessionPicker,
    Tree,
    ProcessOverview,
    Prompt,
}

fn active_modal(app: &App) -> ActiveModal {
    if app.has_document() {
        ActiveModal::Document
    } else if app.has_skill_picker() {
        ActiveModal::SkillPicker
    } else if app.has_session_picker() {
        ActiveModal::SessionPicker
    } else if app.has_tree() {
        ActiveModal::Tree
    } else if app.has_process_overview() {
        ActiveModal::ProcessOverview
    } else if app.has_prompt() {
        ActiveModal::Prompt
    } else {
        ActiveModal::None
    }
}

/// Dismiss the modal that currently owns focus. Returns `false` when no
/// modal is active.
fn dismiss_active_modal(app: &mut App, ui_trace: &mut Option<UiTraceRecorder>) -> bool {
    match active_modal(app) {
        ActiveModal::Document => app.dismiss_document(),
        ActiveModal::Prompt => {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_prompt_dismiss();
            }
            app.dismiss_prompt();
        }
        ActiveModal::Tree => app.dismiss_tree(),
        ActiveModal::SkillPicker => app.dismiss_skill_picker(),
        ActiveModal::SessionPicker => app.dismiss_session_picker(),
        ActiveModal::ProcessOverview => app.dismiss_process_overview(),
        ActiveModal::None => return false,
    }
    true
}

pub(super) fn handle_surface_input(
    ui_extensions: &TuiExtensions,
    event: &TuiInputEvent,
    session: Option<&LashSession>,
    app: &mut App,
) -> bool {
    if session.is_none() {
        return false;
    }
    match ui_extensions.handle_input(event, TuiExtensionContext) {
        TuiInputOutcome::Ignored => false,
        TuiInputOutcome::Handled(effects) => {
            apply_ui_host_effects(app, effects);
            app.dirty = true;
            true
        }
    }
}

/// Route a key press to the handler for the active mode. Order matters and
/// mirrors the historical fall-through: global shortcuts win first, then
/// focused modal overlays, then history scrolling, surface/shortcut routing,
/// and finally the default input dispatch.
pub(super) async fn dispatch_key_event(
    key: KeyEvent,
    ctx: &mut SessionCtx<'_>,
) -> anyhow::Result<bool> {
    // With kitty keyboard protocol, ignore Release/Repeat events
    if key.kind != KeyEventKind::Press {
        return Ok(false);
    }
    ctx.app.dirty = true;

    if let Some(result) = shortcuts::handle_global_shortcut_key(key, ctx).await? {
        return Ok(result);
    }

    match active_modal(ctx.app) {
        ActiveModal::Document => {
            modals::handle_document_key(key, ctx)?;
            return Ok(false);
        }
        ActiveModal::SkillPicker => {
            modals::handle_skill_picker_key(key, ctx);
            return Ok(false);
        }
        ActiveModal::SessionPicker => {
            modals::handle_session_picker_key(key, ctx).await;
            return Ok(false);
        }
        ActiveModal::Tree => return modals::handle_tree_key(key, ctx).await,
        ActiveModal::ProcessOverview => {
            modals::handle_process_overview_key(key, ctx).await;
            return Ok(false);
        }
        ActiveModal::Prompt => {
            modals::handle_prompt_key(key, ctx);
            return Ok(false);
        }
        ActiveModal::None => {}
    }

    let (width, height) = ctx.terminal.size()?;
    let vh = render::history_viewport_height(ctx.app, width, height);
    if let Some(action) = mouse::classify_always_on_scroll_key(key, ctx.app, vh) {
        mouse::apply_scroll_input_action(ctx.app, ctx.terminal, ctx.ui_trace, action);
        return Ok(false);
    }

    if let Some(event) = normalize_event(&TermEvent::Key(key))
        && handle_surface_input(ctx.ui_extensions, &event, ctx.runtime.as_ref(), ctx.app)
    {
        return Ok(false);
    }

    if let Some(chord) = key_chord_from_event(key)
        && let Some(shortcut) = ctx.ui_extensions.shortcut_for(chord)
    {
        if ctx.runtime.is_none() {
            push_system_message(ctx.app, "No active session for UI shortcut.".to_string());
            return Ok(false);
        }
        match ctx
            .ui_extensions
            .invoke_shortcut(&shortcut, TuiExtensionContext)
            .await
        {
            Ok(effects) => {
                crate::ui_effects::apply_ui_host_effects_with_runtime(
                    ctx.app,
                    ctx.ui_extensions,
                    ctx.runtime,
                    effects,
                )
                .await
            }
            Err(err) => push_system_message(ctx.app, err),
        }
        return Ok(false);
    }

    turns::handle_input_mode_key(key, ctx).await
}
