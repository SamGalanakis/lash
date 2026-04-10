mod commands;
mod runtime;
#[cfg(test)]
mod tests;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crossterm::event::{Event as TermEvent, KeyCode, KeyEventKind, KeyModifiers};
use lash::provider::{LashConfig, ProviderKind};
use lash::session_model::Message;
use lash::*;
use lash_tui::Terminal;
use lash_ui::{
    KeyChord as UiKeyChord, KeyCode as UiKeyCode, KeyModifiers as UiKeyModifiers,
    UiCommandInvocation, UiContext, UiExtensions,
};
use tokio::sync::mpsc;
use tokio::task;
use tokio_util::sync::CancellationToken;

use crate::app::{self, App, DisplayBlock, PreparedTurn};
use crate::command;
use crate::event::AppEvent;
use crate::fork;
use crate::input_items::{build_items_from_editor_input, insert_inline_marker};
use crate::render;
use crate::resume;
use crate::session_log::{self, SessionLogger};
use crate::ui_action::{UiAction, UiActionContext, UiActionOutcome, apply_ui_action};
use crate::ui_trace::{
    UiTraceRecorder, disable_aux_op_recording, drain_aux_ops_into, enable_aux_op_recording,
    render_screen_text,
};
use crate::update;
use crate::{Args, scratch_tui, setup};
use crate::{
    apply_ui_host_effects, controls_text, copy_binding, ensure_supported_execution_mode,
    execution_mode_label, execution_mode_usage, hash12, help_text, info_text,
    latest_user_prompt_hash, normalize_prepared_turn_for_dispatch, parse_execution_mode,
    parse_model_selection, persist_live_runtime_snapshot, persist_root_session_state,
    push_system_message, queued_turn_edit_binding, resolve_model_selection, resolve_model_variant,
    shell_escape_command, sync_ui_extensions, turn_has_visible_output, validate_model_selection,
    variant_lines, version_text,
};

use self::commands::{
    dispatch_next_queued_turn, handle_parsed_slash_command, parse_slash_command,
    promote_pending_steers_to_queue, slash_command_runs_out_of_band_while_running,
};
#[cfg(test)]
pub(crate) use self::runtime::injected_image_part_indices;
use self::runtime::{
    RuntimeRunResult, apply_pending_reconfigure, copy_selected_text_or_last_response,
    make_turn_input, send_user_message,
};
pub(crate) use self::runtime::{
    generate_session_name, make_injected_plugin_message, notify_desktop,
};

#[derive(Clone)]
struct TurnReplayPayload {
    prepared_turn: PreparedTurn,
    turn_input: TurnInput,
    execution_mode: ExecutionMode,
}

struct AppEventSink {
    tx: mpsc::UnboundedSender<AppEvent>,
    stream_id: u64,
}

fn cleared_session_state(policy: SessionPolicy) -> SessionStateEnvelope {
    SessionStateEnvelope {
        policy,
        ..SessionStateEnvelope::default()
    }
}

fn log_runtime_handoff(
    phase: &str,
    app: &App,
    runtime: &Option<LashRuntime>,
    runtime_return_rx_present: bool,
    cancel_token_present: bool,
    active_stream_id: u64,
) {
    tracing::debug!(
        phase,
        app_running = app.running,
        runtime_present = runtime.is_some(),
        runtime_return_rx_present,
        cancel_token_present,
        queued_turns = app.queued_turns.len(),
        pending_steers = app.pending_steers.len(),
        active_stream_id,
        live_turn = app.live_turn.as_ref().map(|turn| turn.status_text.as_str()),
        "interactive runtime handoff"
    );
}

const TEXT_DELTA_REDRAW_INTERVAL: std::time::Duration = std::time::Duration::from_millis(33);

#[derive(Default)]
struct PendingTextDeltaBuffer {
    content: String,
    flush_at: Option<tokio::time::Instant>,
}

impl PendingTextDeltaBuffer {
    fn push(&mut self, content: String) {
        self.push_at(content, tokio::time::Instant::now());
    }

    fn push_at(&mut self, content: String, now: tokio::time::Instant) {
        if content.is_empty() {
            return;
        }
        if self.content.is_empty() {
            self.flush_at = Some(now + TEXT_DELTA_REDRAW_INTERVAL);
        }
        self.content.push_str(&content);
    }

    fn flush_deadline(&self) -> Option<tokio::time::Instant> {
        self.flush_at
    }

    fn should_flush(&self, now: tokio::time::Instant) -> bool {
        self.flush_at.is_some_and(|deadline| deadline <= now)
    }

    fn take_event(&mut self) -> Option<SessionEvent> {
        if self.content.is_empty() {
            self.flush_at = None;
            return None;
        }
        self.flush_at = None;
        Some(SessionEvent::TextDelta {
            content: std::mem::take(&mut self.content),
        })
    }
}

fn flush_pending_text_deltas(
    app: &mut App,
    pending: &mut PendingTextDeltaBuffer,
    ui_trace: Option<&mut UiTraceRecorder>,
) {
    if let Some(event) = pending.take_event() {
        if let Some(recorder) = ui_trace {
            recorder.record_session_event(&event);
        }
        app.handle_session_event(event);
        app.dirty = true;
    }
}

fn record_queue_turn(ui_trace: &mut Option<UiTraceRecorder>, turn: &PreparedTurn) {
    if let Some(recorder) = ui_trace.as_mut() {
        recorder.record_queue_turn(turn);
    }
}

fn is_copy_shortcut(key: crossterm::event::KeyEvent) -> bool {
    copy_binding().matches(key)
}

fn should_preserve_selection_for_key(key: crossterm::event::KeyEvent) -> bool {
    if is_copy_shortcut(key) {
        return true;
    }

    key.modifiers.intersects(
        KeyModifiers::SHIFT | KeyModifiers::CONTROL | KeyModifiers::ALT | KeyModifiers::SUPER,
    )
}

fn record_queue_pending_steer(ui_trace: &mut Option<UiTraceRecorder>, turn: &PreparedTurn) {
    if let Some(recorder) = ui_trace.as_mut() {
        recorder.record_queue_pending_steer(turn);
    }
}

fn drain_aux_trace_ops(ui_trace: &mut Option<UiTraceRecorder>) {
    if let Some(recorder) = ui_trace.as_mut() {
        drain_aux_ops_into(recorder);
    }
}

#[async_trait::async_trait]
impl EventSink for AppEventSink {
    async fn emit(&self, event: SessionEvent) {
        let _ = self.tx.send(AppEvent::Session {
            stream_id: self.stream_id,
            event,
        });
    }
}

fn key_chord_from_event(key: crossterm::event::KeyEvent) -> Option<UiKeyChord> {
    let code = match key.code {
        KeyCode::Tab | KeyCode::BackTab => UiKeyCode::Tab,
        KeyCode::Enter => UiKeyCode::Enter,
        KeyCode::Esc => UiKeyCode::Esc,
        KeyCode::Up => UiKeyCode::Up,
        KeyCode::Down => UiKeyCode::Down,
        KeyCode::PageUp => UiKeyCode::PageUp,
        KeyCode::PageDown => UiKeyCode::PageDown,
        KeyCode::Char(ch) => UiKeyCode::Char(ch),
        _ => return None,
    };
    Some(UiKeyChord {
        code,
        modifiers: UiKeyModifiers {
            shift: key.modifiers.contains(KeyModifiers::SHIFT)
                || matches!(key.code, KeyCode::BackTab),
            control: key.modifiers.contains(KeyModifiers::CONTROL),
            alt: key.modifiers.contains(KeyModifiers::ALT),
        },
    })
}

fn current_ui_action_context(app: &App, terminal: &Terminal) -> UiActionContext {
    let (width, height) = terminal.size().unwrap_or((80, 24));
    let viewport_height = render::history_viewport_height(app, width, height);
    let prompt_max_scroll = if let Some(prompt) = app.prompt_state() {
        let input_area_height = height.saturating_sub(
            1 + if app.live_turn.is_some() { 1 } else { 0 }
                + render::queue_preview_lines_snapshot(app, width).len() as u16,
        );
        let visible_height = input_area_height as usize;
        let inner_width = width.saturating_sub(2) as usize;
        render::prompt_max_scroll(prompt, inner_width.max(1), visible_height)
    } else {
        0
    };
    UiActionContext {
        viewport_width: width as usize,
        viewport_height,
        prompt_max_scroll,
    }
}

fn apply_terminal_action(app: &mut App, terminal: &Terminal, action: UiAction) -> UiActionOutcome {
    let context = current_ui_action_context(app, terminal);
    apply_ui_action(app, action, context)
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_app(
    mut terminal: Terminal,
    runtime: LashRuntime,
    plugin_host: PluginHost,
    dynamic_tools: Arc<DynamicToolProvider>,
    turn_injection_bridge: TurnInjectionBridge,
    logger: &mut SessionLogger,
    args: &Args,
    mut provider: Provider,
    model: String,
    initial_context_window: u64,
    session_name: String,
    model_catalog: Arc<CachedModelCatalog>,
    store: Arc<Store>,
    mut toolset_hash: String,
    initial_model_variant: Option<String>,
    initial_execution_mode: ExecutionMode,
    startup_system_message: Option<String>,
) -> anyhow::Result<()> {
    let mut app = App::new(model, session_name);
    let ui_extensions = Arc::new(
        UiExtensions::builtin()
            .map_err(|err| anyhow::anyhow!("failed to build UI extensions: {err}"))?,
    );
    app.set_ui_extensions(Arc::clone(&ui_extensions));
    app.context_window = Some(initial_context_window);
    app.context_usage_excludes_cached_input = provider.input_usage_excludes_cached_tokens();
    let mut current_model_variant = initial_model_variant.or_else(|| {
        provider
            .default_model_variant(&app.model)
            .map(str::to_string)
    });
    app.set_model_variant(current_model_variant.clone());
    let mut current_execution_mode = initial_execution_mode;
    app.load_history();
    let mut history: Vec<Message> = Vec::new();
    let mut turn_counter: usize = 0;
    let mut session_manager = runtime
        .session_manager()
        .map_err(|err| anyhow::anyhow!(err.to_string()))?;
    let mut runtime = Some(runtime);
    let mut current_context_strategy = runtime
        .as_ref()
        .map(|rt| rt.export_state().policy.context_strategy)
        .unwrap_or_else(lash::default_context_strategy);
    let mut desired_dynamic = dynamic_tools.export_state();
    let mut pending_reconfigure = false;
    let mut ui_trace = args.debug_ui_trace.as_ref().map(|path| {
        let (width, height) = terminal.size().unwrap_or((80, 24));
        UiTraceRecorder::new(
            path,
            width,
            height,
            args.debug_ui_trace_interval_ms
                .map(std::time::Duration::from_millis),
        )
    });
    if let Some(recorder) = ui_trace.as_mut() {
        recorder.capture_app_context(&app);
    }
    if ui_trace.is_some() {
        enable_aux_op_recording();
    }

    // Cancellation token for interrupting a running session
    let mut cancel_token: Option<CancellationToken> = None;

    // Unified event channel
    let (app_tx, mut app_rx) = mpsc::unbounded_channel::<AppEvent>();

    // Stop/pause flags for terminal event pumps.
    let stop = Arc::new(AtomicBool::new(false));
    let paused = Arc::new(AtomicBool::new(false));

    // Spawn terminal event reader using poll() with timeout so it can stop
    let term_tx = app_tx.clone();
    let stop_reader = Arc::clone(&stop);
    let paused_reader = Arc::clone(&paused);
    tokio::task::spawn_blocking(move || {
        while !stop_reader.load(Ordering::Relaxed) {
            if paused_reader.load(Ordering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_millis(50));
                continue;
            }
            // Poll with 50ms timeout so we can check the stop flag
            if crossterm::event::poll(std::time::Duration::from_millis(50)).unwrap_or(false) {
                match crossterm::event::read() {
                    Ok(ev) => {
                        if term_tx.send(AppEvent::Terminal(ev)).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        }
    });

    // Tick timer for spinner animation
    let tick_tx = app_tx.clone();
    let stop_tick = Arc::clone(&stop);
    let paused_tick = Arc::clone(&paused);
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_millis(100));
        loop {
            interval.tick().await;
            if stop_tick.load(Ordering::Relaxed) {
                break;
            }
            if paused_tick.load(Ordering::Relaxed) {
                continue;
            }
            if tick_tx.send(AppEvent::Tick).is_err() {
                break;
            }
        }
    });

    #[cfg(unix)]
    {
        // SIGTERM handler for graceful shutdown
        let sigterm_tx = app_tx.clone();
        tokio::spawn(async move {
            use tokio::signal::unix::{SignalKind, signal};
            if let Ok(mut sig) = signal(SignalKind::terminate()) {
                sig.recv().await;
                let _ = sigterm_tx.send(AppEvent::Quit);
            }
        });
    }

    // Oneshot for receiving runtime back after a run completes
    let mut runtime_return_rx: Option<tokio::sync::oneshot::Receiver<RuntimeRunResult>> = None;
    let mut last_turn: Option<TurnReplayPayload> = None;
    let mut active_stream_id: u64 = 0;
    let mut pending_clear_after_return = false;
    let mut pending_text_deltas = PendingTextDeltaBuffer::default();

    sync_ui_extensions(
        &mut app,
        ui_extensions.as_ref(),
        &plugin_host,
        Arc::clone(&session_manager),
    )
    .await;
    if let Some(message) = startup_system_message {
        push_system_message(&mut app, message);
    }

    if let Some(filename) = args.resume.as_deref() {
        if let Err(err) = resume::load_resumed_session(
            filename,
            &mut app,
            &mut history,
            &mut runtime,
            &mut turn_counter,
            &mut current_execution_mode,
            &mut current_context_strategy,
            &provider,
            &mut current_model_variant,
            &dynamic_tools,
            &mut desired_dynamic,
            model_catalog.as_ref(),
        )
        .await
        {
            push_system_message(&mut app, err);
        } else {
            if let Some(rt) = runtime.as_ref() {
                match rt.session_manager() {
                    Ok(manager) => session_manager = manager,
                    Err(err) => push_system_message(
                        &mut app,
                        format!("Failed to refresh session manager: {}", err),
                    ),
                }
            }
            sync_ui_extensions(
                &mut app,
                ui_extensions.as_ref(),
                &plugin_host,
                Arc::clone(&session_manager),
            )
            .await;
            toolset_hash = hash12(
                &serde_json::to_vec(&dynamic_tools.definitions())
                    .unwrap_or_else(|_| b"[]".to_vec()),
            );
        }
        if let Some(prompt) = args
            .resume_prompt
            .as_deref()
            .map(str::trim)
            .filter(|prompt| !prompt.is_empty())
        {
            if let Err(e) = apply_pending_reconfigure(
                &dynamic_tools,
                &mut desired_dynamic,
                &mut pending_reconfigure,
                &mut runtime,
            )
            .await
            {
                push_system_message(
                    &mut app,
                    format!(
                        "Pending runtime reconfigure failed; startup prompt blocked: {}",
                        e
                    ),
                );
            } else {
                toolset_hash = hash12(
                    &serde_json::to_vec(&dynamic_tools.definitions())
                        .unwrap_or_else(|_| b"[]".to_vec()),
                );
                let prepared = PreparedTurn::prepare(prompt.to_string(), Vec::new(), &app.skills);
                let (items, image_blobs) =
                    build_items_from_editor_input(&prepared.effective_text, Vec::new());
                let turn_input = make_turn_input(&prepared, items, image_blobs);
                let current_dynamic_state = dynamic_tools.export_state();
                send_user_message(
                    prepared.clone(),
                    turn_input.clone(),
                    &mut app,
                    ui_trace.as_mut(),
                    logger,
                    &mut runtime,
                    &mut history,
                    &mut runtime_return_rx,
                    &mut cancel_token,
                    &mut active_stream_id,
                    &app_tx,
                    &provider,
                    &current_dynamic_state,
                    &toolset_hash,
                );
                last_turn = Some(TurnReplayPayload {
                    prepared_turn: prepared,
                    turn_input,
                    execution_mode: current_execution_mode,
                });
            }
        }
    }

    let update_tx = app_tx.clone();
    tokio::spawn(async move {
        if let Some(message) = update::background_notification_message().await {
            let _ = update_tx.send(AppEvent::UpdateCheckFinished { message });
        }
    });

    loop {
        if pending_text_deltas.should_flush(tokio::time::Instant::now()) {
            flush_pending_text_deltas(&mut app, &mut pending_text_deltas, ui_trace.as_mut());
        }

        log_runtime_handoff(
            "loop_top",
            &app,
            &runtime,
            runtime_return_rx.is_some(),
            cancel_token.is_some(),
            active_stream_id,
        );

        if !app.running
            && runtime.is_some()
            && runtime_return_rx.is_none()
            && app.has_queued_messages()
        {
            tracing::debug!("dispatching queued turn from idle-ready trigger");
            dispatch_next_queued_turn(
                &mut app,
                &mut ui_trace,
                &mut terminal,
                logger,
                args,
                &paused,
                &plugin_host,
                ui_extensions.as_ref(),
                &dynamic_tools,
                &mut runtime,
                &mut history,
                &mut turn_counter,
                &mut last_turn,
                &mut runtime_return_rx,
                &mut cancel_token,
                &mut active_stream_id,
                &mut provider,
                &mut current_model_variant,
                &mut current_execution_mode,
                &mut current_context_strategy,
                &mut session_manager,
                &mut desired_dynamic,
                &mut pending_reconfigure,
                model_catalog.as_ref(),
                &mut toolset_hash,
                &app_tx,
                &mut pending_clear_after_return,
            )
            .await?;
            continue;
        }

        // Check if runtime turn completed — reclaim runtime + updated history
        if let Some(ref mut rx) = runtime_return_rx {
            match rx.try_recv() {
                Ok(done) => {
                    tracing::debug!(
                        stream_id = done.stream_id,
                        status = ?done.result.status,
                        done_reason = ?done.result.done_reason,
                        active_stream_id,
                        "runtime return received in interactive loop"
                    );
                    flush_pending_text_deltas(
                        &mut app,
                        &mut pending_text_deltas,
                        ui_trace.as_mut(),
                    );
                    runtime = Some(done.runtime);
                    if done.stream_id != active_stream_id || pending_clear_after_return {
                        if let Some(rt) = runtime.as_mut() {
                            let preserved_policy = rt.export_state().policy;
                            let _ = rt.reset_session().await;
                            rt.set_state(cleared_session_state(preserved_policy));
                        }
                        history.clear();
                        turn_counter = 0;
                        app.token_usage = TokenUsage::default();
                        app.stop_turn();
                        app.clear_mode_indicators();
                        app.set_model_variant(current_model_variant.clone());
                        runtime_return_rx = None;
                        cancel_token = None;
                        pending_clear_after_return = false;
                        app.dirty = true;
                        log_runtime_handoff(
                            "runtime_return_ignored_or_cleared",
                            &app,
                            &runtime,
                            runtime_return_rx.is_some(),
                            cancel_token.is_some(),
                            active_stream_id,
                        );
                        continue;
                    }
                    let interrupted = matches!(done.result.status, TurnStatus::Interrupted);
                    let no_visible_output = matches!(done.result.status, TurnStatus::Completed)
                        && !turn_has_visible_output(&done.result);
                    let mut state = done.result.state;
                    tracing::info!(
                        iteration = state.iteration,
                        status = ?done.result.status,
                        reason = ?done.result.done_reason,
                        assistant_chars = done.result.assistant_output.safe_text.len(),
                        plugin_visible_output = done.result.has_plugin_visible_output,
                        "runtime turn completed"
                    );
                    if no_visible_output {
                        let raw = done.result.assistant_output.raw_text.trim();
                        if raw.is_empty() {
                            push_system_message(
                                &mut app,
                                "Model returned no usable output. Use `/retry` to replay the last turn.",
                            );
                        } else {
                            let mut preview: String = raw.chars().take(48).collect();
                            if raw.chars().count() > 48 {
                                preview.push_str("...");
                            }
                            let preview = preview.replace('`', "'");
                            push_system_message(
                                &mut app,
                                format!(
                                    "Model returned malformed output (`{}`). Use `/retry` to replay the last turn.",
                                    preview
                                ),
                            );
                        }
                    }

                    // Snapshot execution-mode-local state after each completed turn so resume can restore exact state.
                    let snapshot_hash = if interrupted {
                        None
                    } else if let Some(rt) = runtime.as_mut() {
                        match rt.snapshot_execution_state().await {
                            Ok(Some(blob)) => {
                                let snapshot_hash = hash12(&blob);
                                state.execution_state_snapshot = Some(blob);
                                Some(snapshot_hash)
                            }
                            Ok(None) => None,
                            Err(e) => {
                                push_system_message(
                                    &mut app,
                                    format!(
                                        "Warning: failed to snapshot execution state for resume: {}",
                                        e
                                    ),
                                );
                                None
                            }
                        }
                    } else {
                        None
                    };

                    history = state.messages.clone();
                    turn_counter = state.iteration;
                    app.token_usage = state.token_usage.clone();
                    app.last_prompt_usage = state.last_prompt_usage.clone();
                    current_context_strategy = state.policy.context_strategy;
                    tracing::debug!(
                        stream_id = done.stream_id,
                        iteration = state.iteration,
                        status = ?done.result.status,
                        reason = ?done.result.done_reason,
                        messages = state.messages.len(),
                        blocks = app.blocks.len(),
                        had_live_turn = app.live_turn.is_some(),
                        running = app.running,
                        "reconciling completed runtime turn"
                    );
                    let persisted_execution_mode = state.policy.execution_mode;
                    let persisted_context_strategy = state.policy.context_strategy;
                    let persisted_dynamic_state = dynamic_tools.export_state();

                    if interrupted {
                        let had_manual_interrupt_message = matches!(
                            app.blocks.last(),
                            Some(DisplayBlock::SystemMessage(message))
                                if message == crate::util::manual_interrupt_message()
                        );
                        let ui_resume_state = app.ui_resume_state();
                        if let Some(context_window) = app.context_window {
                            persist_live_runtime_snapshot(
                                &store,
                                DurableTurnSnapshot {
                                    messages: state.messages.clone(),
                                    tool_calls: state.tool_calls.clone(),
                                    iteration: state.iteration,
                                },
                                &ui_resume_state,
                                &persisted_dynamic_state,
                                &provider,
                                &app.model,
                                context_window,
                                persisted_execution_mode,
                                persisted_context_strategy,
                                current_model_variant.as_deref(),
                                &toolset_hash,
                                app.token_usage.clone(),
                                app.last_prompt_usage.clone(),
                            );
                        }
                        if let Some(rt) = runtime.as_mut() {
                            rt.set_state(state.clone());
                        }
                        let interrupted_message = if had_manual_interrupt_message {
                            crate::util::manual_interrupt_message().to_string()
                        } else {
                            "Cancelled.".to_string()
                        };
                        app.stop_turn();
                        promote_pending_steers_to_queue(&mut app, &mut ui_trace);
                        if app.has_queued_messages() {
                            push_system_message(&mut app, interrupted_message);
                        } else {
                            app.blocks = app::project_interrupted_blocks(
                                &state.messages,
                                &state.tool_calls,
                                &ui_resume_state,
                                interrupted_message,
                            );
                            app.invalidate_height_cache();
                            app.scroll_to_bottom();
                        }
                        runtime_return_rx = None;
                        cancel_token = None;
                        dispatch_next_queued_turn(
                            &mut app,
                            &mut ui_trace,
                            &mut terminal,
                            logger,
                            args,
                            &paused,
                            &plugin_host,
                            ui_extensions.as_ref(),
                            &dynamic_tools,
                            &mut runtime,
                            &mut history,
                            &mut turn_counter,
                            &mut last_turn,
                            &mut runtime_return_rx,
                            &mut cancel_token,
                            &mut active_stream_id,
                            &mut provider,
                            &mut current_model_variant,
                            &mut current_execution_mode,
                            &mut current_context_strategy,
                            &mut session_manager,
                            &mut desired_dynamic,
                            &mut pending_reconfigure,
                            model_catalog.as_ref(),
                            &mut toolset_hash,
                            &app_tx,
                            &mut pending_clear_after_return,
                        )
                        .await?;
                        continue;
                    }

                    state.task_state = None;
                    let final_output = app::latest_assistant_text_from_messages(&state.messages)
                        .or_else(|| {
                            (!done.result.assistant_output.safe_text.is_empty())
                                .then(|| done.result.assistant_output.safe_text.clone())
                        });
                    let ui_resume_state =
                        app.finish_turn_for_resume_with_output(final_output.as_deref());
                    persist_root_session_state(
                        &store,
                        &mut state,
                        &ui_resume_state,
                        &persisted_dynamic_state,
                        &provider,
                        &app.model,
                        app.context_window
                            .expect("app context_window must be set before persisting state"),
                        persisted_execution_mode,
                        persisted_context_strategy,
                        current_model_variant.as_deref(),
                        &toolset_hash,
                        latest_user_prompt_hash(&history),
                        snapshot_hash,
                    );
                    if let Some(rt) = runtime.as_mut() {
                        rt.set_state(state.clone());
                    }
                    runtime_return_rx = None;
                    cancel_token = None;
                    log_runtime_handoff(
                        "runtime_return_completed",
                        &app,
                        &runtime,
                        runtime_return_rx.is_some(),
                        cancel_token.is_some(),
                        active_stream_id,
                    );
                    let leftover_injections =
                        turn_injection_bridge.drain().unwrap_or_else(|_| Vec::new());
                    if !leftover_injections.is_empty() {
                        promote_pending_steers_to_queue(&mut app, &mut ui_trace);
                    }
                    dispatch_next_queued_turn(
                        &mut app,
                        &mut ui_trace,
                        &mut terminal,
                        logger,
                        args,
                        &paused,
                        &plugin_host,
                        ui_extensions.as_ref(),
                        &dynamic_tools,
                        &mut runtime,
                        &mut history,
                        &mut turn_counter,
                        &mut last_turn,
                        &mut runtime_return_rx,
                        &mut cancel_token,
                        &mut active_stream_id,
                        &mut provider,
                        &mut current_model_variant,
                        &mut current_execution_mode,
                        &mut current_context_strategy,
                        &mut session_manager,
                        &mut desired_dynamic,
                        &mut pending_reconfigure,
                        model_catalog.as_ref(),
                        &mut toolset_hash,
                        &app_tx,
                        &mut pending_clear_after_return,
                    )
                    .await?;
                }
                Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                    tracing::debug!("runtime return channel closed before delivering runtime");
                    flush_pending_text_deltas(
                        &mut app,
                        &mut pending_text_deltas,
                        ui_trace.as_mut(),
                    );
                    app.stop_turn();
                    runtime_return_rx = None;
                    cancel_token = None;
                    log_runtime_handoff(
                        "runtime_return_channel_closed",
                        &app,
                        &runtime,
                        runtime_return_rx.is_some(),
                        cancel_token.is_some(),
                        active_stream_id,
                    );
                }
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {}
            }
        }

        // Draw only when dirty
        if app.dirty {
            // Pre-compute height cache before immutable borrow in draw
            let (width, height) = terminal.size()?;
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.set_size(width, height);
            }
            let vh = render::history_viewport_height(&app, width, height);
            let vw = width as usize;
            app.ensure_height_cache_pub(vw, vh);
            app.refresh_follow_output_anchor(vw, vh);
            // Clamp scroll_offset (especially for scroll_to_bottom's usize::MAX)
            let total = app.total_content_height(vw, vh);
            let max_scroll = total.saturating_sub(vh);
            app.scroll_offset = app.scroll_offset.min(max_scroll);

            app.history_area = render::history_area(&app, width, height);
            terminal.draw(|frame| scratch_tui::draw(frame, &mut app))?;
            if let Some(recorder) = ui_trace.as_mut() {
                let screen = render_screen_text(&mut app, width, height);
                recorder.maybe_record_render_checkpoint(&screen);
            }
            app.dirty = false;
        }

        // Wait for next event
        let event = if let Some(deadline) = pending_text_deltas.flush_deadline() {
            tokio::select! {
                biased;
                _ = tokio::time::sleep_until(deadline) => {
                    flush_pending_text_deltas(&mut app, &mut pending_text_deltas, ui_trace.as_mut());
                    continue;
                }
                maybe_event = app_rx.recv() => match maybe_event {
                    Some(event) => event,
                    None => break,
                },
            }
        } else {
            match app_rx.recv().await {
                Some(event) => event,
                None => break,
            }
        };

        if !matches!(
            &event,
            AppEvent::Session {
                event: SessionEvent::TextDelta { .. },
                ..
            }
        ) {
            flush_pending_text_deltas(&mut app, &mut pending_text_deltas, ui_trace.as_mut());
        }

        match event {
            AppEvent::Terminal(TermEvent::Paste(text)) => {
                app.dirty = true;

                if app.has_wait() {
                    continue;
                }

                if app.has_prompt() && app.is_prompt_text_entry() {
                    if let Some(recorder) = ui_trace.as_mut() {
                        recorder.record_prompt_insert_text(text.clone());
                    }
                    let _ = apply_terminal_action(
                        &mut app,
                        &terminal,
                        UiAction::PromptInsertText(text),
                    );
                    continue;
                }

                if let Some(recorder) = ui_trace.as_mut() {
                    recorder.record_input_insert_text(text.clone());
                }
                let _ = apply_terminal_action(
                    &mut app,
                    &terminal,
                    UiAction::InputInsertPastedText(text),
                );
            }
            AppEvent::ClipboardImageReady { id, png } => {
                app.dirty = true;
                match png {
                    Ok(png_bytes) => {
                        let _ = app.complete_pending_image(id, png_bytes);
                    }
                    Err(err) => {
                        let removed = app.fail_pending_image(id);
                        push_system_message(
                            &mut app,
                            if removed {
                                format!("Failed to process pasted image: {err}")
                            } else {
                                format!(
                                    "Failed to process pasted image #{id}: {err} (marker was already removed)"
                                )
                            },
                        );
                    }
                }
                app.update_suggestions();
            }
            AppEvent::UpdateCheckFinished { message } => {
                app.dirty = true;
                push_system_message(&mut app, message);
            }
            AppEvent::Terminal(TermEvent::Key(key)) => {
                // With kitty keyboard protocol, ignore Release/Repeat events
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                app.dirty = true;
                let copy_shortcut = is_copy_shortcut(key);
                tracing::debug!(
                    code = ?key.code,
                    modifiers = ?key.modifiers,
                    kind = ?key.kind,
                    state = ?key.state,
                    selection_visible = app.selection.visible,
                    selection_active = app.selection.active,
                    copy_shortcut,
                    preserve_selection = should_preserve_selection_for_key(key),
                    "received key event"
                );
                // Clear any active text selection on keypress
                if app.selection.visible && !should_preserve_selection_for_key(key) {
                    tracing::debug!("clearing selection on plain keypress");
                    let _ = apply_terminal_action(&mut app, &terminal, UiAction::ClearSelection);
                }

                // Active selection copy should win before generic Ctrl+C handling.
                if app.selection.visible && copy_shortcut {
                    tracing::debug!("selection copy took precedence over generic key handling");
                    copy_selected_text_or_last_response(&app, terminal.size().ok());
                    continue;
                }

                // CTRL+C: dismiss prompt if active, else quit
                if !key.modifiers.contains(KeyModifiers::SHIFT)
                    && key.modifiers.contains(KeyModifiers::CONTROL)
                    && matches!(key.code, KeyCode::Char(c) if c.eq_ignore_ascii_case(&'c'))
                {
                    tracing::debug!(
                        has_prompt = app.has_prompt(),
                        "handling Ctrl+C as dismiss/quit"
                    );
                    if app.has_wait() {
                        app.skip_wait();
                        continue;
                    }
                    if app.has_prompt() {
                        if let Some(recorder) = ui_trace.as_mut() {
                            recorder.record_prompt_dismiss();
                        }
                        let _ = apply_terminal_action(&mut app, &terminal, UiAction::DismissPrompt);
                        continue;
                    }
                    break;
                }

                // ALT+O: reliable full expand toggle across most terminals.
                if !app.has_wait()
                    && key.modifiers.contains(KeyModifiers::ALT)
                    && matches!(key.code, KeyCode::Char(c) if c.eq_ignore_ascii_case(&'o'))
                {
                    app.toggle_full_expand();
                    continue;
                }

                if !app.has_wait() && queued_turn_edit_binding().matches(key) {
                    if let Some((turn, _was_pending)) = app.take_last_queued_turn() {
                        app.restore_prepared_turn(turn);
                        app.update_suggestions();
                    }
                    continue;
                }

                // CTRL+O: cycle expand (0↔1)
                if !app.has_wait()
                    && key.modifiers.contains(KeyModifiers::CONTROL)
                    && matches!(key.code, KeyCode::Char(c) if c.eq_ignore_ascii_case(&'o'))
                {
                    app.cycle_expand();
                    continue;
                }

                // CTRL+Y / CTRL+SHIFT+C: copy current selection when present,
                // otherwise the last assistant response.
                if copy_shortcut {
                    tracing::debug!("copy shortcut matched without active selection precedence");
                    copy_selected_text_or_last_response(&app, terminal.size().ok());
                    continue;
                }

                // CTRL+SHIFT+V: always paste text from clipboard
                if !app.has_wait()
                    && key.modifiers.contains(KeyModifiers::CONTROL)
                    && key.modifiers.contains(KeyModifiers::SHIFT)
                    && key.code == KeyCode::Char('V')
                {
                    if let Ok(mut clipboard) = arboard::Clipboard::new()
                        && let Ok(text) = clipboard.get_text()
                    {
                        app.insert_pasted_text(&text);
                        app.update_suggestions();
                    }
                    continue;
                }

                // CTRL+V: paste image from clipboard (no text fallback)
                if !app.has_wait()
                    && key.modifiers.contains(KeyModifiers::CONTROL)
                    && key.code == KeyCode::Char('v')
                {
                    if let Ok(mut clipboard) = arboard::Clipboard::new()
                        && let Ok(img_data) = clipboard.get_image()
                    {
                        let image_id = app.next_image_marker_id();
                        let marker = format!("[Image #{}]", image_id);
                        insert_inline_marker(&mut app, &marker);
                        app.begin_pending_image(image_id);
                        app.update_suggestions();
                        let app_tx = app_tx.clone();
                        let w = img_data.width as u32;
                        let h = img_data.height as u32;
                        let bytes = img_data.bytes.into_owned();
                        tokio::spawn(async move {
                            let png = task::spawn_blocking(move || {
                                let rgba =
                                    image::RgbaImage::from_raw(w, h, bytes).ok_or_else(|| {
                                        anyhow::anyhow!("Failed to decode pasted image data.")
                                    })?;
                                let mut png_buf = std::io::Cursor::new(Vec::new());
                                rgba.write_to(&mut png_buf, image::ImageFormat::Png)
                                    .map_err(|err| {
                                        anyhow::anyhow!("Failed to encode pasted image: {err}")
                                    })?;
                                Ok::<_, anyhow::Error>(png_buf.into_inner())
                            })
                            .await
                            .unwrap_or_else(|err| {
                                Err(anyhow::anyhow!("Failed to process pasted image: {err}"))
                            });
                            let _ =
                                app_tx.send(AppEvent::ClipboardImageReady { id: image_id, png });
                        });
                    }
                    continue;
                }

                // Escape key behavior depends on state
                if key.code == KeyCode::Esc {
                    if app.has_wait() {
                        app.skip_wait();
                    } else if app.has_prompt() {
                        if let Some(recorder) = ui_trace.as_mut() {
                            recorder.record_prompt_dismiss();
                        }
                        app.dismiss_prompt();
                    } else if app.has_skill_picker() {
                        let _ = apply_terminal_action(
                            &mut app,
                            &terminal,
                            UiAction::DismissSkillPicker,
                        );
                    } else if app.has_session_picker() {
                        let _ = apply_terminal_action(
                            &mut app,
                            &terminal,
                            UiAction::DismissSessionPicker,
                        );
                    } else if app.running {
                        // Interrupt running session
                        app.note_manual_interrupt_requested();
                        if let Some(token) = cancel_token.take() {
                            token.cancel();
                        }
                    }
                    // When idle with no dialog: no-op
                    continue;
                }

                // ── Always-on scroll keys (work in all states) ──
                {
                    let (width, height) = terminal.size()?;
                    let vw = width as usize;
                    let vh = render::history_viewport_height(&app, width, height);
                    let half_page = vh / 2;

                    if app.has_prompt() {
                        if key.modifiers.contains(KeyModifiers::CONTROL)
                            && key.code == KeyCode::Char('u')
                        {
                            let _ = apply_terminal_action(
                                &mut app,
                                &terminal,
                                UiAction::PromptScrollUp(half_page),
                            );
                            app.dirty = true;
                            continue;
                        }
                        if key.modifiers.contains(KeyModifiers::CONTROL)
                            && key.code == KeyCode::Char('d')
                        {
                            let _ = apply_terminal_action(
                                &mut app,
                                &terminal,
                                UiAction::PromptScrollDown(half_page),
                            );
                            app.dirty = true;
                            continue;
                        }
                        if key.code == KeyCode::PageUp {
                            let _ = apply_terminal_action(
                                &mut app,
                                &terminal,
                                UiAction::PromptScrollUp(vh),
                            );
                            app.dirty = true;
                            continue;
                        }
                        if key.code == KeyCode::PageDown {
                            let _ = apply_terminal_action(
                                &mut app,
                                &terminal,
                                UiAction::PromptScrollDown(vh),
                            );
                            app.dirty = true;
                            continue;
                        }
                    }

                    // Ctrl+U / Ctrl+D: half-page scroll
                    if key.modifiers.contains(KeyModifiers::CONTROL)
                        && key.code == KeyCode::Char('u')
                    {
                        if let Some(recorder) = ui_trace.as_mut() {
                            recorder.record_scroll_up(half_page);
                        }
                        let _ = apply_terminal_action(
                            &mut app,
                            &terminal,
                            UiAction::ScrollUp(half_page),
                        );
                        app.dirty = true;
                        continue;
                    }
                    if key.modifiers.contains(KeyModifiers::CONTROL)
                        && key.code == KeyCode::Char('d')
                    {
                        if let Some(recorder) = ui_trace.as_mut() {
                            recorder.record_scroll_down(half_page);
                        }
                        let _ = apply_ui_action(
                            &mut app,
                            UiAction::ScrollDown(half_page),
                            UiActionContext {
                                viewport_width: vw,
                                viewport_height: vh,
                                prompt_max_scroll: 0,
                            },
                        );
                        app.dirty = true;
                        continue;
                    }

                    // PgUp / PgDn
                    if key.code == KeyCode::PageUp {
                        if let Some(recorder) = ui_trace.as_mut() {
                            recorder.record_scroll_up(vh);
                        }
                        let _ = apply_terminal_action(&mut app, &terminal, UiAction::ScrollUp(vh));
                        app.dirty = true;
                        continue;
                    }
                    if key.code == KeyCode::PageDown {
                        if let Some(recorder) = ui_trace.as_mut() {
                            recorder.record_scroll_down(vh);
                        }
                        let _ = apply_ui_action(
                            &mut app,
                            UiAction::ScrollDown(vh),
                            UiActionContext {
                                viewport_width: vw,
                                viewport_height: vh,
                                prompt_max_scroll: 0,
                            },
                        );
                        app.dirty = true;
                        continue;
                    }

                    if app.has_prompt() && !app.is_prompt_text_entry() {
                        if key.code == KeyCode::Up {
                            let _ = apply_terminal_action(
                                &mut app,
                                &terminal,
                                UiAction::PromptScrollUp(1),
                            );
                            app.dirty = true;
                            continue;
                        }
                        if key.code == KeyCode::Down {
                            let _ = apply_terminal_action(
                                &mut app,
                                &terminal,
                                UiAction::PromptScrollDown(1),
                            );
                            app.dirty = true;
                            continue;
                        }
                    }

                    if app.has_wait() {
                        if matches!(key.code, KeyCode::Up | KeyCode::Char('k')) {
                            if let Some(recorder) = ui_trace.as_mut() {
                                recorder.record_scroll_up(1);
                            }
                            let _ =
                                apply_terminal_action(&mut app, &terminal, UiAction::ScrollUp(1));
                            app.dirty = true;
                            continue;
                        }
                        if matches!(key.code, KeyCode::Down | KeyCode::Char('j')) {
                            if let Some(recorder) = ui_trace.as_mut() {
                                recorder.record_scroll_down(1);
                            }
                            let _ = apply_ui_action(
                                &mut app,
                                UiAction::ScrollDown(1),
                                UiActionContext {
                                    viewport_width: vw,
                                    viewport_height: vh,
                                    prompt_max_scroll: 0,
                                },
                            );
                            app.dirty = true;
                            continue;
                        }
                    }
                }

                // ── Skill picker key handling ──
                if app.has_skill_picker() {
                    match key.code {
                        KeyCode::Up | KeyCode::Char('k') => {
                            let _ =
                                apply_terminal_action(&mut app, &terminal, UiAction::SkillPickerUp);
                        }
                        KeyCode::Down | KeyCode::Char('j') => {
                            let _ = apply_terminal_action(
                                &mut app,
                                &terminal,
                                UiAction::SkillPickerDown,
                            );
                        }
                        KeyCode::Enter => {
                            if let UiActionOutcome::SkillPicked(Some(name)) = apply_terminal_action(
                                &mut app,
                                &terminal,
                                UiAction::SubmitSkillPicker,
                            ) {
                                app.set_input(format!("${} ", name));
                            }
                        }
                        _ => {}
                    }
                    continue;
                }

                // ── Session picker key handling ──
                if app.has_session_picker() {
                    match key.code {
                        KeyCode::Up | KeyCode::Char('k') => {
                            let _ = apply_terminal_action(
                                &mut app,
                                &terminal,
                                UiAction::SessionPickerUp,
                            );
                        }
                        KeyCode::Down | KeyCode::Char('j') => {
                            let _ = apply_terminal_action(
                                &mut app,
                                &terminal,
                                UiAction::SessionPickerDown,
                            );
                        }
                        KeyCode::Enter => {
                            if let UiActionOutcome::SessionPicked(Some(filename)) =
                                apply_terminal_action(
                                    &mut app,
                                    &terminal,
                                    UiAction::SubmitSessionPicker,
                                )
                            {
                                match resume::load_resumed_session(
                                    &filename,
                                    &mut app,
                                    &mut history,
                                    &mut runtime,
                                    &mut turn_counter,
                                    &mut current_execution_mode,
                                    &mut current_context_strategy,
                                    &provider,
                                    &mut current_model_variant,
                                    &dynamic_tools,
                                    &mut desired_dynamic,
                                    model_catalog.as_ref(),
                                )
                                .await
                                {
                                    Ok(()) => {
                                        if let Some(rt) = runtime.as_ref() {
                                            match rt.session_manager() {
                                                Ok(manager) => session_manager = manager,
                                                Err(err) => push_system_message(
                                                    &mut app,
                                                    format!(
                                                        "Failed to refresh session manager: {}",
                                                        err
                                                    ),
                                                ),
                                            }
                                        }
                                        sync_ui_extensions(
                                            &mut app,
                                            ui_extensions.as_ref(),
                                            &plugin_host,
                                            Arc::clone(&session_manager),
                                        )
                                        .await;
                                        toolset_hash = hash12(
                                            &serde_json::to_vec(&dynamic_tools.definitions())
                                                .unwrap_or_else(|_| b"[]".to_vec()),
                                        );
                                    }
                                    Err(err) => {
                                        app.blocks.push(DisplayBlock::SystemMessage(err));
                                        app.invalidate_height_cache();
                                        app.scroll_to_bottom();
                                    }
                                }
                            }
                        }
                        _ => {} // ignore other keys while picker is open
                    }
                    continue;
                }

                // ── Prompt (ask dialog) key handling ──
                if app.has_prompt() {
                    let editing_text = app.is_prompt_text_entry();
                    match key.code {
                        KeyCode::Tab if app.prompt_supports_note() => {
                            if let Some(recorder) = ui_trace.as_mut() {
                                recorder.record_prompt_toggle_note_focus();
                            }
                            let _ = apply_terminal_action(
                                &mut app,
                                &terminal,
                                UiAction::PromptToggleNoteFocus,
                            );
                        }
                        KeyCode::Up if !editing_text && app.prompt_supports_note() => {
                            if let Some(recorder) = ui_trace.as_mut() {
                                recorder.record_prompt_up();
                            }
                            let _ = apply_terminal_action(&mut app, &terminal, UiAction::PromptUp);
                        }
                        KeyCode::Down if !editing_text && app.prompt_supports_note() => {
                            if let Some(recorder) = ui_trace.as_mut() {
                                recorder.record_prompt_down();
                            }
                            let _ =
                                apply_terminal_action(&mut app, &terminal, UiAction::PromptDown);
                        }
                        KeyCode::Char(' ') if app.is_prompt_multi_select() && !editing_text => {
                            if let Some(recorder) = ui_trace.as_mut() {
                                recorder.record_prompt_toggle_current_option();
                            }
                            let _ = apply_terminal_action(
                                &mut app,
                                &terminal,
                                UiAction::PromptToggleCurrentOption,
                            );
                        }
                        KeyCode::BackTab if editing_text => {
                            if let Some(recorder) = ui_trace.as_mut() {
                                recorder.record_prompt_insert_text("\n");
                            }
                            let _ = apply_terminal_action(
                                &mut app,
                                &terminal,
                                UiAction::PromptInsertText("\n".to_string()),
                            );
                        }
                        KeyCode::Enter => {
                            if let Some(recorder) = ui_trace.as_mut() {
                                recorder.record_submit_prompt();
                            }
                            let _ =
                                apply_terminal_action(&mut app, &terminal, UiAction::SubmitPrompt);
                        }
                        KeyCode::Char(c) if editing_text => {
                            if let Some(recorder) = ui_trace.as_mut() {
                                recorder.record_prompt_insert_text(c.to_string());
                            }
                            let _ = apply_terminal_action(
                                &mut app,
                                &terminal,
                                UiAction::PromptInsertText(c.to_string()),
                            );
                        }
                        KeyCode::Backspace if editing_text => {
                            if let Some(recorder) = ui_trace.as_mut() {
                                recorder.record_prompt_backspace();
                            }
                            let _ = apply_terminal_action(
                                &mut app,
                                &terminal,
                                UiAction::PromptBackspace,
                            );
                        }
                        _ => {}
                    }
                    continue;
                }

                if app.has_wait() {
                    if key.modifiers.contains(KeyModifiers::CONTROL)
                        && key.code == KeyCode::Char('j')
                    {
                        app.resume_wait();
                    }
                    continue;
                }

                if let Some(chord) = key_chord_from_event(key)
                    && let Some(shortcut) = ui_extensions.shortcut_for(chord)
                {
                    match ui_extensions
                        .invoke_shortcut(
                            &shortcut,
                            UiContext {
                                plugin_host: &plugin_host,
                                session_id: crate::ROOT_SESSION_ID,
                                session_manager: Arc::clone(&session_manager),
                            },
                        )
                        .await
                    {
                        Ok(effects) => apply_ui_host_effects(&mut app, effects),
                        Err(err) => push_system_message(&mut app, err),
                    }
                    continue;
                }

                match key.code {
                    // Tab: complete selected suggestion
                    KeyCode::Tab if app.has_suggestions() => {
                        if let Some(recorder) = ui_trace.as_mut() {
                            recorder.record_suggestion_complete();
                        }
                        let _ = apply_terminal_action(
                            &mut app,
                            &terminal,
                            UiAction::SuggestionComplete,
                        );
                    }
                    KeyCode::Tab => {
                        let queued = normalize_prepared_turn_for_dispatch(
                            app.take_prepared_turn(),
                            &app.skills,
                        );
                        app.update_suggestions();
                        if app.has_pending_image_jobs() {
                            app.restore_prepared_turn(queued);
                            push_system_message(
                                &mut app,
                                "Wait for pasted images to finish processing before sending or queueing this draft.",
                            );
                            continue;
                        }
                        let parsed_command = parse_slash_command(
                            &queued.display_text,
                            &app.skills,
                            ui_extensions.as_ref(),
                        );
                        let is_host_slash_command = parsed_command.is_some();
                        if queued.is_empty()
                            || shell_escape_command(&queued.display_text).is_some()
                            || (is_host_slash_command && !app.running)
                        {
                            app.restore_prepared_turn(queued);
                            continue;
                        }
                        if app.running {
                            if let Some(cmd) = parsed_command
                                && slash_command_runs_out_of_band_while_running(&cmd)
                            {
                                if let Some(recorder) = ui_trace.as_mut() {
                                    recorder.record_slash_command(queued.display_text.clone());
                                }
                                if handle_parsed_slash_command(
                                    cmd,
                                    &mut terminal,
                                    &mut app,
                                    logger,
                                    args,
                                    &paused,
                                    &plugin_host,
                                    ui_extensions.as_ref(),
                                    &dynamic_tools,
                                    &mut runtime,
                                    &mut history,
                                    &mut turn_counter,
                                    &mut last_turn,
                                    &mut runtime_return_rx,
                                    &mut cancel_token,
                                    &mut active_stream_id,
                                    &mut provider,
                                    &mut current_model_variant,
                                    &mut current_execution_mode,
                                    &mut current_context_strategy,
                                    &mut session_manager,
                                    &mut desired_dynamic,
                                    &mut pending_reconfigure,
                                    model_catalog.as_ref(),
                                    &mut toolset_hash,
                                    &app_tx,
                                    &mut pending_clear_after_return,
                                )
                                .await?
                                {
                                    break;
                                }
                                continue;
                            }
                            record_queue_turn(&mut ui_trace, &queued);
                            app.queue_turn(queued.clone());
                            continue;
                        }
                        if runtime.is_none() {
                            tracing::debug!(
                                queued = queued.display_text,
                                app_running = app.running,
                                runtime_return_rx_present = runtime_return_rx.is_some(),
                                "queueing turn because runtime handoff is still in progress"
                            );
                            record_queue_turn(&mut ui_trace, &queued);
                            app.queue_turn(queued);
                            continue;
                        }

                        let (items, image_blobs) = build_items_from_editor_input(
                            &queued.effective_text,
                            queued.images.clone(),
                        );
                        let turn_input = make_turn_input(&queued, items, image_blobs);
                        let current_dynamic_state = dynamic_tools.export_state();
                        send_user_message(
                            queued.clone(),
                            turn_input.clone(),
                            &mut app,
                            ui_trace.as_mut(),
                            logger,
                            &mut runtime,
                            &mut history,
                            &mut runtime_return_rx,
                            &mut cancel_token,
                            &mut active_stream_id,
                            &app_tx,
                            &provider,
                            &current_dynamic_state,
                            &toolset_hash,
                        );
                        last_turn = Some(TurnReplayPayload {
                            prepared_turn: queued,
                            turn_input,
                            execution_mode: current_execution_mode,
                        });
                    }
                    // Up/Down: navigate suggestions when popup is visible
                    KeyCode::Up if app.has_suggestions() => {
                        if let Some(recorder) = ui_trace.as_mut() {
                            recorder.record_suggestion_up();
                        }
                        let _ = apply_terminal_action(&mut app, &terminal, UiAction::SuggestionUp);
                    }
                    KeyCode::Down if app.has_suggestions() => {
                        if let Some(recorder) = ui_trace.as_mut() {
                            recorder.record_suggestion_down();
                        }
                        let _ =
                            apply_terminal_action(&mut app, &terminal, UiAction::SuggestionDown);
                    }
                    KeyCode::Enter => {
                        // Shift+Enter or Alt+Enter → insert newline
                        if key.modifiers.contains(KeyModifiers::SHIFT)
                            || key.modifiers.contains(KeyModifiers::ALT)
                        {
                            app.insert_char('\n');
                            app.update_suggestions();
                            continue;
                        }

                        let queued = normalize_prepared_turn_for_dispatch(
                            app.take_prepared_turn(),
                            &app.skills,
                        );
                        app.update_suggestions();
                        if app.has_pending_image_jobs() {
                            app.restore_prepared_turn(queued);
                            push_system_message(
                                &mut app,
                                "Wait for pasted images to finish processing before sending or queueing this draft.",
                            );
                            continue;
                        }
                        if queued.is_empty() {
                            continue;
                        }

                        let parsed_command = parse_slash_command(
                            &queued.display_text,
                            &app.skills,
                            ui_extensions.as_ref(),
                        );
                        let is_host_slash_command = parsed_command.is_some();

                        if app.running {
                            if let Some(cmd) = parsed_command
                                && slash_command_runs_out_of_band_while_running(&cmd)
                            {
                                if let Some(recorder) = ui_trace.as_mut() {
                                    recorder.record_slash_command(queued.display_text.clone());
                                }
                                if handle_parsed_slash_command(
                                    cmd,
                                    &mut terminal,
                                    &mut app,
                                    logger,
                                    args,
                                    &paused,
                                    &plugin_host,
                                    ui_extensions.as_ref(),
                                    &dynamic_tools,
                                    &mut runtime,
                                    &mut history,
                                    &mut turn_counter,
                                    &mut last_turn,
                                    &mut runtime_return_rx,
                                    &mut cancel_token,
                                    &mut active_stream_id,
                                    &mut provider,
                                    &mut current_model_variant,
                                    &mut current_execution_mode,
                                    &mut current_context_strategy,
                                    &mut session_manager,
                                    &mut desired_dynamic,
                                    &mut pending_reconfigure,
                                    model_catalog.as_ref(),
                                    &mut toolset_hash,
                                    &app_tx,
                                    &mut pending_clear_after_return,
                                )
                                .await?
                                {
                                    break;
                                }
                                continue;
                            }
                            if is_host_slash_command {
                                record_queue_turn(&mut ui_trace, &queued);
                                app.queue_turn(queued.clone());
                                continue;
                            }
                            if shell_escape_command(&queued.display_text).is_some() {
                                push_system_message(
                                    &mut app,
                                    "Shell escapes cannot be injected into the active turn. Wait for completion or use `Tab` to queue a later turn.",
                                );
                                app.restore_prepared_turn(queued);
                                continue;
                            }
                            let injection = make_injected_plugin_message(&queued);
                            match turn_injection_bridge.enqueue(vec![injection]) {
                                Ok(()) => {
                                    record_queue_pending_steer(&mut ui_trace, &queued);
                                    app.queue_pending_steer(queued.clone());
                                }
                                Err(err) => {
                                    push_system_message(
                                        &mut app,
                                        format!("Failed to queue current-turn injection: {}", err),
                                    );
                                    app.restore_prepared_turn(queued);
                                }
                            }
                            continue;
                        }
                        if runtime.is_none() {
                            tracing::debug!(
                                queued = queued.display_text,
                                app_running = app.running,
                                runtime_return_rx_present = runtime_return_rx.is_some(),
                                "queueing turn because runtime handoff is still in progress"
                            );
                            record_queue_turn(&mut ui_trace, &queued);
                            app.queue_turn(queued);
                            continue;
                        }

                        // Shell escape: !command
                        if let Some(cmd_str) = shell_escape_command(&queued.display_text) {
                            if !cmd_str.is_empty() {
                                app.blocks
                                    .push(DisplayBlock::UserInput(queued.display_text.clone()));
                                app.invalidate_height_cache();

                                use tokio::process::Command as TokioCommand;
                                let result = tokio::time::timeout(
                                    std::time::Duration::from_secs(30),
                                    TokioCommand::new("bash").arg("-c").arg(cmd_str).output(),
                                )
                                .await;

                                match result {
                                    Ok(Ok(output)) => {
                                        let stdout =
                                            String::from_utf8_lossy(&output.stdout).to_string();
                                        let stderr =
                                            String::from_utf8_lossy(&output.stderr).to_string();
                                        let error = if !output.status.success() {
                                            let mut err = stderr.clone();
                                            if let Some(code) = output.status.code() {
                                                if !err.is_empty() && !err.ends_with('\n') {
                                                    err.push('\n');
                                                }
                                                err.push_str(&format!("[exit code: {}]", code));
                                            }
                                            Some(err)
                                        } else if !stderr.is_empty() {
                                            Some(stderr)
                                        } else {
                                            None
                                        };
                                        app.blocks.push(DisplayBlock::ShellOutput {
                                            command: cmd_str.to_string(),
                                            output: stdout.trim_end().to_string(),
                                            error: error.map(|e| e.trim_end().to_string()),
                                        });
                                    }
                                    Ok(Err(e)) => {
                                        app.blocks.push(DisplayBlock::Error(format!(
                                            "Failed to run '{}': {}",
                                            cmd_str, e
                                        )));
                                    }
                                    Err(_) => {
                                        app.blocks.push(DisplayBlock::Error(format!(
                                            "Command '{}' timed out after 30s. Try a narrower command or run it in smaller steps.",
                                            cmd_str
                                        )));
                                    }
                                }
                                app.invalidate_height_cache();
                                app.scroll_to_bottom();
                            }
                            continue;
                        }

                        // Try slash command
                        if let Some(cmd) = parse_slash_command(
                            &queued.display_text,
                            &app.skills,
                            ui_extensions.as_ref(),
                        ) {
                            if let Some(recorder) = ui_trace.as_mut() {
                                recorder.record_slash_command(queued.display_text.clone());
                            }
                            if handle_parsed_slash_command(
                                cmd,
                                &mut terminal,
                                &mut app,
                                logger,
                                args,
                                &paused,
                                &plugin_host,
                                ui_extensions.as_ref(),
                                &dynamic_tools,
                                &mut runtime,
                                &mut history,
                                &mut turn_counter,
                                &mut last_turn,
                                &mut runtime_return_rx,
                                &mut cancel_token,
                                &mut active_stream_id,
                                &mut provider,
                                &mut current_model_variant,
                                &mut current_execution_mode,
                                &mut current_context_strategy,
                                &mut session_manager,
                                &mut desired_dynamic,
                                &mut pending_reconfigure,
                                model_catalog.as_ref(),
                                &mut toolset_hash,
                                &app_tx,
                                &mut pending_clear_after_return,
                            )
                            .await?
                            {
                                break;
                            }
                            continue;
                        }

                        // Handle "quit"/"exit" without slash prefix
                        if queued.display_text == "quit" || queued.display_text == "exit" {
                            break;
                        }

                        // Regular user message — send to the active session
                        if let Err(e) = apply_pending_reconfigure(
                            &dynamic_tools,
                            &mut desired_dynamic,
                            &mut pending_reconfigure,
                            &mut runtime,
                        )
                        .await
                        {
                            push_system_message(
                                &mut app,
                                format!(
                                    "Pending runtime reconfigure failed; message not sent: {}",
                                    e
                                ),
                            );
                            continue;
                        }
                        toolset_hash = hash12(
                            &serde_json::to_vec(&dynamic_tools.definitions())
                                .unwrap_or_else(|_| b"[]".to_vec()),
                        );
                        let (items, image_blobs) = build_items_from_editor_input(
                            &queued.effective_text,
                            queued.images.clone(),
                        );
                        let turn_input = make_turn_input(&queued, items, image_blobs);
                        let current_dynamic_state = dynamic_tools.export_state();
                        send_user_message(
                            queued.clone(),
                            turn_input.clone(),
                            &mut app,
                            ui_trace.as_mut(),
                            logger,
                            &mut runtime,
                            &mut history,
                            &mut runtime_return_rx,
                            &mut cancel_token,
                            &mut active_stream_id,
                            &app_tx,
                            &provider,
                            &current_dynamic_state,
                            &toolset_hash,
                        );
                        last_turn = Some(TurnReplayPayload {
                            prepared_turn: queued,
                            turn_input,
                            execution_mode: current_execution_mode,
                        });
                    }
                    KeyCode::Backspace => {
                        if let Some(recorder) = ui_trace.as_mut() {
                            recorder.record_input_backspace();
                        }
                        let _ =
                            apply_terminal_action(&mut app, &terminal, UiAction::InputBackspace);
                    }
                    KeyCode::Delete => {
                        if let Some(recorder) = ui_trace.as_mut() {
                            recorder.record_input_delete();
                        }
                        let _ = apply_terminal_action(&mut app, &terminal, UiAction::InputDelete);
                    }
                    KeyCode::Left => {
                        if let Some(recorder) = ui_trace.as_mut() {
                            recorder.record_move_cursor_left();
                        }
                        let _ =
                            apply_terminal_action(&mut app, &terminal, UiAction::MoveCursorLeft);
                    }
                    KeyCode::Right => {
                        if let Some(recorder) = ui_trace.as_mut() {
                            recorder.record_move_cursor_right();
                        }
                        let _ =
                            apply_terminal_action(&mut app, &terminal, UiAction::MoveCursorRight);
                    }
                    KeyCode::Home => {
                        if let Some(recorder) = ui_trace.as_mut() {
                            recorder.record_move_cursor_home();
                        }
                        let _ =
                            apply_terminal_action(&mut app, &terminal, UiAction::MoveCursorHome);
                    }
                    KeyCode::End => {
                        if let Some(recorder) = ui_trace.as_mut() {
                            recorder.record_move_cursor_end();
                        }
                        let _ = apply_terminal_action(&mut app, &terminal, UiAction::MoveCursorEnd);
                    }
                    KeyCode::Up => {
                        if let Some(recorder) = ui_trace.as_mut() {
                            recorder.record_history_up();
                        }
                        let _ = apply_terminal_action(&mut app, &terminal, UiAction::HistoryUp);
                    }
                    KeyCode::Down => {
                        if let Some(recorder) = ui_trace.as_mut() {
                            recorder.record_history_down();
                        }
                        let _ = apply_terminal_action(&mut app, &terminal, UiAction::HistoryDown);
                    }
                    KeyCode::Char(c) => {
                        if let Some(recorder) = ui_trace.as_mut() {
                            recorder.record_input_insert_text(c.to_string());
                        }
                        let _ = apply_terminal_action(
                            &mut app,
                            &terminal,
                            UiAction::InputInsertText(c.to_string()),
                        );
                    }
                    _ => {}
                }
            }
            AppEvent::Terminal(TermEvent::Mouse(mouse)) => {
                use crossterm::event::{MouseButton, MouseEventKind};
                // Some terminals (notably kitty) can paint transient hover/search
                // decorations on top of the alt-screen. Repaint on any mouse event
                // so ignored motion events do not leave visual artifacts behind.
                app.dirty = true;
                let ha = app.history_area;
                let in_history = mouse.row >= ha.y
                    && mouse.row < ha.y + ha.height
                    && mouse.column >= ha.x
                    && mouse.column < ha.x + ha.width;

                if app.has_prompt() {
                    match mouse.kind {
                        MouseEventKind::ScrollUp => {
                            let _ = apply_terminal_action(
                                &mut app,
                                &terminal,
                                UiAction::PromptScrollUp(3),
                            );
                            app.dirty = true;
                            continue;
                        }
                        MouseEventKind::ScrollDown => {
                            let _ = apply_terminal_action(
                                &mut app,
                                &terminal,
                                UiAction::PromptScrollDown(3),
                            );
                            app.dirty = true;
                            continue;
                        }
                        _ => {}
                    }
                }

                match mouse.kind {
                    MouseEventKind::Down(MouseButton::Left) if in_history => {
                        tracing::debug!(
                            row = mouse.row,
                            column = mouse.column,
                            "selection started from mouse down"
                        );
                        let vrow = app.scroll_offset + (mouse.row - ha.y) as usize;
                        app.selection.anchor = (mouse.column, vrow);
                        app.selection.end = (mouse.column, vrow);
                        app.selection.active = true;
                        app.selection.visible = false;
                        app.dirty = true;
                    }
                    MouseEventKind::Drag(MouseButton::Left) if app.selection.active => {
                        let col = mouse.column.clamp(ha.x, ha.x + ha.width.saturating_sub(1));

                        // Auto-scroll when dragging above or below the history area
                        let scroll_lines = 3usize;
                        if mouse.row < ha.y {
                            if let Some(recorder) = ui_trace.as_mut() {
                                recorder.record_scroll_up(scroll_lines);
                            }
                            app.scroll_up(scroll_lines);
                        } else if mouse.row >= ha.y + ha.height {
                            let (width, height) = terminal.size()?;
                            let vh = render::history_viewport_height(&app, width, height);
                            let vw = width as usize;
                            if let Some(recorder) = ui_trace.as_mut() {
                                recorder.record_scroll_down(scroll_lines);
                            }
                            app.scroll_down(scroll_lines, vh, vw);
                        }

                        let clamped_row = mouse.row.clamp(ha.y, ha.y + ha.height.saturating_sub(1));
                        let vrow = app.scroll_offset + (clamped_row - ha.y) as usize;
                        app.selection.end = (col, vrow);
                        app.selection.visible = true;
                        tracing::debug!(
                            row = mouse.row,
                            column = mouse.column,
                            selection_anchor = ?app.selection.anchor,
                            selection_end = ?app.selection.end,
                            "selection updated from mouse drag"
                        );
                        app.dirty = true;
                    }
                    MouseEventKind::Up(MouseButton::Left) if app.selection.active => {
                        tracing::debug!(
                            selection_anchor = ?app.selection.anchor,
                            selection_end = ?app.selection.end,
                            selection_visible = app.selection.visible,
                            "selection finished on mouse up"
                        );
                        app.selection.active = false;
                    }
                    MouseEventKind::ScrollUp => {
                        // Scroll extends selection if actively dragging, otherwise just scrolls
                        let scroll_lines = 3;
                        if let Some(recorder) = ui_trace.as_mut() {
                            recorder.record_scroll_up(scroll_lines);
                        }
                        app.scroll_up(scroll_lines);
                        if app.selection.active {
                            // Extend selection to top of viewport after scroll
                            let vrow = app.scroll_offset;
                            app.selection.end = (ha.x, vrow);
                            app.selection.visible = true;
                        }
                        app.dirty = true;
                    }
                    MouseEventKind::ScrollDown => {
                        let (width, height) = terminal.size()?;
                        let vh = render::history_viewport_height(&app, width, height);
                        let vw = width as usize;
                        let scroll_lines = 3;
                        if let Some(recorder) = ui_trace.as_mut() {
                            recorder.record_scroll_down(scroll_lines);
                        }
                        app.scroll_down(scroll_lines, vh, vw);
                        if app.selection.active {
                            // Extend selection to bottom of viewport after scroll
                            let vrow = app.scroll_offset + vh.saturating_sub(1);
                            app.selection.end = (ha.x + ha.width, vrow);
                            app.selection.visible = true;
                        }
                        app.dirty = true;
                    }
                    _ => {}
                }
            }
            AppEvent::Terminal(TermEvent::FocusGained) => {
                app.focused = true;
                app.dirty = true;
            }
            AppEvent::Terminal(TermEvent::FocusLost) => {
                app.focused = false;
                app.dirty = true;
            }
            AppEvent::Terminal(_) => {
                // Resize events, etc.
                app.dirty = true;
            }
            AppEvent::Tick => {
                app.on_tick();
                if app.wait_timed_out() {
                    app.timeout_wait();
                }
            }
            AppEvent::Session { stream_id, event } => {
                if stream_id != active_stream_id {
                    continue;
                }
                if let SessionEvent::TextDelta { content } = event {
                    pending_text_deltas.push(content);
                    continue;
                }
                app.dirty = true;
                if let SessionEvent::DurableSnapshot { snapshot } = event {
                    if runtime_return_rx.is_some()
                        && let Some(context_window) = app.context_window
                    {
                        persist_live_runtime_snapshot(
                            &store,
                            snapshot,
                            &app.ui_resume_state(),
                            &desired_dynamic,
                            &provider,
                            &app.model,
                            context_window,
                            current_execution_mode,
                            current_context_strategy,
                            current_model_variant.as_deref(),
                            &toolset_hash,
                            app.token_usage.clone(),
                            app.last_prompt_usage.clone(),
                        );
                    }
                    continue;
                }
                // Intercept Prompt events — set up dialog state instead of passing to handle_session_event
                if let SessionEvent::Prompt {
                    request,
                    response_tx,
                } = event
                {
                    let ui_effects =
                        ui_extensions.effects_for_session_event(&SessionEvent::Prompt {
                            request: request.clone(),
                            response_tx: response_tx.clone(),
                        });
                    apply_ui_host_effects(&mut app, ui_effects);
                    if let Some(recorder) = ui_trace.as_mut() {
                        recorder.record_emit_prompt(&request);
                    }
                    if request.is_wait() {
                        app.show_wait(app::WaitState::from_request(request, response_tx));
                    } else {
                        let focus = if request.is_freeform() {
                            crate::overlay::PromptFocus::Text
                        } else {
                            crate::overlay::PromptFocus::Options
                        };
                        app.show_prompt(app::PromptState {
                            request,
                            focus,
                            cursor: 0,
                            scroll_offset: 0,
                            selected: Default::default(),
                            reply_text: String::new(),
                            reply_cursor: 0,
                            response_tx,
                        });
                    }
                } else {
                    if let Some(recorder) = ui_trace.as_mut() {
                        recorder.record_session_event(&event);
                    }
                    let ui_effects = ui_extensions.effects_for_session_event(&event);
                    let plugin_notification =
                        if let SessionEvent::PluginEvent { event, .. } = &event {
                            crate::plugin_surface::desktop_notification_effect(event)
                        } else {
                            None
                        };
                    app.handle_session_event(event);
                    apply_ui_host_effects(&mut app, ui_effects);
                    if let Some(effect) = plugin_notification {
                        apply_ui_host_effects(&mut app, vec![effect]);
                    }
                }
            }
            AppEvent::Quit => break,
        }

        drain_aux_trace_ops(&mut ui_trace);
    }

    flush_pending_text_deltas(&mut app, &mut pending_text_deltas, ui_trace.as_mut());
    drain_aux_trace_ops(&mut ui_trace);

    if let Some(recorder) = ui_trace.take() {
        let (width, height) = terminal.size().unwrap_or((80, 24));
        let mut recorder = recorder;
        recorder.set_size(width, height);
        let final_screen = render_screen_text(&mut app, width, height);
        recorder.finish(&final_screen)?;
    }
    disable_aux_op_recording();

    // Signal reader thread and tick timer to stop
    stop.store(true, Ordering::Relaxed);

    // Save input history
    app.save_history();

    Ok(())
}
