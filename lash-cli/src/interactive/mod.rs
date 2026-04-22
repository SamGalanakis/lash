mod commands;
mod helpers;
mod input_handling;
mod runtime;
#[cfg(test)]
mod tests;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crossterm::event::Event as TermEvent;
use lash::session_model::Message;
use lash::*;
use lash_tui::{InputEvent as TuiInputEvent, Terminal, normalize_event};
use lash_ui::{UiCommandInvocation, UiContext, UiExtensions};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::app::{self, App, DisplayBlock, PreparedTurn};
use crate::command;
use crate::event::AppEvent;
use crate::render;
use crate::resume;
use crate::session_log::{self, SessionLogger};
use crate::turn_runner::{RuntimeRunResult, make_turn_input};
use crate::ui_action::UiAction;
use crate::ui_trace::{
    UiTraceRecorder, disable_aux_op_recording, enable_aux_op_recording, render_screen_text,
};
use crate::update;
use crate::{Args, scratch_tui};
use crate::{
    apply_ui_host_effects, controls_text, hash12, help_text, info_text,
    normalize_prepared_turn_for_dispatch, push_system_message, sync_ui_extensions,
    turn_has_visible_output, version_text,
};

use self::helpers::{
    AppEventSink, PendingTextDeltaBuffer, TurnReplayPayload, cleared_session_state,
    drain_aux_trace_ops, flush_pending_text_deltas, log_runtime_handoff, record_queue_turn,
};

use self::commands::{dispatch_next_queued_turn, promote_pending_steers_to_queue};
#[cfg(test)]
pub(crate) use self::runtime::injected_image_part_indices;
#[cfg(test)]
pub(crate) use self::runtime::make_injected_plugin_message;
use self::runtime::{apply_pending_reconfigure, send_user_message, sync_runtime_tool_surface};
pub(crate) use self::runtime::{generate_session_name, notify_desktop};

use self::input_handling::{
    apply_terminal_action, apply_ui_host_effect, handle_key_event, handle_mouse_event,
    handle_surface_input, process_pending_monitor_wakes,
};

// Items used only by tests via `use super::*;` in tests.rs.
#[cfg(test)]
#[allow(unused_imports)]
use self::helpers::{
    TEXT_DELTA_REDRAW_INTERVAL, is_copy_shortcut, should_preserve_selection_for_key,
};
#[cfg(test)]
#[allow(unused_imports)]
use self::input_handling::enqueue_pending_monitor_wakes;

#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_app(
    mut terminal: Terminal,
    runtime: LashRuntime,
    plugin_host: PluginHost,
    dynamic_tools: Arc<DynamicToolProvider>,
    turn_injection_bridge: TurnInjectionBridge,
    turn_input_injection_bridge: TurnInputInjectionBridge,
    logger: &mut SessionLogger,
    args: &Args,
    mut provider: ProviderHandle,
    model: String,
    initial_context_window: u64,
    session_name: String,
    model_catalog: Arc<CachedModelCatalog>,
    _store: Arc<Store>,
    mut toolset_hash: String,
    initial_model_variant: Option<String>,
    initial_execution_mode: ExecutionMode,
    startup_system_message: Option<String>,
) -> anyhow::Result<()> {
    let mut app = App::new(model, session_name);
    let extra_ui_extensions: Vec<Arc<dyn lash_ui::UiExtension>> = vec![Arc::new(
        lash_autoresearch::AutoresearchUiExtension::default(),
    )];
    let ui_extensions = Arc::new(
        UiExtensions::with_builtins(extra_ui_extensions)
            .map_err(|err| anyhow::anyhow!("failed to build UI extensions: {err}"))?,
    );
    app.set_ui_extensions(Arc::clone(&ui_extensions));
    if let Some(plugin_session) = runtime.plugin_session() {
        app.plugin_commands = plugin_session.command_catalog();
    }
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
    let mut last_ui_sync = tokio::time::Instant::now();

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
                let turn_input = make_turn_input(&prepared);
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
                    &current_dynamic_state,
                )
                .await;
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

        if app.has_pending_monitor_wakes() {
            match process_pending_monitor_wakes(
                &mut app,
                &mut ui_trace,
                logger,
                &mut runtime,
                &mut history,
                &mut runtime_return_rx,
                &mut cancel_token,
                &mut active_stream_id,
                &app_tx,
                &turn_input_injection_bridge,
                &desired_dynamic,
            )
            .await
            {
                Ok(true) => continue,
                Ok(false) => {}
                Err(err) => {
                    push_system_message(&mut app, format!("Failed to inject monitor wake: {err}"))
                }
            }
        }

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
                    if let Err(err) = sync_runtime_tool_surface(&mut runtime).await {
                        push_system_message(
                            &mut app,
                            format!("Failed to sync tool surface after turn: {err}"),
                        );
                    }
                    if done.stream_id != active_stream_id || pending_clear_after_return {
                        app.recycle_unaccepted_monitor_wakes();
                        if let Some(rt) = runtime.as_mut() {
                            let preserved_policy = rt.export_state().policy;
                            let _ = rt.reset_session().await;
                            rt.set_persisted_state(lash::PersistedSessionState::from_state(
                                cleared_session_state(preserved_policy),
                            ));
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
                    let state = done.result.state;
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

                    history = state.project_messages();
                    turn_counter = state.iteration;
                    app.token_usage = state.token_usage.clone();
                    app.last_prompt_usage = state.last_prompt_usage.clone();
                    tracing::debug!(
                        stream_id = done.stream_id,
                        iteration = state.iteration,
                        status = ?done.result.status,
                        reason = ?done.result.done_reason,
                        messages = state.projected_messages().len(),
                        blocks = app.blocks.len(),
                        had_live_turn = app.live_turn.is_some(),
                        running = app.running,
                        "reconciling completed runtime turn"
                    );
                    if interrupted {
                        let had_manual_interrupt_message = matches!(
                            app.blocks.last(),
                            Some(DisplayBlock::SystemMessage(message))
                                if message == crate::util::manual_interrupt_message()
                        );
                        let mut ui_projection_state = app.ui_projection_state();
                        ui_projection_state.live_assistant_text = app::interrupted_assistant_tail(
                            &app.blocks,
                            &done.result.assistant_output.safe_text,
                        );
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
                            let projected_messages = state.project_messages();
                            let projected_tool_calls = state.project_tool_calls();
                            app.blocks = app::project_interrupted_blocks(
                                &projected_messages,
                                &projected_tool_calls,
                                &ui_projection_state,
                                interrupted_message,
                            );
                            app.invalidate_height_cache();
                            app.scroll_to_bottom();
                        }
                        app.recycle_unaccepted_monitor_wakes();
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

                    let projected_messages = state.project_messages();
                    let final_output = app::latest_assistant_text_from_messages(
                        &projected_messages,
                    )
                    .or_else(|| {
                        (!done.result.assistant_output.safe_text.is_empty())
                            .then(|| done.result.assistant_output.safe_text.clone())
                    });
                    let _ui_projection_state =
                        app.finish_turn_for_projection_with_output(final_output.as_deref());
                    app.recycle_unaccepted_monitor_wakes();
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
                    app.recycle_unaccepted_monitor_wakes();
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
            let capabilities = terminal.capabilities();
            terminal
                .draw(|frame| scratch_tui::draw_with_capabilities(frame, &mut app, capabilities))?;
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

                if app.has_prompt() {
                    if app.is_prompt_text_entry() {
                        if let Some(recorder) = ui_trace.as_mut() {
                            recorder.record_prompt_insert_text(text.clone());
                        }
                        let _ = apply_terminal_action(
                            &mut app,
                            &terminal,
                            UiAction::PromptInsertText(text),
                        );
                    }
                    // Prompt modal is focused: swallow pastes that don't
                    // belong in its text field instead of leaking them
                    // into the composer.
                    continue;
                }

                if let Some(event) = normalize_event(&TermEvent::Paste(text.clone()))
                    && handle_surface_input(
                        ui_extensions.as_ref(),
                        &event,
                        &plugin_host,
                        &session_manager,
                        &mut app,
                    )
                {
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
                if handle_key_event(
                    key,
                    &mut app,
                    &mut terminal,
                    &mut ui_trace,
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
                    &mut session_manager,
                    &mut desired_dynamic,
                    &mut pending_reconfigure,
                    model_catalog.as_ref(),
                    &mut toolset_hash,
                    &app_tx,
                    &mut pending_clear_after_return,
                    &turn_input_injection_bridge,
                )
                .await?
                {
                    break;
                }
            }
            AppEvent::Terminal(TermEvent::Mouse(mouse)) => {
                handle_mouse_event(
                    mouse,
                    &mut app,
                    &terminal,
                    &mut ui_trace,
                    ui_extensions.as_ref(),
                    &plugin_host,
                    &session_manager,
                )?;
            }
            AppEvent::Terminal(TermEvent::FocusGained) => {
                app.focused = true;
                let _ = handle_surface_input(
                    ui_extensions.as_ref(),
                    &TuiInputEvent::FocusGained,
                    &plugin_host,
                    &session_manager,
                    &mut app,
                );
                app.dirty = true;
            }
            AppEvent::Terminal(TermEvent::FocusLost) => {
                app.focused = false;
                let _ = handle_surface_input(
                    ui_extensions.as_ref(),
                    &TuiInputEvent::FocusLost,
                    &plugin_host,
                    &session_manager,
                    &mut app,
                );
                app.dirty = true;
            }
            AppEvent::Terminal(term_event) => {
                // Resize events, etc.
                if let Some(event) = normalize_event(&term_event) {
                    let _ = handle_surface_input(
                        ui_extensions.as_ref(),
                        &event,
                        &plugin_host,
                        &session_manager,
                        &mut app,
                    );
                }
                app.dirty = true;
            }
            AppEvent::Tick => {
                app.on_tick();
                let _ = handle_surface_input(
                    ui_extensions.as_ref(),
                    &TuiInputEvent::Tick,
                    &plugin_host,
                    &session_manager,
                    &mut app,
                );
                if last_ui_sync.elapsed() >= std::time::Duration::from_millis(250) {
                    sync_ui_extensions(
                        &mut app,
                        ui_extensions.as_ref(),
                        &plugin_host,
                        Arc::clone(&session_manager),
                    )
                    .await;
                    last_ui_sync = tokio::time::Instant::now();
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
                    if let SessionEvent::InjectedTurnInputAccepted { messages, .. } = &event {
                        app.acknowledge_monitor_wakes(messages);
                    }
                    app.handle_session_event(event);
                    for effect in ui_effects {
                        apply_ui_host_effect(
                            &mut app,
                            effect,
                            &mut history,
                            &mut runtime,
                            &mut turn_counter,
                            &mut current_execution_mode,
                            &provider,
                            &mut current_model_variant,
                            &dynamic_tools,
                            &mut desired_dynamic,
                            model_catalog.as_ref(),
                            &mut session_manager,
                            ui_extensions.as_ref(),
                            &plugin_host,
                        )
                        .await;
                    }
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

    terminal.restore();

    // Save input history
    app.save_history();

    Ok(())
}
