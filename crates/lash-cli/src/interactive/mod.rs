mod commands;
mod helpers;
mod input_handling;
mod runtime;
#[cfg(test)]
mod tests;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crossterm::event::Event as TermEvent;
use lash::{LashSession, ModeId, TurnEvent, provider::ProviderHandle};
use lash_core::runtime::RuntimeSessionState;
use lash_core::session_model::Message;
use lash_core::{TokenUsage, ToolState};
use lash_sqlite_store::Store;
use lash_tui::{InputEvent as TuiInputEvent, Terminal, normalize_event};
use lash_tui_extensions::{TuiExtensionContext, TuiExtensions, TuiSlashInvocation};
use tokio_util::sync::CancellationToken;

use crate::app::{self, App, PreparedTurn, UiTimelineItem};
use crate::command;
use crate::event::{AppEvent, AppEventPump};
use crate::model_catalog::CachedModelCatalog;
use crate::prompt_tool::CliPromptBridge;
use crate::render;
use crate::resume;
use crate::session_bootstrap::CliSessionOpener;
use crate::session_log::{self, SessionLogger};
use crate::turn_runner::{RuntimeRunResult, make_turn_input};
use crate::ui_trace::{
    UiTraceRecorder, disable_aux_op_recording, enable_aux_op_recording, render_screen_text,
};
use crate::update;
use crate::{Args, scratch_tui};
use crate::{
    apply_ui_host_effects, controls_text, hash12, help_text, info_text, push_system_message,
    turn_has_visible_output, version_text,
};

use self::helpers::{
    TurnReplayPayload, UiSnapshotWorker, cleared_session_state, drain_aux_trace_ops,
    log_runtime_transition, record_queue_turn,
};

use self::commands::{dispatch_next_queued_turn, promote_pending_steers_to_queue};
#[cfg(test)]
pub(crate) use self::runtime::injected_image_part_indices;
#[cfg(test)]
pub(crate) use self::runtime::make_injected_plugin_message;
use self::runtime::{apply_pending_reconfigure, send_user_message, sync_runtime_tool_surface};
pub(crate) use self::runtime::{generate_session_name, notify_desktop};

use self::input_handling::{
    SessionCtx, dispatch_key_event, handle_mouse_event, handle_surface_input,
};

// Items used only by tests via `use super::*;` in tests.rs.
#[cfg(test)]
#[allow(unused_imports)]
use self::helpers::{
    TEXT_DELTA_REDRAW_INTERVAL, is_copy_shortcut, should_preserve_selection_for_key,
};

fn session_activity_is_current(
    stream_id: u64,
    active_stream_id: u64,
    runtime_active: bool,
) -> bool {
    runtime_active && stream_id == active_stream_id
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_app(
    mut terminal: Terminal,
    session: LashSession,
    runtime_factory: CliSessionOpener,
    lash_config: crate::config::LashConfig,
    prompt_bridge: CliPromptBridge,
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
    initial_execution_mode: ModeId,
    startup_system_message: Option<String>,
) -> anyhow::Result<()> {
    let initial_session_id = session.session_id();
    let mut app = App::new(model, session_name, initial_session_id);
    let (chrome_ext, chrome_state) = crate::chrome_ui::ChromeTuiExtension::new();
    let extra_ui_extensions: Vec<Arc<dyn lash_tui_extensions::TuiExtension>> = vec![
        Arc::new(lash_autoresearch::AutoresearchTuiExtension::default()),
        chrome_ext,
    ];
    let ui_extensions = Arc::new(
        TuiExtensions::with_builtins(extra_ui_extensions)
            .map_err(|err| anyhow::anyhow!("failed to build UI extensions: {err}"))?,
    );
    app.set_ui_extensions(Arc::clone(&ui_extensions));
    app.set_chrome_state(chrome_state);
    app.usage.context_window = Some(initial_context_window);
    app.usage.context_usage_excludes_cached_input = provider.input_usage_excludes_cached_tokens();
    let mut current_model_variant = initial_model_variant.or_else(|| {
        crate::provider_metadata::default_model_variant_for_provider(
            provider.kind(),
            &app.model,
            provider.supported_variants(&app.model),
        )
        .map(str::to_string)
    });
    app.set_model_variant(current_model_variant.clone());
    let mut current_execution_mode = initial_execution_mode;
    app.load_history();
    let mut history: Vec<Message> = Vec::new();
    let mut turn_counter: usize = 0;
    let mut runtime = Some(session);
    let mut desired_tool_state = runtime
        .as_ref()
        .expect("session initialized")
        .control()
        .tools()
        .state()
        .await?;
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
    let mut event_pump = AppEventPump::new();
    let app_tx = event_pump.sender();
    prompt_bridge.set_event_tx(app_tx.clone());
    let mut snapshot_worker = UiSnapshotWorker::spawn(app_tx.clone(), Arc::clone(&ui_extensions));

    // Kick off the background `@`-completion file index. Starting it here
    // (rather than lazily on the first `@` keystroke) means the walk is
    // usually finished before the user can type a query, so the popup serves
    // real matches immediately. Cwd is captured once; lash-cli has no
    // mid-session cwd-change events and a stale-after-startup index is the
    // showcased v1 behavior.
    if let Ok(cwd) = std::env::current_dir() {
        let index_tx = app_tx.clone();
        let index = lash_file_index::FileIndex::for_root(
            cwd,
            Box::new(move || {
                let _ = index_tx.send(AppEvent::FileIndexReady);
            }),
        );
        app.install_file_index(index);
    }

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
    let mut last_ui_sync = tokio::time::Instant::now();

    let _ = app_tx.send(AppEvent::RequestUiSnapshot);
    if let Some(message) = startup_system_message {
        push_system_message(&mut app, message);
    }

    if let Some(filename) = args.resume.as_deref() {
        if let Err(err) = resume::load_resumed_session(
            filename,
            &mut app,
            logger,
            &mut history,
            &mut runtime,
            &mut turn_counter,
            &mut current_execution_mode,
            &provider,
            &mut current_model_variant,
            &mut desired_tool_state,
            model_catalog.as_ref(),
        )
        .await
        {
            push_system_message(&mut app, err);
        } else {
            let _ = app_tx.send(AppEvent::RequestUiSnapshot);
            toolset_hash = hash12(
                &serde_json::to_vec(&desired_tool_state.tool_manifests())
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
                &mut desired_tool_state,
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
                    &serde_json::to_vec(&desired_tool_state.tool_manifests())
                        .unwrap_or_else(|_| b"[]".to_vec()),
                );
                let prepared = PreparedTurn::prepare(prompt.to_string(), Vec::new(), &app.skills);
                let turn_input = make_turn_input(&prepared);
                let current_tool_state = desired_tool_state.clone();
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
                    &current_tool_state,
                )
                .await;
                last_turn = Some(TurnReplayPayload {
                    prepared_turn: prepared,
                    turn_input,
                    execution_mode: current_execution_mode.clone(),
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
        log_runtime_transition(
            "loop_top",
            &app,
            &runtime,
            runtime_return_rx.is_some(),
            cancel_token.is_some(),
            active_stream_id,
        );

        if !app.running && runtime.is_some() && runtime_return_rx.is_none() {
            tracing::debug!("dispatching queued turn from idle-ready trigger");
            if dispatch_next_queued_turn(
                &mut app,
                &mut ui_trace,
                &mut terminal,
                logger,
                args,
                &paused,
                ui_extensions.as_ref(),
                &runtime_factory,
                &lash_config,
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
                &mut desired_tool_state,
                &mut pending_reconfigure,
                model_catalog.as_ref(),
                &mut toolset_hash,
                &app_tx,
                &mut pending_clear_after_return,
            )
            .await?
            {
                continue;
            }
        }

        // Check if runtime turn completed — reclaim runtime + updated history
        if let Some(ref mut rx) = runtime_return_rx {
            match rx.try_recv() {
                Ok(done) => {
                    tracing::debug!(
                        stream_id = done.stream_id,
                        outcome = ?done.result.outcome,
                        active_stream_id,
                        "runtime return received in interactive loop"
                    );
                    if let Err(err) = sync_runtime_tool_surface(&mut runtime).await {
                        push_system_message(
                            &mut app,
                            format!("Failed to sync tool surface after turn: {err}"),
                        );
                    }
                    if done.stream_id != active_stream_id || pending_clear_after_return {
                        if let Some(rt) = runtime.as_mut() {
                            let preserved_policy = rt.policy_snapshot();
                            rt.control()
                                .state()
                                .set_persisted(RuntimeSessionState::from_snapshot(
                                    cleared_session_state(preserved_policy),
                                ))
                                .await?;
                        }
                        history.clear();
                        turn_counter = 0;
                        app.usage.token_usage = TokenUsage::default();
                        app.stop_turn();
                        app.clear_mode_indicators();
                        app.set_model_variant(current_model_variant.clone());
                        runtime_return_rx = None;
                        cancel_token = None;
                        pending_clear_after_return = false;
                        app.dirty = true;
                        log_runtime_transition(
                            "runtime_return_ignored_or_cleared",
                            &app,
                            &runtime,
                            runtime_return_rx.is_some(),
                            cancel_token.is_some(),
                            active_stream_id,
                        );
                        continue;
                    }
                    let interrupted = matches!(
                        &done.result.outcome,
                        lash_core::TurnOutcome::Stopped(lash_core::TurnStop::Cancelled)
                    );
                    let no_visible_output = matches!(
                        &done.result.outcome,
                        lash_core::TurnOutcome::Finished(_)
                            | lash_core::TurnOutcome::AgentFrameSwitch { .. }
                    ) && !turn_has_visible_output(&done.result);
                    let state = done.result.state;
                    tracing::info!(
                        turn_index = state.turn_index,
                        outcome = ?done.result.outcome,
                        assistant_chars = done.result.assistant_output.safe_text.len(),
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

                    let read_view = state.read_view();
                    history = read_view.messages().to_vec();
                    turn_counter = state.turn_index;
                    app.usage.token_usage = state.token_usage.clone();
                    app.usage.last_prompt_usage = state.last_prompt_usage.clone();
                    tracing::debug!(
                        stream_id = done.stream_id,
                        turn_index = state.turn_index,
                        outcome = ?done.result.outcome,
                        messages = read_view.messages().len(),
                        blocks = app.timeline.len(),
                        had_live_turn = app.live.turn.is_some(),
                        running = app.running,
                        "reconciling completed runtime turn"
                    );
                    if interrupted {
                        let had_manual_interrupt_message = matches!(
                            app.timeline.last(),
                            Some(UiTimelineItem::SystemMessage(message))
                                if message == crate::util::manual_interrupt_message()
                        );
                        let mut ui_projection_state = app.ui_projection_state();
                        ui_projection_state.live_assistant_text = app::interrupted_assistant_tail(
                            &app.timeline,
                            &done.result.assistant_output.safe_text,
                        );
                        let interrupted_message = if had_manual_interrupt_message {
                            crate::util::manual_interrupt_message().to_string()
                        } else {
                            "Cancelled.".to_string()
                        };
                        app.stop_turn();
                        app.timeline = app::interrupted_blocks_from_read_view(
                            &read_view,
                            &ui_projection_state,
                            interrupted_message.clone(),
                        );
                        app.invalidate_height_cache();
                        app.scroll_to_bottom();
                        promote_pending_steers_to_queue(&mut app, &mut ui_trace);
                        runtime_return_rx = None;
                        cancel_token = None;
                        dispatch_next_queued_turn(
                            &mut app,
                            &mut ui_trace,
                            &mut terminal,
                            logger,
                            args,
                            &paused,
                            ui_extensions.as_ref(),
                            &runtime_factory,
                            &lash_config,
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
                            &mut desired_tool_state,
                            &mut pending_reconfigure,
                            model_catalog.as_ref(),
                            &mut toolset_hash,
                            &app_tx,
                            &mut pending_clear_after_return,
                        )
                        .await?;
                        continue;
                    }

                    app.finish_turn_from_read_view(&read_view);
                    runtime_return_rx = None;
                    cancel_token = None;
                    log_runtime_transition(
                        "runtime_return_completed",
                        &app,
                        &runtime,
                        runtime_return_rx.is_some(),
                        cancel_token.is_some(),
                        active_stream_id,
                    );
                    dispatch_next_queued_turn(
                        &mut app,
                        &mut ui_trace,
                        &mut terminal,
                        logger,
                        args,
                        &paused,
                        ui_extensions.as_ref(),
                        &runtime_factory,
                        &lash_config,
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
                        &mut desired_tool_state,
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
                    app.stop_turn();
                    runtime_return_rx = None;
                    cancel_token = None;
                    log_runtime_transition(
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
            let render_started = std::time::Instant::now();
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
            let render_elapsed = render_started.elapsed();
            if render_elapsed > std::time::Duration::from_millis(16) {
                tracing::warn!(duration_ms = render_elapsed.as_millis(), "slow TUI render");
            } else {
                tracing::trace!(
                    duration_ms = render_elapsed.as_millis(),
                    "TUI render completed"
                );
            }
        }

        // Wait for next event
        let Some(queued_event) = event_pump.recv().await else {
            break;
        };
        let input_latency = queued_event.enqueued_at.elapsed();
        let (high_depth, normal_depth, low_depth) = event_pump.lane_depths();
        if input_latency > std::time::Duration::from_millis(100) {
            tracing::warn!(
                lane = ?queued_event.lane,
                latency_ms = input_latency.as_millis(),
                high_depth,
                normal_depth,
                low_depth,
                "slow TUI event handling latency"
            );
        } else {
            tracing::trace!(
                lane = ?queued_event.lane,
                latency_ms = input_latency.as_millis(),
                high_depth,
                normal_depth,
                low_depth,
                "TUI event dequeued"
            );
        }
        let handler_started = std::time::Instant::now();
        let event = queued_event.event;

        match event {
            AppEvent::FrameRequested => {
                event_pump.mark_frame_consumed();
                app.dirty = true;
            }
            AppEvent::RequestUiSnapshot => {
                if let Some(session) = runtime.as_ref() {
                    snapshot_worker.request(session.clone());
                }
            }
            AppEvent::UiSnapshot { generation, result } => {
                let Some(session) = runtime.as_ref().cloned() else {
                    continue;
                };
                if !snapshot_worker.complete(generation, session) {
                    tracing::debug!(generation, "discarding stale UI snapshot");
                    continue;
                }
                tracing::trace!(
                    generation,
                    duration_ms = result.duration.as_millis(),
                    timed_out = result.timed_out,
                    "applying UI snapshot"
                );
                if let Some(tasks) = result.processes {
                    app.update_processes(tasks);
                }
                for diagnostic in result.diagnostics {
                    tracing::warn!(%diagnostic, "UI snapshot diagnostic");
                }
                apply_ui_host_effects(&mut app, result.effects);
                app_tx.request_frame();
            }
            AppEvent::Terminal(TermEvent::Paste(text)) => {
                app.dirty = true;

                if app.has_prompt() {
                    if app.is_prompt_text_entry() {
                        if let Some(recorder) = ui_trace.as_mut() {
                            recorder.record_prompt_insert_text(text.clone());
                        }
                        app.prompt_insert_text(&text);
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
                        runtime.as_ref(),
                        &mut app,
                    )
                {
                    continue;
                }

                if let Some(recorder) = ui_trace.as_mut() {
                    recorder.record_input_insert_text(text.clone());
                }
                app.insert_pasted_text(&text);
                app.update_suggestions();
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
            AppEvent::FileIndexReady => {
                // Walker just finished. If the user is currently sitting on
                // an "indexing files…" placeholder for an `@`-token, refresh
                // the popup so they see real matches without typing again.
                if app.suggestion_kind() == crate::editor::SuggestionKind::Indexing {
                    app.update_suggestions();
                    app.dirty = true;
                }
            }
            AppEvent::Terminal(TermEvent::Key(key)) => {
                let mut ctx = SessionCtx {
                    app: &mut app,
                    terminal: &mut terminal,
                    ui_trace: &mut ui_trace,
                    logger: &mut *logger,
                    args,
                    paused: &paused,
                    ui_extensions: ui_extensions.as_ref(),
                    runtime_factory: &runtime_factory,
                    lash_config: &lash_config,
                    runtime: &mut runtime,
                    history: &mut history,
                    turn_counter: &mut turn_counter,
                    last_turn: &mut last_turn,
                    runtime_return_rx: &mut runtime_return_rx,
                    cancel_token: &mut cancel_token,
                    active_stream_id: &mut active_stream_id,
                    provider: &mut provider,
                    current_model_variant: &mut current_model_variant,
                    current_execution_mode: &mut current_execution_mode,
                    desired_tool_state: &mut desired_tool_state,
                    pending_reconfigure: &mut pending_reconfigure,
                    model_catalog: model_catalog.as_ref(),
                    toolset_hash: &mut toolset_hash,
                    app_tx: &app_tx,
                    pending_clear_after_return: &mut pending_clear_after_return,
                };
                if dispatch_key_event(key, &mut ctx).await? {
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
                    &runtime,
                )?;
            }
            AppEvent::Terminal(TermEvent::FocusGained) => {
                app.focused = true;
                let _ = handle_surface_input(
                    ui_extensions.as_ref(),
                    &TuiInputEvent::FocusGained,
                    runtime.as_ref(),
                    &mut app,
                );
                app.dirty = true;
            }
            AppEvent::Terminal(TermEvent::FocusLost) => {
                app.focused = false;
                let _ = handle_surface_input(
                    ui_extensions.as_ref(),
                    &TuiInputEvent::FocusLost,
                    runtime.as_ref(),
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
                        runtime.as_ref(),
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
                    runtime.as_ref(),
                    &mut app,
                );
                if last_ui_sync.elapsed() >= std::time::Duration::from_millis(250) {
                    let _ = app_tx.send(AppEvent::RequestUiSnapshot);
                    last_ui_sync = tokio::time::Instant::now();
                }
            }
            AppEvent::Session {
                stream_id,
                activity,
            } => {
                if !session_activity_is_current(
                    stream_id,
                    active_stream_id,
                    runtime_return_rx.is_some(),
                ) {
                    tracing::trace!(
                        stream_id,
                        active_stream_id,
                        runtime_active = runtime_return_rx.is_some(),
                        "dropping stale runtime activity"
                    );
                    continue;
                }
                let activity = *activity;
                if matches!(activity.event, TurnEvent::AssistantProseDelta { .. }) {
                    if let Some(recorder) = ui_trace.as_mut() {
                        recorder.record_turn_activity(&activity);
                    }
                    app.handle_turn_activity(activity);
                    app.dirty = true;
                    continue;
                }
                app.dirty = true;
                if let Some(recorder) = ui_trace.as_mut() {
                    recorder.record_turn_activity(&activity);
                }
                let ui_effects = ui_extensions.effects_for_turn_event(&activity.event);
                app.handle_turn_activity(activity);
                apply_ui_host_effects(&mut app, ui_effects);
            }
            AppEvent::Prompt {
                request,
                response_tx,
            } => {
                app.dirty = true;
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
            }
            AppEvent::Quit => break,
        }

        let handler_elapsed = handler_started.elapsed();
        if handler_elapsed > std::time::Duration::from_millis(16) {
            tracing::warn!(
                duration_ms = handler_elapsed.as_millis(),
                "slow foreground TUI handler"
            );
        }
        drain_aux_trace_ops(&mut ui_trace);
    }

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
