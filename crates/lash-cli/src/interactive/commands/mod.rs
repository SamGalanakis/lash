mod provider;
mod session;
mod tools;

pub(crate) use session::switch_to_session_identifier;

use super::runtime::send_queued_work;
use super::runtime::sync_runtime_tool_surface;
use super::*;
use crate::SkillCatalog;
use crate::app::PendingImage;
use crate::turn_runner::make_turn_input;

#[derive(Clone)]
pub(super) enum ParsedSlashCommand {
    Builtin(command::Command),
    Ui(TuiSlashInvocation),
}

/// The mutable interactive-loop state a slash command may touch.
///
/// Bundling it into one value keeps the dispatcher and command handlers from
/// threading two dozen individual references through every call. The command
/// handlers destructure it back into locals at the top, so their bodies are
/// unchanged; the loop builds one of these (with explicit reborrows) per
/// dispatch.
pub(super) struct SlashCommandCtx<'a> {
    pub(super) terminal: &'a mut Terminal,
    pub(super) app: &'a mut App,
    pub(super) logger: &'a mut SessionLogger,
    pub(super) args: &'a Args,
    pub(super) paused: &'a Arc<AtomicBool>,
    pub(super) ui_extensions: &'a TuiExtensions,
    pub(super) runtime_factory: &'a crate::session_bootstrap::CliSessionOpener,
    pub(super) lash_config: &'a crate::config::LashConfig,
    pub(super) runtime: &'a mut Option<LashSession>,
    pub(super) history: &'a mut Vec<Message>,
    pub(super) turn_counter: &'a mut usize,
    pub(super) last_turn: &'a mut Option<TurnReplayPayload>,
    pub(super) runtime_return_rx: &'a mut Option<tokio::sync::oneshot::Receiver<RuntimeRunResult>>,
    pub(super) cancel_token: &'a mut Option<CancellationToken>,
    pub(super) active_stream_id: &'a mut u64,
    pub(super) provider: &'a mut ProviderHandle,
    pub(super) current_model_variant: &'a mut Option<String>,
    pub(super) current_execution_mode: &'a mut ModeId,
    pub(super) desired_tool_state: &'a mut ToolState,
    pub(super) pending_reconfigure: &'a mut bool,
    pub(super) model_catalog: &'a CachedModelCatalog,
    pub(super) toolset_hash: &'a mut String,
    pub(super) app_tx: &'a crate::event::AppEventTx,
    pub(super) pending_clear_after_return: &'a mut bool,
}

pub(super) fn parse_slash_command(
    input: &str,
    skills: &SkillCatalog,
    ui_extensions: &TuiExtensions,
) -> Option<ParsedSlashCommand> {
    command::parse(input, skills)
        .map(ParsedSlashCommand::Builtin)
        .or_else(|| {
            ui_extensions
                .parse_command(input)
                .map(ParsedSlashCommand::Ui)
        })
}

pub(super) fn slash_command_runs_out_of_band_while_running(cmd: &ParsedSlashCommand) -> bool {
    match cmd {
        ParsedSlashCommand::Builtin(command) => command::runs_out_of_band_while_running(command),
        ParsedSlashCommand::Ui(command) => command.allow_while_running(),
    }
}

async fn handle_ui_command(
    invocation: TuiSlashInvocation,
    app: &mut App,
    ui_extensions: &TuiExtensions,
    runtime: &mut Option<LashSession>,
) {
    let Some(session) = runtime.as_ref() else {
        push_system_message(app, "No active session for UI command.".to_string());
        return;
    };
    match ui_extensions
        .invoke_parsed_command(
            &invocation,
            TuiExtensionContext {
                actions: &session.plugin_actions(),
            },
        )
        .await
    {
        Ok(effects) => {
            if let Err(err) = sync_runtime_tool_surface(runtime).await {
                push_system_message(app, format!("Failed to sync tool surface: {err}"));
            }
            apply_ui_host_effects(app, effects)
        }
        Err(err) => push_system_message(app, err),
    }
}

pub(super) fn promote_pending_steers_to_queue(
    app: &mut App,
    ui_trace: &mut Option<UiTraceRecorder>,
) {
    while let Some(turn) = app.queues.pending_steers.pop_front() {
        record_queue_turn(ui_trace, &turn);
        app.queue_turn(turn);
    }
}

fn current_epoch_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u128::from(u64::MAX)) as u64
}

fn ready_batches_for_idle_dispatch(
    batches: &[lash_core::QueuedWorkBatch],
) -> Vec<lash_core::QueuedWorkBatch> {
    let now = current_epoch_ms();
    let ready = batches
        .iter()
        .filter(|batch| batch.available_at_ms <= now)
        .collect::<Vec<_>>();
    let Some(first) = ready.first() else {
        return Vec::new();
    };
    let first_slot_policy = first.slot_policy;
    let first_delivery_policy = first.delivery_policy;
    let first_merge_key = first.merge_key.clone();
    let mut selected = Vec::new();
    for batch in ready {
        if selected.len() >= 64 {
            break;
        }
        if selected.is_empty() {
            selected.push((*batch).clone());
            if first_slot_policy == lash_core::SlotPolicy::Exclusive {
                break;
            }
            continue;
        }
        if first_slot_policy != lash_core::SlotPolicy::Join
            || batch.slot_policy != lash_core::SlotPolicy::Join
            || batch.delivery_policy != first_delivery_policy
            || batch.merge_key != first_merge_key
        {
            break;
        }
        selected.push((*batch).clone());
    }
    selected
}

fn turn_input_display_text(input: &lash_core::TurnInput) -> String {
    input
        .items
        .iter()
        .filter_map(|item| match item {
            lash_core::InputItem::Text { text } => Some(text.as_str()),
            lash_core::InputItem::ImageRef { .. } => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

fn prepared_turn_from_queued_input(input: &lash_core::TurnInput) -> Option<PreparedTurn> {
    let text = turn_input_display_text(input);
    let images = input
        .items
        .iter()
        .filter_map(|item| match item {
            lash_core::InputItem::ImageRef { id } => input.image_blobs.get(id),
            lash_core::InputItem::Text { .. } => None,
        })
        .enumerate()
        .map(|(idx, png_bytes)| PendingImage {
            id: idx + 1,
            png_bytes: png_bytes.clone(),
        })
        .collect::<Vec<_>>();
    if text.trim().is_empty() && images.is_empty() {
        return None;
    }
    Some(PreparedTurn::prepare_with_effective_text(
        text.clone(),
        text,
        images,
    ))
}

fn display_turns_for_queued_batches(
    app: &mut App,
    batches: &[lash_core::QueuedWorkBatch],
) -> Vec<PreparedTurn> {
    let mut turns = Vec::new();
    for batch in batches {
        let queue_draft_id = batch.source_key.as_deref().and_then(|source| {
            source
                .strip_prefix("host:")
                .or_else(|| source.strip_prefix("injection:"))
        });
        for item in &batch.items {
            let lash_core::QueuedWorkPayload::TurnInput { input } = &item.payload else {
                continue;
            };
            if let Some(draft_id) = queue_draft_id
                && let Some(turn) = app.take_queued_turn_by_draft_id(draft_id)
            {
                turns.push(turn);
                continue;
            }
            let content = turn_input_display_text(input);
            if let Some(turn) = app.take_matching_queued_turn(&content) {
                turns.push(turn);
                continue;
            }
            if let Some(turn) = prepared_turn_from_queued_input(input) {
                turns.push(turn);
            }
        }
    }
    turns
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn dispatch_next_queued_turn(
    app: &mut App,
    ui_trace: &mut Option<UiTraceRecorder>,
    _terminal: &mut Terminal,
    logger: &mut SessionLogger,
    _args: &Args,
    _paused: &Arc<AtomicBool>,
    _ui_extensions: &TuiExtensions,
    _runtime_factory: &crate::session_bootstrap::CliSessionOpener,
    _lash_config: &crate::config::LashConfig,
    runtime: &mut Option<LashSession>,
    _history: &mut Vec<Message>,
    _turn_counter: &mut usize,
    last_turn: &mut Option<TurnReplayPayload>,
    runtime_return_rx: &mut Option<tokio::sync::oneshot::Receiver<RuntimeRunResult>>,
    cancel_token: &mut Option<CancellationToken>,
    active_stream_id: &mut u64,
    _provider: &mut ProviderHandle,
    _current_model_variant: &mut Option<String>,
    current_execution_mode: &mut ModeId,
    desired_tool_state: &mut ToolState,
    pending_reconfigure: &mut bool,
    _model_catalog: &CachedModelCatalog,
    toolset_hash: &mut String,
    app_tx: &crate::event::AppEventTx,
    _pending_clear_after_return: &mut bool,
) -> anyhow::Result<bool> {
    let Some(session) = runtime.as_ref().cloned() else {
        tracing::debug!("queued dispatch paused because runtime is unavailable");
        return Ok(false);
    };
    let queued_batches = session.queued_work().await?;
    let ready_batches = ready_batches_for_idle_dispatch(&queued_batches);
    if ready_batches.is_empty() {
        return Ok(false);
    }

    if let Err(e) =
        apply_pending_reconfigure(desired_tool_state, pending_reconfigure, runtime).await
    {
        push_system_message(
            app,
            format!(
                "Pending runtime reconfigure failed; queued message not sent: {}",
                e
            ),
        );
        return Ok(false);
    }
    *toolset_hash = hash12(
        &serde_json::to_vec(&desired_tool_state.tool_manifests())
            .unwrap_or_else(|_| b"[]".to_vec()),
    );
    let display_turns = display_turns_for_queued_batches(app, &ready_batches);
    if let Some(first_turn) = display_turns.first().cloned() {
        *last_turn = Some(TurnReplayPayload {
            turn_input: make_turn_input(&first_turn),
            prepared_turn: first_turn,
            execution_mode: current_execution_mode.clone(),
        });
    }
    send_queued_work(
        display_turns,
        app,
        ui_trace.as_mut(),
        logger,
        runtime,
        runtime_return_rx,
        cancel_token,
        active_stream_id,
        app_tx,
    )
    .await;
    Ok(true)
}

pub(super) async fn handle_parsed_slash_command(
    command: ParsedSlashCommand,
    ctx: SlashCommandCtx<'_>,
) -> anyhow::Result<bool> {
    match command {
        ParsedSlashCommand::Builtin(command) => handle_slash_command(command, ctx).await,
        ParsedSlashCommand::Ui(command) => {
            handle_ui_command(command, ctx.app, ctx.ui_extensions, ctx.runtime).await;
            Ok(false)
        }
    }
}

async fn handle_slash_command(
    cmd: command::Command,
    ctx: SlashCommandCtx<'_>,
) -> anyhow::Result<bool> {
    let SlashCommandCtx {
        terminal,
        app,
        logger,
        args: _args,
        paused,
        ui_extensions: _,
        runtime_factory,
        lash_config: _lash_config,
        runtime,
        history,
        turn_counter,
        last_turn,
        runtime_return_rx,
        cancel_token,
        active_stream_id,
        provider,
        current_model_variant,
        current_execution_mode,
        desired_tool_state,
        pending_reconfigure,
        model_catalog,
        toolset_hash,
        app_tx,
        pending_clear_after_return,
    } = ctx;
    match cmd {
        command::Command::Exit => Ok(true),
        command::Command::Clear => {
            session::handle_clear(
                app,
                runtime_factory,
                logger,
                runtime,
                history,
                turn_counter,
                last_turn,
                active_stream_id,
                provider,
                current_model_variant,
                current_execution_mode,
                desired_tool_state,
                pending_clear_after_return,
            )
            .await
        }
        command::Command::Version => {
            push_system_message(app, version_text());
            Ok(false)
        }
        command::Command::Info => {
            let model = app.model.clone();
            let context_window = app.usage.context_window;
            let cwd = app.cwd.clone();
            let session_name = app.session_name.clone();
            let standard_context_approach = (current_execution_mode == &ModeId::standard())
                .then(lash_standard_plugins::StandardContextApproach::default);
            let session_db_path = logger.db_path().to_string_lossy().to_string();
            push_system_message(
                app,
                info_text(
                    provider,
                    &model,
                    current_model_variant.as_deref(),
                    current_execution_mode,
                    standard_context_approach.as_ref(),
                    context_window,
                    Some((desired_tool_state.len(), toolset_hash)),
                    &cwd,
                    Some(&session_name),
                    Some(&logger.session_id),
                    Some(&session_db_path),
                ),
            );
            Ok(false)
        }
        command::Command::Model(new_model) => {
            provider::handle_model(
                new_model,
                app,
                runtime,
                provider,
                current_model_variant,
                model_catalog,
            )
            .await
        }
        command::Command::Variant(new_variant) => {
            provider::handle_variant(new_variant, app, runtime, provider, current_model_variant)
                .await
        }
        command::Command::Mode(new_mode) => {
            provider::handle_mode(new_mode, app, current_execution_mode)
        }
        command::Command::ChangeProvider => {
            provider::handle_change_provider(
                terminal,
                app,
                paused,
                provider,
                current_model_variant,
                model_catalog,
                runtime,
            )
            .await
        }
        command::Command::Logout => provider::handle_logout(app, provider),
        command::Command::Retry => {
            session::handle_retry(
                app,
                logger,
                runtime,
                history,
                last_turn,
                runtime_return_rx,
                cancel_token,
                active_stream_id,
                current_execution_mode,
                desired_tool_state,
                pending_reconfigure,
                toolset_hash,
                app_tx,
            )
            .await
        }
        command::Command::Controls => {
            push_system_message(app, controls_text(app.ui_extensions()));
            Ok(false)
        }
        command::Command::Fork => {
            session::handle_fork(
                app,
                logger,
                runtime,
                provider,
                current_model_variant,
                toolset_hash,
            )
            .await
        }
        command::Command::Tree => session::handle_tree(app, runtime).await,
        command::Command::Help => {
            let help = help_text(&app.skills, app.ui_extensions());
            push_system_message(app, help);
            Ok(false)
        }
        command::Command::Resume(name) => {
            session::handle_resume(
                name,
                app,
                logger,
                runtime_factory,
                runtime,
                history,
                turn_counter,
                last_turn,
                provider,
                current_model_variant,
                current_execution_mode,
                desired_tool_state,
                model_catalog,
                toolset_hash,
            )
            .await
        }
        command::Command::Tools(raw) => {
            tools::handle_tools(
                raw,
                app,
                runtime,
                desired_tool_state,
                pending_reconfigure,
                current_execution_mode.clone(),
            )
            .await
        }
        command::Command::Reconfigure(raw) => {
            tools::handle_reconfigure(
                raw,
                app,
                runtime,
                desired_tool_state,
                pending_reconfigure,
                toolset_hash,
            )
            .await
        }
        command::Command::Skills => session::handle_skills(app),
        command::Command::Compact(argument) => {
            let Some(rt) = runtime.as_mut() else {
                push_system_message(app, "Compaction is unavailable while a turn is running.");
                return Ok(false);
            };
            let trigger = lash_core::RewriteTrigger::Manual {
                instructions: argument,
            };
            match rt.control().state().rewrite_history(trigger).await {
                Ok(true) => {
                    let read_view = rt.read_view();
                    history.clear();
                    history.extend(read_view.messages().iter().cloned());
                    app.timeline =
                        app::timeline_from_read_view(&read_view, &app.ui_projection_state());
                    app.invalidate_height_cache();
                    app.scroll_to_bottom();
                    push_system_message(app, "Compaction summary inserted.");
                }
                Ok(false) => push_system_message(
                    app,
                    "Nothing to compact yet — the conversation is still short.",
                ),
                Err(err) => push_system_message(app, format!("Compaction failed: {err}")),
            }
            Ok(false)
        }
    }
}
