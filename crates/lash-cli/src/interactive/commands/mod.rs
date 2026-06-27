mod provider;
mod session;

pub(crate) use session::switch_to_session_identifier;

use super::runtime::send_queued_work;
use super::runtime::sync_runtime_tool_catalog;
use super::*;
use crate::SkillCatalog;
use crate::info::{controls_document, help_document, info_document};
use crate::turn_runner::make_turn_input;
use lash_core::runtime::ExecutionScope;

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
    pub(super) runtime_factory: &'a crate::startup::session::CliSessionOpener,
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
    pub(super) current_execution_mode: &'a mut ExecutionMode,
    pub(super) active_tool_state: &'a mut ToolState,
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
    if runtime.is_none() {
        push_system_message(app, "No active session for UI command.".to_string());
        return;
    }
    match ui_extensions
        .invoke_parsed_command(&invocation, TuiExtensionContext)
        .await
    {
        Ok(effects) => {
            if let Err(err) = sync_runtime_tool_catalog(runtime).await {
                push_system_message(app, format!("Failed to sync tool catalog: {err}"));
            }
            crate::ui_effects::apply_ui_host_effects_with_runtime(
                app,
                ui_extensions,
                runtime,
                effects,
            )
            .await
        }
        Err(err) => push_system_message(app, err),
    }
}

fn turn_input_has_visible_content(input: &lash_core::TurnInput) -> bool {
    input.items.iter().any(|item| match item {
        lash_core::InputItem::Text { text } => !text.trim().is_empty(),
        lash_core::InputItem::ImageRef { id } => input.image_blobs.contains_key(id),
    })
}

fn display_turns_for_pending_inputs(
    app: &mut App,
    inputs: &[lash_core::PendingTurnInput],
) -> Vec<PreparedTurn> {
    let mut turns = Vec::new();
    for input in inputs {
        if let Some(turn) = app.take_prepared_turn_for_pending_input(input) {
            turns.push(turn);
        }
    }
    turns
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn dispatch_queued_turn(
    app: &mut App,
    ui_trace: &mut Option<UiTraceRecorder>,
    _terminal: &mut Terminal,
    logger: &mut SessionLogger,
    _args: &Args,
    _paused: &Arc<AtomicBool>,
    _ui_extensions: &TuiExtensions,
    _runtime_factory: &crate::startup::session::CliSessionOpener,
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
    current_execution_mode: &mut ExecutionMode,
    app_tx: &crate::event::AppEventTx,
    _pending_clear_after_return: &mut bool,
) -> anyhow::Result<bool> {
    let Some(session) = runtime.as_ref().cloned() else {
        tracing::debug!("queued dispatch paused because runtime is unavailable");
        return Ok(false);
    };
    let pending_inputs = session.pending_turn_inputs().await?;
    let ready_inputs = pending_inputs
        .into_iter()
        .filter(|input| {
            input.state.is_next_turn_pending()
                && turn_input_has_visible_content(&input.input)
                && !app.queued_input_preview_suppressed(input)
        })
        .take(64)
        .collect::<Vec<_>>();
    if ready_inputs.is_empty() {
        return Ok(false);
    }

    let ready_input_ids = ready_inputs
        .iter()
        .map(|input| input.input_id.clone())
        .collect::<Vec<_>>();
    app.suppress_queue_preview_inputs(ready_input_ids.iter().map(String::as_str));
    app.remove_pending_turn_inputs(&ready_input_ids);
    let display_turns = display_turns_for_pending_inputs(app, &ready_inputs);
    if let Some(first_turn) = display_turns.first().cloned() {
        *last_turn = Some(TurnReplayPayload {
            turn_input: make_turn_input(&first_turn),
            prepared_turn: first_turn,
            execution_mode: *current_execution_mode,
        });
    }
    send_queued_work(
        Vec::new(),
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
        ParsedSlashCommand::Builtin(command) => handle_builtin_command(command, ctx).await,
        ParsedSlashCommand::Ui(command) => {
            handle_ui_command(command, ctx.app, ctx.ui_extensions, ctx.runtime).await;
            Ok(false)
        }
    }
}

pub(super) async fn handle_builtin_command(
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
        active_tool_state,
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
                active_tool_state,
                pending_clear_after_return,
            )
            .await
        }
        command::Command::Version => {
            push_system_message(app, version_text());
            Ok(false)
        }
        command::Command::Info => {
            if let Some(session) = runtime.as_ref()
                && let Ok(state) = session.admin().tools().state().await
            {
                *active_tool_state = state;
                *toolset_hash = hash12(
                    &serde_json::to_vec(&active_tool_state.tool_manifests())
                        .unwrap_or_else(|_| b"[]".to_vec()),
                );
            }
            let model = app.model.clone();
            let context_window = app.usage.context_window;
            let cwd = app.cwd.clone();
            let session_name = app.session_name.clone();
            let standard_context_approach = (current_execution_mode == &ExecutionMode::Standard)
                .then(lash_standard_plugins::StandardContextApproach::default);
            let session_db_path = logger.db_path().to_string_lossy().to_string();
            app.show_document(info_document(
                provider,
                &model,
                current_model_variant.as_deref(),
                current_execution_mode,
                standard_context_approach.as_ref(),
                context_window,
                Some((active_tool_state.len(), toolset_hash)),
                &cwd,
                Some(&session_name),
                Some(&logger.session_id),
                Some(&session_db_path),
            ));
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
                toolset_hash,
                app_tx,
            )
            .await
        }
        command::Command::Controls => {
            app.show_document(controls_document(app.ui_extensions()));
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
            app.show_document(help_document(&app.skills, app.ui_extensions()));
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
                active_tool_state,
                model_catalog,
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
            let effect_host = rt.effect_host();
            let scoped_effect_controller = effect_host.scoped(ExecutionScope::runtime_operation(
                format!("cli-compact:{}:{}", rt.session_id(), uuid::Uuid::new_v4()),
            ));
            let scoped_effect_controller = match scoped_effect_controller {
                Ok(controller) => controller,
                Err(err) => {
                    push_system_message(app, format!("Compaction failed: {err}"));
                    return Ok(false);
                }
            };
            match rt
                .admin()
                .state()
                .compact_context(argument, scoped_effect_controller)
                .await
            {
                Ok(true) => {
                    let read_view = rt.read_view();
                    history.clear();
                    history.extend(read_view.messages().iter().cloned());
                    app.timeline =
                        app::timeline_from_read_view(&read_view, &app.ui_projection_state());
                    app.invalidate_height_cache();
                    app.scroll_to_bottom();
                    push_system_message(app, "Compaction frame opened.");
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
