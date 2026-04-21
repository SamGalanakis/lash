mod plugin;
mod provider;
mod session;
mod tools;

use super::runtime::send_user_message;
use super::runtime::sync_runtime_tool_surface;
use super::*;
use crate::turn_runner::make_turn_input;

#[derive(Clone)]
pub(super) enum ParsedSlashCommand {
    Builtin(command::Command),
    Ui(UiCommandInvocation),
}

pub(super) fn parse_slash_command(
    input: &str,
    skills: &SkillCatalog,
    ui_extensions: &UiExtensions,
    plugin_commands: &[lash::CommandDef],
) -> Option<ParsedSlashCommand> {
    command::parse(input, skills, plugin_commands)
        .map(ParsedSlashCommand::Builtin)
        .or_else(|| {
            ui_extensions
                .parse_command(input)
                .map(ParsedSlashCommand::Ui)
        })
}

pub(super) fn slash_command_runs_out_of_band_while_running(
    cmd: &ParsedSlashCommand,
    plugin_commands: &[lash::CommandDef],
) -> bool {
    match cmd {
        ParsedSlashCommand::Builtin(command) => {
            command::runs_out_of_band_while_running(command, plugin_commands)
        }
        ParsedSlashCommand::Ui(command) => command.allow_while_running(),
    }
}

async fn handle_ui_command(
    invocation: UiCommandInvocation,
    app: &mut App,
    ui_extensions: &UiExtensions,
    plugin_host: &PluginHost,
    session_manager: &Arc<dyn SessionManager>,
    runtime: &mut Option<LashRuntime>,
) {
    match ui_extensions
        .invoke_command(
            &invocation,
            UiContext {
                plugin_host,
                session_id: crate::ROOT_SESSION_ID,
                session_manager: Arc::clone(session_manager),
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
    while let Some(turn) = app.pending_steers.pop_front() {
        record_queue_turn(ui_trace, &turn);
        app.queue_turn(turn);
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn dispatch_next_queued_turn(
    app: &mut App,
    ui_trace: &mut Option<UiTraceRecorder>,
    terminal: &mut Terminal,
    logger: &mut SessionLogger,
    args: &Args,
    paused: &Arc<AtomicBool>,
    plugin_host: &PluginHost,
    ui_extensions: &UiExtensions,
    dynamic_tools: &Arc<DynamicToolProvider>,
    runtime: &mut Option<LashRuntime>,
    history: &mut Vec<Message>,
    turn_counter: &mut usize,
    last_turn: &mut Option<TurnReplayPayload>,
    runtime_return_rx: &mut Option<tokio::sync::oneshot::Receiver<RuntimeRunResult>>,
    cancel_token: &mut Option<CancellationToken>,
    active_stream_id: &mut u64,
    provider: &mut ProviderHandle,
    current_model_variant: &mut Option<String>,
    current_execution_mode: &mut ExecutionMode,
    session_manager: &mut Arc<dyn SessionManager>,
    desired_dynamic: &mut DynamicStateSnapshot,
    pending_reconfigure: &mut bool,
    model_catalog: &CachedModelCatalog,
    toolset_hash: &mut String,
    app_tx: &mpsc::UnboundedSender<AppEvent>,
    pending_clear_after_return: &mut bool,
) -> anyhow::Result<()> {
    while let Some((queued, was_pending)) = app.take_next_queued_turn() {
        if runtime.is_none() {
            tracing::debug!(
                queued = queued.display_text,
                was_pending,
                "queued dispatch paused because runtime is still unavailable"
            );
            app.requeue_front(queued, was_pending);
            return Ok(());
        }
        let queued = normalize_prepared_turn_for_dispatch(queued, &app.skills);
        if let Some(cmd) = parse_slash_command(
            &queued.display_text,
            &app.skills,
            ui_extensions,
            &app.plugin_commands,
        ) {
            if let Some(recorder) = ui_trace.as_mut() {
                recorder.record_slash_command(queued.display_text.clone());
            }
            if handle_parsed_slash_command(
                cmd,
                terminal,
                app,
                logger,
                args,
                paused,
                plugin_host,
                ui_extensions,
                dynamic_tools,
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
                session_manager,
                desired_dynamic,
                pending_reconfigure,
                model_catalog,
                toolset_hash,
                app_tx,
                pending_clear_after_return,
            )
            .await?
            {
                return Ok(());
            }
            continue;
        }

        if let Err(e) =
            apply_pending_reconfigure(dynamic_tools, desired_dynamic, pending_reconfigure, runtime)
                .await
        {
            push_system_message(
                app,
                format!(
                    "Pending runtime reconfigure failed; queued message not sent: {}",
                    e
                ),
            );
            app.requeue_front(queued, was_pending);
            return Ok(());
        }
        *toolset_hash = hash12(
            &serde_json::to_vec(&dynamic_tools.definitions()).unwrap_or_else(|_| b"[]".to_vec()),
        );
        let turn_input = make_turn_input(&queued);
        let current_dynamic_state = dynamic_tools.export_state();
        send_user_message(
            queued.clone(),
            turn_input.clone(),
            app,
            ui_trace.as_mut(),
            logger,
            runtime,
            history,
            runtime_return_rx,
            cancel_token,
            active_stream_id,
            app_tx,
            &current_dynamic_state,
        )
        .await;
        *last_turn = Some(TurnReplayPayload {
            prepared_turn: queued,
            turn_input,
            execution_mode: *current_execution_mode,
        });
        return Ok(());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn handle_parsed_slash_command(
    command: ParsedSlashCommand,
    terminal: &mut Terminal,
    app: &mut App,
    logger: &mut SessionLogger,
    args: &Args,
    paused: &Arc<AtomicBool>,
    plugin_host: &PluginHost,
    ui_extensions: &UiExtensions,
    dynamic_tools: &Arc<DynamicToolProvider>,
    runtime: &mut Option<LashRuntime>,
    history: &mut Vec<Message>,
    turn_counter: &mut usize,
    last_turn: &mut Option<TurnReplayPayload>,
    runtime_return_rx: &mut Option<tokio::sync::oneshot::Receiver<RuntimeRunResult>>,
    cancel_token: &mut Option<CancellationToken>,
    active_stream_id: &mut u64,
    provider: &mut ProviderHandle,
    current_model_variant: &mut Option<String>,
    current_execution_mode: &mut ExecutionMode,
    session_manager: &mut Arc<dyn SessionManager>,
    desired_dynamic: &mut DynamicStateSnapshot,
    pending_reconfigure: &mut bool,
    model_catalog: &CachedModelCatalog,
    toolset_hash: &mut String,
    app_tx: &mpsc::UnboundedSender<AppEvent>,
    pending_clear_after_return: &mut bool,
) -> anyhow::Result<bool> {
    match command {
        ParsedSlashCommand::Builtin(command) => {
            handle_slash_command(
                command,
                terminal,
                app,
                logger,
                args,
                paused,
                plugin_host,
                dynamic_tools,
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
                session_manager,
                desired_dynamic,
                pending_reconfigure,
                model_catalog,
                toolset_hash,
                app_tx,
                pending_clear_after_return,
            )
            .await
        }
        ParsedSlashCommand::Ui(command) => {
            handle_ui_command(
                command,
                app,
                ui_extensions,
                plugin_host,
                session_manager,
                runtime,
            )
            .await;
            Ok(false)
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn handle_slash_command(
    cmd: command::Command,
    terminal: &mut Terminal,
    app: &mut App,
    logger: &mut SessionLogger,
    _args: &Args,
    paused: &Arc<AtomicBool>,
    plugin_host: &PluginHost,
    dynamic_tools: &Arc<DynamicToolProvider>,
    runtime: &mut Option<LashRuntime>,
    history: &mut Vec<Message>,
    turn_counter: &mut usize,
    last_turn: &mut Option<TurnReplayPayload>,
    runtime_return_rx: &mut Option<tokio::sync::oneshot::Receiver<RuntimeRunResult>>,
    cancel_token: &mut Option<CancellationToken>,
    active_stream_id: &mut u64,
    provider: &mut ProviderHandle,
    current_model_variant: &mut Option<String>,
    current_execution_mode: &mut ExecutionMode,
    session_manager: &mut Arc<dyn SessionManager>,
    desired_dynamic: &mut DynamicStateSnapshot,
    pending_reconfigure: &mut bool,
    model_catalog: &CachedModelCatalog,
    toolset_hash: &mut String,
    app_tx: &mpsc::UnboundedSender<AppEvent>,
    pending_clear_after_return: &mut bool,
) -> anyhow::Result<bool> {
    match cmd {
        command::Command::Exit => Ok(true),
        command::Command::Clear => {
            session::handle_clear(
                app,
                plugin_host,
                runtime,
                history,
                turn_counter,
                last_turn,
                active_stream_id,
                current_model_variant,
                current_execution_mode,
                session_manager,
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
            let context_window = app.context_window;
            let cwd = app.cwd.clone();
            let session_name = app.session_name.clone();
            let context_approach = runtime
                .as_ref()
                .map(|rt| rt.export_state().policy.context_approach)
                .unwrap_or_default();
            let session_db_path = logger.db_path().to_string_lossy().to_string();
            push_system_message(
                app,
                info_text(
                    provider,
                    &model,
                    current_model_variant.as_deref(),
                    *current_execution_mode,
                    &context_approach,
                    context_window,
                    dynamic_tools.definitions().len(),
                    toolset_hash,
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
                dynamic_tools,
                runtime,
                history,
                last_turn,
                runtime_return_rx,
                cancel_token,
                active_stream_id,
                current_execution_mode,
                desired_dynamic,
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
                dynamic_tools,
                runtime,
                provider,
                current_model_variant,
                toolset_hash,
            )
            .await
        }
        command::Command::Tree => session::handle_tree(app, runtime),
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
                plugin_host,
                dynamic_tools,
                runtime,
                history,
                turn_counter,
                last_turn,
                provider,
                current_model_variant,
                current_execution_mode,
                session_manager,
                desired_dynamic,
                model_catalog,
                toolset_hash,
            )
            .await
        }
        command::Command::Tools(raw) => tools::handle_tools(
            raw,
            app,
            dynamic_tools,
            desired_dynamic,
            pending_reconfigure,
            *current_execution_mode,
        ),
        command::Command::Reconfigure(raw) => {
            tools::handle_reconfigure(
                raw,
                app,
                dynamic_tools,
                runtime,
                desired_dynamic,
                pending_reconfigure,
                toolset_hash,
            )
            .await
        }
        command::Command::Skills => session::handle_skills(app),
        command::Command::Plugin { name, argument } => {
            plugin::handle_plugin(name, argument, app, runtime, history, session_manager).await
        }
    }
}
