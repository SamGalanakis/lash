use super::runtime::{parse_kv_args, register_builtin_tool, send_user_message};
use super::*;
use crate::app::projected_blocks_from_state;
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
        Ok(effects) => apply_ui_host_effects(app, effects),
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
    provider: &mut Provider,
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
        );
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
    provider: &mut Provider,
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
            handle_ui_command(command, app, ui_extensions, plugin_host, session_manager).await;
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
    provider: &mut Provider,
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
        command::Command::Exit => return Ok(true),
        command::Command::Clear => {
            app.clear();
            app.set_model_variant(current_model_variant.clone());
            history.clear();
            *turn_counter = 0;
            *last_turn = None;
            app.token_usage = TokenUsage::default();
            *active_stream_id = active_stream_id.wrapping_add(1);
            if let Some(rt) = runtime.as_mut() {
                let _ = rt.reset_session().await;
                let mut state = SessionStateEnvelope {
                    session_id: "root".to_string(),
                    policy: SessionPolicy {
                        execution_mode: *current_execution_mode,
                        ..rt.export_state().policy
                    },
                    session_graph: lash::SessionGraph::default(),
                    iteration: *turn_counter,
                    token_usage: app.token_usage.clone(),
                    last_prompt_usage: None,
                    dynamic_state_ref: None,
                    dynamic_state_generation: None,
                    dynamic_state_snapshot: None,
                    plugin_snapshot_ref: None,
                    plugin_snapshot_revision: None,
                    plugin_snapshot: None,
                    execution_state_snapshot: None,
                    token_ledger: Vec::new(),
                    checkpoint_ref: None,
                    persisted_graph_node_count: 0,
                    graph_replace_required: false,
                };
                state.replace_projection(history, &[]);
                rt.set_state(state);
                match rt.session_manager() {
                    Ok(manager) => *session_manager = manager,
                    Err(err) => push_system_message(
                        app,
                        format!("Failed to refresh session manager: {}", err),
                    ),
                }
                let ui_extensions = app.ui_extensions_handle();
                sync_ui_extensions(
                    app,
                    ui_extensions.as_ref(),
                    plugin_host,
                    Arc::clone(session_manager),
                )
                .await;
                *pending_clear_after_return = false;
            } else {
                *pending_clear_after_return = true;
            }
        }
        command::Command::Version => {
            push_system_message(app, version_text());
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
                ),
            );
        }
        command::Command::Model(new_model) => {
            let Some(new_model) = new_model else {
                let mut lines = vec![
                    format!("Current model: `{}`", app.model),
                    format!("Provider: {}", provider.label()),
                ];
                lines.extend(variant_lines(
                    provider,
                    &app.model,
                    current_model_variant.as_deref(),
                ));
                if let Some(window) = app.context_window {
                    lines.push(format!("Context window: {}", window));
                }
                lines.push("Usage: `/model <name>`".to_string());
                lines.push("Use `/variant` to inspect or change the active variant.".to_string());
                push_system_message(app, lines.join("\n"));
                return Ok(false);
            };
            let selection = match parse_model_selection(&new_model) {
                Ok(s) => s,
                Err(e) => {
                    push_system_message(app, format!("Invalid model input: {}", e));
                    return Ok(false);
                }
            };
            if let Err(e) = validate_model_selection(provider, &selection) {
                push_system_message(app, format!("Model rejected: {}", e));
                return Ok(false);
            }
            let resolved_model_spec =
                match resolve_model_selection(provider, &selection, model_catalog) {
                    Ok(spec) => spec,
                    Err(err) => {
                        push_system_message(app, format!("Model rejected: {}", err));
                        return Ok(false);
                    }
                };
            let model_variant = provider
                .default_model_variant(&selection.model)
                .map(str::to_string);
            if let Some(rt) = runtime.as_mut() {
                rt.update_session_config(
                    None,
                    Some(selection.model.clone()),
                    Some(model_variant.clone()),
                    Some(resolved_model_spec.context_window() as usize),
                )
                .await;
            }
            *current_model_variant = model_variant;
            app.context_window = Some(resolved_model_spec.context_window());
            app.context_usage_excludes_cached_input = provider.input_usage_excludes_cached_tokens();
            app.model = selection.model.clone();
            app.set_model_variant(current_model_variant.clone());
            let mut msg = format!("Model set to `{}`", app.model);
            if let Some(variant) = current_model_variant.as_deref() {
                msg.push_str(&format!("\nVariant reset to `{}`", variant));
                msg.push_str("\nUse `/variant` to pick a different provider-native preset.");
            } else {
                msg.push_str("\nThis model does not expose configurable variants.");
            }
            if let Some(window) = app.context_window {
                msg.push_str(&format!("\nContext window: {}", window));
            }
            push_system_message(app, msg);
        }
        command::Command::Variant(new_variant) => {
            let Some(new_variant) = new_variant else {
                let mut lines = vec![
                    format!("Current model: `{}`", app.model),
                    format!("Provider: {}", provider.label()),
                ];
                lines.extend(variant_lines(
                    provider,
                    &app.model,
                    current_model_variant.as_deref(),
                ));
                push_system_message(app, lines.join("\n"));
                return Ok(false);
            };
            let variant =
                match resolve_model_variant(provider, &app.model, Some(new_variant.as_str())) {
                    Ok(variant) => variant,
                    Err(err) => {
                        push_system_message(app, format!("Variant rejected: {}", err));
                        return Ok(false);
                    }
                };
            if let Some(rt) = runtime.as_mut() {
                rt.update_session_config(None, None, Some(variant.clone()), None)
                    .await;
            }
            *current_model_variant = variant;
            app.set_model_variant(current_model_variant.clone());
            let mut lines = vec![format!("Model: `{}`", app.model)];
            if let Some(variant) = current_model_variant.as_deref() {
                lines.push(format!("Variant set to `{}`", variant));
            } else {
                lines.push("Variant reset to provider default `(none)`.".to_string());
            }
            lines.extend(variant_lines(
                provider,
                &app.model,
                current_model_variant.as_deref(),
            ));
            push_system_message(app, lines.join("\n"));
        }
        command::Command::Mode(new_mode) => {
            let Some(new_mode) = new_mode else {
                push_system_message(
                    app,
                    format!(
                        "Current execution mode: `{}`\nThis is locked for the current session.\nStart a new session to use a different mode.\nUsage: `/mode {}`",
                        execution_mode_label(*current_execution_mode),
                        execution_mode_usage()
                    ),
                );
                return Ok(false);
            };
            let new_mode =
                match parse_execution_mode(&new_mode).and_then(ensure_supported_execution_mode) {
                    Ok(mode) => mode,
                    Err(err) => {
                        push_system_message(app, format!("Invalid execution mode: {}", err));
                        return Ok(false);
                    }
                };
            if new_mode == *current_execution_mode {
                push_system_message(
                    app,
                    format!(
                        "Execution mode is already `{}`.\nThis is locked for the current session.",
                        execution_mode_label(*current_execution_mode)
                    ),
                );
            } else {
                push_system_message(
                    app,
                    format!(
                        "Execution mode is locked for the current session (`{}`).\nStart a new session with `--mode {}` to use `{}`.",
                        execution_mode_label(*current_execution_mode),
                        execution_mode_label(new_mode),
                        execution_mode_label(new_mode)
                    ),
                );
            }
        }
        command::Command::ChangeProvider => {
            paused.store(true, Ordering::Relaxed);
            let _ = crossterm::execute!(std::io::stdout(), crossterm::event::DisableFocusChange);
            let previous_kind = provider.kind();
            let previous_provider = provider.clone();
            let previous_model = app.model.clone();
            let previous_context_window = app.context_window;
            let previous_context_usage = app.context_usage_excludes_cached_input;
            let previous_variant = current_model_variant.clone();

            terminal.restore();
            let existing_cfg = LashConfig::load();
            let setup_result = setup::run_setup_with_existing(existing_cfg.as_ref()).await;
            *terminal = Terminal::enter()?;
            paused.store(false, Ordering::Relaxed);

            match setup_result {
                Ok(mut new_cfg) => {
                    if let Err(e) = new_cfg.active_provider_mut().ensure_fresh().await {
                        push_system_message(
                            app,
                            format!("Provider setup completed, but token refresh failed: {}", e),
                        );
                        *provider = previous_provider;
                        app.model = previous_model;
                        app.context_window = previous_context_window;
                        app.context_usage_excludes_cached_input = previous_context_usage;
                        *current_model_variant = previous_variant;
                        app.set_model_variant(current_model_variant.clone());
                        return Ok(false);
                    }
                    if let Err(e) = new_cfg.save() {
                        push_system_message(
                            app,
                            format!("Provider updated, but saving config failed: {}", e),
                        );
                    }
                    *provider = new_cfg.active_provider().clone();
                    if let Err(err) = model_catalog
                        .refresh_if_stale(lash::model_info::DEFAULT_REFRESH_INTERVAL)
                        .await
                    {
                        push_system_message(
                            app,
                            format!("Warning: failed to refresh models.dev catalog: {}", err),
                        );
                    }
                    let new_model = provider.default_model().to_string();
                    let selection = match parse_model_selection(&new_model) {
                        Ok(s) => s,
                        Err(e) => {
                            push_system_message(
                                app,
                                format!("Provider default model is invalid: {}", e),
                            );
                            *provider = previous_provider;
                            app.model = previous_model;
                            app.context_window = previous_context_window;
                            app.context_usage_excludes_cached_input = previous_context_usage;
                            *current_model_variant = previous_variant;
                            app.set_model_variant(current_model_variant.clone());
                            return Ok(false);
                        }
                    };
                    if let Err(e) = validate_model_selection(provider, &selection) {
                        push_system_message(
                            app,
                            format!("Provider default model failed validation: {}", e),
                        );
                        *provider = previous_provider;
                        app.model = previous_model;
                        app.context_window = previous_context_window;
                        app.context_usage_excludes_cached_input = previous_context_usage;
                        *current_model_variant = previous_variant;
                        app.set_model_variant(current_model_variant.clone());
                        return Ok(false);
                    }
                    let resolved_model_spec =
                        match resolve_model_selection(provider, &selection, model_catalog) {
                            Ok(spec) => spec,
                            Err(err) => {
                                push_system_message(
                                    app,
                                    format!("Provider default model failed validation: {}", err),
                                );
                                *provider = previous_provider;
                                app.model = previous_model;
                                app.context_window = previous_context_window;
                                app.context_usage_excludes_cached_input = previous_context_usage;
                                *current_model_variant = previous_variant;
                                app.set_model_variant(current_model_variant.clone());
                                return Ok(false);
                            }
                        };
                    let model_variant = provider
                        .default_model_variant(&selection.model)
                        .map(str::to_string);
                    if let Some(rt) = runtime.as_mut() {
                        rt.update_session_config(
                            Some(provider.clone()),
                            Some(selection.model.clone()),
                            Some(model_variant.clone()),
                            Some(resolved_model_spec.context_window() as usize),
                        )
                        .await;
                    }
                    *current_model_variant = model_variant;
                    app.context_window = Some(resolved_model_spec.context_window());
                    app.context_usage_excludes_cached_input =
                        provider.input_usage_excludes_cached_tokens();
                    app.model = selection.model.clone();
                    app.set_model_variant(current_model_variant.clone());
                    let saved_kinds = new_cfg
                        .provider_kinds()
                        .into_iter()
                        .map(ProviderKind::cli_label)
                        .collect::<Vec<_>>()
                        .join(", ");
                    push_system_message(
                        app,
                        format!(
                            "Provider {}: {}\nSaved providers: {}\nModel set to default: `{}`",
                            if provider.kind() == previous_kind {
                                "reauthenticated"
                            } else {
                                "switched"
                            },
                            provider.label(),
                            saved_kinds,
                            selection.model
                        ),
                    );
                }
                Err(e) => {
                    let msg = e.to_string();
                    if msg.contains("Setup cancelled") {
                        push_system_message(
                            app,
                            "Provider setup cancelled. Current provider unchanged.",
                        );
                    } else {
                        push_system_message(
                            app,
                            format!(
                                "Provider setup failed: {}. Current provider unchanged.",
                                msg
                            ),
                        );
                    }
                }
            }
        }
        command::Command::Logout => {
            let active_kind = provider.kind();
            match LashConfig::load() {
                Some(mut cfg) => {
                    if !cfg.has_provider(active_kind) {
                        push_system_message(app, "The active provider is not stored on disk.");
                        return Ok(false);
                    }
                    if cfg.provider_count() == 1 {
                        match LashConfig::clear() {
                            Ok(()) => push_system_message(
                                app,
                                format!(
                                    "Removed stored credentials for {}.\n\nThis running session may continue using in-memory credentials.\nUse `/provider` or `/login` to sign in again without restarting.",
                                    active_kind.cli_label()
                                ),
                            ),
                            Err(e) => push_system_message(
                                app,
                                format!("Failed to remove credentials: {}", e),
                            ),
                        }
                    } else {
                        cfg.remove_provider(active_kind);
                        let next_kind = cfg.active_provider_kind();
                        match cfg.save() {
                            Ok(()) => push_system_message(
                                app,
                                format!(
                                    "Removed stored credentials for {}.\nNew sessions will default to {}.\n\nThis running session may continue using in-memory credentials.",
                                    active_kind.cli_label(),
                                    next_kind.cli_label()
                                ),
                            ),
                            Err(e) => push_system_message(
                                app,
                                format!("Failed to save updated provider registry: {}", e),
                            ),
                        }
                    }
                }
                None => push_system_message(app, "No stored provider credentials found."),
            }
        }
        command::Command::Retry => {
            if let Some(previous) = last_turn.clone() {
                if let Err(e) = apply_pending_reconfigure(
                    dynamic_tools,
                    desired_dynamic,
                    pending_reconfigure,
                    runtime,
                )
                .await
                {
                    push_system_message(
                        app,
                        format!("Pending runtime reconfigure failed; retry blocked: {}", e),
                    );
                    return Ok(false);
                }
                *toolset_hash = hash12(
                    &serde_json::to_vec(&dynamic_tools.definitions())
                        .unwrap_or_else(|_| b"[]".to_vec()),
                );
                *current_execution_mode = previous.execution_mode;
                let current_dynamic_state = dynamic_tools.export_state();
                send_user_message(
                    previous.prepared_turn.clone(),
                    previous.turn_input.clone(),
                    app,
                    None,
                    logger,
                    runtime,
                    history,
                    runtime_return_rx,
                    cancel_token,
                    active_stream_id,
                    app_tx,
                    &current_dynamic_state,
                );
            } else {
                push_system_message(app, "No previous turn payload to retry yet.");
            }
        }
        command::Command::Controls => {
            push_system_message(app, controls_text(app.ui_extensions()));
        }
        command::Command::Fork => {
            let current_dynamic_state = dynamic_tools.export_state();
            match fork::fork_current_session(
                runtime.as_mut(),
                logger,
                &app.ui_resume_state(),
                provider,
                &app.model,
                app.context_window
                    .expect("app context_window must be set before forking"),
                current_model_variant.as_deref(),
                toolset_hash,
                &current_dynamic_state,
            )
            .await
            {
                Ok((child_filename, child_session_name)) => {
                    let exe = match std::env::current_exe() {
                        Ok(exe) => exe,
                        Err(err) => {
                            push_system_message(
                                app,
                                format!("Fork created but launcher lookup failed: {}", err),
                            );
                            return Ok(false);
                        }
                    };
                    let child_args = vec!["--resume".to_string(), child_filename.clone()];
                    match fork::spawn_in_new_terminal(&exe, &child_args) {
                        Ok(()) => push_system_message(
                            app,
                            format!("Forked into `{}` ({})", child_session_name, child_filename),
                        ),
                        Err(err) => push_system_message(
                            app,
                            format!("Fork created but launch failed: {}", err),
                        ),
                    }
                }
                Err(err) => push_system_message(app, format!("Fork failed: {}", err)),
            }
        }
        command::Command::Tree => {
            if app.has_prompt() || app.has_wait() {
                push_system_message(
                    app,
                    "Close the active prompt or wait state before opening /tree.",
                );
                return Ok(false);
            }
            let Some(rt) = runtime.as_ref() else {
                push_system_message(
                    app,
                    "Branch navigation is unavailable while a turn is running.",
                );
                return Ok(false);
            };
            let roots = crate::tree::current_message_tree(rt);
            if roots.is_empty() {
                push_system_message(app, "No messages yet.");
            } else {
                app.show_tree(roots);
            }
        }
        command::Command::Help => {
            let help = help_text(&app.skills, app.ui_extensions());
            push_system_message(app, help);
        }
        command::Command::Resume(name) => {
            if let Some(filename) = name {
                match resume::load_resumed_session(
                    &filename,
                    app,
                    history,
                    runtime,
                    turn_counter,
                    current_execution_mode,
                    provider,
                    current_model_variant,
                    dynamic_tools,
                    desired_dynamic,
                    model_catalog,
                )
                .await
                {
                    Ok(()) => {
                        *last_turn = None;
                        if let Some(rt) = runtime.as_ref() {
                            match rt.session_manager() {
                                Ok(manager) => *session_manager = manager,
                                Err(err) => push_system_message(
                                    app,
                                    format!("Failed to refresh session manager: {}", err),
                                ),
                            }
                        }
                        let ui_extensions = app.ui_extensions_handle();
                        sync_ui_extensions(
                            app,
                            ui_extensions.as_ref(),
                            plugin_host,
                            Arc::clone(session_manager),
                        )
                        .await;
                        *toolset_hash = hash12(
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
            } else {
                const SESSION_PICKER_LIMIT: usize = 50;
                let current_session_id = logger.session_id.clone();
                let mut sessions = task::spawn_blocking(move || {
                    let mut s = session_log::list_recent_sessions(SESSION_PICKER_LIMIT + 1);
                    s.retain(|si| si.session_id != current_session_id);
                    s
                })
                .await
                .unwrap_or_default();
                if sessions.is_empty() {
                    app.blocks.push(DisplayBlock::SystemMessage(
                        "No sessions found.".to_string(),
                    ));
                    app.invalidate_height_cache();
                    app.scroll_to_bottom();
                } else {
                    sessions.truncate(SESSION_PICKER_LIMIT);
                    app.show_session_picker(sessions);
                }
            }
        }
        command::Command::Tools(raw) => {
            let raw = raw.unwrap_or_default();
            let raw_trim = raw.trim();
            if raw_trim.is_empty() {
                let active = dynamic_tools.export_state();
                let mut lines = vec![
                    format!("Dynamic tools (generation {}):", active.base_generation),
                    format!(
                        "Pending reconfigure: {}",
                        if *pending_reconfigure { "yes" } else { "no" }
                    ),
                ];
                for (name, spec) in &desired_dynamic.tools {
                    let enabled = desired_dynamic.enabled_tools.contains(name);
                    lines.push(format!(
                        "  - {} [{}] adapter={}{}",
                        name,
                        spec.definition.returns,
                        spec.adapter_id,
                        if enabled { " (enabled)" } else { " (disabled)" }
                    ));
                }
                if desired_dynamic.tools.is_empty() {
                    lines.push("  (none)".to_string());
                }
                push_system_message(app, lines.join("\n"));
                return Ok(false);
            }

            let mut parts = raw_trim.split_whitespace();
            let sub = parts.next().unwrap_or_default();
            match sub {
                "add" => {
                    let mut add_parts = raw_trim.splitn(4, ' ');
                    let _ = add_parts.next();
                    let Some(name) = add_parts.next() else {
                        push_system_message(
                            app,
                            "Usage: /tools add <name> <handler> [description]",
                        );
                        return Ok(false);
                    };
                    let Some(handler_id) = add_parts.next() else {
                        push_system_message(
                            app,
                            "Usage: /tools add <name> <handler> [description]",
                        );
                        return Ok(false);
                    };
                    let description = add_parts.next().map(|v| v.trim().to_string());
                    match register_builtin_tool(
                        dynamic_tools,
                        name,
                        handler_id,
                        description,
                        *current_execution_mode,
                    ) {
                        Ok(def) => {
                            desired_dynamic.tools.insert(
                                name.to_string(),
                                DynamicToolSpec {
                                    definition: def,
                                    adapter_id: "inprocess".to_string(),
                                },
                            );
                            desired_dynamic.enabled_tools.insert(name.to_string());
                            *pending_reconfigure = true;
                            push_system_message(
                                app,
                                format!(
                                    "Tool `{}` staged with handler `{}`. Apply with `/reconfigure apply` or send the next turn.",
                                    name, handler_id
                                ),
                            );
                        }
                        Err(e) => push_system_message(app, e),
                    }
                }
                "rm" | "remove" => {
                    let Some(name) = parts.next() else {
                        push_system_message(app, "Usage: /tools rm <name>");
                        return Ok(false);
                    };
                    if desired_dynamic.tools.remove(name).is_some() {
                        desired_dynamic.enabled_tools.remove(name);
                        *pending_reconfigure = true;
                        push_system_message(app, format!("Tool `{name}` staged for removal."));
                    } else {
                        push_system_message(app, format!("Tool `{name}` not found."));
                    }
                }
                "update" => {
                    let mut update_parts = raw_trim.splitn(3, ' ');
                    let _ = update_parts.next();
                    let Some(name) = update_parts.next() else {
                        push_system_message(app, "Usage: /tools update <name> key=value ...");
                        return Ok(false);
                    };
                    let kv_raw = update_parts.next().unwrap_or_default();
                    let kv = parse_kv_args(kv_raw);
                    let Some(spec) = desired_dynamic.tools.get_mut(name) else {
                        push_system_message(app, format!("Tool `{name}` not found."));
                        return Ok(false);
                    };
                    if let Some(desc) = kv.get("description") {
                        spec.definition.description = desc.clone();
                    }
                    if let Some(returns) = kv.get("returns") {
                        spec.definition.returns = returns.clone();
                    }
                    if let Some(inject) = kv.get("injected") {
                        spec.definition.injected = inject == "true";
                    }
                    *pending_reconfigure = true;
                    push_system_message(app, format!("Tool `{name}` staged for update."));
                }
                "enable" => {
                    let Some(name) = parts.next() else {
                        push_system_message(app, "Usage: /tools enable <name>");
                        return Ok(false);
                    };
                    if desired_dynamic.tools.contains_key(name) {
                        desired_dynamic.enabled_tools.insert(name.to_string());
                        *pending_reconfigure = true;
                        push_system_message(app, format!("Tool `{name}` staged for enable."));
                    } else {
                        push_system_message(app, format!("Tool `{name}` not found."));
                    }
                }
                "disable" => {
                    let Some(name) = parts.next() else {
                        push_system_message(app, "Usage: /tools disable <name>");
                        return Ok(false);
                    };
                    if desired_dynamic.tools.contains_key(name) {
                        desired_dynamic.enabled_tools.remove(name);
                        *pending_reconfigure = true;
                        push_system_message(app, format!("Tool `{name}` staged for disable."));
                    } else {
                        push_system_message(app, format!("Tool `{name}` not found."));
                    }
                }
                _ => push_system_message(
                    app,
                    "Unknown /tools subcommand. Try: add, rm, update, enable, disable",
                ),
            }
        }
        command::Command::Reconfigure(raw) => {
            let action = raw.unwrap_or_else(|| "status".to_string());
            match action.trim() {
                "" | "status" => {
                    push_system_message(
                        app,
                        format!(
                            "Reconfigure status: pending={} current_generation={} base_generation={}",
                            pending_reconfigure,
                            dynamic_tools.generation(),
                            desired_dynamic.base_generation
                        ),
                    );
                }
                "clear" => {
                    *desired_dynamic = dynamic_tools.export_state();
                    *pending_reconfigure = false;
                    push_system_message(app, "Cleared pending dynamic runtime changes.");
                }
                "apply" => match apply_pending_reconfigure(
                    dynamic_tools,
                    desired_dynamic,
                    pending_reconfigure,
                    runtime,
                )
                .await
                {
                    Ok(generation) => {
                        *toolset_hash = hash12(
                            &serde_json::to_vec(&dynamic_tools.definitions())
                                .unwrap_or_else(|_| b"[]".to_vec()),
                        );
                        push_system_message(
                            app,
                            format!(
                                "Dynamic runtime reconfigured successfully (generation {}).",
                                generation
                            ),
                        )
                    }
                    Err(e) => push_system_message(app, format!("Reconfigure failed: {e}")),
                },
                _ => push_system_message(
                    app,
                    "Unknown /reconfigure action. Try: status, apply, clear",
                ),
            }
        }
        command::Command::Skills => {
            app.skills = SkillCatalog::load();
            let items: Vec<(String, String)> = app
                .skills
                .iter()
                .map(|s| (s.name.clone(), s.description.clone()))
                .collect();
            if items.is_empty() {
                app.blocks.push(DisplayBlock::SystemMessage(
                    "No skills found.\n\
                     Add skill directories to ~/.lash/skills/ or .agents/lash/skills/\n\
                     Each skill is a directory with a SKILL.md file."
                        .to_string(),
                ));
                app.invalidate_height_cache();
                app.scroll_to_bottom();
            } else {
                app.show_skill_picker(items);
            }
        }
        command::Command::Plugin { name, argument } => {
            // `/compact` is a built-in plugin command but its handler needs
            // to mutate the live runtime state, which the in-tree plugin
            // command surface cannot do today. Route it through
            // `LashRuntime::rewrite_history` directly so the rewritten
            // messages take effect immediately.
            if name == "/compact" {
                let Some(rt) = runtime.as_mut() else {
                    push_system_message(app, "Compaction is unavailable while a turn is running.");
                    return Ok(false);
                };
                let trigger = lash::RewriteTrigger::Manual {
                    instructions: argument,
                };
                match rt.rewrite_history(trigger).await {
                    Ok(true) => {
                        let state = rt.export_state();
                        history.clear();
                        let projected_messages = state.project_messages();
                        let projected_tool_calls = state.project_tool_calls();
                        history.extend(projected_messages.clone());
                        app.blocks = projected_blocks_from_state(
                            &projected_messages,
                            &projected_tool_calls,
                            &app.ui_resume_state(),
                        );
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
                return Ok(false);
            }
            let plugin_session = runtime.as_ref().and_then(|rt| rt.plugin_session());
            let Some(plugin_session) = plugin_session else {
                push_system_message(
                    app,
                    format!("Plugin command `{name}` is unavailable (no active session)."),
                );
                return Ok(false);
            };
            match plugin_session
                .invoke_command(&name, argument, Arc::clone(session_manager))
                .await
            {
                Ok(lash::CommandOutcome::Handled) => {}
                Ok(lash::CommandOutcome::Message(msg)) => push_system_message(app, msg),
                Ok(lash::CommandOutcome::Error(msg)) => {
                    push_system_message(app, format!("Plugin command `{name}` failed: {msg}"));
                }
                Err(err) => {
                    push_system_message(app, format!("Plugin command `{name}` error: {err}"));
                }
            }
        }
    }

    Ok(false)
}
