use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crossterm::event::{Event as TermEvent, KeyCode, KeyEventKind, KeyModifiers};
use lash::provider::{LashConfig, ProviderKind};
use lash::session_model::{Message, MessageRole, Part, PartKind, PruneState};
use lash::*;
use lash_ui::{
    KeyChord as UiKeyChord, KeyCode as UiKeyCode, KeyModifiers as UiKeyModifiers,
    UiCommandInvocation, UiContext, UiExtensions,
};
use ratatui::DefaultTerminal;
use tokio::sync::mpsc;
use tokio::task;
use tokio_util::sync::CancellationToken;

use crate::app::{self, App, DisplayBlock, PreparedTurn};
use crate::command;
use crate::event::AppEvent;
use crate::fork;
use crate::input_items::{build_items_from_editor_input, insert_inline_marker};
use crate::resume;
use crate::session_log::{self, SessionLogger};
use crate::update;
use crate::{Args, setup, ui};
use crate::{
    apply_ui_host_effects, cleanup_terminal, configure_terminal_ui, controls_text,
    ensure_supported_execution_mode, execution_mode_label, execution_mode_usage, hash12, help_text,
    info_text, latest_user_prompt_hash, normalize_prepared_turn_for_dispatch, parse_execution_mode,
    parse_model_selection, persist_live_runtime_snapshot, persist_root_session_state,
    push_system_message, resolve_model_selection, resolve_model_variant, shell_escape_command,
    sync_ui_extensions, turn_has_visible_output, validate_model_selection, variant_lines,
    version_text,
};

/// Returned by the spawned runtime task so we can reclaim ownership.
struct RuntimeRunResult {
    stream_id: u64,
    runtime: LashRuntime,
    result: AssembledTurn,
}

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

#[async_trait::async_trait]
impl EventSink for AppEventSink {
    async fn emit(&self, event: SessionEvent) {
        let _ = self.tx.send(AppEvent::Session {
            stream_id: self.stream_id,
            event,
        });
    }
}

#[derive(Clone)]
enum ParsedSlashCommand {
    Builtin(command::Command),
    Ui(UiCommandInvocation),
}

fn parse_slash_command(
    input: &str,
    skills: &SkillCatalog,
    ui_extensions: &UiExtensions,
) -> Option<ParsedSlashCommand> {
    command::parse(input, skills)
        .map(ParsedSlashCommand::Builtin)
        .or_else(|| {
            ui_extensions
                .parse_command(input)
                .map(ParsedSlashCommand::Ui)
        })
}

fn slash_command_runs_out_of_band_while_running(cmd: &ParsedSlashCommand) -> bool {
    match cmd {
        ParsedSlashCommand::Builtin(command) => command::runs_out_of_band_while_running(command),
        ParsedSlashCommand::Ui(command) => command.allow_while_running(),
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

#[allow(clippy::too_many_arguments)]
async fn handle_parsed_slash_command(
    command: ParsedSlashCommand,
    terminal: &mut DefaultTerminal,
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
    current_context_strategy: &mut ContextStrategy,
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
                current_context_strategy,
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
    terminal: &mut DefaultTerminal,
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
    current_context_strategy: &mut ContextStrategy,
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
                rt.set_state(SessionStateEnvelope {
                    session_id: "root".to_string(),
                    policy: SessionPolicy {
                        execution_mode: *current_execution_mode,
                        context_strategy: *current_context_strategy,
                        ..rt.export_state().policy
                    },
                    messages: history.clone(),
                    tool_calls: Vec::new(),
                    iteration: *turn_counter,
                    token_usage: app.token_usage.clone(),
                    last_prompt_usage: None,
                    task_state: None,
                    replay_manifest: None,
                    plugin_snapshot: None,
                    repl_snapshot: None,
                });
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
                // Runtime is still being reclaimed from a just-finished turn; clear state as soon
                // as it returns.
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
            push_system_message(
                app,
                info_text(
                    provider,
                    &model,
                    current_model_variant.as_deref(),
                    *current_execution_mode,
                    context_window,
                    dynamic_tools.definitions().len(),
                    toolset_hash,
                    *current_context_strategy,
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
                    None,
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
                rt.update_session_config(None, None, Some(variant.clone()), None, None)
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

            cleanup_terminal();
            let existing_cfg = LashConfig::load();
            let setup_result = setup::run_setup_with_existing(existing_cfg.as_ref()).await;
            *terminal = ratatui::init();
            configure_terminal_ui()?;
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
                            None,
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
                    logger,
                    runtime,
                    history,
                    runtime_return_rx,
                    cancel_token,
                    active_stream_id,
                    app_tx,
                    provider,
                    &current_dynamic_state,
                    toolset_hash,
                );
            } else {
                push_system_message(app, "No previous turn payload to retry yet.");
            }
        }
        command::Command::Controls => {
            push_system_message(app, controls_text(app.ui_extensions()));
        }
        command::Command::Fork(prompt) => {
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
                    let mut child_args = vec!["--resume".to_string(), child_filename.clone()];
                    if let Some(prompt) = prompt
                        .as_deref()
                        .map(str::trim)
                        .filter(|prompt| !prompt.is_empty())
                    {
                        child_args.push("--resume-prompt".to_string());
                        child_args.push(prompt.to_string());
                    }
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
                    current_context_strategy,
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
    }

    Ok(false)
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_app(
    mut terminal: DefaultTerminal,
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
                let turn_input = make_turn_input(&mut app, items, image_blobs);
                let current_dynamic_state = dynamic_tools.export_state();
                send_user_message(
                    prepared.clone(),
                    turn_input.clone(),
                    &mut app,
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
        // Check if runtime turn completed — reclaim runtime + updated history
        if let Some(ref mut rx) = runtime_return_rx {
            match rx.try_recv() {
                Ok(done) => {
                    runtime = Some(done.runtime);
                    if done.stream_id != active_stream_id || pending_clear_after_return {
                        if let Some(rt) = runtime.as_mut() {
                            let _ = rt.reset_session().await;
                            rt.set_state(SessionStateEnvelope::default());
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
                        continue;
                    }
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

                    // Snapshot REPL after each completed turn so resume can restore exact state.
                    let snapshot_hash = if let Some(rt) = runtime.as_mut() {
                        if matches!(state.policy.execution_mode, ExecutionMode::Repl) {
                            match rt.snapshot_repl().await {
                                Ok(blob) => {
                                    let snapshot_hash = hash12(&blob);
                                    state.repl_snapshot = Some(blob);
                                    Some(snapshot_hash)
                                }
                                Err(e) => {
                                    push_system_message(
                                        &mut app,
                                        format!(
                                            "Warning: failed to snapshot REPL state for resume: {}",
                                            e
                                        ),
                                    );
                                    None
                                }
                            }
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    state.task_state = None;

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
                    let final_output = app::latest_assistant_text_from_messages(&state.messages)
                        .or_else(|| {
                            (!done.result.assistant_output.safe_text.is_empty())
                                .then(|| done.result.assistant_output.safe_text.clone())
                        });
                    let ui_resume_state =
                        app.finish_turn_for_resume_with_output(final_output.as_deref());

                    let persisted_execution_mode = state.policy.execution_mode;
                    let persisted_context_strategy = state.policy.context_strategy;
                    let persisted_dynamic_state = dynamic_tools.export_state();
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
                    let leftover_injections =
                        turn_injection_bridge.drain().unwrap_or_else(|_| Vec::new());
                    if !leftover_injections.is_empty() {
                        while let Some(turn) = app.pending_steers.pop_front() {
                            app.queue_turn(turn);
                        }
                    }

                    if let Some((queued, was_pending)) = app.take_next_queued_turn() {
                        let queued = normalize_prepared_turn_for_dispatch(queued, &app.skills);
                        if let Some(cmd) = parse_slash_command(
                            &queued.display_text,
                            &app.skills,
                            ui_extensions.as_ref(),
                        ) {
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
                                    "Pending runtime reconfigure failed; queued message not sent: {}",
                                    e
                                ),
                            );
                            app.requeue_front(queued, was_pending);
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
                        let turn_input = make_turn_input(&mut app, items, image_blobs);
                        let current_dynamic_state = dynamic_tools.export_state();
                        send_user_message(
                            queued.clone(),
                            turn_input.clone(),
                            &mut app,
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
                }
                Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                    app.stop_turn();
                    runtime_return_rx = None;
                    cancel_token = None;
                }
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {}
            }
        }

        // Draw only when dirty
        if app.dirty {
            // Pre-compute height cache before immutable borrow in draw
            let size = terminal.size()?;
            let vh = ui::history_viewport_height(&app, size.width, size.height);
            let vw = size.width as usize;
            app.ensure_height_cache_pub(vw, vh);
            app.refresh_follow_output_anchor(vw, vh);
            // Clamp scroll_offset (especially for scroll_to_bottom's usize::MAX)
            let total = app.total_content_height(vw, vh);
            let max_scroll = total.saturating_sub(vh);
            app.scroll_offset = app.scroll_offset.min(max_scroll);

            app.history_area = ui::history_area(&app, size.width, size.height);
            terminal.draw(|frame| ui::draw(frame, &app))?;
            app.dirty = false;
        }

        // Wait for next event
        let event = match app_rx.recv().await {
            Some(e) => e,
            None => break,
        };

        match event {
            AppEvent::Terminal(TermEvent::Paste(text)) => {
                app.dirty = true;

                if app.has_prompt() && app.is_prompt_text_entry() {
                    app.prompt_insert_text(&text);
                    continue;
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
            AppEvent::Terminal(TermEvent::Key(key)) => {
                // With kitty keyboard protocol, ignore Release/Repeat events
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                app.dirty = true;
                // Clear any active text selection on keypress
                if app.selection.visible {
                    app.clear_selection();
                }
                // CTRL+C: dismiss prompt if active, else quit
                if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
                    if app.has_prompt() {
                        app.dismiss_prompt();
                        continue;
                    }
                    break;
                }

                // ALT+O: reliable full expand toggle across most terminals.
                if key.modifiers.contains(KeyModifiers::ALT)
                    && matches!(key.code, KeyCode::Char(c) if c.eq_ignore_ascii_case(&'o'))
                {
                    app.toggle_full_expand();
                    continue;
                }

                if key.modifiers.contains(KeyModifiers::ALT) && key.code == KeyCode::Up {
                    if let Some((turn, _was_pending)) = app.take_last_queued_turn() {
                        app.restore_prepared_turn(turn);
                        app.update_suggestions();
                    }
                    continue;
                }

                // CTRL+O: cycle expand (0↔1)
                if key.modifiers.contains(KeyModifiers::CONTROL)
                    && matches!(key.code, KeyCode::Char(c) if c.eq_ignore_ascii_case(&'o'))
                {
                    app.cycle_expand();
                    continue;
                }

                // CTRL+Y: copy last response to clipboard
                if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('y') {
                    copy_last_response(&app);
                    continue;
                }

                // CTRL+SHIFT+V: always paste text from clipboard
                if key.modifiers.contains(KeyModifiers::CONTROL)
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
                if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('v') {
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
                    if app.has_prompt() {
                        app.dismiss_prompt();
                    } else if app.has_skill_picker() {
                        app.dismiss_skill_picker();
                    } else if app.has_session_picker() {
                        app.dismiss_session_picker();
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
                    let size = terminal.size()?;
                    let vh = ui::history_viewport_height(&app, size.width, size.height);
                    let vw = size.width as usize;
                    let half_page = vh / 2;

                    // Ctrl+U / Ctrl+D: half-page scroll
                    if key.modifiers.contains(KeyModifiers::CONTROL)
                        && key.code == KeyCode::Char('u')
                    {
                        app.scroll_up(half_page);
                        app.dirty = true;
                        continue;
                    }
                    if key.modifiers.contains(KeyModifiers::CONTROL)
                        && key.code == KeyCode::Char('d')
                    {
                        app.scroll_down(half_page, vh, vw);
                        app.dirty = true;
                        continue;
                    }

                    // PgUp / PgDn
                    if key.code == KeyCode::PageUp {
                        app.scroll_up(vh);
                        app.dirty = true;
                        continue;
                    }
                    if key.code == KeyCode::PageDown {
                        app.scroll_down(vh, vh, vw);
                        app.dirty = true;
                        continue;
                    }
                }

                // ── Skill picker key handling ──
                if app.has_skill_picker() {
                    match key.code {
                        KeyCode::Up | KeyCode::Char('k') => app.skill_picker_up(),
                        KeyCode::Down | KeyCode::Char('j') => app.skill_picker_down(),
                        KeyCode::Enter => {
                            if let Some(name) = app.take_skill_pick() {
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
                        KeyCode::Up | KeyCode::Char('k') => app.session_picker_up(),
                        KeyCode::Down | KeyCode::Char('j') => app.session_picker_down(),
                        KeyCode::Enter => {
                            if let Some(filename) = app.take_session_pick() {
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
                            app.prompt_toggle_note_focus();
                        }
                        KeyCode::Up if !editing_text => app.prompt_up(),
                        KeyCode::Down if !editing_text => app.prompt_down(),
                        KeyCode::Char(' ') if app.is_prompt_multi_select() && !editing_text => {
                            app.prompt_toggle_current_option();
                        }
                        KeyCode::BackTab if editing_text => app.prompt_insert_char('\n'),
                        KeyCode::Enter => {
                            let _ = app.take_prompt_response();
                        }
                        KeyCode::Char(c) if editing_text => {
                            app.prompt_insert_char(c);
                        }
                        KeyCode::Backspace if editing_text => {
                            app.prompt_backspace();
                        }
                        _ => {}
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
                        app.complete_suggestion();
                        app.update_suggestions();
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
                            app.queue_turn(queued.clone());
                            app.preview_queued_turn(&queued, false);
                            continue;
                        }
                        if runtime.is_none() {
                            push_system_message(
                                &mut app,
                                "Runtime is still finalizing the previous turn. Please retry in a moment.",
                            );
                            app.restore_prepared_turn(queued);
                            continue;
                        }

                        let (items, image_blobs) = build_items_from_editor_input(
                            &queued.effective_text,
                            queued.images.clone(),
                        );
                        let turn_input = make_turn_input(&mut app, items, image_blobs);
                        let current_dynamic_state = dynamic_tools.export_state();
                        send_user_message(
                            queued.clone(),
                            turn_input.clone(),
                            &mut app,
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
                        app.suggestion_up();
                    }
                    KeyCode::Down if app.has_suggestions() => {
                        app.suggestion_down();
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
                                app.queue_turn(queued.clone());
                                app.preview_queued_turn(&queued, false);
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
                                    app.queue_pending_steer(queued.clone());
                                    app.preview_queued_turn(&queued, true);
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
                            push_system_message(
                                &mut app,
                                "Runtime is still finalizing the previous turn. Please retry in a moment.",
                            );
                            app.restore_prepared_turn(queued);
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
                        let turn_input = make_turn_input(&mut app, items, image_blobs);
                        let current_dynamic_state = dynamic_tools.export_state();
                        send_user_message(
                            queued.clone(),
                            turn_input.clone(),
                            &mut app,
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
                        app.backspace();
                        app.update_suggestions();
                    }
                    KeyCode::Delete => {
                        app.delete();
                        app.update_suggestions();
                    }
                    KeyCode::Left => app.move_cursor_left(),
                    KeyCode::Right => app.move_cursor_right(),
                    KeyCode::Home => app.move_cursor_home(),
                    KeyCode::End => app.move_cursor_end(),
                    KeyCode::Up => app.history_up(),
                    KeyCode::Down => app.history_down(),
                    KeyCode::Char(c) => {
                        app.insert_char(c);
                        app.update_suggestions();
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

                match mouse.kind {
                    MouseEventKind::Down(MouseButton::Left) if in_history => {
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
                            app.scroll_up(scroll_lines);
                        } else if mouse.row >= ha.y + ha.height {
                            let size = terminal.size()?;
                            let vh = ui::history_viewport_height(&app, size.width, size.height);
                            let vw = size.width as usize;
                            app.scroll_down(scroll_lines, vh, vw);
                        }

                        let clamped_row = mouse.row.clamp(ha.y, ha.y + ha.height.saturating_sub(1));
                        let vrow = app.scroll_offset + (clamped_row - ha.y) as usize;
                        app.selection.end = (col, vrow);
                        app.selection.visible = true;
                        app.dirty = true;
                    }
                    MouseEventKind::Up(MouseButton::Left) if app.selection.active => {
                        app.selection.active = false;
                        if app.selection.visible {
                            // Force a draw so the buffer reflects current content, then extract
                            let size = terminal.size()?;
                            app.history_area = ui::history_area(&app, size.width, size.height);
                            let completed = terminal.draw(|frame| ui::draw(frame, &app))?;
                            let text = ui::extract_selected_text(
                                completed.buffer,
                                &app.selection,
                                app.history_area,
                                app.scroll_offset,
                            );
                            if !text.is_empty()
                                && let Ok(mut clipboard) = arboard::Clipboard::new()
                            {
                                let _ = clipboard.set_text(text);
                            }
                        }
                    }
                    MouseEventKind::ScrollUp => {
                        // Scroll extends selection if actively dragging, otherwise just scrolls
                        let scroll_lines = 3;
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
                        let size = terminal.size()?;
                        let vh = ui::history_viewport_height(&app, size.width, size.height);
                        let vw = size.width as usize;
                        let scroll_lines = 3;
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
            }
            AppEvent::Session { stream_id, event } => {
                if stream_id != active_stream_id {
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
                    let focus = if request.is_freeform() {
                        crate::overlay::PromptFocus::Text
                    } else {
                        crate::overlay::PromptFocus::Options
                    };
                    app.show_prompt(app::PromptState {
                        request,
                        focus,
                        cursor: 0,
                        selected: Default::default(),
                        reply_text: String::new(),
                        reply_cursor: 0,
                        response_tx,
                    });
                } else {
                    let ui_effects = ui_extensions.effects_for_session_event(&event);
                    let is_done = matches!(&event, SessionEvent::Done);
                    app.handle_session_event(event);
                    apply_ui_host_effects(&mut app, ui_effects);
                    if is_done && !app.focused {
                        notify_done();
                    }
                }
            }
            AppEvent::Quit => break,
        }
    }

    // Signal reader thread and tick timer to stop
    stop.store(true, Ordering::Relaxed);

    // Save input history
    app.save_history();

    Ok(())
}

fn make_turn_input(
    _app: &mut App,
    items: Vec<InputItem>,
    image_blobs: HashMap<String, Vec<u8>>,
) -> TurnInput {
    TurnInput {
        items,
        image_blobs,
        mode: Some(RunMode::Normal),
    }
}

fn append_turn_input_message(messages: &mut Vec<Message>, turn_input: &TurnInput) {
    let user_id = format!("m{}", messages.len());
    let mut image_ids = Vec::new();
    let mut user_parts = Vec::new();

    for item in &turn_input.items {
        match item {
            InputItem::Text { text } => {
                if text.is_empty() {
                    continue;
                }
                user_parts.push(Part {
                    id: format!("{}.p{}", user_id, user_parts.len()),
                    kind: PartKind::Text,
                    content: text.clone(),
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: PruneState::Intact,
                });
            }
            InputItem::FileRef { path } => {
                user_parts.push(Part {
                    id: format!("{}.p{}", user_id, user_parts.len()),
                    kind: PartKind::Text,
                    content: format!("[file: {path}]"),
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: PruneState::Intact,
                });
            }
            InputItem::DirRef { path } => {
                user_parts.push(Part {
                    id: format!("{}.p{}", user_id, user_parts.len()),
                    kind: PartKind::Text,
                    content: format!("[directory: {}]", path.trim_end_matches('/')),
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: PruneState::Intact,
                });
            }
            InputItem::ImageRef { id } => {
                let Some(bytes) = turn_input.image_blobs.get(id) else {
                    continue;
                };
                if image_ids.iter().any(|candidate| candidate == id) {
                    continue;
                }
                image_ids.push(id.clone());
                user_parts.push(Part {
                    id: format!("{}.p{}", user_id, user_parts.len()),
                    kind: PartKind::Image,
                    content: String::new(),
                    attachment: Some(lash::session_model::message::PartAttachment {
                        mime: "image/png".to_string(),
                        url: lash::session_model::message::data_url_for_bytes("image/png", bytes),
                        filename: None,
                    }),
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: PruneState::Intact,
                });
            }
        }
    }

    if user_parts.is_empty() {
        user_parts.push(Part {
            id: format!("{user_id}.p0"),
            kind: PartKind::Text,
            content: String::new(),
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            prune_state: PruneState::Intact,
        });
    }

    messages.push(Message {
        id: user_id,
        role: MessageRole::User,
        parts: user_parts,
        origin: None,
    });
}

fn pending_turn_snapshot(
    state: &SessionStateEnvelope,
    turn_input: &TurnInput,
) -> DurableTurnSnapshot {
    let mut messages = state.messages.clone();
    append_turn_input_message(&mut messages, turn_input);
    DurableTurnSnapshot {
        messages,
        tool_calls: state.tool_calls.clone(),
        iteration: state.iteration,
    }
}

pub(crate) fn make_injected_plugin_message(turn: &PreparedTurn) -> PluginMessage {
    let (items, image_blobs) =
        build_items_from_editor_input(&turn.effective_text, turn.images.clone());
    let mut parts = Vec::new();
    let mut image_ids = Vec::new();
    for item in items {
        match item {
            InputItem::Text { text } => {
                if text.is_empty() {
                    continue;
                }
                parts.push(Part {
                    id: String::new(),
                    kind: PartKind::Text,
                    content: text,
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: PruneState::Intact,
                });
            }
            InputItem::FileRef { path } => {
                parts.push(Part {
                    id: String::new(),
                    kind: PartKind::Text,
                    content: format!("[file: {path}]"),
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: PruneState::Intact,
                });
            }
            InputItem::DirRef { path } => {
                parts.push(Part {
                    id: String::new(),
                    kind: PartKind::Text,
                    content: format!("[directory: {}]", path.trim_end_matches('/')),
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: PruneState::Intact,
                });
            }
            InputItem::ImageRef { id } => {
                let Some(bytes) = image_blobs.get(&id) else {
                    continue;
                };
                if image_ids.iter().any(|candidate| candidate == &id) {
                    continue;
                }
                image_ids.push(id.clone());
                parts.push(Part {
                    id: String::new(),
                    kind: PartKind::Image,
                    content: String::new(),
                    attachment: Some(lash::session_model::message::PartAttachment {
                        mime: "image/png".to_string(),
                        url: lash::session_model::message::data_url_for_bytes("image/png", bytes),
                        filename: None,
                    }),
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: PruneState::Intact,
                });
            }
        }
    }

    PluginMessage {
        role: MessageRole::User,
        content: turn.effective_text.clone(),
        parts,
        images: Vec::new(),
    }
}

#[cfg(test)]
pub(crate) fn injected_image_part_indices(message: &PluginMessage) -> Vec<usize> {
    message
        .parts
        .iter()
        .enumerate()
        .filter_map(|(idx, part)| matches!(part.kind, PartKind::Image).then_some(idx))
        .collect()
}

fn parse_kv_args(raw: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for token in raw.split_whitespace() {
        if let Some((k, v)) = token.split_once('=') {
            out.insert(k.trim().to_string(), v.trim().to_string());
        }
    }
    out
}

fn register_builtin_tool(
    dynamic_tools: &Arc<DynamicToolProvider>,
    tool_name: &str,
    handler_id: &str,
    description_override: Option<String>,
    _execution_mode: ExecutionMode,
) -> Result<ToolDefinition, String> {
    let adapter = dynamic_tools.inprocess_adapter();
    let def = match handler_id {
        "echo" => {
            let handler: InProcessToolHandler = Arc::new(|args, _context, _progress| {
                Box::pin(async move {
                    let text = args
                        .get("text")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string();
                    ToolResult::ok(serde_json::json!(text))
                })
            });
            let def = ToolDefinition {
                name: tool_name.to_string(),
                description: description_override
                    .unwrap_or_else(|| "Echoes back the `text` argument.".to_string()),
                params: vec![ToolParam::typed("text", "str")],
                returns: "str".to_string(),
                examples: vec![format!("{tool_name}(text=\"hello\")")],
                enabled: true,
                injected: false,
                input_schema_override: None,
                output_schema_override: None,
            };
            adapter.register_tool(def.clone(), handler);
            def
        }
        "time" => {
            let handler: InProcessToolHandler = Arc::new(|_args, _context, _progress| {
                Box::pin(async move {
                    ToolResult::ok(serde_json::json!(chrono::Utc::now().to_rfc3339()))
                })
            });
            let def = ToolDefinition {
                name: tool_name.to_string(),
                description: description_override
                    .unwrap_or_else(|| "Returns the current UTC timestamp (RFC3339).".to_string()),
                params: vec![],
                returns: "str".to_string(),
                examples: vec![format!("{tool_name}()")],
                enabled: true,
                injected: false,
                input_schema_override: None,
                output_schema_override: None,
            };
            adapter.register_tool(def.clone(), handler);
            def
        }
        "uuid" => {
            let handler: InProcessToolHandler = Arc::new(|_args, _context, _progress| {
                Box::pin(async move {
                    ToolResult::ok(serde_json::json!(uuid::Uuid::new_v4().to_string()))
                })
            });
            let def = ToolDefinition {
                name: tool_name.to_string(),
                description: description_override
                    .unwrap_or_else(|| "Returns a random UUIDv4 string.".to_string()),
                params: vec![],
                returns: "str".to_string(),
                examples: vec![format!("{tool_name}()")],
                enabled: true,
                injected: false,
                input_schema_override: None,
                output_schema_override: None,
            };
            adapter.register_tool(def.clone(), handler);
            def
        }
        other => {
            return Err(format!(
                "Unknown handler `{other}`. Supported handlers: echo, time, uuid"
            ));
        }
    };

    Ok(def)
}

async fn apply_pending_reconfigure(
    dynamic_tools: &Arc<DynamicToolProvider>,
    desired_dynamic: &mut DynamicStateSnapshot,
    pending_reconfigure: &mut bool,
    runtime: &mut Option<LashRuntime>,
) -> Result<u64, String> {
    if !*pending_reconfigure {
        return Ok(dynamic_tools.generation());
    }

    let previous = dynamic_tools.export_state();
    let generation = match dynamic_tools.apply_state(desired_dynamic.clone()) {
        Ok(g) => g,
        Err(e) => {
            desired_dynamic.base_generation = dynamic_tools.generation();
            return Err(e.to_string());
        }
    };

    if let Some(rt) = runtime.as_mut()
        && let Err(e) = rt.refresh_session_execution_surface().await
    {
        let mut rollback = previous.clone();
        rollback.base_generation = dynamic_tools.generation();
        let _ = dynamic_tools.apply_state(rollback);
        let _ = rt.refresh_session_execution_surface().await;
        desired_dynamic.base_generation = dynamic_tools.generation();
        return Err(format!(
            "Failed to apply runtime reconfigure (state rolled back): {e}"
        ));
    }

    *desired_dynamic = dynamic_tools.export_state();
    *pending_reconfigure = false;
    Ok(generation)
}

/// Send a user message to the runtime: push display block and spawn turn run.
#[allow(clippy::too_many_arguments)]
fn send_user_message(
    prepared_turn: PreparedTurn,
    turn_input: TurnInput,
    app: &mut App,
    logger: &mut SessionLogger,
    runtime: &mut Option<LashRuntime>,
    _history: &mut Vec<Message>,
    runtime_return_rx: &mut Option<tokio::sync::oneshot::Receiver<RuntimeRunResult>>,
    cancel_token: &mut Option<CancellationToken>,
    active_stream_id: &mut u64,
    app_tx: &mpsc::UnboundedSender<AppEvent>,
    provider: &Provider,
    dynamic_state: &DynamicStateSnapshot,
    toolset_hash: &str,
) {
    let already_visible = if !prepared_turn.display_text.is_empty() {
        app.commit_pending_user_preview(&prepared_turn.display_text)
    } else {
        false
    };
    if !prepared_turn.display_text.is_empty() && !already_visible {
        app.push_prepared_user_input(&prepared_turn);
    }
    app.start_turn();
    app.resume_contextual_follow_output();
    app.keep_latest_user_block_visible();

    let mut rt = runtime
        .take()
        .expect("runtime should be available when not running");
    let persisted_state = rt.export_state();
    persist_live_runtime_snapshot(
        logger.store().as_ref(),
        pending_turn_snapshot(&persisted_state, &turn_input),
        &app.ui_resume_state(),
        dynamic_state,
        provider,
        &app.model,
        app.context_window
            .expect("app context_window must be set before dispatching a turn"),
        persisted_state.policy.execution_mode,
        persisted_state.policy.context_strategy,
        app.model_variant.as_deref(),
        toolset_hash,
        persisted_state.token_usage.clone(),
        persisted_state.last_prompt_usage.clone(),
    );
    tracing::info!(
        mode = ?turn_input.mode,
        items = turn_input.items.len(),
        images = turn_input.image_blobs.len(),
        "dispatching runtime turn"
    );
    let (return_tx, return_rx) = tokio::sync::oneshot::channel();
    *runtime_return_rx = Some(return_rx);

    let cancel = CancellationToken::new();
    *cancel_token = Some(cancel.clone());
    *active_stream_id = active_stream_id.wrapping_add(1);
    let stream_id = *active_stream_id;

    let sink_tx = app_tx.clone();
    tokio::spawn(async move {
        let sink = AppEventSink {
            tx: sink_tx,
            stream_id,
        };
        let result = match rt.stream_turn(turn_input, &sink, cancel).await {
            Ok(turn) => turn,
            Err(e) => AssembledTurn {
                state: rt.export_state(),
                status: TurnStatus::Failed,
                assistant_output: AssistantOutput {
                    safe_text: String::new(),
                    raw_text: String::new(),
                    state: OutputState::EmptyOutput,
                },
                has_plugin_visible_output: false,
                done_reason: DoneReason::RuntimeError,
                execution: ExecutionSummary {
                    mode: rt.export_state().policy.execution_mode,
                    had_tool_calls: false,
                    had_code_execution: false,
                },
                token_usage: TokenUsage::default(),
                tool_calls: Vec::new(),
                code_outputs: Vec::new(),
                errors: vec![TurnIssue {
                    kind: "runtime".to_string(),
                    code: Some(e.code),
                    message: e.message,
                }],
            },
        };
        let _ = return_tx.send(RuntimeRunResult {
            stream_id,
            runtime: rt,
            result,
        });
    });
}

/// Send a desktop notification that the session finished.
fn notify_done() {
    // Ensure the icon exists in $LASH_HOME
    let icon_path = lash::lash_home().join("icon.svg");
    if !icon_path.exists() {
        let _ = std::fs::write(&icon_path, include_bytes!("../assets/icon.svg"));
    }
    let _ = std::process::Command::new("notify-send")
        .args(["-a", "lash", "-i"])
        .arg(&icon_path)
        .args(["lash", "Response complete"])
        .spawn();
}

/// Generate a unique session name like "juniper-mountain".
/// Scans existing session files for collisions.
pub(crate) fn generate_session_name(sessions_dir: &std::path::Path) -> String {
    use rand::Rng;

    const ADJECTIVES: &[&str] = &[
        "alpine", "amber", "ancient", "ashen", "autumn", "blazing", "bright", "calm", "cedar",
        "coastal", "copper", "coral", "crimson", "crystal", "dappled", "deep", "desert", "distant",
        "dusky", "ember", "fading", "fern", "flint", "foggy", "forest", "frozen", "gentle",
        "gilded", "glacial", "golden", "granite", "hollow", "iron", "ivory", "jade", "keen",
        "lofty", "lunar", "marble", "misty", "mossy", "northern", "obsidian", "onyx", "opal",
        "pale", "pine", "quiet", "radiant", "rugged", "rustic", "sandy", "silver", "silent",
        "solar", "stone", "sunlit", "tidal", "twilight", "verdant", "violet", "wild", "winter",
    ];
    const NOUNS: &[&str] = &[
        "basin",
        "birch",
        "bluff",
        "boulder",
        "brook",
        "canyon",
        "cavern",
        "cliff",
        "cove",
        "creek",
        "delta",
        "dune",
        "falls",
        "field",
        "fjord",
        "glade",
        "gorge",
        "grove",
        "harbor",
        "heath",
        "hill",
        "island",
        "lake",
        "ledge",
        "marsh",
        "meadow",
        "mesa",
        "mountain",
        "oasis",
        "ocean",
        "pass",
        "peak",
        "plain",
        "plateau",
        "pond",
        "prairie",
        "ravine",
        "reef",
        "ridge",
        "river",
        "shore",
        "slope",
        "spring",
        "stone",
        "summit",
        "terrace",
        "thicket",
        "timber",
        "trail",
        "tundra",
        "vale",
        "valley",
        "vista",
        "volcano",
        "waterfall",
        "willow",
        "woods",
    ];

    // Collect existing session names from session files
    let mut existing = std::collections::HashSet::new();
    if let Ok(entries) = std::fs::read_dir(sessions_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("db")
                && let Some(filename) = path.file_name().and_then(|name| name.to_str())
                && let Ok(start) = session_log::load_session_start(filename)
            {
                existing.insert(start.session_name);
            }
        }
    }

    let mut rng = rand::rng();
    loop {
        let adj = ADJECTIVES[rng.random_range(0..ADJECTIVES.len())];
        let noun = NOUNS[rng.random_range(0..NOUNS.len())];
        let name = format!("{}-{}", adj, noun);
        if !existing.contains(&name) {
            return name;
        }
    }
}

/// Copy the last assistant response to the system clipboard.
fn copy_last_response(app: &App) {
    let last_text = app.blocks.iter().rev().find_map(|b| {
        if let DisplayBlock::AssistantText(text) = b {
            Some(text.clone())
        } else {
            None
        }
    });
    if let Some(text) = last_text
        && let Ok(mut clipboard) = arboard::Clipboard::new()
    {
        let _ = clipboard.set_text(text);
    }
}
