mod provider;
mod session;

pub(crate) use session::switch_to_session_identifier;

use super::runtime::send_queued_work;
use super::runtime::sync_runtime_tool_surface;
use super::*;
use crate::SkillCatalog;
use crate::turn_runner::make_turn_input;
use crate::{controls_document, help_document, info_document};
use lash_core::runtime::{EffectScope, QueuedWorkBatch, QueuedWorkPayload, SlotPolicy};

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

fn current_epoch_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u128::from(u64::MAX)) as u64
}

fn ready_batches_for_idle_dispatch(batches: &[QueuedWorkBatch]) -> Vec<QueuedWorkBatch> {
    let now = current_epoch_ms();
    let ready = batches
        .iter()
        .filter(|batch| batch.available_at_ms <= now)
        .collect::<Vec<_>>();
    let Some(first) = ready.first() else {
        return Vec::new();
    };
    if !batch_has_visible_turn_input(first) {
        return Vec::new();
    }
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
            if first_slot_policy == SlotPolicy::Exclusive {
                break;
            }
            continue;
        }
        if first_slot_policy != SlotPolicy::Join
            || batch.slot_policy != SlotPolicy::Join
            || batch.delivery_policy != first_delivery_policy
            || batch.merge_key != first_merge_key
        {
            break;
        }
        selected.push((*batch).clone());
    }
    selected
}

fn batch_has_visible_turn_input(batch: &QueuedWorkBatch) -> bool {
    batch.items.iter().any(|item| match &item.payload {
        QueuedWorkPayload::TurnInput { input } => turn_input_has_visible_content(input),
        QueuedWorkPayload::ProcessWake { .. } | QueuedWorkPayload::SessionCommand { .. } => false,
    })
}

fn turn_input_has_visible_content(input: &lash_core::TurnInput) -> bool {
    input.items.iter().any(|item| match item {
        lash_core::InputItem::Text { text } => !text.trim().is_empty(),
        lash_core::InputItem::ImageRef { id } => input.image_blobs.contains_key(id),
    })
}

fn display_turns_for_queued_batches(
    app: &mut App,
    batches: &[QueuedWorkBatch],
) -> Vec<PreparedTurn> {
    let mut turns = Vec::new();
    for batch in batches {
        if let Some(turn) = app.take_prepared_turn_for_queued_batch(batch) {
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

    let ready_batch_ids = ready_batches
        .iter()
        .map(|batch| batch.batch_id.clone())
        .collect::<Vec<_>>();
    app.suppress_queue_preview_batches(ready_batch_ids.iter().map(String::as_str));
    app.remove_queued_work_batches(&ready_batch_ids);
    let display_turns = display_turns_for_queued_batches(app, &ready_batches);
    if let Some(first_turn) = display_turns.first().cloned() {
        *last_turn = Some(TurnReplayPayload {
            turn_input: make_turn_input(&first_turn),
            prepared_turn: first_turn,
            execution_mode: current_execution_mode.clone(),
        });
    }
    send_queued_work(
        ready_batch_ids,
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
                && let Ok(state) = session.control().tools().state().await
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
            let standard_context_approach = (current_execution_mode == &ModeId::standard())
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
            let scoped_effect_controller = effect_host.scoped(EffectScope::runtime_operation(
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
                .control()
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

#[cfg(test)]
mod tests {
    use super::*;
    use lash_core::runtime::{DeliveryPolicy, MergeKey, QueuedWorkItem};

    fn queued_batch(
        enqueue_seq: u64,
        payload: QueuedWorkPayload,
        slot_policy: SlotPolicy,
    ) -> QueuedWorkBatch {
        QueuedWorkBatch {
            batch_id: format!("batch-{enqueue_seq}"),
            session_id: "test-session".to_string(),
            enqueue_seq,
            source_key: Some(format!("source-{enqueue_seq}")),
            delivery_policy: DeliveryPolicy::EarliestSafeBoundary,
            slot_policy,
            merge_key: MergeKey::Never,
            available_at_ms: 0,
            enqueued_at_ms: 0,
            items: vec![QueuedWorkItem {
                item_id: format!("batch-{enqueue_seq}:item:0"),
                payload,
            }],
        }
    }

    fn process_wake_payload() -> QueuedWorkPayload {
        serde_json::from_value(serde_json::json!({
            "type": "process_wake",
            "wake": {
                "wake_id": "wake:test",
                "target_session_id": "test-session",
                "target_scope_id": "session:test-session",
                "process_id": "process:test",
                "sequence": 1,
                "event_type": "process.wake",
                "event_invocation": {
                    "scope": {
                        "session_id": "test-session"
                    },
                    "subject": {
                        "type": "process_event",
                        "process_id": "process:test",
                        "sequence": 1,
                        "event_type": "process.wake"
                    }
                },
                "dedupe_key": "process:test:1",
                "input": "wake payload",
                "created_at_ms": 0
            }
        }))
        .expect("process wake payload")
    }

    #[test]
    fn idle_dispatch_does_not_claim_process_wake_as_foreground_turn() {
        let batches = vec![
            queued_batch(1, process_wake_payload(), SlotPolicy::Exclusive),
            queued_batch(
                2,
                QueuedWorkPayload::turn_input(lash_core::TurnInput::text("user turn")),
                SlotPolicy::Exclusive,
            ),
        ];

        assert!(ready_batches_for_idle_dispatch(&batches).is_empty());
    }

    #[test]
    fn idle_dispatch_claims_visible_turn_input_when_it_is_first_ready_batch() {
        let batches = vec![queued_batch(
            1,
            QueuedWorkPayload::turn_input(lash_core::TurnInput::text("user turn")),
            SlotPolicy::Exclusive,
        )];

        let ready = ready_batches_for_idle_dispatch(&batches);

        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].batch_id, "batch-1");
    }

    #[test]
    fn idle_dispatch_does_not_claim_empty_turn_input_as_foreground_turn() {
        let batches = vec![queued_batch(
            1,
            QueuedWorkPayload::turn_input(lash_core::TurnInput::text("")),
            SlotPolicy::Exclusive,
        )];

        assert!(ready_batches_for_idle_dispatch(&batches).is_empty());
    }
}
