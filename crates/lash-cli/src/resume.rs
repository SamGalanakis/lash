use std::sync::Arc;

use lash::LashSession;
use lash::admin::SessionConfigPatch;
use lash_core::session_model::{
    Message, MessageRole, Part, PartKind, PruneState, fresh_message_id,
};
use lash_core::store::HydratedSessionCheckpoint;
use lash_core::{
    PersistedSessionConfig, PersistedTurnState, PromptUsage, ProviderHandle, TokenUsage, ToolState,
};
use lash_sqlite_store::Store;

use crate::app::{App, UiTimelineItem};
use crate::execution_settings::ExecutionMode;
use crate::model_catalog::CachedModelCatalog;
use crate::session_log;

fn push_history_system_message(history: &mut Vec<Message>, content: String) {
    let sys_id = fresh_message_id();
    history.push(Message {
        id: sys_id.clone(),
        role: MessageRole::System,
        parts: lash_core::shared_parts(vec![Part {
            id: format!("{}.p0", sys_id),
            kind: PartKind::Text,
            content,
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            tool_replay: None,
            prune_state: PruneState::Intact,
            reasoning_meta: None,
            response_meta: None,
        }]),
        origin: None,
    });
}

fn execution_state_reset_message() -> String {
    "Session resumed, but execution state could not be restored. Recreate any in-memory state you still need.".to_string()
}

async fn restore_execution_state_if_present<'a>(
    runtime: &'a mut Option<LashSession>,
    app: &'a mut App,
    history: &'a mut Vec<Message>,
    snapshot: Option<&'a [u8]>,
) {
    let Some(rt) = runtime.as_mut() else {
        return;
    };
    match snapshot {
        Some(snapshot) => match rt.admin().state().restore_execution(snapshot).await {
            Ok(()) => {
                app.timeline.push(UiTimelineItem::SystemMessage(
                    "Execution state restored from snapshot.".to_string(),
                ));
            }
            Err(e) => {
                push_history_system_message(
                    history,
                    format!(
                        "Session resumed, but execution-state restore failed ({}). Recreate any in-memory state you still need.",
                        e
                    ),
                );
            }
        },
        None => {
            push_history_system_message(history, execution_state_reset_message());
        }
    }
}

fn restored_token_usage(turn_state: Option<&PersistedTurnState>) -> Option<TokenUsage> {
    let usage = turn_state?.token_usage.clone();
    (usage.total() > 0).then_some(usage)
}

fn normalized_last_prompt_usage(last_prompt_usage: Option<PromptUsage>) -> Option<PromptUsage> {
    let mut usage = last_prompt_usage?;
    if usage.context_budget_tokens == 0 && usage.prompt_context_tokens > 0 {
        usage.context_budget_tokens = usage.prompt_context_tokens;
    }
    Some(usage)
}

async fn restore_model_from_graph_config(
    config: Option<&PersistedSessionConfig>,
    app: &mut App,
    runtime: &mut Option<LashSession>,
    provider: &ProviderHandle,
    _model_catalog: &CachedModelCatalog,
) -> Result<(), String> {
    let Some(config) = config else {
        return Ok(());
    };
    let restored_model = config.model.id.as_str();
    if restored_model.is_empty() {
        return Ok(());
    }
    provider
        .validate_model_name(restored_model)
        .map_err(|err| {
            format!(
                "Cannot resume session with model `{}`: {}",
                restored_model, err
            )
        })?;
    let restored_context_window = config.model.context_window_tokens() as u64;
    app.model = restored_model.to_string();
    app.usage.context_window = Some(restored_context_window);
    app.usage.context_usage_excludes_cached_input = provider.input_usage_excludes_cached_tokens();
    if let Some(rt) = runtime.as_mut() {
        let _ = rt
            .admin()
            .config()
            .update(SessionConfigPatch {
                model: Some(config.model.clone()),
                ..SessionConfigPatch::default()
            })
            .await;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn apply_graph_resume_state(
    graph: lash_core::SessionGraph,
    config: Option<lash_core::PersistedSessionConfig>,
    _token_ledger: Vec<lash_core::TokenLedgerEntry>,
    checkpoint: Option<HydratedSessionCheckpoint>,
    _checkpoint_ref: Option<lash_core::BlobRef>,
    history: &mut Vec<Message>,
    runtime: &mut Option<LashSession>,
    app: &mut App,
    turn_counter: &mut usize,
    execution_mode: &mut ExecutionMode,
    provider: &ProviderHandle,
    current_model_variant: &mut Option<String>,
    active_tool_state: &mut ToolState,
    model_catalog: &CachedModelCatalog,
) -> Result<(), String> {
    let mut graph = graph;
    // If the persisted leaf points to a node that no longer exists
    // (compaction rewrote the graph, or the stored session is from
    // an older schema), walk back to the most recent message rather
    // than projecting an empty transcript.
    if graph.heal_orphaned_leaf() {
        tracing::warn!("session graph leaf was orphaned on resume; healed to most recent message");
    }
    let read_view = lash_core::SessionReadView::from_snapshot(&lash_core::SessionSnapshot {
        session_graph: graph.clone(),
        ..lash_core::SessionSnapshot::default()
    });
    let messages = read_view.messages().to_vec();
    *history = messages.clone();

    if let Some(tool_state) = checkpoint
        .as_ref()
        .and_then(|checkpoint| checkpoint.tool_state.clone())
        && let Some(session) = runtime.as_ref()
    {
        // Cold resume restores the persisted snapshot verbatim (adopting its
        // generation), not a generation-checked delta — a session whose tool
        // surface reached generation ≥ 2 would be rejected by `apply_state`.
        let _ = session
            .admin()
            .tools()
            .advanced()
            .restore_state(tool_state)
            .await;
        if let Ok(state) = session.admin().tools().state().await {
            *active_tool_state = state;
        }
    }

    let turn_state = checkpoint.as_ref().map(|checkpoint| &checkpoint.turn_state);
    *turn_counter = turn_state
        .as_ref()
        .map(|state| state.turn_index)
        .unwrap_or(0);
    app.usage.token_usage = restored_token_usage(turn_state).unwrap_or_default();
    app.usage.last_prompt_usage = normalized_last_prompt_usage(
        turn_state
            .as_ref()
            .and_then(|state| state.last_prompt_usage.clone()),
    );

    restore_model_from_graph_config(config.as_ref(), app, runtime, provider, model_catalog).await?;

    *current_model_variant = config
        .as_ref()
        .and_then(|state| state.model.variant.clone())
        .or_else(|| {
            crate::provider_metadata::default_model_variant_for_provider(
                provider.kind(),
                &app.model,
                provider.supported_variants(&app.model),
            )
            .map(str::to_string)
        });
    app.set_model_variant(current_model_variant.clone());

    let _ = execution_mode;

    let execution_state_snapshot = checkpoint
        .as_ref()
        .and_then(|checkpoint| checkpoint.execution_state.clone());
    restore_execution_state_if_present(runtime, app, history, execution_state_snapshot.as_deref())
        .await;

    if let Some(rt) = runtime.as_mut() {
        rt.admin()
            .config()
            .update(SessionConfigPatch {
                provider: Some(provider.clone()),
                model: config.as_ref().map(|config| {
                    let mut model = config.model.clone();
                    model.variant = current_model_variant.clone();
                    model
                }),
                ..SessionConfigPatch::default()
            })
            .await
            .map_err(|err| err.to_string())?;
        let _ = rt
            .admin()
            .commands()
            .refresh_tool_catalog("resume provider update", "resume-provider-update")
            .await;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub async fn load_resumed_session(
    identifier: &str,
    app: &mut App,
    logger: &mut session_log::SessionLogger,
    history: &mut Vec<Message>,
    runtime: &mut Option<LashSession>,
    turn_counter: &mut usize,
    execution_mode: &mut ExecutionMode,
    provider: &ProviderHandle,
    current_model_variant: &mut Option<String>,
    active_tool_state: &mut ToolState,
    model_catalog: &CachedModelCatalog,
) -> Result<(), String> {
    let filename = session_log::filename_for_session_identifier(identifier)
        .await
        .unwrap_or_else(|| identifier.to_string());
    let loaded = session_log::load_session(&filename)
        .await
        .map_err(|err| format!("Could not load: {err}"))?;
    let store = Arc::new(
        Store::open(&session_log::sessions_dir().join(&loaded.filename))
            .await
            .map_err(|err| format!("Could not open session database: {err}"))?,
    );
    *logger = session_log::SessionLogger::resume(store, &loaded.filename)
        .await
        .map_err(|err| format!("Could not resume session logger: {err}"))?;
    *history = loaded.messages;
    app.timeline = loaded.blocks;
    app.session_id = loaded.session_id;
    app.session_name = loaded.session_name;
    app.usage.last_response_usage = loaded.last_token_usage;
    app.usage.last_prompt_usage = None;
    app.plugin_mode_indicators = loaded.plugin_mode_indicators;
    app.timeline.push(UiTimelineItem::SystemMessage(format!(
        "Resumed: {}",
        loaded.filename
    )));
    restore_session_state(
        &filename,
        history,
        runtime,
        app,
        turn_counter,
        execution_mode,
        provider,
        current_model_variant,
        active_tool_state,
        model_catalog,
    )
    .await?;
    app.stop_turn();
    app.live.tool_output = loaded.live_tool_output;
    app.set_execution_mode_label(execution_mode);
    app.invalidate_height_cache();
    app.resume_follow_output();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub async fn restore_session_state(
    session_filename: &str,
    history: &mut Vec<Message>,
    runtime: &mut Option<LashSession>,
    app: &mut App,
    turn_counter: &mut usize,
    execution_mode: &mut ExecutionMode,
    provider: &ProviderHandle,
    current_model_variant: &mut Option<String>,
    active_tool_state: &mut ToolState,
    model_catalog: &CachedModelCatalog,
) -> Result<(), String> {
    let db_path = session_log::sessions_dir().join(session_filename);

    if !db_path.exists() {
        push_history_system_message(history, execution_state_reset_message());
        return Ok(());
    }

    let resume_store = match Store::open(&db_path).await {
        Ok(s) => s,
        Err(err) => {
            app.timeline.push(UiTimelineItem::SystemMessage(format!(
                "Could not open session database: {err}"
            )));
            return Ok(());
        }
    };

    if let Some(head) = resume_store.load_session_head().await {
        let checkpoint = match head.checkpoint_ref.as_ref() {
            Some(blob_ref) => resume_store.get_checkpoint(blob_ref).await,
            None => None,
        };
        return apply_graph_resume_state(
            head.graph,
            Some(head.config),
            head.token_ledger,
            checkpoint,
            head.checkpoint_ref,
            history,
            runtime,
            app,
            turn_counter,
            execution_mode,
            provider,
            current_model_variant,
            active_tool_state,
            model_catalog,
        )
        .await;
    }

    if runtime.is_some() {
        push_history_system_message(history, execution_state_reset_message());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{EnvVarGuard, TempDirGuard, env_lock};

    use crate::model_catalog::MemoryModelCatalogStore;

    async fn persist_session_head(
        store: &Store,
        graph: lash_core::SessionGraph,
        checkpoint: lash_core::store::HydratedSessionCheckpoint,
    ) {
        let checkpoint_ref = store.put_checkpoint(&checkpoint).await.checkpoint_ref;
        store
            .save_session_head(lash_core::store::SessionHead {
                session_id: "root".to_string(),
                head_revision: 0,
                agent_frames: Vec::new(),
                current_agent_frame_id: String::new(),
                graph,
                config: lash_core::PersistedSessionConfig {
                    provider_id: "openai_generic".to_string(),
                    model: lash_core::ModelSpec::from_token_limits("gpt-5", None, 200_000, None)
                        .expect("valid model spec"),
                },
                checkpoint_ref: Some(checkpoint_ref),
                token_ledger: Vec::new(),
            })
            .await;
    }

    fn state_with_graph(
        messages: Vec<Message>,
        iteration: usize,
        token_usage: TokenUsage,
        last_prompt_usage: Option<PromptUsage>,
    ) -> (
        lash_core::SessionGraph,
        lash_core::store::HydratedSessionCheckpoint,
    ) {
        (
            lash_core::SessionGraph::from_active_read_state(&messages),
            lash_core::store::HydratedSessionCheckpoint {
                turn_state: lash_core::PersistedTurnState {
                    turn_index: iteration,
                    token_usage,
                    last_prompt_usage,
                    protocol_turn_options: Default::default(),
                },
                tool_state_ref: None,
                tool_state: Some(ToolState::default()),
                plugin_snapshot_ref: None,
                plugin_snapshot_revision: None,
                plugin_snapshot: None,
                execution_state_ref: None,
                execution_state: None,
            },
        )
    }

    fn build_model_catalog() -> CachedModelCatalog {
        CachedModelCatalog::models_dev(Arc::new(MemoryModelCatalogStore::new(None)), None)
            .expect("catalog")
    }

    #[tokio::test]
    async fn restore_session_state_restores_graph_turn_state_into_app_and_runtime() {
        let _env_guard = env_lock().lock().await;
        let temp = TempDirGuard::new("lash-resume-usage");
        let _lash_home = EnvVarGuard::set("LASH_HOME", temp.path());
        let sessions_dir = crate::paths::lash_home().join("sessions");
        std::fs::create_dir_all(&sessions_dir).expect("sessions dir");

        let db_path = sessions_dir.join("resume-usage.db");
        let store = Store::open(&db_path).await.expect("store");
        let (graph, checkpoint) = state_with_graph(
            Vec::new(),
            7,
            TokenUsage {
                input_tokens: 1200,
                output_tokens: 340,
                cached_input_tokens: 80,
                reasoning_tokens: 55,
            },
            Some(PromptUsage {
                prompt_context_tokens: 4096,
                input_tokens: 3900,
                cached_input_tokens: 196,
                context_budget_tokens: 0,
            }),
        );
        persist_session_head(&store, graph, checkpoint).await;

        let provider = lash_core::ProviderHandle::new(
            lash_provider_openai::OpenAiCompatibleProvider::new(
                "test-key",
                "https://example.invalid/v1",
            )
            .into_components(),
        );
        let mut active_tool_state = ToolState::default();
        let model_catalog = build_model_catalog();

        let mut app = App::new(
            "gpt-5".into(),
            "resume-usage".into(),
            "test-session-id".into(),
        );
        let mut history = Vec::new();
        let mut runtime = None;
        let mut turn_counter = 0;
        let mut execution_mode = ExecutionMode::Standard;
        let mut current_model_variant = None;

        restore_session_state(
            "resume-usage.db",
            &mut history,
            &mut runtime,
            &mut app,
            &mut turn_counter,
            &mut execution_mode,
            &provider,
            &mut current_model_variant,
            &mut active_tool_state,
            &model_catalog,
        )
        .await
        .expect("restore");

        assert_eq!(turn_counter, 7);
        assert_eq!(execution_mode, ExecutionMode::Standard);
        assert_eq!(app.usage.token_usage.input_tokens, 1200);
        assert_eq!(app.usage.token_usage.output_tokens, 340);
        assert_eq!(app.usage.token_usage.cached_input_tokens, 80);
        assert_eq!(app.usage.token_usage.reasoning_tokens, 55);
        let app_prompt_usage = app.usage.last_prompt_usage.expect("app prompt usage");
        assert_eq!(app_prompt_usage.prompt_context_tokens, 4096);
        assert_eq!(app_prompt_usage.context_budget_tokens, 4096);
    }
}
