use std::sync::Arc;

use lash::session_model::{Message, MessageRole, Part, PartKind, PruneState, fresh_message_id};
use lash::{
    CachedModelCatalog, ExecutionMode, PersistedSessionConfig, PersistedTurnState, PromptUsage,
    ProviderHandle, TokenUsage, ToolState,
};
use lash_embed::control::SessionConfigPatch;
use lash_embed::{LashSession, ModelSelection};
use lash_sqlite_store::Store;

use crate::app::{App, UiTimelineItem};
use crate::session_log;

fn push_history_system_message(history: &mut Vec<Message>, content: String) {
    let sys_id = fresh_message_id();
    history.push(Message {
        id: sys_id.clone(),
        role: MessageRole::System,
        parts: lash::shared_parts(vec![Part {
            id: format!("{}.p0", sys_id),
            kind: PartKind::Text,
            content,
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            tool_item_id: None,
            tool_signature: None,
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
        Some(snapshot) => match rt.control().state().restore_execution(snapshot).await {
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
    model_catalog: &CachedModelCatalog,
) -> Result<(), String> {
    let Some(config) = config else {
        return Ok(());
    };
    let restored_model = config.configured_model.as_str();
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
    let restored_context_window = Some(config.context_window).filter(|window| *window > 0)
        .or_else(|| {
            let snapshot = model_catalog.snapshot();
            provider
                .resolve_model_spec(restored_model, &snapshot)
                .ok()
                .map(|spec| spec.context_window())
        })
        .ok_or_else(|| {
            format!(
                "Cannot resume session with model `{}`: no context-window entry was saved and the supplied model catalog does not contain it.",
                restored_model
            )
        })?;
    app.model = restored_model.to_string();
    app.context_window = Some(restored_context_window);
    app.context_usage_excludes_cached_input = provider.input_usage_excludes_cached_tokens();
    if let Some(rt) = runtime.as_mut() {
        let _ = rt
            .control()
            .config()
            .update(SessionConfigPatch {
                model: Some(ModelSelection::new(app.model.clone(), None)),
                max_context_tokens: Some(restored_context_window as usize),
                ..SessionConfigPatch::default()
            })
            .await;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn apply_graph_resume_state(
    graph: lash::SessionGraph,
    config: Option<lash::PersistedSessionConfig>,
    _token_ledger: Vec<lash::TokenLedgerEntry>,
    checkpoint: Option<lash::HydratedSessionCheckpoint>,
    _checkpoint_ref: Option<lash::BlobRef>,
    history: &mut Vec<Message>,
    runtime: &mut Option<LashSession>,
    app: &mut App,
    turn_counter: &mut usize,
    execution_mode: &mut ExecutionMode,
    provider: &ProviderHandle,
    current_model_variant: &mut Option<String>,
    desired_tool_state: &mut ToolState,
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
    let read_view = lash::SessionReadView::from_exported_state(&lash::SessionStateEnvelope {
        session_graph: graph.clone(),
        ..lash::SessionStateEnvelope::default()
    });
    let messages = read_view.messages().to_vec();
    let _tool_calls = read_view.tool_calls().to_vec();
    *history = messages.clone();

    if let Some(tool_state) = checkpoint
        .as_ref()
        .and_then(|checkpoint| checkpoint.tool_state.clone())
        && let Some(session) = runtime.as_ref()
    {
        let _ = session.control().tools().apply_state(tool_state).await;
        if let Ok(state) = session.control().tools().state().await {
            *desired_tool_state = state;
        }
    }

    let turn_state = checkpoint.as_ref().map(|checkpoint| &checkpoint.turn_state);
    *turn_counter = turn_state
        .as_ref()
        .map(|state| state.turn_index)
        .unwrap_or(0);
    app.token_usage = restored_token_usage(turn_state).unwrap_or_default();
    app.last_prompt_usage = normalized_last_prompt_usage(
        turn_state
            .as_ref()
            .and_then(|state| state.last_prompt_usage.clone()),
    );

    restore_model_from_graph_config(config.as_ref(), app, runtime, provider, model_catalog).await?;

    *current_model_variant = config
        .as_ref()
        .and_then(|state| state.model_variant.clone())
        .or_else(|| {
            provider
                .default_model_variant(&app.model)
                .map(str::to_string)
        });
    app.set_model_variant(current_model_variant.clone());

    let requested_execution_mode = config.as_ref().map(|state| state.execution_mode.clone());
    let restored_execution_mode = requested_execution_mode
        .clone()
        .and_then(|mode| crate::ensure_supported_execution_mode(mode).ok())
        .unwrap_or_else(lash::default_execution_mode);
    *execution_mode = restored_execution_mode.clone();

    if let Some(requested_execution_mode) = requested_execution_mode.as_ref()
        && !lash::execution_mode_supported(&restored_execution_mode)
    {
        app.timeline.push(UiTimelineItem::SystemMessage(format!(
            "This build does not support `{}` mode; resuming in `standard`.",
            crate::execution_mode_label(requested_execution_mode)
        )));
    }

    let execution_state_snapshot = checkpoint
        .as_ref()
        .and_then(|checkpoint| checkpoint.execution_state.clone());
    restore_execution_state_if_present(runtime, app, history, execution_state_snapshot.as_deref())
        .await;

    if let Some(rt) = runtime.as_mut() {
        rt.control()
            .config()
            .update(SessionConfigPatch {
                provider: Some(provider.clone()),
                model: Some(ModelSelection::new(
                    app.model.clone(),
                    current_model_variant.clone(),
                )),
                max_context_tokens: app.context_window.map(|window| window as usize),
                ..SessionConfigPatch::default()
            })
            .await
            .map_err(|err| err.to_string())?;
        let _ = rt.control().tools().refresh_surface().await;
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
    desired_tool_state: &mut ToolState,
    model_catalog: &CachedModelCatalog,
) -> Result<(), String> {
    let filename = session_log::filename_for_session_identifier(identifier)
        .unwrap_or_else(|| identifier.to_string());
    let loaded =
        session_log::load_session(&filename).map_err(|err| format!("Could not load: {err}"))?;
    let store = Arc::new(
        Store::open(&session_log::sessions_dir().join(&loaded.filename))
            .map_err(|err| format!("Could not open session database: {err}"))?,
    );
    *logger = session_log::SessionLogger::resume(store, &loaded.filename)
        .map_err(|err| format!("Could not resume session logger: {err}"))?;
    *history = loaded.messages;
    app.timeline = loaded.blocks;
    app.session_id = loaded.session_id;
    app.session_name = loaded.session_name;
    app.last_response_usage = loaded.last_token_usage;
    app.last_prompt_usage = None;
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
        desired_tool_state,
        model_catalog,
    )
    .await?;
    app.stop_turn();
    app.live_tool_output = loaded.live_tool_output;
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
    desired_tool_state: &mut ToolState,
    model_catalog: &CachedModelCatalog,
) -> Result<(), String> {
    let db_path = session_log::sessions_dir().join(session_filename);

    if !db_path.exists() {
        push_history_system_message(history, execution_state_reset_message());
        return Ok(());
    }

    let resume_store = match Store::open(&db_path) {
        Ok(s) => s,
        Err(err) => {
            app.timeline.push(UiTimelineItem::SystemMessage(format!(
                "Could not open session database: {err}"
            )));
            return Ok(());
        }
    };

    if let Some(head) = resume_store.load_session_head() {
        let checkpoint = head
            .checkpoint_ref
            .as_ref()
            .and_then(|blob_ref| resume_store.get_checkpoint(blob_ref));
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
            desired_tool_state,
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

    use lash::MemoryModelCatalogStore;

    fn persist_session_head(
        store: &Store,
        graph: lash::SessionGraph,
        checkpoint: lash::HydratedSessionCheckpoint,
    ) {
        let checkpoint_ref = store.put_checkpoint(&checkpoint).checkpoint_ref;
        store.save_session_head(lash::SessionHead {
            session_id: "root".to_string(),
            head_revision: 0,
            graph,
            config: lash::PersistedSessionConfig {
                provider_id: "openai_generic".to_string(),
                configured_model: "gpt-5".to_string(),
                context_window: 200_000,
                execution_mode: ExecutionMode::standard(),
                standard_context_approach: Some(lash::StandardContextApproach::default()),
                model_variant: None,
            },
            checkpoint_ref: Some(checkpoint_ref),
            token_ledger: Vec::new(),
        });
    }

    fn state_with_graph(
        messages: Vec<Message>,
        iteration: usize,
        token_usage: TokenUsage,
        last_prompt_usage: Option<PromptUsage>,
    ) -> (lash::SessionGraph, lash::HydratedSessionCheckpoint) {
        (
            lash::SessionGraph::from_active_read_state(&messages, &[]),
            lash::HydratedSessionCheckpoint {
                turn_state: lash::PersistedTurnState {
                    turn_index: iteration,
                    token_usage,
                    last_prompt_usage,
                    mode_turn_options: Default::default(),
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
        let store = Store::open(&db_path).expect("store");
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
        persist_session_head(&store, graph, checkpoint);

        let provider = lash::ProviderHandle::new(
            lash_provider_openai::OpenAiGenericProvider::new(
                "test-key",
                "https://example.invalid/v1",
            )
            .into_components(),
        );
        let mut desired_tool_state = ToolState::default();
        let model_catalog = build_model_catalog();

        let mut app = App::new(
            "gpt-5".into(),
            "resume-usage".into(),
            "test-session-id".into(),
        );
        let mut history = Vec::new();
        let mut runtime = None;
        let mut turn_counter = 0;
        let mut execution_mode = ExecutionMode::standard();
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
            &mut desired_tool_state,
            &model_catalog,
        )
        .await
        .expect("restore");

        assert_eq!(turn_counter, 7);
        assert_eq!(execution_mode, ExecutionMode::standard());
        assert_eq!(app.token_usage.input_tokens, 1200);
        assert_eq!(app.token_usage.output_tokens, 340);
        assert_eq!(app.token_usage.cached_input_tokens, 80);
        assert_eq!(app.token_usage.reasoning_tokens, 55);
        let app_prompt_usage = app.last_prompt_usage.expect("app prompt usage");
        assert_eq!(app_prompt_usage.prompt_context_tokens, 4096);
        assert_eq!(app_prompt_usage.context_budget_tokens, 4096);
    }
}
