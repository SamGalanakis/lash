use std::sync::Arc;

use lash::session_model::{Message, MessageRole, Part, PartKind, PruneState, fresh_message_id};
use lash::{
    CachedModelCatalog, DynamicStateSnapshot, DynamicToolProvider, ExecutionMode, LashRuntime,
    PersistedSessionConfig, PersistedTurnState, PromptUsage, Provider, SessionStateEnvelope, Store,
    TokenUsage,
};

use crate::app::{App, DisplayBlock, projected_blocks_from_state};
use crate::resume_snapshot;
use crate::session_log;

fn push_history_system_message(history: &mut Vec<Message>, content: String) {
    let sys_id = fresh_message_id();
    history.push(Message {
        id: sys_id.clone(),
        role: MessageRole::System,
        parts: vec![Part {
            id: format!("{}.p0", sys_id),
            kind: PartKind::Text,
            content,
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            prune_state: PruneState::Intact,
        }],
        user_input: None,
        origin: None,
    });
}

fn execution_state_reset_message() -> String {
    "Session resumed, but execution state could not be restored. Recreate any in-memory state you still need.".to_string()
}

async fn restore_execution_state_if_present<'a>(
    runtime: &'a mut Option<LashRuntime>,
    app: &'a mut App,
    history: &'a mut Vec<Message>,
    snapshot: Option<&'a [u8]>,
) {
    let Some(rt) = runtime.as_mut() else {
        return;
    };
    match snapshot {
        Some(snapshot) => match rt.restore_execution_state(snapshot).await {
            Ok(()) => {
                app.blocks.push(DisplayBlock::SystemMessage(
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
    runtime: &mut Option<LashRuntime>,
    provider: &Provider,
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
        rt.update_session_config(
            None,
            Some(app.model.clone()),
            None,
            Some(restored_context_window as usize),
        )
        .await;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn apply_graph_resume_state(
    graph: lash::SessionGraph,
    history: &mut Vec<Message>,
    runtime: &mut Option<LashRuntime>,
    app: &mut App,
    turn_counter: &mut usize,
    execution_mode: &mut ExecutionMode,
    provider: &Provider,
    current_model_variant: &mut Option<String>,
    dynamic_tools: &Arc<DynamicToolProvider>,
    desired_dynamic: &mut DynamicStateSnapshot,
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
    let messages = graph.project_messages();
    let tool_calls = graph.project_tool_calls();
    *history = messages.clone();

    if let Some(dynamic_state) = graph.latest_dynamic_state() {
        let _ = dynamic_tools.apply_state(dynamic_state);
        *desired_dynamic = dynamic_tools.export_state();
    }

    let turn_state = graph.latest_turn_state();
    *turn_counter = turn_state
        .as_ref()
        .map(|state| state.iteration)
        .unwrap_or(0);
    app.token_usage = restored_token_usage(turn_state.as_ref()).unwrap_or_default();
    app.last_prompt_usage = normalized_last_prompt_usage(
        turn_state
            .as_ref()
            .and_then(|state| state.last_prompt_usage.clone()),
    );

    let config = graph.latest_session_config();
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

    let requested_execution_mode = config.as_ref().map(|state| state.execution_mode);
    let restored_execution_mode = requested_execution_mode
        .and_then(|mode| crate::ensure_supported_execution_mode(mode).ok())
        .unwrap_or_else(lash::default_execution_mode);
    *execution_mode = restored_execution_mode;

    if let Some(requested_execution_mode) = requested_execution_mode.as_ref()
        && !lash::execution_mode_supported(restored_execution_mode)
    {
        app.blocks.push(DisplayBlock::SystemMessage(format!(
            "This build does not support `{}` mode; resuming in `standard`.",
            crate::execution_mode_label(*requested_execution_mode)
        )));
    }

    let execution_state_snapshot = graph.latest_execution_state().unwrap_or(None);
    restore_execution_state_if_present(runtime, app, history, execution_state_snapshot.as_deref())
        .await;

    if let Some(rt) = runtime.as_mut() {
        rt.update_session_config(None, None, Some(current_model_variant.clone()), None)
            .await;
        let _ = rt.refresh_session_execution_surface().await;
        let mut restored_policy = rt.export_state().policy;
        restored_policy.execution_mode = restored_execution_mode;
        restored_policy.model = app.model.clone();
        restored_policy.model_variant = current_model_variant.clone();
        restored_policy.provider = provider.clone();
        if let Some(context_window) = app.context_window {
            restored_policy.max_context_tokens = Some(context_window as usize);
        }
        rt.set_state(SessionStateEnvelope {
            session_id: crate::ROOT_SESSION_ID.to_string(),
            policy: restored_policy,
            session_graph: graph,
            messages,
            tool_calls,
            iteration: *turn_counter,
            token_usage: app.token_usage.clone(),
            last_prompt_usage: app.last_prompt_usage.clone(),
            execution_state_snapshot,
        });
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub async fn load_resumed_session(
    filename: &str,
    app: &mut App,
    history: &mut Vec<Message>,
    runtime: &mut Option<LashRuntime>,
    turn_counter: &mut usize,
    execution_mode: &mut ExecutionMode,
    provider: &Provider,
    current_model_variant: &mut Option<String>,
    dynamic_tools: &Arc<DynamicToolProvider>,
    desired_dynamic: &mut DynamicStateSnapshot,
    model_catalog: &CachedModelCatalog,
) -> Result<(), String> {
    let loaded =
        session_log::load_session(filename).map_err(|err| format!("Could not load: {err}"))?;
    *history = loaded.messages;
    app.blocks = loaded.blocks;
    app.last_response_usage = loaded.last_token_usage;
    app.last_prompt_usage = None;
    app.plugin_mode_indicators = loaded.plugin_mode_indicators;
    app.blocks.push(DisplayBlock::SystemMessage(format!(
        "Resumed: {}",
        filename
    )));
    restore_session_state(
        filename,
        history,
        runtime,
        app,
        turn_counter,
        execution_mode,
        provider,
        current_model_variant,
        dynamic_tools,
        desired_dynamic,
        model_catalog,
    )
    .await?;
    app.stop_turn();
    app.streaming_output = loaded.streaming_output;
    app.streaming_output_hidden = loaded.streaming_output_hidden;
    app.streaming_output_partial = loaded.streaming_output_partial;
    app.invalidate_height_cache();
    app.resume_follow_output();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub async fn load_resumed_session_by_id(
    session_id: &str,
    app: &mut App,
    history: &mut Vec<Message>,
    runtime: &mut Option<LashRuntime>,
    turn_counter: &mut usize,
    execution_mode: &mut ExecutionMode,
    provider: &Provider,
    current_model_variant: &mut Option<String>,
    dynamic_tools: &Arc<DynamicToolProvider>,
    desired_dynamic: &mut DynamicStateSnapshot,
    model_catalog: &CachedModelCatalog,
) -> Result<(), String> {
    let filename = session_log::filename_for_session_id(session_id)
        .ok_or_else(|| format!("Could not find session `{session_id}`"))?;
    load_resumed_session(
        &filename,
        app,
        history,
        runtime,
        turn_counter,
        execution_mode,
        provider,
        current_model_variant,
        dynamic_tools,
        desired_dynamic,
        model_catalog,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn restore_session_state(
    session_filename: &str,
    history: &mut Vec<Message>,
    runtime: &mut Option<LashRuntime>,
    app: &mut App,
    turn_counter: &mut usize,
    execution_mode: &mut ExecutionMode,
    provider: &Provider,
    current_model_variant: &mut Option<String>,
    dynamic_tools: &Arc<DynamicToolProvider>,
    desired_dynamic: &mut DynamicStateSnapshot,
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
            app.blocks.push(DisplayBlock::SystemMessage(format!(
                "Could not open session database: {err}"
            )));
            return Ok(());
        }
    };

    if let Some(live) = resume_snapshot::load_live_resume_snapshot(&resume_store) {
        let live_messages = live.graph.project_messages();
        let live_tool_calls = live.graph.project_tool_calls();
        *history = live_messages.clone();
        app.blocks = projected_blocks_from_state(&live_messages, &live_tool_calls, &live.ui_state);
        app.last_response_usage = live.ui_state.last_response_usage.clone();
        app.plugin_mode_indicators = live.ui_state.plugin_mode_indicators.clone();
        app.streaming_output = live.ui_state.streaming_output.clone();
        app.streaming_output_hidden = live.ui_state.streaming_output_hidden;
        app.streaming_output_partial = live.ui_state.streaming_output_partial.clone();
        app.invalidate_height_cache();
        app.blocks.push(DisplayBlock::SystemMessage(
            "Interrupted runtime state restored from a live snapshot.".to_string(),
        ));
        return apply_graph_resume_state(
            live.graph,
            history,
            runtime,
            app,
            turn_counter,
            execution_mode,
            provider,
            current_model_variant,
            dynamic_tools,
            desired_dynamic,
            model_catalog,
        )
        .await;
    }

    if let Some(graph) = resume_store.load_session_graph() {
        return apply_graph_resume_state(
            graph,
            history,
            runtime,
            app,
            turn_counter,
            execution_mode,
            provider,
            current_model_variant,
            dynamic_tools,
            desired_dynamic,
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
    use crate::ui_resume;

    use lash::{
        MemoryModelCatalogStore, PluginHost, PluginSpecFactory, RuntimeHostConfig, RuntimeServices,
        ToolProvider,
    };

    fn persist_session_graph(
        store: &Store,
        graph: lash::SessionGraph,
        ui_state: crate::app::UiResumeState,
    ) {
        ui_resume::save_ui_resume_state(store, &ui_state);
        store.save_session_graph(graph);
    }

    fn graph_with_state(
        messages: Vec<Message>,
        iteration: usize,
        token_usage: TokenUsage,
        last_prompt_usage: Option<PromptUsage>,
    ) -> lash::SessionGraph {
        let mut graph = lash::SessionGraph::from_projection(&messages, &[]);
        graph.record_runtime_state(
            &lash::PersistedSessionConfig {
                provider_id: "openai_generic".to_string(),
                configured_model: "gpt-5".to_string(),
                context_window: 200_000,
                execution_mode: ExecutionMode::Standard,
                model_variant: None,
            },
            &lash::PersistedTurnState {
                iteration,
                token_usage,
                last_prompt_usage,
            },
            Some(&DynamicStateSnapshot {
                base_generation: 0,
                tools: std::collections::BTreeMap::new(),
                enabled_tools: std::collections::BTreeSet::new(),
            }),
            None,
            None,
        );
        graph
    }

    fn text_message(id: &str, role: MessageRole, content: &str) -> Message {
        Message {
            id: id.to_string(),
            role,
            parts: vec![Part {
                id: format!("{id}.p0"),
                kind: PartKind::Text,
                content: content.to_string(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                prune_state: PruneState::Intact,
            }],
            user_input: None,
            origin: None,
        }
    }

    async fn build_runtime(
        provider: &Provider,
    ) -> (
        Arc<DynamicToolProvider>,
        DynamicStateSnapshot,
        CachedModelCatalog,
        LashRuntime,
    ) {
        struct EmptyTools;

        #[async_trait::async_trait]
        impl ToolProvider for EmptyTools {
            fn definitions(&self) -> Vec<lash::ToolDefinition> {
                Vec::new()
            }

            async fn execute(&self, _name: &str, _args: &serde_json::Value) -> lash::ToolResult {
                lash::ToolResult::err(serde_json::json!("Unknown tool"))
            }
        }

        let tools: Arc<dyn ToolProvider> = Arc::new(EmptyTools);
        let model_catalog =
            CachedModelCatalog::models_dev(Arc::new(MemoryModelCatalogStore::new(None)), None)
                .expect("catalog");
        let plugins = PluginHost::new(vec![Arc::new(PluginSpecFactory::new(
            "resume_tools",
            Arc::new(
                move |_ctx| Ok(lash::PluginSpec::new().with_tool_provider(Arc::clone(&tools))),
            ),
        ))])
        .with_dynamic_tools()
        .build_standard_session("root", None)
        .expect("plugins");
        let dynamic_tools = plugins.dynamic_tools().expect("dynamic tools");
        let desired_dynamic = dynamic_tools.export_state();
        let runtime_services = RuntimeServices::new(plugins);
        let runtime = LashRuntime::from_state(
            lash::SessionPolicy {
                execution_mode: ExecutionMode::Standard,
                provider: provider.clone(),
                model: "gpt-5".into(),
                max_context_tokens: Some(200_000),
                ..lash::SessionPolicy::default()
            },
            RuntimeHostConfig::default(),
            runtime_services,
            SessionStateEnvelope::default(),
        )
        .await
        .expect("runtime");
        (dynamic_tools, desired_dynamic, model_catalog, runtime)
    }

    #[tokio::test]
    async fn restore_session_state_restores_graph_turn_state_into_app_and_runtime() {
        let _env_guard = env_lock().lock().await;
        let temp = TempDirGuard::new("lash-resume-usage");
        let _lash_home = EnvVarGuard::set("LASH_HOME", temp.path());
        let sessions_dir = lash::lash_home().join("sessions");
        std::fs::create_dir_all(&sessions_dir).expect("sessions dir");

        let db_path = sessions_dir.join("resume-usage.db");
        let store = Store::open(&db_path).expect("store");
        persist_session_graph(
            &store,
            graph_with_state(
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
            ),
            crate::app::UiResumeState::default(),
        );

        let provider = Provider::OpenAiGeneric {
            api_key: "test-key".into(),
            base_url: "https://example.invalid/v1".into(),
            options: lash::ProviderOptions::default(),
        };
        let (dynamic_tools, mut desired_dynamic, model_catalog, runtime) =
            build_runtime(&provider).await;

        let mut app = App::new("gpt-5".into(), "resume-usage".into());
        let mut history = Vec::new();
        let mut runtime = Some(runtime);
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
            &dynamic_tools,
            &mut desired_dynamic,
            &model_catalog,
        )
        .await
        .expect("restore");

        assert_eq!(turn_counter, 7);
        assert_eq!(execution_mode, ExecutionMode::Standard);
        assert_eq!(app.token_usage.input_tokens, 1200);
        assert_eq!(app.token_usage.output_tokens, 340);
        assert_eq!(app.token_usage.cached_input_tokens, 80);
        assert_eq!(app.token_usage.reasoning_tokens, 55);
        let app_prompt_usage = app.last_prompt_usage.expect("app prompt usage");
        assert_eq!(app_prompt_usage.prompt_context_tokens, 4096);
        assert_eq!(app_prompt_usage.context_budget_tokens, 4096);

        let restored_state = runtime.expect("runtime").export_state();
        assert_eq!(restored_state.iteration, 7);
        assert_eq!(restored_state.token_usage.input_tokens, 1200);
        assert_eq!(restored_state.token_usage.reasoning_tokens, 55);
    }

    #[tokio::test]
    async fn restore_session_state_prefers_live_graph() {
        let _env_guard = env_lock().lock().await;
        let temp = TempDirGuard::new("lash-resume-live-snapshot");
        let _lash_home = EnvVarGuard::set("LASH_HOME", temp.path());
        let sessions_dir = lash::lash_home().join("sessions");
        std::fs::create_dir_all(&sessions_dir).expect("sessions dir");

        let db_path = sessions_dir.join("resume-live.db");
        let store = Store::open(&db_path).expect("store");
        persist_session_graph(
            &store,
            graph_with_state(
                vec![text_message(
                    "old",
                    MessageRole::User,
                    "canonical history message",
                )],
                1,
                TokenUsage::default(),
                None,
            ),
            crate::app::UiResumeState::default(),
        );

        let live_messages = vec![text_message(
            "m0",
            MessageRole::User,
            "live snapshot message",
        )];
        crate::resume_snapshot::save_live_resume_snapshot(
            &store,
            &SessionStateEnvelope {
                session_id: crate::ROOT_SESSION_ID.to_string(),
                policy: lash::SessionPolicy {
                    execution_mode: ExecutionMode::Standard,
                    ..lash::SessionPolicy::default()
                },
                session_graph: graph_with_state(
                    live_messages.clone(),
                    2,
                    TokenUsage::default(),
                    None,
                ),
                messages: live_messages.clone(),
                tool_calls: Vec::new(),
                iteration: 2,
                token_usage: TokenUsage::default(),
                last_prompt_usage: None,
                execution_state_snapshot: None,
            },
            &crate::app::UiResumeState::default(),
            &DynamicStateSnapshot {
                base_generation: 0,
                tools: std::collections::BTreeMap::new(),
                enabled_tools: std::collections::BTreeSet::new(),
            },
        )
        .expect("live snapshot");

        let provider = Provider::OpenAiGeneric {
            api_key: "test-key".into(),
            base_url: "https://example.invalid/v1".into(),
            options: lash::ProviderOptions::default(),
        };
        let (dynamic_tools, mut desired_dynamic, model_catalog, runtime) =
            build_runtime(&provider).await;

        let mut app = App::new("gpt-5".into(), "resume-live".into());
        let mut history = Vec::new();
        let mut runtime = Some(runtime);
        let mut turn_counter = 0;
        let mut execution_mode = ExecutionMode::Standard;
        let mut current_model_variant = None;

        restore_session_state(
            "resume-live.db",
            &mut history,
            &mut runtime,
            &mut app,
            &mut turn_counter,
            &mut execution_mode,
            &provider,
            &mut current_model_variant,
            &dynamic_tools,
            &mut desired_dynamic,
            &model_catalog,
        )
        .await
        .expect("restore");

        assert!(!history.is_empty());
        assert_eq!(history[0].parts[0].content, "live snapshot message");
        let restored_runtime = runtime.expect("runtime").export_state();
        assert_eq!(restored_runtime.iteration, 2);
        assert_eq!(restored_runtime.messages.len(), 1);
        assert_eq!(
            restored_runtime.messages[0].parts[0].content,
            "live snapshot message"
        );
    }

    #[tokio::test]
    async fn load_resumed_session_clears_stale_live_turn_ui() {
        let _env_guard = env_lock().lock().await;
        let temp = TempDirGuard::new("lash-load-resume-live-turn");
        let _lash_home = EnvVarGuard::set("LASH_HOME", temp.path());
        let sessions_dir = lash::lash_home().join("sessions");
        std::fs::create_dir_all(&sessions_dir).expect("sessions dir");

        let filename = "resume-ui.db";
        let db_path = sessions_dir.join(filename);
        let store = Store::open(&db_path).expect("store");
        persist_session_graph(
            &store,
            graph_with_state(
                vec![text_message("m0", MessageRole::User, "hello")],
                1,
                TokenUsage::default(),
                None,
            ),
            crate::app::UiResumeState {
                streaming_output: vec!["started git status --short".to_string()],
                streaming_output_hidden: 1,
                streaming_output_partial: "partial".to_string(),
                ..crate::app::UiResumeState::default()
            },
        );

        let provider = Provider::OpenAiGeneric {
            api_key: "test-key".into(),
            base_url: "https://example.invalid/v1".into(),
            options: lash::ProviderOptions::default(),
        };
        let model_catalog =
            CachedModelCatalog::models_dev(Arc::new(MemoryModelCatalogStore::new(None)), None)
                .expect("catalog");
        let plugins = PluginHost::new(vec![])
            .with_dynamic_tools()
            .build_standard_session("root", None)
            .expect("plugins");
        let dynamic_tools = plugins.dynamic_tools().expect("dynamic tools");
        let mut desired_dynamic = dynamic_tools.export_state();

        let mut app = App::new("gpt-5".into(), "resume-ui".into());
        let mut history = Vec::new();
        let mut runtime: Option<LashRuntime> = None;
        let mut turn_counter = 0;
        let mut execution_mode = ExecutionMode::Standard;
        let mut current_model_variant = None;

        load_resumed_session(
            filename,
            &mut app,
            &mut history,
            &mut runtime,
            &mut turn_counter,
            &mut execution_mode,
            &provider,
            &mut current_model_variant,
            &dynamic_tools,
            &mut desired_dynamic,
            &model_catalog,
        )
        .await
        .expect("load resumed session");

        assert!(!app.running);
        assert!(app.live_turn.is_none());
        assert_eq!(
            app.streaming_output,
            vec!["started git status --short".to_string()]
        );
        assert_eq!(app.streaming_output_hidden, 1);
        assert_eq!(app.streaming_output_partial, "partial");
    }
}
