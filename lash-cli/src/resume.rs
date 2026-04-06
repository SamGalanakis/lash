use std::sync::Arc;

use lash::session_model::{Message, MessageRole, Part, PartKind, PruneState};
use lash::{
    CachedModelCatalog, ContextStrategy, DynamicStateSnapshot, DynamicToolProvider, ExecutionMode,
    LashRuntime, PromptUsage, Provider, SessionStateEnvelope, Store, TokenUsage,
};

use crate::app::{App, DisplayBlock, apply_ui_resume_state_to_blocks, blocks_from_transcript};
use crate::resume_snapshot;
use crate::session_log;

fn push_history_system_message(history: &mut Vec<Message>, content: String) {
    let sys_id = format!("m{}", history.len());
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
        origin: None,
    });
}

fn repl_reset_message() -> String {
    "Session resumed. Your REPL environment was reset — re-import modules and recreate any state you need.".to_string()
}

fn restored_token_usage(state: &lash::store::SessionState) -> Option<TokenUsage> {
    let usage = TokenUsage {
        input_tokens: state.input_tokens,
        output_tokens: state.output_tokens,
        cached_input_tokens: state.cached_input_tokens,
        reasoning_tokens: state.reasoning_tokens,
    };
    (usage.total() > 0).then_some(usage)
}

fn restored_last_prompt_usage(config_value: Option<&serde_json::Value>) -> Option<PromptUsage> {
    let mut usage: PromptUsage = config_value
        .and_then(|value| value.get("last_prompt_usage").cloned())
        .and_then(|value| serde_json::from_value(value).ok())?;
    if usage.context_budget_tokens == 0 && usage.prompt_context_tokens > 0 {
        usage.context_budget_tokens = usage.prompt_context_tokens;
    }
    Some(usage)
}

#[allow(clippy::too_many_arguments)]
pub async fn load_resumed_session(
    filename: &str,
    app: &mut App,
    history: &mut Vec<Message>,
    runtime: &mut Option<LashRuntime>,
    turn_counter: &mut usize,
    execution_mode: &mut ExecutionMode,
    context_strategy: &mut ContextStrategy,
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
        context_strategy,
        provider,
        current_model_variant,
        dynamic_tools,
        desired_dynamic,
        model_catalog,
    )
    .await?;
    tracing::debug!(
        session_file = filename,
        history = history.len(),
        blocks = app.blocks.len(),
        "restored resumed session state"
    );
    // Resumed sessions restore persisted history and durable state, but they do
    // not reconnect to a still-running turn. Keep live-turn chrome cleared while
    // preserving any persisted streaming output transcript.
    app.stop_turn();
    app.streaming_output = loaded.streaming_output;
    app.streaming_output_hidden = loaded.streaming_output_hidden;
    app.streaming_output_partial = loaded.streaming_output_partial;
    app.invalidate_height_cache();
    app.resume_follow_output();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub async fn restore_session_state(
    session_filename: &str,
    history: &mut Vec<Message>,
    runtime: &mut Option<LashRuntime>,
    app: &mut App,
    turn_counter: &mut usize,
    execution_mode: &mut ExecutionMode,
    context_strategy: &mut ContextStrategy,
    provider: &Provider,
    current_model_variant: &mut Option<String>,
    dynamic_tools: &Arc<DynamicToolProvider>,
    desired_dynamic: &mut DynamicStateSnapshot,
    model_catalog: &CachedModelCatalog,
) -> Result<(), String> {
    let db_path = session_log::sessions_dir().join(session_filename);

    if !db_path.exists() {
        push_history_system_message(history, repl_reset_message());
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
        *history = live.state.messages.clone();
        app.blocks = blocks_from_transcript(&live.state.messages, &live.state.tool_calls);
        apply_ui_resume_state_to_blocks(&mut app.blocks, &live.ui_state);
        app.last_response_usage = live.ui_state.last_response_usage.clone();
        app.plugin_mode_indicators = live.ui_state.plugin_mode_indicators.clone();
        app.streaming_output = live.ui_state.streaming_output.clone();
        app.streaming_output_hidden = live.ui_state.streaming_output_hidden;
        app.streaming_output_partial = live.ui_state.streaming_output_partial.clone();
        app.invalidate_height_cache();
        app.blocks.push(DisplayBlock::SystemMessage(
            "Interrupted runtime state restored from a live snapshot.".to_string(),
        ));

        let _ = dynamic_tools.apply_state(live.dynamic_state.clone());
        *desired_dynamic = dynamic_tools.export_state();

        *turn_counter = live.state.iteration;
        app.token_usage = live.state.token_usage.clone();
        app.last_prompt_usage = live.state.last_prompt_usage.clone();

        let replay_manifest = live.state.replay_manifest.clone();
        if let Some(restored_model) = replay_manifest
            .as_ref()
            .and_then(|m| m.get("configured_model"))
            .and_then(|m| m.as_str())
            .filter(|model| !model.is_empty())
        {
            provider
                .validate_model_name(restored_model)
                .map_err(|err| {
                    format!(
                        "Cannot resume session with model `{}`: {}",
                        restored_model, err
                    )
                })?;
            let restored_context_window = replay_manifest
                .as_ref()
                .and_then(|m| m.get("context_window"))
                .and_then(|m| m.as_u64())
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
                    None,
                )
                .await;
            }
        }

        *current_model_variant = replay_manifest
            .as_ref()
            .and_then(|m| m.get("model_variant").or_else(|| m.get("reasoning_effort")))
            .and_then(|m| m.as_str())
            .map(str::to_string)
            .or_else(|| {
                provider
                    .default_model_variant(&app.model)
                    .map(str::to_string)
            });
        app.set_model_variant(current_model_variant.clone());

        let requested_execution_mode = replay_manifest
            .as_ref()
            .and_then(|m| m.get("execution_mode"))
            .and_then(|m| m.as_str())
            .map(str::to_string)
            .and_then(|raw| crate::parse_execution_mode(&raw).ok());
        let restored_execution_mode = requested_execution_mode
            .and_then(|mode| crate::ensure_supported_execution_mode(mode).ok())
            .unwrap_or(live.state.policy.execution_mode);
        let restored_context_strategy = live.state.policy.context_strategy;
        *execution_mode = restored_execution_mode;
        *context_strategy = restored_context_strategy;
        if matches!(requested_execution_mode, Some(ExecutionMode::Repl))
            && !lash::execution_mode_supported(ExecutionMode::Repl)
        {
            app.blocks.push(DisplayBlock::SystemMessage(
                "This build does not support REPL mode; resuming in `standard`.".to_string(),
            ));
        }

        if matches!(restored_execution_mode, ExecutionMode::Repl) {
            if let Some(ref repl_snapshot) = live.state.repl_snapshot {
                if let Some(rt) = runtime.as_mut() {
                    rt.update_session_config(
                        None,
                        None,
                        None,
                        None,
                        Some(restored_context_strategy),
                    )
                    .await;
                    match rt.restore_repl(repl_snapshot).await {
                        Ok(()) => {
                            app.blocks.push(DisplayBlock::SystemMessage(
                                "REPL state restored from snapshot.".to_string(),
                            ));
                        }
                        Err(e) => {
                            push_history_system_message(
                                history,
                                format!(
                                    "Session resumed but REPL restore failed ({}). Re-import modules and recreate any state you need.",
                                    e
                                ),
                            );
                        }
                    }
                }
            } else {
                push_history_system_message(history, repl_reset_message());
            }
        }

        if let Some(rt) = runtime.as_mut() {
            rt.update_session_config(
                None,
                None,
                Some(current_model_variant.clone()),
                None,
                Some(restored_context_strategy),
            )
            .await;
            let _ = rt.refresh_session_execution_surface().await;
            let mut restored_state = live.state.clone();
            restored_state.policy.execution_mode = restored_execution_mode;
            restored_state.policy.context_strategy = restored_context_strategy;
            restored_state.task_state = None;
            rt.set_state(restored_state);
        }
        return Ok(());
    }

    if let Some(state) = resume_store.load_session_state() {
        let transcript_entries = resume_store.transcript_load();
        *history = lash::transcript_messages(&transcript_entries);
        let config_value = serde_json::from_str::<serde_json::Value>(&state.config_json).ok();
        let resumed_live_snapshot = config_value
            .as_ref()
            .and_then(|value| value.get("task_state"))
            .and_then(|value| value.get("kind"))
            .and_then(|value| value.as_str())
            == Some("live_resume");
        if resumed_live_snapshot {
            app.blocks.push(DisplayBlock::SystemMessage(
                "Interrupted runtime state restored from a live snapshot.".to_string(),
            ));
        }
        if let Some(dynamic_state) = config_value
            .as_ref()
            .and_then(|v| v.get("dynamic_state").cloned())
            .and_then(|v| serde_json::from_value::<DynamicStateSnapshot>(v).ok())
        {
            let _ = dynamic_tools.apply_state(dynamic_state);
            *desired_dynamic = dynamic_tools.export_state();
        }
        *turn_counter = state.iteration.max(0) as usize;
        if let Some(restored_model) = config_value
            .as_ref()
            .and_then(|v| {
                v.get("manifest")
                    .and_then(|m| m.get("configured_model"))
                    .and_then(|m| m.as_str())
            })
            .filter(|model| !model.is_empty())
        {
            provider
                .validate_model_name(restored_model)
                .map_err(|err| {
                    format!(
                        "Cannot resume session with model `{}`: {}",
                        restored_model, err
                    )
                })?;
            let restored_context_window = config_value
                .as_ref()
                .and_then(|v| {
                    v.get("manifest")
                        .and_then(|m| m.get("context_window"))
                        .and_then(|m| m.as_u64())
                })
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
                    None,
                )
                .await;
            }
        }
        *current_model_variant = config_value
            .as_ref()
            .and_then(|v| {
                v.get("manifest")
                    .and_then(|m| m.get("model_variant").or_else(|| m.get("reasoning_effort")))
                    .and_then(|m| m.as_str())
                    .map(str::to_string)
            })
            .or_else(|| {
                provider
                    .default_model_variant(&app.model)
                    .map(str::to_string)
            });
        app.set_model_variant(current_model_variant.clone());
        let requested_execution_mode = config_value
            .as_ref()
            .and_then(|v| {
                v.get("manifest")
                    .and_then(|m| m.get("execution_mode"))
                    .and_then(|m| m.as_str())
                    .map(str::to_string)
            })
            .and_then(|raw| crate::parse_execution_mode(&raw).ok());
        let restored_execution_mode = requested_execution_mode
            .and_then(|mode| crate::ensure_supported_execution_mode(mode).ok())
            .unwrap_or_else(lash::default_execution_mode);
        let restored_context_strategy = config_value
            .as_ref()
            .and_then(|v| v.get("context_strategy").cloned())
            .and_then(|v| serde_json::from_value::<ContextStrategy>(v).ok())
            .unwrap_or_else(lash::default_context_strategy)
            .validate()
            .unwrap_or_else(|_| lash::default_context_strategy());
        *execution_mode = restored_execution_mode;
        *context_strategy = restored_context_strategy;
        if matches!(requested_execution_mode, Some(ExecutionMode::Repl))
            && !lash::execution_mode_supported(ExecutionMode::Repl)
        {
            app.blocks.push(DisplayBlock::SystemMessage(
                "This build does not support REPL mode; resuming in `standard`.".to_string(),
            ));
        }
        if let Some(usage) = restored_token_usage(&state) {
            app.token_usage = usage;
        }

        if matches!(restored_execution_mode, ExecutionMode::Repl) {
            if let Some(ref repl_snapshot) = state.repl_snapshot {
                if let Some(rt) = runtime.as_mut() {
                    rt.update_session_config(
                        None,
                        None,
                        None,
                        None,
                        Some(restored_context_strategy),
                    )
                    .await;
                    match rt.restore_repl(repl_snapshot).await {
                        Ok(()) => {
                            app.blocks.push(DisplayBlock::SystemMessage(
                                "REPL state restored from snapshot.".to_string(),
                            ));
                        }
                        Err(e) => {
                            push_history_system_message(
                                history,
                                format!(
                                    "Session resumed but REPL restore failed ({}). Re-import modules and recreate any state you need.",
                                    e
                                ),
                            );
                        }
                    }
                }
            } else {
                push_history_system_message(history, repl_reset_message());
            }
        }

        if let Some(rt) = runtime.as_mut() {
            rt.update_session_config(
                None,
                None,
                Some(current_model_variant.clone()),
                None,
                Some(restored_context_strategy),
            )
            .await;
            let _ = rt.refresh_session_execution_surface().await;
            let replay_manifest = config_value
                .as_ref()
                .and_then(|v| v.get("manifest").cloned());
            let plugin_snapshot = config_value
                .as_ref()
                .and_then(|v| v.get("plugin_snapshot").cloned())
                .and_then(|v| serde_json::from_value(v).ok());
            let last_prompt_usage = restored_last_prompt_usage(config_value.as_ref());
            let tool_calls = lash::transcript_tool_calls(&transcript_entries);
            app.last_prompt_usage = last_prompt_usage.clone();
            rt.set_state(SessionStateEnvelope {
                session_id: crate::ROOT_SESSION_ID.to_string(),
                policy: lash::SessionPolicy {
                    execution_mode: restored_execution_mode,
                    context_strategy: restored_context_strategy,
                    ..rt.export_state().policy
                },
                messages: history.clone(),
                tool_calls,
                iteration: *turn_counter,
                token_usage: app.token_usage.clone(),
                last_prompt_usage,
                task_state: None,
                replay_manifest,
                plugin_snapshot,
                repl_snapshot: state.repl_snapshot.clone(),
            });
        }
    } else if matches!(*execution_mode, ExecutionMode::Repl) {
        push_history_system_message(history, repl_reset_message());
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

    fn persist_transcript(
        store: &Store,
        messages: Vec<Message>,
        ui_state: crate::app::UiResumeState,
    ) {
        let existing = store.load_session_state();
        let config = existing
            .as_ref()
            .and_then(|state| serde_json::from_str::<serde_json::Value>(&state.config_json).ok())
            .unwrap_or_else(|| serde_json::json!({}));
        store.save_session_state(lash::SessionState {
            iteration: existing.as_ref().map(|state| state.iteration).unwrap_or(0),
            config_json: ui_resume::with_ui_resume_state(config, &ui_state).to_string(),
            repl_snapshot: existing
                .as_ref()
                .and_then(|state| state.repl_snapshot.clone()),
            input_tokens: existing
                .as_ref()
                .map(|state| state.input_tokens)
                .unwrap_or(0),
            output_tokens: existing
                .as_ref()
                .map(|state| state.output_tokens)
                .unwrap_or(0),
            cached_input_tokens: existing
                .as_ref()
                .map(|state| state.cached_input_tokens)
                .unwrap_or(0),
            reasoning_tokens: existing
                .as_ref()
                .map(|state| state.reasoning_tokens)
                .unwrap_or(0),
        });
        let keyspaces = lash::semantic_transcript_keyspaces(&messages, &[]);
        store.transcript_replace_keyspaces(&keyspaces);
    }

    #[test]
    fn restored_token_usage_includes_reasoning_tokens() {
        let state = lash::store::SessionState {
            iteration: 3,
            config_json: "{}".into(),
            repl_snapshot: None,
            input_tokens: 1200,
            output_tokens: 340,
            cached_input_tokens: 80,
            reasoning_tokens: 55,
        };

        let usage = restored_token_usage(&state).expect("usage");
        assert_eq!(usage.input_tokens, 1200);
        assert_eq!(usage.output_tokens, 340);
        assert_eq!(usage.cached_input_tokens, 80);
        assert_eq!(usage.reasoning_tokens, 55);
    }

    #[test]
    fn restored_last_prompt_usage_reads_persisted_snapshot() {
        let config = serde_json::json!({
            "last_prompt_usage": {
                "prompt_context_tokens": 4096,
                "input_tokens": 3900,
                "cached_input_tokens": 196
            }
        });

        let usage = restored_last_prompt_usage(Some(&config)).expect("prompt usage");
        assert_eq!(usage.prompt_context_tokens, 4096);
        assert_eq!(usage.input_tokens, 3900);
        assert_eq!(usage.cached_input_tokens, 196);
        assert_eq!(usage.context_budget_tokens, 4096);
    }

    #[tokio::test]
    async fn restore_session_state_restores_persisted_usage_into_app_and_runtime() {
        let _env_guard = env_lock().lock().await;
        let temp = TempDirGuard::new("lash-resume-usage");
        let _lash_home = EnvVarGuard::set("LASH_HOME", temp.path());
        let sessions_dir = lash::lash_home().join("sessions");
        std::fs::create_dir_all(&sessions_dir).expect("sessions dir");

        let db_path = sessions_dir.join("resume-usage.db");
        let store = Store::open(&db_path).expect("store");
        let config_json = serde_json::json!({
            "manifest": {
                "execution_mode": "standard"
            },
            "context_strategy": {
                "type": "rolling_context"
            },
            "last_prompt_usage": {
                "prompt_context_tokens": 4096,
                "input_tokens": 3900,
                "cached_input_tokens": 196
            }
        })
        .to_string();
        store.save_session_state(lash::SessionState {
            iteration: 7,
            config_json,
            repl_snapshot: None,
            input_tokens: 1200,
            output_tokens: 340,
            cached_input_tokens: 80,
            reasoning_tokens: 55,
        });
        persist_transcript(&store, Vec::new(), crate::app::UiResumeState::default());

        let provider = Provider::OpenAiGeneric {
            api_key: "test-key".into(),
            base_url: "https://example.invalid/v1".into(),
            options: lash::ProviderOptions::default(),
        };
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
        let mut desired_dynamic = dynamic_tools.export_state();
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

        let mut app = App::new("gpt-5".into(), "resume-usage".into());
        let mut history = Vec::new();
        let mut runtime = Some(runtime);
        let mut turn_counter = 0;
        let mut execution_mode = ExecutionMode::Standard;
        let mut context_strategy = lash::default_context_strategy();
        let mut current_model_variant = None;

        restore_session_state(
            "resume-usage.db",
            &mut history,
            &mut runtime,
            &mut app,
            &mut turn_counter,
            &mut execution_mode,
            &mut context_strategy,
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
        assert_eq!(context_strategy, lash::ContextStrategy::RollingContext);
        assert_eq!(app.token_usage.input_tokens, 1200);
        assert_eq!(app.token_usage.output_tokens, 340);
        assert_eq!(app.token_usage.cached_input_tokens, 80);
        assert_eq!(app.token_usage.reasoning_tokens, 55);
        let app_prompt_usage = app.last_prompt_usage.expect("app prompt usage");
        assert_eq!(app_prompt_usage.prompt_context_tokens, 4096);
        assert_eq!(app_prompt_usage.context_budget_tokens, 4096);

        let restored_state = runtime.expect("runtime").export_state();
        assert_eq!(restored_state.iteration, 7);
        assert_eq!(
            restored_state.policy.execution_mode,
            ExecutionMode::Standard
        );
        assert_eq!(
            restored_state.policy.context_strategy,
            lash::ContextStrategy::RollingContext
        );
        assert_eq!(restored_state.token_usage.input_tokens, 1200);
        assert_eq!(restored_state.token_usage.output_tokens, 340);
        assert_eq!(restored_state.token_usage.cached_input_tokens, 80);
        assert_eq!(restored_state.token_usage.reasoning_tokens, 55);
        let prompt_usage = restored_state.last_prompt_usage.expect("prompt usage");
        assert_eq!(prompt_usage.prompt_context_tokens, 4096);
        assert_eq!(prompt_usage.input_tokens, 3900);
        assert_eq!(prompt_usage.cached_input_tokens, 196);
        assert_eq!(prompt_usage.context_budget_tokens, 4096);
    }

    #[tokio::test]
    async fn restore_session_state_keeps_loaded_history_for_interrupted_turns() {
        let _env_guard = env_lock().lock().await;
        let temp = TempDirGuard::new("lash-resume-live-snapshot");
        let _lash_home = EnvVarGuard::set("LASH_HOME", temp.path());
        let sessions_dir = lash::lash_home().join("sessions");
        std::fs::create_dir_all(&sessions_dir).expect("sessions dir");

        let db_path = sessions_dir.join("resume-live.db");
        let store = Store::open(&db_path).expect("store");
        let live_messages = vec![Message {
            id: "m0".into(),
            role: MessageRole::User,
            parts: vec![Part {
                id: "m0.p0".into(),
                kind: PartKind::Text,
                content: "live snapshot message".into(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                prune_state: PruneState::Intact,
            }],
            origin: None,
        }];
        crate::resume_snapshot::save_live_resume_snapshot(
            &store,
            &SessionStateEnvelope {
                session_id: crate::ROOT_SESSION_ID.to_string(),
                policy: lash::SessionPolicy {
                    execution_mode: ExecutionMode::Standard,
                    context_strategy: lash::ContextStrategy::RollingContext,
                    ..lash::SessionPolicy::default()
                },
                messages: live_messages.clone(),
                tool_calls: Vec::new(),
                iteration: 2,
                token_usage: TokenUsage::default(),
                last_prompt_usage: None,
                task_state: Some(serde_json::json!({
                    "kind": "live_resume",
                    "status": "running"
                })),
                replay_manifest: Some(serde_json::json!({
                    "configured_model": "gpt-5",
                    "context_window": 200000,
                    "execution_mode": "standard"
                })),
                plugin_snapshot: None,
                repl_snapshot: None,
            },
            &crate::app::UiResumeState::default(),
            &DynamicStateSnapshot {
                base_generation: 0,
                tools: std::collections::BTreeMap::new(),
                enabled_tools: std::collections::BTreeSet::new(),
            },
        )
        .expect("live snapshot");
        persist_transcript(
            &store,
            vec![Message {
                id: "old".into(),
                role: MessageRole::User,
                parts: vec![Part {
                    id: "old.p0".into(),
                    kind: PartKind::Text,
                    content: "canonical transcript message".into(),
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: PruneState::Intact,
                }],
                origin: None,
            }],
            crate::app::UiResumeState::default(),
        );

        let provider = Provider::OpenAiGeneric {
            api_key: "test-key".into(),
            base_url: "https://example.invalid/v1".into(),
            options: lash::ProviderOptions::default(),
        };
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
            "resume_live_tools",
            Arc::new(
                move |_ctx| Ok(lash::PluginSpec::new().with_tool_provider(Arc::clone(&tools))),
            ),
        ))])
        .with_dynamic_tools()
        .build_standard_session("root", None)
        .expect("plugins");
        let dynamic_tools = plugins.dynamic_tools().expect("dynamic tools");
        let mut desired_dynamic = dynamic_tools.export_state();
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

        let mut app = App::new("gpt-5".into(), "resume-live".into());
        let mut history = vec![Message {
            id: "m0".into(),
            role: MessageRole::User,
            parts: vec![Part {
                id: "m0.p0".into(),
                kind: PartKind::Text,
                content: "canonical transcript message".into(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                prune_state: PruneState::Intact,
            }],
            origin: None,
        }];
        let mut runtime = Some(runtime);
        let mut turn_counter = 0;
        let mut execution_mode = ExecutionMode::Standard;
        let mut context_strategy = lash::default_context_strategy();
        let mut current_model_variant = None;

        restore_session_state(
            "resume-live.db",
            &mut history,
            &mut runtime,
            &mut app,
            &mut turn_counter,
            &mut execution_mode,
            &mut context_strategy,
            &provider,
            &mut current_model_variant,
            &dynamic_tools,
            &mut desired_dynamic,
            &model_catalog,
        )
        .await
        .expect("restore");

        assert_eq!(history.len(), 1);
        assert_eq!(history[0].role, MessageRole::User);
        assert_eq!(history[0].parts.len(), 1);
        assert_eq!(history[0].parts[0].content, "live snapshot message");
        let restored_runtime = runtime.expect("runtime");
        let restored_messages = restored_runtime.export_state().messages;
        assert_eq!(restored_messages.len(), 1);
        assert_eq!(restored_messages[0].role, MessageRole::User);
        assert_eq!(restored_messages[0].parts.len(), 1);
        assert_eq!(
            restored_messages[0].parts[0].content,
            "live snapshot message"
        );
        assert!(app.blocks.iter().any(|block| matches!(
            block,
            DisplayBlock::SystemMessage(msg)
                if msg == "Interrupted runtime state restored from a live snapshot."
        )));
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
        let messages = vec![Message {
            id: "m0".into(),
            role: MessageRole::User,
            parts: vec![Part {
                id: "m0.p0".into(),
                kind: PartKind::Text,
                content: "hello".into(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                prune_state: PruneState::Intact,
            }],
            origin: None,
        }];
        let config_json = serde_json::json!({
            "manifest": {
                "execution_mode": "standard"
            }
        })
        .to_string();
        store.save_session_state(lash::SessionState {
            iteration: 1,
            config_json,
            repl_snapshot: None,
            input_tokens: 0,
            output_tokens: 0,
            cached_input_tokens: 0,
            reasoning_tokens: 0,
        });
        persist_transcript(
            &store,
            messages,
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
        let mut context_strategy = lash::default_context_strategy();
        let mut current_model_variant = None;

        load_resumed_session(
            filename,
            &mut app,
            &mut history,
            &mut runtime,
            &mut turn_counter,
            &mut execution_mode,
            &mut context_strategy,
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
        assert!(app.blocks.iter().any(|block| matches!(
            block,
            DisplayBlock::SystemMessage(msg) if msg == &format!("Resumed: {}", filename)
        )));
    }
}
