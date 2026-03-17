use std::sync::Arc;

use lash_core::agent::{Message, MessageRole, Part, PartKind, PruneState};
use lash_core::{
    AgentStateEnvelope, CachedModelCatalog, ContextStrategy, DynamicStateSnapshot,
    DynamicToolProvider, ExecutionMode, LashRuntime, PromptUsage, Provider, Store, TokenUsage,
};

use crate::app::{App, DisplayBlock};
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

fn restored_token_usage(state: &lash_core::store::AgentState) -> Option<TokenUsage> {
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
    let Some(loaded) = session_log::load_session(filename) else {
        return Err(format!("Could not load: {}", filename));
    };
    *history = loaded.messages;
    app.blocks = loaded.blocks;
    app.last_response_usage = loaded.last_token_usage;
    app.last_prompt_usage = None;
    app.plugin_mode_indicators = loaded.plugin_mode_indicators;
    app.blocks.push(DisplayBlock::SystemMessage(format!(
        "Resumed: {}",
        filename
    )));
    restore_agent_state(
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
    app.invalidate_height_cache();
    app.resume_follow_output();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub async fn restore_agent_state(
    jsonl_filename: &str,
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
    let stem = jsonl_filename.trim_end_matches(".jsonl");
    let db_filename = format!("{}.db", stem);
    let db_path = session_log::sessions_dir().join(&db_filename);

    if !db_path.exists() {
        push_history_system_message(history, repl_reset_message());
        return Ok(());
    }

    let resume_store = match Store::open(&db_path) {
        Ok(s) => s,
        Err(_) => {
            app.blocks.push(DisplayBlock::SystemMessage(
                "Could not open session database.".to_string(),
            ));
            return Ok(());
        }
    };

    if let Some(state) = resume_store.load_agent_state("root") {
        let config_value = serde_json::from_str::<serde_json::Value>(&state.config_json).ok();
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
            .unwrap_or_else(lash_core::default_execution_mode);
        let restored_context_strategy = config_value
            .as_ref()
            .and_then(|v| v.get("context_strategy").cloned())
            .and_then(|v| serde_json::from_value::<ContextStrategy>(v).ok())
            .unwrap_or_else(lash_core::default_context_strategy)
            .validate()
            .unwrap_or_else(|_| lash_core::default_context_strategy());
        *execution_mode = restored_execution_mode;
        *context_strategy = restored_context_strategy;
        if matches!(requested_execution_mode, Some(ExecutionMode::Repl))
            && !lash_core::execution_mode_supported(ExecutionMode::Repl)
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
            let tool_calls = serde_json::from_str(&state.tool_calls_json).unwrap_or_default();
            app.last_prompt_usage = last_prompt_usage.clone();
            rt.set_state(AgentStateEnvelope {
                agent_id: "root".to_string(),
                policy: lash_core::SessionPolicy {
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

    use lash_core::{
        MemoryModelCatalogStore, PluginHost, PluginSpecFactory, RuntimeHostConfig, RuntimeServices,
        ToolProvider,
    };

    #[test]
    fn restored_token_usage_includes_reasoning_tokens() {
        let state = lash_core::store::AgentState {
            agent_id: "root".into(),
            messages_json: "[]".into(),
            tool_calls_json: "[]".into(),
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
    async fn restore_agent_state_restores_persisted_usage_into_app_and_runtime() {
        let _env_guard = env_lock().lock().await;
        let temp = TempDirGuard::new("lash-resume-usage");
        let _lash_home = EnvVarGuard::set("LASH_HOME", temp.path());
        let sessions_dir = lash_core::lash_home().join("sessions");
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
        store.save_agent_state(lash_core::store::AgentStateSave {
            agent_id: "root",
            messages_json: "[]",
            tool_calls_json: "[]",
            iteration: 7,
            config_json: &config_json,
            repl_snapshot: None,
            input_tokens: 1200,
            output_tokens: 340,
            cached_input_tokens: 80,
            reasoning_tokens: 55,
        });

        let provider = Provider::OpenAiGeneric {
            api_key: "test-key".into(),
            base_url: "https://example.invalid/v1".into(),
            options: lash_core::ProviderOptions::default(),
        };
        struct EmptyTools;

        #[async_trait::async_trait]
        impl ToolProvider for EmptyTools {
            fn definitions(&self) -> Vec<lash_core::ToolDefinition> {
                Vec::new()
            }

            async fn execute(
                &self,
                _name: &str,
                _args: &serde_json::Value,
            ) -> lash_core::ToolResult {
                lash_core::ToolResult::err(serde_json::json!("Unknown tool"))
            }
        }

        let tools: Arc<dyn ToolProvider> = Arc::new(EmptyTools);
        let model_catalog =
            CachedModelCatalog::models_dev(Arc::new(MemoryModelCatalogStore::new(None)), None)
                .expect("catalog");
        let plugins = PluginHost::new(vec![Arc::new(PluginSpecFactory::new(
            "resume_tools",
            Arc::new(move |_ctx| {
                Ok(lash_core::PluginSpec::new().with_tool_provider(Arc::clone(&tools)))
            }),
        ))])
        .with_dynamic_tools()
        .build_standard_session("root", None)
        .expect("plugins");
        let dynamic_tools = plugins.dynamic_tools().expect("dynamic tools");
        let mut desired_dynamic = dynamic_tools.export_state();
        let runtime_services = RuntimeServices::new(plugins);
        let runtime = LashRuntime::from_state(
            lash_core::SessionPolicy {
                execution_mode: ExecutionMode::Standard,
                provider: provider.clone(),
                model: "gpt-5".into(),
                max_context_tokens: Some(200_000),
                ..lash_core::SessionPolicy::default()
            },
            RuntimeHostConfig::default(),
            runtime_services,
            AgentStateEnvelope::default(),
        )
        .await
        .expect("runtime");

        let mut app = App::new("gpt-5".into(), "resume-usage".into());
        let mut history = Vec::new();
        let mut runtime = Some(runtime);
        let mut turn_counter = 0;
        let mut execution_mode = ExecutionMode::Standard;
        let mut context_strategy = lash_core::default_context_strategy();
        let mut current_model_variant = None;

        restore_agent_state(
            "resume-usage.jsonl",
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
        assert_eq!(context_strategy, lash_core::ContextStrategy::RollingContext);
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
            lash_core::ContextStrategy::RollingContext
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
}
