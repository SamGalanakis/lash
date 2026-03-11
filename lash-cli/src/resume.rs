use std::sync::Arc;

use lash_core::agent::{Message, MessageRole, Part, PartKind, PruneState};
use lash_core::{
    AgentStateEnvelope, ContextFoldingConfig, DynamicStateSnapshot, DynamicToolProvider,
    ExecutionMode, LashRuntime, Provider, Store,
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

#[allow(clippy::too_many_arguments)]
pub async fn load_resumed_session(
    filename: &str,
    app: &mut App,
    history: &mut Vec<Message>,
    runtime: &mut Option<LashRuntime>,
    turn_counter: &mut usize,
    execution_mode: &mut ExecutionMode,
    provider: &Provider,
    current_reasoning_effort: &mut Option<String>,
    dynamic_tools: &Arc<DynamicToolProvider>,
    desired_dynamic: &mut DynamicStateSnapshot,
) -> Result<(), String> {
    let Some(loaded) = session_log::load_session(filename) else {
        return Err(format!("Could not load: {}", filename));
    };
    *history = loaded.messages;
    app.blocks = loaded.blocks;
    app.last_response_usage = loaded.last_token_usage;
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
        provider,
        current_reasoning_effort,
        dynamic_tools,
        desired_dynamic,
    )
    .await;
    app.invalidate_height_cache();
    app.scroll_to_bottom();
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
    provider: &Provider,
    current_reasoning_effort: &mut Option<String>,
    dynamic_tools: &Arc<DynamicToolProvider>,
    desired_dynamic: &mut DynamicStateSnapshot,
) {
    let stem = jsonl_filename.trim_end_matches(".jsonl");
    let db_filename = format!("{}.db", stem);
    let db_path = session_log::sessions_dir().join(&db_filename);

    if !db_path.exists() {
        push_history_system_message(history, repl_reset_message());
        return;
    }

    let resume_store = match Store::open(&db_path) {
        Ok(s) => s,
        Err(_) => {
            app.blocks.push(DisplayBlock::SystemMessage(
                "Could not open session database.".to_string(),
            ));
            return;
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
            app.model = restored_model.to_string();
            app.context_window = provider.context_window(&app.model);
            if let Some(rt) = runtime.as_mut() {
                rt.set_model(app.model.clone());
            }
        }
        *current_reasoning_effort = config_value
            .as_ref()
            .and_then(|v| {
                v.get("manifest")
                    .and_then(|m| m.get("reasoning_effort"))
                    .and_then(|m| m.as_str())
                    .map(str::to_string)
            })
            .or_else(|| {
                provider
                    .reasoning_effort_for_model(&app.model)
                    .map(str::to_string)
            });
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
        let restored_context_folding = config_value
            .as_ref()
            .and_then(|v| v.get("context_folding").cloned())
            .and_then(|v| serde_json::from_value::<ContextFoldingConfig>(v).ok())
            .and_then(|cfg| cfg.validate().ok())
            .unwrap_or_default();
        *execution_mode = restored_execution_mode;
        if matches!(requested_execution_mode, Some(ExecutionMode::Repl))
            && !lash_core::execution_mode_supported(ExecutionMode::Repl)
        {
            app.blocks.push(DisplayBlock::SystemMessage(
                "This build does not support REPL mode; resuming in `standard`.".to_string(),
            ));
        }
        if state.input_tokens > 0 || state.output_tokens > 0 {
            app.token_usage = lash_core::TokenUsage {
                input_tokens: state.input_tokens,
                output_tokens: state.output_tokens,
                cached_input_tokens: state.cached_input_tokens,
                reasoning_tokens: 0,
            };
        }

        if matches!(restored_execution_mode, ExecutionMode::Repl) {
            if let Some(ref repl_snapshot) = state.repl_snapshot {
                if let Some(rt) = runtime.as_mut() {
                    rt.set_context_folding(restored_context_folding);
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

        let active_subs = resume_store.list_active_agents(Some("root"));
        for sub in &active_subs {
            let prompt = serde_json::from_str::<serde_json::Value>(&sub.config_json)
                .ok()
                .and_then(|v| v.get("prompt").and_then(|p| p.as_str()).map(String::from))
                .unwrap_or_else(|| format!("sub-agent {}", sub.agent_id));

            push_history_system_message(
                history,
                format!(
                    "Sub-agent '{}' was interrupted mid-task (iteration {}). You may re-delegate if needed.",
                    prompt, sub.iteration,
                ),
            );

            resume_store.mark_agent_done(&sub.agent_id);
        }

        if !active_subs.is_empty() {
            app.blocks.push(DisplayBlock::SystemMessage(format!(
                "{} interrupted sub-agent(s) noted in context.",
                active_subs.len()
            )));
        }

        if let Some(rt) = runtime.as_mut() {
            rt.set_reasoning_effort(current_reasoning_effort.clone());
            rt.set_capabilities(lash_core::agent_capabilities_from_profile(
                &dynamic_tools.profile(),
            ));
            let _ = rt
                .reconfigure_session(
                    dynamic_tools.capabilities_payload_json(),
                    dynamic_tools.generation(),
                )
                .await;
            let replay_manifest = config_value
                .as_ref()
                .and_then(|v| v.get("manifest").cloned());
            let plugin_snapshot = config_value
                .as_ref()
                .and_then(|v| v.get("plugin_snapshot").cloned())
                .and_then(|v| serde_json::from_value(v).ok());
            rt.set_state(AgentStateEnvelope {
                agent_id: "root".to_string(),
                context_folding: restored_context_folding,
                messages: history.clone(),
                iteration: *turn_counter,
                token_usage: app.token_usage.clone(),
                execution_mode: restored_execution_mode,
                task_state: None,
                subagent_state: None,
                replay_manifest,
                plugin_snapshot,
                repl_snapshot: state.repl_snapshot.clone(),
            });
        }
    } else if matches!(*execution_mode, ExecutionMode::Repl) {
        push_history_system_message(history, repl_reset_message());
    }
}
