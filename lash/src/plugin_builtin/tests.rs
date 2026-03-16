use std::sync::Arc;

use serde_json::json;

use super::history::HistoryTools;
use super::plan_mode::{PlanModePluginConfig, PlanModePluginFactory};
use super::plan_tracker::PlanTrackerPluginFactory;
use super::*;
use crate::agent::PromptSectionName;
use crate::instructions::InstructionSource;
use crate::plugin::{
    MessageMutatorContext, MessageMutatorHook, PluginDirective, PluginError, PromptHookContext,
    ToolCallHookContext, ToolSurfaceContext,
};
use crate::store::Store;
use crate::tools::StateToolsPluginFactory;
use crate::{
    AgentStateEnvelope, AssembledTurn, AssistantOutput, ContextFoldingConfig, DoneReason,
    ExecutionMode, MessageRole, OutputState, PluginHost, PromptUsage, SessionCreateRequest,
    SessionHandle, SessionManager, SessionSnapshot, TokenUsage, ToolDefinition, TurnHookContext,
    TurnInput, TurnResultHookContext, TurnStatus,
};

struct MockSessionManager;

struct StaticInstructionSource {
    text: String,
    read_text: String,
}

impl InstructionSource for StaticInstructionSource {
    fn system_instructions(&self) -> String {
        self.text.clone()
    }

    fn context_instructions_for_reads(&self, _read_paths: &[String]) -> String {
        self.read_text.clone()
    }
}

#[async_trait::async_trait]
impl SessionManager for MockSessionManager {
    async fn snapshot_current(&self) -> Result<SessionSnapshot, PluginError> {
        Ok(AgentStateEnvelope::default())
    }

    async fn snapshot_session(&self, _session_id: &str) -> Result<SessionSnapshot, PluginError> {
        Ok(AgentStateEnvelope::default())
    }

    async fn tool_catalog(&self, _session_id: &str) -> Result<Vec<serde_json::Value>, PluginError> {
        Ok(Vec::new())
    }

    async fn create_session(
        &self,
        request: SessionCreateRequest,
    ) -> Result<SessionHandle, PluginError> {
        Ok(SessionHandle {
            session_id: request.agent_id.unwrap_or_else(|| "child".to_string()),
            config: crate::SessionConfigSnapshot {
                provider_kind: crate::provider::ProviderKind::OpenAiGeneric,
                model: "mock-model".to_string(),
                model_variant: None,
                execution_mode: ExecutionMode::Standard,
                context_folding: ContextFoldingConfig::default(),
                context_window: None,
                max_turns: None,
                include_soul: false,
                sub_agent: false,
            },
        })
    }

    async fn close_session(&self, _session_id: &str) -> Result<(), PluginError> {
        Ok(())
    }

    async fn start_turn_stream(
        &self,
        session_id: &str,
        _input: TurnInput,
    ) -> Result<crate::plugin::SessionTurnHandle, PluginError> {
        let (_tx, rx) = tokio::sync::mpsc::channel(1);
        Ok(crate::plugin::SessionTurnHandle {
            turn_id: format!("{session_id}-turn"),
            session_id: session_id.to_string(),
            config: crate::SessionConfigSnapshot {
                provider_kind: crate::provider::ProviderKind::OpenAiGeneric,
                model: "mock-model".to_string(),
                model_variant: None,
                execution_mode: ExecutionMode::Standard,
                context_folding: ContextFoldingConfig::default(),
                context_window: None,
                max_turns: None,
                include_soul: false,
                sub_agent: false,
            },
            events: rx,
        })
    }

    async fn await_turn(&self, turn_id: &str) -> Result<AssembledTurn, PluginError> {
        Ok(empty_turn(turn_id.trim_end_matches("-turn")))
    }

    async fn cancel_turn(&self, _turn_id: &str) -> Result<(), PluginError> {
        Ok(())
    }
}

fn empty_turn(session_id: &str) -> AssembledTurn {
    AssembledTurn {
        state: AgentStateEnvelope {
            agent_id: session_id.to_string(),
            execution_mode: ExecutionMode::Standard,
            context_folding: ContextFoldingConfig::default(),
            ..Default::default()
        },
        status: TurnStatus::Completed,
        assistant_output: AssistantOutput {
            safe_text: String::new(),
            raw_text: String::new(),
            state: OutputState::Usable,
        },
        done_reason: DoneReason::ModelStop,
        execution: crate::ExecutionSummary {
            mode: ExecutionMode::Standard,
            had_tool_calls: false,
            had_code_execution: false,
        },
        token_usage: TokenUsage::default(),
        tool_calls: Vec::new(),
        code_outputs: Vec::new(),
        errors: Vec::new(),
    }
}

#[cfg(feature = "sqlite-store")]
#[tokio::test]
async fn history_plugin_emits_turn_prompt_note_only_after_compaction() {
    let store = Arc::new(Store::memory().expect("store"));
    let host = PluginHost::new(vec![Arc::new(BuiltinHistoryPluginFactory::new(store))]);
    let session = host.build_standard_session("root", None).expect("session");
    let manager: Arc<dyn SessionManager> = Arc::new(MockSessionManager);

    let empty = session
        .collect_prompt_contributions(PromptHookContext {
            session_id: "root".to_string(),
            host: Arc::clone(&manager),
            prompt: crate::PromptContext {
                tool_names: vec!["search_history".to_string()],
                ..Default::default()
            },
            state: AgentStateEnvelope::default(),
        })
        .await
        .expect("prompt");
    assert_eq!(empty.len(), 1);

    let messages = vec![
        crate::Message {
            id: "u1".to_string(),
            role: MessageRole::User,
            parts: vec![crate::Part {
                id: "u1.p0".to_string(),
                kind: crate::PartKind::Text,
                content: "oldest user".repeat(20),
                tool_call_id: None,
                tool_name: None,
                prune_state: crate::PruneState::Intact,
            }],
            origin: None,
        },
        crate::Message {
            id: "a1".to_string(),
            role: MessageRole::Assistant,
            parts: vec![crate::Part {
                id: "a1.p0".to_string(),
                kind: crate::PartKind::Text,
                content: "oldest assistant".repeat(20),
                tool_call_id: None,
                tool_name: None,
                prune_state: crate::PruneState::Intact,
            }],
            origin: None,
        },
        crate::Message {
            id: "u2".to_string(),
            role: MessageRole::User,
            parts: vec![crate::Part {
                id: "u2.p0".to_string(),
                kind: crate::PartKind::Text,
                content: "older user".repeat(20),
                tool_call_id: None,
                tool_name: None,
                prune_state: crate::PruneState::Intact,
            }],
            origin: None,
        },
        crate::Message {
            id: "a2".to_string(),
            role: MessageRole::Assistant,
            parts: vec![crate::Part {
                id: "a2.p0".to_string(),
                kind: crate::PartKind::Text,
                content: "older assistant".repeat(20),
                tool_call_id: None,
                tool_name: None,
                prune_state: crate::PruneState::Intact,
            }],
            origin: None,
        },
        crate::Message {
            id: "u3".to_string(),
            role: MessageRole::User,
            parts: vec![crate::Part {
                id: "u3.p0".to_string(),
                kind: crate::PartKind::Text,
                content: "recent user".repeat(20),
                tool_call_id: None,
                tool_name: None,
                prune_state: crate::PruneState::Intact,
            }],
            origin: None,
        },
        crate::Message {
            id: "a3".to_string(),
            role: MessageRole::Assistant,
            parts: vec![crate::Part {
                id: "a3.p0".to_string(),
                kind: crate::PartKind::Text,
                content: "recent assistant".repeat(20),
                tool_call_id: None,
                tool_name: None,
                prune_state: crate::PruneState::Intact,
            }],
            origin: None,
        },
        crate::Message {
            id: "u4".to_string(),
            role: MessageRole::User,
            parts: vec![crate::Part {
                id: "u4.p0".to_string(),
                kind: crate::PartKind::Text,
                content: "latest user".repeat(20),
                tool_call_id: None,
                tool_name: None,
                prune_state: crate::PruneState::Intact,
            }],
            origin: None,
        },
        crate::Message {
            id: "a4".to_string(),
            role: MessageRole::Assistant,
            parts: vec![crate::Part {
                id: "a4.p0".to_string(),
                kind: crate::PartKind::Text,
                content: "latest assistant".repeat(20),
                tool_call_id: None,
                tool_name: None,
                prune_state: crate::PruneState::Intact,
            }],
            origin: None,
        },
    ];
    let compacted = session
        .mutate_messages(
            MessageMutatorContext {
                hook: MessageMutatorHook::AfterTokenCount,
                session_id: "root".to_string(),
                state: AgentStateEnvelope::default(),
                host: Arc::clone(&manager),
                turn: None,
                prompt_usage: Some(PromptUsage {
                    prompt_context_tokens: 70,
                    input_tokens: 70,
                    cached_input_tokens: 0,
                    context_budget_tokens: 70,
                }),
                max_context_tokens: Some(100),
                context_folding: Some(ContextFoldingConfig::default()),
            },
            messages,
        )
        .await
        .expect("mutate messages");
    assert!(!compacted.iter().any(|message| {
        message
            .parts
            .iter()
            .any(|part| part.content.contains("oldest user"))
    }));

    let contributions = session
        .collect_prompt_contributions(PromptHookContext {
            session_id: "root".to_string(),
            host: manager,
            prompt: crate::PromptContext {
                tool_names: vec!["search_history".to_string()],
                ..Default::default()
            },
            state: AgentStateEnvelope::default(),
        })
        .await
        .expect("prompt");
    assert_eq!(contributions.len(), 2);
    assert!(
        contributions
            .iter()
            .any(|contribution| contribution.content.contains("Older turns were archived"))
    );
}

#[cfg(feature = "sqlite-store")]
#[tokio::test]
async fn history_plugin_hides_search_history_until_context_is_archived() {
    let store = Arc::new(Store::memory().expect("store"));
    let host = PluginHost::new(vec![Arc::new(BuiltinHistoryPluginFactory::new(store))]);
    let session = host.build_standard_session("root", None).expect("session");
    let manager: Arc<dyn SessionManager> = Arc::new(MockSessionManager);
    let defs = session
        .tools()
        .definitions()
        .into_iter()
        .filter(|def| def.name == "search_history")
        .collect::<Vec<_>>();
    assert_eq!(defs.len(), 1);
    assert!(!defs[0].enabled);
    assert!(!defs[0].injected);

    let surface = session
        .resolve_tool_surface(ToolSurfaceContext {
            session_id: "root".to_string(),
            mode: ExecutionMode::Standard,
            tools: defs.clone(),
        })
        .expect("tool surface");
    assert_eq!(surface.tools.len(), 1);
    assert!(!surface.tools[0].injected);
    assert!(!surface.tools[0].enabled);

    let archived_result = session
        .mutate_messages(
            MessageMutatorContext {
                hook: MessageMutatorHook::AfterTokenCount,
                session_id: "root".to_string(),
                state: AgentStateEnvelope::default(),
                host: Arc::clone(&manager),
                turn: None,
                prompt_usage: Some(PromptUsage {
                    prompt_context_tokens: 70,
                    input_tokens: 70,
                    cached_input_tokens: 0,
                    context_budget_tokens: 70,
                }),
                max_context_tokens: Some(100),
                context_folding: Some(ContextFoldingConfig::default()),
            },
            vec![
                crate::Message {
                    id: "u1".to_string(),
                    role: MessageRole::User,
                    parts: vec![crate::Part {
                        id: "u1.p0".to_string(),
                        kind: crate::PartKind::Text,
                        content: "oldest user".repeat(20),
                        tool_call_id: None,
                        tool_name: None,
                        prune_state: crate::PruneState::Intact,
                    }],
                    origin: None,
                },
                crate::Message {
                    id: "a1".to_string(),
                    role: MessageRole::Assistant,
                    parts: vec![crate::Part {
                        id: "a1.p0".to_string(),
                        kind: crate::PartKind::Text,
                        content: "oldest assistant".repeat(20),
                        tool_call_id: None,
                        tool_name: None,
                        prune_state: crate::PruneState::Intact,
                    }],
                    origin: None,
                },
                crate::Message {
                    id: "u2".to_string(),
                    role: MessageRole::User,
                    parts: vec![crate::Part {
                        id: "u2.p0".to_string(),
                        kind: crate::PartKind::Text,
                        content: "older user".repeat(20),
                        tool_call_id: None,
                        tool_name: None,
                        prune_state: crate::PruneState::Intact,
                    }],
                    origin: None,
                },
                crate::Message {
                    id: "a2".to_string(),
                    role: MessageRole::Assistant,
                    parts: vec![crate::Part {
                        id: "a2.p0".to_string(),
                        kind: crate::PartKind::Text,
                        content: "older assistant".repeat(20),
                        tool_call_id: None,
                        tool_name: None,
                        prune_state: crate::PruneState::Intact,
                    }],
                    origin: None,
                },
                crate::Message {
                    id: "u3".to_string(),
                    role: MessageRole::User,
                    parts: vec![crate::Part {
                        id: "u3.p0".to_string(),
                        kind: crate::PartKind::Text,
                        content: "recent user".repeat(20),
                        tool_call_id: None,
                        tool_name: None,
                        prune_state: crate::PruneState::Intact,
                    }],
                    origin: None,
                },
                crate::Message {
                    id: "a3".to_string(),
                    role: MessageRole::Assistant,
                    parts: vec![crate::Part {
                        id: "a3.p0".to_string(),
                        kind: crate::PartKind::Text,
                        content: "recent assistant".repeat(20),
                        tool_call_id: None,
                        tool_name: None,
                        prune_state: crate::PruneState::Intact,
                    }],
                    origin: None,
                },
                crate::Message {
                    id: "u4".to_string(),
                    role: MessageRole::User,
                    parts: vec![crate::Part {
                        id: "u4.p0".to_string(),
                        kind: crate::PartKind::Text,
                        content: "latest user".repeat(20),
                        tool_call_id: None,
                        tool_name: None,
                        prune_state: crate::PruneState::Intact,
                    }],
                    origin: None,
                },
                crate::Message {
                    id: "a4".to_string(),
                    role: MessageRole::Assistant,
                    parts: vec![crate::Part {
                        id: "a4.p0".to_string(),
                        kind: crate::PartKind::Text,
                        content: "latest assistant".repeat(20),
                        tool_call_id: None,
                        tool_name: None,
                        prune_state: crate::PruneState::Intact,
                    }],
                    origin: None,
                },
            ],
        )
        .await
        .expect("mutate messages");
    assert!(!archived_result.is_empty());

    let surface = session
        .resolve_tool_surface(ToolSurfaceContext {
            session_id: "root".to_string(),
            mode: ExecutionMode::Standard,
            tools: defs,
        })
        .expect("tool surface");
    assert_eq!(surface.tools.len(), 1);
    assert!(surface.tools[0].injected);
    assert!(surface.tools[0].enabled);
}

#[cfg(feature = "sqlite-store")]
#[test]
fn search_history_lists_all_without_query_in_stable_turn_order() {
    let store = Arc::new(Store::memory().expect("store"));
    store.history_add_turn(
        "root",
        &json!({
            "index": 2,
            "user_message": "second",
            "prose": "",
            "code": "",
            "output": "",
            "tool_calls": [],
            "files_read": [],
            "files_written": [],
        }),
    );
    store.history_add_turn(
        "root",
        &json!({
            "index": 1,
            "user_message": "first",
            "prose": "",
            "code": "",
            "output": "",
            "tool_calls": [],
            "files_read": [],
            "files_written": [],
        }),
    );
    let tools = HistoryTools::new(store);

    let result = tools.search_history(&json!({ "__agent_id__": "root" }));

    assert!(result.success);
    let items = result.result.as_array().cloned().unwrap_or_default();
    assert_eq!(items.len(), 2);
    assert_eq!(items[0]["turn"], 1);
    assert_eq!(items[1]["turn"], 2);
    assert!(items[0].get("score").is_none());
    assert!(items[0].get("field_hits").is_none());
}

#[tokio::test]
async fn prompt_context_plugin_contributes_environment_and_project_instruction_sections() {
    let host = PluginHost::new(vec![Arc::new(BuiltinPromptContextPluginFactory::new(
        Arc::new(StaticInstructionSource {
            text: "Repo rules".to_string(),
            read_text: String::new(),
        }),
        PromptContextPluginConfig::default(),
    ))]);
    let session = host.build_standard_session("root", None).expect("session");
    let contributions = session
        .collect_prompt_contributions(PromptHookContext {
            session_id: "root".to_string(),
            host: Arc::new(MockSessionManager),
            prompt: crate::PromptContext::default(),
            state: AgentStateEnvelope::default(),
        })
        .await
        .expect("prompt contributions");

    assert!(contributions.iter().any(|contribution| {
        contribution.section == PromptSectionName::Environment
            && contribution.content.contains("Working directory:")
    }));
    assert!(contributions.iter().any(|contribution| {
        contribution.section == PromptSectionName::Guidance
            && contribution.content == "### Project Instructions\nRepo rules"
    }));
}

#[test]
fn repl_tool_surface_plugin_shapes_search_surface_and_omitted_tool_note() {
    let host = PluginHost::new(vec![Arc::new(StateToolsPluginFactory::new())]);
    let session = host.build_standard_session("root", None).expect("session");

    let surface = session
        .resolve_tool_surface(ToolSurfaceContext {
            session_id: "root".to_string(),
            mode: ExecutionMode::Repl,
            tools: vec![
                ToolDefinition {
                    name: "search_tools".to_string(),
                    description: "Discover tools".to_string(),
                    params: vec![],
                    returns: "list".to_string(),
                    examples: vec![],
                    enabled: true,
                    injected: false,
                },
                ToolDefinition {
                    name: "read_file".to_string(),
                    description: "Read files".to_string(),
                    params: vec![],
                    returns: "str".to_string(),
                    examples: vec![],
                    enabled: true,
                    injected: true,
                },
                ToolDefinition {
                    name: "apply_patch".to_string(),
                    description: "Apply patches".to_string(),
                    params: vec![],
                    returns: "str".to_string(),
                    examples: vec![],
                    enabled: true,
                    injected: false,
                },
            ],
        })
        .expect("tool surface");

    assert!(
        surface
            .tools
            .iter()
            .any(|tool| tool.name == "search_tools" && tool.enabled && !tool.injected)
    );
    assert!(surface.tool_list_notes.iter().any(|note| {
        note.contains("additional tool(s) are available but omitted from this prompt")
    }));
}

#[test]
fn repl_tool_surface_plugin_hides_search_tools_when_nothing_is_omitted() {
    let host = PluginHost::new(vec![Arc::new(StateToolsPluginFactory::new())]);
    let session = host.build_standard_session("root", None).expect("session");

    let surface = session
        .resolve_tool_surface(ToolSurfaceContext {
            session_id: "root".to_string(),
            mode: ExecutionMode::Repl,
            tools: vec![
                ToolDefinition {
                    name: "search_tools".to_string(),
                    description: "Discover tools".to_string(),
                    params: vec![],
                    returns: "list".to_string(),
                    examples: vec![],
                    enabled: true,
                    injected: false,
                },
                ToolDefinition {
                    name: "read_file".to_string(),
                    description: "Read files".to_string(),
                    params: vec![],
                    returns: "str".to_string(),
                    examples: vec![],
                    enabled: true,
                    injected: true,
                },
            ],
        })
        .expect("tool surface");

    assert_eq!(surface.tools.len(), 2);
    assert!(
        surface
            .tools
            .iter()
            .any(|tool| tool.name == "search_tools" && !tool.enabled && !tool.injected)
    );
    assert!(surface.tool_list_notes.is_empty());
}

#[tokio::test]
async fn plan_tracker_plugin_registers_update_plan_and_restores_state() {
    let host = PluginHost::new(vec![Arc::new(PlanTrackerPluginFactory)]);
    let session = host.build_standard_session("root", None).expect("session");
    let tracker = session.tools();

    let result = tracker
        .execute(
            "update_plan",
            &json!({
                "explanation": "Mapped the runtime/plugin seam.",
                "plan": [
                    {"step":"Inspect planning hooks","status":"completed"},
                    {"step":"Split plugin ownership","status":"in_progress"}
                ]
            }),
        )
        .await;
    assert!(result.success);
    assert_eq!(result.result.as_str(), Some("Plan updated"));

    let snapshot = session.snapshot().expect("snapshot");
    let restored = host
        .build_standard_session("restored", Some(&snapshot))
        .expect("restored");
    let restored_tracker = restored.tools();
    let second_result = restored_tracker
        .execute(
            "update_plan",
            &json!({
                "plan": [
                    {"step":"Inspect planning hooks","status":"completed"},
                    {"step":"Split plugin ownership","status":"completed"}
                ]
            }),
        )
        .await;
    assert!(second_result.success);
}

#[tokio::test]
async fn plan_mode_plugin_toggle_and_status_round_trip() {
    let host = PluginHost::new(vec![Arc::new(PlanModePluginFactory::default())]);
    let session = host.build_standard_session("root", None).expect("session");
    let manager: Arc<dyn SessionManager> = Arc::new(MockSessionManager);

    let status = session
        .invoke_external(
            "plan_mode.status",
            json!({}),
            None,
            true,
            Arc::clone(&manager),
        )
        .await
        .expect("status");
    assert!(status.success);
    assert_eq!(
        status.result.get("enabled").and_then(|v| v.as_bool()),
        Some(false)
    );

    let enabled = session
        .invoke_external(
            "plan_mode.enable",
            json!({}),
            None,
            true,
            Arc::clone(&manager),
        )
        .await
        .expect("enable");
    assert!(enabled.success);
    assert_eq!(
        enabled.result.get("enabled").and_then(|v| v.as_bool()),
        Some(true)
    );

    let snapshot = session.snapshot().expect("snapshot");
    let restored = host
        .build_standard_session("restored", Some(&snapshot))
        .expect("restored");
    let restored_status = restored
        .invoke_external("plan_mode.status", json!({}), None, true, manager)
        .await
        .expect("status");
    assert_eq!(
        restored_status
            .result
            .get("enabled")
            .and_then(|v| v.as_bool()),
        Some(true)
    );

    restored
        .restore(&crate::PluginSessionSnapshot::default())
        .expect("reset restore");
    let reset_status = restored
        .invoke_external(
            "plan_mode.status",
            json!({}),
            None,
            true,
            Arc::new(MockSessionManager),
        )
        .await
        .expect("status");
    assert_eq!(
        reset_status.result.get("enabled").and_then(|v| v.as_bool()),
        Some(false)
    );
}

#[tokio::test]
async fn plan_mode_plugin_injects_guidance_and_blocks_implementation_tools() {
    let host = PluginHost::new(vec![Arc::new(PlanModePluginFactory::default())]);
    let session = host.build_standard_session("root", None).expect("session");
    let manager: Arc<dyn SessionManager> = Arc::new(MockSessionManager);

    session
        .invoke_external(
            "plan_mode.enable",
            json!({}),
            None,
            true,
            Arc::clone(&manager),
        )
        .await
        .expect("enable");

    let before_turn = session
        .before_turn(TurnHookContext {
            session_id: "root".to_string(),
            state: AgentStateEnvelope::default(),
            host: Arc::clone(&manager),
        })
        .await
        .expect("before_turn");
    assert!(before_turn.iter().any(|emitted| matches!(
        &emitted.value,
        PluginDirective::EnqueueMessages { messages }
            if messages.iter().any(|message|
                message.role == MessageRole::System
                    && message.content.contains("Plan Mode (Conversational)")
                    && message.content.contains("PHASE 1 - Ground in the environment"))
    )));
    assert!(before_turn.iter().any(|emitted| {
        emitted.plugin_id == "plan_mode"
            && matches!(
                &emitted.value,
                PluginDirective::EmitEvents { events }
                    if events.iter().any(|event| matches!(
                        event,
                        crate::plugin::PluginSurfaceEvent::ModeIndicatorUpsert { label, .. }
                            if label == "plan"
                    ))
            )
    }));

    let allowed_exec = session
        .before_tool_call(ToolCallHookContext {
            session_id: "root".to_string(),
            tool_name: "exec_command".to_string(),
            args: json!({"cmd":"cargo test -q"}),
            host: Arc::clone(&manager),
        })
        .await
        .expect("before_tool_call");
    assert!(!allowed_exec.is_empty());

    let allowed = session
        .before_tool_call(ToolCallHookContext {
            session_id: "root".to_string(),
            tool_name: "read_file".to_string(),
            args: json!({"path":"src/main.rs"}),
            host: Arc::clone(&manager),
        })
        .await
        .expect("before_tool_call");
    assert!(allowed.is_empty());

    let surface = session
        .resolve_tool_surface(ToolSurfaceContext {
            session_id: "root".to_string(),
            mode: ExecutionMode::Standard,
            tools: vec![
                ToolDefinition {
                    name: "search_tools".to_string(),
                    description: "Discover tools".to_string(),
                    params: vec![],
                    returns: "list".to_string(),
                    examples: vec![],
                    enabled: true,
                    injected: false,
                },
                ToolDefinition {
                    name: "update_plan".to_string(),
                    description: "Update plan".to_string(),
                    params: vec![],
                    returns: "str".to_string(),
                    examples: vec![],
                    enabled: true,
                    injected: true,
                },
                ToolDefinition {
                    name: "read_file".to_string(),
                    description: "Read files".to_string(),
                    params: vec![],
                    returns: "str".to_string(),
                    examples: vec![],
                    enabled: true,
                    injected: true,
                },
            ],
        })
        .expect("tool surface");
    assert!(
        surface
            .tools
            .iter()
            .find(|tool| tool.name == "update_plan")
            .is_some_and(|tool| !tool.enabled && !tool.injected)
    );

    let checklist_blocked = session
        .before_tool_call(ToolCallHookContext {
            session_id: "root".to_string(),
            tool_name: "update_plan".to_string(),
            args: json!({
                "plan": [{"step":"Inspect planning hooks","status":"in_progress"}]
            }),
            host: Arc::clone(&manager),
        })
        .await
        .expect("before_tool_call");
    assert!(checklist_blocked.iter().any(|emitted| matches!(
        &emitted.value,
        PluginDirective::AbortTurn { code, .. } if code == "plan_mode_tool_blocked"
    )));

    let read_allowed = session
        .before_tool_call(ToolCallHookContext {
            session_id: "root".to_string(),
            tool_name: "read_file".to_string(),
            args: json!({
                "path":"src/main.rs"
            }),
            host: Arc::clone(&manager),
        })
        .await
        .expect("before_tool_call");
    assert!(read_allowed.is_empty());

    session
        .invoke_external(
            "plan_mode.disable",
            json!({}),
            None,
            true,
            Arc::clone(&manager),
        )
        .await
        .expect("disable");
    session
        .invoke_external(
            "plan_mode.enable",
            json!({}),
            None,
            true,
            Arc::clone(&manager),
        )
        .await
        .expect("enable");

    let checkpoint = session
        .at_checkpoint(crate::CheckpointHookContext {
            session_id: "root".to_string(),
            checkpoint: crate::CheckpointKind::AfterWork,
            state: AgentStateEnvelope::default(),
            host: Arc::clone(&manager),
        })
        .await
        .expect("checkpoint");
    assert!(checkpoint.iter().any(|emitted| matches!(
        &emitted.value,
        PluginDirective::EnqueueMessages { messages }
            if messages.iter().any(|message|
                message.content.contains("Plan Mode (Conversational)")
                    && message.content.contains("Finalization rule"))
    )));
    assert!(checkpoint.iter().any(|emitted| matches!(
        &emitted.value,
        PluginDirective::EnqueueMessages { messages }
            if messages.iter().any(|message| message.content.contains("<proposed_plan>"))
    )));

    session
        .after_turn(TurnResultHookContext {
            session_id: "root".to_string(),
            turn: empty_turn("root"),
            host: manager,
        })
        .await
        .expect("after_turn");
}

#[tokio::test]
async fn plan_mode_plugin_uses_configured_blocked_tool_set() {
    let host = PluginHost::new(vec![Arc::new(PlanModePluginFactory::new(
        PlanModePluginConfig::default().with_blocked_tools(["read_file"]),
    ))]);
    let session = host.build_standard_session("root", None).expect("session");
    let manager: Arc<dyn SessionManager> = Arc::new(MockSessionManager);

    session
        .invoke_external(
            "plan_mode.enable",
            json!({}),
            None,
            true,
            Arc::clone(&manager),
        )
        .await
        .expect("enable");

    let blocked = session
        .before_tool_call(ToolCallHookContext {
            session_id: "root".to_string(),
            tool_name: "read_file".to_string(),
            args: json!({"path":"src/main.rs"}),
            host: Arc::clone(&manager),
        })
        .await
        .expect("before_tool_call");
    assert!(blocked.iter().any(|emitted| matches!(
        &emitted.value,
        PluginDirective::AbortTurn { code, .. } if code == "plan_mode_tool_blocked"
    )));

    let allowed = session
        .before_tool_call(ToolCallHookContext {
            session_id: "root".to_string(),
            tool_name: "update_plan".to_string(),
            args: json!({
                "plan": [{"step":"Inspect planning hooks","status":"in_progress"}]
            }),
            host: Arc::clone(&manager),
        })
        .await
        .expect("before_tool_call");
    assert!(allowed.is_empty());
}

#[tokio::test]
async fn plan_mode_plugin_suppresses_proposed_plan_tags_and_emits_panel_events() {
    let host = PluginHost::new(vec![Arc::new(PlanModePluginFactory::default())]);
    let session = host.build_standard_session("root", None).expect("session");
    let manager: Arc<dyn SessionManager> = Arc::new(MockSessionManager);

    session
        .invoke_external(
            "plan_mode.enable",
            json!({}),
            None,
            true,
            Arc::clone(&manager),
        )
        .await
        .expect("enable");
    session
        .before_turn(TurnHookContext {
            session_id: "root".to_string(),
            state: AgentStateEnvelope::default(),
            host: Arc::clone(&manager),
        })
        .await
        .expect("before_turn");

    let stream = session
        .transform_assistant_stream(
            "root",
            "Start\n<proposed_plan>\n- Step one\n".to_string(),
            Arc::clone(&manager),
        )
        .await
        .expect("stream");
    assert_eq!(stream.len(), 1);
    assert_eq!(stream[0].plugin_id, "plan_mode");
    assert_eq!(stream[0].value.chunk, "Start\n");
    assert!(stream[0].value.events.iter().any(|event| matches!(
        event,
        crate::plugin::PluginSurfaceEvent::PanelUpsert { title, content, .. }
            if title == "PROPOSED PLAN" && content.contains("Step one")
    )));

    let response = session
        .transform_assistant_response(
            "root",
            crate::llm::types::LlmResponse {
                full_text: "Start\n<proposed_plan>\n- Step one\n</proposed_plan>\nDone.".into(),
                deltas: Vec::new(),
                parts: vec![crate::llm::types::LlmOutputPart::Text {
                    text: "Start\n<proposed_plan>\n- Step one\n</proposed_plan>\nDone.".into(),
                }],
                usage: crate::llm::types::LlmUsage::default(),
                request_body: None,
                http_summary: None,
            },
            manager,
        )
        .await
        .expect("response");
    assert_eq!(response.len(), 1);
    assert_eq!(response[0].value.response.full_text, "Start\n\nDone.");
}
