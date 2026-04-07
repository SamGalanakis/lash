use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use serde_json::json;
use tokio::sync::Mutex;

use super::plan_mode::{PlanModePluginConfig, PlanModePluginFactory};
use super::plan_tracker::PlanTrackerPluginFactory;
use super::*;
use crate::instructions::InstructionSource;
use crate::plugin::{
    PluginDirective, PluginError, PromptHookContext, ToolCallHookContext, ToolSurfaceContext,
};
use crate::session_model::PromptSectionName;
use crate::tools::StateToolsPluginFactory;
use crate::{
    AssembledTurn, AssistantOutput, DoneReason, ExecutionMode, MessageRole, OutputState,
    PluginHost, SessionCreateRequest, SessionHandle, SessionManager, SessionPolicy,
    SessionSnapshot, SessionStateEnvelope, TokenUsage, ToolDefinition, TurnHookContext, TurnInput,
    TurnResultHookContext, TurnStatus,
};

struct MockSessionManager;

fn mock_snapshot(run_session_id: &str) -> SessionSnapshot {
    SessionStateEnvelope {
        session_id: "root".to_string(),
        policy: SessionPolicy {
            execution_mode: ExecutionMode::Standard,
            context_strategy: crate::ContextStrategy::RollingContext,
            session_id: Some(run_session_id.to_string()),
            ..Default::default()
        },
        ..Default::default()
    }
}

struct StaticInstructionSource {
    text: String,
    read_text: String,
}

fn plan_mode_env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

struct CurrentDirGuard {
    original: PathBuf,
}

impl CurrentDirGuard {
    fn set(path: &Path) -> Self {
        let original = std::env::current_dir().expect("current dir");
        std::env::set_current_dir(path).expect("set current dir");
        Self { original }
    }
}

impl Drop for CurrentDirGuard {
    fn drop(&mut self) {
        std::env::set_current_dir(&self.original).expect("restore current dir");
    }
}

fn plan_file_path(root: &Path, run_session_id: &str) -> PathBuf {
    root.join(crate::legacy_repo_local_lash_dir())
        .join("plans")
        .join(format!("{run_session_id}.md"))
}

fn ready_plan_markdown() -> &'static str {
    r#"# Plan

## Goal
- Update the plan-mode behavior cleanly.

## Steps
- Tighten the allowlist.
- Validate the plan before handoff.
- Execute the implementation turn after approval.

## Files
- lash/src/plugin_builtin/plan_mode.rs
- lash-ui/src/lib.rs

## Risks
- Status/panel sync could drift if toggle handling is incomplete.

## Verification
- cargo test -p lash --lib
- cargo test -p lash-ui
"#
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
        Ok(mock_snapshot("run-session"))
    }

    async fn snapshot_session(&self, _session_id: &str) -> Result<SessionSnapshot, PluginError> {
        Ok(mock_snapshot("run-session"))
    }

    async fn tool_catalog(&self, _session_id: &str) -> Result<Vec<serde_json::Value>, PluginError> {
        Ok(Vec::new())
    }

    async fn create_session(
        &self,
        request: SessionCreateRequest,
    ) -> Result<SessionHandle, PluginError> {
        Ok(SessionHandle {
            session_id: request.session_id.unwrap_or_else(|| "child".to_string()),
            parent_session_id: request.parent_session_id,
            policy: crate::SessionPolicy {
                provider: crate::Provider::OpenAiGeneric {
                    api_key: String::new(),
                    base_url: "https://example.invalid/v1".to_string(),
                    options: crate::ProviderOptions::default(),
                },
                model: "mock-model".to_string(),
                execution_mode: ExecutionMode::Standard,
                context_strategy: crate::ContextStrategy::RollingContext,
                ..Default::default()
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
            policy: crate::SessionPolicy {
                provider: crate::Provider::OpenAiGeneric {
                    api_key: String::new(),
                    base_url: "https://example.invalid/v1".to_string(),
                    options: crate::ProviderOptions::default(),
                },
                model: "mock-model".to_string(),
                execution_mode: ExecutionMode::Standard,
                context_strategy: crate::ContextStrategy::RollingContext,
                session_id: Some("run-session".to_string()),
                ..Default::default()
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
        state: SessionStateEnvelope {
            session_id: session_id.to_string(),
            policy: SessionPolicy {
                execution_mode: ExecutionMode::Standard,
                context_strategy: crate::ContextStrategy::RollingContext,
                ..Default::default()
            },
            ..Default::default()
        },
        status: TurnStatus::Completed,
        assistant_output: AssistantOutput {
            safe_text: String::new(),
            raw_text: String::new(),
            state: OutputState::Usable,
        },
        has_plugin_visible_output: false,
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
            state: SessionStateEnvelope::default(),
        })
        .await
        .expect("prompt contributions");

    assert!(contributions.iter().any(|contribution| {
        contribution.section == PromptSectionName::Environment
            && contribution.content.contains("Working directory:")
    }));
    assert!(contributions.iter().any(|contribution| {
        contribution.section == PromptSectionName::Environment
            && contribution.content.contains("Current date (UTC):")
    }));
    assert!(contributions.iter().any(|contribution| {
        contribution.section == PromptSectionName::Guidance
            && contribution.block == "project_instructions"
            && contribution.title.as_deref() == Some("Project Instructions")
            && contribution.content == "Repo rules"
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
                    input_schema_override: None,
                    output_schema_override: None,
                },
                ToolDefinition {
                    name: "read_file".to_string(),
                    description: "Read files".to_string(),
                    params: vec![],
                    returns: "str".to_string(),
                    examples: vec![],
                    enabled: true,
                    injected: true,
                    input_schema_override: None,
                    output_schema_override: None,
                },
                ToolDefinition {
                    name: "apply_patch".to_string(),
                    description: "Apply patches".to_string(),
                    params: vec![],
                    returns: "str".to_string(),
                    examples: vec![],
                    enabled: true,
                    injected: false,
                    input_schema_override: None,
                    output_schema_override: None,
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
                    input_schema_override: None,
                    output_schema_override: None,
                },
                ToolDefinition {
                    name: "read_file".to_string(),
                    description: "Read files".to_string(),
                    params: vec![],
                    returns: "str".to_string(),
                    examples: vec![],
                    enabled: true,
                    injected: true,
                    input_schema_override: None,
                    output_schema_override: None,
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
    let _guard = plan_mode_env_lock().lock().await;
    let temp = tempfile::tempdir().expect("tempdir");
    let _cwd = CurrentDirGuard::set(temp.path());
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
    assert_eq!(
        enabled
            .result
            .get("plan_path")
            .and_then(|value| value.as_str()),
        Some(".lash/plans/run-session.md")
    );
    assert_eq!(
        enabled
            .result
            .get("ready")
            .and_then(|value| value.as_bool()),
        Some(false)
    );
    assert!(plan_file_path(temp.path(), "run-session").is_file());

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
    let _guard = plan_mode_env_lock().lock().await;
    let temp = tempfile::tempdir().expect("tempdir");
    let _cwd = CurrentDirGuard::set(temp.path());
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
            state: mock_snapshot("run-session"),
            host: Arc::clone(&manager),
        })
        .await
        .expect("before_turn");
    assert!(before_turn.iter().any(|emitted| matches!(
        &emitted.value,
        PluginDirective::EnqueueMessages { messages }
            if messages.iter().any(|message|
                message.role == MessageRole::System
                    && message.content.contains("Plan mode:")
                    && message.content.contains(".lash/plans/run-session.md")
                    && message.content.contains("plan_exit()"))
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
    assert!(before_turn.iter().any(|emitted| {
        emitted.plugin_id == "plan_mode"
            && matches!(
                &emitted.value,
                PluginDirective::EmitEvents { events }
                    if events.iter().any(|event| matches!(
                        event,
                        crate::plugin::PluginSurfaceEvent::PanelUpsert { title, .. }
                            if title == "PLAN"
                    ))
            )
    }));
    assert!(plan_file_path(temp.path(), "run-session").is_file());

    let blocked_exec = session
        .before_tool_call(ToolCallHookContext {
            session_id: "root".to_string(),
            tool_name: "exec_command".to_string(),
            args: json!({"cmd":"cargo test -q"}),
            host: Arc::clone(&manager),
        })
        .await
        .expect("before_tool_call");
    assert!(!blocked_exec.is_empty());

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
                    input_schema_override: None,
                    output_schema_override: None,
                },
                ToolDefinition {
                    name: "update_plan".to_string(),
                    description: "Update plan".to_string(),
                    params: vec![],
                    returns: "str".to_string(),
                    examples: vec![],
                    enabled: true,
                    injected: true,
                    input_schema_override: None,
                    output_schema_override: None,
                },
                ToolDefinition {
                    name: "read_file".to_string(),
                    description: "Read files".to_string(),
                    params: vec![],
                    returns: "str".to_string(),
                    examples: vec![],
                    enabled: true,
                    injected: true,
                    input_schema_override: None,
                    output_schema_override: None,
                },
                ToolDefinition {
                    name: "search_web".to_string(),
                    description: "Search the web".to_string(),
                    params: vec![],
                    returns: "list".to_string(),
                    examples: vec![],
                    enabled: true,
                    injected: true,
                    input_schema_override: None,
                    output_schema_override: None,
                },
                ToolDefinition {
                    name: "apply_patch".to_string(),
                    description: "Apply patches".to_string(),
                    params: vec![],
                    returns: "dict".to_string(),
                    examples: vec![],
                    enabled: true,
                    injected: true,
                    input_schema_override: None,
                    output_schema_override: None,
                },
                ToolDefinition {
                    name: "plan_exit".to_string(),
                    description: "Exit plan mode".to_string(),
                    params: vec![],
                    returns: "dict".to_string(),
                    examples: vec![],
                    enabled: false,
                    injected: false,
                    input_schema_override: None,
                    output_schema_override: None,
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
    assert!(
        surface
            .tools
            .iter()
            .find(|tool| tool.name == "search_web")
            .is_some_and(|tool| tool.enabled && tool.injected)
    );
    assert!(
        surface
            .tools
            .iter()
            .find(|tool| tool.name == "plan_exit")
            .is_some_and(|tool| tool.enabled && tool.injected)
    );
    assert!(
        surface
            .tool_list_notes
            .iter()
            .any(|note| note.contains(".lash/plans/run-session.md"))
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

    let web_allowed = session
        .before_tool_call(ToolCallHookContext {
            session_id: "root".to_string(),
            tool_name: "search_web".to_string(),
            args: json!({
                "query":"surrealdb datetime best practices"
            }),
            host: Arc::clone(&manager),
        })
        .await
        .expect("before_tool_call");
    assert!(web_allowed.is_empty());

    let plan_patch_allowed = session
        .before_tool_call(ToolCallHookContext {
            session_id: "root".to_string(),
            tool_name: "apply_patch".to_string(),
            args: json!({
                "input": "*** Begin Patch\n*** Add File: .lash/plans/run-session.md\n+# Plan\n*** End Patch"
            }),
            host: Arc::clone(&manager),
        })
        .await
        .expect("before_tool_call");
    assert!(plan_patch_allowed.is_empty());

    let repo_patch_blocked = session
        .before_tool_call(ToolCallHookContext {
            session_id: "root".to_string(),
            tool_name: "apply_patch".to_string(),
            args: json!({
                "input": "*** Begin Patch\n*** Add File: src/main.rs\n+fn main() {}\n*** End Patch"
            }),
            host: Arc::clone(&manager),
        })
        .await
        .expect("before_tool_call");
    assert!(repo_patch_blocked.iter().any(|emitted| matches!(
        &emitted.value,
        PluginDirective::AbortTurn { code, message }
            if code == "plan_mode_tool_blocked"
                && message.contains(".lash/plans/run-session.md")
    )));

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
            state: mock_snapshot("run-session"),
            host: Arc::clone(&manager),
        })
        .await
        .expect("checkpoint");
    assert!(checkpoint.iter().any(|emitted| matches!(
        &emitted.value,
        PluginDirective::EnqueueMessages { messages }
            if messages.iter().any(|message|
                message.content.contains(".lash/plans/run-session.md")
                    && message.content.contains("plan_exit()"))
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
async fn plan_mode_plugin_uses_configured_allowlist() {
    let _guard = plan_mode_env_lock().lock().await;
    let temp = tempfile::tempdir().expect("tempdir");
    let _cwd = CurrentDirGuard::set(temp.path());
    let host = PluginHost::new(vec![Arc::new(PlanModePluginFactory::new(
        PlanModePluginConfig::default().with_allowed_tools(["apply_patch", "read_file"]),
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

    let blocked = session
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
    assert!(blocked.iter().any(|emitted| matches!(
        &emitted.value,
        PluginDirective::AbortTurn { code, .. } if code == "plan_mode_tool_blocked"
    )));

    let apply_patch_allowed = session
        .before_tool_call(ToolCallHookContext {
            session_id: "root".to_string(),
            tool_name: "apply_patch".to_string(),
            args: json!({
                "input": "*** Begin Patch\n*** Add File: .lash/plans/run-session.md\n+# Plan\n*** End Patch"
            }),
            host: Arc::clone(&manager),
        })
        .await
        .expect("before_tool_call");
    assert!(apply_patch_allowed.is_empty());
}

#[tokio::test]
async fn plan_mode_tool_exit_disables_mode_after_user_approval() {
    struct PromptingSessionManager;

    #[async_trait::async_trait]
    impl SessionManager for PromptingSessionManager {
        async fn snapshot_current(&self) -> Result<SessionSnapshot, PluginError> {
            Ok(mock_snapshot("run-session"))
        }

        async fn snapshot_session(
            &self,
            _session_id: &str,
        ) -> Result<SessionSnapshot, PluginError> {
            Ok(mock_snapshot("run-session"))
        }

        async fn tool_catalog(
            &self,
            _session_id: &str,
        ) -> Result<Vec<serde_json::Value>, PluginError> {
            Ok(Vec::new())
        }

        async fn create_session(
            &self,
            request: SessionCreateRequest,
        ) -> Result<SessionHandle, PluginError> {
            let base = MockSessionManager;
            base.create_session(request).await
        }

        async fn close_session(&self, session_id: &str) -> Result<(), PluginError> {
            let base = MockSessionManager;
            base.close_session(session_id).await
        }

        async fn start_turn_stream(
            &self,
            session_id: &str,
            input: TurnInput,
        ) -> Result<crate::plugin::SessionTurnHandle, PluginError> {
            let base = MockSessionManager;
            base.start_turn_stream(session_id, input).await
        }

        async fn await_turn(&self, turn_id: &str) -> Result<AssembledTurn, PluginError> {
            let base = MockSessionManager;
            base.await_turn(turn_id).await
        }

        async fn cancel_turn(&self, turn_id: &str) -> Result<(), PluginError> {
            let base = MockSessionManager;
            base.cancel_turn(turn_id).await
        }

        async fn prompt_user(
            &self,
            request: crate::PromptRequest,
        ) -> Result<crate::PromptResponse, PluginError> {
            assert!(request.question.contains(".lash/plans/run-session.md"));
            assert!(request.allows_note());
            Ok(crate::PromptResponse::Single {
                selection: "Exit plan mode".to_string(),
                note: Some("start with the safe slice".to_string()),
            })
        }
    }

    let _guard = plan_mode_env_lock().lock().await;
    let temp = tempfile::tempdir().expect("tempdir");
    let _cwd = CurrentDirGuard::set(temp.path());
    std::fs::create_dir_all(
        plan_file_path(temp.path(), "run-session")
            .parent()
            .expect("parent"),
    )
    .expect("plan dir");
    std::fs::write(
        plan_file_path(temp.path(), "run-session"),
        ready_plan_markdown(),
    )
    .expect("write plan");

    let host = PluginHost::new(vec![Arc::new(PlanModePluginFactory::new(
        PlanModePluginConfig::default(),
    ))]);
    let session = host.build_standard_session("root", None).expect("session");
    let manager: Arc<dyn SessionManager> = Arc::new(PromptingSessionManager);

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
            state: mock_snapshot("run-session"),
            host: Arc::clone(&manager),
        })
        .await
        .expect("before_turn");

    let result = session
        .tools()
        .execute_with_context(
            "plan_exit",
            &json!({}),
            &crate::ToolExecutionContext {
                session_id: "root".to_string(),
                host: Arc::clone(&manager),
            },
        )
        .await;
    assert!(result.success);
    assert_eq!(
        result
            .result
            .get("approved")
            .and_then(|value| value.as_bool()),
        Some(true)
    );
    assert!(
        result
            .result
            .get("next_turn_input")
            .and_then(|value| value.as_str())
            .is_some_and(|value| {
                value.contains("Execute the plan in `.lash/plans/run-session.md`.")
                    && value.contains("User note: start with the safe slice")
            })
    );

    let status = session
        .invoke_external("plan_mode.status", json!({}), None, true, manager)
        .await
        .expect("status");
    assert_eq!(
        status
            .result
            .get("enabled")
            .and_then(|value| value.as_bool()),
        Some(false)
    );
}

#[tokio::test]
async fn plan_mode_tool_exit_requires_ready_plan() {
    let _guard = plan_mode_env_lock().lock().await;
    let temp = tempfile::tempdir().expect("tempdir");
    let _cwd = CurrentDirGuard::set(temp.path());
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

    let result = session
        .tools()
        .execute_with_context(
            "plan_exit",
            &json!({}),
            &crate::ToolExecutionContext {
                session_id: "root".to_string(),
                host: Arc::clone(&manager),
            },
        )
        .await;
    assert!(!result.success);
    assert!(
        result
            .result
            .as_str()
            .is_some_and(|value| value.contains("Need: Goal (fill in), Steps (fill in), Files (fill in), Risks (fill in), Verification (fill in)."))
    );
}

#[tokio::test]
async fn plan_mode_plugin_does_not_rewrite_assistant_output() {
    let _guard = plan_mode_env_lock().lock().await;
    let temp = tempfile::tempdir().expect("tempdir");
    let _cwd = CurrentDirGuard::set(temp.path());
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

    let stream = session
        .transform_assistant_stream(
            "root",
            "Keep this text exactly.".to_string(),
            Arc::clone(&manager),
        )
        .await
        .expect("stream");
    assert!(stream.is_empty());

    let response = session
        .transform_assistant_response(
            "root",
            crate::llm::types::LlmResponse {
                full_text: "Keep this text exactly.".into(),
                deltas: Vec::new(),
                parts: vec![crate::llm::types::LlmOutputPart::Text {
                    text: "Keep this text exactly.".into(),
                }],
                usage: crate::llm::types::LlmUsage::default(),
                request_body: None,
                http_summary: None,
            },
            manager,
        )
        .await
        .expect("response");
    assert!(response.is_empty());
}
