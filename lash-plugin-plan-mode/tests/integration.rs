use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use serde_json::json;
use tokio::sync::Mutex;

use lash_plugin_plan_mode::{PlanModePluginConfig, PlanModePluginFactory};

use lash::plugin::{
    PluginDirective, PluginError, SessionPluginMode, SessionStartPoint, ToolCallHookContext,
    ToolResultHookContext, ToolSurfaceContext,
};
use lash::{
    AssembledTurn, DynamicToolProvider, ExecutionMode, MessageRole, PersistedSessionState,
    PluginHost, RuntimeSessionHost, SessionCreateRequest, SessionHandle, SessionPolicy,
    SessionReadView, SessionSnapshot, SessionSnapshotHost, SessionStateEnvelope, ToolCatalogHost,
    ToolDefinition, ToolProvider, ToolResult, TurnHookContext, TurnInput, TurnResultHookContext,
};
use lash::{DynamicToolHost, SessionLifecycleHost, TurnHost};

use lash::testing::{MockSessionManager, mock_assembled_turn};

#[async_trait::async_trait]
trait PlanTestHostCore: Send + Sync {
    async fn snapshot_current(&self) -> Result<SessionSnapshot, PluginError>;
    async fn snapshot_session(&self, session_id: &str) -> Result<SessionSnapshot, PluginError>;
    async fn tool_catalog(&self, session_id: &str) -> Result<Vec<serde_json::Value>, PluginError>;
    async fn dynamic_tool_state(
        &self,
        _session_id: &str,
    ) -> Result<lash::DynamicStateSnapshot, PluginError> {
        Err(PluginError::Session(
            "dynamic tool state is unavailable in this session".to_string(),
        ))
    }
    async fn apply_dynamic_tool_state(
        &self,
        _session_id: &str,
        _snapshot: lash::DynamicStateSnapshot,
    ) -> Result<u64, PluginError> {
        Err(PluginError::Session(
            "dynamic tool state mutation is unavailable in this session".to_string(),
        ))
    }
    async fn create_session(
        &self,
        request: SessionCreateRequest,
    ) -> Result<SessionHandle, PluginError>;
    async fn close_session(&self, session_id: &str) -> Result<(), PluginError>;
    async fn start_turn_stream(
        &self,
        session_id: &str,
        input: TurnInput,
    ) -> Result<lash::plugin::SessionTurnHandle, PluginError>;
    async fn await_turn(&self, turn_id: &str) -> Result<AssembledTurn, PluginError>;
    async fn cancel_turn(&self, turn_id: &str) -> Result<(), PluginError>;
    async fn prompt_user(
        &self,
        _request: lash::PromptRequest,
    ) -> Result<lash::PromptResponse, PluginError> {
        Err(PluginError::Session("prompt unavailable".to_string()))
    }
}

macro_rules! impl_plan_test_host {
    ($ty:ty) => {
        #[async_trait::async_trait]
        impl lash::SessionSnapshotHost for $ty {
            async fn snapshot_current(&self) -> Result<SessionSnapshot, PluginError> {
                PlanTestHostCore::snapshot_current(self).await
            }
            async fn snapshot_session(
                &self,
                session_id: &str,
            ) -> Result<SessionSnapshot, PluginError> {
                PlanTestHostCore::snapshot_session(self, session_id).await
            }
        }

        #[async_trait::async_trait]
        impl lash::ToolCatalogHost for $ty {
            async fn tool_catalog(
                &self,
                session_id: &str,
            ) -> Result<Vec<serde_json::Value>, PluginError> {
                PlanTestHostCore::tool_catalog(self, session_id).await
            }
        }

        #[async_trait::async_trait]
        impl lash::DynamicToolHost for $ty {
            async fn dynamic_tool_state(
                &self,
                session_id: &str,
            ) -> Result<lash::DynamicStateSnapshot, PluginError> {
                PlanTestHostCore::dynamic_tool_state(self, session_id).await
            }
            async fn apply_dynamic_tool_state(
                &self,
                session_id: &str,
                snapshot: lash::DynamicStateSnapshot,
            ) -> Result<u64, PluginError> {
                PlanTestHostCore::apply_dynamic_tool_state(self, session_id, snapshot).await
            }
        }

        #[async_trait::async_trait]
        impl lash::SessionLifecycleHost for $ty {
            async fn create_session(
                &self,
                request: SessionCreateRequest,
            ) -> Result<SessionHandle, PluginError> {
                PlanTestHostCore::create_session(self, request).await
            }
            async fn close_session(&self, session_id: &str) -> Result<(), PluginError> {
                PlanTestHostCore::close_session(self, session_id).await
            }
        }

        #[async_trait::async_trait]
        impl lash::TurnHost for $ty {
            async fn start_turn_stream(
                &self,
                session_id: &str,
                input: TurnInput,
            ) -> Result<lash::plugin::SessionTurnHandle, PluginError> {
                PlanTestHostCore::start_turn_stream(self, session_id, input).await
            }
            async fn await_turn(&self, turn_id: &str) -> Result<AssembledTurn, PluginError> {
                PlanTestHostCore::await_turn(self, turn_id).await
            }
            async fn cancel_turn(&self, turn_id: &str) -> Result<(), PluginError> {
                PlanTestHostCore::cancel_turn(self, turn_id).await
            }
        }

        #[async_trait::async_trait]
        impl lash_plugin_plan_mode::PlanModePrompt for $ty {
            async fn prompt_user(
                &self,
                request: lash::PromptRequest,
            ) -> Result<lash::PromptResponse, PluginError> {
                PlanTestHostCore::prompt_user(self, request).await
            }
        }

        impl lash::TaskHost for $ty {}
        impl lash::MonitorHost for $ty {}
        impl lash::SessionGraphHost for $ty {}
        impl lash::DirectCompletionHost for $ty {}
        impl lash::TraceHost for $ty {}
    };
}

fn mock_snapshot(run_session_id: &str) -> SessionSnapshot {
    PersistedSessionState::from_state(SessionStateEnvelope {
        session_id: "root".to_string(),
        policy: SessionPolicy {
            execution_mode: ExecutionMode::standard(),
            session_id: Some(run_session_id.to_string()),
            ..Default::default()
        },
        ..Default::default()
    })
}

fn mock_read_view(run_session_id: &str) -> SessionReadView {
    let snapshot = mock_snapshot(run_session_id);
    SessionReadView::from_persisted_state(&snapshot)
}

fn test_tool(
    name: &str,
    description: &str,
    availability: lash::ToolAvailabilityConfig,
    execution_mode: lash::ToolExecutionMode,
) -> ToolDefinition {
    ToolDefinition::new(
        name,
        description,
        ToolDefinition::default_input_schema(),
        serde_json::json!({ "type": "object", "additionalProperties": true }),
    )
    .with_availability(availability)
    .with_execution_mode(execution_mode)
}

struct PlanModeDynamicTools;

#[async_trait::async_trait]
impl ToolProvider for PlanModeDynamicTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![
            test_tool(
                "plan_exit",
                "Ask whether to exit plan mode.",
                lash::ToolAvailabilityConfig::hidden(),
                lash::ToolExecutionMode::Parallel,
            )
            .with_examples(vec!["plan_exit()".to_string()]),
        ]
    }

    async fn execute(&self, name: &str, _args: &serde_json::Value) -> ToolResult {
        ToolResult::err_fmt(format_args!("unexpected tool call: {name}"))
    }
}

fn mock_session_manager(run_session_id: &str) -> MockSessionManager {
    MockSessionManager::default()
        .with_snapshot(mock_snapshot(run_session_id))
        .with_turn(mock_assembled_turn(run_session_id, ""))
        .with_dynamic_tool_provider(
            DynamicToolProvider::from_tool_provider(Arc::new(PlanModeDynamicTools))
                .expect("plan mode dynamic tools"),
        )
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
    root.join(".lash")
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
- lash-tui-extensions/src/lib.rs

## Risks
- Status/panel sync could drift if toggle handling is incomplete.

## Verification
- cargo test -p lash --lib
- cargo test -p lash-tui-extensions
"#
}

fn empty_turn(session_id: &str) -> AssembledTurn {
    mock_assembled_turn(session_id, "")
}

fn plan_mode_host(plan_factory: PlanModePluginFactory) -> PluginHost {
    let mut factories = lash::testing::test_mode_factories();
    factories.push(Arc::new(plan_factory));
    PluginHost::new(factories)
}

#[tokio::test]
async fn plan_mode_plugin_enable_toggle_and_restore_round_trip() {
    let _guard = plan_mode_env_lock().lock().await;
    let temp = tempfile::tempdir().expect("tempdir");
    let _cwd = CurrentDirGuard::set(temp.path());
    let host = plan_mode_host(PlanModePluginFactory::default());
    let session = host.build_standard_session("root", None).expect("session");
    let manager: Arc<dyn RuntimeSessionHost> = Arc::new(mock_session_manager("run-session"));

    let enabled = session
        .invoke_external("plan_mode.enable", json!({}), None, true, manager.clone())
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
    assert!(plan_file_path(temp.path(), "run-session").is_file());

    let snapshot = session.snapshot().expect("snapshot");
    let restored = host
        .build_standard_session("restored", Some(&snapshot))
        .expect("restored");
    let restored_toggle = restored
        .invoke_external("plan_mode.toggle", json!({}), None, true, manager)
        .await
        .expect("toggle restored");
    assert_eq!(
        restored_toggle
            .result
            .get("enabled")
            .and_then(|v| v.as_bool()),
        Some(false)
    );

    restored
        .restore(&lash::PluginSessionSnapshot::default())
        .expect("reset restore");
    let reset_toggle = restored
        .invoke_external(
            "plan_mode.toggle",
            json!({}),
            None,
            true,
            Arc::new(mock_session_manager("run-session")),
        )
        .await
        .expect("toggle reset");
    assert_eq!(
        reset_toggle.result.get("enabled").and_then(|v| v.as_bool()),
        Some(true)
    );
}

#[tokio::test]
async fn plan_mode_toggles_dynamic_plan_exit_tool_state() {
    let _guard = plan_mode_env_lock().lock().await;
    let temp = tempfile::tempdir().expect("tempdir");
    let _cwd = CurrentDirGuard::set(temp.path());
    let host = plan_mode_host(PlanModePluginFactory::default());
    let session = host.build_standard_session("root", None).expect("session");
    let manager = Arc::new(mock_session_manager("run-session"));
    let manager_host: Arc<dyn RuntimeSessionHost> = manager.clone();

    let initial = manager
        .dynamic_tool_state("root")
        .await
        .expect("initial dynamic tool state");
    assert!(initial.tools.get("plan_exit").is_some_and(|tool| {
        tool.definition
            .effective_availability(&lash::ExecutionMode::standard())
            == lash::ToolAvailability::Hidden
    }));

    session
        .invoke_external(
            "plan_mode.enable",
            json!({}),
            None,
            true,
            manager_host.clone(),
        )
        .await
        .expect("enable");

    let enabled = manager
        .dynamic_tool_state("root")
        .await
        .expect("enabled dynamic tool state");
    assert!(enabled.tools.get("plan_exit").is_some_and(|tool| {
        tool.definition
            .effective_availability(&lash::ExecutionMode::standard())
            == lash::ToolAvailability::Documented
    }));

    session
        .invoke_external("plan_mode.disable", json!({}), None, true, manager_host)
        .await
        .expect("disable");

    let disabled = manager
        .dynamic_tool_state("root")
        .await
        .expect("disabled dynamic tool state");
    assert!(disabled.tools.get("plan_exit").is_some_and(|tool| {
        tool.definition
            .effective_availability(&lash::ExecutionMode::standard())
            == lash::ToolAvailability::Hidden
    }));
}

#[tokio::test]
async fn plan_mode_plugin_injects_guidance_and_blocks_implementation_tools() {
    let _guard = plan_mode_env_lock().lock().await;
    let temp = tempfile::tempdir().expect("tempdir");
    let _cwd = CurrentDirGuard::set(temp.path());
    let host = plan_mode_host(PlanModePluginFactory::default());
    let session = host.build_standard_session("root", None).expect("session");
    let manager: Arc<dyn RuntimeSessionHost> = Arc::new(mock_session_manager("run-session"));

    session
        .invoke_external("plan_mode.enable", json!({}), None, true, manager.clone())
        .await
        .expect("enable");

    let before_turn = session
        .before_turn(TurnHookContext {
            session_id: "root".to_string(),
            state: mock_read_view("run-session"),
            host: manager.clone(),
            turn_context: lash::TurnContext::default(),
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
                    && message.content.contains("plan_exit()")
                    && message.content.contains("single source of truth")
                    && message.content.contains("panel only shows the file path"))
    )));
    assert!(before_turn.iter().any(|emitted| {
        emitted.plugin_id == "plan_mode"
            && matches!(
                &emitted.value,
                PluginDirective::EmitEvents { events }
                    if events.iter().any(|event| matches!(
                        event,
                        lash::plugin::PluginSurfaceEvent::ModeIndicatorUpsert { label, .. }
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
                        lash::plugin::PluginSurfaceEvent::PanelUpsert { title, content, .. }
                            if title == "PLAN"
                                && content.contains("Path: `.lash/plans/run-session.md`")
                    ))
            )
    }));
    assert!(plan_file_path(temp.path(), "run-session").is_file());

    let blocked_exec = session
        .before_tool_call(ToolCallHookContext {
            session_id: "root".to_string(),
            tool_name: "exec_command".to_string(),
            args: json!({"cmd":"cargo test -q"}),
            host: manager.clone(),
            turn_context: lash::TurnContext::default(),
        })
        .await
        .expect("before_tool_call");
    assert!(!blocked_exec.is_empty());

    let allowed = session
        .before_tool_call(ToolCallHookContext {
            session_id: "root".to_string(),
            tool_name: "read_file".to_string(),
            args: json!({"path":"src/main.rs"}),
            host: manager.clone(),
            turn_context: lash::TurnContext::default(),
        })
        .await
        .expect("before_tool_call");
    assert!(allowed.is_empty());

    let surface = session
        .resolve_tool_surface(ToolSurfaceContext {
            session_id: "root".to_string(),
            mode: ExecutionMode::standard(),
            tools: vec![
                test_tool(
                    "search_tools",
                    "Discover tools",
                    lash::ToolAvailabilityConfig::callable(),
                    lash::ToolExecutionMode::Parallel,
                ),
                test_tool(
                    "read_file",
                    "Read files",
                    lash::ToolAvailabilityConfig::documented(),
                    lash::ToolExecutionMode::Parallel,
                ),
                test_tool(
                    "search_web",
                    "Search the web",
                    lash::ToolAvailabilityConfig::documented(),
                    lash::ToolExecutionMode::Parallel,
                ),
                test_tool(
                    "apply_patch",
                    "Apply patches",
                    lash::ToolAvailabilityConfig::documented(),
                    lash::ToolExecutionMode::Serial,
                ),
                test_tool(
                    "plan_exit",
                    "Exit plan mode",
                    lash::ToolAvailabilityConfig::hidden(),
                    lash::ToolExecutionMode::Parallel,
                ),
            ],
            tool_access: lash::SessionToolAccess::default(),
            subagent: None,
        })
        .expect("tool surface");
    assert!(
        surface
            .tools
            .iter()
            .find(|tool| tool.definition.name == "search_web")
            .is_some_and(|tool| tool.availability == lash::ToolAvailability::Documented)
    );
    assert!(
        surface
            .tools
            .iter()
            .find(|tool| tool.definition.name == "plan_exit")
            .is_some_and(|tool| tool.availability == lash::ToolAvailability::Documented)
    );
    assert!(
        surface
            .tool_list_notes
            .iter()
            .any(|note| note.contains(".lash/plans/run-session.md"))
    );

    let read_allowed = session
        .before_tool_call(ToolCallHookContext {
            session_id: "root".to_string(),
            tool_name: "read_file".to_string(),
            args: json!({
                "path":"src/main.rs"
            }),
            host: manager.clone(),
            turn_context: lash::TurnContext::default(),
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
            host: manager.clone(),
            turn_context: lash::TurnContext::default(),
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
            host: manager.clone(),
            turn_context: lash::TurnContext::default(),
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
            host: manager.clone(),
            turn_context: lash::TurnContext::default(),
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
        .invoke_external("plan_mode.disable", json!({}), None, true, manager.clone())
        .await
        .expect("disable");
    session
        .invoke_external("plan_mode.enable", json!({}), None, true, manager.clone())
        .await
        .expect("enable");

    let checkpoint = session
        .at_checkpoint(lash::CheckpointHookContext {
            session_id: "root".to_string(),
            checkpoint: lash::CheckpointKind::AfterWork,
            state: mock_read_view("run-session"),
            host: manager.clone(),
        })
        .await
        .expect("checkpoint");
    assert!(checkpoint.iter().any(|emitted| matches!(
        &emitted.value,
        PluginDirective::EnqueueMessages { messages }
            if messages.iter().any(|message|
                message.content.contains(".lash/plans/run-session.md")
                    && message.content.contains("plan_exit()")
                    && message.content.contains("single source of truth")
                    && message.content.contains("panel only shows the file path"))
    )));

    session
        .after_turn(TurnResultHookContext {
            session_id: "root".to_string(),
            turn: Arc::new(lash::TurnResultSummary::from_assembled(&empty_turn("root"))),
            host: manager,
        })
        .await
        .expect("after_turn");
}

#[tokio::test]
async fn plan_mode_does_not_reinject_entry_guidance_on_later_turns() {
    let _guard = plan_mode_env_lock().lock().await;
    let temp = tempfile::tempdir().expect("tempdir");
    let _cwd = CurrentDirGuard::set(temp.path());
    let host = plan_mode_host(PlanModePluginFactory::default());
    let session = host.build_standard_session("root", None).expect("session");
    let manager: Arc<dyn RuntimeSessionHost> = Arc::new(mock_session_manager("run-session"));

    session
        .invoke_external("plan_mode.enable", json!({}), None, true, manager.clone())
        .await
        .expect("enable");

    let first_before_turn = session
        .before_turn(TurnHookContext {
            session_id: "root".to_string(),
            state: mock_read_view("run-session"),
            host: manager.clone(),
            turn_context: lash::TurnContext::default(),
        })
        .await
        .expect("first before_turn");
    assert!(first_before_turn.iter().any(|emitted| matches!(
        &emitted.value,
        PluginDirective::EnqueueMessages { messages }
            if messages.iter().any(|message| message.content.contains("Plan mode:"))
    )));

    session
        .after_turn(TurnResultHookContext {
            session_id: "root".to_string(),
            turn: Arc::new(lash::TurnResultSummary::from_assembled(&empty_turn("root"))),
            host: manager.clone(),
        })
        .await
        .expect("after_turn");

    let second_before_turn = session
        .before_turn(TurnHookContext {
            session_id: "root".to_string(),
            state: mock_read_view("run-session"),
            host: manager.clone(),
            turn_context: lash::TurnContext::default(),
        })
        .await
        .expect("second before_turn");
    assert!(second_before_turn.is_empty());
}

#[tokio::test]
async fn plan_mode_plugin_uses_configured_allowlist() {
    let _guard = plan_mode_env_lock().lock().await;
    let temp = tempfile::tempdir().expect("tempdir");
    let _cwd = CurrentDirGuard::set(temp.path());
    let host = plan_mode_host(PlanModePluginFactory::new(
        PlanModePluginConfig::default().with_allowed_tools(["apply_patch", "read_file"]),
    ));
    let session = host.build_standard_session("root", None).expect("session");
    let manager: Arc<dyn RuntimeSessionHost> = Arc::new(mock_session_manager("run-session"));

    session
        .invoke_external("plan_mode.enable", json!({}), None, true, manager.clone())
        .await
        .expect("enable");

    let allowed = session
        .before_tool_call(ToolCallHookContext {
            session_id: "root".to_string(),
            tool_name: "read_file".to_string(),
            args: json!({"path":"src/main.rs"}),
            host: manager.clone(),
            turn_context: lash::TurnContext::default(),
        })
        .await
        .expect("before_tool_call");
    assert!(allowed.is_empty());

    let apply_patch_allowed = session
        .before_tool_call(ToolCallHookContext {
            session_id: "root".to_string(),
            tool_name: "apply_patch".to_string(),
            args: json!({
                "input": "*** Begin Patch\n*** Add File: .lash/plans/run-session.md\n+# Plan\n*** End Patch"
            }),
            host: manager.clone(),
            turn_context: lash::TurnContext::default(),
        })
        .await
        .expect("before_tool_call");
    assert!(apply_patch_allowed.is_empty());
}

#[tokio::test]
async fn plan_mode_tool_exit_disables_mode_after_user_approval() {
    struct PromptingSessionManager {
        base: MockSessionManager,
    }
    impl_plan_test_host!(PromptingSessionManager);

    #[async_trait::async_trait]
    impl PlanTestHostCore for PromptingSessionManager {
        async fn snapshot_current(&self) -> Result<SessionSnapshot, PluginError> {
            self.base.snapshot_current().await
        }

        async fn snapshot_session(&self, session_id: &str) -> Result<SessionSnapshot, PluginError> {
            self.base.snapshot_session(session_id).await
        }

        async fn tool_catalog(
            &self,
            session_id: &str,
        ) -> Result<Vec<serde_json::Value>, PluginError> {
            self.base.tool_catalog(session_id).await
        }

        async fn dynamic_tool_state(
            &self,
            session_id: &str,
        ) -> Result<lash::DynamicStateSnapshot, PluginError> {
            self.base.dynamic_tool_state(session_id).await
        }

        async fn apply_dynamic_tool_state(
            &self,
            session_id: &str,
            snapshot: lash::DynamicStateSnapshot,
        ) -> Result<u64, PluginError> {
            self.base
                .apply_dynamic_tool_state(session_id, snapshot)
                .await
        }

        async fn create_session(
            &self,
            request: SessionCreateRequest,
        ) -> Result<SessionHandle, PluginError> {
            self.base.create_session(request).await
        }

        async fn close_session(&self, session_id: &str) -> Result<(), PluginError> {
            self.base.close_session(session_id).await
        }

        async fn start_turn_stream(
            &self,
            session_id: &str,
            input: TurnInput,
        ) -> Result<lash::plugin::SessionTurnHandle, PluginError> {
            self.base.start_turn_stream(session_id, input).await
        }

        async fn await_turn(&self, turn_id: &str) -> Result<AssembledTurn, PluginError> {
            self.base.await_turn(turn_id).await
        }

        async fn cancel_turn(&self, turn_id: &str) -> Result<(), PluginError> {
            self.base.cancel_turn(turn_id).await
        }

        async fn prompt_user(
            &self,
            request: lash::PromptRequest,
        ) -> Result<lash::PromptResponse, PluginError> {
            assert!(request.question.contains(".lash/plans/run-session.md"));
            assert!(request.allows_note());
            let panel = request.panel.expect("plan review panel");
            assert_eq!(panel.title, "PLAN");
            assert_eq!(panel.markdown, ready_plan_markdown().trim_end());
            Ok(lash::PromptResponse::Single {
                selection: "Start implementing now".to_string(),
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

    let manager = Arc::new(PromptingSessionManager {
        base: mock_session_manager("run-session"),
    });
    let manager_host: Arc<dyn RuntimeSessionHost> = manager.clone();
    let host = plan_mode_host(
        PlanModePluginFactory::new(PlanModePluginConfig::default()).with_prompt(manager.clone()),
    );
    let session = host.build_standard_session("root", None).expect("session");

    session
        .invoke_external(
            "plan_mode.enable",
            json!({}),
            None,
            true,
            manager_host.clone(),
        )
        .await
        .expect("enable");
    session
        .before_turn(TurnHookContext {
            session_id: "root".to_string(),
            state: mock_read_view("run-session"),
            host: manager_host.clone(),
            turn_context: lash::TurnContext::default(),
        })
        .await
        .expect("before_turn");

    let result = session
        .tools()
        .execute_with_context(
            "plan_exit",
            &json!({}),
            &lash::ToolExecutionContext {
                session_id: "root".to_string(),
                host: manager_host.clone(),
                cancellation_token: None,
                async_task_id: None,
                turn_context: lash::TurnContext::default(),
                tool_call_id: None,
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
                value.contains(
                    "The user approved the plan. Execute the plan in `.lash/plans/run-session.md` now"
                )
                    && value.contains("User note: start with the safe slice")
            })
    );

    let dynamic = lash::DynamicToolHost::dynamic_tool_state(manager.as_ref(), "root")
        .await
        .expect("dynamic tool state");
    assert!(dynamic.tools.get("plan_exit").is_some_and(|tool| {
        tool.definition
            .effective_availability(&lash::ExecutionMode::standard())
            == lash::ToolAvailability::Hidden
    }));
}

#[tokio::test]
async fn plan_mode_tool_exit_allows_exit_without_validation() {
    struct PromptingSessionManager;
    impl_plan_test_host!(PromptingSessionManager);

    #[async_trait::async_trait]
    impl PlanTestHostCore for PromptingSessionManager {
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
            let base = mock_session_manager("run-session");
            base.create_session(request).await
        }

        async fn close_session(&self, session_id: &str) -> Result<(), PluginError> {
            let base = mock_session_manager("run-session");
            base.close_session(session_id).await
        }

        async fn start_turn_stream(
            &self,
            session_id: &str,
            input: TurnInput,
        ) -> Result<lash::plugin::SessionTurnHandle, PluginError> {
            let base = mock_session_manager("run-session");
            base.start_turn_stream(session_id, input).await
        }

        async fn await_turn(&self, turn_id: &str) -> Result<AssembledTurn, PluginError> {
            let base = mock_session_manager("run-session");
            base.await_turn(turn_id).await
        }

        async fn cancel_turn(&self, turn_id: &str) -> Result<(), PluginError> {
            let base = mock_session_manager("run-session");
            base.cancel_turn(turn_id).await
        }

        async fn prompt_user(
            &self,
            request: lash::PromptRequest,
        ) -> Result<lash::PromptResponse, PluginError> {
            assert_eq!(
                request.question,
                "Review the plan in `.lash/plans/run-session.md`. What next?"
            );
            let panel = request.panel.expect("plan review panel");
            assert_eq!(panel.title, "PLAN");
            assert!(panel.markdown.contains("# Plan"));
            Ok(lash::PromptResponse::Single {
                selection: "Start implementing now".to_string(),
                note: None,
            })
        }
    }

    let _guard = plan_mode_env_lock().lock().await;
    let temp = tempfile::tempdir().expect("tempdir");
    let _cwd = CurrentDirGuard::set(temp.path());
    let manager = Arc::new(PromptingSessionManager);
    let host = plan_mode_host(PlanModePluginFactory::default().with_prompt(manager.clone()));
    let session = host.build_standard_session("root", None).expect("session");
    let manager: Arc<dyn RuntimeSessionHost> = manager;

    session
        .invoke_external("plan_mode.enable", json!({}), None, true, manager.clone())
        .await
        .expect("enable");

    session
        .before_turn(TurnHookContext {
            session_id: "root".to_string(),
            state: mock_read_view("run-session"),
            host: manager.clone(),
            turn_context: lash::TurnContext::default(),
        })
        .await
        .expect("before_turn");

    let result = session
        .tools()
        .execute_with_context(
            "plan_exit",
            &json!({}),
            &lash::ToolExecutionContext {
                session_id: "root".to_string(),
                host: manager.clone(),
                turn_context: lash::TurnContext::default(),
                cancellation_token: None,
                async_task_id: None,
                tool_call_id: None,
            },
        )
        .await;
    assert!(result.success);
    assert!(
        result
            .result
            .get("approved")
            .and_then(|value| value.as_bool())
            == Some(true)
    );
    assert_eq!(
        result
            .result
            .get("execution_mode")
            .and_then(|value| value.as_str()),
        Some("current_session")
    );
}

#[tokio::test]
async fn plan_mode_tool_exit_can_execute_with_fresh_context() {
    struct PromptingSessionManager;
    impl_plan_test_host!(PromptingSessionManager);

    #[async_trait::async_trait]
    impl PlanTestHostCore for PromptingSessionManager {
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
            let base = mock_session_manager("run-session");
            base.create_session(request).await
        }

        async fn close_session(&self, session_id: &str) -> Result<(), PluginError> {
            let base = mock_session_manager("run-session");
            base.close_session(session_id).await
        }

        async fn start_turn_stream(
            &self,
            session_id: &str,
            input: TurnInput,
        ) -> Result<lash::plugin::SessionTurnHandle, PluginError> {
            let base = mock_session_manager("run-session");
            base.start_turn_stream(session_id, input).await
        }

        async fn await_turn(&self, turn_id: &str) -> Result<AssembledTurn, PluginError> {
            let base = mock_session_manager("run-session");
            base.await_turn(turn_id).await
        }

        async fn cancel_turn(&self, turn_id: &str) -> Result<(), PluginError> {
            let base = mock_session_manager("run-session");
            base.cancel_turn(turn_id).await
        }

        async fn prompt_user(
            &self,
            _request: lash::PromptRequest,
        ) -> Result<lash::PromptResponse, PluginError> {
            Ok(lash::PromptResponse::Single {
                selection: "Start in fresh context".to_string(),
                note: None,
            })
        }
    }

    let _guard = plan_mode_env_lock().lock().await;
    let temp = tempfile::tempdir().expect("tempdir");
    let _cwd = CurrentDirGuard::set(temp.path());
    let manager = Arc::new(PromptingSessionManager);
    let host = plan_mode_host(PlanModePluginFactory::default().with_prompt(manager.clone()));
    let session = host.build_standard_session("root", None).expect("session");
    let manager: Arc<dyn RuntimeSessionHost> = manager;

    session
        .invoke_external("plan_mode.enable", json!({}), None, true, manager.clone())
        .await
        .expect("enable");

    session
        .before_turn(TurnHookContext {
            session_id: "root".to_string(),
            state: mock_read_view("run-session"),
            host: manager.clone(),
            turn_context: lash::TurnContext::default(),
        })
        .await
        .expect("before_turn");

    let result = session
        .tools()
        .execute_with_context(
            "plan_exit",
            &json!({}),
            &lash::ToolExecutionContext {
                session_id: "root".to_string(),
                host: manager.clone(),
                turn_context: lash::TurnContext::default(),
                cancellation_token: None,
                async_task_id: None,
                tool_call_id: None,
            },
        )
        .await;
    assert!(result.success);
    assert_eq!(
        result
            .result
            .get("execution_mode")
            .and_then(|value| value.as_str()),
        Some("fresh_context")
    );
    assert!(
        result.result.get("fresh_context_input").is_none(),
        "seed prompt is now carried by SessionCreateRequest::first_turn_input, not the tool result"
    );
}

#[tokio::test]
async fn plan_mode_after_tool_call_creates_fresh_context_session_on_approval() {
    #[derive(Clone, Default)]
    struct CapturingSessionManager {
        created: Arc<std::sync::Mutex<Vec<SessionCreateRequest>>>,
    }
    impl_plan_test_host!(CapturingSessionManager);

    #[async_trait::async_trait]
    impl PlanTestHostCore for CapturingSessionManager {
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
            self.created.lock().expect("created").push(request.clone());
            Ok(SessionHandle {
                session_id: request
                    .session_id
                    .clone()
                    .unwrap_or_else(|| "new-session".to_string()),
                parent_session_id: request.relation.parent_session_id().map(ToOwned::to_owned),
                policy: request.policy.clone().unwrap_or_default(),
            })
        }

        async fn close_session(&self, _session_id: &str) -> Result<(), PluginError> {
            Ok(())
        }

        async fn start_turn_stream(
            &self,
            _session_id: &str,
            _input: TurnInput,
        ) -> Result<lash::plugin::SessionTurnHandle, PluginError> {
            Err(PluginError::Session("unused".to_string()))
        }

        async fn await_turn(&self, _turn_id: &str) -> Result<AssembledTurn, PluginError> {
            Err(PluginError::Session("unused".to_string()))
        }

        async fn cancel_turn(&self, _turn_id: &str) -> Result<(), PluginError> {
            Ok(())
        }
    }

    let _guard = plan_mode_env_lock().lock().await;
    let temp = tempfile::tempdir().expect("tempdir");
    let _cwd = CurrentDirGuard::set(temp.path());
    let host = plan_mode_host(PlanModePluginFactory::default());
    let session = host.build_standard_session("root", None).expect("session");
    let manager = Arc::new(CapturingSessionManager::default());

    session
        .invoke_external("plan_mode.enable", json!({}), None, true, manager.clone())
        .await
        .expect("enable");

    let directives = session
        .after_tool_call(ToolResultHookContext {
            session_id: "root".to_string(),
            tool_name: "plan_exit".to_string(),
            args: json!({}),
            result: ToolResult::ok(json!({
                "approved": true,
                "plan_path": ".lash/plans/run-session.md",
                "execution_mode": "fresh_context",
            })),
            duration_ms: 1,
            host: manager.clone(),
            turn_context: lash::TurnContext::default(),
        })
        .await
        .expect("after_tool_call");

    let create_request = directives
        .iter()
        .find_map(|owned| match &owned.value {
            PluginDirective::CreateSession { request } => Some(request.as_ref()),
            _ => None,
        })
        .expect("create session directive");
    assert_eq!(manager.created.lock().expect("created").len(), 0);
    assert!(matches!(create_request.start, SessionStartPoint::Empty));
    assert_eq!(create_request.plugin_mode, SessionPluginMode::Fresh);
    assert!(
        create_request.initial_nodes.is_empty(),
        "fresh-context execution should let the host drive the seed prompt as the first user turn"
    );
    let seed = create_request
        .first_turn_input
        .as_ref()
        .expect("first_turn_input set on CreateSession directive");
    assert_eq!(seed.role, lash::MessageRole::User);
    assert_eq!(
        seed.content,
        "Do a full, faithful implementation of the plan found at: .lash/plans/run-session.md"
    );
    assert!(
        directives
            .iter()
            .any(|owned| matches!(owned.value, PluginDirective::EmitEvents { .. }))
    );
    assert!(
        directives
            .iter()
            .any(|owned| matches!(owned.value, PluginDirective::HandoffSession { .. }))
    );
}

#[tokio::test]
async fn plan_mode_plugin_does_not_rewrite_assistant_output() {
    let _guard = plan_mode_env_lock().lock().await;
    let temp = tempfile::tempdir().expect("tempdir");
    let _cwd = CurrentDirGuard::set(temp.path());
    let host = plan_mode_host(PlanModePluginFactory::default());
    let session = host.build_standard_session("root", None).expect("session");
    let manager: Arc<dyn RuntimeSessionHost> = Arc::new(mock_session_manager("run-session"));

    session
        .invoke_external("plan_mode.enable", json!({}), None, true, manager.clone())
        .await
        .expect("enable");

    let stream = session
        .transform_assistant_stream(
            "root",
            "Keep this text exactly.".to_string(),
            manager.clone(),
        )
        .await
        .expect("stream");
    assert!(stream.is_empty());

    let response = session
        .transform_assistant_response(
            "root",
            lash::llm::types::LlmResponse {
                full_text: "Keep this text exactly.".into(),
                deltas: Vec::new(),
                parts: vec![lash::llm::types::LlmOutputPart::Text {
                    text: "Keep this text exactly.".into(),
                    response_meta: None,
                }],
                usage: lash::llm::types::LlmUsage::default(),
                provider_usage: None,
                request_body: None,
                http_summary: None,
            },
            manager,
        )
        .await
        .expect("response");
    assert!(response.is_empty());
}
