use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use serde_json::json;
use tokio::sync::Mutex;

use lash_plugin_plan_mode::{PlanModePluginConfig, PlanModePluginFactory};

use lash_core::plugin::runtime_host::RuntimeSessionHost;
use lash_core::plugin::{
    PluginDirective, PluginError, SessionPluginMode, SessionStartPoint, ToolCallHookContext,
    ToolResultHookContext, ToolSurfaceContext,
};
use lash_core::{
    AssembledTurn, ExecutionMode, MessageRole, PersistedSessionState, PluginHost,
    SessionCreateRequest, SessionHandle, SessionPolicy, SessionReadView, SessionSnapshot,
    SessionStateEnvelope, ToolContract, ToolDefinition, ToolManifest, ToolProvider, ToolRegistry,
    ToolResult, TurnHookContext, TurnResultHookContext,
};

use lash_core::testing::{MockSessionManager, mock_assembled_turn};

#[async_trait::async_trait]
trait PlanTestHostCore: Send + Sync {
    async fn snapshot_current(&self) -> Result<SessionSnapshot, PluginError>;
    async fn snapshot_session(&self, session_id: &str) -> Result<SessionSnapshot, PluginError>;
    async fn tool_catalog(&self, session_id: &str) -> Result<Vec<serde_json::Value>, PluginError>;
    async fn tool_state(&self, _session_id: &str) -> Result<lash_core::ToolState, PluginError> {
        Err(PluginError::Session(
            "tool state is unavailable in this session".to_string(),
        ))
    }
    async fn apply_tool_state(
        &self,
        _session_id: &str,
        _snapshot: lash_core::ToolState,
    ) -> Result<u64, PluginError> {
        Err(PluginError::Session(
            "tool state mutation is unavailable in this session".to_string(),
        ))
    }
    async fn create_session(
        &self,
        request: SessionCreateRequest,
    ) -> Result<SessionHandle, PluginError>;
    async fn close_session(&self, session_id: &str) -> Result<(), PluginError>;
    async fn prompt_user(
        &self,
        _request: lash_plugin_plan_mode::PlanModePromptRequest,
    ) -> Result<lash_plugin_plan_mode::PlanModePromptResponse, PluginError> {
        Err(PluginError::Session("prompt unavailable".to_string()))
    }
}

macro_rules! impl_plan_test_host {
    ($ty:ty) => {
        #[async_trait::async_trait]
        impl lash_core::plugin::runtime_host::RuntimeSessionHost for $ty {
            async fn snapshot_current(&self) -> Result<SessionSnapshot, PluginError> {
                PlanTestHostCore::snapshot_current(self).await
            }
            async fn snapshot_session(
                &self,
                session_id: &str,
            ) -> Result<SessionSnapshot, PluginError> {
                PlanTestHostCore::snapshot_session(self, session_id).await
            }
            async fn tool_catalog(
                &self,
                session_id: &str,
            ) -> Result<Vec<serde_json::Value>, PluginError> {
                PlanTestHostCore::tool_catalog(self, session_id).await
            }
            async fn tool_state(
                &self,
                session_id: &str,
            ) -> Result<lash_core::ToolState, PluginError> {
                PlanTestHostCore::tool_state(self, session_id).await
            }
            async fn apply_tool_state(
                &self,
                session_id: &str,
                snapshot: lash_core::ToolState,
            ) -> Result<u64, PluginError> {
                PlanTestHostCore::apply_tool_state(self, session_id, snapshot).await
            }
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
        impl lash_plugin_plan_mode::PlanModePrompt for $ty {
            async fn prompt_user(
                &self,
                request: lash_plugin_plan_mode::PlanModePromptRequest,
            ) -> Result<lash_plugin_plan_mode::PlanModePromptResponse, PluginError> {
                PlanTestHostCore::prompt_user(self, request).await
            }
        }
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
    availability: lash_core::ToolAvailabilityConfig,
    execution_mode: lash_core::ToolExecutionMode,
) -> ToolDefinition {
    ToolDefinition::raw(
        name,
        description,
        ToolDefinition::default_input_schema(),
        serde_json::json!({ "type": "object", "additionalProperties": true }),
    )
    .with_availability(availability)
    .with_execution_mode(execution_mode)
}

struct PlanModeTestTools;

#[async_trait::async_trait]
impl ToolProvider for PlanModeTestTools {
    fn tool_manifests(&self) -> Vec<ToolManifest> {
        vec![plan_mode_test_tool_definition().manifest()]
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<ToolContract>> {
        (name == "plan_exit").then(|| Arc::new(plan_mode_test_tool_definition().contract()))
    }

    async fn execute(&self, call: lash_core::ToolCall<'_>) -> ToolResult {
        ToolResult::err_fmt(format_args!("unexpected tool call: {}", call.name))
    }
}

fn plan_mode_test_tool_definition() -> ToolDefinition {
    test_tool(
        "plan_exit",
        "Ask whether to exit plan mode.",
        lash_core::ToolAvailabilityConfig::off(),
        lash_core::ToolExecutionMode::Parallel,
    )
    .with_examples(vec!["plan_exit()".to_string()])
}

fn mock_session_manager(run_session_id: &str) -> MockSessionManager {
    MockSessionManager::default()
        .with_snapshot(mock_snapshot(run_session_id))
        .with_turn(mock_assembled_turn(run_session_id, ""))
        .with_tool_registry(
            ToolRegistry::from_tool_provider(Arc::new(PlanModeTestTools))
                .expect("plan mode tool registry"),
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
- crates/lash/src/plugin_builtin/plan_mode.rs
- crates/lash-tui-extensions/src/lib.rs

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
    let mut factories = lash_core::testing::test_mode_factories();
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
        .invoke_plugin_action("plan_mode.enable", json!({}), None, true, manager.clone())
        .await
        .expect("enable");
    assert!(enabled.is_success());
    let enabled_value = enabled.value_for_projection();
    assert_eq!(
        enabled_value.get("enabled").and_then(|v| v.as_bool()),
        Some(true)
    );
    assert_eq!(
        enabled_value
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
        .invoke_plugin_action("plan_mode.toggle", json!({}), None, true, manager)
        .await
        .expect("toggle restored");
    let restored_toggle_value = restored_toggle.value_for_projection();
    assert_eq!(
        restored_toggle_value
            .get("enabled")
            .and_then(|v| v.as_bool()),
        Some(false)
    );

    restored
        .restore(&lash_core::PluginSessionSnapshot::default())
        .expect("reset restore");
    let reset_toggle = restored
        .invoke_plugin_action(
            "plan_mode.toggle",
            json!({}),
            None,
            true,
            Arc::new(mock_session_manager("run-session")),
        )
        .await
        .expect("toggle reset");
    let reset_toggle_value = reset_toggle.value_for_projection();
    assert_eq!(
        reset_toggle_value.get("enabled").and_then(|v| v.as_bool()),
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
        .tool_state("root")
        .await
        .expect("initial tool state");
    assert!(initial.get("plan_exit").is_some_and(|tool| {
        tool.manifest()
            .effective_availability(&lash_core::ExecutionMode::standard())
            == lash_core::ToolAvailability::Off
    }));

    session
        .invoke_plugin_action(
            "plan_mode.enable",
            json!({}),
            None,
            true,
            manager_host.clone(),
        )
        .await
        .expect("enable");

    let enabled = manager
        .tool_state("root")
        .await
        .expect("enabled tool state");
    assert!(enabled.get("plan_exit").is_some_and(|tool| {
        tool.manifest()
            .effective_availability(&lash_core::ExecutionMode::standard())
            == lash_core::ToolAvailability::Showcased
    }));

    session
        .invoke_plugin_action("plan_mode.disable", json!({}), None, true, manager_host)
        .await
        .expect("disable");

    let disabled = manager
        .tool_state("root")
        .await
        .expect("disabled tool state");
    assert!(disabled.get("plan_exit").is_some_and(|tool| {
        tool.manifest()
            .effective_availability(&lash_core::ExecutionMode::standard())
            == lash_core::ToolAvailability::Off
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
        .invoke_plugin_action("plan_mode.enable", json!({}), None, true, manager.clone())
        .await
        .expect("enable");

    let before_turn = session
        .before_turn(TurnHookContext {
            session_id: "root".to_string(),
            state: mock_read_view("run-session"),
            host: manager.clone(),
            turn_context: lash_core::TurnContext::default(),
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
                    && message.content.contains("host can surface the file path"))
    )));
    assert!(before_turn.iter().any(|emitted| {
        emitted.plugin_id == "plan_mode"
            && matches!(
                &emitted.value,
                PluginDirective::EmitRuntimeEvents { events }
                    if events.iter().any(|event| matches!(
                        event,
                        lash_core::plugin::PluginRuntimeEvent::Custom { name, payload }
                            if name == "plan_mode.state"
                                && payload.get("enabled").and_then(|value| value.as_bool()) == Some(true)
                                && payload.get("plan_path").and_then(|value| value.as_str()) == Some(".lash/plans/run-session.md")
                    ))
            )
    }));
    assert!(plan_file_path(temp.path(), "run-session").is_file());

    let blocked_exec = session
        .before_tool_call(ToolCallHookContext::new(
            "root".to_string(),
            "exec_command".to_string(),
            json!({"cmd":"cargo test -q"}),
            lash_core::ToolArgumentProjectionPolicy::default(),
            lash_core::TurnContext::default(),
            manager.clone(),
        ))
        .await
        .expect("before_tool_call");
    assert!(!blocked_exec.is_empty());

    let allowed = session
        .before_tool_call(ToolCallHookContext::new(
            "root".to_string(),
            "read_file".to_string(),
            json!({"path":"src/main.rs"}),
            lash_core::ToolArgumentProjectionPolicy::default(),
            lash_core::TurnContext::default(),
            manager.clone(),
        ))
        .await
        .expect("before_tool_call");
    assert!(allowed.is_empty());

    let tools = vec![
        test_tool(
            "search_tools",
            "Discover tools",
            lash_core::ToolAvailabilityConfig::callable(),
            lash_core::ToolExecutionMode::Parallel,
        ),
        test_tool(
            "read_file",
            "Read files",
            lash_core::ToolAvailabilityConfig::showcased(),
            lash_core::ToolExecutionMode::Parallel,
        ),
        test_tool(
            "search_web",
            "Search the web",
            lash_core::ToolAvailabilityConfig::showcased(),
            lash_core::ToolExecutionMode::Parallel,
        ),
        test_tool(
            "apply_patch",
            "Apply patches",
            lash_core::ToolAvailabilityConfig::showcased(),
            lash_core::ToolExecutionMode::Serial,
        ),
        test_tool(
            "plan_exit",
            "Exit plan mode",
            lash_core::ToolAvailabilityConfig::off(),
            lash_core::ToolExecutionMode::Parallel,
        ),
    ];
    let contracts = tools
        .iter()
        .map(|tool| (tool.name.clone(), Arc::new(tool.contract())))
        .collect::<std::collections::BTreeMap<_, _>>();
    let manifests = tools.into_iter().map(|tool| tool.manifest()).collect();
    let surface = session
        .resolve_tool_surface(ToolSurfaceContext {
            session_id: "root".to_string(),
            mode: ExecutionMode::standard(),
            tools: manifests,
            resolve_contract: Some(Arc::new(move |name| contracts.get(name).cloned())),
            tool_access: lash_core::SessionToolAccess::default(),
            subagent: None,
        })
        .expect("tool surface");
    assert!(
        surface
            .tools
            .iter()
            .find(|tool| tool.manifest.name == "search_web")
            .is_some_and(|tool| tool.availability == lash_core::ToolAvailability::Showcased)
    );
    assert!(
        surface
            .tools
            .iter()
            .find(|tool| tool.manifest.name == "plan_exit")
            .is_some_and(|tool| tool.availability == lash_core::ToolAvailability::Showcased)
    );
    assert!(
        surface
            .tool_list_notes
            .iter()
            .any(|note| note.contains(".lash/plans/run-session.md"))
    );

    let read_allowed = session
        .before_tool_call(ToolCallHookContext::new(
            "root".to_string(),
            "read_file".to_string(),
            json!({
                "path":"src/main.rs"
            }),
            lash_core::ToolArgumentProjectionPolicy::default(),
            lash_core::TurnContext::default(),
            manager.clone(),
        ))
        .await
        .expect("before_tool_call");
    assert!(read_allowed.is_empty());

    let web_allowed = session
        .before_tool_call(ToolCallHookContext::new(
            "root".to_string(),
            "search_web".to_string(),
            json!({
                "query":"surrealdb datetime best practices"
            }),
            lash_core::ToolArgumentProjectionPolicy::default(),
            lash_core::TurnContext::default(),
            manager.clone(),
        ))
        .await
        .expect("before_tool_call");
    assert!(web_allowed.is_empty());

    let plan_patch_allowed = session
        .before_tool_call(ToolCallHookContext::new("root".to_string(), "apply_patch".to_string(), json!({
                "input": "*** Begin Patch\n*** Add File: .lash/plans/run-session.md\n+# Plan\n*** End Patch"
            }), lash_core::ToolArgumentProjectionPolicy::default(), lash_core::TurnContext::default(), manager.clone()))
        .await
        .expect("before_tool_call");
    assert!(plan_patch_allowed.is_empty());

    let repo_patch_blocked = session
        .before_tool_call(ToolCallHookContext::new(
            "root".to_string(),
            "apply_patch".to_string(),
            json!({
                "input": "*** Begin Patch\n*** Add File: src/main.rs\n+fn main() {}\n*** End Patch"
            }),
            lash_core::ToolArgumentProjectionPolicy::default(),
            lash_core::TurnContext::default(),
            manager.clone(),
        ))
        .await
        .expect("before_tool_call");
    assert!(repo_patch_blocked.iter().any(|emitted| matches!(
        &emitted.value,
        PluginDirective::AbortTurn { code, message }
            if code == "plan_mode_tool_blocked"
                && message.contains(".lash/plans/run-session.md")
    )));

    session
        .invoke_plugin_action("plan_mode.disable", json!({}), None, true, manager.clone())
        .await
        .expect("disable");
    session
        .invoke_plugin_action("plan_mode.enable", json!({}), None, true, manager.clone())
        .await
        .expect("enable");

    let checkpoint = session
        .at_checkpoint(lash_core::CheckpointHookContext {
            session_id: "root".to_string(),
            checkpoint: lash_core::CheckpointKind::AfterWork,
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
                    && message.content.contains("host can surface the file path"))
    )));

    session
        .after_turn(TurnResultHookContext {
            session_id: "root".to_string(),
            turn: Arc::new(lash_core::TurnResultSummary::from_assembled(&empty_turn(
                "root",
            ))),
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
        .invoke_plugin_action("plan_mode.enable", json!({}), None, true, manager.clone())
        .await
        .expect("enable");

    let first_before_turn = session
        .before_turn(TurnHookContext {
            session_id: "root".to_string(),
            state: mock_read_view("run-session"),
            host: manager.clone(),
            turn_context: lash_core::TurnContext::default(),
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
            turn: Arc::new(lash_core::TurnResultSummary::from_assembled(&empty_turn(
                "root",
            ))),
            host: manager.clone(),
        })
        .await
        .expect("after_turn");

    let second_before_turn = session
        .before_turn(TurnHookContext {
            session_id: "root".to_string(),
            state: mock_read_view("run-session"),
            host: manager.clone(),
            turn_context: lash_core::TurnContext::default(),
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
        .invoke_plugin_action("plan_mode.enable", json!({}), None, true, manager.clone())
        .await
        .expect("enable");

    let allowed = session
        .before_tool_call(ToolCallHookContext::new(
            "root".to_string(),
            "read_file".to_string(),
            json!({"path":"src/main.rs"}),
            lash_core::ToolArgumentProjectionPolicy::default(),
            lash_core::TurnContext::default(),
            manager.clone(),
        ))
        .await
        .expect("before_tool_call");
    assert!(allowed.is_empty());

    let apply_patch_allowed = session
        .before_tool_call(ToolCallHookContext::new("root".to_string(), "apply_patch".to_string(), json!({
                "input": "*** Begin Patch\n*** Add File: .lash/plans/run-session.md\n+# Plan\n*** End Patch"
            }), lash_core::ToolArgumentProjectionPolicy::default(), lash_core::TurnContext::default(), manager.clone()))
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

        async fn tool_state(&self, session_id: &str) -> Result<lash_core::ToolState, PluginError> {
            self.base.tool_state(session_id).await
        }

        async fn apply_tool_state(
            &self,
            session_id: &str,
            snapshot: lash_core::ToolState,
        ) -> Result<u64, PluginError> {
            self.base.apply_tool_state(session_id, snapshot).await
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

        async fn prompt_user(
            &self,
            request: lash_plugin_plan_mode::PlanModePromptRequest,
        ) -> Result<lash_plugin_plan_mode::PlanModePromptResponse, PluginError> {
            assert!(request.question.contains(".lash/plans/run-session.md"));
            assert!(request.allow_note);
            let review = request.review.expect("plan review");
            assert_eq!(review.title, "PLAN");
            assert_eq!(review.markdown, ready_plan_markdown().trim_end());
            Ok(lash_plugin_plan_mode::PlanModePromptResponse::Single {
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
        .invoke_plugin_action(
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
            turn_context: lash_core::TurnContext::default(),
        })
        .await
        .expect("before_turn");

    let plan_exit_args = json!({});
    let plan_exit_ctx = lash_core::testing::mock_tool_context_with_host(manager_host.clone());
    let result = session
        .tools()
        .execute(lash_core::ToolCall {
            name: "plan_exit",
            args: &plan_exit_args,
            context: &plan_exit_ctx,
            progress: None,
        })
        .await;
    assert!(result.is_success());
    let result_value = result.value_for_projection();
    assert_eq!(
        result_value
            .get("approved")
            .and_then(|value| value.as_bool()),
        Some(true)
    );
    assert!(
        result_value
            .get("next_turn_input")
            .and_then(|value| value.as_str())
            .is_some_and(|value| {
                value.contains(
                    "The user approved the plan. Execute the plan in `.lash/plans/run-session.md` now"
                )
                    && value.contains("User note: start with the safe slice")
            })
    );

    let dynamic =
        lash_core::plugin::runtime_host::RuntimeSessionHost::tool_state(manager.as_ref(), "root")
            .await
            .expect("tool state");
    assert!(dynamic.get("plan_exit").is_some_and(|tool| {
        tool.manifest()
            .effective_availability(&lash_core::ExecutionMode::standard())
            == lash_core::ToolAvailability::Off
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

        async fn prompt_user(
            &self,
            request: lash_plugin_plan_mode::PlanModePromptRequest,
        ) -> Result<lash_plugin_plan_mode::PlanModePromptResponse, PluginError> {
            assert_eq!(
                request.question,
                "Review the plan in `.lash/plans/run-session.md`. What next?"
            );
            let review = request.review.expect("plan review");
            assert_eq!(review.title, "PLAN");
            assert!(review.markdown.contains("# Plan"));
            Ok(lash_plugin_plan_mode::PlanModePromptResponse::Single {
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
        .invoke_plugin_action("plan_mode.enable", json!({}), None, true, manager.clone())
        .await
        .expect("enable");

    session
        .before_turn(TurnHookContext {
            session_id: "root".to_string(),
            state: mock_read_view("run-session"),
            host: manager.clone(),
            turn_context: lash_core::TurnContext::default(),
        })
        .await
        .expect("before_turn");

    let plan_exit_args = json!({});
    let plan_exit_ctx = lash_core::testing::mock_tool_context_with_host(manager.clone());
    let result = session
        .tools()
        .execute(lash_core::ToolCall {
            name: "plan_exit",
            args: &plan_exit_args,
            context: &plan_exit_ctx,
            progress: None,
        })
        .await;
    assert!(result.is_success());
    let result_value = result.value_for_projection();
    assert!(
        result_value
            .get("approved")
            .and_then(|value| value.as_bool())
            == Some(true)
    );
    assert_eq!(
        result_value
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

        async fn prompt_user(
            &self,
            _request: lash_plugin_plan_mode::PlanModePromptRequest,
        ) -> Result<lash_plugin_plan_mode::PlanModePromptResponse, PluginError> {
            Ok(lash_plugin_plan_mode::PlanModePromptResponse::Single {
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
        .invoke_plugin_action("plan_mode.enable", json!({}), None, true, manager.clone())
        .await
        .expect("enable");

    session
        .before_turn(TurnHookContext {
            session_id: "root".to_string(),
            state: mock_read_view("run-session"),
            host: manager.clone(),
            turn_context: lash_core::TurnContext::default(),
        })
        .await
        .expect("before_turn");

    let plan_exit_args = json!({});
    let plan_exit_ctx = lash_core::testing::mock_tool_context_with_host(manager.clone());
    let result = session
        .tools()
        .execute(lash_core::ToolCall {
            name: "plan_exit",
            args: &plan_exit_args,
            context: &plan_exit_ctx,
            progress: None,
        })
        .await;
    assert!(result.is_success());
    let result_value = result.value_for_projection();
    assert_eq!(
        result_value
            .get("execution_mode")
            .and_then(|value| value.as_str()),
        Some("fresh_context")
    );
    assert!(
        result_value.get("fresh_context_input").is_none(),
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
    }

    let _guard = plan_mode_env_lock().lock().await;
    let temp = tempfile::tempdir().expect("tempdir");
    let _cwd = CurrentDirGuard::set(temp.path());
    let host = plan_mode_host(PlanModePluginFactory::default());
    let session = host.build_standard_session("root", None).expect("session");
    let manager = Arc::new(CapturingSessionManager::default());

    session
        .invoke_plugin_action("plan_mode.enable", json!({}), None, true, manager.clone())
        .await
        .expect("enable");

    let directives = session
        .after_tool_call(ToolResultHookContext::new(
            "root".to_string(),
            "plan_exit".to_string(),
            json!({}),
            ToolResult::ok(json!({
                "approved": true,
                "plan_path": ".lash/plans/run-session.md",
                "execution_mode": "fresh_context",
            })),
            1,
            lash_core::TurnContext::default(),
            manager.clone(),
        ))
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
    assert_eq!(seed.role, lash_core::MessageRole::User);
    assert_eq!(
        seed.content,
        "Do a full, faithful implementation of the plan found at: .lash/plans/run-session.md"
    );
    assert!(
        directives
            .iter()
            .any(|owned| matches!(owned.value, PluginDirective::EmitRuntimeEvents { .. }))
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
        .invoke_plugin_action("plan_mode.enable", json!({}), None, true, manager.clone())
        .await
        .expect("enable");

    let stream = session
        .transform_assistant_stream("root", "Keep this text exactly.".to_string())
        .await
        .expect("stream");
    assert!(stream.is_empty());

    let response = session
        .transform_assistant_response(
            "root",
            lash_core::llm::types::LlmResponse {
                full_text: "Keep this text exactly.".into(),
                parts: vec![lash_core::llm::types::LlmOutputPart::Text {
                    text: "Keep this text exactly.".into(),
                    response_meta: None,
                }],
                usage: lash_core::llm::types::LlmUsage::default(),
                terminal_reason: lash_core::LlmTerminalReason::Stop,
                terminal_diagnostic: None,
                provider_usage: None,
                request_body: None,
                http_summary: None,
            },
        )
        .await
        .expect("response");
    assert!(response.is_empty());
}
