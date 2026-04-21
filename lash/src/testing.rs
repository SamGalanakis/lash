//! In-tree test fixtures shared across the lash crate's test modules.
//!
//! Cuts down on per-test-module `MockSessionManager` boilerplate by
//! providing a configurable mock implementation plus a couple of small
//! builders for common policy / turn fixtures.

use std::sync::Mutex;

use crate::plugin::{
    PluginError, SessionCreateRequest, SessionHandle, SessionManager, SessionSnapshot,
    SessionTurnHandle,
};
use crate::{
    AssembledTurn, AssistantOutput, DoneReason, ExecutionMode, ExecutionSummary, OutputState,
    PersistedSessionState, Provider, ProviderOptions, SessionPolicy, SessionStateEnvelope,
    TokenUsage, TurnInput, TurnStatus,
};

/// Build a `SessionPolicy` populated with the canonical mock provider
/// + model used by lash's in-tree tests.
pub fn mock_session_policy() -> SessionPolicy {
    SessionPolicy {
        provider: Provider::OpenAiGeneric {
            api_key: String::new(),
            base_url: "https://example.invalid/v1".to_string(),
            options: ProviderOptions::default(),
        },
        model: "mock-model".to_string(),
        execution_mode: ExecutionMode::Standard,
        ..Default::default()
    }
}

/// Build an empty `AssembledTurn` whose assistant text is `summary`.
pub fn mock_assembled_turn(session_id: &str, summary: &str) -> AssembledTurn {
    AssembledTurn {
        state: SessionStateEnvelope {
            session_id: session_id.to_string(),
            policy: SessionPolicy {
                execution_mode: ExecutionMode::Standard,
                ..Default::default()
            },
            ..Default::default()
        },
        status: TurnStatus::Completed,
        assistant_output: AssistantOutput {
            safe_text: summary.to_string(),
            raw_text: summary.to_string(),
            state: OutputState::Usable,
        },
        has_plugin_visible_output: false,
        done_reason: DoneReason::ModelStop,
        execution: ExecutionSummary {
            mode: ExecutionMode::Standard,
            had_tool_calls: false,
            had_code_execution: false,
        },
        token_usage: TokenUsage::default(),
        tool_calls: Vec::new(),
        errors: Vec::new(),
        typed_finish: None,
    }
}

/// Configurable mock for the [`SessionManager`] trait. Tests override
/// the snapshot, tool catalog, and turn outcome via the builder
/// methods; mutations (`create_session`, `cancel_turn`, `close_session`)
/// are recorded so tests can assert against them.
pub struct MockSessionManager {
    pub snapshot: SessionSnapshot,
    pub tool_catalog: Vec<serde_json::Value>,
    pub turn: AssembledTurn,
    pub dynamic_tools: Option<crate::DynamicToolProvider>,
    pub created: Mutex<Vec<SessionCreateRequest>>,
    pub cancelled: Mutex<Vec<String>>,
    pub closed: Mutex<Vec<String>>,
}

impl Default for MockSessionManager {
    fn default() -> Self {
        Self {
            snapshot: PersistedSessionState::default(),
            tool_catalog: Vec::new(),
            turn: mock_assembled_turn("root", ""),
            dynamic_tools: None,
            created: Mutex::new(Vec::new()),
            cancelled: Mutex::new(Vec::new()),
            closed: Mutex::new(Vec::new()),
        }
    }
}

impl MockSessionManager {
    #[allow(dead_code)]
    pub fn with_snapshot(mut self, snapshot: SessionSnapshot) -> Self {
        self.snapshot = snapshot;
        self
    }

    pub fn with_tool_catalog(mut self, catalog: Vec<serde_json::Value>) -> Self {
        self.tool_catalog = catalog;
        self
    }

    pub fn with_turn(mut self, turn: AssembledTurn) -> Self {
        self.turn = turn;
        self
    }

    #[allow(dead_code)]
    pub fn with_dynamic_tool_provider(mut self, dynamic_tools: crate::DynamicToolProvider) -> Self {
        self.dynamic_tools = Some(dynamic_tools);
        self
    }

    /// Snapshot of the requests captured by `create_session`. Panics if
    /// the lock is poisoned (a panic from another test thread).
    pub fn created_snapshot(&self) -> Vec<SessionCreateRequest> {
        self.created.lock().expect("created lock").clone()
    }
}

#[async_trait::async_trait]
impl SessionManager for MockSessionManager {
    async fn snapshot_current(&self) -> Result<SessionSnapshot, PluginError> {
        Ok(self.snapshot.clone())
    }

    async fn snapshot_session(&self, _session_id: &str) -> Result<SessionSnapshot, PluginError> {
        Ok(self.snapshot.clone())
    }

    async fn tool_catalog(&self, _session_id: &str) -> Result<Vec<serde_json::Value>, PluginError> {
        Ok(self.tool_catalog.clone())
    }

    async fn dynamic_tool_state(
        &self,
        _session_id: &str,
    ) -> Result<crate::DynamicStateSnapshot, PluginError> {
        self.dynamic_tools
            .as_ref()
            .map(crate::DynamicToolProvider::export_state)
            .ok_or_else(|| {
                PluginError::Session(
                    "dynamic tool state is unavailable in this session".to_string(),
                )
            })
    }

    async fn apply_dynamic_tool_state(
        &self,
        _session_id: &str,
        snapshot: crate::DynamicStateSnapshot,
    ) -> Result<u64, PluginError> {
        let Some(dynamic_tools) = self.dynamic_tools.as_ref() else {
            return Err(PluginError::Session(
                "dynamic tool state mutation is unavailable in this session".to_string(),
            ));
        };
        dynamic_tools
            .apply_state(snapshot)
            .map_err(|err| PluginError::Session(err.to_string()))
    }

    async fn create_session(
        &self,
        request: SessionCreateRequest,
    ) -> Result<SessionHandle, PluginError> {
        self.created
            .lock()
            .expect("created lock")
            .push(request.clone());
        Ok(SessionHandle {
            session_id: request
                .session_id
                .clone()
                .unwrap_or_else(|| "child".to_string()),
            parent_session_id: request.parent_session_id.clone(),
            policy: request.policy.unwrap_or_else(mock_session_policy),
        })
    }

    async fn close_session(&self, session_id: &str) -> Result<(), PluginError> {
        self.closed
            .lock()
            .expect("closed lock")
            .push(session_id.to_string());
        Ok(())
    }

    async fn start_turn_stream(
        &self,
        session_id: &str,
        _input: TurnInput,
    ) -> Result<SessionTurnHandle, PluginError> {
        let (_tx, rx) = tokio::sync::mpsc::channel(1);
        Ok(SessionTurnHandle {
            turn_id: format!("{session_id}-turn"),
            session_id: session_id.to_string(),
            policy: mock_session_policy(),
            events: rx,
        })
    }

    async fn await_turn(&self, _turn_id: &str) -> Result<AssembledTurn, PluginError> {
        Ok(self.turn.clone())
    }

    async fn cancel_turn(&self, turn_id: &str) -> Result<(), PluginError> {
        self.cancelled
            .lock()
            .expect("cancelled lock")
            .push(turn_id.to_string());
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────
// Minimal in-tree plugin fake advertising support for a given
// `ContextApproachKind`. Lash tests use this instead of pulling in
// `lash-plugin-rolling-history` / `lash-plugin-observational-memory`
// as dev-deps, which would create a dev-dep cycle.
// ─────────────────────────────────────────────────────────────────────

use std::sync::Arc;

use crate::context_approach::ContextApproachKind;
use crate::plugin::{PluginFactory, PluginSessionContext, PluginSpec, SessionPlugin};

pub struct FakeContextApproachPluginFactory {
    id: &'static str,
    approaches: &'static [ContextApproachKind],
}

impl FakeContextApproachPluginFactory {
    pub fn rolling_history() -> Self {
        Self {
            id: "fake_rolling_history",
            approaches: &[ContextApproachKind::RollingHistory],
        }
    }

    pub fn observational_memory() -> Self {
        Self {
            id: "fake_observational_memory",
            approaches: &[ContextApproachKind::ObservationalMemory],
        }
    }
}

impl PluginFactory for FakeContextApproachPluginFactory {
    fn id(&self) -> &'static str {
        self.id
    }

    fn supported_context_approaches(&self) -> &'static [ContextApproachKind] {
        self.approaches
    }

    fn build(
        &self,
        ctx: &PluginSessionContext,
    ) -> Result<Arc<dyn SessionPlugin>, crate::plugin::PluginError> {
        crate::plugin::StaticPluginFactory::new(self.id, PluginSpec::new()).build(ctx)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Test mode plugin fakes.
//
// Exposed publicly under the `testing` feature so downstream plugin
// crates (e.g. `lash-plugin-plan-mode`) can wire minimal fake mode
// plugins into their integration tests without depending on the real
// `lash-mode-standard` / `lash-mode-rlm` crates (which would create a
// dev-dep cycle through the plugin crates those modes already
// include).
// ─────────────────────────────────────────────────────────────────────
pub use test_mode_fakes::test_mode_factories;

mod test_mode_fakes {
    use std::sync::Arc;

    use async_trait::async_trait;

    use super::*;
    use crate::plugin::{
        ModeProtocolDriverPlugin, ModeRuntimeContext, ModeSessionContext, ModeSessionPlugin,
        PluginFactory, PluginRegistrar, PluginSessionContext, SessionPlugin,
    };
    use crate::sansio::{
        CompletedToolCall, DriverAction, DriverContextView, ProtocolDriverHandle, WaitingExecState,
        WaitingLlmState,
    };
    use crate::{ExecResponse, ModeBuildInput, ModeConfig, ModePreamble};
    use lash_sansio::llm::types::LlmResponse;

    /// Factories that register minimal fake mode plugins for lash's own
    /// unit tests and downstream plugin crate integration tests.
    /// Production callers embed the real `lash-mode-standard` /
    /// `lash-mode-rlm` crates instead.
    pub fn test_mode_factories() -> Vec<Arc<dyn PluginFactory>> {
        vec![
            Arc::new(TestModeFactory {
                id: "mode_standard",
                mode: ExecutionMode::Standard,
            }),
            Arc::new(TestModeFactory {
                id: "mode_rlm",
                mode: ExecutionMode::Rlm,
            }),
        ]
    }

    struct TestModeFactory {
        id: &'static str,
        mode: ExecutionMode,
    }

    impl PluginFactory for TestModeFactory {
        fn id(&self) -> &'static str {
            self.id
        }

        fn build(&self, ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
            Ok(Arc::new(TestModePlugin {
                id: self.id,
                active: ctx.execution_mode == self.mode,
                mode: self.mode,
            }))
        }
    }

    struct TestModePlugin {
        id: &'static str,
        active: bool,
        mode: ExecutionMode,
    }

    impl SessionPlugin for TestModePlugin {
        fn id(&self) -> &'static str {
            self.id
        }

        fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
            if !self.active {
                return Ok(());
            }
            reg.mode()
                .session(Arc::new(TestModeSession { mode: self.mode }))?;
            reg.mode()
                .native_tools(Arc::new(TestModeNativeTools { mode: self.mode }))?;
            reg.mode()
                .protocol_driver(Arc::new(TestProtocolDriver { mode: self.mode }))?;
            Ok(())
        }
    }

    struct TestModeSession {
        mode: ExecutionMode,
    }

    #[async_trait]
    impl ModeSessionPlugin for TestModeSession {
        async fn initialize_session(
            &self,
            mut ctx: ModeSessionContext<'_>,
        ) -> Result<(), crate::SessionError> {
            if matches!(self.mode, ExecutionMode::Rlm) {
                ctx.start_lashlang_runtime().await?;
            }
            Ok(())
        }

        fn configure_runtime_from_request(
            &self,
            mut ctx: ModeRuntimeContext<'_>,
            request: &crate::SessionCreateRequest,
        ) {
            if matches!(self.mode, ExecutionMode::Rlm)
                && let crate::ModeExtras::Rlm(extras) = &request.mode_extras
            {
                ctx.set_termination_mode(extras.termination.clone());
            }
        }
    }

    struct TestModeNativeTools {
        mode: ExecutionMode,
    }

    #[async_trait]
    impl crate::plugin::ModeNativeToolsPlugin for TestModeNativeTools {
        fn definitions(&self) -> Vec<crate::ToolDefinition> {
            use crate::runtime_controls::{
                monitor_tool_definition, tasks_list_tool_definition, tasks_stop_tool_definition,
            };
            use crate::tools::batch::batch_tool_definition;
            let mut tools = vec![
                monitor_tool_definition(),
                tasks_list_tool_definition(),
                tasks_stop_tool_definition(),
            ];
            if self.mode == ExecutionMode::Standard {
                tools.insert(0, batch_tool_definition());
            }
            tools
        }

        async fn execute(
            &self,
            context: &crate::tool_dispatch::ToolDispatchContext,
            name: &str,
            args: &serde_json::Value,
            progress: Option<&crate::ProgressSender>,
        ) -> Option<crate::ToolResult> {
            use crate::runtime_controls::{
                MonitorToolSpec, execute_monitor_tool_call, execute_tasks_list_tool_call,
                execute_tasks_stop_tool_call,
            };
            match name {
                "batch" if self.mode == ExecutionMode::Standard => {
                    Some(execute_test_batch(context, args, progress).await)
                }
                "monitor" => {
                    let spec = match MonitorToolSpec::from_args(args) {
                        Ok(spec) => spec,
                        Err(result) => return Some(result),
                    };
                    Some(execute_monitor_tool_call(context, spec).await)
                }
                "tasks_list" => Some(execute_tasks_list_tool_call(context).await),
                "tasks_stop" => Some(execute_tasks_stop_tool_call(context, args).await),
                _ => None,
            }
        }
    }

    /// Minimal batch executor used by lash's own tests (mirrors the
    /// behavior of `lash-mode-standard`'s `execute_batch_tool_call`).
    async fn execute_test_batch(
        context: &crate::tool_dispatch::ToolDispatchContext,
        args: &serde_json::Value,
        progress: Option<&crate::ProgressSender>,
    ) -> crate::ToolResult {
        use crate::tool_dispatch::{ParallelToolCallSpec, dispatch_parallel_tool_calls};

        const MAX: usize = 25;
        let Some(raw_calls) = args.get("tool_calls").and_then(|v| v.as_array()) else {
            return crate::ToolResult::err_fmt("Missing required parameter: tool_calls");
        };
        if raw_calls.is_empty() {
            return crate::ToolResult::err_fmt("Invalid tool_calls: expected at least one call");
        }

        let mut results = Vec::new();
        let mut parallel_specs = Vec::new();
        for (index, item) in raw_calls.iter().enumerate().take(MAX) {
            let Some(obj) = item.as_object() else {
                return crate::ToolResult::err_fmt(format_args!(
                    "Invalid tool_calls[{index}]: expected object with tool and parameters"
                ));
            };
            let Some(tool) = obj
                .get("tool")
                .and_then(|v| v.as_str())
                .map(str::trim)
                .filter(|t| !t.is_empty())
            else {
                return crate::ToolResult::err_fmt(format_args!(
                    "Invalid tool_calls[{index}].tool: expected non-empty string"
                ));
            };
            if tool == "batch" {
                results.push(serde_json::json!({
                    "index": index,
                    "tool": tool,
                    "success": false,
                    "duration_ms": 0,
                    "error": "Tool 'batch' is not allowed inside batch",
                }));
                continue;
            }
            let parameters = obj
                .get("parameters")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({}));
            parallel_specs.push(ParallelToolCallSpec {
                index,
                tool_name: tool.to_string(),
                args: parameters,
            });
        }

        let mut images = Vec::new();
        let outcomes =
            dispatch_parallel_tool_calls(Arc::new(context.clone()), parallel_specs, progress).await;
        for outcome in outcomes {
            images.extend(outcome.images);
            let mut record = serde_json::Map::new();
            record.insert("index".to_string(), serde_json::json!(outcome.index));
            record.insert("tool".to_string(), serde_json::json!(outcome.record.tool));
            record.insert(
                "success".to_string(),
                serde_json::json!(outcome.record.success),
            );
            record.insert(
                "duration_ms".to_string(),
                serde_json::json!(outcome.record.duration_ms),
            );
            record.insert(
                if outcome.record.success {
                    "result"
                } else {
                    "error"
                }
                .to_string(),
                outcome.record.result,
            );
            results.push(serde_json::Value::Object(record));
        }

        for overflow_index in MAX..raw_calls.len() {
            results.push(serde_json::json!({
                "index": overflow_index,
                "tool": raw_calls
                    .get(overflow_index)
                    .and_then(|item| item.get("tool"))
                    .and_then(|value| value.as_str())
                    .unwrap_or("unknown"),
                "success": false,
                "duration_ms": 0,
                "error": "Maximum of 25 tool calls allowed in batch",
            }));
        }

        results.sort_by_key(|r| {
            r.get("index")
                .and_then(|value| value.as_u64())
                .unwrap_or(u64::MAX)
        });
        crate::ToolResult::with_images(true, serde_json::json!({ "results": results }), images)
    }

    struct TestProtocolDriver {
        mode: ExecutionMode,
    }

    impl ModeProtocolDriverPlugin for TestProtocolDriver {
        fn mode_id(&self) -> &'static str {
            self.mode.plugin_id()
        }

        fn build_preamble(&self, input: ModeBuildInput) -> ModePreamble {
            ModePreamble {
                config: ModeConfig {
                    protocol: Arc::new(TestDriver),
                    sync_execution_surface: false,
                },
                tool_specs: Arc::new(input.tool_surface.model_tool_specs()),
                tool_names: input.tool_surface.tool_names(),
                omitted_tool_count: 0,
                execution_prompt: String::new(),
                prompt_contributions: input.extra_prompt_contributions,
            }
        }
    }

    /// Minimal Standard-style driver used by lash's own test suite. Mirrors
    /// the parts of the real `lash-mode-standard::StandardDriver` that
    /// production tests depend on: extract tool calls + assistant text from
    /// the LLM response, append the assistant message, dispatch tools, and
    /// finish-checkpoint when there are no tools. Reasoning parts are
    /// surfaced but without the interleave ordering the real driver uses —
    /// no test asserts that ordering.
    struct TestDriver;

    impl ProtocolDriverHandle for TestDriver {
        fn prepare_iteration(&self, ctx: DriverContextView<'_>) -> Vec<DriverAction> {
            vec![DriverAction::StartLlm {
                request: ctx.build_llm_request(true),
                driver_state: None,
            }]
        }

        fn handle_llm_success(
            &self,
            ctx: DriverContextView<'_>,
            _waiting: WaitingLlmState,
            llm_response: LlmResponse,
            text_streamed: bool,
        ) -> Vec<DriverAction> {
            use crate::sansio::{CheckpointResumeAction, PendingToolCall};
            use crate::{
                CheckpointKind, Message, MessageRole, Part, PartKind, PruneState, SessionEvent,
            };
            use lash_sansio::llm::types::LlmOutputPart;
            use lash_sansio::session_model::fresh_message_id;
            use lash_sansio::session_model::make_error_event;

            let parts = crate::normalized_response_parts(&llm_response);
            let mut assistant_text = String::new();
            let mut tool_calls: Vec<(String, String, String, Option<String>)> = Vec::new();
            let mut actions = Vec::new();

            for part in parts {
                match part {
                    LlmOutputPart::Text { text } => {
                        if !text.is_empty() {
                            let previous_len = assistant_text.len();
                            crate::append_assistant_text_part(&mut assistant_text, &text);
                            if !text_streamed {
                                actions.push(DriverAction::Emit(SessionEvent::TextDelta {
                                    content: assistant_text[previous_len..].to_string(),
                                }));
                            }
                        }
                    }
                    LlmOutputPart::Reasoning { .. } => {}
                    LlmOutputPart::ToolCall {
                        call_id,
                        tool_name,
                        input_json,
                        item_id,
                        signature: _,
                    } => {
                        tool_calls.push((call_id, tool_name, input_json, item_id));
                    }
                }
            }

            actions.push(DriverAction::Emit(SessionEvent::LlmResponse {
                iteration: ctx.iteration(),
                content: assistant_text.clone(),
                duration_ms: 0,
            }));

            if tool_calls.is_empty() {
                if assistant_text.trim().is_empty() {
                    actions.push(DriverAction::Emit(make_error_event(
                        "llm_provider",
                        Some("empty_response"),
                        "Model returned no assistant text or tool calls.",
                        None,
                    )));
                    actions.push(DriverAction::Finish);
                    return actions;
                }
                let asst_id = fresh_message_id();
                let parts_out = vec![Part {
                    id: format!("{asst_id}.p0"),
                    kind: PartKind::Prose,
                    content: assistant_text,
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    tool_item_id: None,
                    tool_signature: None,
                    prune_state: PruneState::Intact,
                    reasoning_meta: None,
                }];
                actions.push(DriverAction::AppendMessages(vec![Message {
                    id: asst_id,
                    role: MessageRole::Assistant,
                    parts: parts_out,
                    user_input: None,
                    origin: None,
                }]));
                actions.push(DriverAction::StartCheckpoint {
                    checkpoint: CheckpointKind::BeforeCompletion,
                    on_empty: CheckpointResumeAction::Finish,
                });
                return actions;
            }

            let asst_id = fresh_message_id();
            let mut assistant_parts = Vec::new();
            if !assistant_text.trim().is_empty() {
                assistant_parts.push(Part {
                    id: format!("{}.p{}", asst_id, assistant_parts.len()),
                    kind: PartKind::Prose,
                    content: assistant_text,
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    tool_item_id: None,
                    tool_signature: None,
                    prune_state: PruneState::Intact,
                    reasoning_meta: None,
                });
            }
            let mut calls = Vec::new();
            for (call_id, tool_name, input_json, item_id) in tool_calls {
                assistant_parts.push(Part {
                    id: format!("{}.p{}", asst_id, assistant_parts.len()),
                    kind: PartKind::ToolCall,
                    content: input_json.clone(),
                    attachment: None,
                    tool_call_id: Some(call_id.clone()),
                    tool_name: Some(tool_name.clone()),
                    tool_item_id: item_id.clone(),
                    tool_signature: None,
                    prune_state: PruneState::Intact,
                    reasoning_meta: None,
                });
                let args = serde_json::from_str::<serde_json::Value>(&input_json)
                    .unwrap_or_else(|_| serde_json::json!({}));
                calls.push(PendingToolCall {
                    call_id,
                    tool_name,
                    args,
                    item_id,
                });
            }
            if !assistant_parts.is_empty() {
                actions.push(DriverAction::AppendMessages(vec![Message {
                    id: asst_id,
                    role: MessageRole::Assistant,
                    parts: assistant_parts,
                    user_input: None,
                    origin: None,
                }]));
            }
            actions.push(DriverAction::StartTools { calls });
            actions
        }

        fn handle_tool_results(
            &self,
            ctx: DriverContextView<'_>,
            completed: Vec<CompletedToolCall>,
        ) -> Vec<DriverAction> {
            use crate::sansio::CheckpointResumeAction;
            use crate::{
                CheckpointKind, Message, MessageRole, Part, PartKind, PruneState, SessionEvent,
            };
            use lash_sansio::session_model::{
                format_tool_result_content, fresh_message_id, reassign_part_ids,
            };
            let mut actions = Vec::new();
            let mut result_parts = Vec::new();
            for outcome in completed {
                result_parts.push(Part {
                    id: String::new(),
                    kind: PartKind::ToolResult,
                    content: format_tool_result_content(
                        outcome.model_result.success,
                        &outcome.model_result.result,
                    ),
                    attachment: None,
                    tool_call_id: Some(outcome.call_id.clone()),
                    tool_name: Some(outcome.tool_name.clone()),
                    tool_item_id: None,
                    tool_signature: None,
                    prune_state: PruneState::Intact,
                    reasoning_meta: None,
                });
            }
            if !result_parts.is_empty() {
                let user_id = fresh_message_id();
                reassign_part_ids(&user_id, &mut result_parts);
                actions.push(DriverAction::AppendMessages(vec![Message {
                    id: user_id,
                    role: MessageRole::User,
                    parts: result_parts,
                    user_input: None,
                    origin: None,
                }]));
            }
            actions.push(DriverAction::AdvanceIteration);
            let next_iteration = ctx.iteration() + 1;
            if let Some(max_turns) = ctx.max_turns()
                && next_iteration >= ctx.run_offset() + max_turns
            {
                actions.push(DriverAction::AppendMessages(vec![
                    crate::turn_limit_exhausted_message(max_turns),
                ]));
                actions.push(DriverAction::Finish);
                let _ = SessionEvent::Done;
                return actions;
            }
            actions.push(DriverAction::StartCheckpoint {
                checkpoint: CheckpointKind::AfterWork,
                on_empty: CheckpointResumeAction::PrepareIteration,
            });
            actions
        }

        fn handle_exec_result(
            &self,
            _ctx: DriverContextView<'_>,
            _waiting: WaitingExecState,
            _result: Result<ExecResponse, String>,
        ) -> Vec<DriverAction> {
            Vec::new()
        }
    }
} // mod test_mode_fakes
