use super::*;
use crate::llm::types::{LlmContentBlock, LlmMessage, LlmRole, LlmToolChoice};
use crate::plugin::{ModeProtocolDriverPlugin, ModeSessionPlugin};
use crate::sansio::{CompletedToolCall, ExecutionSurfaceSync, PendingToolCall};
use crate::{DirectCompletion, DirectLlmCompletion, DirectRequest, ExecResponse, PluginError};

#[derive(Clone, Debug)]
struct EffectHostRecord {
    kind: RuntimeEffectKind,
    origin: EffectOrigin,
    turn_id: Option<String>,
    effect_id: String,
    idempotency_key: String,
}

#[derive(Clone, Default)]
struct RecordingEffectHost {
    records: Arc<Mutex<Vec<EffectHostRecord>>>,
}

impl RecordingEffectHost {
    fn records(&self) -> Vec<EffectHostRecord> {
        self.records.lock().expect("effect records").clone()
    }

    fn count_kind(&self, kind: RuntimeEffectKind) -> usize {
        self.records()
            .iter()
            .filter(|record| record.kind == kind)
            .count()
    }

    fn record(&self, invocation: &EffectInvocation) {
        self.records
            .lock()
            .expect("effect records")
            .push(EffectHostRecord {
                kind: invocation.metadata.effect_kind,
                origin: invocation.metadata.origin.clone(),
                turn_id: invocation.metadata.turn_id.clone(),
                effect_id: invocation.metadata.effect_id.clone(),
                idempotency_key: invocation.metadata.idempotency_key.clone(),
            });
    }
}

#[async_trait::async_trait]
impl RuntimeEffectHost for RecordingEffectHost {
    async fn llm_call(
        &self,
        invocation: EffectInvocation,
        request: Arc<LlmRequest>,
        executor: TurnEffectLocalExecutor<'_, '_>,
    ) -> (Result<LlmResponse, LlmCallError>, bool) {
        self.record(&invocation);
        executor.llm_call(request).await
    }

    async fn direct_completion(
        &self,
        invocation: EffectInvocation,
        request: DirectRequest,
        normalized_request: LlmRequest,
        model: String,
        usage_source: String,
        mut executor: DirectEffectLocalExecutor,
    ) -> Result<DirectCompletion, PluginError> {
        self.record(&invocation);
        executor
            .direct_completion(request, normalized_request, model, usage_source)
            .await
    }

    async fn direct_llm_completion(
        &self,
        invocation: EffectInvocation,
        request: LlmRequest,
        usage_source: String,
        mut executor: DirectEffectLocalExecutor,
    ) -> Result<DirectLlmCompletion, PluginError> {
        self.record(&invocation);
        executor.direct_llm_completion(request, usage_source).await
    }

    async fn tool_call(
        &self,
        invocation: EffectInvocation,
        call: PendingToolCall,
        executor: TurnEffectLocalExecutor<'_, '_>,
    ) -> CompletedToolCall {
        self.record(&invocation);
        executor.tool_call(call, invocation.metadata).await
    }

    async fn exec_code(
        &self,
        invocation: EffectInvocation,
        code: String,
        executor: TurnEffectLocalExecutor<'_, '_>,
    ) -> Result<ExecResponse, String> {
        self.record(&invocation);
        executor.exec_code(code).await
    }

    async fn checkpoint(
        &self,
        invocation: EffectInvocation,
        checkpoint: CheckpointKind,
        executor: TurnEffectLocalExecutor<'_, '_>,
    ) -> Result<(Vec<PluginMessage>, Vec<PluginMessage>), RuntimeError> {
        self.record(&invocation);
        executor.checkpoint(checkpoint).await
    }

    async fn sync_execution_surface(
        &self,
        invocation: EffectInvocation,
        update_machine_config: bool,
        executor: TurnEffectLocalExecutor<'_, '_>,
    ) -> Result<Option<ExecutionSurfaceSync>, String> {
        self.record(&invocation);
        executor.sync_execution_surface(update_machine_config).await
    }

    async fn sleep(&self, invocation: EffectInvocation, _duration: std::time::Duration) {
        self.record(&invocation);
    }
}

fn host_with_effect_recorder(recorder: RecordingEffectHost) -> EmbeddedRuntimeHost {
    EmbeddedRuntimeHost::new(RuntimeCoreConfig::default().with_effect_host(Arc::new(recorder)))
}

fn effect_test_mode() -> ExecutionMode {
    ExecutionMode::new("effect_host_test")
}

#[tokio::test]
async fn standard_turn_llm_and_checkpoint_effects_cross_host_once() {
    let recorder = RecordingEffectHost::default();
    let transport = mock_provider(vec![MockCall {
        stream_events: Vec::new(),
        response: Ok(LlmResponse {
            full_text: "Done".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "Done".to_string(),
                response_meta: None,
            }],
            usage: LlmUsage {
                input_tokens: 3,
                output_tokens: 2,
                cached_input_tokens: 0,
                reasoning_tokens: 0,
            },
            ..LlmResponse::default()
        }),
    }]);
    let mut runtime = runtime_with_plugins_and_tools_and_host(
        Vec::new(),
        Arc::new(EmptyTools),
        transport,
        host_with_effect_recorder(recorder.clone()),
    )
    .await;

    let turn = runtime
        .run_turn_assembled(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "hello".to_string(),
                }],
                image_blobs: HashMap::new(),
                mode_turn_options: None,
                trace_turn_id: None,
                mode_extension: None,
                turn_context: crate::TurnContext::default(),
            },
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    assert!(matches!(turn.outcome, TurnOutcome::Finished(_)));
    assert_eq!(recorder.count_kind(RuntimeEffectKind::LlmCall), 1);
    assert_eq!(recorder.count_kind(RuntimeEffectKind::Checkpoint), 1);
    assert!(recorder.records().iter().all(|record| {
        record.origin == EffectOrigin::Turn && record.idempotency_key.starts_with("root:")
    }));
}

#[tokio::test]
async fn scoped_borrowed_effect_host_uses_required_stable_turn_id() {
    let recorder = RecordingEffectHost::default();
    assert!(RuntimeEffectScope::new(&recorder, "").is_err());
    let transport = mock_provider(vec![MockCall {
        stream_events: Vec::new(),
        response: Ok(LlmResponse {
            full_text: "Done".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "Done".to_string(),
                response_meta: None,
            }],
            ..LlmResponse::default()
        }),
    }]);
    let mut runtime = runtime_with_plugins_and_tools_and_host(
        Vec::new(),
        Arc::new(EmptyTools),
        transport,
        EmbeddedRuntimeHost::new(RuntimeCoreConfig::default()),
    )
    .await;

    let effect_scope =
        RuntimeEffectScope::new(&recorder, "stable-scoped-turn").expect("effect scope");
    let turn = runtime
        .stream_turn_with_effect_scope(
            TurnInput::text("hello"),
            &NoopEventSink,
            effect_scope,
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    assert!(matches!(turn.outcome, TurnOutcome::Finished(_)));
    assert!(
        recorder
            .records()
            .iter()
            .all(|record| record.idempotency_key.contains("stable-scoped-turn"))
    );
}

#[tokio::test]
async fn scoped_borrowed_effect_host_reaches_tool_direct_completions() {
    struct DirectTool;

    fn direct_tool_definition() -> crate::ToolDefinition {
        crate::ToolDefinition::raw(
            "direct_tool",
            "Issue a direct completion from inside a tool",
            serde_json::json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
            serde_json::json!({ "type": "object", "additionalProperties": true }),
        )
    }

    #[async_trait::async_trait]
    impl crate::ToolProvider for DirectTool {
        fn tool_manifests(&self) -> Vec<crate::ToolManifest> {
            vec![direct_tool_definition().manifest()]
        }

        fn resolve_contract(&self, name: &str) -> Option<Arc<crate::ToolContract>> {
            (name == "direct_tool").then(|| Arc::new(direct_tool_definition().contract()))
        }

        async fn execute(&self, call: crate::ToolCall<'_>) -> crate::ToolResult {
            let completion = call
                .context
                .direct_completion(
                    crate::DirectRequest::text("mock-model", "nested"),
                    "tool-direct",
                )
                .await
                .expect("tool direct completion");
            crate::ToolResult::ok(serde_json::json!({ "text": completion.text }))
        }
    }

    let default_recorder = RecordingEffectHost::default();
    let scoped_recorder = RecordingEffectHost::default();
    let transport = mock_provider(vec![
        MockCall {
            stream_events: Vec::new(),
            response: Ok(LlmResponse {
                full_text: String::new(),
                parts: vec![LlmOutputPart::ToolCall {
                    call_id: "direct-call-1".to_string(),
                    tool_name: "direct_tool".to_string(),
                    input_json: serde_json::json!({}).to_string(),
                    replay: None,
                }],
                ..LlmResponse::default()
            }),
        },
        MockCall {
            stream_events: Vec::new(),
            response: Ok(LlmResponse {
                full_text: "nested answer".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "nested answer".to_string(),
                    response_meta: None,
                }],
                ..LlmResponse::default()
            }),
        },
        MockCall {
            stream_events: Vec::new(),
            response: Ok(LlmResponse {
                full_text: "finished".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "finished".to_string(),
                    response_meta: None,
                }],
                ..LlmResponse::default()
            }),
        },
    ]);
    let mut runtime = runtime_with_plugins_and_tools_and_host(
        Vec::new(),
        Arc::new(DirectTool),
        transport,
        host_with_effect_recorder(default_recorder.clone()),
    )
    .await;

    let effect_scope =
        RuntimeEffectScope::new(&scoped_recorder, "scoped-tool-direct").expect("effect scope");
    let turn = runtime
        .stream_turn_with_effect_scope(
            TurnInput::text("use direct tool"),
            &NoopEventSink,
            effect_scope,
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    assert!(matches!(turn.outcome, TurnOutcome::Finished(_)));
    assert_eq!(
        scoped_recorder.count_kind(RuntimeEffectKind::DirectCompletion),
        1
    );
    assert_eq!(
        default_recorder.count_kind(RuntimeEffectKind::DirectCompletion),
        0
    );
    assert!(scoped_recorder.records().iter().any(|record| {
        record.kind == RuntimeEffectKind::DirectCompletion
            && record.idempotency_key.contains("tool:direct-call-1")
    }));
}

#[tokio::test]
async fn scoped_retry_sleep_records_turn_and_parent_tool_identity() {
    struct RetryOnceTool {
        attempts: Arc<std::sync::atomic::AtomicUsize>,
    }

    fn retry_once_tool_definition() -> crate::ToolDefinition {
        crate::ToolDefinition::raw(
            "retry_once",
            "Fails once with a safe retry.",
            serde_json::json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
            serde_json::json!({ "type": "object", "additionalProperties": true }),
        )
        .with_retry_policy(crate::ToolRetryPolicy::safe(2, 1, 1))
    }

    #[async_trait::async_trait]
    impl crate::ToolProvider for RetryOnceTool {
        fn tool_manifests(&self) -> Vec<crate::ToolManifest> {
            vec![retry_once_tool_definition().manifest()]
        }

        fn resolve_contract(&self, name: &str) -> Option<Arc<crate::ToolContract>> {
            (name == "retry_once").then(|| Arc::new(retry_once_tool_definition().contract()))
        }

        async fn execute(&self, _call: crate::ToolCall<'_>) -> crate::ToolResult {
            let attempt = self
                .attempts
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if attempt == 0 {
                return crate::ToolResult::retryable_failure(
                    crate::ToolFailureClass::External,
                    "transient",
                    "transient failure",
                    Some(1),
                );
            }
            crate::ToolResult::ok(serde_json::json!({ "ok": true }))
        }
    }

    let recorder = RecordingEffectHost::default();
    let transport = mock_provider(vec![
        MockCall {
            stream_events: Vec::new(),
            response: Ok(LlmResponse {
                full_text: String::new(),
                parts: vec![LlmOutputPart::ToolCall {
                    call_id: "retry-call-1".to_string(),
                    tool_name: "retry_once".to_string(),
                    input_json: serde_json::json!({}).to_string(),
                    replay: None,
                }],
                ..LlmResponse::default()
            }),
        },
        MockCall {
            stream_events: Vec::new(),
            response: Ok(LlmResponse {
                full_text: "finished".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "finished".to_string(),
                    response_meta: None,
                }],
                ..LlmResponse::default()
            }),
        },
    ]);
    let mut runtime = runtime_with_plugins_and_tools_and_host(
        Vec::new(),
        Arc::new(RetryOnceTool {
            attempts: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }),
        transport,
        EmbeddedRuntimeHost::new(RuntimeCoreConfig::default()),
    )
    .await;

    let effect_scope =
        RuntimeEffectScope::new(&recorder, "scoped-retry-sleep").expect("effect scope");
    let turn = runtime
        .stream_turn_with_effect_scope(
            TurnInput::text("use retry tool"),
            &NoopEventSink,
            effect_scope,
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    assert!(matches!(turn.outcome, TurnOutcome::Finished(_)));
    let sleep_records = recorder
        .records()
        .into_iter()
        .filter(|record| record.kind == RuntimeEffectKind::Sleep)
        .collect::<Vec<_>>();
    assert_eq!(sleep_records.len(), 1);
    let sleep = &sleep_records[0];
    assert_eq!(sleep.turn_id.as_deref(), Some("scoped-retry-sleep"));
    assert!(sleep.effect_id.contains("retry-call-1"));
    assert!(sleep.idempotency_key.contains("scoped-retry-sleep"));
    assert!(sleep.idempotency_key.contains("retry-call-1"));
}

#[tokio::test]
async fn tool_call_effect_crosses_host_per_logical_call_and_runs_local_tools() {
    let recorder = RecordingEffectHost::default();
    let transport = mock_provider(vec![
        MockCall {
            stream_events: Vec::new(),
            response: Ok(LlmResponse {
                full_text: String::new(),
                parts: vec![
                    LlmOutputPart::ToolCall {
                        call_id: "call-1".to_string(),
                        tool_name: "echo_tool".to_string(),
                        input_json: serde_json::json!({"value": "hi"}).to_string(),
                        replay: None,
                    },
                    LlmOutputPart::ToolCall {
                        call_id: "call-2".to_string(),
                        tool_name: "echo_tool".to_string(),
                        input_json: serde_json::json!({"value": "there"}).to_string(),
                        replay: None,
                    },
                ],
                ..LlmResponse::default()
            }),
        },
        MockCall {
            stream_events: Vec::new(),
            response: Ok(LlmResponse {
                full_text: "finished".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "finished".to_string(),
                    response_meta: None,
                }],
                ..LlmResponse::default()
            }),
        },
    ]);
    let mut runtime = runtime_with_plugins_and_tools_and_host(
        Vec::new(),
        Arc::new(EchoTool),
        transport,
        host_with_effect_recorder(recorder.clone()),
    )
    .await;

    let turn = runtime
        .run_turn_assembled(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "use the tool".to_string(),
                }],
                image_blobs: HashMap::new(),
                mode_turn_options: None,
                trace_turn_id: None,
                mode_extension: None,
                turn_context: crate::TurnContext::default(),
            },
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    assert!(matches!(turn.outcome, TurnOutcome::Finished(_)));
    assert_eq!(recorder.count_kind(RuntimeEffectKind::ToolCall), 2);
    let tool_keys = recorder
        .records()
        .into_iter()
        .filter(|record| record.kind == RuntimeEffectKind::ToolCall)
        .map(|record| record.idempotency_key)
        .collect::<Vec<_>>();
    assert!(tool_keys.iter().any(|key| key.ends_with(":call-1")));
    assert!(tool_keys.iter().any(|key| key.ends_with(":call-2")));
    assert!(
        active_tool_calls(&turn.state)
            .iter()
            .any(|record| record.tool == "echo_tool" && record.output.is_success())
    );
}

#[tokio::test]
async fn exec_and_execution_surface_effects_cross_host_once() {
    let recorder = RecordingEffectHost::default();
    let mode = effect_test_mode();
    let policy = SessionPolicy {
        execution_mode: mode.clone(),
        provider: mock_provider(Vec::new()).into_handle(),
        model: "mock-model".to_string(),
        max_context_tokens: Some(200_000),
        ..SessionPolicy::default()
    };
    let plugin_session = crate::PluginHost::new(vec![Arc::new(EffectHostTestModeFactory)])
        .build_session("root", mode, None, None)
        .expect("plugins");
    let mut runtime = LashRuntime::from_embedded_state(
        policy,
        host_with_effect_recorder(recorder.clone()),
        RuntimeServices::new(plugin_session),
        PersistedSessionState::default(),
    )
    .await
    .expect("runtime");

    let turn = runtime
        .run_turn_assembled(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "run code".to_string(),
                }],
                image_blobs: HashMap::new(),
                mode_turn_options: None,
                trace_turn_id: None,
                mode_extension: None,
                turn_context: crate::TurnContext::default(),
            },
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    assert!(matches!(turn.outcome, TurnOutcome::Finished(_)));
    assert_eq!(
        recorder.count_kind(RuntimeEffectKind::SyncExecutionSurface),
        1
    );
    assert_eq!(recorder.count_kind(RuntimeEffectKind::ExecCode), 1);
}

#[tokio::test]
async fn direct_completion_crosses_host_and_records_usage_and_trace() {
    let recorder = RecordingEffectHost::default();
    let trace_path = unique_trace_path("direct-completion");
    let transport = mock_provider(vec![MockCall {
        stream_events: Vec::new(),
        response: Ok(LlmResponse {
            full_text: "direct answer".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "direct answer".to_string(),
                response_meta: None,
            }],
            usage: LlmUsage {
                input_tokens: 7,
                output_tokens: 5,
                cached_input_tokens: 1,
                reasoning_tokens: 2,
            },
            ..LlmResponse::default()
        }),
    }]);
    let host = EmbeddedRuntimeHost::new(
        RuntimeCoreConfig::default()
            .with_effect_host(Arc::new(recorder.clone()))
            .with_trace_jsonl_path(Some(trace_path.clone())),
    );
    let runtime =
        runtime_with_plugins_and_tools_and_host(Vec::new(), Arc::new(EmptyTools), transport, host)
            .await;

    let manager = runtime.session_manager().expect("session manager");
    let mut request = crate::DirectRequest::text("mock-model", "summarize");
    request.originating_tool_call_id = Some("originating-tool-call".to_string());
    let completion = manager
        .direct_completion(request, "direct-test")
        .await
        .expect("direct completion");

    assert_eq!(completion.text, "direct answer");
    assert_eq!(completion.usage.input_tokens, 7);
    assert_eq!(recorder.count_kind(RuntimeEffectKind::DirectCompletion), 1);
    assert!(recorder.records().iter().any(|record| {
        record.kind == RuntimeEffectKind::DirectCompletion
            && record
                .idempotency_key
                .contains("tool:originating-tool-call")
    }));
    assert_token_ledger_entry(&runtime, "direct-test", "mock-model", &completion.usage);
    assert_trace_contains_completed_llm_call(&trace_path, Some("originating-tool-call"));
}

#[tokio::test]
async fn direct_llm_completion_crosses_host_and_records_usage_and_trace() {
    let recorder = RecordingEffectHost::default();
    let trace_path = unique_trace_path("direct-llm-completion");
    let transport = mock_provider(vec![MockCall {
        stream_events: Vec::new(),
        response: Ok(LlmResponse {
            full_text: "raw direct answer".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "raw direct answer".to_string(),
                response_meta: None,
            }],
            usage: LlmUsage {
                input_tokens: 4,
                output_tokens: 6,
                cached_input_tokens: 0,
                reasoning_tokens: 1,
            },
            ..LlmResponse::default()
        }),
    }]);
    let host = EmbeddedRuntimeHost::new(
        RuntimeCoreConfig::default()
            .with_effect_host(Arc::new(recorder.clone()))
            .with_trace_jsonl_path(Some(trace_path.clone())),
    );
    let runtime =
        runtime_with_plugins_and_tools_and_host(Vec::new(), Arc::new(EmptyTools), transport, host)
            .await;

    let manager = runtime.session_manager().expect("session manager");
    let request = LlmRequest {
        model: "mock-model".to_string(),
        messages: vec![LlmMessage::new(
            LlmRole::User,
            vec![LlmContentBlock::Text {
                text: Arc::from("raw prompt"),
                response_meta: None,
                cache_breakpoint: false,
            }],
        )],
        attachments: Vec::new(),
        tools: Arc::new(Vec::new()),
        tool_choice: LlmToolChoice::None,
        model_variant: None,
        session_id: None,
        output_spec: None,
        stream_events: None,
        provider_trace: None,
    };
    let completion = manager
        .direct_llm_completion(request, "direct-llm-test")
        .await
        .expect("direct llm completion");

    assert_eq!(completion.response.full_text, "raw direct answer");
    assert_eq!(completion.usage.output_tokens, 6);
    assert_eq!(
        recorder.count_kind(RuntimeEffectKind::DirectLlmCompletion),
        1
    );
    assert_token_ledger_entry(&runtime, "direct-llm-test", "mock-model", &completion.usage);
    assert_trace_contains_completed_llm_call(&trace_path, None);
}

#[test]
fn runtime_effect_executor_has_no_legacy_future_api() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source_files = [
        manifest_dir.join("src/runtime/effect_host.rs"),
        manifest_dir.join("src/runtime/turn_driver.rs"),
        manifest_dir.join("src/runtime/session_manager/direct.rs"),
    ];
    let legacy_future_type = ["Effect", "Future"].concat();
    let legacy_constructor = ["Runtime", "Effect", "Executor", "::new"].concat();
    for path in source_files {
        let source = std::fs::read_to_string(&path).expect("read runtime effect source");
        assert!(
            !source.contains(&legacy_future_type),
            "{} still mentions {legacy_future_type}",
            path.display()
        );
        assert!(
            !source.contains(&legacy_constructor),
            "{} still mentions {legacy_constructor}",
            path.display()
        );
    }
}

fn unique_trace_path(prefix: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "lash-{prefix}-{}-{}.jsonl",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ))
}

fn assert_trace_contains_completed_llm_call(
    trace_path: &PathBuf,
    originating_tool_call_id: Option<&str>,
) {
    let logged = std::fs::read_to_string(trace_path).expect("read trace");
    let entries = logged
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("json trace entry"))
        .collect::<Vec<_>>();
    assert!(
        entries.iter().any(|entry| {
            let origin_matches = match originating_tool_call_id {
                Some(expected) => {
                    entry
                        .pointer("/context/originating_tool_call_id")
                        .and_then(|value| value.as_str())
                        == Some(expected)
                }
                None => entry.pointer("/context/originating_tool_call_id").is_none(),
            };
            entry.get("type").and_then(|value| value.as_str()) == Some("llm_call_completed")
                && entry.get("usage").is_some()
                && origin_matches
        }),
        "expected direct completion trace with usage: {entries:?}"
    );
}

fn assert_token_ledger_entry(runtime: &LashRuntime, source: &str, model: &str, usage: &TokenUsage) {
    let ledger = runtime.shared_token_ledger.lock().expect("token ledger");
    assert_eq!(ledger.len(), 1, "unexpected ledger: {ledger:?}");
    let entry = &ledger[0];
    assert_eq!(entry.source, source);
    assert_eq!(entry.model, model);
    assert_eq!(entry.usage.input_tokens, usage.input_tokens);
    assert_eq!(entry.usage.output_tokens, usage.output_tokens);
    assert_eq!(entry.usage.cached_input_tokens, usage.cached_input_tokens);
    assert_eq!(entry.usage.reasoning_tokens, usage.reasoning_tokens);
}

struct EffectHostTestModeFactory;

impl crate::PluginFactory for EffectHostTestModeFactory {
    fn id(&self) -> &'static str {
        "effect_host_test_mode"
    }

    fn build(
        &self,
        ctx: &crate::PluginSessionContext,
    ) -> Result<Arc<dyn crate::SessionPlugin>, crate::PluginError> {
        Ok(Arc::new(EffectHostTestModePlugin {
            active: ctx.execution_mode == effect_test_mode(),
        }))
    }
}

struct EffectHostTestModePlugin {
    active: bool,
}

impl crate::SessionPlugin for EffectHostTestModePlugin {
    fn id(&self) -> &'static str {
        "effect_host_test_mode"
    }

    fn register(&self, registrar: &mut crate::PluginRegistrar) -> Result<(), crate::PluginError> {
        if self.active {
            registrar
                .mode()
                .session(Arc::new(EffectHostTestModeSession))?;
            registrar
                .mode()
                .protocol_driver(Arc::new(EffectHostTestProtocolDriver))?;
        }
        Ok(())
    }
}

struct EffectHostTestModeSession;

#[async_trait::async_trait]
impl ModeSessionPlugin for EffectHostTestModeSession {
    async fn execute_code(
        &self,
        _ctx: crate::ModeExecutionContext<'_>,
        _request: crate::ExecRequest,
    ) -> Result<crate::ExecResponse, crate::SessionError> {
        Ok(crate::ExecResponse {
            output: "exec output".to_string(),
            observations: Vec::new(),
            observation_truncation: Vec::new(),
            tool_calls: Vec::new(),
            images: Vec::new(),
            printed_images: Vec::new(),
            error: None,
            duration_ms: 1,
            terminal_finish: None,
        })
    }
}

struct EffectHostTestProtocolDriver;

impl ModeProtocolDriverPlugin for EffectHostTestProtocolDriver {
    fn mode_id(&self) -> &str {
        "effect_host_test"
    }

    fn build_preamble(&self, input: crate::ModeBuildInput) -> crate::ModePreamble {
        crate::ModePreamble {
            config: crate::ModeConfig::chat(
                Arc::new(EffectHostTestDriver),
                true,
                Arc::new(effect_host_turn_limit_final_message),
            ),
            tool_specs: input.tool_surface.model_tool_specs(),
            tool_names: input.tool_surface.tool_names(),
            tool_names_fingerprint: input.tool_surface.tool_names_fingerprint(),
            omitted_tool_count: 0,
            execution_prompt: Arc::from(""),
            prompt_contributions: input.extra_prompt_contributions,
        }
    }
}

fn effect_host_turn_limit_final_message(message_id: String, max_turns: usize) -> crate::Message {
    crate::Message {
        id: message_id.clone(),
        role: crate::MessageRole::System,
        parts: crate::shared_parts(vec![crate::Part {
            id: format!("{message_id}.p0"),
            kind: crate::PartKind::Error,
            content: format!("Turn limit reached ({max_turns}) before a final test response."),
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            tool_replay: None,
            prune_state: crate::PruneState::Intact,
            reasoning_meta: None,
            response_meta: None,
        }]),
        origin: None,
    }
}

struct EffectHostTestDriver;

impl lash_sansio::ProtocolDriverHandle<crate::HostModeProtocol> for EffectHostTestDriver {
    fn prepare_mode_iteration(
        &self,
        _ctx: crate::DriverContextView<'_>,
    ) -> Vec<crate::DriverAction> {
        vec![crate::DriverAction::StartExec {
            code: "print('effect host')".to_string(),
            driver_state: serde_json::Value::Null,
        }]
    }

    fn handle_llm_success(
        &self,
        _ctx: crate::DriverContextView<'_>,
        _waiting: lash_sansio::WaitingLlmState<crate::HostModeProtocol>,
        _llm_response: LlmResponse,
        _text_streamed: bool,
    ) -> Vec<crate::DriverAction> {
        Vec::new()
    }

    fn handle_tool_results(
        &self,
        _ctx: crate::DriverContextView<'_>,
        _completed: Vec<crate::sansio::CompletedToolCall>,
    ) -> Vec<crate::DriverAction> {
        Vec::new()
    }

    fn handle_exec_result(
        &self,
        _ctx: crate::DriverContextView<'_>,
        _waiting: lash_sansio::WaitingExecState<crate::HostModeProtocol>,
        result: Result<crate::ExecResponse, String>,
    ) -> Vec<crate::DriverAction> {
        match result {
            Ok(response) => vec![crate::DriverAction::Finish(TurnOutcome::Finished(
                TurnFinish::SubmittedValue {
                    value: serde_json::json!(response.output),
                },
            ))],
            Err(_error) => vec![crate::DriverAction::Finish(TurnOutcome::Stopped(
                TurnStop::RuntimeError,
            ))],
        }
    }
}
