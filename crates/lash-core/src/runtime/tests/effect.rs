use super::*;
use crate::llm::types::{LlmAttachment, LlmContentBlock, LlmMessage, LlmRole, LlmToolChoice};
use crate::plugin::{ProtocolDriverPlugin, ProtocolSessionPlugin};
use crate::runtime::EmbeddedDurableTurnStore;
use std::time::Duration;

#[derive(Clone, Debug)]
struct EffectControllerRecord {
    kind: RuntimeEffectKind,
    turn_id: Option<String>,
    effect_id: String,
    replay_key: String,
    checkpoint_hash: Option<String>,
}

#[derive(Clone, Default)]
struct RecordingEffectController {
    records: Arc<Mutex<Vec<EffectControllerRecord>>>,
    envelopes: Arc<Mutex<Vec<String>>>,
    llm_calls: Arc<Mutex<usize>>,
}

impl RecordingEffectController {
    fn records(&self) -> Vec<EffectControllerRecord> {
        self.records.lock().expect("effect records").clone()
    }

    fn envelopes(&self) -> Vec<String> {
        self.envelopes.lock().expect("effect envelopes").clone()
    }

    fn count_kind(&self, kind: RuntimeEffectKind) -> usize {
        self.records()
            .iter()
            .filter(|record| record.kind == kind)
            .count()
    }

    fn record(&self, invocation: &RuntimeInvocation) {
        self.records
            .lock()
            .expect("effect records")
            .push(EffectControllerRecord {
                kind: invocation.effect_kind().expect("effect kind"),
                turn_id: invocation.scope.turn_id.clone(),
                effect_id: invocation.effect_id().expect("effect id").to_string(),
                replay_key: invocation.replay_key().expect("replay key").to_string(),
                checkpoint_hash: invocation.checkpoint_hash.clone(),
            });
    }
}

fn test_effect_invocation(
    session_id: &str,
    turn_id: &str,
    turn_index: usize,
    protocol_iteration: usize,
    effect_id: &str,
    kind: RuntimeEffectKind,
    replay_key: &str,
) -> RuntimeInvocation {
    RuntimeInvocation::effect(
        RuntimeScope::for_turn(session_id, turn_id, turn_index, protocol_iteration),
        effect_id,
        kind,
        replay_key,
        Some("0".repeat(64)),
    )
}

fn runtime_host_config_with_effect_controller(
    effect_controller: Arc<dyn RuntimeEffectController>,
) -> RuntimeHostConfig {
    let mut config = RuntimeHostConfig::in_memory();
    config.control.effect_controller = effect_controller;
    config
}

fn runtime_host_config_with_provider(provider: crate::ProviderHandle) -> RuntimeHostConfig {
    let mut config = RuntimeHostConfig::in_memory();
    config.providers.provider_resolver = Arc::new(crate::SingleProviderResolver::new(provider));
    config
}

fn runtime_host_config_with_durability(
    attachment_store: Arc<dyn crate::AttachmentStore>,
    artifact_store: Arc<dyn lashlang::LashlangArtifactStore>,
) -> RuntimeHostConfig {
    let mut config = RuntimeHostConfig::in_memory();
    config.durability.attachment_store = attachment_store;
    config.durability.lashlang_artifact_store = artifact_store;
    config
}

#[async_trait::async_trait]
impl RuntimeEffectController for RecordingEffectController {
    async fn execute_effect(
        &self,
        envelope: RuntimeEffectEnvelope,
        _local_executor: crate::RuntimeEffectLocalExecutor<'_>,
    ) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError> {
        self.envelopes
            .lock()
            .expect("effect envelopes")
            .push(serde_json::to_string(&envelope).expect("serialize effect envelope"));
        self.record(&envelope.invocation);
        match envelope.command {
            RuntimeEffectCommand::LlmCall { request } => {
                let mut llm_calls = self.llm_calls.lock().expect("llm calls");
                *llm_calls += 1;
                let first_call = *llm_calls == 1;
                let prompt = format!("{:?}", request.messages);
                let parts = if first_call && prompt.contains("use the tool") {
                    vec![
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
                    ]
                } else if first_call && prompt.contains("use direct tool") {
                    vec![LlmOutputPart::ToolCall {
                        call_id: "direct-call-1".to_string(),
                        tool_name: "direct_tool".to_string(),
                        input_json: serde_json::json!({}).to_string(),
                        replay: None,
                    }]
                } else if first_call && prompt.contains("use retry tool") {
                    vec![LlmOutputPart::ToolCall {
                        call_id: "retry-call-1".to_string(),
                        tool_name: "retry_once".to_string(),
                        input_json: serde_json::json!({}).to_string(),
                        replay: None,
                    }]
                } else {
                    vec![LlmOutputPart::Text {
                        text: "finished".to_string(),
                        response_meta: None,
                    }]
                };
                Ok(RuntimeEffectOutcome::LlmCall {
                    result: Ok(LlmResponse {
                        full_text: if parts
                            .iter()
                            .any(|part| matches!(part, LlmOutputPart::Text { .. }))
                        {
                            "finished".to_string()
                        } else {
                            String::new()
                        },
                        parts,
                        usage: LlmUsage {
                            input_tokens: 1,
                            output_tokens: 1,
                            cached_input_tokens: 0,
                            reasoning_tokens: 0,
                        },
                        ..LlmResponse::default()
                    }),
                    text_streamed: false,
                })
            }
            RuntimeEffectCommand::ToolCall { call } => {
                let output = crate::ToolCallOutput::success(serde_json::json!({"ok": true}));
                Ok(RuntimeEffectOutcome::ToolCall {
                    result: crate::sansio::CompletedToolCall {
                        call_id: call.call_id.clone(),
                        tool_name: call.tool_name.clone(),
                        args: call.args,
                        model_return: crate::ModelToolReturn {
                            call_id: call.call_id,
                            tool_name: call.tool_name,
                            parts: vec![crate::ModelToolReturnPart::text("ok")],
                        },
                        output,
                        duration_ms: 0,
                        replay: call.replay,
                    },
                })
            }
            RuntimeEffectCommand::Process { .. } => Err(RuntimeEffectControllerError::new(
                "process_unexpected",
                "recording effect controller does not execute processes",
            )),
            RuntimeEffectCommand::Checkpoint { .. } => Ok(RuntimeEffectOutcome::Checkpoint {
                result: Ok(crate::CheckpointDelivery::default()),
            }),
            RuntimeEffectCommand::SyncExecutionSurface { .. } => {
                Ok(RuntimeEffectOutcome::SyncExecutionSurface { result: Ok(None) })
            }
            RuntimeEffectCommand::ExecCode { .. } => Ok(RuntimeEffectOutcome::ExecCode {
                result: Ok(crate::ExecResponse {
                    observations: Vec::new(),
                    observation_truncation: Vec::new(),
                    tool_calls: Vec::new(),
                    images: Vec::new(),
                    printed_images: Vec::new(),
                    error: None,
                    duration_ms: 0,
                    terminal_finish: Some(serde_json::json!("ok")),
                }),
            }),
            RuntimeEffectCommand::Sleep { .. } => Ok(RuntimeEffectOutcome::Sleep),
            RuntimeEffectCommand::Direct { request, .. } => {
                // Both the text-only (`direct_completion`) and full-response
                // (`direct_llm_completion`) client methods now flow through the
                // single `Direct` effect; they differ only in how the caller
                // projects the resulting `LlmResponse`. The full-response tests
                // submit a "raw prompt" message or an image attachment, so use
                // those to pick the response text/usage the assertions expect.
                let prompt = format!("{:?}", request.messages);
                let is_full = prompt.contains("raw prompt") || !request.attachments.is_empty();
                let (text, usage) = if is_full {
                    (
                        "raw direct answer",
                        LlmUsage {
                            input_tokens: 4,
                            output_tokens: 6,
                            cached_input_tokens: 0,
                            reasoning_tokens: 1,
                        },
                    )
                } else {
                    (
                        "direct answer",
                        LlmUsage {
                            input_tokens: 7,
                            output_tokens: 5,
                            cached_input_tokens: 1,
                            reasoning_tokens: 2,
                        },
                    )
                };
                Ok(RuntimeEffectOutcome::Direct {
                    result: Ok(LlmResponse {
                        full_text: text.to_string(),
                        parts: vec![LlmOutputPart::Text {
                            text: text.to_string(),
                            response_meta: None,
                        }],
                        usage,
                        ..LlmResponse::default()
                    }),
                })
            }
        }
    }
}

#[derive(Clone, Default)]
struct ProcessJournalController {
    calls: Arc<Mutex<usize>>,
}

impl ProcessJournalController {
    fn call_count(&self) -> usize {
        *self.calls.lock().expect("process journal calls")
    }
}

#[async_trait::async_trait]
impl RuntimeEffectController for ProcessJournalController {
    async fn execute_effect(
        &self,
        envelope: RuntimeEffectEnvelope,
        _local_executor: crate::RuntimeEffectLocalExecutor<'_>,
    ) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError> {
        *self.calls.lock().expect("process journal calls") += 1;
        match envelope.command {
            RuntimeEffectCommand::Process { command } => match command {
                ProcessCommand::List { .. } => Ok(RuntimeEffectOutcome::Process {
                    result: ProcessEffectOutcome::List {
                        entries: Vec::new(),
                    },
                }),
                ProcessCommand::Transfer { .. } => Ok(RuntimeEffectOutcome::Process {
                    result: ProcessEffectOutcome::Transfer,
                }),
                ProcessCommand::DeleteSession { session_id } => Ok(RuntimeEffectOutcome::Process {
                    result: ProcessEffectOutcome::DeleteSession {
                        report: crate::ProcessSessionDeleteReport {
                            session_id,
                            ..Default::default()
                        },
                    },
                }),
                ProcessCommand::Start { registration, .. } => Ok(RuntimeEffectOutcome::Process {
                    result: ProcessEffectOutcome::Start {
                        record: ProcessRecord::from_registration(registration),
                    },
                }),
                ProcessCommand::Await { .. } => Ok(RuntimeEffectOutcome::Process {
                    result: ProcessEffectOutcome::Await {
                        output: ProcessAwaitOutput::from_tool_output(
                            crate::ToolCallOutput::success(serde_json::json!({"ok": true})),
                        ),
                    },
                }),
                ProcessCommand::Cancel { process_id, .. } => Ok(RuntimeEffectOutcome::Process {
                    result: ProcessEffectOutcome::Cancel {
                        record: ProcessRecord::from_registration(ProcessRegistration::new(
                            process_id,
                            ProcessInput::External {
                                metadata: serde_json::json!({}),
                            },
                        )),
                    },
                }),
                ProcessCommand::Signal { process_id, .. } => Ok(RuntimeEffectOutcome::Process {
                    result: ProcessEffectOutcome::Signal {
                        event: crate::ProcessEvent {
                            process_id,
                            sequence: 1,
                            event_type: "process.signal".to_string(),
                            payload: serde_json::json!({}),
                            invocation: crate::RuntimeInvocation {
                                scope: crate::RuntimeScope::new("test-session"),
                                subject: crate::RuntimeSubject::Process {
                                    process_id: "test-process".to_string(),
                                },
                                caused_by: None,
                                replay: None,
                                checkpoint_hash: None,
                            },
                            semantics: crate::ProcessEventSemantics::default(),
                            occurred_at: std::time::SystemTime::UNIX_EPOCH,
                        },
                    },
                }),
            },
            other => Err(RuntimeEffectControllerError::new(
                "unexpected_effect",
                format!("expected process effect, got {}", other.kind().as_str()),
            )),
        }
    }
}

#[derive(Clone)]
struct DelayedSleepController {
    delay: Duration,
}

#[async_trait::async_trait]
impl RuntimeEffectController for DelayedSleepController {
    async fn execute_effect(
        &self,
        envelope: RuntimeEffectEnvelope,
        _local_executor: crate::RuntimeEffectLocalExecutor<'_>,
    ) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError> {
        tokio::time::sleep(self.delay).await;
        match envelope.command {
            RuntimeEffectCommand::Sleep { .. } => Ok(RuntimeEffectOutcome::Sleep),
            other => Err(RuntimeEffectControllerError::new(
                "unexpected_effect",
                format!("expected sleep effect, got {}", other.kind().as_str()),
            )),
        }
    }
}

#[derive(Default)]
struct DurableAttachmentRequiredController;

#[async_trait::async_trait]
impl RuntimeEffectController for DurableAttachmentRequiredController {
    fn durability_tier(&self) -> crate::DurabilityTier {
        crate::DurabilityTier::Durable
    }

    fn requires_durable_attachment_store(&self) -> bool {
        true
    }

    async fn execute_effect(
        &self,
        envelope: RuntimeEffectEnvelope,
        local_executor: crate::RuntimeEffectLocalExecutor<'_>,
    ) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError> {
        local_executor.execute(envelope).await
    }
}

/// An attachment store that reports a durable persistence tier while keeping the
/// in-memory backing. Lets the durable-substrate scope checks reach the
/// artifact/session facets without standing up a real durable backend.
#[derive(Default)]
struct DurableInMemoryAttachmentStore {
    inner: crate::InMemoryAttachmentStore,
}

impl crate::AttachmentStore for DurableInMemoryAttachmentStore {
    fn persistence(&self) -> crate::AttachmentStorePersistence {
        crate::AttachmentStorePersistence::Durable
    }

    fn put(
        &self,
        bytes: Vec<u8>,
        meta: lash_sansio::AttachmentCreateMeta,
    ) -> Result<lash_sansio::AttachmentRef, crate::AttachmentStoreError> {
        self.inner.put(bytes, meta)
    }

    fn get(
        &self,
        id: &lash_sansio::AttachmentId,
    ) -> Result<crate::StoredAttachment, crate::AttachmentStoreError> {
        self.inner.get(id)
    }
}

/// A Lashlang artifact store that reports a durable tier over in-memory storage.
#[derive(Default)]
struct DurableInMemoryArtifactStore {
    inner: lashlang::InMemoryLashlangArtifactStore,
}

impl lashlang::LashlangArtifactStore for DurableInMemoryArtifactStore {
    fn durability_tier(&self) -> crate::DurabilityTier {
        crate::DurabilityTier::Durable
    }

    fn put_module_artifact(
        &self,
        artifact: &lashlang::ModuleArtifact,
    ) -> Result<(), lashlang::ArtifactStoreError> {
        self.inner.put_module_artifact(artifact)
    }

    fn get_module_artifact(
        &self,
        module_ref: &lashlang::ModuleRef,
    ) -> Result<Option<lashlang::ModuleArtifact>, lashlang::ArtifactStoreError> {
        self.inner.get_module_artifact(module_ref)
    }
}

/// Build the single-turn mock transport shared by the durable-substrate scope
/// tests.
fn done_once_provider() -> TestProvider {
    mock_provider(vec![MockCall {
        stream_events: Vec::new(),
        response: Ok(LlmResponse {
            full_text: "Done".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "Done".to_string(),
                response_meta: None,
            }],
            ..LlmResponse::default()
        }),
    }])
}

struct RejectingEffectController;

#[async_trait::async_trait]
impl RuntimeEffectController for RejectingEffectController {
    async fn execute_effect(
        &self,
        envelope: RuntimeEffectEnvelope,
        _local_executor: crate::RuntimeEffectLocalExecutor<'_>,
    ) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError> {
        Err(RuntimeEffectControllerError::new(
            "test_controller_rejected",
            format!("rejected {}", envelope.command.kind().as_str()),
        ))
    }
}

struct WrongOutcomeEffectController;

#[async_trait::async_trait]
impl RuntimeEffectController for WrongOutcomeEffectController {
    async fn execute_effect(
        &self,
        _envelope: RuntimeEffectEnvelope,
        _local_executor: crate::RuntimeEffectLocalExecutor<'_>,
    ) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError> {
        Ok(RuntimeEffectOutcome::Sleep)
    }
}

fn host_with_effect_recorder(recorder: RecordingEffectController) -> EmbeddedRuntimeHost {
    let mut config = runtime_host_config_with_effect_controller(Arc::new(recorder));
    config.providers.provider_resolver = Arc::new(crate::SingleProviderResolver::new(
        mock_provider(Vec::new()).into_handle(),
    ));
    EmbeddedRuntimeHost::new(config)
}

#[tokio::test]
async fn standard_turn_llm_and_checkpoint_effects_cross_controller_once() {
    let recorder = RecordingEffectController::default();
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
                protocol_turn_options: None,
                trace_turn_id: None,
                protocol_extension: None,
                turn_context: crate::TurnContext::default(),
            },
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    assert!(matches!(turn.outcome, TurnOutcome::Finished(_)));
    assert_eq!(recorder.count_kind(RuntimeEffectKind::LlmCall), 1);
    assert_eq!(recorder.count_kind(RuntimeEffectKind::Checkpoint), 1);
    assert!(
        recorder
            .records()
            .iter()
            .all(|record| record.turn_id.is_some() && record.replay_key.starts_with("root:"))
    );
    assert!(recorder.records().iter().all(|record| {
        record
            .checkpoint_hash
            .as_deref()
            .is_some_and(|hash| hash.len() == 64 && hash.chars().all(|ch| ch.is_ascii_hexdigit()))
    }));
}

#[tokio::test]
async fn durable_turn_persists_checkpoints_and_journal_then_clears_on_commit() {
    let recorder = RecordingEffectController::default();
    let store = Arc::new(RecordingStore::default());
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
    let mut runtime = runtime_with_plugins_and_tools_and_host_and_store(
        Vec::new(),
        Arc::new(EmptyTools),
        transport,
        host_with_effect_recorder(recorder),
        store.clone(),
    )
    .await;

    let turn = runtime
        .run_turn_assembled(TurnInput::text("hello"), CancellationToken::new())
        .await
        .expect("turn");

    assert!(matches!(turn.outcome, TurnOutcome::Finished(_)));
    assert!(store.runtime_turn_checkpoint_save_count() >= 1);
    assert!(store.runtime_effect_journal_save_count() >= 1);
    assert!(
        store.runtime_turn_lease_renew_count()
            >= store.runtime_turn_checkpoint_save_count()
                + store.runtime_effect_journal_save_count()
    );
    assert_eq!(store.runtime_turn_checkpoint_count(), 0);
    assert_eq!(store.runtime_effect_journal_count(), 0);
}

#[tokio::test]
async fn durable_controller_error_abandons_lease_and_preserves_resume_state() {
    #[derive(Clone)]
    struct FailToolCallController {
        inner: RecordingEffectController,
    }

    #[async_trait::async_trait]
    impl RuntimeEffectController for FailToolCallController {
        async fn execute_effect(
            &self,
            envelope: RuntimeEffectEnvelope,
            local_executor: crate::RuntimeEffectLocalExecutor<'_>,
        ) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError> {
            if envelope.invocation.effect_kind() == Some(RuntimeEffectKind::ToolCall) {
                return Err(RuntimeEffectControllerError::new(
                    "controller_failed",
                    "tool call controller failed",
                ));
            }
            self.inner.execute_effect(envelope, local_executor).await
        }
    }

    let recorder = RecordingEffectController::default();
    let store = Arc::new(RecordingStore::default());
    let transport = mock_provider(Vec::new());
    let host = EmbeddedRuntimeHost::new(runtime_host_config_with_effect_controller(Arc::new(
        FailToolCallController {
            inner: recorder.clone(),
        },
    )));
    let mut runtime = runtime_with_plugins_and_tools_and_host_and_store(
        Vec::new(),
        Arc::new(EchoTool),
        transport,
        host,
        store.clone(),
    )
    .await;

    let err = runtime
        .run_turn_assembled(TurnInput::text("use the tool"), CancellationToken::new())
        .await
        .expect_err("controller failure should abort durable turn");

    assert_eq!(
        err.code,
        crate::RuntimeErrorCode::Other("controller_failed".to_string())
    );
    assert_eq!(store.runtime_turn_lease_abandon_count(), 1);
    assert!(store.runtime_turn_checkpoint_count() >= 1);
    assert!(store.runtime_effect_journal_count() >= 1);
}

#[tokio::test]
async fn durable_finalize_error_abandons_lease_and_preserves_resume_state() {
    let store = Arc::new(RecordingStore::default());
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
    let plugin = Arc::new(StaticPluginFactory::new(
        "bad_after_turn",
        crate::PluginSpec::new().with_after_turn(Arc::new(|_| {
            Box::pin(async {
                Ok(vec![crate::PluginDirective::AbortTurn {
                    code: "after_turn_abort".to_string(),
                    message: "after turn abort is invalid during finalization".to_string(),
                }])
            })
        })),
    ));
    let mut runtime = runtime_with_plugins_and_tools_and_host_and_store(
        vec![plugin],
        Arc::new(EmptyTools),
        transport,
        test_host_config(),
        store.clone(),
    )
    .await;

    let err = runtime
        .run_turn_assembled(
            TurnInput::text("hello").with_trace_turn_id("finalize-error-turn"),
            CancellationToken::new(),
        )
        .await
        .expect_err("after_turn abort should fail finalization");

    assert_eq!(err.code, crate::RuntimeErrorCode::PluginFinalizeTurn);
    assert_eq!(store.runtime_turn_lease_abandon_count(), 1);
    assert!(store.runtime_turn_checkpoint_count() >= 1);
    assert!(store.runtime_effect_journal_count() >= 1);
}

#[tokio::test]
async fn durable_turn_resume_uses_leased_finisher_and_clears_resume_state() {
    #[derive(Clone)]
    struct FailToolCallController {
        inner: RecordingEffectController,
    }

    #[async_trait::async_trait]
    impl RuntimeEffectController for FailToolCallController {
        async fn execute_effect(
            &self,
            envelope: RuntimeEffectEnvelope,
            local_executor: crate::RuntimeEffectLocalExecutor<'_>,
        ) -> Result<RuntimeEffectOutcome, RuntimeEffectControllerError> {
            if envelope.invocation.effect_kind() == Some(RuntimeEffectKind::ToolCall) {
                return Err(RuntimeEffectControllerError::new(
                    "controller_failed",
                    "tool call controller failed",
                ));
            }
            self.inner.execute_effect(envelope, local_executor).await
        }
    }

    let turn_id = "resume-shared-finisher-turn";
    let store = Arc::new(RecordingStore::default());
    let first_recorder = RecordingEffectController::default();
    let failing_host = EmbeddedRuntimeHost::new(runtime_host_config_with_effect_controller(
        Arc::new(FailToolCallController {
            inner: first_recorder,
        }),
    ));
    let mut failing_runtime = runtime_with_plugins_and_tools_and_host_and_store(
        Vec::new(),
        Arc::new(EchoTool),
        mock_provider(Vec::new()),
        failing_host,
        store.clone(),
    )
    .await;

    failing_runtime
        .run_turn_assembled(
            TurnInput::text("use the tool").with_trace_turn_id(turn_id),
            CancellationToken::new(),
        )
        .await
        .expect_err("first turn should stop before final commit");
    assert!(store.runtime_turn_checkpoint_count() >= 1);
    assert!(store.runtime_effect_journal_count() >= 1);

    let resume_recorder = RecordingEffectController::default();
    let resume_host = EmbeddedRuntimeHost::new(runtime_host_config_with_effect_controller(
        Arc::new(resume_recorder.clone()),
    ));
    let mut resume_runtime = runtime_with_plugins_and_tools_and_host_and_store(
        Vec::new(),
        Arc::new(EchoTool),
        mock_provider(Vec::new()),
        resume_host,
        store.clone(),
    )
    .await;

    let resumed = resume_runtime
        .resume_turn(
            turn_id,
            TurnOptions::new(CancellationToken::new()).with_events(&NoopEventSink),
        )
        .await
        .expect("resume turn");

    assert!(matches!(resumed.outcome, TurnOutcome::Finished(_)));
    assert_eq!(store.runtime_turn_checkpoint_count(), 0);
    assert_eq!(store.runtime_effect_journal_count(), 0);
    assert!(resume_recorder.count_kind(RuntimeEffectKind::ToolCall) >= 1);
}

#[tokio::test]
async fn effect_journal_replays_without_reinvoking_controller_and_rejects_hash_mismatch() {
    let recorder = RecordingEffectController::default();
    let store = RecordingStore::default();
    let mut lease = store
        .claim_runtime_turn_lease("root", "turn-1", "test", 60_000)
        .await
        .expect("lease");
    let invocation = test_effect_invocation(
        "root",
        "turn-1",
        1,
        0,
        "sleep",
        RuntimeEffectKind::Sleep,
        "root:turn-1:1:0:sleep:1",
    );
    let envelope = RuntimeEffectEnvelope::new(
        invocation.clone(),
        RuntimeEffectCommand::Sleep { duration_ms: 1 },
    );

    let first = crate::runtime::execute_embedded_journaled_effect(
        &store,
        &mut lease,
        &recorder,
        envelope.clone(),
        crate::RuntimeEffectLocalExecutor::unavailable(),
    )
    .await
    .expect("first effect");
    let second = crate::runtime::execute_embedded_journaled_effect(
        &store,
        &mut lease,
        &recorder,
        envelope,
        crate::RuntimeEffectLocalExecutor::unavailable(),
    )
    .await
    .expect("replayed effect");

    assert!(matches!(first, RuntimeEffectOutcome::Sleep));
    assert!(matches!(second, RuntimeEffectOutcome::Sleep));
    assert_eq!(recorder.count_kind(RuntimeEffectKind::Sleep), 1);

    let mismatched = crate::runtime::execute_embedded_journaled_effect(
        &store,
        &mut lease,
        &recorder,
        RuntimeEffectEnvelope::new(invocation, RuntimeEffectCommand::Sleep { duration_ms: 2 }),
        crate::RuntimeEffectLocalExecutor::unavailable(),
    )
    .await
    .expect_err("hash mismatch should fail");
    assert_eq!(mismatched.code, "runtime_effect_journal_hash_mismatch");
}

#[tokio::test]
async fn journaled_turn_effect_requires_lease_before_controller_execution() {
    let recorder = RecordingEffectController::default();
    let store = RecordingStore::default();
    let invocation = test_effect_invocation(
        "root",
        "turn-1",
        1,
        0,
        "sleep",
        RuntimeEffectKind::Sleep,
        "root:turn-1:1:0:sleep:1",
    );

    let mut wrong_lease = store
        .claim_runtime_turn_lease("root", "other-turn", "test", 60_000)
        .await
        .expect("wrong lease");
    let err = crate::runtime::execute_embedded_journaled_effect(
        &store,
        &mut wrong_lease,
        &recorder,
        RuntimeEffectEnvelope::new(invocation, RuntimeEffectCommand::Sleep { duration_ms: 1 }),
        crate::RuntimeEffectLocalExecutor::unavailable(),
    )
    .await
    .expect_err("missing turn lease must fail");

    assert_eq!(err.code, "runtime_turn_lease_required");
    assert_eq!(recorder.count_kind(RuntimeEffectKind::Sleep), 0);
}

#[tokio::test]
async fn process_effect_journal_replays_without_reinvoking_controller_and_rejects_hash_mismatch() {
    let controller = ProcessJournalController::default();
    let store = RecordingStore::default();
    let mut lease = store
        .claim_runtime_turn_lease("root", "turn-process", "test", 60_000)
        .await
        .expect("lease");
    let invocation = test_effect_invocation(
        "root",
        "turn-process",
        1,
        0,
        "process:list:scope-a:live",
        RuntimeEffectKind::Process,
        "root:turn-process:1:0:process:list",
    );
    let envelope = RuntimeEffectEnvelope::new(
        invocation.clone(),
        RuntimeEffectCommand::Process {
            command: ProcessCommand::List {
                owner_scope: ProcessScope::new("scope-a"),
                mode: crate::ProcessListMode::Live,
            },
        },
    );

    let first = crate::runtime::execute_embedded_journaled_effect(
        &store,
        &mut lease,
        &controller,
        envelope.clone(),
        crate::RuntimeEffectLocalExecutor::unavailable(),
    )
    .await
    .expect("first process effect");
    let second = crate::runtime::execute_embedded_journaled_effect(
        &store,
        &mut lease,
        &controller,
        envelope,
        crate::RuntimeEffectLocalExecutor::unavailable(),
    )
    .await
    .expect("replayed process effect");

    assert!(matches!(
        first,
        RuntimeEffectOutcome::Process {
            result: ProcessEffectOutcome::List { .. }
        }
    ));
    assert!(matches!(
        second,
        RuntimeEffectOutcome::Process {
            result: ProcessEffectOutcome::List { .. }
        }
    ));
    assert_eq!(controller.call_count(), 1);

    let mismatched = crate::runtime::execute_embedded_journaled_effect(
        &store,
        &mut lease,
        &controller,
        RuntimeEffectEnvelope::new(
            invocation,
            RuntimeEffectCommand::Process {
                command: ProcessCommand::List {
                    owner_scope: ProcessScope::new("scope-b"),
                    mode: crate::ProcessListMode::Live,
                },
            },
        ),
        crate::RuntimeEffectLocalExecutor::unavailable(),
    )
    .await
    .expect_err("hash mismatch should fail");
    assert_eq!(mismatched.code, "runtime_effect_journal_hash_mismatch");
    assert_eq!(controller.call_count(), 1);
}

#[tokio::test]
async fn journaled_effect_renews_lease_while_pending() {
    let store = RecordingStore::default();
    let mut lease = store
        .claim_runtime_turn_lease("root", "turn-long-effect", "test", 60_000)
        .await
        .expect("lease");
    let invocation = test_effect_invocation(
        "root",
        "turn-long-effect",
        1,
        0,
        "long-sleep",
        RuntimeEffectKind::Sleep,
        "root:turn-long-effect:1:0:sleep:long",
    );
    let outcome = crate::runtime::execute_embedded_journaled_effect(
        &store,
        &mut lease,
        &DelayedSleepController {
            delay: Duration::from_millis(80),
        },
        RuntimeEffectEnvelope::new(invocation, RuntimeEffectCommand::Sleep { duration_ms: 80 }),
        crate::RuntimeEffectLocalExecutor::unavailable(),
    )
    .await
    .expect("long effect");

    assert!(matches!(outcome, RuntimeEffectOutcome::Sleep));
    assert!(store.runtime_turn_lease_renew_count() >= 2);
    assert_eq!(store.runtime_effect_journal_save_count(), 1);
}

#[tokio::test]
async fn journaled_effect_does_not_save_outcome_when_pending_lease_renewal_fails() {
    let store = RecordingStore::default();
    let mut lease = store
        .claim_runtime_turn_lease("root", "turn-expiring-effect", "test", 1)
        .await
        .expect("lease");
    let invocation = test_effect_invocation(
        "root",
        "turn-expiring-effect",
        1,
        0,
        "expiring-sleep",
        RuntimeEffectKind::Sleep,
        "root:turn-expiring-effect:1:0:sleep:expiring",
    );

    let err = crate::runtime::execute_embedded_journaled_effect(
        &store,
        &mut lease,
        &DelayedSleepController {
            delay: Duration::from_millis(80),
        },
        RuntimeEffectEnvelope::new(invocation, RuntimeEffectCommand::Sleep { duration_ms: 80 }),
        crate::RuntimeEffectLocalExecutor::unavailable(),
    )
    .await
    .expect_err("expired pending lease renewal should fail");

    assert_eq!(err.code, "runtime_store");
    assert_eq!(store.runtime_effect_journal_save_count(), 0);
}

#[tokio::test]
async fn turn_effect_envelope_carries_checkpoint_digest_not_checkpoint_payload() {
    let recorder = RecordingEffectController::default();
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
    let large_marker = format!("large-turn-marker-{}", "x".repeat(16_384));

    let turn = runtime
        .run_turn_assembled(
            TurnInput::text(large_marker.clone()),
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    assert!(matches!(turn.outcome, TurnOutcome::Finished(_)));
    let checkpoint_envelope = recorder
        .envelopes()
        .into_iter()
        .find(|encoded| {
            serde_json::from_str::<RuntimeEffectEnvelope>(encoded)
                .expect("decode envelope")
                .command
                .kind()
                == RuntimeEffectKind::Checkpoint
        })
        .expect("checkpoint envelope");
    let decoded: RuntimeEffectEnvelope =
        serde_json::from_str(&checkpoint_envelope).expect("decode checkpoint envelope");
    assert!(
        decoded
            .invocation
            .checkpoint_hash
            .as_deref()
            .is_some_and(|hash| hash.len() == 64 && hash.chars().all(|ch| ch.is_ascii_hexdigit()))
    );
    assert!(!checkpoint_envelope.contains("\"turn_checkpoint\":"));
    assert!(!checkpoint_envelope.contains(&large_marker));
    assert!(!checkpoint_envelope.contains("\"messages\""));
    assert!(!checkpoint_envelope.contains("\"events\""));
}

#[tokio::test]
async fn controller_rejection_fails_turn_explicitly() {
    let mut runtime = runtime_with_plugins_and_tools_and_host(
        Vec::new(),
        Arc::new(EmptyTools),
        mock_provider(Vec::new()),
        EmbeddedRuntimeHost::new(runtime_host_config_with_effect_controller(Arc::new(
            RejectingEffectController,
        ))),
    )
    .await;

    let turn = runtime
        .run_turn_assembled(TurnInput::text("hello"), CancellationToken::new())
        .await
        .expect("turn");

    assert!(matches!(
        turn.outcome,
        TurnOutcome::Stopped(TurnStop::RuntimeError)
    ));
    assert!(turn.errors.iter().any(|issue| {
        issue.kind == "runtime_effect_controller"
            && issue.code.as_deref() == Some("test_controller_rejected")
    }));
}

#[tokio::test]
async fn wrong_controller_outcome_fails_turn_explicitly() {
    let mut runtime = runtime_with_plugins_and_tools_and_host(
        Vec::new(),
        Arc::new(EmptyTools),
        mock_provider(Vec::new()),
        EmbeddedRuntimeHost::new(runtime_host_config_with_effect_controller(Arc::new(
            WrongOutcomeEffectController,
        ))),
    )
    .await;

    let turn = runtime
        .run_turn_assembled(TurnInput::text("hello"), CancellationToken::new())
        .await
        .expect("turn");

    assert!(matches!(
        turn.outcome,
        TurnOutcome::Stopped(TurnStop::RuntimeError)
    ));
    assert!(turn.errors.iter().any(|issue| {
        issue.kind == "runtime_effect_controller"
            && issue.code.as_deref() == Some("runtime_effect_wrong_outcome")
    }));
}

#[tokio::test]
async fn scoped_borrowed_effect_controller_uses_required_stable_turn_id() {
    let recorder = RecordingEffectController::default();
    assert!(DurableTurnScope::new(&recorder, "").is_err());
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
        EmbeddedRuntimeHost::new(RuntimeHostConfig::in_memory()),
    )
    .await;

    let durable_turn_scope =
        DurableTurnScope::new(&recorder, "stable-scoped-turn").expect("durable turn scope");
    let turn = runtime
        .stream_turn(
            TurnInput::text("hello"),
            TurnOptions::new(CancellationToken::new())
                .with_events(&NoopEventSink)
                .with_durable_turn_scope(durable_turn_scope),
        )
        .await
        .expect("turn");

    assert!(matches!(turn.outcome, TurnOutcome::Finished(_)));
    assert!(
        recorder
            .records()
            .iter()
            .all(|record| record.replay_key.contains("stable-scoped-turn"))
    );
}

#[tokio::test]
async fn durable_controller_rejects_ephemeral_attachment_store_before_turn_runs() {
    let controller = DurableAttachmentRequiredController;
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
        EmbeddedRuntimeHost::new(RuntimeHostConfig::in_memory()),
    )
    .await;
    let durable_turn_scope =
        DurableTurnScope::new(&controller, "durable-turn").expect("durable turn scope");

    let err = runtime
        .stream_turn(
            TurnInput::text("hello"),
            TurnOptions::new(CancellationToken::new())
                .with_events(&NoopEventSink)
                .with_durable_turn_scope(durable_turn_scope),
        )
        .await
        .expect_err("ephemeral attachment store should be rejected");

    assert_eq!(
        err.code,
        crate::RuntimeErrorCode::DurableSubstrateRequired {
            facet: crate::DurableSubstrateFacet::AttachmentStore,
        }
    );
}

#[tokio::test]
async fn durable_controller_rejects_ephemeral_artifact_store_before_turn_runs() {
    // Durable attachment store clears the first facet check, so the scope
    // boundary must reject on the ephemeral Lashlang artifact store next.
    let controller = DurableAttachmentRequiredController;
    let mut runtime = runtime_with_plugins_and_tools_and_host(
        Vec::new(),
        Arc::new(EmptyTools),
        done_once_provider(),
        EmbeddedRuntimeHost::new({
            let mut config = RuntimeHostConfig::in_memory();
            config.durability.attachment_store =
                Arc::new(DurableInMemoryAttachmentStore::default());
            config
        }),
    )
    .await;
    let durable_turn_scope =
        DurableTurnScope::new(&controller, "durable-artifact-turn").expect("durable turn scope");

    let err = runtime
        .stream_turn(
            TurnInput::text("hello"),
            TurnOptions::new(CancellationToken::new())
                .with_events(&NoopEventSink)
                .with_durable_turn_scope(durable_turn_scope),
        )
        .await
        .expect_err("ephemeral artifact store should be rejected");

    assert_eq!(
        err.code,
        crate::RuntimeErrorCode::DurableSubstrateRequired {
            facet: crate::DurableSubstrateFacet::ArtifactStore,
        }
    );
}

#[tokio::test]
async fn durable_controller_rejects_ephemeral_session_store_before_turn_runs() {
    // Durable attachment + artifact stores clear the first two facet checks, so
    // an ephemeral (default-tier) session store backing the turn must be the
    // facet that fails. The in-memory `RecordingStore` reports the Inline tier.
    let controller = DurableAttachmentRequiredController;
    let store = Arc::new(RecordingStore::default());
    let mut runtime = runtime_with_plugins_and_tools_and_host_and_store(
        Vec::new(),
        Arc::new(EmptyTools),
        done_once_provider(),
        EmbeddedRuntimeHost::new(runtime_host_config_with_durability(
            Arc::new(DurableInMemoryAttachmentStore::default()),
            Arc::new(DurableInMemoryArtifactStore::default()),
        )),
        store,
    )
    .await;
    let durable_turn_scope =
        DurableTurnScope::new(&controller, "durable-session-turn").expect("durable turn scope");

    let err = runtime
        .stream_turn(
            TurnInput::text("hello"),
            TurnOptions::new(CancellationToken::new())
                .with_events(&NoopEventSink)
                .with_durable_turn_scope(durable_turn_scope),
        )
        .await
        .expect_err("ephemeral session store should be rejected");

    assert_eq!(
        err.code,
        crate::RuntimeErrorCode::DurableSubstrateRequired {
            facet: crate::DurableSubstrateFacet::SessionStore,
        }
    );
}

#[tokio::test]
async fn durable_controller_with_all_durable_stores_runs_turn() {
    // Positive control: a durable controller wired against fully durable stores
    // must NOT be rejected by the scope boundary — the check is a loud guard
    // against ephemeral substrate, not a blanket veto on durable controllers.
    let controller = DurableAttachmentRequiredController;
    let mut runtime = runtime_with_plugins_and_tools_and_host(
        Vec::new(),
        Arc::new(EmptyTools),
        done_once_provider(),
        EmbeddedRuntimeHost::new(runtime_host_config_with_durability(
            Arc::new(DurableInMemoryAttachmentStore::default()),
            Arc::new(DurableInMemoryArtifactStore::default()),
        )),
    )
    .await;
    let durable_turn_scope =
        DurableTurnScope::new(&controller, "durable-ok-turn").expect("durable turn scope");

    let turn = runtime
        .stream_turn(
            TurnInput::text("hello"),
            TurnOptions::new(CancellationToken::new())
                .with_events(&NoopEventSink)
                .with_durable_turn_scope(durable_turn_scope),
        )
        .await
        .expect("durable controller + all-durable stores should run");

    assert!(matches!(turn.outcome, TurnOutcome::Finished(_)));
}

#[tokio::test]
async fn scoped_borrowed_effect_controller_reaches_tool_direct_completions() {
    struct DirectTool;

    fn direct_tool_definition() -> crate::ToolDefinition {
        crate::ToolDefinition::raw(
            "tool:direct_tool",
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
                .direct_completions()
                .complete(
                    crate::DirectRequest::text("mock-model", "nested"),
                    "tool-direct",
                )
                .await
                .expect("tool direct completion");
            crate::ToolResult::ok(serde_json::json!({ "text": completion.text }))
        }
    }

    let default_recorder = RecordingEffectController::default();
    let scoped_recorder = RecordingEffectController::default();
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

    let durable_turn_scope =
        DurableTurnScope::new(&scoped_recorder, "scoped-tool-direct").expect("durable turn scope");
    let turn = runtime
        .stream_turn(
            TurnInput::text("use direct tool"),
            TurnOptions::new(CancellationToken::new())
                .with_events(&NoopEventSink)
                .with_durable_turn_scope(durable_turn_scope),
        )
        .await
        .expect("turn");

    assert!(matches!(turn.outcome, TurnOutcome::Finished(_)));
    assert_eq!(scoped_recorder.count_kind(RuntimeEffectKind::ToolCall), 1);
    assert_eq!(default_recorder.count_kind(RuntimeEffectKind::Direct), 0);
    assert!(scoped_recorder.records().iter().any(|record| {
        record.kind == RuntimeEffectKind::ToolCall && record.replay_key.contains("direct-call-1")
    }));
}

#[tokio::test]
async fn scoped_retry_sleep_records_turn_and_parent_tool_identity() {
    struct RetryOnceTool {
        attempts: Arc<std::sync::atomic::AtomicUsize>,
    }

    fn retry_once_tool_definition() -> crate::ToolDefinition {
        crate::ToolDefinition::raw(
            "tool:retry_once",
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

    let recorder = RecordingEffectController::default();
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
        EmbeddedRuntimeHost::new(RuntimeHostConfig::in_memory()),
    )
    .await;

    let durable_turn_scope =
        DurableTurnScope::new(&recorder, "scoped-retry-sleep").expect("durable turn scope");
    let turn = runtime
        .stream_turn(
            TurnInput::text("use retry tool"),
            TurnOptions::new(CancellationToken::new())
                .with_events(&NoopEventSink)
                .with_durable_turn_scope(durable_turn_scope),
        )
        .await
        .expect("turn");

    assert!(matches!(turn.outcome, TurnOutcome::Finished(_)));
    let tool_records = recorder
        .records()
        .into_iter()
        .filter(|record| record.kind == RuntimeEffectKind::ToolCall)
        .collect::<Vec<_>>();
    assert_eq!(tool_records.len(), 1);
    let tool = &tool_records[0];
    assert_eq!(tool.turn_id.as_deref(), Some("scoped-retry-sleep"));
    assert!(tool.effect_id.contains("retry-call-1"));
    assert!(tool.replay_key.contains("scoped-retry-sleep"));
    assert!(tool.replay_key.contains("retry-call-1"));
}

#[tokio::test]
async fn tool_call_effect_crosses_controller_per_logical_call_and_runs_local_tools() {
    let recorder = RecordingEffectController::default();
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
                protocol_turn_options: None,
                trace_turn_id: None,
                protocol_extension: None,
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
        .map(|record| record.replay_key)
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
async fn exec_and_execution_surface_effects_cross_controller_once() {
    let recorder = RecordingEffectController::default();
    let policy = SessionPolicy {
        provider_id: "mock".to_string(),
        model: crate::ModelSpec::from_token_limits("mock-model", None, 200_000, None, None)
            .expect("valid model spec"),
        ..SessionPolicy::default()
    };
    let plugin_session =
        crate::PluginHost::new(vec![Arc::new(EffectControllerTestProtocolFactory {
            install_code_executor: true,
        })])
        .build_session("root", None)
        .expect("plugins");
    let mut runtime = LashRuntime::from_embedded_state(
        policy,
        host_with_effect_recorder(recorder.clone()),
        RuntimeServices::new(plugin_session),
        RuntimeSessionState::default(),
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
                protocol_turn_options: None,
                trace_turn_id: None,
                protocol_extension: None,
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
async fn start_exec_without_code_executor_stops_as_runtime_error() {
    let policy = SessionPolicy {
        provider_id: "mock".to_string(),
        model: crate::ModelSpec::from_token_limits("mock-model", None, 200_000, None, None)
            .expect("valid model spec"),
        ..SessionPolicy::default()
    };
    let plugin_session =
        crate::PluginHost::new(vec![Arc::new(EffectControllerTestProtocolFactory {
            install_code_executor: false,
        })])
        .build_session("root", None)
        .expect("plugins");
    let mut runtime = LashRuntime::from_embedded_state(
        policy,
        EmbeddedRuntimeHost::new(runtime_host_config_with_provider(
            mock_provider(Vec::new()).into_handle(),
        )),
        RuntimeServices::new(plugin_session),
        RuntimeSessionState::default(),
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
                protocol_turn_options: None,
                trace_turn_id: None,
                protocol_extension: None,
                turn_context: crate::TurnContext::default(),
            },
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    assert!(matches!(
        turn.outcome,
        TurnOutcome::Stopped(TurnStop::RuntimeError)
    ));
    assert!(turn.errors.iter().any(|issue| {
        issue
            .message
            .contains("code execution is not available in this session")
    }));
}

#[tokio::test]
async fn direct_completion_crosses_controller_and_records_usage_and_trace() {
    let recorder = RecordingEffectController::default();
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
    let host = EmbeddedRuntimeHost::new({
        let mut config = runtime_host_config_with_effect_controller(Arc::new(recorder.clone()));
        config.tracing.trace_sink = Some(Arc::new(lash_trace::JsonlTraceSink::new(
            trace_path.clone(),
        )));
        config
    });
    let runtime =
        runtime_with_plugins_and_tools_and_host(Vec::new(), Arc::new(EmptyTools), transport, host)
            .await;

    let manager = runtime.runtime_session_services().expect("session manager");
    let direct = manager.direct_completion_client(
        RuntimeEffectControllerHandle::shared(Arc::new(recorder.clone())),
        None,
        None,
    );
    let mut request = crate::DirectRequest::text("mock-model", "summarize");
    request.caused_by = Some(CausalRef::ToolCall {
        session_id: "root".to_string(),
        call_id: "originating-tool-call".to_string(),
    });
    let completion = direct
        .direct_completion(request, "direct-test")
        .await
        .expect("direct completion");

    assert_eq!(completion.text, "direct answer");
    assert_eq!(completion.usage.input_tokens, 7);
    assert_eq!(recorder.count_kind(RuntimeEffectKind::Direct), 1);
    assert!(recorder.records().iter().any(|record| {
        record.kind == RuntimeEffectKind::Direct
            && record
                .replay_key
                .contains("cause:tool_call:root:originating-tool-call")
    }));
    let ledger = runtime.shared_token_ledger.lock().expect("token ledger");
    assert_eq!(ledger.len(), 1);
    assert_eq!(ledger[0].source, "direct-test");
    assert_eq!(ledger[0].model, "mock-model");
    assert_eq!(ledger[0].usage.input_tokens, 7);
}

#[tokio::test]
async fn in_turn_direct_completion_requires_lease_and_journals_under_it() {
    let recorder = RecordingEffectController::default();
    let store = Arc::new(RecordingStore::default());
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
                cached_input_tokens: 0,
                reasoning_tokens: 0,
            },
            ..LlmResponse::default()
        }),
    }]);
    let host = EmbeddedRuntimeHost::new(runtime_host_config_with_effect_controller(Arc::new(
        recorder.clone(),
    )));
    let runtime = runtime_with_plugins_and_tools_and_host_and_store(
        Vec::new(),
        Arc::new(EmptyTools),
        transport,
        host,
        store.clone(),
    )
    .await;
    let lease = store
        .claim_runtime_turn_lease("root", "turn-direct", "test", 60_000)
        .await
        .expect("lease");

    let manager = runtime.runtime_session_services().expect("session manager");
    let direct = manager.direct_completion_client(
        RuntimeEffectControllerHandle::shared(Arc::new(recorder.clone())),
        Some("turn-direct".to_string()),
        Some(lease),
    );
    let completion = direct
        .direct_completion(
            crate::DirectRequest::text("mock-model", "summarize"),
            "direct-test",
        )
        .await
        .expect("direct completion");

    assert_eq!(completion.text, "direct answer");
    assert!(recorder.records().iter().any(|record| {
        record.kind == RuntimeEffectKind::Direct && record.turn_id.as_deref() == Some("turn-direct")
    }));
    assert_eq!(store.runtime_effect_journal_save_count(), 1);
    assert!(store.runtime_turn_lease_renew_count() >= 1);

    // A direct effect must record usage into the shared in-memory ledger only;
    // that ledger is drained and persisted exactly once by the lease-fenced
    // turn-commit pipeline. The direct path must NOT issue its own
    // out-of-band `commit_runtime_state` mid-turn: doing so races the owning
    // turn's head-revision CAS, and — because the effect's `apply` re-runs on
    // journal replay — would double-merge the usage into the durable cost
    // ledger on every crash/resume. Since the path performs zero durable
    // commits, replay cannot double-count it.
    assert_eq!(
        *store
            .runtime_commit_count
            .lock()
            .expect("runtime commit count"),
        0,
        "in-turn direct completion must not commit runtime state out-of-band"
    );
    let ledger = runtime.shared_token_ledger.lock().expect("token ledger");
    assert_eq!(ledger.len(), 1);
    assert_eq!(ledger[0].usage.input_tokens, 7);
}

#[tokio::test]
async fn direct_effect_restores_required_streaming_for_provider_execution() {
    let saw_stream_events = Arc::new(AtomicBool::new(false));
    let captured = Arc::clone(&saw_stream_events);
    let transport = TestProvider::builder()
        .kind("stream-required")
        .requires_streaming(true)
        .complete(move |request| {
            let captured = Arc::clone(&captured);
            async move {
                captured.store(request.stream_events.is_some(), Ordering::SeqCst);
                Ok(LlmResponse {
                    full_text: "direct answer".to_string(),
                    parts: vec![LlmOutputPart::Text {
                        text: "direct answer".to_string(),
                        response_meta: None,
                    }],
                    ..LlmResponse::default()
                })
            }
        })
        .build();
    let runtime = runtime_with_plugins_and_tools_and_host(
        Vec::new(),
        Arc::new(EmptyTools),
        transport,
        EmbeddedRuntimeHost::new(RuntimeHostConfig::in_memory()),
    )
    .await;

    let manager = runtime.runtime_session_services().expect("session manager");
    let direct = manager.direct_completion_client(
        RuntimeEffectControllerHandle::shared(Arc::new(InlineRuntimeEffectController)),
        None,
        None,
    );
    let completion = direct
        .direct_completion(
            crate::DirectRequest::text("mock-model", "summarize"),
            "direct-test",
        )
        .await
        .expect("direct completion");

    assert_eq!(completion.text, "direct answer");
    assert!(saw_stream_events.load(Ordering::SeqCst));
}

#[tokio::test]
async fn direct_llm_completion_crosses_controller_and_records_usage_and_trace() {
    let recorder = RecordingEffectController::default();
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
    let host = EmbeddedRuntimeHost::new({
        let mut config = runtime_host_config_with_effect_controller(Arc::new(recorder.clone()));
        config.tracing.trace_sink = Some(Arc::new(lash_trace::JsonlTraceSink::new(
            trace_path.clone(),
        )));
        config
    });
    let runtime =
        runtime_with_plugins_and_tools_and_host(Vec::new(), Arc::new(EmptyTools), transport, host)
            .await;

    let manager = runtime.runtime_session_services().expect("session manager");
    let direct = manager.direct_completion_client(
        RuntimeEffectControllerHandle::shared(Arc::new(recorder.clone())),
        None,
        None,
    );
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
        generation: crate::GenerationOptions::default(),
        provider_trace: None,
    };
    let completion = direct
        .direct_llm_completion(request, "direct-llm-test")
        .await
        .expect("direct llm completion");

    assert_eq!(completion.response.full_text, "raw direct answer");
    assert_eq!(completion.usage.output_tokens, 6);
    assert_eq!(recorder.count_kind(RuntimeEffectKind::Direct), 1);
    let ledger = runtime.shared_token_ledger.lock().expect("token ledger");
    assert_eq!(ledger.len(), 1);
    assert_eq!(ledger[0].source, "direct-llm-test");
    assert_eq!(ledger[0].model, "mock-model");
    assert_eq!(ledger[0].usage.input_tokens, 4);
}

#[tokio::test]
async fn direct_llm_completion_envelope_stores_attachment_refs_not_bytes() {
    let recorder = RecordingEffectController::default();
    let runtime = runtime_with_plugins_and_tools_and_host(
        Vec::new(),
        Arc::new(EmptyTools),
        mock_provider(Vec::new()),
        EmbeddedRuntimeHost::new(RuntimeHostConfig::in_memory()),
    )
    .await;

    let image_bytes = vec![137, 80, 78, 71];
    let expected_attachment_id = crate::attachments::content_id(&image_bytes).to_string();
    let request = LlmRequest {
        model: "mock-model".to_string(),
        messages: vec![LlmMessage::new(
            LlmRole::User,
            vec![LlmContentBlock::Image { attachment_idx: 0 }],
        )],
        attachments: vec![LlmAttachment::bytes("image/png", image_bytes)],
        tools: Arc::new(Vec::new()),
        tool_choice: LlmToolChoice::None,
        model_variant: None,
        session_id: None,
        output_spec: None,
        stream_events: None,
        generation: crate::GenerationOptions::default(),
        provider_trace: None,
    };

    let manager = runtime.runtime_session_services().expect("session manager");
    let direct = manager.direct_completion_client(
        RuntimeEffectControllerHandle::shared(Arc::new(recorder.clone())),
        None,
        None,
    );
    let completion = direct
        .direct_llm_completion(request, "direct-image-test")
        .await
        .expect("direct llm completion");

    assert_eq!(completion.response.full_text, "raw direct answer");
    let envelope = recorder
        .envelopes()
        .into_iter()
        .find(|envelope| envelope.contains("\"type\":\"direct\""))
        .expect("direct llm envelope");
    assert!(!envelope.contains("\"data\""));
    assert!(envelope.contains(&expected_attachment_id));
}

fn effect_module_sources(manifest_dir: &std::path::Path) -> Vec<PathBuf> {
    let dir = manifest_dir.join("src/runtime/effect");
    let mut paths = std::fs::read_dir(&dir)
        .expect("read effect module directory")
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("rs"))
        .collect::<Vec<_>>();
    paths.sort();
    paths
}

#[test]
fn runtime_effect_executor_has_no_legacy_future_api() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source_files = effect_module_sources(&manifest_dir)
        .into_iter()
        .chain([
            manifest_dir.join("src/runtime/turn_driver.rs"),
            manifest_dir.join("src/runtime/session_manager/direct.rs"),
        ])
        .collect::<Vec<_>>();
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

#[test]
fn runtime_effect_controller_cutover_has_no_legacy_host_request_or_fallback_symbols() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source_files = effect_module_sources(&manifest_dir)
        .into_iter()
        .chain([
            manifest_dir.join("src/runtime/turn_driver.rs"),
            manifest_dir.join("src/runtime/session_manager/direct.rs"),
            manifest_dir.join("src/tool_dispatch.rs"),
            manifest_dir.join("src/runtime/assembly.rs"),
        ])
        .collect::<Vec<_>>();
    let forbidden = [
        ["Runtime", "Effect", "Host"].concat(),
        ["Local", "Runtime", "Effect", "Host"].concat(),
        ["Runtime", "Effect", "Request"].concat(),
        ["Background", "Task", "Start", "Request"].concat(),
        ["missing", "_tool", "_result", "_completed", "_call"].concat(),
        ["fallback", "_assistant", "_output", "_from", "_state"].concat(),
        ["b", "\"", "un", "serializable", "\""].concat(),
    ];
    for path in source_files {
        let source = std::fs::read_to_string(&path).expect("read runtime effect source");
        for symbol in &forbidden {
            assert!(
                !source.contains(symbol.as_str()),
                "{} still mentions {symbol}",
                path.display()
            );
        }
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

struct EffectControllerTestProtocolFactory {
    install_code_executor: bool,
}

impl crate::PluginFactory for EffectControllerTestProtocolFactory {
    fn id(&self) -> &'static str {
        "protocol_standard"
    }

    fn build(
        &self,
        _ctx: &crate::PluginSessionContext,
    ) -> Result<Arc<dyn crate::SessionPlugin>, crate::PluginError> {
        Ok(Arc::new(EffectControllerTestProtocolPlugin {
            install_code_executor: self.install_code_executor,
        }))
    }
}

struct EffectControllerTestProtocolPlugin {
    install_code_executor: bool,
}

impl crate::SessionPlugin for EffectControllerTestProtocolPlugin {
    fn id(&self) -> &'static str {
        "effect_controller_test_protocol"
    }

    fn register(&self, registrar: &mut crate::PluginRegistrar) -> Result<(), crate::PluginError> {
        registrar
            .protocol()
            .session(Arc::new(EffectControllerTestProtocolSession))?;
        if self.install_code_executor {
            registrar
                .execution()
                .code_executor(Arc::new(EffectControllerTestCodeExecutor))?;
        }
        registrar
            .protocol()
            .protocol_driver(Arc::new(EffectControllerTestProtocolDriver))?;
        Ok(())
    }
}

struct EffectControllerTestProtocolSession;

#[async_trait::async_trait]
impl ProtocolSessionPlugin for EffectControllerTestProtocolSession {}

struct EffectControllerTestCodeExecutor;

#[async_trait::async_trait]
impl crate::plugin::CodeExecutorPlugin for EffectControllerTestCodeExecutor {
    async fn execute_code(
        &self,
        _ctx: crate::RuntimeExecutionContext<'_>,
        _request: crate::ExecRequest,
    ) -> Result<crate::ExecResponse, crate::SessionError> {
        Ok(crate::ExecResponse {
            observations: vec!["exec output".to_string()],
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

struct EffectControllerTestProtocolDriver;

impl ProtocolDriverPlugin for EffectControllerTestProtocolDriver {
    fn build_preamble(&self, input: crate::ProtocolBuildInput) -> crate::TurnDriverPreamble {
        crate::TurnDriverPreamble {
            config: crate::TurnDriverConfig::chat(
                Arc::new(EffectControllerTestDriver),
                true,
                Arc::new(effect_controller_turn_limit_final_message),
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

fn effect_controller_turn_limit_final_message(
    message_id: String,
    max_turns: usize,
) -> crate::Message {
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

struct EffectControllerTestDriver;

impl lash_sansio::ProtocolDriverHandle<crate::HostTurnProtocol> for EffectControllerTestDriver {
    fn prepare_protocol_iteration(
        &self,
        _ctx: crate::DriverContextView<'_>,
    ) -> Vec<crate::DriverAction> {
        vec![crate::DriverAction::StartExec {
            code: "print('effect controller')".to_string(),
            driver_state: crate::ProtocolDriverState::new(
                "effect_controller_test_protocol",
                serde_json::Value::Null,
            ),
        }]
    }

    fn handle_llm_success(
        &self,
        _ctx: crate::DriverContextView<'_>,
        _waiting: lash_sansio::WaitingLlmState<crate::HostTurnProtocol>,
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
        _waiting: lash_sansio::WaitingExecState<crate::HostTurnProtocol>,
        result: Result<crate::ExecResponse, String>,
    ) -> Vec<crate::DriverAction> {
        match result {
            Ok(response) => vec![crate::DriverAction::Finish(TurnOutcome::Finished(
                TurnFinish::SubmittedValue {
                    value: serde_json::json!(response.observations.join("\n")),
                },
            ))],
            Err(error) => vec![
                crate::DriverAction::Emit(crate::SessionEvent::Error {
                    message: error,
                    envelope: None,
                }),
                crate::DriverAction::Finish(TurnOutcome::Stopped(TurnStop::RuntimeError)),
            ],
        }
    }
}
