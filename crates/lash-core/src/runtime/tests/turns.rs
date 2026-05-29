use super::*;

async fn standard_runtime_with_transport_and_queue_store(
    transport: TestProvider,
) -> (LashRuntime, Arc<RecordingStore>) {
    let store = Arc::new(RecordingStore::default());
    let runtime_store: Arc<dyn crate::store::RuntimePersistence> = store.clone();
    let mut runtime = runtime_with_plugins_and_tools_and_host_and_store(
        Vec::new(),
        Arc::new(EmptyTools),
        transport,
        test_host_config(),
        runtime_store,
    )
    .await;
    runtime.host.process_registry = Some(Arc::new(crate::TestLocalProcessRegistry::default()));
    (runtime, store)
}

async fn append_process_wake_to_queue(
    registry: &dyn crate::ProcessRegistry,
    store: &RecordingStore,
    process_id: &str,
    request: crate::ProcessEventAppendRequest,
) -> crate::ProcessWakeDelivery {
    let appended = registry
        .append_event(process_id, request)
        .await
        .expect("append wake");
    let wake = appended.wake_delivery.expect("wake delivery");
    crate::store::RuntimePersistence::enqueue_queued_work(
        store,
        crate::process_wake_batch_draft(wake.clone()),
    )
    .await
    .expect("enqueue wake");
    wake
}

async fn enqueue_turn_input_for_checkpoint(
    store: &RecordingStore,
    session_id: &str,
    source_key: Option<String>,
    input: TurnInput,
) {
    let mut draft = crate::QueuedWorkBatchDraft::new(
        session_id.to_string(),
        crate::DeliveryPolicy::EarliestSafeBoundary,
        crate::SlotPolicy::Join,
        vec![crate::QueuedWorkPayload::turn_input(input)],
    );
    draft.source_key = source_key;
    crate::store::RuntimePersistence::enqueue_queued_work(store, draft)
        .await
        .expect("enqueue turn input");
}

#[tokio::test]
async fn session_config_change_hook_receives_context_window_updates() {
    let observed = Arc::new(tokio::sync::Mutex::new(Vec::new()));
    let observed_hook = Arc::clone(&observed);
    let plugin = Arc::new(RuntimeTestPluginFactory {
        build: Arc::new(move |_| {
            let observed = Arc::clone(&observed_hook);
            Ok(Arc::new(RuntimeTestPlugin {
                before_turn: None,
                checkpoint: None,
                tool_result_projector: None,
                runtime_event: Some(Arc::new(move |event| {
                    let observed = Arc::clone(&observed);
                    Box::pin(async move {
                        if let crate::plugin::PluginLifecycleEvent::SessionConfigChanged(ctx) =
                            event
                        {
                            observed.lock().await.push((ctx.previous, ctx.current));
                        }
                        Ok(())
                    })
                })),
                external_registrar: None,
            }))
        }),
    });
    let transport = mock_provider(Vec::new());
    let mut runtime = runtime_with_plugins(vec![plugin], transport).await;

    let alt_provider = TestProvider::builder()
        .kind("alt")
        .complete_error("alt provider not wired")
        .build();
    let alt_model = crate::ModelSpec::from_token_limits("alt-model", None, 123_456, None, None)
        .expect("valid model spec");
    runtime
        .update_session_config(
            Some(alt_provider.into_handle()),
            Some(alt_model.clone()),
            None,
        )
        .await;

    let changes = observed.lock().await;
    assert_eq!(changes.len(), 1);
    let (previous, current) = &changes[0];
    assert_eq!(previous.provider.kind(), "mock");
    assert_eq!(current.provider.kind(), "alt");
    assert_eq!(current.model.id, "alt-model");
    assert_ne!(
        previous.context_window_tokens(),
        current.context_window_tokens()
    );
}

#[tokio::test]
async fn plugin_before_turn_can_abort_and_inject_messages() {
    let plugin = Arc::new(RuntimeTestPluginFactory {
        build: Arc::new(|_| {
            Ok(Arc::new(RuntimeTestPlugin {
                before_turn: Some(Arc::new(|_| {
                    Box::pin(async {
                        Ok(vec![
                            crate::PluginDirective::EnqueueMessages {
                                messages: vec![crate::PluginMessage::text(
                                    crate::MessageRole::System,
                                    "plugin preface",
                                )],
                            },
                            crate::PluginDirective::AbortTurn {
                                code: "blocked".to_string(),
                                message: "plugin stopped the turn".to_string(),
                            },
                        ])
                    })
                })),
                checkpoint: None,
                tool_result_projector: None,
                runtime_event: None,
                external_registrar: None,
            }))
        }),
    });
    let transport = mock_provider(Vec::new());
    let mut runtime = runtime_with_plugins(vec![plugin], transport).await;

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

    assert!(matches!(&turn.outcome, TurnOutcome::Stopped(_)));
    assert!(matches!(
        &turn.outcome,
        TurnOutcome::Stopped(TurnStop::PluginAbort)
    ));
    assert!(turn.errors.iter().any(|issue| issue.kind == "plugin"));
    assert!(
        active_conversation_messages(&turn.state)
            .iter()
            .any(|message| {
                message
                    .parts
                    .iter()
                    .any(|part| part.content.contains("plugin preface"))
            })
    );
}

#[tokio::test]
async fn normal_turn_stores_effective_user_text_in_state() {
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
    let mut runtime = runtime_with_plugins(Vec::new(), transport).await;

    let turn = runtime
        .run_turn_assembled(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "/yolopush\n\n<skill>\nbody\n</skill>".to_string(),
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

    let read_model = turn.state.read_model();
    let user_message = read_model
        .messages
        .iter()
        .find(|message| message.role == MessageRole::User)
        .expect("user message");
    assert_eq!(
        user_message.parts.first().map(|part| part.content.as_str()),
        Some("/yolopush\n\n<skill>\nbody\n</skill>")
    );
}

#[tokio::test]
async fn retryable_llm_failures_exhaust_and_fail_turn() {
    let transport = mock_provider(vec![
        MockCall {
            stream_events: Vec::new(),
            response: Err(
                crate::llm::transport::LlmTransportError::new("provider unavailable")
                    .retryable(true)
                    .with_code("http_500"),
            ),
        },
        MockCall {
            stream_events: Vec::new(),
            response: Err(
                crate::llm::transport::LlmTransportError::new("provider unavailable")
                    .retryable(true)
                    .with_code("http_500"),
            ),
        },
        MockCall {
            stream_events: Vec::new(),
            response: Err(
                crate::llm::transport::LlmTransportError::new("provider unavailable")
                    .retryable(true)
                    .with_code("http_500"),
            ),
        },
        MockCall {
            stream_events: Vec::new(),
            response: Err(
                crate::llm::transport::LlmTransportError::new("provider unavailable")
                    .retryable(true)
                    .with_code("http_500"),
            ),
        },
    ]);
    let mut runtime = runtime_with_plugins(Vec::new(), transport).await;

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

    assert!(matches!(&turn.outcome, TurnOutcome::Stopped(_)));
    assert!(matches!(
        &turn.outcome,
        TurnOutcome::Stopped(TurnStop::ProviderError)
    ));
    assert!(turn.errors.iter().any(|issue| issue.kind == "llm_provider"));
    assert!(
        turn.errors
            .iter()
            .any(|issue| issue.message.contains("provider unavailable"))
    );
}

#[tokio::test]
async fn queued_checkpoint_input_continues_standard_turn() {
    let transport = mock_provider(vec![
        MockCall {
            stream_events: Vec::new(),
            response: Ok(LlmResponse {
                full_text: "First answer.".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "First answer.".to_string(),
                    response_meta: None,
                }],
                ..LlmResponse::default()
            }),
        },
        MockCall {
            stream_events: Vec::new(),
            response: Ok(LlmResponse {
                full_text: "Second answer.".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "Second answer.".to_string(),
                    response_meta: None,
                }],
                ..LlmResponse::default()
            }),
        },
    ]);
    let (mut runtime, store) = standard_runtime_with_transport_and_queue_store(transport).await;
    enqueue_turn_input_for_checkpoint(
        store.as_ref(),
        "root",
        None,
        TurnInput::text("one more thing"),
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

    assert!(
        active_conversation_messages(&turn.state)
            .iter()
            .any(|message| {
                message.role == MessageRole::Assistant
                    && message
                        .parts
                        .iter()
                        .any(|part| part.content.contains("Second answer."))
            })
    );
    assert!(
        active_conversation_messages(&turn.state)
            .iter()
            .all(|message| {
                !(message.role == MessageRole::User
                    && message
                        .parts
                        .iter()
                        .any(|part| part.content == "one more thing"))
            })
    );
}

#[tokio::test]
async fn queued_checkpoint_input_preserves_images() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let captured_requests = Arc::clone(&requests);
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let captured_calls = Arc::clone(&calls);
    let transport = TestProvider::builder()
        .kind("mock")
        .complete(move |request| {
            let captured_requests = Arc::clone(&captured_requests);
            let captured_calls = Arc::clone(&captured_calls);
            async move {
                captured_requests
                    .lock()
                    .expect("request capture lock")
                    .push(request);
                let call = captured_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let text = if call == 0 {
                    "First answer."
                } else {
                    "Second answer."
                };
                Ok(LlmResponse {
                    full_text: text.to_string(),
                    parts: vec![LlmOutputPart::Text {
                        text: text.to_string(),
                        response_meta: None,
                    }],
                    ..LlmResponse::default()
                })
            }
        })
        .build();
    let (mut runtime, store) = standard_runtime_with_transport_and_queue_store(transport).await;
    enqueue_turn_input_for_checkpoint(
        store.as_ref(),
        "root",
        None,
        TurnInput::text("see image").with_image_ref("test-image", vec![1, 2, 3]),
    )
    .await;

    runtime
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

    let requests = requests.lock().expect("request capture lock").clone();
    assert_eq!(requests.len(), 2);
    assert!(requests[1].messages.iter().any(|message| {
        message.role == crate::llm::types::LlmRole::User
            && message
                .blocks
                .iter()
                .any(|block| matches!(block, crate::llm::types::LlmContentBlock::Image { .. }))
    }));
}

#[tokio::test]
async fn checkpoint_hook_can_inject_messages() {
    let plugin = Arc::new(RuntimeTestPluginFactory {
        build: Arc::new(|_| {
            Ok(Arc::new(RuntimeTestPlugin {
                before_turn: None,
                checkpoint: Some(Arc::new(|ctx| {
                    Box::pin(async move {
                        if ctx.checkpoint == crate::CheckpointKind::BeforeCompletion {
                            Ok(vec![crate::PluginDirective::EnqueueMessages {
                                messages: vec![crate::PluginMessage::text(
                                    crate::MessageRole::System,
                                    "checkpoint injected",
                                )],
                            }])
                        } else {
                            Ok(Vec::new())
                        }
                    })
                })),
                tool_result_projector: None,
                runtime_event: None,
                external_registrar: None,
            }))
        }),
    });
    let transport = mock_provider(vec![
        MockCall {
            stream_events: Vec::new(),
            response: Ok(LlmResponse {
                full_text: "First answer.".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "First answer.".to_string(),
                    response_meta: None,
                }],
                ..LlmResponse::default()
            }),
        },
        MockCall {
            stream_events: Vec::new(),
            response: Ok(LlmResponse {
                full_text: "Second answer.".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "Second answer.".to_string(),
                    response_meta: None,
                }],
                ..LlmResponse::default()
            }),
        },
    ]);
    let mut runtime = runtime_with_plugins(vec![plugin], transport).await;

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

    assert!(
        active_conversation_messages(&turn.state)
            .iter()
            .any(|message| {
                message.role == MessageRole::System
                    && message
                        .parts
                        .iter()
                        .any(|part| part.content == "checkpoint injected")
            })
    );
}

#[tokio::test]
async fn queued_checkpoint_input_accepts_active_turn_without_persisting_duplicate_user_message() {
    let transport = mock_provider(vec![
        MockCall {
            stream_events: Vec::new(),
            response: Ok(LlmResponse {
                full_text: "first".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "first".to_string(),
                    response_meta: None,
                }],
                ..LlmResponse::default()
            }),
        },
        MockCall {
            stream_events: Vec::new(),
            response: Ok(LlmResponse {
                full_text: "answer".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "answer".to_string(),
                    response_meta: None,
                }],
                ..LlmResponse::default()
            }),
        },
    ]);
    let (mut runtime, store) = standard_runtime_with_transport_and_queue_store(transport).await;
    enqueue_turn_input_for_checkpoint(
        store.as_ref(),
        "root",
        Some("host:follow-up-id".to_string()),
        TurnInput::text("follow up"),
    )
    .await;
    let sink = RecordingSink::default();
    let assembled = runtime
        .stream_turn(
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
            TurnOptions::new(CancellationToken::new()).with_events(&sink),
        )
        .await
        .expect("turn");

    let mut saw_injected_accept = false;
    for event in sink.snapshot() {
        if let crate::SessionEvent::InjectedTurnInputAccepted { inputs, .. } = event {
            saw_injected_accept = inputs.iter().any(|input| {
                input.id.as_deref() == Some("follow-up-id")
                    && input.message.role == crate::MessageRole::User
                    && input.message.content == "follow up"
            });
        }
    }
    assert!(
        saw_injected_accept,
        "expected injected turn input accepted event"
    );

    let projected = active_conversation_messages(&assembled.state);
    let follow_up_count = projected
        .iter()
        .filter(|message| {
            message.role == crate::MessageRole::User
                && message.parts.iter().any(|part| part.content == "follow up")
        })
        .count();
    assert_eq!(
        follow_up_count, 0,
        "injected active-turn input must stay out of persisted history"
    );
    assert!(projected.iter().any(|message| {
        message.role == crate::MessageRole::User
            && message.parts.iter().any(|part| part.content == "hello")
    }));
}

#[tokio::test]
async fn pending_process_wake_drains_into_idle_queued_turn_as_turn_event() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let captured_requests = Arc::clone(&requests);
    let transport = TestProvider::builder()
        .kind("mock")
        .requires_streaming(true)
        .complete(move |req| {
            let captured_requests = Arc::clone(&captured_requests);
            async move {
                captured_requests
                    .lock()
                    .expect("request capture lock")
                    .push(req);
                Ok(LlmResponse {
                    full_text: "saw event".to_string(),
                    parts: vec![LlmOutputPart::Text {
                        text: "saw event".to_string(),
                        response_meta: None,
                    }],
                    ..LlmResponse::default()
                })
            }
        })
        .build();
    let (mut runtime, store) = standard_runtime_with_transport_and_queue_store(transport).await;
    let registry = runtime
        .host
        .process_registry
        .as_ref()
        .expect("process registry")
        .clone();
    let target_scope = crate::ProcessScope::new("root");
    registry
        .register_process(
            crate::ProcessRegistration::new(
                "wake-proc",
                crate::ProcessInput::External {
                    metadata: serde_json::Value::Null,
                },
            )
            .with_extra_event_types(crate::lashlang_process_event_types()),
        )
        .await
        .expect("register wake process");
    let wake = append_process_wake_to_queue(
        registry.as_ref(),
        store.as_ref(),
        "wake-proc",
        crate::ProcessEventAppendRequest::new(
            "process.wake",
            json!({
                "text": "deploy complete",
                "value": {
                    "status": "deploy complete"
                }
            }),
        )
        .with_wake_target_scope(target_scope.clone()),
    )
    .await;

    let turn_events = RecordingTurnEvents::default();
    runtime
        .stream_next_queued_work(
            TurnOptions::new(CancellationToken::new()).with_turn_events(&turn_events),
        )
        .await
        .expect("turn")
        .expect("queued turn");

    let events = turn_events.snapshot();
    let queued_started = events
        .iter()
        .position(|activity| matches!(&activity.event, crate::TurnEvent::QueuedWorkStarted { .. }))
        .expect("queued work started event");
    let model_started = events
        .iter()
        .position(|activity| {
            matches!(
                &activity.event,
                crate::TurnEvent::ModelRequestStarted { .. }
            )
        })
        .expect("model request started event");
    assert!(
        queued_started < model_started,
        "queued work should be announced before model output starts"
    );
    let crate::TurnEvent::QueuedWorkStarted {
        boundary,
        batch_ids,
        causes,
    } = &events[queued_started].event
    else {
        panic!("expected queued work started event");
    };
    assert_eq!(*boundary, crate::QueuedWorkClaimBoundary::Idle);
    assert_eq!(batch_ids.len(), 1);
    assert!(causes.iter().any(|cause| {
        cause.event_type == "process.wake"
            && cause.id == wake.wake_id
            && cause.text.contains("deploy complete")
    }));

    let requests = {
        let guard = requests.lock().expect("request capture lock");
        guard.clone()
    };
    assert_eq!(requests.len(), 1);
    let request = &requests[0];
    let message_text = |message: &crate::llm::types::LlmMessage| {
        message
            .blocks
            .iter()
            .filter_map(|block| match block {
                crate::llm::types::LlmContentBlock::Text { text, .. } => Some(text.as_ref()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    let turn_event_user_messages = request
        .messages
        .iter()
        .filter(|message| {
            message.role == crate::llm::types::LlmRole::User
                && message_text(message).contains("=== TURN EVENTS ===")
        })
        .collect::<Vec<_>>();
    assert_eq!(turn_event_user_messages.len(), 1);
    let turn_event_text = message_text(turn_event_user_messages[0]);
    assert!(turn_event_text.contains("Background process wake"));
    assert!(turn_event_text.contains("deploy complete"));
    assert!(request.messages.iter().all(|message| {
        message.role != crate::llm::types::LlmRole::System
            || !message_text(message).contains("deploy complete")
    }));
    assert!(request.messages.iter().all(|message| {
        message.role != crate::llm::types::LlmRole::User || !message.is_blank()
    }));
    assert!(
        active_conversation_messages(&runtime.state)
            .iter()
            .all(|message| {
                !(message.role == crate::MessageRole::User
                    && message
                        .parts
                        .iter()
                        .all(|part| part.content.trim().is_empty()))
            }),
        "empty wake turns must not synthesize blank user history"
    );
    assert!(
        crate::store::RuntimePersistence::list_queued_work(store.as_ref(), "root")
            .await
            .expect("queued work after commit")
            .is_empty()
    );
}

#[tokio::test]
async fn durable_process_wake_drains_as_committed_event_history_and_acknowledges() {
    let transport = mock_provider(vec![
        MockCall {
            stream_events: Vec::new(),
            response: Ok(LlmResponse {
                full_text: "first answer".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "first answer".to_string(),
                    response_meta: None,
                }],
                ..LlmResponse::default()
            }),
        },
        MockCall {
            stream_events: Vec::new(),
            response: Ok(LlmResponse {
                full_text: "acknowledged".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "acknowledged".to_string(),
                    response_meta: None,
                }],
                ..LlmResponse::default()
            }),
        },
    ]);
    let (mut runtime, store) = standard_runtime_with_transport_and_queue_store(transport).await;
    let registry = runtime
        .host
        .process_registry
        .as_ref()
        .expect("process registry")
        .clone();
    let target_scope = crate::ProcessScope::new("root");
    registry
        .register_process(
            crate::ProcessRegistration::new(
                "wake-proc",
                crate::ProcessInput::External {
                    metadata: serde_json::Value::Null,
                },
            )
            .with_extra_event_types(crate::lashlang_process_event_types()),
        )
        .await
        .expect("register wake process");
    let wake = append_process_wake_to_queue(
        registry.as_ref(),
        store.as_ref(),
        "wake-proc",
        crate::ProcessEventAppendRequest::new(
            "process.wake",
            json!({
                "text": "deploy complete",
                "value": {
                    "status": "deploy complete"
                }
            }),
        )
        .with_wake_target_scope(target_scope.clone()),
    )
    .await;
    let expected_wake_id = wake.wake_id.clone();
    let expected_text = "Background process wake\nProcess: wake-proc\nEvent: process.wake #1\nWake input:\ndeploy complete";

    let sink = RecordingSink::default();
    let turn_events = RecordingTurnEvents::default();
    runtime
        .stream_turn(
            TurnInput::text("hello"),
            TurnOptions::new(CancellationToken::new())
                .with_events(&sink)
                .with_turn_events(&turn_events),
        )
        .await
        .expect("turn");

    let turn_event_snapshot = turn_events.snapshot();
    let queued_started = turn_event_snapshot
        .iter()
        .find(|activity| matches!(&activity.event, crate::TurnEvent::QueuedWorkStarted { .. }))
        .expect("queued work started event");
    let crate::TurnEvent::QueuedWorkStarted {
        boundary, causes, ..
    } = &queued_started.event
    else {
        panic!("expected queued work started event");
    };
    assert_eq!(
        *boundary,
        crate::QueuedWorkClaimBoundary::ActiveTurnCheckpoint
    );
    assert!(causes.iter().any(|cause| {
        cause.event_type == "process.wake"
            && cause.id == expected_wake_id
            && cause.text == expected_text
    }));

    assert!(
        sink.snapshot().into_iter().all(|event| {
            !matches!(
                event,
                crate::SessionEvent::InjectedMessagesCommitted { messages, .. }
                    if messages.iter().any(|message| message.content == expected_text)
            )
        }),
        "durable wake events must not be bridged as injected plugin messages"
    );
    assert!(
        crate::store::RuntimePersistence::list_queued_work(store.as_ref(), "root")
            .await
            .expect("queued work after commit")
            .is_empty()
    );
    let wake_history = active_conversation_messages(&runtime.state)
        .into_iter()
        .find(|message| {
            message.role == crate::MessageRole::Event
                && message
                    .parts
                    .iter()
                    .any(|part| part.content == expected_text)
        })
        .expect("wake history message");
    assert!(matches!(
        wake_history.origin,
        Some(crate::MessageOrigin::Process {
            process_id,
            event_type,
            sequence,
            wake_id,
        }) if process_id == "wake-proc"
            && event_type == "process.wake"
            && sequence == 1
            && wake_id.as_deref() == Some(expected_wake_id.as_str())
    ));
    assert!(
        active_conversation_messages(&runtime.state)
            .iter()
            .all(|message| {
                !((message.role == crate::MessageRole::System
                    || message.role == crate::MessageRole::User)
                    && message
                        .parts
                        .iter()
                        .any(|part| part.content == expected_text))
            }),
        "durable wake must not enter history as provider system text"
    );
}

#[tokio::test]
async fn external_invoke_can_create_session_from_current_snapshot() {
    let plugin = Arc::new(RuntimeTestPluginFactory {
        build: Arc::new(|_| {
            Ok(Arc::new(RuntimeTestPlugin {
                before_turn: None,
                checkpoint: None,
                tool_result_projector: None,
                runtime_event: None,
                external_registrar: Some(Arc::new(|reg| {
                    reg.actions().op(
                        crate::PluginActionDef {
                            name: "test.spawn".to_string(),
                            description: "spawn".to_string(),
                            kind: crate::PluginActionKind::Command,
                            session_param: crate::SessionParam::Optional,
                            input_schema: json!({}),
                            output_schema: json!({}),
                        },
                        Arc::new(|ctx, _args| {
                            Box::pin(async move {
                                let handle = ctx
                                    .host
                                    .create_session(
                                        crate::SessionCreateRequest::root(
                                            crate::SessionStartPoint::CurrentSession,
                                            crate::PluginOptions::default(),
                                        )
                                        .with_session_id("branched")
                                        .with_plugin_source(
                                            crate::SessionPluginSource::CurrentSessionFork,
                                        )
                                        .with_initial_nodes(vec![crate::SessionAppendNode::message(
                                            crate::PluginMessage::text(
                                                crate::MessageRole::User,
                                                "branch seed",
                                            ),
                                        )]),
                                    )
                                    .await
                                    .map_err(|err| crate::ToolResult::err_fmt(err.to_string()));
                                match handle {
                                    Ok(handle) => {
                                        let snapshot = ctx
                                            .host
                                            .snapshot_session(&handle.session_id)
                                            .await
                                            .map_err(|err| {
                                                crate::ToolResult::err_fmt(err.to_string())
                                            });
                                        match snapshot {
                                            Ok(snapshot) => crate::ToolResult::ok(json!({
                                                "session_id": handle.session_id,
                                                "message_count": snapshot.read_model().messages.len(),
                                            })),
                                            Err(err) => err,
                                        }
                                    }
                                    Err(err) => err,
                                }
                            })
                        }),
                    )
                })),
            }))
        }),
    });
    let transport = mock_provider(Vec::new());
    let mut runtime = runtime_with_plugins(vec![plugin], transport).await;

    append_message(
        &mut runtime.state,
        Message {
            id: "m0".to_string(),
            role: MessageRole::User,
            parts: vec![Part {
                id: "m0.p0".to_string(),
                kind: PartKind::Text,
                content: "root msg".to_string(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                tool_replay: None,
                prune_state: PruneState::Intact,
                reasoning_meta: None,
                response_meta: None,
            }]
            .into(),
            origin: None,
        },
    );

    let result = runtime
        .invoke_plugin_action("test.spawn", json!({}), None)
        .await
        .expect("invoke");
    assert!(result.is_success());
    assert_eq!(
        result
            .value_for_projection()
            .get("session_id")
            .and_then(|value| value.as_str()),
        Some("branched")
    );
    assert_eq!(
        result
            .value_for_projection()
            .get("message_count")
            .and_then(|value| value.as_u64()),
        Some(2)
    );
}

#[tokio::test]
async fn session_manager_can_run_child_session_turn() {
    let transport = mock_provider(vec![MockCall {
        stream_events: vec![
            LlmStreamEvent::Delta("child ".to_string()),
            LlmStreamEvent::Part(LlmOutputPart::Text {
                text: "session".to_string(),
                response_meta: None,
            }),
            LlmStreamEvent::Usage(LlmUsage {
                input_tokens: 7,
                output_tokens: 2,
                cached_input_tokens: 0,
                reasoning_tokens: 1,
            }),
        ],
        response: Ok(LlmResponse {
            full_text: "child session".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "child session".to_string(),
                response_meta: None,
            }],
            ..LlmResponse::default()
        }),
    }]);
    let runtime = runtime_with_plugins(Vec::new(), transport).await;
    let manager = runtime.session_manager().expect("session manager");
    let handle = manager
        .create_session(
            crate::SessionCreateRequest::root(
                crate::SessionStartPoint::Empty,
                crate::PluginOptions::default(),
            )
            .with_session_id("child")
            .with_plugin_source(crate::SessionPluginSource::CurrentSessionFork),
        )
        .await
        .expect("child session");
    let assembled = manager
        .start_turn(
            &handle.session_id,
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
        )
        .await
        .expect("child turn");
    assert_eq!(handle.session_id, "child");
    assert_eq!(handle.policy.model.id, "mock-model");
    assert_eq!(assembled.state.session_id, "child");
}

#[tokio::test]
async fn session_manager_persists_child_sessions_in_separate_store() {
    let factory = RecordingSessionStoreFactory::default();
    let host = test_host_config().with_session_store_factory(Arc::new(factory.clone()));
    let mut runtime = runtime_with_plugins_and_tools_and_host(
        Vec::new(),
        Arc::new(EmptyTools),
        mock_provider(Vec::new()),
        host,
    )
    .await;
    append_message(
        &mut runtime.state,
        Message {
            id: "u1".to_string(),
            role: MessageRole::User,
            parts: vec![Part {
                id: "u1.p0".to_string(),
                kind: PartKind::Text,
                content: "parent hello".to_string(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                tool_replay: None,
                prune_state: PruneState::Intact,
                reasoning_meta: None,
                response_meta: None,
            }]
            .into(),
            origin: None,
        },
    );
    runtime.state.turn_index = 3;

    let manager = runtime.session_manager().expect("session manager");
    let handle = manager
        .create_session(
            crate::SessionCreateRequest::child_session(
                "root",
                crate::SessionStartPoint::CurrentSession,
                crate::PluginOptions::default(),
            )
            .with_session_id("child-store")
            .with_plugin_source(crate::SessionPluginSource::CurrentSessionFork),
        )
        .await
        .expect("child session");

    assert_eq!(handle.session_id, "child-store");
    let stores = factory.stores();
    assert_eq!(stores.len(), 1);
    let meta = crate::store::RuntimePersistence::load_session_meta(stores[0].as_ref())
        .await
        .expect("load session meta")
        .expect("session meta");
    assert_eq!(meta.session_id, "child-store");
    assert_eq!(meta.parent_session_id(), Some("root"));
    let read = crate::store::RuntimePersistence::load_session(
        stores[0].as_ref(),
        crate::store::SessionReadScope::FullGraph,
    )
    .await
    .expect("load session")
    .expect("session read");
    let graph = read.graph;
    let read_model = graph.read_model();
    let messages = read_model.messages.as_slice();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].parts[0].content, "parent hello");
    let checkpoint = read.checkpoint.expect("checkpoint");
    let turn_state = checkpoint.turn_state;
    assert_eq!(turn_state.turn_index, 3);
}

#[tokio::test]
async fn child_relation_does_not_replace_active_session() {
    let mut runtime = runtime_with_plugins(Vec::new(), mock_provider(Vec::new())).await;
    let manager = runtime.session_manager().expect("session manager");
    manager
        .create_session(
            crate::SessionCreateRequest::child_session(
                runtime.session_id(),
                crate::SessionStartPoint::Empty,
                crate::PluginOptions::default(),
            )
            .with_session_id("ordinary-child")
            .with_plugin_source(crate::SessionPluginSource::CurrentSessionFork),
        )
        .await
        .expect("child session");

    assert_eq!(runtime.session_id(), "root");
    let assembled = runtime
        .run_turn_assembled(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "parent turn".to_string(),
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
        .expect("parent turn");

    assert_eq!(assembled.state.session_id, "root");
    assert_eq!(assembled.state.turn_index, 1);
}

#[tokio::test]
async fn session_manager_rejects_duplicate_child_session_ids() {
    let runtime = runtime_with_plugins(Vec::new(), mock_provider(Vec::new())).await;
    let manager = runtime.session_manager().expect("session manager");
    manager
        .create_session(
            crate::SessionCreateRequest::root(
                crate::SessionStartPoint::Empty,
                crate::PluginOptions::default(),
            )
            .with_session_id("child")
            .with_plugin_source(crate::SessionPluginSource::CurrentSessionFork),
        )
        .await
        .expect("first child session");

    let err = manager
        .create_session(
            crate::SessionCreateRequest::root(
                crate::SessionStartPoint::Empty,
                crate::PluginOptions::default(),
            )
            .with_session_id("child")
            .with_plugin_source(crate::SessionPluginSource::CurrentSessionFork),
        )
        .await
        .expect_err("duplicate child session should fail");
    assert!(err.to_string().contains("already exists"));
}

#[tokio::test]
async fn runtime_can_activate_managed_child_session() {
    let mut runtime = runtime_with_plugins(Vec::new(), mock_provider(Vec::new())).await;
    let manager = runtime.session_manager().expect("session manager");
    manager
        .create_session(
            crate::SessionCreateRequest::child(
                runtime.session_id(),
                crate::SessionStartPoint::Empty,
                runtime.policy.clone(),
                crate::PluginOptions::default(),
                "test",
            )
            .with_session_id("child")
            .with_plugin_source(crate::SessionPluginSource::CurrentSessionFork),
        )
        .await
        .expect("child session");

    runtime
        .activate_managed_session("child")
        .await
        .expect("activate child");

    assert_eq!(runtime.session_id(), "child");
    assert!(
        manager
            .start_turn(
                "child",
                TurnInput {
                    items: vec![InputItem::Text {
                        text: "old manager should not own activated child".to_string(),
                    }],
                    image_blobs: HashMap::new(),
                    protocol_turn_options: None,
                    trace_turn_id: None,
                    protocol_extension: None,
                    turn_context: crate::TurnContext::default(),
                },
            )
            .await
            .is_err(),
        "activated child runtime should leave the parent manager registry"
    );
}
