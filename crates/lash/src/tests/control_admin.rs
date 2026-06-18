use super::*;

#[derive(Default)]
struct DenyCancelAbility {
    calls: StdMutex<Vec<(lash_core::ProcessCancelSource, String)>>,
}

impl DenyCancelAbility {
    fn calls(&self) -> Vec<(lash_core::ProcessCancelSource, String)> {
        self.calls.lock().expect("cancel calls").clone()
    }
}

#[async_trait]
impl lash_core::ProcessCancelAbility for DenyCancelAbility {
    async fn cancel(
        &self,
        _processes: &dyn lash_core::ProcessService,
        request: lash_core::ProcessCancelRequest<'_>,
    ) -> std::result::Result<lash_core::ProcessRecord, lash_core::PluginError> {
        self.calls
            .lock()
            .expect("cancel calls")
            .push((request.source, request.process_id.to_string()));
        Err(lash_core::PluginError::Session(
            "denied by host".to_string(),
        ))
    }
}

struct NoopProcessRunHandle;

#[async_trait]
impl lash_core::ProcessRunHandle for NoopProcessRunHandle {
    async fn claim_and_run_pending(&self) -> std::result::Result<(), lash_core::PluginError> {
        Ok(())
    }
}

#[derive(Default)]
struct RecordingCancelAbility {
    calls: StdMutex<Vec<(lash_core::ProcessCancelSource, String, Option<String>)>>,
}

impl RecordingCancelAbility {
    fn calls(&self) -> Vec<(lash_core::ProcessCancelSource, String, Option<String>)> {
        self.calls.lock().expect("cancel calls").clone()
    }
}

#[async_trait]
impl lash_core::ProcessCancelAbility for RecordingCancelAbility {
    async fn cancel(
        &self,
        processes: &dyn lash_core::ProcessService,
        request: lash_core::ProcessCancelRequest<'_>,
    ) -> std::result::Result<lash_core::ProcessRecord, lash_core::PluginError> {
        self.calls.lock().expect("cancel calls").push((
            request.source,
            request.process_id.to_string(),
            request.reason.clone(),
        ));
        lash_core::DefaultProcessCancelAbility
            .cancel(processes, request)
            .await
    }
}

struct FixedCompactor;

#[async_trait]
impl lash_core::ContextCompactor for FixedCompactor {
    fn id(&self) -> &'static str {
        "test.fixed_compactor"
    }

    async fn compact(
        &self,
        ctx: &lash_core::CompactionContext<'_>,
    ) -> std::result::Result<Option<lash_core::ContextCompaction>, lash_core::ContextError> {
        assert_eq!(
            ctx.instructions.as_deref(),
            Some("focus on durable summary")
        );
        assert!(
            ctx.state
                .messages()
                .iter()
                .any(|message| message.parts[0].content.contains("old durable request"))
        );
        Ok(Some(lash_core::ContextCompaction::new(vec![
            lash_core::SessionAppendNode::message(
                lash_core::PluginMessage::text(
                    lash_core::MessageRole::Assistant,
                    "Compaction summary:\nold durable request summarized",
                )
                .with_origin(lash_core::MessageOrigin::Plugin {
                    plugin_id: "test_compactor".to_string(),
                    transient: false,
                }),
            ),
        ])))
    }
}

#[tokio::test]
async fn session_operations_delegate_to_runtime() -> Result<()> {
    let core = standard_core();
    let session = core.session("session-ops").open().await?;

    session.turn(TurnInput::text("usage")).run().await?;
    let usage = session.usage_report();
    assert_eq!(usage.usage.output_tokens, 2);
    session
        .admin()
        .commands()
        .refresh_tool_catalog("control admin test", "control-admin-refresh")
        .await?;
    session.processes().await_all().await?;
    assert!(session.processes().list().await?.is_empty());
    let err = session
        .admin()
        .state()
        .snapshot_execution()
        .await
        .expect_err("standard protocol has no code executor to snapshot");
    assert!(matches!(
        err,
        EmbedError::Session(SessionError::CodeExecutionUnavailable)
    ));
    let err = session
        .admin()
        .state()
        .restore_execution(&[1, 2, 3])
        .await
        .expect_err("standard protocol has no code executor to restore");
    assert!(matches!(
        err,
        EmbedError::Session(SessionError::CodeExecutionUnavailable)
    ));
    Ok(())
}

#[tokio::test]
async fn compact_context_opens_compaction_frame_and_preserves_prior_frame() -> Result<()> {
    let core = explicit_ephemeral_facets(StandardCore::builder())
        .provider(mock_provider())
        .model(mock_model_spec())
        .plugin(Arc::new(StaticPluginFactory::new(
            "test-compactor",
            lash_core::PluginSpec::new().with_context_compactor(100, Arc::new(FixedCompactor)),
        )))
        .build()?;
    let session = core.session("compact-context").open().await?;
    session
        .turn(TurnInput::text("old durable request"))
        .run()
        .await?;
    let before = session.admin().state().persist_current().await?;
    let previous_frame_id = before.current_agent_frame_id.clone();
    let observation_cursor = session.observe().current_observation().cursor;
    assert!(
        before.session_graph.nodes.iter().any(|node| {
            node.agent_frame_id.as_deref() == Some(previous_frame_id.as_str())
                && node
                    .message()
                    .is_some_and(|message| message.parts[0].content.contains("old durable request"))
        }),
        "initial frame should contain the original request"
    );

    let compacted = session
        .admin()
        .state()
        .compact_context(
            Some("focus on durable summary".to_string()),
            runtime_operation_scope("compact-context-test"),
        )
        .await?;

    assert!(compacted);
    let read_view = session.read_view();
    assert_eq!(read_view.messages().len(), 1);
    assert_eq!(
        read_view.messages()[0].parts[0].content,
        "Compaction summary:\nold durable request summarized"
    );
    assert!(matches!(
        read_view.messages()[0].origin.as_ref(),
        Some(lash_core::MessageOrigin::Plugin { plugin_id, .. }) if plugin_id == "test_compactor"
    ));
    let SessionResume::Replayed { events } =
        session.observe().resume_from_cursor(&observation_cursor)?
    else {
        panic!("recent cursor should replay compaction observation events");
    };
    assert!(
        events.windows(2).any(|window| matches!(
            (&window[0].payload, &window[1].payload),
            (
                lash_core::SessionObservationEventPayload::AgentFrameSwitched { .. },
                lash_core::SessionObservationEventPayload::Committed { .. }
            )
        )),
        "expected AgentFrameSwitched immediately followed by Committed, got {events:?}"
    );

    let after = session.admin().state().persist_current().await?;
    let current = after
        .agent_frames
        .iter()
        .find(|frame| frame.frame_id == after.current_agent_frame_id)
        .expect("current frame");
    assert_eq!(
        current.reason.as_str(),
        lash_core::AgentFrameReason::COMPACTION
    );
    assert_eq!(
        current.previous_frame_id.as_deref(),
        Some(previous_frame_id.as_str())
    );
    assert_eq!(
        current.assignment.policy.provider_id,
        before.agent_frames[0].assignment.policy.provider_id
    );
    assert_eq!(
        current.protocol_turn_options.payload,
        before.agent_frames[0].protocol_turn_options.payload
    );
    assert!(
        after.session_graph.nodes.iter().any(|node| {
            node.agent_frame_id.as_deref() == Some(previous_frame_id.as_str())
                && node
                    .message()
                    .is_some_and(|message| message.parts[0].content.contains("old durable request"))
        }),
        "previous frame content should remain durable after compaction"
    );
    assert!(
        after.session_graph.nodes.iter().any(|node| {
            node.agent_frame_id.as_deref() == Some(after.current_agent_frame_id.as_str())
                && node.message().is_some_and(|message| {
                    message.parts[0]
                        .content
                        .contains("old durable request summarized")
                })
        }),
        "compaction summary should be scoped to the new frame"
    );
    Ok(())
}

#[tokio::test]
async fn session_commands_enqueue_idempotently_by_source_key() -> Result<()> {
    let core = explicit_ephemeral_facets(StandardCore::builder())
        .provider(mock_provider())
        .model(mock_model_spec())
        .store_factory(Arc::new(lash_core::InMemorySessionStoreFactory::new()))
        .build()?;
    let session = core.session("command-idempotency").open().await?;

    let first = session
        .commands()
        .refresh_tool_catalog("test refresh", "same-refresh")
        .await?;
    let second = session
        .commands()
        .refresh_tool_catalog("test refresh", "same-refresh")
        .await?;

    assert_eq!(first.batch_id, second.batch_id);
    assert_eq!(
        first.source_key,
        "command:refresh_tool_catalog:same-refresh"
    );
    let queued = session.queued_work().await?;
    assert_eq!(queued.len(), 1);
    assert!(matches!(
        &queued[0].items[0].payload,
        lash_core::runtime::QueuedWorkPayload::SessionCommand { .. }
    ));
    Ok(())
}

#[tokio::test]
async fn queue_enqueue_and_cancel_emit_typed_observation_events() -> Result<()> {
    let core = explicit_ephemeral_facets(StandardCore::builder())
        .provider(mock_provider())
        .model(mock_model_spec())
        .store_factory(Arc::new(lash_core::InMemorySessionStoreFactory::new()))
        .build()?;
    let session = core.session("queue-observation-events").open().await?;
    let cursor = session.observe().current_observation().cursor;

    session
        .enqueue(TurnInput::text("queued observation"))
        .id("queue-observation")
        .send()
        .await?;
    let queued = session.queued_work().await?;
    let batch_id = queued.first().expect("queued batch").batch_id.clone();
    session.cancel_queued_work_batch(&batch_id).await?;

    let SessionResume::Replayed { events } = session.observe().resume_from_cursor(&cursor)? else {
        panic!("recent cursor should replay queue observation events");
    };
    assert!(events.iter().any(|event| matches!(
        &event.payload,
        lash_core::SessionObservationEventPayload::QueueChanged { kind, batch_ids }
            if *kind == lash_core::SessionQueueEventKind::Enqueued
                && batch_ids.as_slice() == std::slice::from_ref(&batch_id)
    )));
    assert!(events.iter().any(|event| matches!(
        &event.payload,
        lash_core::SessionObservationEventPayload::QueueChanged { kind, batch_ids }
            if *kind == lash_core::SessionQueueEventKind::Cancelled
                && batch_ids.as_slice() == std::slice::from_ref(&batch_id)
    )));
    Ok(())
}

#[tokio::test]
async fn process_start_and_cancel_emit_typed_observation_events() -> Result<()> {
    let core = explicit_ephemeral_facets(StandardCore::builder())
        .provider(mock_provider())
        .model(mock_model_spec())
        .store_factory(Arc::new(lash_core::InMemorySessionStoreFactory::new()))
        .process_registry(Arc::new(TestLocalProcessRegistry::default()))
        .build()?;
    let session = core.session("process-observation-events").open().await?;
    let cursor = session.observe().current_observation().cursor;
    let process_id = "observed-process";

    session
        .processes()
        .start(
            lash_core::ProcessStartRequest::external(
                process_id,
                lash_core::ProcessOriginator::host(),
                serde_json::Value::Null,
            )
            .with_grant(Some(lash_core::ProcessStartGrant {
                session_scope: lash_core::SessionScope::new("request-descriptor"),
                descriptor: lash_core::ProcessHandleDescriptor::new(
                    Some("test"),
                    Some("observed process"),
                ),
            })),
            inline_scope(lash_core::ExecutionScope::process(process_id)),
        )
        .await?;
    session
        .processes()
        .cancel(
            process_id,
            inline_scope(lash_core::ExecutionScope::process(process_id)),
        )
        .await?;

    let SessionResume::Replayed { events } = session.observe().resume_from_cursor(&cursor)? else {
        panic!("recent cursor should replay process observation events");
    };
    assert!(events.iter().any(|event| matches!(
        &event.payload,
        lash_core::SessionObservationEventPayload::ProcessChanged { kind, process_ids }
            if *kind == SessionProcessEventKind::Started
                && process_ids.len() == 1
                && process_ids[0] == process_id
    )));
    assert!(events.iter().any(|event| matches!(
        &event.payload,
        lash_core::SessionObservationEventPayload::ProcessChanged { kind, process_ids }
            if *kind == SessionProcessEventKind::Cancelled
                && process_ids.len() == 1
                && process_ids[0] == process_id
    )));
    Ok(())
}

#[tokio::test]
async fn trigger_emit_does_not_append_session_node_or_queue_work() -> Result<()> {
    let trigger = lash_core::TriggerEvent::new(
        "Button",
        "ui.button",
        "pressed",
        lash_core::LashSchema::any(),
    );
    let core = explicit_ephemeral_facets(StandardCore::builder())
        .provider(mock_provider())
        .model(mock_model_spec())
        .plugin(Arc::new(StaticPluginFactory::new(
            "button-triggers",
            lash_core::PluginSpec::new().with_trigger_event(trigger),
        )))
        .store_factory(Arc::new(lash_core::InMemorySessionStoreFactory::new()))
        .build()?;
    let session = core.session("command-trigger").open().await?;
    let before = session.admin().state().persist_current().await?;

    let source_key = lash_core::empty_trigger_source_key("ui.button.pressed")?;
    let scoped_effect_controller = lash_core::ScopedEffectController::shared(
        Arc::new(lash_core::InlineRuntimeEffectController),
        lash_core::ExecutionScope::runtime_operation("trigger:button-press-1"),
    )?;
    let report = core
        .triggers()
        .emit(
            lash_core::TriggerOccurrenceRequest::new(
                "ui.button.pressed",
                source_key,
                serde_json::json!({ "pressed": true }),
                "button-press-1",
            )
            .with_source(serde_json::json!({})),
            scoped_effect_controller,
        )
        .await?;
    assert!(!report.occurrence_id.is_empty());
    assert!(report.started_process_ids.is_empty());

    assert!(session.queued_work().await?.is_empty());
    let persisted = session.admin().state().persist_current().await?;
    assert_eq!(
        persisted.session_graph.leaf_node_id,
        before.session_graph.leaf_node_id
    );
    let trigger_nodes = persisted
        .session_graph
        .nodes
        .iter()
        .filter_map(|node| node.plugin())
        .filter(|(plugin_type, body)| {
            *plugin_type == "lash.trigger" && body["source_type"] == "ui.button.pressed"
        })
        .collect::<Vec<_>>();
    assert!(trigger_nodes.is_empty());
    Ok(())
}

#[tokio::test]
async fn observation_reads_do_not_wait_for_active_turn() -> Result<()> {
    let (entered_tx, entered_rx) = oneshot::channel();
    let (release_tx, release_rx) = oneshot::channel();
    let core = explicit_ephemeral_facets(StandardCore::builder())
        .provider(checkpoint_gated_provider(entered_tx, release_rx))
        .model(mock_model_spec())
        .tools(Arc::new(AppTools))
        .store_factory(Arc::new(lash_core::InMemorySessionStoreFactory::new()))
        .process_registry(Arc::new(TestLocalProcessRegistry::default()))
        .build()?;
    let session = core.session("nonblocking-observation").open().await?;
    let turn_session = session.clone();
    let scoped_effect_controller = turn_scope(&turn_session.session_id());
    let turn = tokio::spawn(async move {
        turn_session
            .turn(TurnInput::text("blocked"))
            .advanced()
            .run_with_scope(scoped_effect_controller)
            .await
    });

    entered_rx.await.expect("provider entered");

    let observed = tokio::time::timeout(std::time::Duration::from_millis(50), async {
        let _ = session.session_id();
        let _ = session.policy_snapshot();
        let _ = session.read_view();
        let _ = session.usage_report();
        let _ = session.admin().tools().state().await?;
        let _ = session.admin().tools().active_manifests().await?;
        let _ = session.processes().list().await?;
        Result::<()>::Ok(())
    })
    .await
    .expect("observation reads should not wait for the turn");
    observed?;

    release_tx.send(()).expect("release provider");
    turn.await.expect("turn task")?;
    Ok(())
}

#[tokio::test]
async fn processes_cancel_uses_host_cancel_ability() -> Result<()> {
    let ability = Arc::new(DenyCancelAbility::default());
    let runtime_host = RuntimeHostConfig::in_memory().with_process_cancel_ability(ability.clone());
    let core = explicit_ephemeral_facets(StandardCore::builder())
        .provider(mock_provider())
        .model(mock_model_spec())
        .store_factory(Arc::new(lash_core::InMemorySessionStoreFactory::new()))
        .process_registry(Arc::new(TestLocalProcessRegistry::default()))
        .runtime_host_config(runtime_host)
        .build()?;
    let session = core.session("host-cancel").open().await?;
    session
        .processes()
        .start(
            lash_core::ProcessStartRequest::external(
                "host-process",
                lash_core::ProcessOriginator::host(),
                serde_json::Value::Null,
            )
            .with_grant(Some(lash_core::ProcessStartGrant {
                session_scope: lash_core::SessionScope::new("request-descriptor"),
                descriptor: lash_core::ProcessHandleDescriptor::new(
                    Some("test"),
                    Some("host process"),
                ),
            })),
            inline_scope(lash_core::ExecutionScope::process("host-process")),
        )
        .await?;

    let err = session
        .processes()
        .cancel(
            "host-process",
            inline_scope(lash_core::ExecutionScope::process("host-process")),
        )
        .await
        .expect_err("host ability should deny cancellation");

    assert!(
        err.to_string().contains("denied by host"),
        "unexpected error: {err}"
    );
    assert_eq!(
        ability.calls(),
        vec![(
            lash_core::ProcessCancelSource::HostApi,
            "host-process".to_string()
        )]
    );
    Ok(())
}

#[tokio::test]
async fn processes_cancel_all_uses_host_cancel_ability() -> Result<()> {
    let ability = Arc::new(RecordingCancelAbility::default());
    let runtime_host = RuntimeHostConfig::in_memory().with_process_cancel_ability(ability.clone());
    let registry =
        Arc::new(TestLocalProcessRegistry::default()) as Arc<dyn lash_core::ProcessRegistry>;
    let runner = lash_core::ProcessWorkRunner::new(Arc::new(NoopProcessRunHandle));
    let driver = lash_core::ProcessWorkDriver::new(Arc::clone(&registry), runner.poke_handle());
    let core = explicit_ephemeral_facets(StandardCore::builder())
        .provider(mock_provider())
        .model(mock_model_spec())
        .store_factory(Arc::new(lash_core::InMemorySessionStoreFactory::new()))
        .process_work_driver(driver)
        .runtime_host_config(runtime_host)
        .build()?;
    let session = core.session("host-cancel-all").open().await?;
    for process_id in ["host-process-a", "host-process-b"] {
        session
            .processes()
            .start(
                lash_core::ProcessStartRequest::external(
                    process_id,
                    lash_core::ProcessOriginator::host(),
                    serde_json::Value::Null,
                )
                .with_grant(Some(lash_core::ProcessStartGrant {
                    session_scope: lash_core::SessionScope::new("request-descriptor"),
                    descriptor: lash_core::ProcessHandleDescriptor::new(
                        Some("test"),
                        Some(process_id),
                    ),
                })),
                inline_scope(lash_core::ExecutionScope::process(process_id)),
            )
            .await?;
    }

    let mut summaries = session
        .processes()
        .cancel_all(runtime_operation_scope("host-cancel-all"))
        .await?;
    summaries.sort_by(|left, right| left.process_id.cmp(&right.process_id));
    let mut calls = ability.calls();
    calls.sort_by(|left, right| left.1.cmp(&right.1));

    assert_eq!(
        summaries
            .iter()
            .map(|summary| summary.process_id.as_str())
            .collect::<Vec<_>>(),
        vec!["host-process-a", "host-process-b"]
    );
    assert_eq!(
        calls,
        vec![
            (
                lash_core::ProcessCancelSource::HostApi,
                "host-process-a".to_string(),
                Some("requested by host API".to_string())
            ),
            (
                lash_core::ProcessCancelSource::HostApi,
                "host-process-b".to_string(),
                Some("requested by host API".to_string())
            )
        ]
    );
    Ok(())
}

#[tokio::test]
async fn observation_updates_after_completed_turn() -> Result<()> {
    let core = standard_core();
    let session = core.session("observation-after-turn").open().await?;

    assert!(session.read_view().messages().is_empty());
    session
        .turn(TurnInput::text("hello observation"))
        .run()
        .await?;

    let observed = session.observe();
    assert_eq!(observed.read_view().messages().len(), 2);
    assert_eq!(observed.usage_report().usage.output_tokens, 2);
    assert_eq!(observed.policy_snapshot().model.id, "mock-model");
    Ok(())
}

#[tokio::test]
async fn config_and_tool_mutations_publish_observation_immediately() -> Result<()> {
    let core = explicit_ephemeral_facets(StandardCore::builder())
        .provider(mock_provider())
        .model(mock_model_spec())
        .tools(Arc::new(AppTools))
        .build()?;
    let session = core.session("observation-mutations").open().await?;

    session
        .admin()
        .config()
        .set_prompt_template(PromptTemplate::new(vec![
            lash_core::PromptTemplateSection::untitled(vec![lash_core::PromptTemplateEntry::text(
                "updated",
            )]),
        ]))
        .await?;
    assert!(session.policy_snapshot().prompt.template.is_some());

    session
        .admin()
        .tools()
        .set_availability("tool:app_lookup", ToolAvailability::Off)
        .await?;
    let tool_state = session
        .observe()
        .tool_state()
        .expect("tool state should be observable");
    assert_eq!(
        tool_state
            .get(&lash_core::ToolId::from("tool:app_lookup"))
            .and_then(|spec| spec.manifest().availability_override),
        Some(ToolAvailability::Off)
    );
    Ok(())
}

#[tokio::test]
async fn session_control_manages_child_session_lifecycle() -> Result<()> {
    let core = standard_core();
    let session = core.session("parent-control").open().await?;
    let children = session.admin().children();
    let child = children
        .create_session(SessionCreateRequest {
            session_id: Some("child-control".to_string()),
            relation: lash_core::SessionRelation::Child {
                parent_session_id: "parent-control".to_string(),
                caused_by: None,
            },
            start: lash_core::SessionStartPoint::Empty,
            policy: None,
            plugin_source: lash_core::SessionPluginSource::CurrentSessionFork,
            initial_nodes: Vec::new(),
            tool_access: lash_core::SessionToolAccess::default(),
            subagent: None,
            context_overlay: lash_core::SessionContextOverlay::default(),
            plugin_options: lash_core::PluginOptions::default(),
            usage_source: None,
        })
        .await?;

    assert_eq!(child.session_id, "child-control");
    children.close_session(&child.session_id).await?;
    Ok(())
}
