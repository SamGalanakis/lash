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

#[tokio::test]
async fn session_operations_delegate_to_runtime() -> Result<()> {
    let core = standard_core();
    let session = core.session("session-ops").open().await?;

    session.run(TurnInput::text("usage")).await?;
    let usage = session.usage_report();
    assert_eq!(usage.usage.output_tokens, 2);
    session
        .control()
        .commands()
        .refresh_tool_surface("control admin test", None, "control-admin-refresh")
        .await?;
    session.process_control().await_all().await?;
    assert!(session.process_control().list().await?.is_empty());
    let err = session
        .control()
        .state()
        .snapshot_execution()
        .await
        .expect_err("standard protocol has no code executor to snapshot");
    assert!(matches!(
        err,
        EmbedError::Session(SessionError::CodeExecutionUnavailable)
    ));
    let err = session
        .control()
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
async fn session_commands_enqueue_idempotently_by_source_key() -> Result<()> {
    let core = explicit_ephemeral_facets(LashCore::standard())
        .provider(mock_provider())
        .model(mock_model_spec())
        .store_factory(Arc::new(lash_core::InMemorySessionStoreFactory::new()))
        .build()?;
    let session = core.session("command-idempotency").open().await?;

    let first = session
        .commands()
        .refresh_tool_surface("test refresh", None, "same-refresh")
        .await?;
    let second = session
        .commands()
        .refresh_tool_surface("test refresh", None, "same-refresh")
        .await?;

    assert_eq!(first.batch_id, second.batch_id);
    assert_eq!(
        first.source_key,
        "command:refresh_tool_surface:same-refresh"
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
async fn emit_host_event_command_appends_node_and_completes_claim() -> Result<()> {
    let host_event = lash_core::HostEvent::new(
        "Button",
        "ui.button",
        "pressed",
        lash_protocol_rlm::NamedDataType::object("ui.button.Pressed", Vec::new())
            .expect("event type"),
    );
    let core = explicit_ephemeral_facets(LashCore::standard())
        .provider(mock_provider())
        .model(mock_model_spec())
        .plugin(Arc::new(StaticPluginFactory::new(
            "button-host-events",
            lash_core::PluginSpec::new().with_host_event(host_event),
        )))
        .store_factory(Arc::new(lash_core::InMemorySessionStoreFactory::new()))
        .build()?;
    let session = core.session("command-host-event").open().await?;

    let receipt = session
        .commands()
        .emit_host_event(
            "Button",
            "ui.button",
            "pressed",
            serde_json::json!({ "pressed": true }),
            "button-press-1",
        )
        .await?;
    assert_eq!(receipt.source_key, "command:emit_host_event:button-press-1");
    assert_eq!(session.queued_work().await?.len(), 1);

    assert!(session.next_queued_turn().run().await?.is_none());
    assert!(session.queued_work().await?.is_empty());
    let persisted = session.control().state().persist_current().await?;
    let host_event_nodes = persisted
        .session_graph
        .nodes
        .iter()
        .filter_map(|node| node.plugin())
        .filter(|(plugin_type, body)| {
            *plugin_type == "lash.host_event" && body["source_type"] == "ui.button.pressed"
        })
        .collect::<Vec<_>>();
    assert_eq!(host_event_nodes.len(), 1);
    Ok(())
}

#[tokio::test]
async fn observation_reads_do_not_wait_for_active_turn() -> Result<()> {
    let (entered_tx, entered_rx) = oneshot::channel();
    let (release_tx, release_rx) = oneshot::channel();
    let core = explicit_ephemeral_facets(LashCore::standard())
        .provider(checkpoint_gated_provider(entered_tx, release_rx))
        .model(mock_model_spec())
        .tools(Arc::new(AppTools))
        .store_factory(Arc::new(lash_core::InMemorySessionStoreFactory::new()))
        .process_registry(Arc::new(TestLocalProcessRegistry::default()))
        .build()?;
    let session = core.session("nonblocking-observation").open().await?;
    let turn_session = session.clone();
    let turn = tokio::spawn(async move { turn_session.run(TurnInput::text("blocked")).await });

    entered_rx.await.expect("provider entered");

    let observed = tokio::time::timeout(std::time::Duration::from_millis(50), async {
        let _ = session.session_id();
        let _ = session.policy_snapshot();
        let _ = session.read_view();
        let _ = session.usage_report();
        let _ = session.control().tools().state().await?;
        let _ = session.control().tools().active_definitions().await?;
        let _ = session.process_control().list().await?;
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
async fn process_control_cancel_uses_host_cancel_ability() -> Result<()> {
    let ability = Arc::new(DenyCancelAbility::default());
    let runtime_host = RuntimeHostConfig::in_memory().with_process_cancel_ability(ability.clone());
    let core = explicit_ephemeral_facets(LashCore::standard())
        .provider(mock_provider())
        .model(mock_model_spec())
        .store_factory(Arc::new(lash_core::InMemorySessionStoreFactory::new()))
        .process_registry(Arc::new(TestLocalProcessRegistry::default()))
        .advanced()
        .runtime_host_config(runtime_host)
        .build()?;
    let session = core.session("host-cancel").open().await?;
    session
        .process_control()
        .start(lash_core::ProcessStartRequest::external(
            "host-process",
            lash_core::ProcessHandleDescriptor::new(Some("test"), Some("host process")),
            serde_json::Value::Null,
        ))
        .await?;

    let err = session
        .process_control()
        .cancel("host-process")
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
async fn process_control_cancel_all_uses_host_cancel_ability() -> Result<()> {
    let ability = Arc::new(RecordingCancelAbility::default());
    let runtime_host = RuntimeHostConfig::in_memory().with_process_cancel_ability(ability.clone());
    let core = explicit_ephemeral_facets(LashCore::standard())
        .provider(mock_provider())
        .model(mock_model_spec())
        .store_factory(Arc::new(lash_core::InMemorySessionStoreFactory::new()))
        .process_registry(Arc::new(TestLocalProcessRegistry::default()))
        .advanced()
        .runtime_host_config(runtime_host)
        .build()?;
    let session = core.session("host-cancel-all").open().await?;
    for process_id in ["host-process-a", "host-process-b"] {
        session
            .process_control()
            .start(lash_core::ProcessStartRequest::external(
                process_id,
                lash_core::ProcessHandleDescriptor::new(Some("test"), Some(process_id)),
                serde_json::Value::Null,
            ))
            .await?;
    }

    let mut summaries = session.process_control().cancel_all().await?;
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
    session.run(TurnInput::text("hello observation")).await?;

    let observed = session.observe();
    assert_eq!(observed.read_view().messages().len(), 2);
    assert_eq!(observed.usage_report().usage.output_tokens, 2);
    assert_eq!(observed.policy_snapshot().model.id, "mock-model");
    Ok(())
}

#[tokio::test]
async fn config_and_tool_mutations_publish_observation_immediately() -> Result<()> {
    let core = explicit_ephemeral_facets(LashCore::standard())
        .provider(mock_provider())
        .model(mock_model_spec())
        .tools(Arc::new(AppTools))
        .build()?;
    let session = core.session("observation-mutations").open().await?;

    session
        .control()
        .config()
        .set_prompt_template(PromptTemplate::new(vec![
            lash_core::PromptTemplateSection::untitled(vec![lash_core::PromptTemplateEntry::text(
                "updated",
            )]),
        ]))
        .await?;
    assert!(session.policy_snapshot().prompt.template.is_some());

    session
        .control()
        .tools()
        .set_availability("app_lookup", ToolAvailability::Off)
        .await?;
    let tool_state = session
        .observe()
        .tool_state()
        .expect("tool state should be observable");
    assert_eq!(
        tool_state
            .get("app_lookup")
            .and_then(|spec| spec.manifest().availability_override),
        Some(ToolAvailability::Off)
    );
    Ok(())
}

#[tokio::test]
async fn child_session_snapshot_does_not_wait_for_child_turn() -> Result<()> {
    let (entered_tx, entered_rx) = oneshot::channel();
    let (release_tx, release_rx) = oneshot::channel();
    let core = explicit_ephemeral_facets(LashCore::standard())
        .provider(checkpoint_gated_provider(entered_tx, release_rx))
        .model(mock_model_spec())
        .store_factory(Arc::new(lash_core::InMemorySessionStoreFactory::new()))
        .process_registry(Arc::new(TestLocalProcessRegistry::default()))
        .build()?;
    let session = core.session("child-observation-parent").open().await?;
    let children = session.control().children();
    children
        .create_session(SessionCreateRequest {
            session_id: Some("child-observation".to_string()),
            relation: lash_core::SessionRelation::Child {
                parent_session_id: "child-observation-parent".to_string(),
                caused_by: None,
            },
            start: lash_core::SessionStartPoint::Empty,
            policy: None,
            plugin_source: lash_core::SessionPluginSource::CurrentSessionFork,
            initial_nodes: Vec::new(),
            tool_access: lash_core::SessionToolAccess::default(),
            subagent: None,
            context_surface: lash_core::SessionContextSurface::default(),
            plugin_options: lash_core::PluginOptions::default(),
            usage_source: None,
        })
        .await?;
    let child_runner = {
        let children = children.clone();
        tokio::spawn(async move {
            children
                .start_turn(
                    "child-observation",
                    "child-observation-turn",
                    TurnInput::text("blocked child"),
                )
                .await
        })
    };

    entered_rx.await.expect("child provider entered");
    let host = session.control().state().session_state_service().await?;
    let snapshot = tokio::time::timeout(std::time::Duration::from_millis(50), async {
        host.snapshot_session("child-observation").await
    })
    .await
    .expect("child snapshot should not wait for the child turn")?;
    assert_eq!(snapshot.session_id, "child-observation");

    release_tx.send(()).expect("release child provider");
    child_runner.await.expect("child turn task")?;
    Ok(())
}

#[tokio::test]
async fn session_control_manages_child_session_turns() -> Result<()> {
    let core = standard_core();
    let session = core.session("parent-control").open().await?;
    let children = session.control().children();
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
            context_surface: lash_core::SessionContextSurface::default(),
            plugin_options: lash_core::PluginOptions::default(),
            usage_source: None,
        })
        .await?;

    let assembled = children
        .start_turn(
            &child.session_id,
            "child-control-turn",
            TurnInput::text("child"),
        )
        .await?;
    assert_eq!(assembled.state.session_id, "child-control");
    children.close_session(&child.session_id).await?;
    Ok(())
}
