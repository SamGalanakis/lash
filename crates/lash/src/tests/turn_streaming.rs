use super::*;
use crate::modes::RlmTurnBuilderExt as _;

#[derive(Clone, Debug)]
struct DurableEffectInvocation {
    kind: lash_core::RuntimeEffectKind,
    turn_id: Option<String>,
    replay_key: Option<String>,
}

#[derive(Default)]
struct RecordingDurableEffectController {
    invocations: StdMutex<Vec<DurableEffectInvocation>>,
}

impl RecordingDurableEffectController {
    fn invocations(&self) -> Vec<DurableEffectInvocation> {
        self.invocations
            .lock()
            .expect("durable effect invocations")
            .clone()
    }
}

#[async_trait]
impl lash_core::RuntimeEffectController for RecordingDurableEffectController {
    fn durability_tier(&self) -> DurabilityTier {
        DurabilityTier::Durable
    }

    fn requires_durable_attachment_store(&self) -> bool {
        true
    }

    async fn execute_effect(
        &self,
        envelope: lash_core::RuntimeEffectEnvelope,
        local_executor: lash_core::RuntimeEffectLocalExecutor<'_>,
    ) -> std::result::Result<lash_core::RuntimeEffectOutcome, lash_core::RuntimeEffectControllerError>
    {
        self.invocations
            .lock()
            .expect("durable effect invocations")
            .push(DurableEffectInvocation {
                kind: envelope.invocation.effect_kind().expect("effect kind"),
                turn_id: envelope.invocation.scope.turn_id.clone(),
                replay_key: envelope.invocation.replay_key().map(ToOwned::to_owned),
            });
        local_executor.execute(envelope).await
    }
}

#[derive(Default)]
struct RecordingInlineEffectController {
    invocations: StdMutex<Vec<DurableEffectInvocation>>,
}

impl RecordingInlineEffectController {
    fn invocations(&self) -> Vec<DurableEffectInvocation> {
        self.invocations
            .lock()
            .expect("inline effect invocations")
            .clone()
    }
}

#[async_trait]
impl lash_core::RuntimeEffectController for RecordingInlineEffectController {
    async fn execute_effect(
        &self,
        envelope: lash_core::RuntimeEffectEnvelope,
        local_executor: lash_core::RuntimeEffectLocalExecutor<'_>,
    ) -> std::result::Result<lash_core::RuntimeEffectOutcome, lash_core::RuntimeEffectControllerError>
    {
        self.invocations
            .lock()
            .expect("inline effect invocations")
            .push(DurableEffectInvocation {
                kind: envelope.invocation.effect_kind().expect("effect kind"),
                turn_id: envelope.invocation.scope.turn_id.clone(),
                replay_key: envelope.invocation.replay_key().map(ToOwned::to_owned),
            });
        local_executor.execute(envelope).await
    }
}

#[derive(Default)]
struct DurableNoopEffectHost;

impl lash_core::EffectHost for DurableNoopEffectHost {
    fn durability_tier(&self) -> DurabilityTier {
        DurabilityTier::Durable
    }

    fn scoped<'run>(
        &'run self,
        scope: lash_core::EffectScope,
    ) -> std::result::Result<lash_core::ScopedEffectController<'run>, lash_core::RuntimeError> {
        lash_core::ScopedEffectController::shared(
            Arc::new(lash_core::InlineRuntimeEffectController),
            scope,
        )
    }

    fn scoped_static(
        &self,
        scope: lash_core::EffectScope,
    ) -> std::result::Result<
        Option<lash_core::ScopedEffectController<'static>>,
        lash_core::RuntimeError,
    > {
        Ok(Some(lash_core::ScopedEffectController::shared(
            Arc::new(lash_core::InlineRuntimeEffectController),
            scope,
        )?))
    }
}

struct BlockingAppTools {
    entered_tx: StdMutex<Option<oneshot::Sender<()>>>,
    release_rx: TokioMutex<Option<oneshot::Receiver<()>>>,
}

impl BlockingAppTools {
    fn new(entered_tx: oneshot::Sender<()>, release_rx: oneshot::Receiver<()>) -> Self {
        Self {
            entered_tx: StdMutex::new(Some(entered_tx)),
            release_rx: TokioMutex::new(Some(release_rx)),
        }
    }
}

#[async_trait]
impl ToolProvider for BlockingAppTools {
    fn tool_manifests(&self) -> Vec<lash_core::ToolManifest> {
        vec![app_tool_definition().manifest()]
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<lash_core::ToolContract>> {
        (name == "app_lookup").then(|| Arc::new(app_tool_definition().contract()))
    }

    async fn execute(&self, call: lash_core::ToolCall<'_>) -> lash_core::ToolResult {
        assert_eq!(call.name, "app_lookup");
        if let Some(tx) = self.entered_tx.lock().expect("entered tx").take() {
            let _ = tx.send(());
        }
        if let Some(rx) = self.release_rx.lock().await.take() {
            let _ = rx.await;
        }
        lash_core::ToolResult::ok(serde_json::json!({ "answer": "ready" }))
    }
}

#[tokio::test]
async fn turn_run_uses_configured_inline_effect_host_without_explicit_effects() -> Result<()> {
    let recorder = Arc::new(RecordingInlineEffectController::default());
    let effect_controller: Arc<dyn lash_core::RuntimeEffectController> = recorder.clone();
    let core = explicit_ephemeral_facets(LashCore::standard())
        .effect_host(Arc::new(lash_core::InlineEffectHost::new(
            effect_controller,
        )))
        .provider(mock_provider())
        .model(mock_model_spec())
        .build()?;
    let session = core.session("inline-default-effect-host").open().await?;

    let output = session.turn(TurnInput::text("inline")).run().await?;

    assert_eq!(output.assistant_message(), Some("echo: inline"));
    let invocations = recorder.invocations();
    assert!(
        invocations
            .iter()
            .any(|record| record.kind == lash_core::RuntimeEffectKind::LlmCall)
    );
    assert!(invocations.iter().all(|record| {
        record
            .turn_id
            .as_deref()
            .is_some_and(|turn_id| !turn_id.trim().is_empty())
    }));
    Ok(())
}

#[tokio::test]
async fn durable_configured_effect_host_requires_explicit_handler_effects() -> Result<()> {
    let core = explicit_ephemeral_facets(LashCore::standard())
        .effect_host(Arc::new(DurableNoopEffectHost))
        .provider(mock_provider())
        .model(mock_model_spec())
        .build()?;
    let session = core.session("durable-default-effect-host").open().await?;

    let err = session
        .turn(TurnInput::text("should fail before provider"))
        .run()
        .await
        .expect_err("durable deployment host should require handler context");

    assert!(matches!(
        err,
        EmbedError::DurableEffectHostRequiresHandlerContext { operation: "turn" }
    ));
    Ok(())
}

#[tokio::test]
async fn turn_id_sets_effect_scope_and_trace_identity() -> Result<()> {
    let recorder = Arc::new(RecordingInlineEffectController::default());
    let effect_controller: Arc<dyn lash_core::RuntimeEffectController> = recorder.clone();
    let core = explicit_ephemeral_facets(LashCore::standard())
        .effect_host(Arc::new(lash_core::InlineEffectHost::new(
            effect_controller,
        )))
        .provider(mock_provider())
        .model(mock_model_spec())
        .build()?;
    let session = core.session("stable-turn-id").open().await?;

    session
        .turn(TurnInput::text("stable"))
        .turn_id("stable-turn")
        .run()
        .await?;

    let llm_invocation = recorder
        .invocations()
        .into_iter()
        .find(|record| record.kind == lash_core::RuntimeEffectKind::LlmCall)
        .expect("llm effect");
    assert_eq!(llm_invocation.turn_id.as_deref(), Some("stable-turn"));
    assert!(
        llm_invocation
            .replay_key
            .as_deref()
            .is_some_and(|key| key.contains("stable-turn"))
    );
    Ok(())
}

#[tokio::test]
async fn explicit_effect_controller_creates_turn_scope_internally() -> Result<()> {
    let recorder = RecordingInlineEffectController::default();
    let core = standard_core();
    let session = core.session("explicit-handler-effects").open().await?;

    session
        .turn(TurnInput::text("handler"))
        .turn_id("handler-turn")
        .effects(&recorder)
        .run()
        .await?;

    let llm_invocation = recorder
        .invocations()
        .into_iter()
        .find(|record| record.kind == lash_core::RuntimeEffectKind::LlmCall)
        .expect("llm effect");
    assert_eq!(llm_invocation.turn_id.as_deref(), Some("handler-turn"));
    Ok(())
}

#[tokio::test]
async fn queued_turn_run_drains_ready_work_and_returns_none_when_idle() -> Result<()> {
    let core = explicit_ephemeral_facets(LashCore::standard())
        .provider(mock_provider())
        .model(mock_model_spec())
        .store_factory(Arc::new(lash_core::InMemorySessionStoreFactory::new()))
        .build()?;
    let session = core.session("queued-turn-run").open().await?;
    let receipt = session
        .enqueue(TurnInput::text("queued work"))
        .id("queued-request")
        .send()
        .await?;

    let output = session
        .queued_turn()
        .batch_ids([receipt.batch_id.clone()])
        .run()
        .await?
        .expect("queued turn should run");

    assert_eq!(output.assistant_message(), Some("echo: queued work"));
    assert!(session.queued_turn().run().await?.is_none());
    Ok(())
}

#[tokio::test]
async fn queued_turn_explicit_effects_create_queue_drain_scope_internally() -> Result<()> {
    let recorder = RecordingInlineEffectController::default();
    let core = explicit_ephemeral_facets(LashCore::standard())
        .provider(mock_provider())
        .model(mock_model_spec())
        .store_factory(Arc::new(lash_core::InMemorySessionStoreFactory::new()))
        .build()?;
    let session = core.session("queued-explicit-effects").open().await?;
    let receipt = session
        .enqueue(TurnInput::text("queued handler"))
        .send()
        .await?;

    let output = session
        .queued_turn()
        .batch_ids([receipt.batch_id])
        .drain_id("handler-drain")
        .effects(&recorder)
        .run()
        .await?
        .expect("queued turn should run");

    assert_eq!(output.assistant_message(), Some("echo: queued handler"));
    let llm_invocation = recorder
        .invocations()
        .into_iter()
        .find(|record| record.kind == lash_core::RuntimeEffectKind::LlmCall)
        .expect("llm effect");
    assert_eq!(llm_invocation.turn_id.as_deref(), Some("handler-drain"));
    Ok(())
}

#[tokio::test]
async fn turn_builder_stream_emits_activities_and_finishes() -> Result<()> {
    let core = standard_core();
    let session = core.session("turn-stream").open().await?;
    let mut stream = session.turn(TurnInput::text("stream me")).stream()?;

    let mut activities = Vec::new();
    while let Some(activity) = stream.next_activity().await {
        activities.push(activity?);
    }
    let result = stream.finish().await?;

    assert!(matches!(
        result.outcome,
        TurnOutcome::Finished(lash_core::TurnFinish::AssistantMessage { .. })
    ));
    assert_eq!(assistant_prose(&activities), "echo: stream me");
    assert!(
        activities
            .iter()
            .any(|activity| matches!(&activity.event, TurnEvent::AssistantProseDelta { .. }))
    );
    Ok(())
}

#[tokio::test]
async fn session_observation_replays_live_activity_and_commit() -> Result<()> {
    let core = standard_core();
    let session = core.session("session-observation-replay").open().await?;
    let cursor = session.observe().current_observation().cursor;

    let output = session.turn(TurnInput::text("observe me")).run().await?;
    assert_eq!(assistant_prose(&output.activities), "echo: observe me");

    let replay = session.observe().resume_from_cursor(&cursor)?;
    let SessionResume::Replayed { events } = replay else {
        panic!("recent cursor should replay live events");
    };
    assert!(events.iter().any(|event| {
        matches!(
            &event.payload,
            lash_core::SessionObservationEventPayload::TurnActivity(activity)
                if matches!(
                    &activity.event,
                    TurnEvent::AssistantProseDelta { text } if text == "echo: observe me"
                )
        )
    }));
    assert!(events.iter().any(|event| {
        matches!(
            &event.payload,
            lash_core::SessionObservationEventPayload::Committed { .. }
        )
    }));
    Ok(())
}

#[tokio::test]
async fn session_observation_rejects_cursor_from_another_session() -> Result<()> {
    let core = standard_core();
    let session = core.session("session-observation-a").open().await?;
    let other = core.session("session-observation-b").open().await?;
    let other_cursor = other.observe().current_observation().cursor;

    let err = session
        .observe()
        .resume_from_cursor(&other_cursor)
        .expect_err("cursor from another session should be rejected");
    assert!(
        err.to_string().contains("session-observation-b")
            && err.to_string().contains("session-observation-a"),
        "unexpected error: {err}"
    );
    Ok(())
}

#[tokio::test]
async fn session_observation_subscription_replays_buffered_events_before_live_events() -> Result<()>
{
    let core = standard_core();
    let session = core
        .session("session-observation-subscribe-replay")
        .open()
        .await?;
    let cursor = session.observe().current_observation().cursor;

    session
        .turn(TurnInput::text("first observed"))
        .run()
        .await?;
    let SessionObservationSubscription::Subscribed(mut subscription) =
        session.observe().subscribe_from_cursor(&cursor)?
    else {
        panic!("recent cursor should subscribe without a gap");
    };

    loop {
        let event =
            tokio::time::timeout(std::time::Duration::from_secs(2), subscription.next_event())
                .await
                .expect("timed out waiting for replayed event")
                .expect("replayed event");
        if observation_assistant_delta(&event).as_deref() == Some("echo: first observed") {
            break;
        }
    }

    session
        .turn(TurnInput::text("second observed"))
        .run()
        .await?;
    loop {
        let event =
            tokio::time::timeout(std::time::Duration::from_secs(2), subscription.next_event())
                .await
                .expect("timed out waiting for live event")
                .expect("live event");
        if observation_assistant_delta(&event).as_deref() == Some("echo: second observed") {
            break;
        }
    }
    Ok(())
}

fn observation_assistant_delta(event: &lash_core::SessionObservationEvent) -> Option<String> {
    match &event.payload {
        lash_core::SessionObservationEventPayload::TurnActivity(activity) => {
            match &activity.event {
                TurnEvent::AssistantProseDelta { text } => Some(text.clone()),
                _ => None,
            }
        }
        _ => None,
    }
}

#[tokio::test]
async fn turn_stream_finish_returns_last_assistant_prose_group() -> Result<()> {
    let core = explicit_ephemeral_facets(LashCore::standard())
        .provider(semantic_group_provider())
        .model(mock_model_spec())
        .build()?;
    let session = core.session("turn-stream-last-group").open().await?;
    let mut stream = session.turn(TurnInput::text("stream groups")).stream()?;

    let mut activities = Vec::new();
    while let Some(activity) = stream.next_activity().await {
        activities.push(activity?);
    }
    let result = stream.finish().await?;

    assert_eq!(assistant_prose(&activities), "firstsecond");
    assert_eq!(result.assistant_message(), Some("second"));
    assert!(result.is_success());
    Ok(())
}

#[tokio::test]
async fn turn_run_collects_activities_and_returns_last_assistant_prose_group() -> Result<()> {
    let core = explicit_ephemeral_facets(LashCore::standard())
        .provider(semantic_group_provider())
        .model(mock_model_spec())
        .build()?;
    let session = core.session("turn-run-last-group").open().await?;

    let collected = session.turn(TurnInput::text("run groups")).run().await?;

    assert_eq!(assistant_prose(&collected.activities), "firstsecond");
    assert_eq!(collected.result.assistant_message(), Some("second"));
    Ok(())
}

#[tokio::test]
async fn retry_status_streams_as_semantic_turn_event() -> Result<()> {
    let core = explicit_ephemeral_facets(LashCore::standard())
        .provider(retry_once_provider())
        .model(mock_model_spec())
        .build()?;
    let session = core.session("retry-status").open().await?;
    let events = RecordingEvents::default();

    let result = session
        .turn(TurnInput::text("hello"))
        .stream_to(&events)
        .await?;

    assert!(matches!(
        result.outcome,
        TurnOutcome::Finished(lash_core::TurnFinish::AssistantMessage { .. })
    ));
    let retry = events
        .snapshot()
        .await
        .into_iter()
        .find(|event| matches!(&event.event, TurnEvent::RetryStatus { .. }))
        .expect("retry status event");
    let TurnEvent::RetryStatus {
        wait_seconds,
        attempt,
        max_attempts,
        reason,
    } = retry.event
    else {
        unreachable!();
    };
    assert_eq!(wait_seconds, 0);
    assert_eq!(attempt, 1);
    assert_eq!(max_attempts, 2);
    assert!(reason.contains("retry me"));
    Ok(())
}

#[tokio::test]
async fn control_turn_accepts_prebuilt_turn_input() -> Result<()> {
    let core = standard_core();
    let session = core.session("raw-turn").open().await?;
    let mut input = TurnInput::text("raw input");
    input.trace_turn_id = Some("host-trace-id".to_string());

    let result = session.turn(input).turn_id("host-trace-id").run().await?;

    assert_eq!(assistant_prose(&result.activities), "echo: raw input");
    Ok(())
}

#[tokio::test]
async fn queued_input_acceptance_streams_semantic_ack_with_id() -> Result<()> {
    let (entered_tx, entered_rx) = oneshot::channel();
    let (release_tx, release_rx) = oneshot::channel();
    let core = explicit_ephemeral_facets(LashCore::standard())
        .provider(checkpoint_gated_provider(entered_tx, release_rx))
        .model(mock_model_spec())
        .store_factory(Arc::new(lash_core::InMemorySessionStoreFactory::new()))
        .build()?;
    let session = core.session("queued-input").open().await?;
    let events = Arc::new(RecordingEvents::default());
    let turn_session = session.clone();
    let turn_events = Arc::clone(&events);
    let turn = tokio::spawn(async move {
        turn_session
            .turn(TurnInput::text("hello"))
            .stream_to(turn_events.as_ref())
            .await
    });

    entered_rx.await.expect("provider entered first call");
    session
        .control()
        .injection()
        .inject_turn_input(
            Some("queue-1".to_string()),
            lash_core::PluginMessage::text(lash_core::MessageRole::User, "queued follow-up"),
        )
        .await?;
    release_tx.send(()).expect("release provider");
    let result = turn.await.expect("turn task")?;

    assert!(matches!(
        result.outcome,
        TurnOutcome::Finished(lash_core::TurnFinish::AssistantMessage { .. })
    ));
    let events = events.snapshot().await;
    assert!(events.iter().any(|event| matches!(
        &event.event,
        TurnEvent::QueuedInputAccepted {
            checkpoint: lash_core::CheckpointKind::BeforeCompletion,
            inputs,
            ..
        } if inputs.iter().any(|input| input.id.as_deref() == Some("queue-1"))
    )));
    let prose = events
        .into_iter()
        .filter_map(|event| match event.event {
            TurnEvent::AssistantProseDelta { text } => Some(text),
            _ => None,
        })
        .collect::<String>();
    assert!(prose.contains("after queued follow-up"));
    Ok(())
}

#[tokio::test]
async fn pre_cancelled_token_yields_cancelled_outcome() -> Result<()> {
    let core = standard_core();
    let session = core.session("pre-cancelled").open().await?;
    let cancel = CancellationToken::new();
    cancel.cancel();

    let output = session
        .turn(TurnInput::text("never runs"))
        .cancel(cancel)
        .run()
        .await?;

    assert!(matches!(
        output.result.outcome,
        TurnOutcome::Stopped(lash_core::TurnStop::Cancelled)
    ));
    Ok(())
}

#[tokio::test]
async fn cancel_running_turns_stops_inflight_turn() -> Result<()> {
    let (started_tx, started_rx) = oneshot::channel::<()>();
    let started_tx = Arc::new(StdMutex::new(Some(started_tx)));
    let provider = crate::testing::TestProvider::builder()
        .kind("embed-test")
        .complete(move |_request| {
            let started_tx = Arc::clone(&started_tx);
            async move {
                if let Some(tx) = started_tx.lock().expect("started signal").take() {
                    let _ = tx.send(());
                }
                // Hang until the turn is cancelled out from under us.
                std::future::pending::<()>().await;
                unreachable!("provider future should be dropped by cancellation")
            }
        })
        .build()
        .into_handle();
    let core = explicit_ephemeral_facets(LashCore::standard())
        .provider(provider)
        .model(mock_model_spec())
        .build()
        .expect("core");
    let session = core.session("cancel-inflight").open().await?;
    let stopper = session.clone();

    let stream = session.turn(TurnInput::text("hang forever")).stream()?;
    started_rx.await.expect("provider reached");
    assert_eq!(stopper.cancel_running_turns(), 1);

    let result = stream.finish().await?;
    assert!(matches!(
        result.outcome,
        TurnOutcome::Stopped(lash_core::TurnStop::Cancelled)
    ));
    // The registry entry is gone once the turn finished.
    assert_eq!(stopper.cancel_running_turns(), 0);
    Ok(())
}

fn hang_on_signal_provider(started_tx: Arc<StdMutex<Vec<oneshot::Sender<()>>>>) -> ProviderHandle {
    crate::testing::TestProvider::builder()
        .kind("embed-test")
        .complete(move |request| {
            let started_tx = Arc::clone(&started_tx);
            async move {
                let user_text = last_user_text(&request);
                if user_text.contains("hang") {
                    if let Some(tx) = started_tx.lock().expect("started signal").pop() {
                        let _ = tx.send(());
                    }
                    // Hang until the turn is cancelled out from under us.
                    std::future::pending::<()>().await;
                    unreachable!("provider future should be dropped by cancellation")
                }
                Ok(text_response(&format!("echo: {user_text}")))
            }
        })
        .build()
        .into_handle()
}

#[tokio::test]
async fn cancel_running_turns_sweeps_lock_queued_turns() -> Result<()> {
    // One opened session serializes turn execution on the runtime writer
    // lock, but a second turn is already registered while it waits for that
    // lock. A stop sweep must reach both: the executing turn aborts, and the
    // parked turn sees its cancelled token the moment it acquires the lock
    // instead of starting a fresh provider call after the user pressed stop.
    let (started_tx, started_rx) = oneshot::channel::<()>();
    let provider = hang_on_signal_provider(Arc::new(StdMutex::new(vec![started_tx])));
    let core = explicit_ephemeral_facets(LashCore::standard())
        .provider(provider)
        .model(mock_model_spec())
        .build()
        .expect("core");
    let session = core.session("cancel-lock-queue").open().await?;

    let first = session.turn(TurnInput::text("hang one")).stream()?;
    started_rx.await.expect("first turn reached the provider");
    let second = session.turn(TurnInput::text("hang two")).stream()?;

    assert_eq!(session.cancel_running_turns(), 2);

    let first = first.finish().await?;
    let second = second.finish().await?;
    assert!(matches!(
        first.outcome,
        TurnOutcome::Stopped(lash_core::TurnStop::Cancelled)
    ));
    assert!(matches!(
        second.outcome,
        TurnOutcome::Stopped(lash_core::TurnStop::Cancelled)
    ));
    assert_eq!(session.cancel_running_turns(), 0);
    Ok(())
}

#[tokio::test]
async fn cancel_running_turns_does_not_cross_separately_opened_handles() -> Result<()> {
    // Each open() builds its own runtime and cancel registry; the documented
    // scope of cancel_running_turns is the opened handle and its clones.
    let (started_tx, started_rx) = oneshot::channel::<()>();
    let provider = hang_on_signal_provider(Arc::new(StdMutex::new(vec![started_tx])));
    let core = explicit_ephemeral_facets(LashCore::standard())
        .provider(provider)
        .model(mock_model_spec())
        .build()
        .expect("core");
    let handle_a = core.session("cancel-scope").open().await?;
    let handle_b = core.session("cancel-scope").open().await?;

    let hanging = handle_a.turn(TurnInput::text("hang here")).stream()?;
    started_rx.await.expect("turn reached the provider");

    // The other handle has its own registry: nothing to cancel there.
    assert_eq!(handle_b.cancel_running_turns(), 0);
    assert_eq!(handle_a.cancel_running_turns(), 1);

    let result = hanging.finish().await?;
    assert!(matches!(
        result.outcome,
        TurnOutcome::Stopped(lash_core::TurnStop::Cancelled)
    ));

    // The untouched handle keeps working.
    let output = handle_b.turn(TurnInput::text("plain")).run().await?;
    assert_eq!(output.assistant_message(), Some("echo: plain"));
    Ok(())
}

#[tokio::test]
async fn cancel_running_turns_reaches_queued_turn_drains() -> Result<()> {
    // Queued drains register in the same session registry as foreground
    // turns, so a stop sweep reaches them too.
    let (started_tx, started_rx) = oneshot::channel::<()>();
    let provider = hang_on_signal_provider(Arc::new(StdMutex::new(vec![started_tx])));
    let core = explicit_ephemeral_facets(LashCore::standard())
        .provider(provider)
        .model(mock_model_spec())
        .store_factory(Arc::new(lash_core::InMemorySessionStoreFactory::new()))
        .build()
        .expect("core");
    let session = core.session("cancel-queued-drain").open().await?;
    let receipt = session
        .enqueue(TurnInput::text("hang queued"))
        .send()
        .await?;

    let drainer = session.clone();
    let drain = tokio::spawn(async move {
        drainer
            .queued_turn()
            .batch_ids([receipt.batch_id])
            .run()
            .await
    });
    started_rx.await.expect("queued drain reached the provider");
    assert_eq!(session.cancel_running_turns(), 1);

    let output = drain
        .await
        .expect("drain task")?
        .expect("queued drain should produce a turn");
    assert!(matches!(
        output.result.outcome,
        TurnOutcome::Stopped(lash_core::TurnStop::Cancelled)
    ));
    Ok(())
}

#[tokio::test]
async fn await_queued_work_batch_resolves_when_drained() -> Result<()> {
    let core = explicit_ephemeral_facets(LashCore::standard())
        .provider(mock_provider())
        .model(mock_model_spec())
        .store_factory(Arc::new(lash_core::InMemorySessionStoreFactory::new()))
        .build()
        .expect("core");
    let session = core.session("await-queued").open().await?;
    let receipt = session
        .enqueue(TurnInput::text("queued work"))
        .send()
        .await?;

    let waiter_session = session.clone();
    let waiter_batch = receipt.batch_id.clone();
    let waiter =
        tokio::spawn(async move { waiter_session.await_queued_work_batch(&waiter_batch).await });

    // Nothing has drained the batch yet, so the waiter must still be pending.
    tokio::time::sleep(std::time::Duration::from_millis(80)).await;
    assert!(!waiter.is_finished(), "waiter resolved before any drain");

    let output = session
        .queued_turn()
        .batch_ids([receipt.batch_id.clone()])
        .run()
        .await?
        .expect("queued turn should run");
    assert_eq!(output.assistant_message(), Some("echo: queued work"));

    tokio::time::timeout(std::time::Duration::from_secs(5), waiter)
        .await
        .expect("waiter should resolve after the drain")
        .expect("waiter task")?;
    Ok(())
}

#[tokio::test]
async fn await_queued_work_batch_resolves_immediately_for_unknown_batch() -> Result<()> {
    let core = explicit_ephemeral_facets(LashCore::standard())
        .provider(mock_provider())
        .model(mock_model_spec())
        .store_factory(Arc::new(lash_core::InMemorySessionStoreFactory::new()))
        .build()
        .expect("core");
    let session = core.session("await-unknown").open().await?;
    tokio::time::timeout(
        std::time::Duration::from_secs(1),
        session.await_queued_work_batch("qwb:never-existed"),
    )
    .await
    .expect("unknown batch must resolve immediately")?;
    Ok(())
}

#[tokio::test]
async fn turn_stream_receives_semantic_activities() -> Result<()> {
    let core = standard_core();
    let session = core.session("semantic-stream").open().await?;
    let turn_events = RecordingEvents::default();

    let result = session
        .turn(TurnInput::text("semantic stream"))
        .cancel(CancellationToken::new())
        .stream_to(&turn_events)
        .await?;

    assert!(matches!(
        result.outcome,
        TurnOutcome::Finished(lash_core::TurnFinish::AssistantMessage { .. })
    ));
    assert!(
        turn_events
            .snapshot()
            .await
            .iter()
            .any(|event| matches!(&event.event, TurnEvent::AssistantProseDelta { .. }))
    );
    Ok(())
}

#[tokio::test]
async fn run_collects_ordered_assistant_prose_activity() -> Result<()> {
    let core = standard_core();
    let session = core.session("main").open().await?;

    let result = session.turn(TurnInput::text("visible")).run().await?;

    assert_eq!(assistant_prose(&result.activities), "echo: visible");
    assert!(
        result
            .activities
            .iter()
            .any(|activity| matches!(&activity.event, TurnEvent::AssistantProseDelta { .. }))
    );
    assert!(
        !result
            .activities
            .iter()
            .any(|activity| matches!(&activity.event, TurnEvent::ToolCallCompleted { .. }))
    );
    assert!(
        !result
            .activities
            .iter()
            .any(|activity| matches!(&activity.event, TurnEvent::CodeBlockCompleted { .. }))
    );
    assert!(matches!(
        result.result.outcome,
        TurnOutcome::Finished(lash_core::TurnFinish::AssistantMessage { .. })
    ));
    assert_eq!(result.result.usage.output_tokens, 2);
    Ok(())
}

#[tokio::test]
async fn private_run_collector_records_ordered_activities() -> Result<()> {
    let collector = RunActivityCollector::default();

    collector
        .emit(test_activity(
            "code-1",
            TurnEvent::CodeBlockStarted {
                language: "lashlang".to_string(),
                code: "x = await tools.app_lookup({})?".to_string(),
                graph_key: None,
            },
        ))
        .await;
    collector
        .emit(test_activity(
            "tool-1",
            TurnEvent::ToolCallCompleted {
                call_id: Some("call-1".to_string()),
                name: "app_lookup".to_string(),
                args: serde_json::json!({}),
                output: lash_core::ToolCallOutput::success(serde_json::json!({ "ok": true })),
                duration_ms: 3,
            },
        ))
        .await;
    collector
        .emit(test_activity(
            "code-1",
            TurnEvent::CodeBlockCompleted {
                language: "lashlang".to_string(),
                output: String::new(),
                error: None,
                success: true,
                duration_ms: 4,
                tool_call_ids: vec!["call-1".to_string()],
                graph_key: None,
            },
        ))
        .await;

    let activities = collector.snapshot();
    assert_eq!(activities.len(), 3);
    assert!(matches!(
        &activities[0].event,
        TurnEvent::CodeBlockStarted { language, code, .. }
            if language == "lashlang" && code == "x = await tools.app_lookup({})?"
    ));
    assert!(matches!(
        &activities[1].event,
        TurnEvent::ToolCallCompleted { name, output, .. }
            if name == "app_lookup" && output.value_for_projection() == serde_json::json!({ "ok": true })
    ));
    assert_eq!(activities[0].correlation_id, activities[2].correlation_id);
    assert!(matches!(
        &activities[2].event,
        TurnEvent::CodeBlockCompleted { language, success, .. }
            if language == "lashlang" && *success
    ));
    Ok(())
}

#[tokio::test]
async fn turn_event_fanout_streams_to_collector_and_live_sink() -> Result<()> {
    let live = Arc::new(RecordingEvents::default());
    let core = explicit_ephemeral_facets(LashCore::standard())
        .provider(tool_roundtrip_provider())
        .model(mock_model_spec())
        .tools(Arc::new(AppTools))
        .store_factory(Arc::new(lash_core::InMemorySessionStoreFactory::new()))
        .process_registry(Arc::new(TestLocalProcessRegistry::default()))
        .build()?;
    let session = core.session("fanout-tool-events").open().await?;

    let output = session
        .turn(TurnInput::text("use tool"))
        .advanced()
        .collect_with_scope(live.as_ref(), turn_scope(&session.session_id()))
        .await?;

    assert!(matches!(
        output.result.outcome,
        TurnOutcome::Finished(lash_core::TurnFinish::AssistantMessage { .. })
    ));
    assert_eq!(
        serde_json::to_value(&output.activities).expect("recorded activities serialize"),
        serde_json::to_value(live.snapshot().await).expect("live activities serialize")
    );
    assert_eq!(assistant_prose(&output.activities), "done");
    assert_eq!(output.assistant_message(), Some("done"));
    assert!(output.is_success());
    let tool_completed = output
        .activities
        .iter()
        .find(|activity| matches!(&activity.event, TurnEvent::ToolCallCompleted { .. }))
        .expect("tool completion");
    assert!(matches!(
        &tool_completed.event,
        TurnEvent::ToolCallCompleted { name, output, .. }
            if name == "app_lookup" && output.value_for_projection() == serde_json::json!({ "ok": true })
    ));
    Ok(())
}

#[tokio::test]
async fn stream_returns_terminal_metadata_without_prose() -> Result<()> {
    let core = standard_core();
    let session = core.session("semantic-events").open().await?;
    let events = RecordingEvents::default();

    let result = session
        .turn(TurnInput::text("stream"))
        .stream_to(&events)
        .await?;

    assert!(matches!(
        result.outcome,
        TurnOutcome::Finished(lash_core::TurnFinish::AssistantMessage { .. })
    ));
    let prose = events
        .snapshot()
        .await
        .into_iter()
        .filter_map(|event| match event.event {
            TurnEvent::AssistantProseDelta { text } => Some(text),
            _ => None,
        })
        .collect::<String>();
    assert_eq!(prose, "echo: stream");
    assert!(!events.snapshot().await.iter().any(|event| matches!(
        &event.event,
        TurnEvent::SubmittedValue { .. } | TurnEvent::ToolValue { .. }
    )));
    Ok(())
}

#[tokio::test]
async fn stream_emits_chronological_tool_events_without_prose_pollution() -> Result<()> {
    let core = explicit_ephemeral_facets(LashCore::standard())
        .provider(tool_roundtrip_provider())
        .model(mock_model_spec())
        .tools(Arc::new(AppTools))
        .store_factory(Arc::new(lash_core::InMemorySessionStoreFactory::new()))
        .process_registry(Arc::new(TestLocalProcessRegistry::default()))
        .build()?;
    let session = core.session("tool-events").open().await?;
    let events = RecordingEvents::default();

    let collected = session
        .turn(TurnInput::text("use tool"))
        .stream_to(&events)
        .await?;

    assert!(matches!(
        collected.outcome,
        TurnOutcome::Finished(lash_core::TurnFinish::AssistantMessage { .. })
    ));
    let events = events.snapshot().await;
    let started = events
        .iter()
        .position(|event| matches!(&event.event, TurnEvent::ToolCallStarted { .. }))
        .expect("tool start event");
    let completed = events
        .iter()
        .position(|event| matches!(&event.event, TurnEvent::ToolCallCompleted { .. }))
        .expect("tool completed event");
    assert!(started < completed);
    let TurnEvent::ToolCallCompleted { output, .. } = &events[completed].event else {
        unreachable!();
    };
    assert_eq!(
        output.value_for_projection(),
        serde_json::json!({ "ok": true })
    );
    let prose = events
        .into_iter()
        .filter_map(|event| match event.event {
            TurnEvent::AssistantProseDelta { text } => Some(text),
            _ => None,
        })
        .collect::<String>();
    assert_eq!(prose, "done");
    assert!(!prose.contains("ok"));
    Ok(())
}

#[test]
fn rlm_tool_calls_stream_from_live_exec_boundary() -> Result<()> {
    run_async_test_on_stack_budget("rlm-live-exec-boundary-test", || {
        rlm_tool_calls_stream_from_live_exec_boundary_inner()
    })
}

async fn rlm_tool_calls_stream_from_live_exec_boundary_inner() -> Result<()> {
    let core = explicit_ephemeral_facets(LashCore::rlm())
        .provider(queued_text_provider(vec![
            "```lashlang\nvalue = await tools.app_lookup({})?\nsubmit \"done\"\n```",
        ]))
        .model(mock_model_spec())
        .tools(Arc::new(AppTools))
        .store_factory(Arc::new(lash_core::InMemorySessionStoreFactory::new()))
        .process_registry(Arc::new(TestLocalProcessRegistry::default()))
        .build()?;
    let session = core.session("rlm-live-tool-events").open().await?;
    let events = Arc::new(RecordingEvents::default());

    let result = session
        .turn(TurnInput::text("use tool"))
        .stream_to(events.as_ref())
        .await?;

    assert!(matches!(
        result.outcome,
        TurnOutcome::Finished(lash_core::TurnFinish::SubmittedValue { .. })
    ));
    let events = events.snapshot().await;
    let code_started = events
        .iter()
        .position(|event| matches!(&event.event, TurnEvent::CodeBlockStarted { .. }))
        .expect("code started");
    let tool_started = events
        .iter()
        .position(|event| matches!(&event.event, TurnEvent::ToolCallStarted { .. }))
        .expect("tool started");
    let tool_completed = events
        .iter()
        .position(|event| matches!(&event.event, TurnEvent::ToolCallCompleted { .. }))
        .expect("tool completed");
    let code_completed = events
        .iter()
        .position(|event| matches!(&event.event, TurnEvent::CodeBlockCompleted { .. }))
        .expect("code completed");
    let terminal_output = events
        .iter()
        .position(|event| matches!(&event.event, TurnEvent::SubmittedValue { .. }))
        .expect("terminal output");
    assert!(code_started < tool_started);
    assert!(tool_started < tool_completed);
    assert!(tool_completed < code_completed);
    assert!(code_completed < terminal_output);
    assert!(!events[code_completed + 1..].iter().any(|event| matches!(
        &event.event,
        TurnEvent::ToolCallStarted { .. } | TurnEvent::ToolCallCompleted { .. }
    )));

    let TurnEvent::ToolCallCompleted {
        call_id, output, ..
    } = &events[tool_completed].event
    else {
        unreachable!();
    };
    assert_eq!(
        output.value_for_projection(),
        serde_json::json!({ "ok": true })
    );
    let TurnEvent::CodeBlockStarted {
        graph_key: started_graph_key,
        ..
    } = &events[code_started].event
    else {
        unreachable!();
    };
    assert!(
        started_graph_key
            .as_deref()
            .is_some_and(|key| key.starts_with("effect:rlm-live-tool-events:")),
        "missing foreground graph key on CodeBlockStarted: {started_graph_key:?}"
    );
    let TurnEvent::CodeBlockCompleted {
        language,
        success,
        error,
        tool_call_ids,
        graph_key: completed_graph_key,
        ..
    } = &events[code_completed].event
    else {
        unreachable!();
    };
    assert_eq!(language, "lashlang");
    assert!(*success);
    assert!(error.is_none());
    assert_eq!(call_id.as_ref(), tool_call_ids.first());
    assert_eq!(tool_call_ids.len(), 1);
    assert_eq!(completed_graph_key, started_graph_key);
    let read_view = result.state.read_view();
    assert!(
        read_view.messages().iter().all(|message| message
            .parts
            .iter()
            .all(|part| part.tool_call_id.as_ref() != tool_call_ids.first())),
        "live RLM tool calls should not be persisted as message history"
    );
    assert_eq!(
        read_view
            .materialized_session_graph()
            .active_path_nodes()
            .into_iter()
            .filter_map(|node| node.event())
            .filter(|event| matches!(event, lash_core::SessionEventRecord::Conversation(_)))
            .count(),
        read_view.messages().len()
    );
    let TurnEvent::SubmittedValue { value } = &events[terminal_output].event else {
        unreachable!();
    };
    assert_eq!(value, &serde_json::json!("done"));
    Ok(())
}

#[test]
fn continue_as_observation_emits_frame_switch_then_commit() -> Result<()> {
    run_async_test_on_stack_budget("continue-as-observation-test", || {
        continue_as_observation_emits_frame_switch_then_commit_inner()
    })
}

async fn continue_as_observation_emits_frame_switch_then_commit_inner() -> Result<()> {
    let core = explicit_ephemeral_facets(LashCore::rlm())
        .provider(queued_text_provider(vec![
            "```lashlang\nawait control.continue_as({ task: \"finish in a fresh frame\" })?\n```",
            "```lashlang\nsubmit \"done after continue_as\"\n```",
        ]))
        .model(mock_model_spec())
        .store_factory(Arc::new(lash_core::InMemorySessionStoreFactory::new()))
        .build()?;
    let session = core.session("continue-as-observation").open().await?;
    let cursor = session.observe().current_observation().cursor;

    let output = session.turn(TurnInput::text("switch frames")).run().await?;
    assert_eq!(
        output.submitted_value(),
        Some(&serde_json::json!("done after continue_as"))
    );

    let SessionResume::Replayed { events } = session.observe().resume_from_cursor(&cursor)? else {
        panic!("recent cursor should replay continue_as observation events");
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
    Ok(())
}

#[test]
fn durable_agent_frame_follow_through_uses_distinct_turn_scopes_and_commits() -> Result<()> {
    run_async_test_on_stack_budget("durable-agent-frame-follow-through-test", || {
        durable_agent_frame_follow_through_uses_distinct_turn_scopes_and_commits_inner()
    })
}

async fn durable_agent_frame_follow_through_uses_distinct_turn_scopes_and_commits_inner()
-> Result<()> {
    let dir = tempfile::tempdir().expect("tempdir");
    let session_id = "agent-frame-durable";
    let root_turn_id = "agent-frame-root-turn";
    let store_factory = Arc::new(lash_sqlite_store::SqliteSessionStoreFactory::new(
        dir.path().join("sessions"),
    ));
    let artifact_store = Arc::new(
        lash_sqlite_store::Store::open(&dir.path().join("artifacts.db"))
            .await
            .expect("open artifact store"),
    );
    let process_registry = Arc::new(
        lash_sqlite_store::SqliteProcessRegistry::open(&dir.path().join("processes.db"))
            .await
            .expect("open process registry"),
    );
    let host_event_store = Arc::new(
        lash_sqlite_store::SqliteHostEventStore::open(&dir.path().join("host-events.db"))
            .await
            .expect("open host event store"),
    );
    let controller = Arc::new(RecordingDurableEffectController::default());
    let scoped_effect_controller = ScopedEffectController::borrowed(
        controller.as_ref(),
        lash_core::EffectScope::turn(session_id, root_turn_id),
    )
    .expect("scoped durable effect controller");
    let core = explicit_ephemeral_facets(LashCore::standard())
        .provider(agent_frame_switch_provider())
        .model(mock_model_spec())
        .tools(Arc::new(AgentFrameSwitchTools))
        .store_factory(store_factory.clone())
        .attachment_store(Arc::new(crate::persistence::FileAttachmentStore::new(
            dir.path().join("attachments"),
        )))
        .lashlang_artifact_store(artifact_store)
        .host_event_store(host_event_store)
        .process_registry(process_registry)
        .build()?;
    let session = core.session(session_id).open().await?;
    let mut input = TurnInput::text("switch frames");
    input.trace_turn_id = Some(root_turn_id.to_string());

    let output = session
        .turn(input)
        .advanced()
        .run_with_scope(scoped_effect_controller)
        .await?;

    assert_eq!(output.assistant_message(), Some("done after frame switch"));
    let follow_turn_id = format!("{root_turn_id}:agent-frame:1");
    let mut llm_turn_ids = controller
        .invocations()
        .into_iter()
        .filter(|record| record.kind == lash_core::RuntimeEffectKind::LlmCall)
        .map(|record| record.turn_id.expect("turn-scoped LLM effect"))
        .collect::<Vec<_>>();
    llm_turn_ids.sort();
    llm_turn_ids.dedup();
    assert_eq!(
        llm_turn_ids,
        vec![root_turn_id.to_string(), follow_turn_id.clone()]
    );
    let replay_keys = controller
        .invocations()
        .into_iter()
        .filter_map(|record| record.replay_key)
        .collect::<Vec<_>>();
    assert!(
        replay_keys.iter().any(|key| key.contains(root_turn_id)),
        "root turn replay keys should include {root_turn_id}: {replay_keys:?}"
    );
    assert!(
        replay_keys.iter().any(|key| key.contains(&follow_turn_id)),
        "follow turn replay keys should include {follow_turn_id}: {replay_keys:?}"
    );

    let conn = rusqlite::Connection::open(store_factory.path_for_session(session_id))
        .expect("open session sqlite store");
    let mut stmt = conn
        .prepare("SELECT turn_id FROM runtime_turn_commits ORDER BY turn_id ASC")
        .expect("prepare turn commits");
    let turn_commit_ids = stmt
        .query_map([], |row| row.get::<_, String>(0))
        .expect("query turn commits")
        .map(|row| row.expect("read turn commit row"))
        .collect::<Vec<_>>();
    assert_eq!(
        turn_commit_ids,
        vec![root_turn_id.to_string(), follow_turn_id]
    );
    Ok(())
}

#[test]
fn process_control_lists_started_lashlang_process_until_awaited() -> Result<()> {
    run_async_test_on_stack_budget("process-control-lashlang-process-test", || {
        process_control_lists_started_lashlang_process_until_awaited_inner()
    })
}

async fn process_control_lists_started_lashlang_process_until_awaited_inner() -> Result<()> {
    let (entered_tx, entered_rx) = oneshot::channel();
    let (release_tx, release_rx) = oneshot::channel();
    let core = explicit_ephemeral_facets(LashCore::rlm())
        .provider(queued_text_provider(vec![
            r#"```lashlang
process lookup(tools: Tools) {
  value = await tools.app_lookup({})?
  finish value
}
h = start lookup(tools: tools)
value = await h
submit value
```"#,
        ]))
        .model(mock_model_spec())
        .tools(Arc::new(BlockingAppTools::new(entered_tx, release_rx)))
        // A started (`start lookup(...)`) process runs in the lease-protected
        // worker's rebuilt runtime, which needs a session store factory; the
        // explicit in-memory factory backs ephemeral process execution.
        .store_factory(Arc::new(lash_core::InMemorySessionStoreFactory::new()))
        .process_registry(Arc::new(TestLocalProcessRegistry::default()))
        .build()?;
    let session = core.session("rlm-process-control-tool").open().await?;
    let turn_session = session.clone();
    let scoped_effect_controller = turn_scope(&turn_session.session_id());
    let turn = tokio::spawn(async move {
        turn_session
            .turn(TurnInput::text("start tool"))
            .advanced()
            .run_with_scope(scoped_effect_controller)
            .await
    });

    tokio::time::timeout(std::time::Duration::from_secs(5), entered_rx)
        .await
        .expect("tool process should start")
        .expect("tool provider entered");

    let processes = session.process_control().list().await?;
    let running_app_lookup = processes.iter().any(|process| {
        process.descriptor.kind.as_deref() == Some("lashlang")
            && process.descriptor.label.as_deref() == Some("lookup")
            && !process.status.is_terminal()
    });
    assert!(
        running_app_lookup,
        "expected running lookup lashlang process, got {processes:?}"
    );

    release_tx.send(()).expect("release tool provider");
    let result = turn.await.expect("turn task")?;
    assert_eq!(
        result.submitted_value(),
        Some(&serde_json::json!({
            "ok": true,
            "value": { "answer": "ready" },
        }))
    );
    Ok(())
}

#[test]
fn lashlang_execution_graph_store_observes_lashlang_process_from_facade() -> Result<()> {
    run_async_test_on_stack_budget("lashlang-graph-store-facade-test", || {
        lashlang_execution_graph_store_observes_lashlang_process_from_facade_inner()
    })
}

async fn lashlang_execution_graph_store_observes_lashlang_process_from_facade_inner() -> Result<()>
{
    let (entered_tx, entered_rx) = oneshot::channel();
    let (release_tx, release_rx) = oneshot::channel();
    let graph_store = Arc::new(crate::tracing::TraceLashlangGraphStore::default());
    let core = explicit_ephemeral_facets(LashCore::rlm())
        .provider(queued_text_provider(vec![
            r#"```lashlang
process lookup(tools: Tools) {
  value = await tools.app_lookup({})?
  finish value
}
h = start lookup(tools: tools)
value = await h
submit value
```"#,
        ]))
        .model(mock_model_spec())
        .tools(Arc::new(BlockingAppTools::new(entered_tx, release_rx)))
        .store_factory(Arc::new(lash_core::InMemorySessionStoreFactory::new()))
        .process_registry(Arc::new(TestLocalProcessRegistry::default()))
        .lashlang_execution_sink(Arc::clone(&graph_store) as Arc<dyn crate::tracing::TraceSink>)
        .build()?;
    let session = core.session("rlm-lashlang-graph-store").open().await?;
    let turn_session = session.clone();
    let scoped_effect_controller = turn_scope(&turn_session.session_id());
    let turn = tokio::spawn(async move {
        turn_session
            .turn(TurnInput::text("start tool"))
            .advanced()
            .run_with_scope(scoped_effect_controller)
            .await
    });

    tokio::time::timeout(std::time::Duration::from_secs(5), entered_rx)
        .await
        .expect("tool process should start")
        .expect("tool provider entered");

    let processes = session.process_control().list().await?;
    let running = processes
        .iter()
        .find(|process| process.descriptor.label.as_deref() == Some("lookup"))
        .expect("running lookup process");
    let graph = graph_store
        .graph(&format!("process:{}", running.process_id))
        .expect("Lashlang graph snapshot");
    assert_eq!(graph.graph_key, format!("process:{}", running.process_id));
    assert_eq!(graph.entry_kind, "process");
    assert_eq!(graph.entry_name, "lookup");
    assert_eq!(graph.status, lash_core::TraceLashlangStatus::Running);
    assert!(!graph.nodes.is_empty());
    assert!(
        graph_store
            .graphs()
            .iter()
            .any(|graph| graph.entry_name == "lookup")
    );

    release_tx.send(()).expect("release tool provider");
    let _ = turn.await.expect("turn task")?;
    Ok(())
}

#[tokio::test]
async fn prose_or_submit_rlm_completion_emits_no_terminal_output() -> Result<()> {
    let core = explicit_ephemeral_facets(LashCore::rlm())
        .provider(queued_text_provider(vec!["done in prose"]))
        .model(mock_model_spec())
        .build()?;
    let session = core.session("rlm-prose-completion").open().await?;
    let events = Arc::new(RecordingEvents::default());

    let result = session
        .turn(TurnInput::text("answer directly"))
        .allow_prose_or_submit()?
        .stream_to(events.as_ref())
        .await?;

    assert!(matches!(
        result.outcome,
        TurnOutcome::Finished(lash_core::TurnFinish::AssistantMessage { .. })
    ));
    let events = events.snapshot().await;
    assert!(!events.iter().any(|event| matches!(
        &event.event,
        TurnEvent::SubmittedValue { .. } | TurnEvent::ToolValue { .. }
    )));
    assert_eq!(assistant_prose(&events), "done in prose");
    let read_view = result.state.read_view();
    let assistant_messages = read_view
        .messages()
        .iter()
        .filter(|message| message.role == lash_core::MessageRole::Assistant)
        .collect::<Vec<_>>();
    assert_eq!(assistant_messages.len(), 1);
    assert_eq!(assistant_messages[0].parts[0].content, "done in prose");
    Ok(())
}

#[tokio::test]
async fn submit_required_rlm_completion_emits_terminal_output() -> Result<()> {
    let core = explicit_ephemeral_facets(LashCore::rlm())
        .provider(queued_text_provider(vec![
            "```lashlang\nsubmit \"done via submit\"\n```",
        ]))
        .model(mock_model_spec())
        .build()?;
    let session = core
        .session("rlm-submit-required-completion")
        .open()
        .await?;
    let events = Arc::new(RecordingEvents::default());

    let result = session
        .turn(TurnInput::text("submit"))
        .require_submit()?
        .stream_to(events.as_ref())
        .await?;

    assert!(matches!(
        result.outcome,
        TurnOutcome::Finished(lash_core::TurnFinish::SubmittedValue { .. })
    ));
    assert_eq!(
        result.submitted_value(),
        Some(&serde_json::json!("done via submit"))
    );
    let events = events.snapshot().await;
    let terminal_output = events
        .iter()
        .find(|event| matches!(&event.event, TurnEvent::SubmittedValue { .. }))
        .expect("terminal output");
    let TurnEvent::SubmittedValue { value } = &terminal_output.event else {
        unreachable!();
    };
    assert_eq!(value, &serde_json::json!("done via submit"));
    Ok(())
}

#[tokio::test]
async fn rlm_failed_code_emits_failed_code_completion_without_fake_tools() -> Result<()> {
    let core = explicit_ephemeral_facets(LashCore::rlm())
        .provider(queued_text_provider(vec![
            "```lashlang\nthis is not valid lashlang\n```",
            "```lashlang\nsubmit \"recovered\"\n```",
        ]))
        .model(mock_model_spec())
        .tools(Arc::new(AppTools))
        .build()?;
    let session = core.session("rlm-failed-code-event").open().await?;
    let events = RecordingEvents::default();

    let _result = session
        .turn(TurnInput::text("bad code"))
        .stream_to(&events)
        .await?;

    let events = events.snapshot().await;
    let failed = events
        .iter()
        .position(|event| {
            matches!(
                &event.event,
                TurnEvent::CodeBlockCompleted {
                    success: false,
                    error: Some(_),
                    ..
                }
            )
        })
        .expect("failed code completion");
    let next_code = events[failed + 1..]
        .iter()
        .position(|event| matches!(&event.event, TurnEvent::CodeBlockStarted { .. }))
        .map(|offset| failed + 1 + offset)
        .unwrap_or(events.len());
    assert!(
        !events[failed + 1..next_code]
            .iter()
            .any(|event| matches!(&event.event, TurnEvent::ToolCallCompleted { .. }))
    );
    Ok(())
}
