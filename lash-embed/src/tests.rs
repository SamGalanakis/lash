use crate::support::*;
use std::collections::VecDeque;

use lash::LlmOutputPart;
use lash::llm::transport::LlmTransportError;
use lash::llm::types::{
    LlmContentBlock, LlmRequest, LlmResponse, LlmRole, LlmStreamEvent, ResponseTextMeta,
};
use tokio::sync::{Mutex as TokioMutex, oneshot};

#[derive(Default)]
struct SnapshotStore {
    read: std::sync::Mutex<Option<lash::PersistedSessionRead>>,
}

impl SnapshotStore {
    fn with_state(state: PersistedSessionState) -> Self {
        let turn_state = state.turn_state();
        let config = lash::PersistedSessionConfig {
            provider_id: state.policy.provider.kind().to_string(),
            configured_model: state.policy.model.clone(),
            context_window: state.policy.max_context_tokens.unwrap_or_default() as u64,
            execution_mode: state.policy.execution_mode.clone(),
            standard_context_approach: state.policy.standard_context_approach.clone(),
            model_variant: state.policy.model_variant.clone(),
        };
        Self {
            read: std::sync::Mutex::new(Some(lash::PersistedSessionRead {
                session_id: state.session_id,
                head_revision: 7,
                config,
                graph: state.session_graph,
                checkpoint_ref: None,
                checkpoint: Some(lash::HydratedSessionCheckpoint {
                    turn_state,
                    tool_state: state.tool_state_snapshot,
                    ..Default::default()
                }),
                token_ledger: Vec::new(),
            })),
        }
    }
}

#[async_trait]
impl lash::RuntimePersistence for SnapshotStore {
    async fn load_session(
        &self,
        _scope: lash::SessionReadScope,
    ) -> std::result::Result<Option<lash::PersistedSessionRead>, lash::store::StoreError> {
        Ok(self.read.lock().expect("snapshot store lock").clone())
    }

    async fn load_node(
        &self,
        _node_id: &str,
    ) -> std::result::Result<Option<lash::SessionNodeRecord>, lash::store::StoreError> {
        Ok(None)
    }

    async fn commit_runtime_state(
        &self,
        commit: lash::RuntimeCommit,
    ) -> std::result::Result<lash::RuntimeCommitResult, lash::store::StoreError> {
        let mut read = self.read.lock().expect("snapshot store lock");
        let existing_graph = read
            .as_ref()
            .map(|read| read.graph.clone())
            .unwrap_or_default();
        let graph = match commit.graph.clone() {
            lash::GraphCommitDelta::ReplaceFull(graph) => graph,
            lash::GraphCommitDelta::Unchanged { leaf_node_id } => {
                let mut graph = existing_graph;
                graph.set_leaf_node_id(leaf_node_id);
                graph
            }
            lash::GraphCommitDelta::Append {
                nodes,
                leaf_node_id,
            } => {
                let mut graph = existing_graph;
                graph.extend_node_records(nodes);
                graph.set_leaf_node_id(leaf_node_id);
                graph
            }
        };
        *read = Some(lash::PersistedSessionRead {
            session_id: commit.session_id.clone(),
            head_revision: 8,
            config: commit.config,
            graph,
            checkpoint_ref: Some(lash::BlobRef("checkpoint".to_string())),
            checkpoint: Some(commit.checkpoint),
            token_ledger: commit.usage_deltas,
        });
        Ok(lash::RuntimeCommitResult {
            head_revision: 8,
            checkpoint_ref: lash::BlobRef("checkpoint".to_string()),
            manifest: lash::SessionCheckpoint::default(),
        })
    }

    async fn save_session_meta(
        &self,
        _meta: lash::SessionMeta,
    ) -> std::result::Result<(), lash::store::StoreError> {
        Ok(())
    }

    async fn load_session_meta(
        &self,
    ) -> std::result::Result<Option<lash::SessionMeta>, lash::store::StoreError> {
        Ok(None)
    }

    async fn tombstone_nodes(
        &self,
        _ids: &[String],
    ) -> std::result::Result<(), lash::store::StoreError> {
        Ok(())
    }

    async fn vacuum(&self) -> std::result::Result<lash::VacuumReport, lash::store::StoreError> {
        Ok(lash::VacuumReport::default())
    }

    async fn gc_unreachable(&self) -> std::result::Result<lash::GcReport, lash::store::StoreError> {
        Ok(lash::GcReport::default())
    }
}

#[derive(Clone)]
struct ReusableStoreFactory {
    store: Arc<dyn lash::RuntimePersistence>,
}

impl lash::SessionStoreFactory for ReusableStoreFactory {
    fn create_store(
        &self,
        _request: &lash::SessionStoreCreateRequest,
    ) -> std::result::Result<Arc<dyn lash::RuntimePersistence>, String> {
        Ok(Arc::clone(&self.store))
    }
}

#[derive(Default)]
struct RecordingStoreFactory {
    requests: std::sync::Mutex<Vec<lash::SessionStoreCreateRequest>>,
}

impl RecordingStoreFactory {
    fn session_ids(&self) -> Vec<String> {
        self.requests
            .lock()
            .expect("recording factory lock")
            .iter()
            .map(|request| request.session_id.clone())
            .collect()
    }
}

impl lash::SessionStoreFactory for RecordingStoreFactory {
    fn create_store(
        &self,
        request: &lash::SessionStoreCreateRequest,
    ) -> std::result::Result<Arc<dyn lash::RuntimePersistence>, String> {
        self.requests
            .lock()
            .expect("recording factory lock")
            .push(request.clone());
        Ok(Arc::new(SnapshotStore::default()))
    }
}

#[derive(Default)]
struct RecordingEvents {
    events: TokioMutex<Vec<TurnActivity>>,
}

impl RecordingEvents {
    async fn snapshot(&self) -> Vec<TurnActivity> {
        self.events.lock().await.clone()
    }
}

#[async_trait]
impl TurnActivitySink for RecordingEvents {
    async fn emit(&self, activity: TurnActivity) {
        self.events.lock().await.push(activity);
    }
}

fn test_activity(correlation_id: &str, event: TurnEvent) -> TurnActivity {
    TurnActivity::new(TurnActivityId::new(correlation_id.to_string()), event)
}

fn assistant_prose(events: &[TurnActivity]) -> String {
    events
        .iter()
        .filter_map(|activity| match &activity.event {
            TurnEvent::AssistantProseDelta { text } => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

struct AppTools;

#[async_trait]
impl ToolProvider for AppTools {
    fn definitions(&self) -> Vec<lash::ToolDefinition> {
        vec![lash::ToolDefinition::raw(
            "app_lookup",
            "Look up app state.",
            serde_json::json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
            serde_json::json!({ "type": "object" }),
        )]
    }

    async fn execute(&self, _call: lash::ToolCall<'_>) -> lash::ToolResult {
        lash::ToolResult::ok(serde_json::json!({ "ok": true }))
    }
}

struct LongTextTools;

#[async_trait]
impl ToolProvider for LongTextTools {
    fn definitions(&self) -> Vec<lash::ToolDefinition> {
        vec![lash::ToolDefinition::raw(
            "app_lookup",
            "Look up verbose app state.",
            serde_json::json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
            serde_json::json!({ "type": "string" }),
        )]
    }

    async fn execute(&self, _call: lash::ToolCall<'_>) -> lash::ToolResult {
        lash::ToolResult::ok(serde_json::json!("abcdefghijklmnopqrstuvwxyz0123456789"))
    }
}

struct SurfacePluginFactory;

impl lash::PluginFactory for SurfacePluginFactory {
    fn id(&self) -> &'static str {
        "surface_test"
    }

    fn build(
        &self,
        _ctx: &lash::PluginSessionContext,
    ) -> std::result::Result<Arc<dyn lash::SessionPlugin>, lash::PluginError> {
        Ok(Arc::new(SurfacePlugin))
    }
}

struct SurfacePlugin;

impl lash::SessionPlugin for SurfacePlugin {
    fn id(&self) -> &'static str {
        "surface_test"
    }

    fn register(
        &self,
        reg: &mut lash::PluginRegistrar,
    ) -> std::result::Result<(), lash::PluginError> {
        reg.output().response(Arc::new(|ctx| {
            Box::pin(async move {
                Ok(lash::AssistantResponseTransform {
                    response: ctx.response,
                    events: vec![lash::PluginSurfaceEvent::Status {
                        key: "surface".to_string(),
                        label: "working".to_string(),
                        detail: Some("details".to_string()),
                        transient_ms: None,
                    }],
                })
            })
        }));
        Ok(())
    }
}

fn mock_provider() -> ProviderHandle {
    lash::testing::TestProvider::builder()
        .kind("embed-test")
        .default_model("mock-model")
        .requires_streaming(true)
        .complete(|request| async move {
            let user_text = last_user_text(&request);
            let reply = format!("echo: {user_text}");
            if let Some(events) = request.stream_events.as_ref() {
                events.send(LlmStreamEvent::Delta(reply.clone()));
            }
            Ok(LlmResponse {
                full_text: reply.clone(),
                parts: vec![LlmOutputPart::Text {
                    text: reply,
                    response_meta: None,
                }],
                usage: lash::llm::types::LlmUsage {
                    input_tokens: user_text.split_whitespace().count() as i64,
                    output_tokens: 2,
                    cached_input_tokens: 0,
                    reasoning_tokens: 0,
                },
                ..LlmResponse::default()
            })
        })
        .build()
        .into_handle()
}

fn tool_roundtrip_provider() -> ProviderHandle {
    let responses = Arc::new(TokioMutex::new(VecDeque::from([
        LlmResponse {
            parts: vec![LlmOutputPart::ToolCall {
                call_id: "call-1".to_string(),
                tool_name: "app_lookup".to_string(),
                input_json: "{}".to_string(),
                item_id: None,
                signature: None,
            }],
            ..LlmResponse::default()
        },
        LlmResponse {
            full_text: "done".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "done".to_string(),
                response_meta: None,
            }],
            ..LlmResponse::default()
        },
    ])));
    lash::testing::TestProvider::builder()
        .kind("embed-test")
        .default_model("mock-model")
        .complete(move |_request| {
            let responses = Arc::clone(&responses);
            async move { Ok(responses.lock().await.pop_front().expect("queued response")) }
        })
        .build()
        .into_handle()
}

fn text_response(text: &str) -> LlmResponse {
    LlmResponse {
        full_text: text.to_string(),
        parts: vec![LlmOutputPart::Text {
            text: text.to_string(),
            response_meta: None,
        }],
        ..LlmResponse::default()
    }
}

fn queued_text_provider(texts: Vec<&'static str>) -> ProviderHandle {
    let responses = Arc::new(TokioMutex::new(VecDeque::from(
        texts
            .into_iter()
            .map(|text| LlmResponse {
                full_text: text.to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: text.to_string(),
                    response_meta: None,
                }],
                ..LlmResponse::default()
            })
            .collect::<Vec<_>>(),
    )));
    lash::testing::TestProvider::builder()
        .kind("embed-test")
        .default_model("mock-model")
        .complete(move |_request| {
            let responses = Arc::clone(&responses);
            async move { Ok(responses.lock().await.pop_front().expect("queued response")) }
        })
        .build()
        .into_handle()
}

fn semantic_group_provider() -> ProviderHandle {
    lash::testing::TestProvider::builder()
        .kind("embed-test")
        .default_model("mock-model")
        .complete(|_request| async move {
            Ok(LlmResponse {
                full_text: "firstsecond".to_string(),
                parts: vec![
                    LlmOutputPart::Text {
                        text: "first".to_string(),
                        response_meta: Some(ResponseTextMeta {
                            id: Some("assistant:first".to_string()),
                            status: None,
                            phase: None,
                        }),
                    },
                    LlmOutputPart::Text {
                        text: "second".to_string(),
                        response_meta: Some(ResponseTextMeta {
                            id: Some("assistant:second".to_string()),
                            status: None,
                            phase: None,
                        }),
                    },
                ],
                ..LlmResponse::default()
            })
        })
        .build()
        .into_handle()
}

fn text_provider(kind: &'static str, model: &'static str, text: &'static str) -> ProviderHandle {
    lash::testing::TestProvider::builder()
        .kind(kind)
        .default_model(model)
        .complete(move |_request| async move {
            Ok(LlmResponse {
                full_text: text.to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: text.to_string(),
                    response_meta: None,
                }],
                ..LlmResponse::default()
            })
        })
        .build()
        .into_handle()
}

type SeenModels = Arc<std::sync::Mutex<Vec<(String, Option<String>)>>>;

fn recording_text_provider(
    kind: &'static str,
    model: &'static str,
    variant: Option<&'static str>,
    text: &'static str,
    seen: SeenModels,
) -> ProviderHandle {
    lash::testing::TestProvider::builder()
        .kind(kind)
        .default_model(model)
        .supported_variants(|_| {
            &[
                "core-variant",
                "session-variant",
                "turn-variant",
                "updated-variant",
                "manual-variant",
            ]
        })
        .default_model_variant(
            move |requested| {
                if requested == model { variant } else { None }
            },
        )
        .complete(move |request| {
            let seen = Arc::clone(&seen);
            async move {
                seen.lock()
                    .expect("seen requests")
                    .push((request.model, request.model_variant));
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
        .build()
        .into_handle()
}

fn last_user_text(request: &LlmRequest) -> String {
    request
        .messages
        .iter()
        .rev()
        .find(|message| message.role == LlmRole::User)
        .map(|message| {
            message
                .blocks
                .iter()
                .filter_map(|block| match block {
                    LlmContentBlock::Text { text, .. } => Some(text.as_ref()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default()
}

fn system_text(request: &LlmRequest) -> String {
    request
        .messages
        .iter()
        .find(|message| message.role == LlmRole::System)
        .map(|message| {
            message
                .blocks
                .iter()
                .filter_map(|block| match block {
                    LlmContentBlock::Text { text, .. } => Some(text.as_ref()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default()
}

fn recording_prompt_provider(seen: Arc<std::sync::Mutex<Vec<String>>>) -> ProviderHandle {
    lash::testing::TestProvider::builder()
        .kind("prompt-test")
        .default_model("mock-model")
        .complete(move |request| {
            let seen = Arc::clone(&seen);
            async move {
                seen.lock()
                    .expect("seen prompts")
                    .push(system_text(&request));
                Ok(LlmResponse {
                    full_text: "ok".to_string(),
                    parts: vec![LlmOutputPart::Text {
                        text: "ok".to_string(),
                        response_meta: None,
                    }],
                    ..LlmResponse::default()
                })
            }
        })
        .build()
        .into_handle()
}

fn retry_once_provider() -> ProviderHandle {
    let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    lash::testing::TestProvider::builder()
        .kind("retry-test")
        .default_model("mock-model")
        .requires_streaming(true)
        .options(lash::ProviderOptions {
            reliability: lash::provider::ProviderReliability::builder()
                .max_attempts(2)
                .base_delay_ms(0)
                .max_delay_ms(0)
                .build(),
            ..lash::ProviderOptions::default()
        })
        .complete(move |_request| {
            let attempts = Arc::clone(&attempts);
            async move {
                if attempts.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                    return Err(LlmTransportError::new("retry me").retryable(true));
                }
                Ok(LlmResponse {
                    full_text: "retried".to_string(),
                    parts: vec![LlmOutputPart::Text {
                        text: "retried".to_string(),
                        response_meta: None,
                    }],
                    ..LlmResponse::default()
                })
            }
        })
        .build()
        .into_handle()
}

fn checkpoint_gated_provider(
    entered_tx: oneshot::Sender<()>,
    release_rx: oneshot::Receiver<()>,
) -> ProviderHandle {
    let entered_tx = Arc::new(std::sync::Mutex::new(Some(entered_tx)));
    let release_rx = Arc::new(TokioMutex::new(Some(release_rx)));
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    lash::testing::TestProvider::builder()
        .kind("checkpoint-gated")
        .default_model("mock-model")
        .complete(move |request| {
            let entered_tx = Arc::clone(&entered_tx);
            let release_rx = Arc::clone(&release_rx);
            let calls = Arc::clone(&calls);
            async move {
                let call = calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if call == 0 {
                    if let Some(tx) = entered_tx.lock().expect("entered tx").take() {
                        let _ = tx.send(());
                    }
                    if let Some(rx) = release_rx.lock().await.take() {
                        let _ = rx.await;
                    }
                    Ok(text_response("first"))
                } else {
                    Ok(text_response(&format!(
                        "after {}",
                        last_user_text(&request)
                    )))
                }
            }
        })
        .build()
        .into_handle()
}

fn standard_core() -> LashCore {
    LashCore::standard()
        .provider(mock_provider())
        .model("mock-model", None)
        .max_context_tokens(200_000)
        .build()
        .expect("standard core")
}

#[tokio::test]
async fn standard_core_runs_mock_turn() -> Result<()> {
    let core = standard_core();
    let session = core.session("main").open().await?;
    let events = RecordingEvents::default();

    let result = session
        .turn(TurnInput::text("hello"))
        .stream(&events)
        .await?;

    assert!(matches!(
        result.outcome,
        TurnOutcome::Finished(lash::TurnFinish::AssistantMessage { .. })
    ));
    let events = events.snapshot().await;
    assert!(
        events
            .iter()
            .any(|event| matches!(&event.event, TurnEvent::AssistantProseDelta { .. }))
    );
    assert!(
        !events
            .iter()
            .any(|event| matches!(&event.event, TurnEvent::ToolCallCompleted { .. }))
    );
    Ok(())
}

#[tokio::test]
async fn turn_builder_into_stream_emits_activities_and_finishes() -> Result<()> {
    let core = standard_core();
    let session = core.session("turn-stream").open().await?;
    let mut stream = session.turn(TurnInput::text("stream me")).into_stream()?;

    let mut activities = Vec::new();
    while let Some(activity) = stream.next_activity().await {
        activities.push(activity?);
    }
    let result = stream.finish().await?;

    assert!(matches!(
        result.outcome,
        TurnOutcome::Finished(lash::TurnFinish::AssistantMessage { .. })
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
async fn turn_stream_finish_returns_last_assistant_prose_group() -> Result<()> {
    let core = LashCore::standard()
        .provider(semantic_group_provider())
        .model("mock-model", None)
        .max_context_tokens(200_000)
        .build()?;
    let session = core.session("turn-stream-last-group").open().await?;
    let mut stream = session
        .turn(TurnInput::text("stream groups"))
        .into_stream()?;

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
    let core = LashCore::standard()
        .provider(semantic_group_provider())
        .model("mock-model", None)
        .max_context_tokens(200_000)
        .build()?;
    let session = core.session("turn-run-last-group").open().await?;

    let collected = session.turn(TurnInput::text("run groups")).run().await?;

    assert_eq!(collected.assistant_messages(), vec!["first", "second"]);
    assert_eq!(collected.assistant_transcript_text(), "firstsecond");
    assert_eq!(collected.result.assistant_message(), Some("second"));
    Ok(())
}

#[tokio::test]
async fn retry_status_streams_as_semantic_turn_event() -> Result<()> {
    let core = LashCore::standard()
        .provider(retry_once_provider())
        .model("mock-model", None)
        .max_context_tokens(200_000)
        .build()?;
    let session = core.session("retry-status").open().await?;
    let events = RecordingEvents::default();

    let result = session
        .turn(TurnInput::text("hello"))
        .stream(&events)
        .await?;

    assert!(matches!(
        result.outcome,
        TurnOutcome::Finished(lash::TurnFinish::AssistantMessage { .. })
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
async fn plugin_surface_streams_as_semantic_turn_event() -> Result<()> {
    let core = LashCore::standard()
        .provider(mock_provider())
        .model("mock-model", None)
        .plugin(Arc::new(SurfacePluginFactory))
        .max_context_tokens(200_000)
        .build()?;
    let session = core.session("plugin-surface").open().await?;
    let events = RecordingEvents::default();

    session
        .turn(TurnInput::text("hello"))
        .stream(&events)
        .await?;

    let surface = events
        .snapshot()
        .await
        .into_iter()
        .find(|event| matches!(&event.event, TurnEvent::PluginSurface { .. }))
        .expect("plugin surface event");
    let TurnEvent::PluginSurface { plugin_id, event } = surface.event else {
        unreachable!();
    };
    assert_eq!(plugin_id, "surface_test");
    assert!(matches!(
        event,
        lash::PluginSurfaceEvent::Status { key, label, .. }
        if key == "surface" && label == "working"
    ));
    Ok(())
}

#[tokio::test]
async fn control_turn_accepts_prebuilt_turn_input() -> Result<()> {
    let core = standard_core();
    let session = core.session("raw-turn").open().await?;
    let mut input = TurnInput::text("raw input");
    input.trace_turn_id = Some("host-trace-id".to_string());

    let result = session.turn(input).run().await?;

    assert_eq!(assistant_prose(&result.activities), "echo: raw input");
    Ok(())
}

#[tokio::test]
async fn queued_input_acceptance_streams_semantic_ack_with_id() -> Result<()> {
    let (entered_tx, entered_rx) = oneshot::channel();
    let (release_tx, release_rx) = oneshot::channel();
    let core = LashCore::standard()
        .provider(checkpoint_gated_provider(entered_tx, release_rx))
        .model("mock-model", None)
        .max_context_tokens(200_000)
        .build()?;
    let session = core.session("queued-input").open().await?;
    let events = Arc::new(RecordingEvents::default());
    let turn_session = session.clone();
    let turn_events = Arc::clone(&events);
    let turn = tokio::spawn(async move {
        turn_session
            .turn(TurnInput::text("hello"))
            .stream(turn_events.as_ref())
            .await
    });

    entered_rx.await.expect("provider entered first call");
    session
        .queue(TurnInput::text("queued follow-up"))
        .id("queue-1")
        .await?;
    release_tx.send(()).expect("release provider");
    let result = turn.await.expect("turn task")?;

    assert!(matches!(
        result.outcome,
        TurnOutcome::Finished(lash::TurnFinish::AssistantMessage { .. })
    ));
    let events = events.snapshot().await;
    assert!(events.iter().any(|event| matches!(
        &event.event,
        TurnEvent::QueuedInputAccepted {
            checkpoint: lash::CheckpointKind::BeforeCompletion,
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
async fn turn_stream_receives_semantic_activities() -> Result<()> {
    let core = standard_core();
    let session = core.session("semantic-stream").open().await?;
    let turn_events = RecordingEvents::default();

    let result = session
        .turn(TurnInput::text("semantic stream"))
        .cancel(CancellationToken::new())
        .stream(&turn_events)
        .await?;

    assert!(matches!(
        result.outcome,
        TurnOutcome::Finished(lash::TurnFinish::AssistantMessage { .. })
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
async fn session_operations_delegate_to_runtime() -> Result<()> {
    let core = standard_core();
    let session = core.session("session-ops").open().await?;

    session.run(TurnInput::text("usage")).await?;
    let usage = session.usage_report();
    assert_eq!(usage.usage.output_tokens, 2);
    session.control().tools().refresh_surface().await?;
    session.control().state().await_background_work().await?;
    assert!(
        session
            .control()
            .state()
            .list_background_tasks()
            .await?
            .is_empty()
    );
    assert!(
        session
            .control()
            .state()
            .snapshot_execution()
            .await?
            .is_none()
    );
    session
        .control()
        .state()
        .restore_execution(&[1, 2, 3])
        .await?;
    Ok(())
}

#[tokio::test]
async fn observation_reads_do_not_wait_for_active_turn() -> Result<()> {
    let (entered_tx, entered_rx) = oneshot::channel();
    let (release_tx, release_rx) = oneshot::channel();
    let core = LashCore::standard()
        .provider(checkpoint_gated_provider(entered_tx, release_rx))
        .model("mock-model", None)
        .max_context_tokens(200_000)
        .tools(Arc::new(AppTools))
        .advanced()
        .session_task_executor(Arc::new(TokioSessionTaskExecutor::default()))
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
        let _ = session.control().state().list_background_tasks().await?;
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
async fn observation_updates_after_completed_turn() -> Result<()> {
    let core = standard_core();
    let session = core.session("observation-after-turn").open().await?;

    assert!(session.read_view().messages().is_empty());
    session.run(TurnInput::text("hello observation")).await?;

    let observed = session.observe();
    assert_eq!(observed.read_view().messages().len(), 2);
    assert_eq!(observed.usage_report().usage.output_tokens, 2);
    assert_eq!(observed.policy_snapshot().model, "mock-model");
    Ok(())
}

#[tokio::test]
async fn config_and_tool_mutations_publish_observation_immediately() -> Result<()> {
    let core = LashCore::standard()
        .provider(mock_provider())
        .model("mock-model", None)
        .max_context_tokens(200_000)
        .tools(Arc::new(AppTools))
        .build()?;
    let session = core.session("observation-mutations").open().await?;

    session
        .control()
        .config()
        .set_prompt_template(PromptTemplate::new(vec![
            lash::PromptTemplateSection::untitled(vec![lash::PromptTemplateEntry::text("updated")]),
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
            .and_then(|spec| spec.definition().availability_override),
        Some(ToolAvailability::Off)
    );
    Ok(())
}

#[tokio::test]
async fn child_session_snapshot_does_not_wait_for_child_turn() -> Result<()> {
    let (entered_tx, entered_rx) = oneshot::channel();
    let (release_tx, release_rx) = oneshot::channel();
    let core = LashCore::standard()
        .provider(checkpoint_gated_provider(entered_tx, release_rx))
        .model("mock-model", None)
        .max_context_tokens(200_000)
        .advanced()
        .session_task_executor(Arc::new(TokioSessionTaskExecutor::default()))
        .build()?;
    let session = core.session("child-observation-parent").open().await?;
    let children = session.control().children();
    children
        .create_session(SessionCreateRequest {
            session_id: Some("child-observation".to_string()),
            relation: lash::SessionRelation::Child {
                parent_session_id: "child-observation-parent".to_string(),
            },
            start: lash::SessionStartPoint::Empty,
            policy: None,
            plugin_mode: lash::SessionPluginMode::InheritCurrent,
            initial_nodes: Vec::new(),
            first_turn_input: None,
            tool_access: lash::SessionToolAccess::default(),
            subagent: None,
            context_surface: lash::SessionContextSurface::default(),
            mode_extras: lash::ModeExtras::default(),
            usage_source: None,
        })
        .await?;
    let handle = children
        .start_turn("child-observation", TurnInput::text("blocked child"))
        .await?;

    entered_rx.await.expect("child provider entered");
    let host = session.control().state().session_manager().await?;
    let snapshot = tokio::time::timeout(std::time::Duration::from_millis(50), async {
        host.snapshot_session("child-observation").await
    })
    .await
    .expect("child snapshot should not wait for the child turn")?;
    assert_eq!(snapshot.session_id, "child-observation");

    release_tx.send(()).expect("release child provider");
    children.await_turn(&handle.turn_id).await?;
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
            relation: lash::SessionRelation::Child {
                parent_session_id: "parent-control".to_string(),
            },
            start: lash::SessionStartPoint::Empty,
            policy: None,
            plugin_mode: lash::SessionPluginMode::InheritCurrent,
            initial_nodes: Vec::new(),
            first_turn_input: None,
            tool_access: lash::SessionToolAccess::default(),
            subagent: None,
            context_surface: lash::SessionContextSurface::default(),
            mode_extras: lash::ModeExtras::default(),
            usage_source: None,
        })
        .await?;

    let handle = children
        .start_turn(&child.session_id, TurnInput::text("child"))
        .await?;
    let assembled = children.await_turn(&handle.turn_id).await?;
    assert_eq!(assembled.state.session_id, "child-control");
    children.close_session(&child.session_id).await?;
    Ok(())
}

#[tokio::test]
async fn prompt_layers_apply_across_core_session_turn_and_mutation_scopes() -> Result<()> {
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let core = LashCore::standard()
        .provider(recording_prompt_provider(Arc::clone(&seen)))
        .model("mock-model", None)
        .max_context_tokens(200_000)
        .prompt_contribution(PromptContribution::guidance("Core", "core guidance"))
        .build()?;
    let session = core
        .session("prompt-api")
        .prompt_contribution(PromptContribution::guidance("Session", "session guidance"))
        .open()
        .await?;

    session
        .turn(TurnInput::text("first"))
        .prompt_contribution(PromptContribution::guidance("Turn", "turn guidance"))
        .run()
        .await?;
    session
        .control()
        .config()
        .replace_prompt_slot(
            PromptSlot::Guidance,
            [PromptContribution::guidance(
                "Replacement",
                "replacement guidance",
            )],
        )
        .await?;
    session.run(TurnInput::text("second")).await?;
    session
        .control()
        .config()
        .clear_prompt_slot(PromptSlot::Guidance)
        .await?;
    session.run(TurnInput::text("third")).await?;

    let prompts = seen.lock().expect("seen prompts");
    assert!(prompts[0].contains("core guidance"));
    assert!(prompts[0].contains("session guidance"));
    assert!(prompts[0].contains("turn guidance"));
    assert!(prompts[1].contains("replacement guidance"));
    assert!(!prompts[1].contains("core guidance"));
    assert!(!prompts[1].contains("session guidance"));
    assert!(!prompts[2].contains("core guidance"));
    assert!(!prompts[2].contains("replacement guidance"));
    Ok(())
}

#[tokio::test]
async fn provider_overrides_apply_at_core_session_turn_and_config_scopes() -> Result<()> {
    let core = LashCore::standard()
        .provider(text_provider("core-provider", "core-model", "core"))
        .model("core-model", None)
        .max_context_tokens(200_000)
        .build()
        .expect("standard core");
    let session = core
        .session("main")
        .provider(text_provider(
            "session-provider",
            "session-model",
            "session",
        ))
        .open()
        .await?;

    let session_result = session.run(TurnInput::text("hello")).await?;
    assert_eq!(assistant_prose(&session_result.activities), "session");

    let turn_result = session
        .turn(TurnInput::text("hello"))
        .provider(text_provider("turn-provider", "turn-model", "turn"))
        .run()
        .await?;
    assert_eq!(assistant_prose(&turn_result.activities), "turn");

    let after_turn = session.run(TurnInput::text("hello")).await?;
    assert_eq!(assistant_prose(&after_turn.activities), "session");

    session
        .control()
        .config()
        .update(SessionConfigPatch {
            provider: Some(text_provider(
                "updated-provider",
                "updated-model",
                "updated",
            )),
            model: Some(ModelSelection::new("updated-model", None)),
            ..SessionConfigPatch::default()
        })
        .await?;

    let updated = session.run(TurnInput::text("hello")).await?;
    assert_eq!(assistant_prose(&updated.activities), "updated");
    Ok(())
}

#[tokio::test]
async fn provider_only_overrides_use_provider_default_model_and_variant() -> Result<()> {
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let core = LashCore::standard()
        .provider(recording_text_provider(
            "core-provider",
            "core-model",
            Some("core-variant"),
            "core",
            Arc::clone(&seen),
        ))
        .max_context_tokens(200_000)
        .build()
        .expect("standard core");
    let session = core
        .session("main")
        .provider(recording_text_provider(
            "session-provider",
            "session-model",
            Some("session-variant"),
            "session",
            Arc::clone(&seen),
        ))
        .open()
        .await?;

    session.run(TurnInput::text("hello")).await?;
    session
        .turn(TurnInput::text("hello"))
        .provider(recording_text_provider(
            "turn-provider",
            "turn-model",
            Some("turn-variant"),
            "turn",
            Arc::clone(&seen),
        ))
        .run()
        .await?;
    session
        .turn(TurnInput::text("hello"))
        .provider(recording_text_provider(
            "manual-provider",
            "manual-default-model",
            Some("turn-variant"),
            "manual",
            Arc::clone(&seen),
        ))
        .model("manual-model", Some("manual-variant".to_string()))
        .run()
        .await?;
    session
        .control()
        .config()
        .update(SessionConfigPatch {
            provider: Some(recording_text_provider(
                "updated-provider",
                "updated-model",
                Some("updated-variant"),
                "updated",
                Arc::clone(&seen),
            )),
            ..SessionConfigPatch::default()
        })
        .await?;
    session.run(TurnInput::text("hello")).await?;

    assert_eq!(
        *seen.lock().expect("seen requests"),
        vec![
            (
                "session-model".to_string(),
                Some("session-variant".to_string())
            ),
            ("turn-model".to_string(), Some("turn-variant".to_string())),
            (
                "manual-model".to_string(),
                Some("manual-variant".to_string())
            ),
            (
                "updated-model".to_string(),
                Some("updated-variant".to_string())
            ),
        ]
    );
    Ok(())
}

#[tokio::test]
async fn embedded_sessions_always_expose_tool_state() -> Result<()> {
    let core = standard_core();
    let session = core.session("dynamic-default").open().await?;

    let state = session.control().tools().state().await?;

    assert!(state.generation() > 0);
    Ok(())
}

#[tokio::test]
async fn registered_static_tools_appear_in_tool_state() -> Result<()> {
    let core = LashCore::standard()
        .provider(mock_provider())
        .model("mock-model", None)
        .max_context_tokens(200_000)
        .tools(Arc::new(AppTools))
        .build()?;
    let session = core.session("static-tools").open().await?;

    let state = session.control().tools().state().await?;

    assert!(state.contains("app_lookup"));
    Ok(())
}

#[tokio::test]
async fn apply_tool_state_and_availability_update_live_catalog() -> Result<()> {
    let core = LashCore::standard()
        .provider(mock_provider())
        .model("mock-model", None)
        .max_context_tokens(200_000)
        .tools(Arc::new(AppTools))
        .build()?;
    let session = core.session("tool-state").open().await?;

    let generation = session
        .control()
        .tools()
        .set_availability_many(&[("app_lookup", ToolAvailability::Showcased)])
        .await?;
    let showcased = session.control().tools().state().await?;

    assert_eq!(showcased.generation(), generation);
    assert_eq!(
        showcased
            .get("app_lookup")
            .and_then(|spec| spec.definition().availability_override),
        Some(ToolAvailability::Showcased)
    );

    let generation = session
        .control()
        .tools()
        .clear_availability_override("app_lookup")
        .await?;
    let cleared = session.control().tools().state().await?;

    assert_eq!(cleared.generation(), generation);
    assert_eq!(
        cleared
            .get("app_lookup")
            .and_then(|spec| spec.definition().availability_override),
        None
    );

    let generation = session
        .control()
        .tools()
        .set_availability("app_lookup", ToolAvailability::Off)
        .await?;
    let off = session.control().tools().state().await?;

    assert_eq!(off.generation(), generation);
    assert_eq!(
        off.get("app_lookup")
            .and_then(|spec| spec.definition().availability_override),
        Some(ToolAvailability::Off)
    );

    let mut callable = off;
    callable
        .set_availability("app_lookup", Some(ToolAvailability::Callable))
        .expect("app tool");
    let generation = session
        .control()
        .tools()
        .advanced()
        .apply_state(callable)
        .await?;
    let callable = session.control().tools().state().await?;

    assert_eq!(callable.generation(), generation);
    assert_eq!(
        callable
            .get("app_lookup")
            .and_then(|spec| spec.definition().availability_override),
        Some(ToolAvailability::Callable)
    );
    Ok(())
}

#[tokio::test]
async fn persisted_session_restores_tool_state() -> Result<()> {
    let core = LashCore::standard()
        .provider(mock_provider())
        .model("mock-model", None)
        .max_context_tokens(200_000)
        .tools(Arc::new(AppTools))
        .build()?;
    let session = core.session("persisted-tools").open().await?;
    session
        .control()
        .tools()
        .set_availability("app_lookup", ToolAvailability::Off)
        .await?;
    let persisted_tool_state = session.control().tools().state().await?;
    let state = PersistedSessionState {
        session_id: "persisted-tools".to_string(),
        policy: lash::SessionPolicy {
            provider: mock_provider(),
            model: "mock-model".to_string(),
            max_context_tokens: Some(200_000),
            execution_mode: lash::ExecutionMode::standard(),
            ..Default::default()
        },
        tool_state_snapshot: Some(persisted_tool_state),
        ..Default::default()
    };
    let store: Arc<dyn lash::RuntimePersistence> = Arc::new(SnapshotStore::with_state(state));
    let reopened_core = LashCore::standard()
        .provider(mock_provider())
        .model("mock-model", None)
        .max_context_tokens(200_000)
        .tools(Arc::new(AppTools))
        .store_factory(Arc::new(ReusableStoreFactory { store }))
        .build()?;

    let reopened = reopened_core.session("persisted-tools").open().await?;
    let state = reopened.control().tools().state().await?;

    assert_eq!(
        state
            .get("app_lookup")
            .and_then(|spec| spec.definition().availability_override),
        Some(ToolAvailability::Off)
    );
    Ok(())
}

#[tokio::test]
async fn rlm_core_opens_rlm_session_and_rejects_standard_session() -> Result<()> {
    let core = LashCore::rlm()
        .provider(mock_provider())
        .model("mock-model", None)
        .max_context_tokens(200_000)
        .build()?;

    let rlm = core.session("rlm").open().await?;
    assert_eq!(rlm.mode(), &ModeId::rlm());

    let err = match core.session("standard").standard().open().await {
        Ok(_) => panic!("standard mode should not be installed"),
        Err(err) => err,
    };
    assert!(matches!(err, EmbedError::ModeNotInstalled { mode } if mode == ModeId::standard()));
    Ok(())
}

#[tokio::test]
async fn rlm_projection_errors_surface_from_mode_extensions() -> Result<()> {
    use lash_mode_rlm::{RlmProjectedBindings, RlmTurnInputExt};

    let core = LashCore::rlm()
        .provider(mock_provider())
        .model("mock-model", None)
        .max_context_tokens(200_000)
        .build()?;
    let session = core.session("rlm").open().await?;
    session
        .control()
        .mode()
        .apply_session_extension(lash_mode_rlm::rlm_session_projection_extension(
            RlmProjectedBindings::new()
                .bind_json("current_query", serde_json::json!("session"))
                .expect("session bind"),
        ))
        .await?;

    let input = TurnInput::text("hello")
        .rlm_project(
            RlmProjectedBindings::new()
                .bind_json("current_query", serde_json::json!("turn"))
                .expect("turn bind"),
        )
        .map_err(|err| EmbedError::Session(SessionError::Protocol(err.to_string())))?;
    let err = match session.turn(input).run().await {
        Ok(_) => panic!("duplicate session and turn projection should fail"),
        Err(err) => err,
    };
    assert!(
        matches!(err, EmbedError::Session(message) if message.to_string().contains("current_query"))
    );
    Ok(())
}

#[tokio::test]
async fn explicit_dual_mode_install_allows_standard_parent_and_rlm_child() -> Result<()> {
    let core = LashCore::builder()
        .install_mode(ModePreset::standard())
        .install_mode(ModePreset::rlm())
        .default_mode(ModeId::standard())
        .provider(mock_provider())
        .model("mock-model", None)
        .max_context_tokens(200_000)
        .build()?;

    let parent = core.session("main").standard().open().await?;
    let child = core.session("child").rlm().parent("main").open().await?;

    assert_eq!(parent.mode(), &ModeId::standard());
    assert_eq!(child.mode(), &ModeId::rlm());
    assert_eq!(child.parent_session_id(), Some("main"));
    Ok(())
}

#[tokio::test]
async fn uninstalled_mode_fails_at_open_time() -> Result<()> {
    let core = standard_core();
    let err = match core.session("rlm").rlm().open().await {
        Ok(_) => panic!("rlm mode should not be installed"),
        Err(err) => err,
    };
    assert!(matches!(err, EmbedError::ModeNotInstalled { mode } if mode == ModeId::rlm()));
    Ok(())
}

#[tokio::test]
async fn run_collects_ordered_assistant_prose_activity() -> Result<()> {
    let core = standard_core();
    let session = core.session("main").open().await?;

    let result = session.run(TurnInput::text("visible")).await?;

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
        TurnOutcome::Finished(lash::TurnFinish::AssistantMessage { .. })
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
                code: "x = (call app_lookup {})?".to_string(),
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
                result: serde_json::json!({ "ok": true }),
                success: true,
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
            },
        ))
        .await;

    let activities = collector.snapshot();
    assert_eq!(activities.len(), 3);
    assert!(matches!(
        &activities[0].event,
        TurnEvent::CodeBlockStarted { language, code }
            if language == "lashlang" && code == "x = (call app_lookup {})?"
    ));
    assert!(matches!(
        &activities[1].event,
        TurnEvent::ToolCallCompleted { name, result, .. }
            if name == "app_lookup" && *result == serde_json::json!({ "ok": true })
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
    let core = LashCore::standard()
        .provider(tool_roundtrip_provider())
        .model("mock-model", None)
        .tools(Arc::new(AppTools))
        .max_context_tokens(200_000)
        .build()?;
    let session = core.session("fanout-tool-events").open().await?;

    let output = session
        .turn(TurnInput::text("use tool"))
        .collect_with(live.as_ref())
        .await?;

    assert!(matches!(
        output.result.outcome,
        TurnOutcome::Finished(lash::TurnFinish::AssistantMessage { .. })
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
        TurnEvent::ToolCallCompleted { name, result, .. }
            if name == "app_lookup" && *result == serde_json::json!({ "ok": true })
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
        .stream(&events)
        .await?;

    assert!(matches!(
        result.outcome,
        TurnOutcome::Finished(lash::TurnFinish::AssistantMessage { .. })
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
    let core = LashCore::standard()
        .provider(tool_roundtrip_provider())
        .model("mock-model", None)
        .tools(Arc::new(AppTools))
        .max_context_tokens(200_000)
        .build()?;
    let session = core.session("tool-events").open().await?;
    let events = RecordingEvents::default();

    let collected = session
        .turn(TurnInput::text("use tool"))
        .stream(&events)
        .await?;

    assert!(matches!(
        collected.outcome,
        TurnOutcome::Finished(lash::TurnFinish::AssistantMessage { .. })
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
    let TurnEvent::ToolCallCompleted { result, .. } = &events[completed].event else {
        unreachable!();
    };
    assert_eq!(*result, serde_json::json!({ "ok": true }));
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

#[tokio::test]
async fn rlm_tool_calls_stream_from_live_exec_boundary() -> Result<()> {
    let core = LashCore::rlm()
        .provider(queued_text_provider(vec![
            "```lashlang\nvalue = (call app_lookup {})?\nsubmit \"done\"\n```",
        ]))
        .model("mock-model", None)
        .tools(Arc::new(AppTools))
        .max_context_tokens(200_000)
        .build()?;
    let session = core.session("rlm-live-tool-events").open().await?;
    let events = Arc::new(RecordingEvents::default());

    let result = session
        .turn(TurnInput::text("use tool"))
        .stream(events.as_ref())
        .await?;

    assert!(matches!(
        result.outcome,
        TurnOutcome::Finished(lash::TurnFinish::SubmittedValue { .. })
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

    let TurnEvent::ToolCallCompleted { result, .. } = &events[tool_completed].event else {
        unreachable!();
    };
    assert_eq!(*result, serde_json::json!({ "ok": true }));
    let TurnEvent::CodeBlockCompleted {
        language,
        success,
        error,
        ..
    } = &events[code_completed].event
    else {
        unreachable!();
    };
    assert_eq!(language, "lashlang");
    assert!(*success);
    assert!(error.is_none());
    let TurnEvent::SubmittedValue { value } = &events[terminal_output].event else {
        unreachable!();
    };
    assert_eq!(value, &serde_json::json!("done"));
    Ok(())
}

#[tokio::test]
async fn prose_or_submit_rlm_completion_emits_no_terminal_output() -> Result<()> {
    let core = LashCore::rlm()
        .provider(queued_text_provider(vec!["done in prose"]))
        .model("mock-model", None)
        .max_context_tokens(200_000)
        .build()?;
    let session = core.session("rlm-prose-completion").open().await?;
    let events = Arc::new(RecordingEvents::default());

    let result = session
        .turn(TurnInput::text("answer directly"))
        .allow_prose_or_submit()?
        .stream(events.as_ref())
        .await?;

    assert!(matches!(
        result.outcome,
        TurnOutcome::Finished(lash::TurnFinish::AssistantMessage { .. })
    ));
    let events = events.snapshot().await;
    assert!(!events.iter().any(|event| matches!(
        &event.event,
        TurnEvent::SubmittedValue { .. } | TurnEvent::ToolValue { .. }
    )));
    assert_eq!(assistant_prose(&events), "done in prose");
    Ok(())
}

#[tokio::test]
async fn submit_required_rlm_completion_emits_terminal_output() -> Result<()> {
    let core = LashCore::rlm()
        .provider(queued_text_provider(vec![
            "```lashlang\nsubmit \"done via submit\"\n```",
        ]))
        .model("mock-model", None)
        .max_context_tokens(200_000)
        .build()?;
    let session = core
        .session("rlm-submit-required-completion")
        .open()
        .await?;
    let events = Arc::new(RecordingEvents::default());

    let result = session
        .turn(TurnInput::text("submit"))
        .require_submit()?
        .stream(events.as_ref())
        .await?;

    assert!(matches!(
        result.outcome,
        TurnOutcome::Finished(lash::TurnFinish::SubmittedValue { .. })
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
async fn tool_completed_activity_is_canonical_while_model_observation_is_projected() -> Result<()> {
    let projection = Arc::new(lash::BuiltinToolResultProjectionPluginFactory::new(
        lash::ToolResultProjectionPluginConfig {
            mode: lash::ToolResultProjectionMode::Bytes,
            limit: 12,
            max_lines: 4,
        },
    ));
    let observed_tool_results = Arc::new(TokioMutex::new(Vec::<String>::new()));
    let observed_tool_results_provider = Arc::clone(&observed_tool_results);
    let responses = Arc::new(TokioMutex::new(VecDeque::from([
        LlmResponse {
            parts: vec![LlmOutputPart::ToolCall {
                call_id: "call-1".to_string(),
                tool_name: "app_lookup".to_string(),
                input_json: "{}".to_string(),
                item_id: None,
                signature: None,
            }],
            ..LlmResponse::default()
        },
        LlmResponse {
            full_text: "done".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "done".to_string(),
                response_meta: None,
            }],
            ..LlmResponse::default()
        },
    ])));
    let standard_provider = lash::testing::TestProvider::builder()
        .kind("embed-test")
        .default_model("mock-model")
        .complete(move |request| {
            let observed_tool_results = Arc::clone(&observed_tool_results_provider);
            let responses = Arc::clone(&responses);
            async move {
                for message in &request.messages {
                    for block in message.blocks.iter() {
                        if let LlmContentBlock::ToolResult { content, .. } = block {
                            observed_tool_results.lock().await.push(content.clone());
                        }
                    }
                }
                Ok(responses.lock().await.pop_front().expect("queued response"))
            }
        })
        .build()
        .into_handle();
    let standard_core = LashCore::standard()
        .provider(standard_provider)
        .model("mock-model", None)
        .tools(Arc::new(LongTextTools))
        .plugin(projection.clone())
        .max_context_tokens(200_000)
        .build()?;
    let standard_session = standard_core.session("standard-projection").open().await?;
    let standard_events = RecordingEvents::default();
    let _ = standard_session
        .turn(TurnInput::text("use tool"))
        .stream(&standard_events)
        .await?;
    let standard_view = standard_events
        .snapshot()
        .await
        .into_iter()
        .find_map(|event| match event.event {
            TurnEvent::ToolCallCompleted { result, .. } => Some(result),
            _ => None,
        })
        .expect("standard tool completion");
    assert_eq!(
        standard_view,
        serde_json::json!("abcdefghijklmnopqrstuvwxyz0123456789")
    );
    let observed = observed_tool_results.lock().await;
    let model_observation = observed
        .iter()
        .find(|content| content.contains("bytes truncated"))
        .expect("projected model observation");
    assert!(model_observation.contains("Full output saved to:"));

    let rlm_core = LashCore::rlm()
        .provider(queued_text_provider(vec![
            "```lashlang\nvalue = (call app_lookup {})?\nsubmit \"done\"\n```",
        ]))
        .model("mock-model", None)
        .tools(Arc::new(LongTextTools))
        .plugin(projection)
        .max_context_tokens(200_000)
        .build()?;
    let rlm_session = rlm_core.session("rlm-projection").open().await?;
    let rlm_events = RecordingEvents::default();
    let _ = rlm_session
        .turn(TurnInput::text("use tool"))
        .stream(&rlm_events)
        .await?;
    let rlm_view = rlm_events
        .snapshot()
        .await
        .into_iter()
        .find_map(|event| match event.event {
            TurnEvent::ToolCallCompleted { result, .. } => Some(result),
            _ => None,
        })
        .expect("rlm tool completion");

    assert_eq!(rlm_view, standard_view);
    Ok(())
}

#[tokio::test]
async fn rlm_failed_code_emits_failed_code_completion_without_fake_tools() -> Result<()> {
    let core = LashCore::rlm()
        .provider(queued_text_provider(vec![
            "```lashlang\nthis is not valid lashlang\n```",
            "```lashlang\nsubmit \"recovered\"\n```",
        ]))
        .model("mock-model", None)
        .tools(Arc::new(AppTools))
        .max_context_tokens(200_000)
        .build()?;
    let session = core.session("rlm-failed-code-event").open().await?;
    let events = RecordingEvents::default();

    let _result = session
        .turn(TurnInput::text("bad code"))
        .stream(&events)
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

#[tokio::test]
async fn store_factory_reopens_persisted_session_state() -> Result<()> {
    let mut state = PersistedSessionState {
        session_id: "persisted".to_string(),
        policy: lash::SessionPolicy {
            provider: mock_provider(),
            model: "mock-model".to_string(),
            max_context_tokens: Some(200_000),
            execution_mode: lash::ExecutionMode::standard(),
            ..Default::default()
        },
        ..Default::default()
    };
    state.append_active_conversation_messages(&[text_message(
        lash::MessageRole::User,
        "already stored",
    )]);
    let store: Arc<dyn lash::RuntimePersistence> = Arc::new(SnapshotStore::with_state(state));
    let core = LashCore::standard()
        .provider(mock_provider())
        .model("mock-model", None)
        .max_context_tokens(200_000)
        .store_factory(Arc::new(ReusableStoreFactory { store }))
        .build()?;

    let reopened = core.session("persisted").open().await?;
    let messages = reopened.read_view().messages().to_vec();
    assert_eq!(messages.len(), 1);
    assert_eq!(message_text(&messages[0]), "already stored");
    Ok(())
}

#[tokio::test]
async fn store_session_id_mismatch_is_rejected() -> Result<()> {
    let state = PersistedSessionState {
        session_id: "actual-session".to_string(),
        policy: lash::SessionPolicy {
            provider: mock_provider(),
            model: "mock-model".to_string(),
            max_context_tokens: Some(200_000),
            execution_mode: lash::ExecutionMode::standard(),
            ..Default::default()
        },
        ..Default::default()
    };
    let store: Arc<dyn lash::RuntimePersistence> = Arc::new(SnapshotStore::with_state(state));
    let core = LashCore::standard()
        .provider(mock_provider())
        .model("mock-model", None)
        .max_context_tokens(200_000)
        .store_factory(Arc::new(ReusableStoreFactory { store }))
        .build()?;

    let err = match core.session("requested-session").open().await {
        Ok(_) => panic!("mismatched store should fail"),
        Err(err) => err,
    };

    assert!(matches!(
        err,
        EmbedError::StoreSessionMismatch {
            loaded,
            requested
        } if loaded == "actual-session" && requested == "requested-session"
    ));
    Ok(())
}

#[tokio::test]
async fn open_with_state_uses_manual_state_and_persists_tool_state() -> Result<()> {
    let mut state = PersistedSessionState {
        session_id: "manual-state".to_string(),
        policy: lash::SessionPolicy {
            provider: mock_provider(),
            model: "mock-model".to_string(),
            max_context_tokens: Some(200_000),
            execution_mode: lash::ExecutionMode::standard(),
            ..Default::default()
        },
        ..Default::default()
    };
    state.append_active_conversation_messages(&[text_message(
        lash::MessageRole::User,
        "manual input",
    )]);
    let store: Arc<dyn lash::RuntimePersistence> = Arc::new(SnapshotStore::default());
    let core = LashCore::standard()
        .provider(mock_provider())
        .model("mock-model", None)
        .max_context_tokens(200_000)
        .tools(Arc::new(AppTools))
        .build()?;

    let opened = core
        .session("manual-state")
        .store(Arc::clone(&store))
        .open_with_state(state)
        .await?;
    assert_eq!(
        message_text(&opened.read_view().messages().to_vec()[0]),
        "manual input"
    );
    opened
        .control()
        .tools()
        .set_availability("app_lookup", ToolAvailability::Off)
        .await?;
    let mut persisted = opened.control().state().persist_current().await?;
    persisted.tool_state_snapshot = Some(opened.control().tools().state().await?);
    drop(opened);

    let reopened = core
        .session("manual-state")
        .store(Arc::clone(&store))
        .open_with_state(persisted)
        .await?;
    let state = reopened.control().tools().state().await?;
    assert_eq!(
        state
            .get("app_lookup")
            .and_then(|spec| spec.definition().availability_override),
        Some(ToolAvailability::Off)
    );
    Ok(())
}

#[tokio::test]
async fn core_store_factory_is_used_for_managed_child_sessions() -> Result<()> {
    let factory = Arc::new(RecordingStoreFactory::default());
    let core = LashCore::standard()
        .provider(mock_provider())
        .model("mock-model", None)
        .max_context_tokens(200_000)
        .store_factory(factory.clone())
        .build()?;
    let session = core.session("root-with-child-store").open().await?;

    session
        .control()
        .children()
        .create_session(SessionCreateRequest {
            session_id: Some("managed-child-store".to_string()),
            relation: lash::SessionRelation::Child {
                parent_session_id: "root-with-child-store".to_string(),
            },
            start: lash::SessionStartPoint::Empty,
            policy: None,
            plugin_mode: lash::SessionPluginMode::InheritCurrent,
            initial_nodes: Vec::new(),
            first_turn_input: None,
            tool_access: lash::SessionToolAccess::default(),
            subagent: None,
            context_surface: lash::SessionContextSurface::default(),
            mode_extras: lash::ModeExtras::default(),
            usage_source: None,
        })
        .await?;

    assert_eq!(
        factory.session_ids(),
        vec![
            "root-with-child-store".to_string(),
            "managed-child-store".to_string()
        ]
    );
    Ok(())
}

#[tokio::test]
async fn explicit_root_store_keeps_configured_child_store_factory() -> Result<()> {
    let factory = Arc::new(RecordingStoreFactory::default());
    let explicit_store: Arc<dyn lash::RuntimePersistence> = Arc::new(SnapshotStore::default());
    let core = LashCore::standard()
        .provider(mock_provider())
        .model("mock-model", None)
        .max_context_tokens(200_000)
        .store_factory(factory.clone())
        .build()?;
    let session = core
        .session("explicit-root-store")
        .store(explicit_store)
        .open()
        .await?;

    session
        .control()
        .children()
        .create_session(SessionCreateRequest {
            session_id: Some("explicit-root-child".to_string()),
            relation: lash::SessionRelation::Child {
                parent_session_id: "explicit-root-store".to_string(),
            },
            start: lash::SessionStartPoint::Empty,
            policy: None,
            plugin_mode: lash::SessionPluginMode::InheritCurrent,
            initial_nodes: Vec::new(),
            first_turn_input: None,
            tool_access: lash::SessionToolAccess::default(),
            subagent: None,
            context_surface: lash::SessionContextSurface::default(),
            mode_extras: lash::ModeExtras::default(),
            usage_source: None,
        })
        .await?;

    assert_eq!(
        factory.session_ids(),
        vec!["explicit-root-child".to_string()]
    );
    Ok(())
}

#[tokio::test]
async fn explicit_session_store_takes_precedence_over_core_store_factory() -> Result<()> {
    let mut explicit_state = PersistedSessionState {
        session_id: "store-precedence".to_string(),
        policy: lash::SessionPolicy {
            provider: mock_provider(),
            model: "mock-model".to_string(),
            max_context_tokens: Some(200_000),
            execution_mode: lash::ExecutionMode::standard(),
            ..Default::default()
        },
        ..Default::default()
    };
    explicit_state.append_active_conversation_messages(&[text_message(
        lash::MessageRole::User,
        "explicit store",
    )]);
    let mut factory_state = explicit_state.clone();
    factory_state.append_active_conversation_messages(&[text_message(
        lash::MessageRole::Assistant,
        "factory store",
    )]);
    let explicit_store: Arc<dyn lash::RuntimePersistence> =
        Arc::new(SnapshotStore::with_state(explicit_state));
    let factory_store: Arc<dyn lash::RuntimePersistence> =
        Arc::new(SnapshotStore::with_state(factory_state));
    let core = LashCore::standard()
        .provider(mock_provider())
        .model("mock-model", None)
        .max_context_tokens(200_000)
        .store_factory(Arc::new(ReusableStoreFactory {
            store: factory_store,
        }))
        .build()?;

    let reopened = core
        .session("store-precedence")
        .store(explicit_store)
        .open()
        .await?;
    let messages = reopened.read_view().messages().to_vec();

    assert_eq!(messages.len(), 1);
    assert_eq!(message_text(&messages[0]), "explicit store");
    Ok(())
}

#[test]
fn turn_result_total_usage_sums_parent_and_children() {
    use lash::{
        ExecutionMode, ExecutionSummary, OutputState, SessionPolicy, SessionStateEnvelope,
        TurnFinish, TurnOutcome,
    };

    let result = TurnResult {
        state: SessionStateEnvelope {
            session_id: "s".to_string(),
            policy: SessionPolicy {
                execution_mode: ExecutionMode::standard(),
                ..Default::default()
            },
            ..Default::default()
        },
        outcome: TurnOutcome::Finished(TurnFinish::AssistantMessage {
            text: "ok".to_string(),
        }),
        assistant_output: AssistantOutput {
            safe_text: "ok".to_string(),
            raw_text: "ok".to_string(),
            state: OutputState::Usable,
        },
        usage: TokenUsage {
            input_tokens: 10,
            output_tokens: 5,
            cached_input_tokens: 2,
            reasoning_tokens: 1,
        },
        children_usage: vec![
            TokenLedgerEntry {
                source: "subagent".to_string(),
                model: "m".to_string(),
                usage: TokenUsage {
                    input_tokens: 7,
                    output_tokens: 3,
                    cached_input_tokens: 4,
                    reasoning_tokens: 0,
                },
            },
            TokenLedgerEntry {
                source: "compaction".to_string(),
                model: "m".to_string(),
                usage: TokenUsage {
                    input_tokens: 1,
                    output_tokens: 0,
                    cached_input_tokens: 0,
                    reasoning_tokens: 0,
                },
            },
        ],
        tool_calls: Vec::new(),
        execution: ExecutionSummary {
            mode: ExecutionMode::standard(),
            had_tool_calls: false,
            had_code_execution: false,
        },
        errors: Vec::new(),
    };

    let total = result.total_usage();
    assert_eq!(total.input_tokens, 10 + 7 + 1);
    assert_eq!(total.output_tokens, 5 + 3);
    assert_eq!(total.cached_input_tokens, 2 + 4);
    assert_eq!(total.reasoning_tokens, 1);
    // Parent's own usage is unchanged.
    assert_eq!(result.usage.input_tokens, 10);
}

fn text_message(role: lash::MessageRole, text: &str) -> lash::Message {
    let id = "stored-message".to_string();
    lash::Message {
        id: id.clone(),
        role,
        parts: lash::shared_parts(vec![lash::Part {
            id: format!("{id}.p0"),
            kind: lash::PartKind::Text,
            content: text.to_string(),
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            tool_item_id: None,
            tool_signature: None,
            prune_state: lash::PruneState::Intact,
            reasoning_meta: None,
            response_meta: None,
        }]),
        origin: None,
    }
}
