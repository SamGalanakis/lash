use crate::support::*;
use std::collections::VecDeque;

use lash_core::LlmOutputPart;
use lash_core::llm::transport::LlmTransportError;
use lash_core::llm::types::{
    LlmContentBlock, LlmRequest, LlmResponse, LlmRole, LlmStreamEvent, ResponseTextMeta,
};
use tokio::sync::{Mutex as TokioMutex, oneshot};

#[derive(Default)]
struct SnapshotStore {
    read: std::sync::Mutex<Option<lash_core::store::PersistedSessionRead>>,
    scopes: std::sync::Mutex<Vec<lash_core::SessionReadScope>>,
    runtime_turn_commits: std::sync::Mutex<
        std::collections::HashMap<
            (String, String),
            (String, lash_core::store::RuntimeCommitResult),
        >,
    >,
}

impl SnapshotStore {
    fn with_state(state: RuntimeSessionState) -> Self {
        let turn_state = state.turn_state();
        let config = lash_core::PersistedSessionConfig {
            provider_id: state.policy.recorded_provider_id().to_string(),
            model: state.policy.model.clone(),
        };
        Self {
            read: std::sync::Mutex::new(Some(lash_core::store::PersistedSessionRead {
                session_id: state.session_id,
                head_revision: 7,
                config,
                agent_frames: state.agent_frames,
                current_agent_frame_id: state.current_agent_frame_id,
                graph: state.session_graph,
                checkpoint_ref: None,
                checkpoint: Some(lash_core::store::HydratedSessionCheckpoint {
                    turn_state,
                    tool_state: state.tool_state_snapshot,
                    ..Default::default()
                }),
                token_ledger: Vec::new(),
            })),
            scopes: std::sync::Mutex::new(Vec::new()),
            runtime_turn_commits: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    fn scopes(&self) -> Vec<lash_core::SessionReadScope> {
        self.scopes.lock().expect("snapshot scopes lock").clone()
    }

    fn set_head_provider_id(&self, provider_id: impl Into<String>) {
        let mut read = self.read.lock().expect("snapshot store lock");
        let Some(read) = read.as_mut() else {
            panic!("snapshot store has no session head");
        };
        let provider_id = provider_id.into();
        read.config.provider_id = provider_id.clone();
        for frame in &mut read.agent_frames {
            frame.assignment.policy.provider_id = provider_id.clone();
        }
        read.head_revision += 1;
    }
}

lash_core::impl_noop_attachment_manifest!(SnapshotStore);

#[async_trait]
impl lash_core::RuntimePersistence for SnapshotStore {
    async fn load_session(
        &self,
        scope: lash_core::SessionReadScope,
    ) -> std::result::Result<
        Option<lash_core::store::PersistedSessionRead>,
        lash_core::store::StoreError,
    > {
        self.scopes
            .lock()
            .expect("snapshot scopes lock")
            .push(scope.clone());
        let mut read = self.read.lock().expect("snapshot store lock").clone();
        if let Some(read) = read.as_mut()
            && let lash_core::SessionReadScope::ActivePath { leaf_node_id } = scope
        {
            if let Some(leaf_node_id) = leaf_node_id {
                read.graph.set_leaf_node_id(Some(leaf_node_id));
            }
            read.graph = read.graph.fork_current_path();
        }
        Ok(read)
    }

    async fn load_node(
        &self,
        _node_id: &str,
    ) -> std::result::Result<Option<lash_core::SessionNodeRecord>, lash_core::store::StoreError>
    {
        Ok(None)
    }

    async fn commit_runtime_state(
        &self,
        commit: lash_core::store::RuntimeCommit,
    ) -> std::result::Result<lash_core::store::RuntimeCommitResult, lash_core::store::StoreError>
    {
        let mut read = self.read.lock().expect("snapshot store lock");
        if let Some(completed) = &commit.turn_commit {
            if completed.session_id != commit.session_id {
                return Err(lash_core::store::StoreError::RuntimeTurnCommitConflict {
                    session_id: completed.session_id.clone(),
                    turn_id: completed.turn_id.clone(),
                });
            }
            let key = (completed.session_id.clone(), completed.turn_id.clone());
            if let Some((stored_hash, result)) = self
                .runtime_turn_commits
                .lock()
                .expect("runtime turn commits lock")
                .get(&key)
                .cloned()
            {
                if stored_hash == completed.turn_commit_hash {
                    return Ok(result);
                }
                return Err(lash_core::store::StoreError::RuntimeTurnCommitConflict {
                    session_id: completed.session_id.clone(),
                    turn_id: completed.turn_id.clone(),
                });
            }
        }
        let existing_graph = read
            .as_ref()
            .map(|read| read.graph.clone())
            .unwrap_or_default();
        let graph = match commit.graph.clone() {
            lash_core::store::GraphCommitDelta::ReplaceFull(graph) => graph,
            lash_core::store::GraphCommitDelta::Unchanged { leaf_node_id } => {
                let mut graph = existing_graph;
                graph.set_leaf_node_id(leaf_node_id);
                graph
            }
            lash_core::store::GraphCommitDelta::Append {
                nodes,
                leaf_node_id,
            } => {
                let mut graph = existing_graph;
                graph.extend_node_records(nodes);
                graph.set_leaf_node_id(leaf_node_id);
                graph
            }
        };
        *read = Some(lash_core::store::PersistedSessionRead {
            session_id: commit.session_id.clone(),
            head_revision: 8,
            config: commit.config,
            agent_frames: commit.agent_frames,
            current_agent_frame_id: commit.current_agent_frame_id,
            graph,
            checkpoint_ref: Some(lash_core::BlobRef("checkpoint".to_string())),
            checkpoint: Some(commit.checkpoint),
            token_ledger: commit.usage_deltas,
        });
        let result = lash_core::store::RuntimeCommitResult {
            head_revision: 8,
            checkpoint_ref: lash_core::BlobRef("checkpoint".to_string()),
            manifest: lash_core::store::SessionCheckpoint::default(),
        };
        if let Some(completed) = &commit.turn_commit {
            self.runtime_turn_commits
                .lock()
                .expect("runtime turn commits lock")
                .insert(
                    (completed.session_id.clone(), completed.turn_id.clone()),
                    (completed.turn_commit_hash.clone(), result.clone()),
                );
        }
        Ok(result)
    }

    async fn enqueue_queued_work(
        &self,
        _batch: lash_core::runtime::QueuedWorkBatchDraft,
    ) -> std::result::Result<lash_core::runtime::QueuedWorkBatch, lash_core::store::StoreError>
    {
        Err(lash_core::store::StoreError::Backend(
            "queued work is not supported by SnapshotStore".to_string(),
        ))
    }

    async fn claim_ready_queued_work(
        &self,
        _session_id: &str,
        _owner_id: &str,
        _boundary: lash_core::runtime::QueuedWorkClaimBoundary,
        _lease_ttl_ms: u64,
        _max_batches: usize,
    ) -> std::result::Result<
        Option<lash_core::runtime::QueuedWorkClaim>,
        lash_core::store::StoreError,
    > {
        Ok(None)
    }

    async fn renew_queued_work_claim(
        &self,
        claim: &lash_core::runtime::QueuedWorkClaim,
        _lease_ttl_ms: u64,
    ) -> std::result::Result<lash_core::runtime::QueuedWorkClaim, lash_core::store::StoreError>
    {
        Err(lash_core::store::StoreError::QueuedWorkClaimExpired {
            session_id: claim.session_id.clone(),
            claim_id: claim.claim_id.clone(),
        })
    }

    async fn abandon_queued_work_claim(
        &self,
        _claim: &lash_core::runtime::QueuedWorkClaim,
    ) -> std::result::Result<(), lash_core::store::StoreError> {
        Ok(())
    }

    async fn cancel_queued_work_batch(
        &self,
        _session_id: &str,
        _batch_id: &str,
    ) -> std::result::Result<
        Option<lash_core::runtime::QueuedWorkBatch>,
        lash_core::store::StoreError,
    > {
        Ok(None)
    }

    async fn list_queued_work(
        &self,
        _session_id: &str,
    ) -> std::result::Result<Vec<lash_core::runtime::QueuedWorkBatch>, lash_core::store::StoreError>
    {
        Ok(Vec::new())
    }

    async fn save_session_meta(
        &self,
        _meta: lash_core::SessionMeta,
    ) -> std::result::Result<(), lash_core::store::StoreError> {
        Ok(())
    }

    async fn load_session_meta(
        &self,
    ) -> std::result::Result<Option<lash_core::SessionMeta>, lash_core::store::StoreError> {
        Ok(None)
    }

    async fn tombstone_nodes(
        &self,
        _ids: &[String],
    ) -> std::result::Result<(), lash_core::store::StoreError> {
        Ok(())
    }

    async fn vacuum(
        &self,
    ) -> std::result::Result<lash_core::VacuumReport, lash_core::store::StoreError> {
        Ok(lash_core::VacuumReport::default())
    }

    async fn gc_unreachable(
        &self,
    ) -> std::result::Result<lash_core::GcReport, lash_core::store::StoreError> {
        Ok(lash_core::GcReport::default())
    }
}

#[derive(Clone)]
struct ReusableStoreFactory {
    store: Arc<dyn lash_core::RuntimePersistence>,
}

#[async_trait::async_trait]
impl lash_core::SessionStoreFactory for ReusableStoreFactory {
    async fn create_store(
        &self,
        _request: &lash_core::SessionStoreCreateRequest,
    ) -> std::result::Result<Arc<dyn lash_core::RuntimePersistence>, String> {
        Ok(Arc::clone(&self.store))
    }

    async fn delete_session(&self, _session_id: &str) -> std::result::Result<(), String> {
        Ok(())
    }
}

struct BoundSessionStore {
    session_id: String,
}

lash_core::impl_noop_attachment_manifest!(BoundSessionStore);

#[async_trait]
impl lash_core::RuntimePersistence for BoundSessionStore {
    async fn load_session(
        &self,
        _scope: lash_core::SessionReadScope,
    ) -> std::result::Result<
        Option<lash_core::store::PersistedSessionRead>,
        lash_core::store::StoreError,
    > {
        Ok(None)
    }

    async fn load_node(
        &self,
        _node_id: &str,
    ) -> std::result::Result<Option<lash_core::SessionNodeRecord>, lash_core::store::StoreError>
    {
        Ok(None)
    }

    async fn commit_runtime_state(
        &self,
        _commit: lash_core::store::RuntimeCommit,
    ) -> std::result::Result<lash_core::store::RuntimeCommitResult, lash_core::store::StoreError>
    {
        unreachable!("test should fail before committing to the reused child store")
    }

    async fn save_session_meta(
        &self,
        _meta: lash_core::SessionMeta,
    ) -> std::result::Result<(), lash_core::store::StoreError> {
        Ok(())
    }

    async fn load_session_meta(
        &self,
    ) -> std::result::Result<Option<lash_core::SessionMeta>, lash_core::store::StoreError> {
        Ok(Some(lash_core::SessionMeta {
            session_id: self.session_id.clone(),
            session_name: self.session_id.clone(),
            created_at: "test".to_string(),
            model: "mock-model".to_string(),
            cwd: None,
            relation: lash_core::SessionRelation::Root,
        }))
    }

    async fn tombstone_nodes(
        &self,
        _ids: &[String],
    ) -> std::result::Result<(), lash_core::store::StoreError> {
        Ok(())
    }

    async fn vacuum(
        &self,
    ) -> std::result::Result<lash_core::VacuumReport, lash_core::store::StoreError> {
        Ok(lash_core::VacuumReport::default())
    }

    async fn gc_unreachable(
        &self,
    ) -> std::result::Result<lash_core::GcReport, lash_core::store::StoreError> {
        Ok(lash_core::GcReport::default())
    }
}

#[derive(Default)]
struct RecordingStoreFactory {
    requests: std::sync::Mutex<Vec<lash_core::SessionStoreCreateRequest>>,
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

#[async_trait::async_trait]
impl lash_core::SessionStoreFactory for RecordingStoreFactory {
    async fn create_store(
        &self,
        request: &lash_core::SessionStoreCreateRequest,
    ) -> std::result::Result<Arc<dyn lash_core::RuntimePersistence>, String> {
        self.requests
            .lock()
            .expect("recording factory lock")
            .push(request.clone());
        Ok(Arc::new(SnapshotStore::default()))
    }

    async fn delete_session(&self, _session_id: &str) -> std::result::Result<(), String> {
        Ok(())
    }
}

#[derive(Default)]
struct DeletingStoreFactory {
    stores: std::sync::Mutex<std::collections::HashMap<String, Arc<SnapshotStore>>>,
}

#[async_trait::async_trait]
impl lash_core::SessionStoreFactory for DeletingStoreFactory {
    async fn create_store(
        &self,
        request: &lash_core::SessionStoreCreateRequest,
    ) -> std::result::Result<Arc<dyn lash_core::RuntimePersistence>, String> {
        let store = self
            .stores
            .lock()
            .expect("deleting factory lock")
            .entry(request.session_id.clone())
            .or_insert_with(|| Arc::new(SnapshotStore::default()))
            .clone();
        Ok(store as Arc<dyn lash_core::RuntimePersistence>)
    }

    async fn delete_session(&self, session_id: &str) -> std::result::Result<(), String> {
        self.stores
            .lock()
            .expect("deleting factory lock")
            .remove(session_id);
        Ok(())
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
    fn tool_manifests(&self) -> Vec<lash_core::ToolManifest> {
        vec![app_tool_definition().manifest()]
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<lash_core::ToolContract>> {
        (name == "app_lookup").then(|| Arc::new(app_tool_definition().contract()))
    }

    async fn execute(&self, _call: lash_core::ToolCall<'_>) -> lash_core::ToolResult {
        lash_core::ToolResult::ok(serde_json::json!({ "ok": true }))
    }
}

struct PendingAppTools {
    key_tx: StdMutex<Option<oneshot::Sender<lash_core::AwaitEventKey>>>,
}

impl PendingAppTools {
    fn new(key_tx: oneshot::Sender<lash_core::AwaitEventKey>) -> Self {
        Self {
            key_tx: StdMutex::new(Some(key_tx)),
        }
    }
}

#[async_trait]
impl ToolProvider for PendingAppTools {
    fn tool_manifests(&self) -> Vec<lash_core::ToolManifest> {
        vec![app_tool_definition().manifest()]
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<lash_core::ToolContract>> {
        (name == "app_lookup").then(|| Arc::new(app_tool_definition().contract()))
    }

    async fn execute(&self, call: lash_core::ToolCall<'_>) -> lash_core::ToolResult {
        assert_eq!(call.name, "app_lookup");
        let key = match call.context.completion_key().await {
            Ok(key) => key,
            Err(err) => return lash_core::ToolResult::err_fmt(err),
        };
        if let Some(tx) = self.key_tx.lock().expect("pending tool key tx").take() {
            let _ = tx.send(key);
        }
        lash_core::ToolResult::pending(lash_core::PendingCompletion::new())
    }
}

struct AgentFrameSwitchTools;

#[async_trait]
impl ToolProvider for AgentFrameSwitchTools {
    fn tool_manifests(&self) -> Vec<lash_core::ToolManifest> {
        vec![agent_frame_switch_tool_definition().manifest()]
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<lash_core::ToolContract>> {
        (name == "switch_frame").then(|| Arc::new(agent_frame_switch_tool_definition().contract()))
    }

    async fn execute(&self, call: lash_core::ToolCall<'_>) -> lash_core::ToolResult {
        assert_eq!(call.name, "switch_frame");
        let task = call
            .args
            .get("task")
            .and_then(serde_json::Value::as_str)
            .expect("task arg")
            .to_string();
        lash_core::ToolResult::ok(serde_json::json!({ "ok": true })).with_control(
            lash_core::ToolControl::SwitchAgentFrame {
                frame_id: "durable-follow-frame".to_string(),
                initial_nodes: Vec::new(),
                task: Some(task),
            },
        )
    }
}

fn agent_frame_switch_tool_definition() -> lash_core::ToolDefinition {
    lash_core::ToolDefinition::raw(
        "tool:switch_frame",
        "switch_frame",
        "Switch to a fresh agent frame.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "task": { "type": "string" }
            },
            "required": ["task"],
            "additionalProperties": false
        }),
        serde_json::json!({ "type": "object" }),
    )
}

fn app_tool_definition() -> lash_core::ToolDefinition {
    lash_core::ToolDefinition::raw(
        "tool:app_lookup",
        "app_lookup",
        "Look up app state.",
        serde_json::json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false
        }),
        serde_json::json!({ "type": "object" }),
    )
    .with_lashlang_binding(lash_core::LashlangToolBinding::new(["tools"], "app_lookup"))
}

struct LongTextTools;

#[async_trait]
impl ToolProvider for LongTextTools {
    fn tool_manifests(&self) -> Vec<lash_core::ToolManifest> {
        vec![long_text_tool_definition().manifest()]
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<lash_core::ToolContract>> {
        (name == "app_lookup").then(|| Arc::new(long_text_tool_definition().contract()))
    }

    async fn execute(&self, _call: lash_core::ToolCall<'_>) -> lash_core::ToolResult {
        lash_core::ToolResult::ok(serde_json::json!("abcdefghijklmnopqrstuvwxyz0123456789"))
    }
}

fn long_text_tool_definition() -> lash_core::ToolDefinition {
    lash_core::ToolDefinition::raw(
        "tool:app_lookup",
        "app_lookup",
        "Look up verbose app state.",
        serde_json::json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false
        }),
        serde_json::json!({ "type": "string" }),
    )
    .with_lashlang_binding(lash_core::LashlangToolBinding::new(["tools"], "app_lookup"))
}

struct SurfacePluginFactory;

impl lash_core::PluginFactory for SurfacePluginFactory {
    fn id(&self) -> &'static str {
        "surface_test"
    }

    fn build(
        &self,
        _ctx: &lash_core::PluginSessionContext,
    ) -> std::result::Result<Arc<dyn lash_core::SessionPlugin>, lash_core::PluginError> {
        Ok(Arc::new(SurfacePlugin))
    }
}

struct SurfacePlugin;

impl lash_core::SessionPlugin for SurfacePlugin {
    fn id(&self) -> &'static str {
        "surface_test"
    }

    fn register(
        &self,
        reg: &mut lash_core::PluginRegistrar,
    ) -> std::result::Result<(), lash_core::PluginError> {
        reg.output().response(Arc::new(|ctx| {
            Box::pin(async move {
                Ok(lash_core::AssistantResponseTransform {
                    response: ctx.response,
                    events: vec![lash_core::PluginRuntimeEvent::Status {
                        key: "surface".to_string(),
                        label: "working".to_string(),
                        detail: Some("details".to_string()),
                    }],
                })
            })
        }));
        Ok(())
    }
}

fn mock_provider() -> ProviderHandle {
    crate::testing::TestProvider::builder()
        .kind("embed-test")
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
                usage: lash_core::llm::types::LlmUsage {
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
                replay: None,
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
    crate::testing::TestProvider::builder()
        .kind("embed-test")
        .complete(move |_request| {
            let responses = Arc::clone(&responses);
            async move { Ok(responses.lock().await.pop_front().expect("queued response")) }
        })
        .build()
        .into_handle()
}

fn agent_frame_switch_provider() -> ProviderHandle {
    let responses = Arc::new(TokioMutex::new(VecDeque::from([
        LlmResponse {
            parts: vec![LlmOutputPart::ToolCall {
                call_id: "switch-call".to_string(),
                tool_name: "switch_frame".to_string(),
                input_json: serde_json::json!({
                    "task": "finish in the next frame"
                })
                .to_string(),
                replay: None,
            }],
            ..LlmResponse::default()
        },
        text_response("done after frame switch"),
    ])));
    crate::testing::TestProvider::builder()
        .kind("embed-test")
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
    crate::testing::TestProvider::builder()
        .kind("embed-test")
        .complete(move |_request| {
            let responses = Arc::clone(&responses);
            async move { Ok(responses.lock().await.pop_front().expect("queued response")) }
        })
        .build()
        .into_handle()
}

fn semantic_group_provider() -> ProviderHandle {
    crate::testing::TestProvider::builder()
        .kind("embed-test")
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

fn text_provider(kind: &'static str, _model: &'static str, text: &'static str) -> ProviderHandle {
    crate::testing::TestProvider::builder()
        .kind(kind)
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
    _model: &'static str,
    _variant: Option<&'static str>,
    text: &'static str,
    seen: SeenModels,
) -> ProviderHandle {
    crate::testing::TestProvider::builder()
        .kind(kind)
        .supported_variants(|_| {
            &[
                "core-variant",
                "session-variant",
                "turn-variant",
                "updated-variant",
                "manual-variant",
            ]
        })
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

fn request_text(request: &LlmRequest) -> String {
    request
        .messages
        .iter()
        .flat_map(|message| message.blocks.iter())
        .filter_map(|block| match block {
            LlmContentBlock::Text { text, .. } => Some(text.as_ref()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn recording_prompt_provider(seen: Arc<std::sync::Mutex<Vec<String>>>) -> ProviderHandle {
    crate::testing::TestProvider::builder()
        .kind("prompt-test")
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

fn recording_request_provider(seen: Arc<std::sync::Mutex<Vec<String>>>) -> ProviderHandle {
    crate::testing::TestProvider::builder()
        .kind("request-test")
        .complete(move |request| {
            let seen = Arc::clone(&seen);
            async move {
                seen.lock()
                    .expect("seen prompts")
                    .push(request_text(&request));
                Ok(text_response("```lashlang\nsubmit \"ok\"\n```"))
            }
        })
        .build()
        .into_handle()
}

fn retry_once_provider() -> ProviderHandle {
    let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    crate::testing::TestProvider::builder()
        .kind("retry-test")
        .requires_streaming(true)
        .options(lash_core::ProviderOptions {
            reliability: lash_core::provider::ProviderReliability::default()
                .max_attempts(2)
                .base_delay_ms(0)
                .max_delay_ms(0),
            ..lash_core::ProviderOptions::default()
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
    crate::testing::TestProvider::builder()
        .kind("checkpoint-gated")
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
    explicit_ephemeral_facets(LashCore::standard())
        .provider(mock_provider())
        .model(mock_model_spec())
        .build()
        .expect("standard core")
}

fn inline_scope(scope: lash_core::ExecutionScope) -> lash_core::ScopedEffectController<'static> {
    lash_core::ScopedEffectController::shared(
        Arc::new(lash_core::InlineRuntimeEffectController),
        scope,
    )
    .expect("inline execution scope")
}

fn turn_scope(session_id: &str) -> lash_core::ScopedEffectController<'static> {
    inline_scope(lash_core::ExecutionScope::turn(
        session_id,
        lash_core::TurnActivityId::fresh().0,
    ))
}

fn runtime_operation_scope(
    scope_id: impl Into<String>,
) -> lash_core::ScopedEffectController<'static> {
    inline_scope(lash_core::ExecutionScope::runtime_operation(scope_id))
}

fn session_delete_scope(session_id: &str) -> lash_core::ScopedEffectController<'static> {
    inline_scope(lash_core::ExecutionScope::session_delete(session_id))
}

fn explicit_ephemeral_facets(
    builder: crate::core::LashCoreBuilder,
) -> crate::core::LashCoreBuilder {
    builder
        .effect_host(Arc::new(crate::durability::InlineEffectHost::default()))
        .lashlang_artifact_store(Arc::new(
            crate::persistence::InMemoryLashlangArtifactStore::new(),
        ))
        .attachment_store(Arc::new(crate::persistence::InMemoryAttachmentStore::new()))
}

fn text_message(role: lash_core::MessageRole, text: &str) -> lash_core::Message {
    let id = "stored-message".to_string();
    lash_core::Message {
        id: id.clone(),
        role,
        parts: lash_core::shared_parts(vec![lash_core::Part {
            id: format!("{id}.p0"),
            kind: lash_core::PartKind::Text,
            content: text.to_string(),
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            tool_replay: None,
            prune_state: lash_core::PruneState::Intact,
            reasoning_meta: None,
            response_meta: None,
        }]),
        origin: None,
    }
}

mod control_admin;
mod core_session_builder;
mod harness;
use harness::{mock_model_spec, model_spec, run_async_test_on_stack_budget};
mod lash_e2e;
mod plugin_stack;
mod processes_endstate;
mod rebuild_conformance;
mod stack_budget;
mod turn_streaming;
