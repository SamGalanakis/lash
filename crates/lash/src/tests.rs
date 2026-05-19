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
    read: std::sync::Mutex<Option<lash_core::PersistedSessionRead>>,
    scopes: std::sync::Mutex<Vec<lash_core::SessionReadScope>>,
    runtime_turn_checkpoints: std::sync::Mutex<
        std::collections::HashMap<(String, String), lash_core::RuntimeTurnCheckpoint>,
    >,
    runtime_effect_journal: std::sync::Mutex<
        std::collections::HashMap<(String, String, String), lash_core::RuntimeEffectJournalRecord>,
    >,
    runtime_turn_leases:
        std::sync::Mutex<std::collections::HashMap<(String, String), lash_core::RuntimeTurnLease>>,
}

impl SnapshotStore {
    fn with_state(state: PersistedSessionState) -> Self {
        let turn_state = state.turn_state();
        let config = lash_core::PersistedSessionConfig {
            provider_id: state.policy.provider.kind().to_string(),
            configured_model: state.policy.model.clone(),
            context_window: state.policy.max_context_tokens.unwrap_or_default() as u64,
            execution_mode: state.policy.execution_mode.clone(),
            standard_context_approach: state.policy.standard_context_approach.clone(),
            model_variant: state.policy.model_variant.clone(),
        };
        Self {
            read: std::sync::Mutex::new(Some(lash_core::PersistedSessionRead {
                session_id: state.session_id,
                head_revision: 7,
                config,
                graph: state.session_graph,
                checkpoint_ref: None,
                checkpoint: Some(lash_core::HydratedSessionCheckpoint {
                    turn_state,
                    tool_state: state.tool_state_snapshot,
                    ..Default::default()
                }),
                token_ledger: Vec::new(),
            })),
            scopes: std::sync::Mutex::new(Vec::new()),
            runtime_turn_checkpoints: std::sync::Mutex::new(std::collections::HashMap::new()),
            runtime_effect_journal: std::sync::Mutex::new(std::collections::HashMap::new()),
            runtime_turn_leases: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    fn scopes(&self) -> Vec<lash_core::SessionReadScope> {
        self.scopes.lock().expect("snapshot scopes lock").clone()
    }

    fn ensure_lease(
        &self,
        lease: &lash_core::RuntimeTurnLease,
    ) -> std::result::Result<(), lash_core::store::StoreError> {
        self.runtime_turn_leases
            .lock()
            .expect("runtime turn leases lock")
            .get(&(lease.session_id.clone(), lease.turn_id.clone()))
            .filter(|current| {
                current.lease_token == lease.lease_token
                    && current.expires_at_epoch_ms > test_current_epoch_ms()
            })
            .ok_or_else(|| lash_core::store::StoreError::RuntimeTurnLeaseExpired {
                session_id: lease.session_id.clone(),
                turn_id: lease.turn_id.clone(),
            })?;
        Ok(())
    }
}

fn test_current_epoch_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or_default()
}

#[async_trait]
impl lash_core::RuntimePersistence for SnapshotStore {
    async fn load_session(
        &self,
        scope: lash_core::SessionReadScope,
    ) -> std::result::Result<Option<lash_core::PersistedSessionRead>, lash_core::store::StoreError>
    {
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
        commit: lash_core::RuntimeCommit,
    ) -> std::result::Result<lash_core::RuntimeCommitResult, lash_core::store::StoreError> {
        let mut read = self.read.lock().expect("snapshot store lock");
        if let Some(completed) = &commit.completed_turn {
            let lease_key = (completed.session_id.clone(), completed.turn_id.clone());
            let lease_matches = self
                .runtime_turn_leases
                .lock()
                .expect("runtime turn leases lock")
                .get(&lease_key)
                .is_some_and(|lease| {
                    lease.lease_token == completed.lease_token
                        && lease.expires_at_epoch_ms > test_current_epoch_ms()
                });
            if !lease_matches {
                return Err(lash_core::store::StoreError::RuntimeTurnLeaseExpired {
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
            lash_core::GraphCommitDelta::ReplaceFull(graph) => graph,
            lash_core::GraphCommitDelta::Unchanged { leaf_node_id } => {
                let mut graph = existing_graph;
                graph.set_leaf_node_id(leaf_node_id);
                graph
            }
            lash_core::GraphCommitDelta::Append {
                nodes,
                leaf_node_id,
            } => {
                let mut graph = existing_graph;
                graph.extend_node_records(nodes);
                graph.set_leaf_node_id(leaf_node_id);
                graph
            }
        };
        *read = Some(lash_core::PersistedSessionRead {
            session_id: commit.session_id.clone(),
            head_revision: 8,
            config: commit.config,
            graph,
            checkpoint_ref: Some(lash_core::BlobRef("checkpoint".to_string())),
            checkpoint: Some(commit.checkpoint),
            token_ledger: commit.usage_deltas,
        });
        if let Some(completed) = &commit.completed_turn {
            let lease_key = (completed.session_id.clone(), completed.turn_id.clone());
            self.runtime_turn_checkpoints
                .lock()
                .expect("runtime turn checkpoints lock")
                .remove(&lease_key);
            self.runtime_turn_leases
                .lock()
                .expect("runtime turn leases lock")
                .remove(&lease_key);
            self.runtime_effect_journal
                .lock()
                .expect("runtime effect journal lock")
                .retain(|(session_id, turn_id, _), _| {
                    session_id != &completed.session_id || turn_id != &completed.turn_id
                });
        }
        Ok(lash_core::RuntimeCommitResult {
            head_revision: 8,
            checkpoint_ref: lash_core::BlobRef("checkpoint".to_string()),
            manifest: lash_core::SessionCheckpoint::default(),
        })
    }

    async fn claim_runtime_turn_lease(
        &self,
        session_id: &str,
        turn_id: &str,
        owner_id: &str,
        lease_ttl_ms: u64,
    ) -> std::result::Result<lash_core::RuntimeTurnLease, lash_core::store::StoreError> {
        let lease = lash_core::RuntimeTurnLease {
            schema_version: lash_core::RUNTIME_TURN_LEASE_SCHEMA_VERSION,
            session_id: session_id.to_string(),
            turn_id: turn_id.to_string(),
            owner_id: owner_id.to_string(),
            lease_token: format!("{session_id}:{turn_id}:{owner_id}"),
            fencing_token: 1,
            claimed_at_epoch_ms: test_current_epoch_ms(),
            expires_at_epoch_ms: test_current_epoch_ms().saturating_add(lease_ttl_ms),
        };
        self.runtime_turn_leases
            .lock()
            .expect("runtime turn leases lock")
            .insert((session_id.to_string(), turn_id.to_string()), lease.clone());
        Ok(lease)
    }

    async fn renew_runtime_turn_lease(
        &self,
        lease: &lash_core::RuntimeTurnLease,
        lease_ttl_ms: u64,
    ) -> std::result::Result<lash_core::RuntimeTurnLease, lash_core::store::StoreError> {
        let renewed = lash_core::RuntimeTurnLease {
            expires_at_epoch_ms: test_current_epoch_ms().saturating_add(lease_ttl_ms),
            ..lease.clone()
        };
        self.runtime_turn_leases
            .lock()
            .expect("runtime turn leases lock")
            .insert(
                (renewed.session_id.clone(), renewed.turn_id.clone()),
                renewed.clone(),
            );
        Ok(renewed)
    }

    async fn abandon_runtime_turn_lease(
        &self,
        lease: &lash_core::RuntimeTurnLease,
    ) -> std::result::Result<(), lash_core::store::StoreError> {
        self.runtime_turn_leases
            .lock()
            .expect("runtime turn leases lock")
            .remove(&(lease.session_id.clone(), lease.turn_id.clone()));
        Ok(())
    }

    async fn save_runtime_turn_checkpoint(
        &self,
        lease: &lash_core::RuntimeTurnLease,
        checkpoint: lash_core::RuntimeTurnCheckpoint,
    ) -> std::result::Result<(), lash_core::store::StoreError> {
        self.ensure_lease(lease)?;
        self.runtime_turn_checkpoints
            .lock()
            .expect("runtime turn checkpoints lock")
            .insert(
                (checkpoint.session_id.clone(), checkpoint.turn_id.clone()),
                checkpoint,
            );
        Ok(())
    }

    async fn load_runtime_turn_checkpoint(
        &self,
        session_id: &str,
        turn_id: &str,
    ) -> std::result::Result<Option<lash_core::RuntimeTurnCheckpoint>, lash_core::store::StoreError>
    {
        Ok(self
            .runtime_turn_checkpoints
            .lock()
            .expect("runtime turn checkpoints lock")
            .get(&(session_id.to_string(), turn_id.to_string()))
            .cloned())
    }

    async fn save_runtime_effect_outcome(
        &self,
        lease: &lash_core::RuntimeTurnLease,
        record: lash_core::RuntimeEffectJournalRecord,
    ) -> std::result::Result<(), lash_core::store::StoreError> {
        self.ensure_lease(lease)?;
        self.runtime_effect_journal
            .lock()
            .expect("runtime effect journal lock")
            .insert(
                (
                    record.session_id.clone(),
                    record.turn_id.clone(),
                    record.idempotency_key.clone(),
                ),
                record,
            );
        Ok(())
    }

    async fn load_runtime_effect_outcome(
        &self,
        session_id: &str,
        turn_id: &str,
        idempotency_key: &str,
    ) -> std::result::Result<
        Option<lash_core::RuntimeEffectJournalRecord>,
        lash_core::store::StoreError,
    > {
        Ok(self
            .runtime_effect_journal
            .lock()
            .expect("runtime effect journal lock")
            .get(&(
                session_id.to_string(),
                turn_id.to_string(),
                idempotency_key.to_string(),
            ))
            .cloned())
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

impl lash_core::SessionStoreFactory for ReusableStoreFactory {
    fn create_store(
        &self,
        _request: &lash_core::SessionStoreCreateRequest,
    ) -> std::result::Result<Arc<dyn lash_core::RuntimePersistence>, String> {
        Ok(Arc::clone(&self.store))
    }
}

struct BoundSessionStore {
    session_id: String,
}

#[async_trait]
impl lash_core::RuntimePersistence for BoundSessionStore {
    async fn load_session(
        &self,
        _scope: lash_core::SessionReadScope,
    ) -> std::result::Result<Option<lash_core::PersistedSessionRead>, lash_core::store::StoreError>
    {
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
        _commit: lash_core::RuntimeCommit,
    ) -> std::result::Result<lash_core::RuntimeCommitResult, lash_core::store::StoreError> {
        unreachable!("test should fail before committing to the reused child store")
    }

    async fn claim_runtime_turn_lease(
        &self,
        session_id: &str,
        turn_id: &str,
        owner_id: &str,
        lease_ttl_ms: u64,
    ) -> std::result::Result<lash_core::RuntimeTurnLease, lash_core::store::StoreError> {
        Ok(lash_core::RuntimeTurnLease {
            schema_version: lash_core::RUNTIME_TURN_LEASE_SCHEMA_VERSION,
            session_id: session_id.to_string(),
            turn_id: turn_id.to_string(),
            owner_id: owner_id.to_string(),
            lease_token: format!("{session_id}:{turn_id}:{owner_id}"),
            fencing_token: 1,
            claimed_at_epoch_ms: 0,
            expires_at_epoch_ms: lease_ttl_ms,
        })
    }

    async fn renew_runtime_turn_lease(
        &self,
        lease: &lash_core::RuntimeTurnLease,
        lease_ttl_ms: u64,
    ) -> std::result::Result<lash_core::RuntimeTurnLease, lash_core::store::StoreError> {
        Ok(lash_core::RuntimeTurnLease {
            expires_at_epoch_ms: lease.expires_at_epoch_ms.saturating_add(lease_ttl_ms),
            ..lease.clone()
        })
    }

    async fn abandon_runtime_turn_lease(
        &self,
        _lease: &lash_core::RuntimeTurnLease,
    ) -> std::result::Result<(), lash_core::store::StoreError> {
        Ok(())
    }

    async fn save_runtime_turn_checkpoint(
        &self,
        _lease: &lash_core::RuntimeTurnLease,
        _checkpoint: lash_core::RuntimeTurnCheckpoint,
    ) -> std::result::Result<(), lash_core::store::StoreError> {
        Ok(())
    }

    async fn load_runtime_turn_checkpoint(
        &self,
        _session_id: &str,
        _turn_id: &str,
    ) -> std::result::Result<Option<lash_core::RuntimeTurnCheckpoint>, lash_core::store::StoreError>
    {
        Ok(None)
    }

    async fn save_runtime_effect_outcome(
        &self,
        _lease: &lash_core::RuntimeTurnLease,
        _record: lash_core::RuntimeEffectJournalRecord,
    ) -> std::result::Result<(), lash_core::store::StoreError> {
        Ok(())
    }

    async fn load_runtime_effect_outcome(
        &self,
        _session_id: &str,
        _turn_id: &str,
        _idempotency_key: &str,
    ) -> std::result::Result<
        Option<lash_core::RuntimeEffectJournalRecord>,
        lash_core::store::StoreError,
    > {
        Ok(None)
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
            parent_session_id: None,
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

impl lash_core::SessionStoreFactory for RecordingStoreFactory {
    fn create_store(
        &self,
        request: &lash_core::SessionStoreCreateRequest,
    ) -> std::result::Result<Arc<dyn lash_core::RuntimePersistence>, String> {
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

fn app_tool_definition() -> lash_core::ToolDefinition {
    lash_core::ToolDefinition::raw(
        "app_lookup",
        "Look up app state.",
        serde_json::json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false
        }),
        serde_json::json!({ "type": "object" }),
    )
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
        "app_lookup",
        "Look up verbose app state.",
        serde_json::json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false
        }),
        serde_json::json!({ "type": "string" }),
    )
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
    lash_core::testing::TestProvider::builder()
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
    lash_core::testing::TestProvider::builder()
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
    lash_core::testing::TestProvider::builder()
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
    lash_core::testing::TestProvider::builder()
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
    lash_core::testing::TestProvider::builder()
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
    lash_core::testing::TestProvider::builder()
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
    lash_core::testing::TestProvider::builder()
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
    lash_core::testing::TestProvider::builder()
        .kind("retry-test")
        .default_model("mock-model")
        .requires_streaming(true)
        .options(lash_core::ProviderOptions {
            reliability: lash_core::provider::ProviderReliability::builder()
                .max_attempts(2)
                .base_delay_ms(0)
                .max_delay_ms(0)
                .build(),
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
    lash_core::testing::TestProvider::builder()
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
mod plugin_stack;
mod turn_streaming;
