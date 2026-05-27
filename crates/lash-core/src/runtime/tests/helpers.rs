use super::*;

pub(crate) fn default_state() -> RuntimeSessionState {
    RuntimeSessionState::default()
}

#[test]
pub(crate) fn stream_accumulator_merges_adjacent_display_reasoning_chunks() {
    let mut accumulator = LlmStreamAccumulator::default();
    accumulator.push_reasoning("I'll".to_string(), None, Vec::new(), None);
    accumulator.push_reasoning(" check".to_string(), None, Vec::new(), None);
    accumulator.push_reasoning(" the time.".to_string(), None, Vec::new(), None);

    assert_eq!(accumulator.parts.len(), 1);
    assert!(matches!(
        &accumulator.parts[0],
        LlmOutputPart::Reasoning { text, .. } if text == "I'll check the time."
    ));
}

#[test]
pub(crate) fn stream_accumulator_enriches_reasoning_delta_with_later_roundtrip_payload() {
    let mut accumulator = LlmStreamAccumulator::default();
    accumulator.push_reasoning("I'll check the time.".to_string(), None, Vec::new(), None);
    accumulator.push_reasoning(
        "I'll check the time.".to_string(),
        Some("rs_1".to_string()),
        vec!["I'll check the time.".to_string()],
        Some("encrypted".to_string()),
    );

    assert_eq!(accumulator.parts.len(), 1);
    assert!(matches!(
        &accumulator.parts[0],
        LlmOutputPart::Reasoning {
            text,
            replay: Some(replay),
            ..
        } if text == "I'll check the time."
            && replay.item_id.as_deref() == Some("rs_1")
            && replay.encrypted_content.as_deref() == Some("encrypted")
    ));
}

#[test]
pub(crate) fn stream_accumulator_preserves_reasoning_when_final_response_has_tool_call() {
    let mut accumulator = LlmStreamAccumulator::default();
    accumulator.push_reasoning("I'll check the time.".to_string(), None, Vec::new(), None);
    accumulator.push_tool_call(
        "call_1".to_string(),
        "exec_command".to_string(),
        "{\"cmd\":\"date\"}".to_string(),
        Some(lash_sansio::llm::types::ProviderReplayMeta {
            item_id: Some("item_1".to_string()),
            opaque: Some("sig".to_string()),
        }),
    );

    let mut response = LlmResponse {
        full_text: String::new(),
        parts: vec![LlmOutputPart::ToolCall {
            call_id: "call_1".to_string(),
            tool_name: "exec_command".to_string(),
            input_json: "{\"cmd\":\"date\"}".to_string(),
            replay: Some(lash_sansio::llm::types::ProviderReplayMeta {
                item_id: Some("item_1".to_string()),
                opaque: Some("sig".to_string()),
            }),
        }],
        ..Default::default()
    };

    accumulator.apply_to_response(&mut response);

    assert_eq!(response.parts.len(), 2);
    assert!(matches!(
        &response.parts[0],
        LlmOutputPart::Reasoning { text, .. } if text == "I'll check the time."
    ));
    assert!(matches!(
        &response.parts[1],
        LlmOutputPart::ToolCall { tool_name, .. } if tool_name == "exec_command"
    ));
}

#[test]
pub(crate) fn stream_accumulator_does_not_duplicate_complete_final_response() {
    let mut accumulator = LlmStreamAccumulator::default();
    accumulator.push_reasoning("I'll answer.".to_string(), None, Vec::new(), None);
    accumulator.push_text("Done.");

    let mut response = LlmResponse {
        full_text: "Done.".to_string(),
        parts: vec![
            LlmOutputPart::Reasoning {
                text: "I'll answer.".to_string(),
                replay: None,
            },
            LlmOutputPart::Text {
                text: "Done.".to_string(),
                response_meta: None,
            },
        ],
        ..Default::default()
    };

    accumulator.apply_to_response(&mut response);

    assert_eq!(response.parts.len(), 2);
    assert!(matches!(
        &response.parts[0],
        LlmOutputPart::Reasoning { text, .. } if text == "I'll answer."
    ));
    assert!(matches!(
        &response.parts[1],
        LlmOutputPart::Text { text, .. } if text == "Done."
    ));
}

pub(crate) trait ReadModelState {
    fn read_model(&self) -> crate::session_graph::SessionReadModel;
}

impl ReadModelState for SessionStateEnvelope {
    fn read_model(&self) -> crate::session_graph::SessionReadModel {
        self.read_model()
    }
}

impl ReadModelState for RuntimeSessionState {
    fn read_model(&self) -> crate::session_graph::SessionReadModel {
        self.read_model()
    }
}

pub(crate) trait ReadModelStateMut: ReadModelState {
    fn append_message(&mut self, message: Message);
}

impl ReadModelStateMut for SessionStateEnvelope {
    fn append_message(&mut self, message: Message) {
        self.session_graph.append_message(message);
    }
}

impl ReadModelStateMut for RuntimeSessionState {
    fn append_message(&mut self, message: Message) {
        self.session_graph.append_message(message);
    }
}

pub(crate) fn active_conversation_messages(state: &impl ReadModelState) -> Vec<Message> {
    state.read_model().messages.as_ref().clone()
}

pub(crate) fn active_tool_calls(state: &impl ReadModelState) -> Vec<ToolCallRecord> {
    state.read_model().tool_calls.as_ref().clone()
}

pub(crate) fn append_message(state: &mut impl ReadModelStateMut, message: Message) {
    state.append_message(message);
}

#[derive(Clone, Default)]
pub(crate) struct RecordingSink {
    pub(crate) events: Arc<Mutex<Vec<SessionEvent>>>,
}

#[async_trait::async_trait]
impl EventSink for RecordingSink {
    async fn emit(&self, event: SessionEvent) {
        self.events.lock().expect("lock sink").push(event);
    }
}

impl RecordingSink {
    pub(crate) fn snapshot(&self) -> Vec<SessionEvent> {
        self.events.lock().expect("lock sink").clone()
    }
}

#[derive(Clone, Default)]
pub(crate) struct RecordingTurnEvents {
    pub(crate) events: Arc<Mutex<Vec<TurnActivity>>>,
}

#[async_trait::async_trait]
impl TurnActivitySink for RecordingTurnEvents {
    async fn emit(&self, activity: TurnActivity) {
        self.events.lock().expect("lock turn events").push(activity);
    }
}

impl RecordingTurnEvents {
    pub(crate) fn snapshot(&self) -> Vec<TurnActivity> {
        self.events.lock().expect("lock turn events").clone()
    }
}

#[derive(Default)]
pub(crate) struct RecordingStore {
    pub(crate) session_head_meta: Mutex<Option<crate::SessionHeadMeta>>,
    pub(crate) session_meta: Mutex<Option<crate::SessionMeta>>,
    pub(crate) session_graph: Mutex<crate::SessionGraph>,
    pub(crate) checkpoint: Mutex<Option<crate::HydratedSessionCheckpoint>>,
    pub(crate) usage_deltas: Mutex<Vec<crate::TokenLedgerEntry>>,
    pub(crate) runtime_commit_count: Mutex<usize>,
    runtime_turn_checkpoints:
        Mutex<std::collections::HashMap<(String, String), crate::RuntimeTurnCheckpoint>>,
    runtime_turn_leases:
        Mutex<std::collections::HashMap<(String, String), crate::RuntimeTurnLease>>,
    runtime_effect_journal: Mutex<
        std::collections::HashMap<(String, String, String), crate::RuntimeEffectJournalRecord>,
    >,
    runtime_turn_checkpoint_save_count: Mutex<usize>,
    runtime_effect_journal_save_count: Mutex<usize>,
    runtime_turn_lease_renew_count: Mutex<usize>,
    runtime_turn_lease_abandon_count: Mutex<usize>,
}

crate::impl_noop_attachment_manifest!(RecordingStore);

#[async_trait::async_trait]
impl crate::store::RuntimePersistence for RecordingStore {
    async fn load_session(
        &self,
        scope: crate::store::SessionReadScope,
    ) -> Result<Option<crate::store::PersistedSessionRead>, crate::store::StoreError> {
        let Some(meta) = self.session_head_meta.lock().expect("lock store").clone() else {
            return Ok(None);
        };
        let mut graph = self.session_graph.lock().expect("lock graph").clone();
        if let crate::store::SessionReadScope::ActivePath { leaf_node_id } = scope {
            if let Some(leaf_node_id) = leaf_node_id.or_else(|| meta.leaf_node_id.clone()) {
                graph.set_leaf_node_id(Some(leaf_node_id));
            }
            graph = graph.fork_current_path();
        }
        Ok(Some(crate::store::PersistedSessionRead {
            session_id: meta.session_id,
            head_revision: meta.head_revision,
            config: meta.config,
            graph,
            checkpoint_ref: meta.checkpoint_ref,
            checkpoint: self.checkpoint.lock().expect("lock checkpoint").clone(),
            token_ledger: merge_usage_delta_entries(
                self.usage_deltas.lock().expect("lock usage deltas").clone(),
            ),
        }))
    }

    async fn load_node(
        &self,
        node_id: &str,
    ) -> Result<Option<crate::SessionNodeRecord>, crate::store::StoreError> {
        Ok(self
            .session_graph
            .lock()
            .expect("lock graph")
            .find_node(node_id)
            .cloned())
    }

    async fn commit_runtime_state(
        &self,
        commit: crate::store::RuntimeCommit,
    ) -> Result<crate::store::RuntimeCommitResult, crate::store::StoreError> {
        let mut meta = self.session_head_meta.lock().expect("lock store");
        let actual = meta.as_ref().map_or(0, |meta| meta.head_revision);
        if let Some(bound) = meta.as_ref().map(|meta| meta.session_id.clone())
            && bound != commit.session_id
        {
            return Err(crate::store::StoreError::SessionBindingMismatch {
                bound_session_id: bound,
                attempted_session_id: commit.session_id,
            });
        }
        if commit.expected_head_revision.is_some() && commit.expected_head_revision != Some(actual)
        {
            return Err(crate::store::StoreError::HeadRevisionConflict {
                expected: commit.expected_head_revision,
                actual,
            });
        }
        if let Some(completed) = &commit.completed_turn {
            let lease_key = (completed.session_id.clone(), completed.turn_id.clone());
            let lease_matches = self
                .runtime_turn_leases
                .lock()
                .expect("lock runtime turn leases")
                .get(&lease_key)
                .is_some_and(|lease| {
                    lease.lease_token == completed.lease_token
                        && lease.expires_at_epoch_ms > current_epoch_ms()
                });
            if !lease_matches {
                return Err(crate::store::StoreError::RuntimeTurnLeaseExpired {
                    session_id: completed.session_id.clone(),
                    turn_id: completed.turn_id.clone(),
                });
            }
        }
        let mut graph = self.session_graph.lock().expect("lock graph");
        let leaf_node_id = match &commit.graph {
            crate::store::GraphCommitDelta::Unchanged { leaf_node_id } => leaf_node_id.clone(),
            crate::store::GraphCommitDelta::Append {
                nodes,
                leaf_node_id,
            } => {
                graph.extend_node_records(nodes.iter().cloned());
                leaf_node_id.clone()
            }
            crate::store::GraphCommitDelta::ReplaceFull(next) => {
                *graph = next.clone();
                next.leaf_node_id.clone()
            }
        };
        self.usage_deltas
            .lock()
            .expect("lock usage deltas")
            .extend(commit.usage_deltas.iter().cloned());
        let checkpoint_ref = crate::BlobRef(format!("recording-checkpoint-{}", actual + 1));
        let manifest = crate::store::SessionCheckpoint {
            turn_state: commit.checkpoint.turn_state.clone(),
            tool_state_ref: commit.checkpoint.tool_state_ref.clone(),
            plugin_snapshot_ref: commit.checkpoint.plugin_snapshot_ref.clone(),
            plugin_snapshot_revision: commit.checkpoint.plugin_snapshot_revision,
            execution_state_ref: commit.checkpoint.execution_state_ref.clone(),
        };
        *self.checkpoint.lock().expect("lock checkpoint") = Some(commit.checkpoint);
        if let Some(completed) = &commit.completed_turn {
            let lease_key = (completed.session_id.clone(), completed.turn_id.clone());
            let lease_matches = self
                .runtime_turn_leases
                .lock()
                .expect("lock runtime turn leases")
                .get(&lease_key)
                .is_some_and(|lease| lease.lease_token == completed.lease_token);
            if lease_matches {
                self.runtime_turn_checkpoints
                    .lock()
                    .expect("lock runtime turn checkpoints")
                    .remove(&lease_key);
                self.runtime_turn_leases
                    .lock()
                    .expect("lock runtime turn leases")
                    .remove(&lease_key);
                self.runtime_effect_journal
                    .lock()
                    .expect("lock runtime effect journal")
                    .retain(|(session_id, turn_id, _), _| {
                        session_id != &completed.session_id || turn_id != &completed.turn_id
                    });
            }
        }
        let head_revision = actual + 1;
        *meta = Some(crate::SessionHeadMeta {
            session_id: commit.session_id,
            head_revision,
            config: commit.config,
            checkpoint_ref: Some(checkpoint_ref.clone()),
            leaf_node_id,
            graph_node_count: graph.nodes.len(),
            token_ledger: Vec::new(),
        });
        *self
            .runtime_commit_count
            .lock()
            .expect("lock runtime commit count") += 1;
        Ok(crate::store::RuntimeCommitResult {
            head_revision,
            checkpoint_ref,
            manifest,
        })
    }

    async fn claim_runtime_turn_lease(
        &self,
        session_id: &str,
        turn_id: &str,
        owner_id: &str,
        lease_ttl_ms: u64,
    ) -> Result<crate::RuntimeTurnLease, crate::store::StoreError> {
        let lease = crate::RuntimeTurnLease {
            schema_version: crate::RUNTIME_TURN_LEASE_SCHEMA_VERSION,
            session_id: session_id.to_string(),
            turn_id: turn_id.to_string(),
            owner_id: owner_id.to_string(),
            lease_token: format!("{session_id}:{turn_id}:{owner_id}"),
            fencing_token: 1,
            claimed_at_epoch_ms: current_epoch_ms(),
            expires_at_epoch_ms: current_epoch_ms().saturating_add(lease_ttl_ms),
        };
        self.runtime_turn_leases
            .lock()
            .expect("lock runtime turn leases")
            .insert((session_id.to_string(), turn_id.to_string()), lease.clone());
        Ok(lease)
    }

    async fn renew_runtime_turn_lease(
        &self,
        lease: &crate::RuntimeTurnLease,
        lease_ttl_ms: u64,
    ) -> Result<crate::RuntimeTurnLease, crate::store::StoreError> {
        self.runtime_turn_leases
            .lock()
            .expect("lock runtime turn leases")
            .get(&(lease.session_id.clone(), lease.turn_id.clone()))
            .filter(|current| {
                current.lease_token == lease.lease_token
                    && current.expires_at_epoch_ms > current_epoch_ms()
            })
            .ok_or_else(|| crate::store::StoreError::RuntimeTurnLeaseExpired {
                session_id: lease.session_id.clone(),
                turn_id: lease.turn_id.clone(),
            })?;
        let renewed = crate::RuntimeTurnLease {
            expires_at_epoch_ms: current_epoch_ms().saturating_add(lease_ttl_ms),
            ..lease.clone()
        };
        self.runtime_turn_leases
            .lock()
            .expect("lock runtime turn leases")
            .insert(
                (renewed.session_id.clone(), renewed.turn_id.clone()),
                renewed.clone(),
            );
        *self
            .runtime_turn_lease_renew_count
            .lock()
            .expect("lock runtime turn lease renew count") += 1;
        Ok(renewed)
    }

    async fn abandon_runtime_turn_lease(
        &self,
        lease: &crate::RuntimeTurnLease,
    ) -> Result<(), crate::store::StoreError> {
        let key = (lease.session_id.clone(), lease.turn_id.clone());
        let mut leases = self
            .runtime_turn_leases
            .lock()
            .expect("lock runtime turn leases");
        if leases.get(&key).is_some_and(|current| {
            current.owner_id == lease.owner_id
                && current.lease_token == lease.lease_token
                && current.fencing_token == lease.fencing_token
        }) {
            leases.remove(&key);
        }
        *self
            .runtime_turn_lease_abandon_count
            .lock()
            .expect("lock runtime turn lease abandon count") += 1;
        Ok(())
    }

    async fn save_runtime_turn_checkpoint(
        &self,
        lease: &crate::RuntimeTurnLease,
        checkpoint: crate::RuntimeTurnCheckpoint,
    ) -> Result<(), crate::store::StoreError> {
        self.runtime_turn_leases
            .lock()
            .expect("lock runtime turn leases")
            .get(&(lease.session_id.clone(), lease.turn_id.clone()))
            .filter(|current| {
                current.lease_token == lease.lease_token
                    && current.expires_at_epoch_ms > current_epoch_ms()
            })
            .ok_or_else(|| crate::store::StoreError::RuntimeTurnLeaseExpired {
                session_id: lease.session_id.clone(),
                turn_id: lease.turn_id.clone(),
            })?;
        self.runtime_turn_checkpoints
            .lock()
            .expect("lock runtime turn checkpoints")
            .insert(
                (checkpoint.session_id.clone(), checkpoint.turn_id.clone()),
                checkpoint,
            );
        *self
            .runtime_turn_checkpoint_save_count
            .lock()
            .expect("lock runtime turn checkpoint save count") += 1;
        Ok(())
    }

    async fn load_runtime_turn_checkpoint(
        &self,
        session_id: &str,
        turn_id: &str,
    ) -> Result<Option<crate::RuntimeTurnCheckpoint>, crate::store::StoreError> {
        Ok(self
            .runtime_turn_checkpoints
            .lock()
            .expect("lock runtime turn checkpoints")
            .get(&(session_id.to_string(), turn_id.to_string()))
            .cloned())
    }

    async fn save_runtime_effect_outcome(
        &self,
        lease: &crate::RuntimeTurnLease,
        record: crate::RuntimeEffectJournalRecord,
    ) -> Result<(), crate::store::StoreError> {
        self.runtime_turn_leases
            .lock()
            .expect("lock runtime turn leases")
            .get(&(lease.session_id.clone(), lease.turn_id.clone()))
            .filter(|current| {
                current.lease_token == lease.lease_token
                    && current.expires_at_epoch_ms > current_epoch_ms()
            })
            .ok_or_else(|| crate::store::StoreError::RuntimeTurnLeaseExpired {
                session_id: lease.session_id.clone(),
                turn_id: lease.turn_id.clone(),
            })?;
        self.runtime_effect_journal
            .lock()
            .expect("lock runtime effect journal")
            .insert(
                (
                    record.session_id.clone(),
                    record.turn_id.clone(),
                    record.idempotency_key.clone(),
                ),
                record,
            );
        *self
            .runtime_effect_journal_save_count
            .lock()
            .expect("lock runtime effect journal save count") += 1;
        Ok(())
    }

    async fn load_runtime_effect_outcome(
        &self,
        session_id: &str,
        turn_id: &str,
        idempotency_key: &str,
    ) -> Result<Option<crate::RuntimeEffectJournalRecord>, crate::store::StoreError> {
        Ok(self
            .runtime_effect_journal
            .lock()
            .expect("lock runtime effect journal")
            .get(&(
                session_id.to_string(),
                turn_id.to_string(),
                idempotency_key.to_string(),
            ))
            .cloned())
    }

    async fn save_session_meta(
        &self,
        meta: crate::store::SessionMeta,
    ) -> Result<(), crate::store::StoreError> {
        *self.session_meta.lock().expect("lock session meta") = Some(meta);
        Ok(())
    }

    async fn load_session_meta(
        &self,
    ) -> Result<Option<crate::store::SessionMeta>, crate::store::StoreError> {
        Ok(self.session_meta.lock().expect("lock session meta").clone())
    }

    async fn tombstone_nodes(&self, _ids: &[String]) -> Result<(), crate::store::StoreError> {
        Ok(())
    }

    async fn vacuum(&self) -> Result<crate::store::VacuumReport, crate::store::StoreError> {
        Ok(crate::store::VacuumReport::default())
    }

    async fn gc_unreachable(&self) -> Result<crate::store::GcReport, crate::store::StoreError> {
        Ok(crate::store::GcReport::default())
    }
}

fn current_epoch_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or_default()
}

impl RecordingStore {
    pub(crate) async fn save_session_head_meta(&self, meta: crate::SessionHeadMeta) {
        *self.session_head_meta.lock().expect("lock store") = Some(meta);
    }

    pub(crate) fn runtime_turn_checkpoint_count(&self) -> usize {
        self.runtime_turn_checkpoints
            .lock()
            .expect("lock runtime turn checkpoints")
            .len()
    }

    pub(crate) fn runtime_effect_journal_count(&self) -> usize {
        self.runtime_effect_journal
            .lock()
            .expect("lock runtime effect journal")
            .len()
    }

    pub(crate) fn runtime_turn_checkpoint_save_count(&self) -> usize {
        *self
            .runtime_turn_checkpoint_save_count
            .lock()
            .expect("lock runtime turn checkpoint save count")
    }

    pub(crate) fn runtime_effect_journal_save_count(&self) -> usize {
        *self
            .runtime_effect_journal_save_count
            .lock()
            .expect("lock runtime effect journal save count")
    }

    pub(crate) fn runtime_turn_lease_renew_count(&self) -> usize {
        *self
            .runtime_turn_lease_renew_count
            .lock()
            .expect("lock runtime turn lease renew count")
    }

    pub(crate) fn runtime_turn_lease_abandon_count(&self) -> usize {
        *self
            .runtime_turn_lease_abandon_count
            .lock()
            .expect("lock runtime turn lease abandon count")
    }
}

#[derive(Debug)]
pub(crate) struct MockCall {
    pub(crate) stream_events: Vec<LlmStreamEvent>,
    pub(crate) response: Result<LlmResponse, LlmTransportError>,
}

pub(crate) fn mock_provider(calls: Vec<MockCall>) -> TestProvider {
    let calls = Arc::new(Mutex::new(calls));
    TestProvider::builder()
        .kind("mock")
        .requires_streaming(true)
        .complete(move |req| {
            let calls = Arc::clone(&calls);
            async move {
                let call = calls.lock().expect("lock calls").remove(0);
                if let Some(tx) = req.stream_events.as_ref() {
                    for event in &call.stream_events {
                        tx.send(event.clone());
                    }
                }
                call.response
            }
        })
        .build()
}

pub(crate) fn standard_test_policy() -> SessionPolicy {
    SessionPolicy {
        provider: mock_provider(Vec::new()).into_handle(),
        model: crate::ModelSpec::from_token_limits("mock-model", None, 200_000, None, None)
            .expect("valid model spec"),
        ..SessionPolicy::default()
    }
}

pub(crate) fn test_host_config() -> EmbeddedRuntimeHost {
    EmbeddedRuntimeHost::new(RuntimeCoreConfig::default())
}

pub(crate) fn test_host_config_with_trace_path(path: PathBuf) -> EmbeddedRuntimeHost {
    EmbeddedRuntimeHost::new(
        RuntimeCoreConfig::default()
            .with_trace_sink(Some(Arc::new(lash_trace::JsonlTraceSink::new(path)))),
    )
}

pub(crate) fn test_host_config_with_trace_path_and_stream_events(
    path: PathBuf,
) -> EmbeddedRuntimeHost {
    EmbeddedRuntimeHost::new(
        RuntimeCoreConfig::default()
            .with_trace_sink(Some(Arc::new(lash_trace::JsonlTraceSink::new(path))))
            .with_trace_level(lash_trace::TraceLevel::Extended),
    )
}

#[derive(Clone, Default)]
pub(crate) struct RecordingSessionStoreFactory {
    stores: Arc<StdMutex<Vec<Arc<RecordingStore>>>>,
}

impl RecordingSessionStoreFactory {
    pub(crate) fn stores(&self) -> Vec<Arc<RecordingStore>> {
        self.stores.lock().expect("store factory").clone()
    }
}

impl SessionStoreFactory for RecordingSessionStoreFactory {
    fn create_store(
        &self,
        request: &SessionStoreCreateRequest,
    ) -> Result<Arc<dyn crate::store::RuntimePersistence>, String> {
        let store = Arc::new(RecordingStore::default());
        *store.session_meta.lock().expect("lock session meta") = Some(crate::SessionMeta {
            session_id: request.session_id.clone(),
            session_name: request.session_id.clone(),
            created_at: "2026-04-06T00:00:00Z".to_string(),
            model: request.policy.model.id.clone(),
            cwd: None,
            relation: request.relation.clone(),
        });
        self.stores
            .lock()
            .expect("store factory")
            .push(Arc::clone(&store));
        Ok(store as Arc<dyn crate::store::RuntimePersistence>)
    }
}

pub(crate) fn plugin_session_with_tools(
    session_id: &str,
    tools: Arc<dyn crate::ToolProvider>,
) -> Arc<crate::PluginSession> {
    let tool_factory = StaticPluginFactory::new(
        "test_tools",
        crate::PluginSpec::new().with_tool_provider(Arc::clone(&tools)),
    );
    let mut factories = crate::testing::test_standard_protocol_factories();
    factories.push(Arc::new(tool_factory));
    crate::PluginHost::new(factories)
        .build_session(session_id, None)
        .expect("plugins")
}

pub(crate) struct EmptyTools;

#[async_trait::async_trait]
impl crate::ToolProvider for EmptyTools {
    fn tool_manifests(&self) -> Vec<crate::ToolManifest> {
        Vec::new()
    }

    fn resolve_contract(&self, _name: &str) -> Option<Arc<crate::ToolContract>> {
        None
    }

    async fn execute(&self, _call: crate::ToolCall<'_>) -> crate::ToolResult {
        crate::ToolResult::err(serde_json::json!("Unknown tool"))
    }
}

pub(crate) async fn standard_runtime_with_transport(transport: TestProvider) -> LashRuntime {
    let tools: Arc<dyn crate::ToolProvider> = Arc::new(EmptyTools);
    let mut runtime = LashRuntime::from_embedded_state(
        standard_test_policy(),
        test_host_config(),
        crate::RuntimeServices::new(plugin_session_with_tools("root", tools)),
        RuntimeSessionState::default(),
    )
    .await
    .expect("runtime");
    runtime.host.process_registry = Some(Arc::new(crate::TestLocalProcessRegistry::default()));
    runtime.policy.provider = transport.clone().into_handle();
    runtime
}
pub(crate) type RuntimeTestPluginBuilder = dyn Fn(&crate::PluginSessionContext) -> Result<Arc<dyn crate::SessionPlugin>, crate::PluginError>
    + Send
    + Sync;
pub(crate) type RuntimeExternalRegistrar =
    dyn Fn(&mut crate::PluginRegistrar) -> Result<(), crate::PluginError> + Send + Sync;

pub(crate) struct RuntimeTestPluginFactory {
    pub(crate) build: Arc<RuntimeTestPluginBuilder>,
}

impl crate::PluginFactory for RuntimeTestPluginFactory {
    fn id(&self) -> &'static str {
        "runtime-test"
    }

    fn build(
        &self,
        ctx: &crate::PluginSessionContext,
    ) -> Result<Arc<dyn crate::SessionPlugin>, crate::PluginError> {
        (self.build)(ctx)
    }
}

pub(crate) struct RuntimeTestPlugin {
    pub(crate) before_turn: Option<crate::plugin::BeforeTurnHook>,
    pub(crate) checkpoint: Option<crate::plugin::CheckpointHook>,
    pub(crate) tool_result_projector: Option<crate::plugin::ToolResultProjector>,
    pub(crate) runtime_event: Option<crate::plugin::PluginLifecycleEventHook>,
    pub(crate) external_registrar: Option<Arc<RuntimeExternalRegistrar>>,
}

impl crate::SessionPlugin for RuntimeTestPlugin {
    fn id(&self) -> &'static str {
        "runtime-test"
    }

    fn register(&self, reg: &mut crate::PluginRegistrar) -> Result<(), crate::PluginError> {
        if let Some(hook) = &self.before_turn {
            reg.turn().before(Arc::clone(hook));
        }
        if let Some(hook) = &self.checkpoint {
            reg.turn().checkpoint(Arc::clone(hook));
        }
        if let Some(projector) = &self.tool_result_projector {
            reg.tool_results().projector(Arc::clone(projector))?;
        }
        if let Some(hook) = &self.runtime_event {
            reg.session().on_event(Arc::clone(hook));
        }
        if let Some(register) = &self.external_registrar {
            register(reg)?;
        }
        Ok(())
    }
}

pub(crate) async fn runtime_with_plugins(
    plugins: Vec<Arc<dyn crate::PluginFactory>>,
    transport: TestProvider,
) -> LashRuntime {
    runtime_with_plugins_and_tools_and_host(
        plugins,
        Arc::new(EmptyTools),
        transport,
        test_host_config(),
    )
    .await
}

pub(crate) async fn runtime_with_plugins_and_tools(
    plugins: Vec<Arc<dyn crate::PluginFactory>>,
    tools: Arc<dyn crate::ToolProvider>,
    transport: TestProvider,
) -> LashRuntime {
    runtime_with_plugins_and_tools_and_host(plugins, tools, transport, test_host_config()).await
}

pub(crate) async fn runtime_with_plugins_and_tools_and_host(
    plugins: Vec<Arc<dyn crate::PluginFactory>>,
    tools: Arc<dyn crate::ToolProvider>,
    transport: TestProvider,
    host: EmbeddedRuntimeHost,
) -> LashRuntime {
    let mut factories = plugins;
    let tools = Arc::clone(&tools);
    factories.push(Arc::new(StaticPluginFactory::new(
        "test_tools",
        crate::PluginSpec::new().with_tool_provider(Arc::clone(&tools)),
    )));
    let plugin_host = crate::PluginHost::new(factories);
    let plugin_session = plugin_host.build_session("root", None).expect("plugins");
    let mut runtime = LashRuntime::from_embedded_state(
        standard_test_policy(),
        host,
        crate::RuntimeServices::new(plugin_session),
        RuntimeSessionState::default(),
    )
    .await
    .expect("runtime");
    runtime.host.process_registry = Some(Arc::new(crate::TestLocalProcessRegistry::default()));
    runtime.policy.provider = transport.clone().into_handle();
    runtime
}

pub(crate) async fn runtime_with_plugins_and_tools_and_host_and_store(
    plugins: Vec<Arc<dyn crate::PluginFactory>>,
    tools: Arc<dyn crate::ToolProvider>,
    transport: TestProvider,
    host: EmbeddedRuntimeHost,
    store: Arc<dyn crate::RuntimePersistence>,
) -> LashRuntime {
    let mut factories = plugins;
    let tools = Arc::clone(&tools);
    factories.push(Arc::new(StaticPluginFactory::new(
        "test_tools",
        crate::PluginSpec::new().with_tool_provider(Arc::clone(&tools)),
    )));
    let plugin_host = crate::PluginHost::new(factories);
    let plugin_session = plugin_host.build_session("root", None).expect("plugins");
    let services =
        crate::PersistentRuntimeServices::new(plugin_session, store).into_runtime_services();
    let mut runtime = LashRuntime::from_embedded_state(
        standard_test_policy(),
        host,
        services,
        RuntimeSessionState::default(),
    )
    .await
    .expect("runtime");
    runtime.policy.provider = transport.clone().into_handle();
    runtime
}

pub(crate) struct EchoTool;

fn echo_tool_definition() -> crate::ToolDefinition {
    crate::ToolDefinition::raw(
        "tool:echo_tool",
        "echo_tool",
        "Return a tool payload",
        serde_json::json!({
            "type": "object",
            "properties": { "value": { "type": "string" } },
            "required": ["value"],
            "additionalProperties": false
        }),
        serde_json::json!({ "type": "object", "additionalProperties": true }),
    )
}

#[async_trait::async_trait]
impl crate::ToolProvider for EchoTool {
    fn tool_manifests(&self) -> Vec<crate::ToolManifest> {
        vec![echo_tool_definition().manifest()]
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<crate::ToolContract>> {
        (name == "echo_tool").then(|| Arc::new(echo_tool_definition().contract()))
    }

    async fn execute(&self, call: crate::ToolCall<'_>) -> crate::ToolResult {
        assert_eq!(call.name, "echo_tool");
        let value = call
            .args
            .get("value")
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        crate::ToolResult::ok(serde_json::json!({
            "payload": format!("raw:{value}")
        }))
    }
}

pub(crate) struct TerminalControlTool {
    pub(crate) controls: Vec<crate::ToolControl>,
}

#[async_trait::async_trait]
impl crate::ToolProvider for TerminalControlTool {
    fn tool_manifests(&self) -> Vec<crate::ToolManifest> {
        (0..self.controls.len())
            .map(|index| terminal_tool_definition(index).manifest())
            .collect()
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<crate::ToolContract>> {
        name.strip_prefix("terminal_tool_")
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|index| *index < self.controls.len())
            .map(|index| Arc::new(terminal_tool_definition(index).contract()))
    }

    async fn execute(&self, call: crate::ToolCall<'_>) -> crate::ToolResult {
        self.result_for(call.name)
    }
}

impl TerminalControlTool {
    fn result_for(&self, name: &str) -> crate::ToolResult {
        let index = name
            .strip_prefix("terminal_tool_")
            .and_then(|value| value.parse::<usize>().ok())
            .expect("known terminal test tool");
        crate::ToolResult::ok(serde_json::json!({ "tool": name }))
            .with_control(self.controls[index].clone())
    }
}

fn terminal_tool_definition(index: usize) -> crate::ToolDefinition {
    crate::ToolDefinition::raw(
        format!("tool:terminal_tool_{index}"),
        format!("terminal_tool_{index}"),
        "Return a terminal control result",
        crate::ToolDefinition::default_input_schema(),
        serde_json::json!({ "type": "object", "additionalProperties": true }),
    )
}

/// Tool that sleeps for 10 seconds unless its future is aborted or the
/// execution-context cancellation token fires. Used to verify that turn
/// cancellation unwinds in-flight tool tasks promptly.
pub(crate) struct SlowTool {
    pub(crate) observed_cancel: Arc<AtomicBool>,
}

#[async_trait::async_trait]
impl crate::ToolProvider for SlowTool {
    fn tool_manifests(&self) -> Vec<crate::ToolManifest> {
        vec![slow_tool_definition().manifest()]
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<crate::ToolContract>> {
        (name == "slow_tool").then(|| Arc::new(slow_tool_definition().contract()))
    }

    async fn execute(&self, call: crate::ToolCall<'_>) -> crate::ToolResult {
        let observed = Arc::clone(&self.observed_cancel);
        if let Some(token) = call.context.cancellation_token() {
            let token = token.clone();
            tokio::select! {
                _ = token.cancelled() => {
                    observed.store(true, Ordering::SeqCst);
                    crate::ToolResult::cancelled("cancelled")
                }
                _ = tokio::time::sleep(std::time::Duration::from_secs(10)) => {
                    crate::ToolResult::ok(serde_json::json!({"status": "completed"}))
                }
            }
        } else {
            tokio::time::sleep(std::time::Duration::from_secs(10)).await;
            crate::ToolResult::ok(serde_json::json!({"status": "completed"}))
        }
    }
}

fn slow_tool_definition() -> crate::ToolDefinition {
    crate::ToolDefinition::raw(
        "tool:slow_tool",
        "slow_tool",
        "Sleep for a long time; respects cancellation.",
        crate::ToolDefinition::default_input_schema(),
        serde_json::json!({ "type": "object", "additionalProperties": true }),
    )
}

pub(crate) struct MemoryProbeTool;

#[async_trait::async_trait]
impl crate::ToolProvider for MemoryProbeTool {
    fn tool_manifests(&self) -> Vec<crate::ToolManifest> {
        vec![memory_probe_tool_definition().manifest()]
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<crate::ToolContract>> {
        (name == "memory_probe").then(|| Arc::new(memory_probe_tool_definition().contract()))
    }

    async fn execute(&self, _call: crate::ToolCall<'_>) -> crate::ToolResult {
        crate::ToolResult::ok(json!("ok"))
    }
}

fn memory_probe_tool_definition() -> crate::ToolDefinition {
    crate::ToolDefinition::raw(
        "tool:memory_probe",
        "memory_probe",
        "probe",
        crate::ToolDefinition::default_input_schema(),
        serde_json::json!({ "type": "string" }),
    )
}

pub(crate) struct ChildSessionTool;

#[async_trait::async_trait]
impl crate::ToolProvider for ChildSessionTool {
    fn tool_manifests(&self) -> Vec<crate::ToolManifest> {
        vec![child_session_tool_definition().manifest()]
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<crate::ToolContract>> {
        (name == "spawn_child").then(|| Arc::new(child_session_tool_definition().contract()))
    }

    async fn execute(&self, call: crate::ToolCall<'_>) -> crate::ToolResult {
        let context = call.context;
        let child = match context
            .sessions()
            .create_session(
                crate::SessionCreateRequest::child_session(
                    context.session_id(),
                    crate::SessionStartPoint::Empty,
                    crate::PluginOptions::default(),
                )
                .with_session_id("subagent-child")
                .with_plugin_source(crate::SessionPluginSource::CurrentSessionFork)
                .with_usage_source("subagent"),
            )
            .await
        {
            Ok(child) => child,
            Err(err) => return crate::ToolResult::err_fmt(format_args!("{err}")),
        };

        let turn = match context
            .sessions()
            .start_turn(
                &child.session_id,
                TurnInput {
                    items: vec![InputItem::Text {
                        text: "child turn".to_string(),
                    }],
                    image_blobs: HashMap::new(),
                    protocol_turn_options: None,
                    trace_turn_id: None,
                    protocol_extension: None,
                    turn_context: crate::TurnContext::default(),
                },
            )
            .await
        {
            Ok(turn) => turn,
            Err(err) => return crate::ToolResult::err_fmt(format_args!("{err}")),
        };

        let _ = context.sessions().close_session(&child.session_id).await;
        let _ = turn;
        crate::ToolResult::ok(json!({ "status": "ok" }))
    }
}

fn child_session_tool_definition() -> crate::ToolDefinition {
    crate::ToolDefinition::raw(
        "tool:spawn_child",
        "spawn_child",
        "spawn a child session",
        crate::ToolDefinition::default_input_schema(),
        serde_json::json!({ "type": "object", "additionalProperties": true }),
    )
}

pub(crate) async fn standard_runtime_with_transport_and_host(
    transport: TestProvider,
    host: EmbeddedRuntimeHost,
) -> LashRuntime {
    let tools: Arc<dyn crate::ToolProvider> = Arc::new(EmptyTools);
    let mut runtime = LashRuntime::from_embedded_state(
        standard_test_policy(),
        host,
        crate::RuntimeServices::new(plugin_session_with_tools("root", tools)),
        RuntimeSessionState::default(),
    )
    .await
    .expect("runtime");
    runtime.policy.provider = transport.clone().into_handle();
    runtime
}

pub(crate) async fn standard_runtime_with_bridge(
    transport: TestProvider,
    turn_injection_bridge: crate::TurnInjectionBridge,
) -> LashRuntime {
    let tools: Arc<dyn crate::ToolProvider> = Arc::new(EmptyTools);
    let mut runtime = LashRuntime::from_embedded_state(
        standard_test_policy(),
        test_host_config(),
        crate::RuntimeServices::new_with_bridges(
            plugin_session_with_tools("root", tools),
            turn_injection_bridge,
            crate::TurnInputInjectionBridge::new(),
        ),
        RuntimeSessionState::default(),
    )
    .await
    .expect("runtime");
    runtime.policy.provider = transport.clone().into_handle();
    runtime
}

pub(crate) async fn standard_runtime_with_input_bridge(
    transport: TestProvider,
    turn_input_injection_bridge: crate::TurnInputInjectionBridge,
) -> LashRuntime {
    let tools: Arc<dyn crate::ToolProvider> = Arc::new(EmptyTools);
    let mut runtime = LashRuntime::from_embedded_state(
        standard_test_policy(),
        test_host_config(),
        crate::RuntimeServices::new_with_bridges(
            plugin_session_with_tools("root", tools),
            crate::TurnInjectionBridge::new(),
            turn_input_injection_bridge,
        ),
        RuntimeSessionState::default(),
    )
    .await
    .expect("runtime");
    runtime.policy.provider = transport.clone().into_handle();
    runtime
}
