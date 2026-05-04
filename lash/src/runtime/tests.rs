use super::*;
use serde_json::json;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::llm::transport::LlmTransportError;
use crate::llm::types::{LlmProviderTraceEvent, LlmUsage};
use crate::plugin::StaticPluginFactory;
use crate::testing::TestProvider;
use tokio_util::sync::CancellationToken;

fn default_state() -> PersistedSessionState {
    PersistedSessionState::default()
}

#[test]
fn stream_fallback_merges_adjacent_display_reasoning_chunks() {
    let mut fallback = StandardStreamFallback::default();
    fallback.push_reasoning("I'll".to_string(), None, Vec::new(), None);
    fallback.push_reasoning(" check".to_string(), None, Vec::new(), None);
    fallback.push_reasoning(" the time.".to_string(), None, Vec::new(), None);

    assert_eq!(fallback.parts.len(), 1);
    assert!(matches!(
        &fallback.parts[0],
        LlmOutputPart::Reasoning { text, .. } if text == "I'll check the time."
    ));
}

trait ReadModelState {
    fn read_model(&self) -> crate::session_graph::SessionReadModel;
}

impl ReadModelState for SessionStateEnvelope {
    fn read_model(&self) -> crate::session_graph::SessionReadModel {
        self.read_model()
    }
}

impl ReadModelState for PersistedSessionState {
    fn read_model(&self) -> crate::session_graph::SessionReadModel {
        self.read_model()
    }
}

trait ReadModelStateMut: ReadModelState {
    fn append_message(&mut self, message: Message);
}

impl ReadModelStateMut for SessionStateEnvelope {
    fn append_message(&mut self, message: Message) {
        self.session_graph.append_message(message);
    }
}

impl ReadModelStateMut for PersistedSessionState {
    fn append_message(&mut self, message: Message) {
        self.session_graph.append_message(message);
    }
}

fn active_conversation_messages(state: &impl ReadModelState) -> Vec<Message> {
    state.read_model().messages.as_ref().clone()
}

fn active_tool_calls(state: &impl ReadModelState) -> Vec<ToolCallRecord> {
    state.read_model().tool_calls.as_ref().clone()
}

fn append_message(state: &mut impl ReadModelStateMut, message: Message) {
    state.append_message(message);
}

#[derive(Clone, Default)]
struct RecordingSink {
    events: Arc<Mutex<Vec<SessionEvent>>>,
}

#[async_trait::async_trait]
impl EventSink for RecordingSink {
    async fn emit(&self, event: SessionEvent) {
        self.events.lock().expect("lock sink").push(event);
    }
}

impl RecordingSink {
    fn snapshot(&self) -> Vec<SessionEvent> {
        self.events.lock().expect("lock sink").clone()
    }
}

#[derive(Default)]
struct RecordingStore {
    session_head_meta: Mutex<Option<crate::SessionHeadMeta>>,
    session_meta: Mutex<Option<crate::SessionMeta>>,
    session_graph: Mutex<crate::SessionGraph>,
    checkpoint: Mutex<Option<crate::HydratedSessionCheckpoint>>,
    usage_deltas: Mutex<Vec<crate::TokenLedgerEntry>>,
}

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
            dynamic_state_ref: commit.checkpoint.dynamic_state_ref.clone(),
            plugin_snapshot_ref: commit.checkpoint.plugin_snapshot_ref.clone(),
            plugin_snapshot_revision: commit.checkpoint.plugin_snapshot_revision,
            execution_state_ref: commit.checkpoint.execution_state_ref.clone(),
        };
        *self.checkpoint.lock().expect("lock checkpoint") = Some(commit.checkpoint);
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
        Ok(crate::store::RuntimeCommitResult {
            head_revision,
            checkpoint_ref,
            manifest,
        })
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

impl RecordingStore {
    async fn save_session_head_meta(&self, meta: crate::SessionHeadMeta) {
        *self.session_head_meta.lock().expect("lock store") = Some(meta);
    }
}

#[derive(Debug)]
struct MockCall {
    stream_events: Vec<LlmStreamEvent>,
    response: Result<LlmResponse, LlmTransportError>,
}

fn mock_provider(calls: Vec<MockCall>) -> TestProvider {
    let calls = Arc::new(Mutex::new(calls));
    TestProvider::builder()
        .kind("mock")
        .default_model("mock-model")
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

fn standard_test_policy() -> SessionPolicy {
    SessionPolicy {
        execution_mode: ExecutionMode::standard(),
        provider: mock_provider(Vec::new()).into_handle(),
        model: "mock-model".to_string(),
        max_context_tokens: Some(200_000),
        ..SessionPolicy::default()
    }
}

fn test_host_config() -> EmbeddedRuntimeHost {
    EmbeddedRuntimeHost::new(RuntimeCoreConfig::default())
}

fn test_host_config_with_trace_path(path: PathBuf) -> EmbeddedRuntimeHost {
    EmbeddedRuntimeHost::new(RuntimeCoreConfig::default().with_trace_jsonl_path(Some(path)))
}

fn test_host_config_with_trace_path_and_stream_events(path: PathBuf) -> EmbeddedRuntimeHost {
    EmbeddedRuntimeHost::new(
        RuntimeCoreConfig::default()
            .with_trace_jsonl_path(Some(path))
            .with_trace_level(lash_trace::TraceLevel::Extended),
    )
}

#[derive(Clone, Default)]
struct RecordingSessionStoreFactory {
    stores: Arc<StdMutex<Vec<Arc<RecordingStore>>>>,
}

impl RecordingSessionStoreFactory {
    fn stores(&self) -> Vec<Arc<RecordingStore>> {
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
            model: request.policy.model.clone(),
            cwd: None,
            parent_session_id: request.parent_session_id.clone(),
        });
        self.stores
            .lock()
            .expect("store factory")
            .push(Arc::clone(&store));
        Ok(store as Arc<dyn crate::store::RuntimePersistence>)
    }
}

fn plugin_session_with_tools(
    session_id: &str,
    mode: ExecutionMode,
    tools: Arc<dyn crate::ToolProvider>,
) -> Arc<crate::PluginSession> {
    let tool_factory = StaticPluginFactory::new(
        "test_tools",
        crate::PluginSpec::new().with_tool_provider(Arc::clone(&tools)),
    );
    crate::PluginHost::new(vec![Arc::new(tool_factory)])
        .build_session(
            session_id,
            mode.clone(),
            (mode == crate::ExecutionMode::standard())
                .then(crate::StandardContextApproach::default),
            None,
        )
        .expect("plugins")
}

#[cfg(feature = "tool-impls")]
fn default_tool_session(
    session_id: &str,
    mode: ExecutionMode,
    enable_user_prompts: bool,
) -> Arc<crate::PluginSession> {
    let mut factories: Vec<Arc<dyn crate::PluginFactory>> = vec![
        Arc::new(crate::BuiltinToolResultProjectionPluginFactory::default()),
        Arc::new(crate::testing::FakeStandardContextApproachPluginFactory::rolling_history()),
        Arc::new(crate::testing::FakeStandardContextApproachPluginFactory::observational_memory()),
        Arc::new(StaticPluginFactory::new(
            "shell",
            crate::PluginSpec::new()
                .with_tool_provider(Arc::new(crate::tools::StandardShell::new()))
                .with_prompt_contributor(Arc::new(move |_ctx| {
                    Box::pin(async move { Ok(crate::tools::shell_prompt_contributions()) })
                })),
        )),
        Arc::new(StaticPluginFactory::new(
            "apply_patch",
            crate::PluginSpec::new().with_tool_provider(Arc::new(crate::tools::ApplyPatchTool)),
        )),
        Arc::new(crate::tools::ReadFilePluginFactory::new(None)),
        Arc::new(StaticPluginFactory::new(
            "glob",
            crate::PluginSpec::new().with_tool_provider(Arc::new(crate::tools::Glob)),
        )),
        Arc::new(StaticPluginFactory::new(
            "grep",
            crate::PluginSpec::new().with_tool_provider(Arc::new(crate::tools::Grep::new())),
        )),
        Arc::new(StaticPluginFactory::new(
            "ls",
            crate::PluginSpec::new().with_tool_provider(Arc::new(crate::tools::Ls)),
        )),
    ];
    if enable_user_prompts {
        factories.push(Arc::new(StaticPluginFactory::new(
            "ask",
            crate::PluginSpec::new().with_tool_provider(Arc::new(crate::tools::AskTool::new())),
        )));
    }
    crate::PluginHost::new(factories)
        .build_session(
            session_id,
            mode.clone(),
            (mode == crate::ExecutionMode::standard())
                .then(crate::StandardContextApproach::default),
            None,
        )
        .expect("plugins")
}

struct EmptyTools;

#[async_trait::async_trait]
impl crate::ToolProvider for EmptyTools {
    fn definitions(&self) -> Vec<crate::ToolDefinition> {
        Vec::new()
    }

    async fn execute(&self, _name: &str, _args: &serde_json::Value) -> crate::ToolResult {
        crate::ToolResult::err(serde_json::json!("Unknown tool"))
    }
}

async fn standard_runtime_with_transport(transport: TestProvider) -> LashRuntime {
    let tools: Arc<dyn crate::ToolProvider> = Arc::new(EmptyTools);
    let mut runtime = LashRuntime::from_embedded_state(
        standard_test_policy(),
        test_host_config(),
        crate::RuntimeServices::new(plugin_session_with_tools(
            "root",
            ExecutionMode::standard(),
            tools,
        )),
        PersistedSessionState::default(),
    )
    .await
    .expect("runtime");
    runtime.policy.provider = transport.clone().into_handle();
    runtime
}

#[test]
fn plugin_host_rejects_observational_memory_without_supporting_plugin() {
    let host = crate::PluginHost::new(vec![Arc::new(
        crate::testing::FakeStandardContextApproachPluginFactory::rolling_history(),
    )]);
    let result = host.build_session(
        "root",
        ExecutionMode::standard(),
        Some(crate::StandardContextApproach::ObservationalMemory(
            crate::ObservationalMemoryConfig::default(),
        )),
        None,
    );
    let err = match result {
        Ok(_) => panic!("OM should require supporting plugin"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains(
            "standard context approach `observational_memory` requires a supporting plugin factory"
        ),
        "unexpected error: {err}"
    );
}

#[test]
fn plugin_host_rejects_standard_context_for_rlm_sessions() {
    let host = crate::PluginHost::new(vec![Arc::new(
        crate::testing::FakeStandardContextApproachPluginFactory::rolling_history(),
    )]);
    let result = host.build_session(
        "root",
        ExecutionMode::new("rlm"),
        Some(crate::StandardContextApproach::default()),
        None,
    );
    let err = match result {
        Ok(_) => panic!("RLM sessions should not accept a standard context approach"),
        Err(err) => err,
    };
    assert!(
        err.to_string()
            .contains("standard context approach only applies to standard execution mode"),
        "unexpected error: {err}"
    );
}

async fn standard_runtime_with_transport_and_background(transport: TestProvider) -> LashRuntime {
    let tools: Arc<dyn crate::ToolProvider> = Arc::new(EmptyTools);
    let host = BackgroundRuntimeHost::new(
        test_host_config(),
        Arc::new(TokioSessionTaskExecutor::default()),
    );
    let mut runtime = LashRuntime::from_background_state(
        standard_test_policy(),
        host,
        crate::RuntimeServices::new(plugin_session_with_tools(
            "root",
            ExecutionMode::standard(),
            tools,
        )),
        PersistedSessionState::default(),
    )
    .await
    .expect("runtime");
    runtime.policy.provider = transport.clone().into_handle();
    runtime
}

async fn standard_runtime_with_shared_background_executor(
    transport: TestProvider,
    executor: Arc<dyn SessionTaskExecutor>,
) -> LashRuntime {
    let tools: Arc<dyn crate::ToolProvider> = Arc::new(EmptyTools);
    let host = BackgroundRuntimeHost::new(test_host_config(), executor);
    let mut runtime = LashRuntime::from_background_state(
        standard_test_policy(),
        host,
        crate::RuntimeServices::new(plugin_session_with_tools(
            "root",
            ExecutionMode::standard(),
            tools,
        )),
        PersistedSessionState::default(),
    )
    .await
    .expect("runtime");
    runtime.policy.provider = transport.clone().into_handle();
    runtime
}

async fn standard_runtime_with_transport_and_host(
    transport: TestProvider,
    host: EmbeddedRuntimeHost,
) -> LashRuntime {
    let tools: Arc<dyn crate::ToolProvider> = Arc::new(EmptyTools);
    let mut runtime = LashRuntime::from_embedded_state(
        standard_test_policy(),
        host,
        crate::RuntimeServices::new(plugin_session_with_tools(
            "root",
            ExecutionMode::standard(),
            tools,
        )),
        PersistedSessionState::default(),
    )
    .await
    .expect("runtime");
    runtime.policy.provider = transport.clone().into_handle();
    runtime
}

#[tokio::test]
async fn runtime_requires_explicit_max_context_tokens() {
    let tools: Arc<dyn crate::ToolProvider> = Arc::new(EmptyTools);
    let result = LashRuntime::from_embedded_state(
        SessionPolicy {
            execution_mode: ExecutionMode::standard(),
            provider: mock_provider(Vec::new()).into_handle(),
            model: "mock-model".to_string(),
            max_context_tokens: None,
            ..SessionPolicy::default()
        },
        test_host_config(),
        crate::RuntimeServices::new(plugin_session_with_tools(
            "root",
            ExecutionMode::standard(),
            tools,
        )),
        PersistedSessionState::default(),
    )
    .await;
    match result {
        Err(SessionError::Protocol(message)) => {
            assert!(message.contains("max_context_tokens"));
        }
        Err(other) => panic!("unexpected session error: {other}"),
        Ok(_) => panic!("runtime should reject implicit model metadata"),
    }
}

#[cfg(feature = "tool-impls")]
fn rlm_test_policy() -> SessionPolicy {
    SessionPolicy {
        execution_mode: ExecutionMode::new("rlm"),
        provider: mock_provider(Vec::new()).into_handle(),
        model: "mock-model".to_string(),
        max_context_tokens: Some(200_000),
        standard_context_approach: None,
        ..SessionPolicy::default()
    }
}

#[cfg(feature = "tool-impls")]
async fn rlm_mode_with_transport(transport: TestProvider) -> LashRuntime {
    let plugins = default_tool_session("root", ExecutionMode::new("rlm"), true);
    let mut runtime = LashRuntime::from_embedded_state(
        rlm_test_policy(),
        test_host_config(),
        crate::RuntimeServices::new_with_bridges(
            plugins,
            crate::TurnInjectionBridge::new(),
            crate::TurnInputInjectionBridge::new(),
        ),
        PersistedSessionState::from_state(SessionStateEnvelope {
            policy: SessionPolicy {
                execution_mode: ExecutionMode::new("rlm"),
                standard_context_approach: None,
                ..Default::default()
            },
            ..SessionStateEnvelope::default()
        }),
    )
    .await
    .expect("runtime");
    runtime.policy.provider = transport.clone().into_handle();
    runtime
}

#[cfg(feature = "tool-impls")]
async fn rlm_mode_with_transport_and_store(
    transport: TestProvider,
    store: Arc<dyn crate::store::RuntimePersistence>,
) -> LashRuntime {
    let plugins = default_tool_session("root", ExecutionMode::new("rlm"), true);
    let mut runtime = LashRuntime::from_persistent_embedded_state(
        rlm_test_policy(),
        test_host_config(),
        crate::PersistentRuntimeServices::new_with_bridges(
            plugins,
            crate::TurnInjectionBridge::new(),
            crate::TurnInputInjectionBridge::new(),
            store,
        ),
        PersistedSessionState::from_state(SessionStateEnvelope {
            policy: SessionPolicy {
                execution_mode: ExecutionMode::new("rlm"),
                standard_context_approach: None,
                ..Default::default()
            },
            ..SessionStateEnvelope::default()
        }),
    )
    .await
    .expect("runtime");
    runtime.policy.provider = transport.clone().into_handle();
    runtime
}

#[cfg(feature = "tool-impls")]
#[tokio::test]
async fn active_tool_catalog_uses_runtime_execution_mode() {
    let runtime = LashRuntime::from_embedded_state(
        rlm_test_policy(),
        test_host_config(),
        crate::RuntimeServices::new(default_tool_session(
            "root",
            ExecutionMode::new("rlm"),
            false,
        )),
        PersistedSessionState::from_state(SessionStateEnvelope {
            policy: SessionPolicy {
                execution_mode: ExecutionMode::new("rlm"),
                standard_context_approach: None,
                ..Default::default()
            },
            ..SessionStateEnvelope::default()
        }),
    )
    .await
    .expect("runtime");
    let catalog = runtime.active_tool_catalog();
    let names: Vec<&str> = catalog
        .iter()
        .filter_map(|item| item.get("name").and_then(|value| value.as_str()))
        .collect();
    assert!(names.contains(&"exec_command"));
    assert!(names.contains(&"start_command"));
    assert!(names.contains(&"write_stdin"));
    assert!(!names.contains(&"shell_wait"));
    assert!(!names.contains(&"shell_read"));
}

async fn standard_runtime_with_bridge(
    transport: TestProvider,
    turn_injection_bridge: crate::TurnInjectionBridge,
) -> LashRuntime {
    let tools: Arc<dyn crate::ToolProvider> = Arc::new(EmptyTools);
    let mut runtime = LashRuntime::from_embedded_state(
        standard_test_policy(),
        test_host_config(),
        crate::RuntimeServices::new_with_bridges(
            plugin_session_with_tools("root", ExecutionMode::standard(), tools),
            turn_injection_bridge,
            crate::TurnInputInjectionBridge::new(),
        ),
        PersistedSessionState::default(),
    )
    .await
    .expect("runtime");
    runtime.policy.provider = transport.clone().into_handle();
    runtime
}

async fn standard_runtime_with_input_bridge(
    transport: TestProvider,
    turn_input_injection_bridge: crate::TurnInputInjectionBridge,
) -> LashRuntime {
    let tools: Arc<dyn crate::ToolProvider> = Arc::new(EmptyTools);
    let mut runtime = LashRuntime::from_embedded_state(
        standard_test_policy(),
        test_host_config(),
        crate::RuntimeServices::new_with_bridges(
            plugin_session_with_tools("root", ExecutionMode::standard(), tools),
            crate::TurnInjectionBridge::new(),
            turn_input_injection_bridge,
        ),
        PersistedSessionState::default(),
    )
    .await
    .expect("runtime");
    runtime.policy.provider = transport.clone().into_handle();
    runtime
}

type RuntimeTestPluginBuilder = dyn Fn(&crate::PluginSessionContext) -> Result<Arc<dyn crate::SessionPlugin>, crate::PluginError>
    + Send
    + Sync;
type RuntimeExternalRegistrar =
    dyn Fn(&mut crate::PluginRegistrar) -> Result<(), crate::PluginError> + Send + Sync;

struct RuntimeTestPluginFactory {
    build: Arc<RuntimeTestPluginBuilder>,
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

struct RuntimeTestPlugin {
    before_turn: Option<crate::plugin::BeforeTurnHook>,
    checkpoint: Option<crate::plugin::CheckpointHook>,
    tool_result_projectors: Vec<(
        crate::ToolResultProjectionHook,
        crate::plugin::ToolResultProjector,
    )>,
    runtime_event: Option<crate::plugin::PluginRuntimeEventHook>,
    external_registrar: Option<Arc<RuntimeExternalRegistrar>>,
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
        for (hook, projector) in &self.tool_result_projectors {
            reg.tool_results().projector(*hook, Arc::clone(projector))?;
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

async fn runtime_with_plugins(
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

async fn runtime_with_plugins_and_tools(
    plugins: Vec<Arc<dyn crate::PluginFactory>>,
    tools: Arc<dyn crate::ToolProvider>,
    transport: TestProvider,
) -> LashRuntime {
    runtime_with_plugins_and_tools_and_host(plugins, tools, transport, test_host_config()).await
}

async fn runtime_with_plugins_and_tools_and_host(
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
    let plugin_session = plugin_host
        .build_standard_session("root", None)
        .expect("plugins");
    let mut runtime = LashRuntime::from_embedded_state(
        standard_test_policy(),
        host,
        crate::RuntimeServices::new(plugin_session),
        PersistedSessionState::default(),
    )
    .await
    .expect("runtime");
    runtime.policy.provider = transport.clone().into_handle();
    runtime
}

struct EchoTool;

#[async_trait::async_trait]
impl crate::ToolProvider for EchoTool {
    fn definitions(&self) -> Vec<crate::ToolDefinition> {
        vec![crate::ToolDefinition::new(
            "echo_tool",
            "Return a tool payload",
            serde_json::json!({
                "type": "object",
                "properties": { "value": { "type": "string" } },
                "required": ["value"],
                "additionalProperties": false
            }),
            serde_json::json!({ "type": "object", "additionalProperties": true }),
        )]
    }

    async fn execute(&self, tool_name: &str, args: &serde_json::Value) -> crate::ToolResult {
        assert_eq!(tool_name, "echo_tool");
        let value = args
            .get("value")
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        crate::ToolResult::ok(serde_json::json!({
            "payload": format!("raw:{value}")
        }))
    }
}

/// Tool that sleeps for 10 seconds unless its future is aborted or the
/// execution-context cancellation token fires. Used to verify that turn
/// cancellation unwinds in-flight tool tasks promptly.
struct SlowTool {
    observed_cancel: Arc<AtomicBool>,
}

#[async_trait::async_trait]
impl crate::ToolProvider for SlowTool {
    fn definitions(&self) -> Vec<crate::ToolDefinition> {
        vec![crate::ToolDefinition::new(
            "slow_tool",
            "Sleep for a long time; respects cancellation.",
            crate::ToolDefinition::default_input_schema(),
            serde_json::json!({ "type": "object", "additionalProperties": true }),
        )]
    }

    async fn execute(&self, _name: &str, _args: &serde_json::Value) -> crate::ToolResult {
        tokio::time::sleep(std::time::Duration::from_secs(10)).await;
        crate::ToolResult::ok(serde_json::json!({"status": "completed"}))
    }

    async fn execute_streaming_with_context(
        &self,
        _name: &str,
        _args: &serde_json::Value,
        context: &crate::ToolExecutionContext,
        _progress: Option<&crate::ProgressSender>,
    ) -> crate::ToolResult {
        let observed = Arc::clone(&self.observed_cancel);
        if let Some(token) = context.cancellation_token.as_ref() {
            let token = token.clone();
            tokio::select! {
                _ = token.cancelled() => {
                    observed.store(true, Ordering::SeqCst);
                    crate::ToolResult::err_fmt("cancelled")
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
                tool_result_projectors: Vec::new(),
                runtime_event: Some(Arc::new(move |event| {
                    let observed = Arc::clone(&observed);
                    Box::pin(async move {
                        if let crate::plugin::PluginRuntimeEvent::SessionConfigChanged(ctx) = event
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
        .default_model("alt-model")
        .complete_error("alt provider not wired")
        .build();
    runtime
        .update_session_config(
            Some(alt_provider.into_handle()),
            Some("alt-model".to_string()),
            Some(None),
            Some(123_456),
        )
        .await;

    let changes = observed.lock().await;
    assert_eq!(changes.len(), 1);
    let (previous, current) = &changes[0];
    assert_eq!(previous.provider.kind(), "mock");
    assert_eq!(current.provider.kind(), "alt");
    assert_eq!(current.model, "alt-model");
    assert_ne!(previous.max_context_tokens, current.max_context_tokens);
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
                tool_result_projectors: Vec::new(),
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
                user_input: None,
                mode: None,
                mode_turn_options: None,
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
async fn normal_turn_preserves_user_input_provenance_in_state() {
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
                user_input: Some(crate::UserInputProvenance {
                    display_text: "/yolopush".to_string(),
                    effective_text: "/yolopush\n\n<skill>\nbody\n</skill>".to_string(),
                    transforms: vec![crate::UserInputTransform::SkillBlockAppend {
                        skill_name: "yolopush".to_string(),
                        skill_path: "/tmp/yolopush/SKILL.md".to_string(),
                    }],
                }),
                mode: None,
                mode_turn_options: None,
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
        user_message
            .user_input
            .as_ref()
            .map(|input| input.display_text.as_str()),
        Some("/yolopush")
    );
    assert_eq!(
        user_message
            .user_input
            .as_ref()
            .map(|input| input.effective_text.as_str()),
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
                user_input: None,
                mode: None,
                mode_turn_options: None,
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
async fn bridge_checkpoint_injection_continues_standard_turn() {
    let bridge = crate::TurnInputInjectionBridge::new();
    bridge
        .enqueue(vec![crate::InjectedTurnInput {
            message: crate::PluginMessage::text(crate::MessageRole::User, "one more thing"),
        }])
        .expect("enqueue");
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
    let mut runtime = standard_runtime_with_input_bridge(transport, bridge).await;

    let turn = runtime
        .run_turn_assembled(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "hello".to_string(),
                }],
                image_blobs: HashMap::new(),
                user_input: None,
                mode: None,
                mode_turn_options: None,
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
async fn bridge_checkpoint_injection_preserves_images() {
    let bridge = crate::TurnInjectionBridge::new();
    bridge
        .enqueue(vec![crate::PluginMessage {
            role: crate::MessageRole::User,
            content: "see image".to_string(),
            parts: vec![
                crate::Part {
                    id: String::new(),
                    kind: crate::PartKind::Image,
                    content: String::new(),
                    attachment: Some(crate::session_model::message::PartAttachment {
                        reference: crate::AttachmentRef {
                            id: crate::AttachmentId::new("test-image"),
                            media_type: crate::MediaType::Image(crate::ImageMediaType::Png),
                            byte_len: 3,
                            width: None,
                            height: None,
                            label: None,
                        },
                    }),
                    tool_call_id: None,
                    tool_name: None,
                    tool_item_id: None,
                    tool_signature: None,
                    prune_state: crate::PruneState::Intact,
                    reasoning_meta: None,
                    response_meta: None,
                },
                crate::Part {
                    id: String::new(),
                    kind: crate::PartKind::Text,
                    content: "see image".to_string(),
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    tool_item_id: None,
                    tool_signature: None,
                    prune_state: crate::PruneState::Intact,
                    reasoning_meta: None,
                    response_meta: None,
                },
            ],
            images: Vec::new(),
            user_input: None,
        }])
        .expect("enqueue");
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
    let mut runtime = standard_runtime_with_bridge(transport, bridge).await;

    let turn = runtime
        .run_turn_assembled(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "hello".to_string(),
                }],
                image_blobs: HashMap::new(),
                user_input: None,
                mode: None,
                mode_turn_options: None,
            },
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    assert!(
        active_conversation_messages(&turn.state)
            .iter()
            .any(|message| {
                message.role == MessageRole::User
                    && message.parts.iter().any(|part| {
                        matches!(part.kind, PartKind::Image) && part.attachment.is_some()
                    })
            })
    );
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
                tool_result_projectors: Vec::new(),
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
                user_input: None,
                mode: None,
                mode_turn_options: None,
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
async fn turn_injection_bridge_accepts_active_turn_input_without_persisting_duplicate_user_message()
{
    let bridge = crate::TurnInputInjectionBridge::new();
    bridge
        .enqueue(vec![crate::InjectedTurnInput {
            message: crate::PluginMessage {
                role: crate::MessageRole::User,
                content: "follow up".to_string(),
                parts: Vec::new(),
                images: Vec::new(),
                user_input: Some(crate::UserInputProvenance {
                    display_text: "follow up".to_string(),
                    effective_text: "follow up".to_string(),
                    transforms: Vec::new(),
                }),
            },
        }])
        .expect("enqueue injected turn input");

    let transport = mock_provider(vec![MockCall {
        stream_events: Vec::new(),
        response: Ok(LlmResponse {
            full_text: "answer".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "answer".to_string(),
                response_meta: None,
            }],
            ..LlmResponse::default()
        }),
    }]);
    let mut runtime = standard_runtime_with_input_bridge(transport, bridge).await;
    let sink = RecordingSink::default();
    let assembled = runtime
        .stream_turn(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "hello".to_string(),
                }],
                image_blobs: HashMap::new(),
                user_input: None,
                mode: None,
                mode_turn_options: None,
            },
            &sink,
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    let mut saw_injected_accept = false;
    for event in sink.snapshot() {
        if let crate::SessionEvent::InjectedTurnInputAccepted { messages, .. } = event {
            saw_injected_accept = messages.iter().any(|message| {
                message.role == crate::MessageRole::User && message.content == "follow up"
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
async fn external_invoke_can_create_session_from_current_snapshot() {
    let plugin = Arc::new(RuntimeTestPluginFactory {
        build: Arc::new(|_| {
            Ok(Arc::new(RuntimeTestPlugin {
                before_turn: None,
                checkpoint: None,
                tool_result_projectors: Vec::new(),
                runtime_event: None,
                external_registrar: Some(Arc::new(|reg| {
                    reg.external().op(
                        crate::ExternalOpDef {
                            name: "test.spawn".to_string(),
                            description: "spawn".to_string(),
                            kind: crate::ExternalOpKind::Command,
                            session_param: crate::SessionParam::Optional,
                            input_schema: json!({}),
                            output_schema: json!({}),
                        },
                        Arc::new(|ctx, _args| {
                            Box::pin(async move {
                                let handle = ctx
                                    .host
                                    .create_session(crate::SessionCreateRequest {
                                        session_id: Some("branched".to_string()),
                                        parent_session_id: None,
                                        start: crate::SessionStartPoint::CurrentSession,
                                        policy: None,
                                        plugin_mode: crate::SessionPluginMode::InheritCurrent,
                                        initial_nodes: vec![crate::SessionAppendNode::message(
                                            crate::PluginMessage::text(
                                                crate::MessageRole::User,
                                                "branch seed",
                                            ),
                                        )],
                                        first_turn_input: None,
                                        tool_access: crate::SessionToolAccess::default(),
            subagent: None,
            context_surface: crate::SessionContextSurface::default(),
                                        mode_extras: crate::ModeExtras::default(),
                                        usage_source: None,
                                    })
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
                tool_item_id: None,
                tool_signature: None,
                prune_state: PruneState::Intact,
                reasoning_meta: None,
                response_meta: None,
            }]
            .into(),
            user_input: None,
            origin: None,
        },
    );

    let result = runtime
        .invoke_external("test.spawn", json!({}), None)
        .await
        .expect("invoke");
    assert!(result.success);
    assert_eq!(
        result
            .result
            .get("session_id")
            .and_then(|value| value.as_str()),
        Some("branched")
    );
    assert_eq!(
        result
            .result
            .get("message_count")
            .and_then(|value| value.as_u64()),
        Some(2)
    );
}

#[tokio::test]
async fn session_manager_can_stream_and_await_child_session_turns() {
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
        .create_session(crate::SessionCreateRequest {
            session_id: Some("child".to_string()),
            parent_session_id: None,
            start: crate::SessionStartPoint::Empty,
            policy: None,
            plugin_mode: crate::SessionPluginMode::InheritCurrent,
            initial_nodes: Vec::new(),
            first_turn_input: None,
            tool_access: crate::SessionToolAccess::default(),
            subagent: None,
            context_surface: crate::SessionContextSurface::default(),
            mode_extras: crate::ModeExtras::default(),
            usage_source: None,
        })
        .await
        .expect("child session");
    let mut turn = manager
        .start_turn_stream(
            &handle.session_id,
            TurnInput {
                items: vec![InputItem::Text {
                    text: "hello".to_string(),
                }],
                image_blobs: HashMap::new(),
                user_input: None,
                mode: None,
                mode_turn_options: None,
            },
        )
        .await
        .expect("child turn");

    let mut saw_stream_event = false;
    while let Some(event) = turn.events.recv().await {
        if matches!(
            event,
            SessionEvent::TextDelta { .. }
                | SessionEvent::Message { .. }
                | SessionEvent::TokenUsage { .. }
                | SessionEvent::Done
        ) {
            saw_stream_event = true;
        }
    }

    let assembled = manager.await_turn(&turn.turn_id).await.expect("assembled");
    assert_eq!(handle.session_id, "child");
    assert_eq!(handle.policy.model, "mock-model");
    assert_eq!(turn.session_id, "child");
    assert_eq!(turn.policy.model, "mock-model");
    assert!(saw_stream_event);
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
                tool_item_id: None,
                tool_signature: None,
                prune_state: PruneState::Intact,
                reasoning_meta: None,
                response_meta: None,
            }]
            .into(),
            user_input: None,
            origin: None,
        },
    );
    runtime.state.iteration = 3;

    let manager = runtime.session_manager().expect("session manager");
    let handle = manager
        .create_session(crate::SessionCreateRequest {
            session_id: Some("child-store".to_string()),
            parent_session_id: Some("root".to_string()),
            start: crate::SessionStartPoint::CurrentSession,
            policy: None,
            plugin_mode: crate::SessionPluginMode::InheritCurrent,
            initial_nodes: Vec::new(),
            first_turn_input: None,
            tool_access: crate::SessionToolAccess::default(),
            subagent: None,
            context_surface: crate::SessionContextSurface::default(),
            mode_extras: crate::ModeExtras::default(),
            usage_source: None,
        })
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
    assert_eq!(meta.parent_session_id.as_deref(), Some("root"));
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
    assert_eq!(turn_state.iteration, 3);
}

#[tokio::test]
async fn session_manager_rejects_duplicate_child_session_ids() {
    let runtime = runtime_with_plugins(Vec::new(), mock_provider(Vec::new())).await;
    let manager = runtime.session_manager().expect("session manager");
    manager
        .create_session(crate::SessionCreateRequest {
            session_id: Some("child".to_string()),
            parent_session_id: None,
            start: crate::SessionStartPoint::Empty,
            policy: None,
            plugin_mode: crate::SessionPluginMode::InheritCurrent,
            initial_nodes: Vec::new(),
            first_turn_input: None,
            tool_access: crate::SessionToolAccess::default(),
            subagent: None,
            context_surface: crate::SessionContextSurface::default(),
            mode_extras: crate::ModeExtras::default(),
            usage_source: None,
        })
        .await
        .expect("first child session");

    let err = manager
        .create_session(crate::SessionCreateRequest {
            session_id: Some("child".to_string()),
            parent_session_id: None,
            start: crate::SessionStartPoint::Empty,
            policy: None,
            plugin_mode: crate::SessionPluginMode::InheritCurrent,
            initial_nodes: Vec::new(),
            first_turn_input: None,
            tool_access: crate::SessionToolAccess::default(),
            subagent: None,
            context_surface: crate::SessionContextSurface::default(),
            mode_extras: crate::ModeExtras::default(),
            usage_source: None,
        })
        .await
        .expect_err("duplicate child session should fail");
    assert!(err.to_string().contains("already exists"));
}

#[tokio::test]
async fn runtime_can_activate_managed_child_session() {
    let mut runtime = runtime_with_plugins(Vec::new(), mock_provider(Vec::new())).await;
    let manager = runtime.session_manager().expect("session manager");
    manager
        .create_session(crate::SessionCreateRequest {
            session_id: Some("child".to_string()),
            parent_session_id: Some(runtime.session_id().to_string()),
            start: crate::SessionStartPoint::Empty,
            policy: None,
            plugin_mode: crate::SessionPluginMode::InheritCurrent,
            initial_nodes: Vec::new(),
            first_turn_input: Some(crate::PluginMessage::text(
                crate::MessageRole::User,
                "run child",
            )),
            tool_access: crate::SessionToolAccess::default(),
            subagent: None,
            context_surface: crate::SessionContextSurface::default(),
            mode_extras: crate::ModeExtras::default(),
            usage_source: Some("test".to_string()),
        })
        .await
        .expect("child session");

    runtime
        .activate_managed_session("child")
        .await
        .expect("activate child");

    assert_eq!(runtime.session_id(), "child");
    let seed = manager
        .take_first_turn_input("child")
        .await
        .expect("seed lookup")
        .expect("seed");
    assert_eq!(seed.content, "run child");
    assert!(
        manager
            .start_turn_stream(
                "child",
                TurnInput {
                    items: vec![InputItem::Text {
                        text: "old manager should not own activated child".to_string(),
                    }],
                    image_blobs: HashMap::new(),
                    user_input: None,
                    mode: None,
                    mode_turn_options: None,
                },
            )
            .await
            .is_err(),
        "activated child runtime should leave the parent manager registry"
    );
}

struct MemoryProbeTool;

#[async_trait::async_trait]
impl crate::ToolProvider for MemoryProbeTool {
    fn definitions(&self) -> Vec<crate::ToolDefinition> {
        vec![crate::ToolDefinition::new(
            "memory_probe",
            "probe",
            crate::ToolDefinition::default_input_schema(),
            serde_json::json!({ "type": "string" }),
        )]
    }

    async fn execute(&self, _name: &str, _args: &serde_json::Value) -> crate::ToolResult {
        crate::ToolResult::ok(json!("ok"))
    }
}

struct ChildSessionTool;

#[async_trait::async_trait]
impl crate::ToolProvider for ChildSessionTool {
    fn definitions(&self) -> Vec<crate::ToolDefinition> {
        vec![crate::ToolDefinition::new(
            "spawn_child",
            "spawn a child session",
            crate::ToolDefinition::default_input_schema(),
            serde_json::json!({ "type": "object", "additionalProperties": true }),
        )]
    }

    async fn execute(&self, _name: &str, _args: &serde_json::Value) -> crate::ToolResult {
        crate::ToolResult::err(json!("session context required"))
    }

    async fn execute_with_context(
        &self,
        _name: &str,
        _args: &serde_json::Value,
        context: &crate::ToolExecutionContext,
    ) -> crate::ToolResult {
        let child = match context
            .host
            .create_session(crate::SessionCreateRequest {
                session_id: Some("subagent-child".to_string()),
                parent_session_id: Some(context.session_id.clone()),
                start: crate::SessionStartPoint::Empty,
                policy: None,
                plugin_mode: crate::SessionPluginMode::InheritCurrent,
                initial_nodes: Vec::new(),
                first_turn_input: None,
                tool_access: crate::SessionToolAccess::default(),
                subagent: None,
                context_surface: crate::SessionContextSurface::default(),
                mode_extras: crate::ModeExtras::default(),
                usage_source: Some("subagent".to_string()),
            })
            .await
        {
            Ok(child) => child,
            Err(err) => return crate::ToolResult::err_fmt(format_args!("{err}")),
        };

        let turn = match context
            .host
            .start_turn_stream(
                &child.session_id,
                TurnInput {
                    items: vec![InputItem::Text {
                        text: "child turn".to_string(),
                    }],
                    image_blobs: HashMap::new(),
                    user_input: None,
                    mode: None,
                    mode_turn_options: None,
                },
            )
            .await
        {
            Ok(turn) => turn,
            Err(err) => return crate::ToolResult::err_fmt(format_args!("{err}")),
        };

        drop(turn.events);

        let result = context.host.await_turn(&turn.turn_id).await;
        let _ = context.host.close_session(&child.session_id).await;
        match result {
            Ok(_) => crate::ToolResult::ok(json!({ "status": "ok" })),
            Err(err) => crate::ToolResult::err_fmt(format_args!("{err}")),
        }
    }

    async fn execute_streaming_with_context(
        &self,
        name: &str,
        args: &serde_json::Value,
        context: &crate::ToolExecutionContext,
        _progress: Option<&crate::ProgressSender>,
    ) -> crate::ToolResult {
        self.execute_with_context(name, args, context).await
    }
}

#[tokio::test]
async fn session_manager_create_session_accepts_custom_context_surface() {
    let runtime = runtime_with_plugins(Vec::new(), mock_provider(Vec::new())).await;
    let manager = runtime.session_manager().expect("session manager");
    let handle = manager
        .create_session(crate::SessionCreateRequest {
            session_id: Some("memory-child".to_string()),
            parent_session_id: None,
            start: crate::SessionStartPoint::Empty,
            policy: None,
            plugin_mode: crate::SessionPluginMode::Fresh,
            initial_nodes: Vec::new(),
            first_turn_input: None,
            tool_access: crate::SessionToolAccess::default(),
            subagent: None,
            context_surface: crate::SessionContextSurface {
                include_base_tools: false,
                tool_providers: vec![Arc::new(MemoryProbeTool)],
                prompt_contributions: vec![crate::PromptContribution::guidance(
                    "Memory Context",
                    "memory child",
                )],
            },
            mode_extras: crate::ModeExtras::default(),
            usage_source: None,
        })
        .await
        .expect("child session");

    let catalog = manager
        .tool_catalog(&handle.session_id)
        .await
        .expect("tool catalog");
    let tool_names = catalog
        .iter()
        .filter_map(|tool| tool.get("name").and_then(|value| value.as_str()))
        .collect::<Vec<_>>();
    assert_eq!(tool_names, vec!["memory_probe"]);
}

#[tokio::test]
async fn inherited_child_session_carries_parent_dynamic_tool_state() {
    let plugin_host = crate::PluginHost::new(vec![Arc::new(StaticPluginFactory::new(
        "memory_probe",
        crate::PluginSpec::new().with_tool_provider(Arc::new(MemoryProbeTool)),
    ))])
    .with_dynamic_tools();
    let plugin_session = plugin_host
        .build_standard_session("root", None)
        .expect("plugins");
    let mut runtime = LashRuntime::from_embedded_state(
        standard_test_policy(),
        test_host_config(),
        crate::RuntimeServices::new(plugin_session),
        PersistedSessionState::default(),
    )
    .await
    .expect("runtime");
    runtime.policy.provider = mock_provider(Vec::new()).into_handle();
    let manager = runtime.session_manager().expect("session manager");
    let mut snapshot = manager
        .dynamic_tool_state("root")
        .await
        .expect("dynamic tool state");
    assert!(snapshot.tools.remove("memory_probe").is_some());
    manager
        .apply_dynamic_tool_state("root", snapshot)
        .await
        .expect("apply dynamic state");

    let handle = manager
        .create_session(crate::SessionCreateRequest {
            session_id: Some("dynamic-child".to_string()),
            parent_session_id: Some("root".to_string()),
            start: crate::SessionStartPoint::Empty,
            policy: None,
            plugin_mode: crate::SessionPluginMode::InheritCurrent,
            initial_nodes: Vec::new(),
            first_turn_input: None,
            tool_access: crate::SessionToolAccess::default(),
            subagent: None,
            context_surface: crate::SessionContextSurface::default(),
            mode_extras: crate::ModeExtras::default(),
            usage_source: None,
        })
        .await
        .expect("child session");

    let catalog = manager
        .tool_catalog(&handle.session_id)
        .await
        .expect("tool catalog");
    let tool_names = catalog
        .iter()
        .filter_map(|tool| tool.get("name").and_then(|value| value.as_str()))
        .collect::<Vec<_>>();
    assert!(
        !tool_names.contains(&"memory_probe"),
        "inherited child should receive the parent's dynamic snapshot, got {tool_names:?}"
    );
}

#[tokio::test]
async fn parent_turn_receives_live_child_token_usage_events() {
    let transport = mock_provider(vec![
        MockCall {
            stream_events: vec![
                LlmStreamEvent::Part(LlmOutputPart::ToolCall {
                    call_id: "tool-1".to_string(),
                    tool_name: "spawn_child".to_string(),
                    input_json: "{}".to_string(),
                    item_id: None,
                    signature: None,
                }),
                LlmStreamEvent::Usage(LlmUsage {
                    input_tokens: 11,
                    output_tokens: 3,
                    cached_input_tokens: 0,
                    reasoning_tokens: 0,
                }),
            ],
            response: Ok(LlmResponse::default()),
        },
        MockCall {
            stream_events: vec![LlmStreamEvent::Usage(LlmUsage {
                input_tokens: 7,
                output_tokens: 2,
                cached_input_tokens: 4,
                reasoning_tokens: 1,
            })],
            response: Ok(LlmResponse {
                full_text: "child session".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "child session".to_string(),
                    response_meta: None,
                }],
                ..LlmResponse::default()
            }),
        },
        MockCall {
            stream_events: Vec::new(),
            response: Ok(LlmResponse {
                full_text: "done".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "done".to_string(),
                    response_meta: None,
                }],
                ..LlmResponse::default()
            }),
        },
    ]);
    let tools: Arc<dyn crate::ToolProvider> = Arc::new(ChildSessionTool);
    let mut runtime = runtime_with_plugins_and_tools(Vec::new(), tools, transport).await;
    let sink = RecordingSink::default();

    let turn = runtime
        .stream_turn(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "run child".to_string(),
                }],
                image_blobs: HashMap::new(),
                user_input: None,
                mode: None,
                mode_turn_options: None,
            },
            &sink,
            CancellationToken::new(),
        )
        .await
        .expect("parent turn");

    assert!(matches!(
        &turn.outcome,
        TurnOutcome::Finished(_) | TurnOutcome::Handoff { .. }
    ));
    let events = sink.snapshot();
    let child_usage_event = events
        .clone()
        .into_iter()
        .find_map(|event| match event {
            SessionEvent::ChildTokenUsage {
                session_id,
                source,
                model,
                usage,
                cumulative,
                ..
            } => Some((session_id, source, model, usage, cumulative)),
            _ => None,
        })
        .unwrap_or_else(|| panic!("child token usage event missing from {events:?}"));
    assert_eq!(child_usage_event.0, "subagent-child");
    assert_eq!(child_usage_event.1, "subagent");
    assert_eq!(child_usage_event.2, "mock-model");
    assert_eq!(child_usage_event.3.input_tokens, 7);
    assert_eq!(child_usage_event.3.output_tokens, 2);
    assert_eq!(child_usage_event.3.cached_input_tokens, 4);
    assert_eq!(child_usage_event.3.reasoning_tokens, 1);
    assert_eq!(child_usage_event.4.cached_input_tokens, 4);

    let usage = runtime.usage_report();
    assert_eq!(usage.by_source["subagent"].input_tokens, 7);
    assert_eq!(usage.by_source["subagent"].output_tokens, 2);
    assert_eq!(usage.by_source["subagent"].cached_input_tokens, 4);
    assert_eq!(usage.by_source["subagent"].reasoning_tokens, 1);
}

#[tokio::test]
async fn parent_turn_keeps_cached_only_child_usage_live() {
    let transport = mock_provider(vec![
        MockCall {
            stream_events: vec![
                LlmStreamEvent::Part(LlmOutputPart::ToolCall {
                    call_id: "tool-1".to_string(),
                    tool_name: "spawn_child".to_string(),
                    input_json: "{}".to_string(),
                    item_id: None,
                    signature: None,
                }),
                LlmStreamEvent::Usage(LlmUsage {
                    input_tokens: 5,
                    output_tokens: 1,
                    cached_input_tokens: 0,
                    reasoning_tokens: 0,
                }),
            ],
            response: Ok(LlmResponse::default()),
        },
        MockCall {
            stream_events: vec![LlmStreamEvent::Usage(LlmUsage {
                input_tokens: 0,
                output_tokens: 0,
                cached_input_tokens: 9,
                reasoning_tokens: 0,
            })],
            response: Ok(LlmResponse {
                full_text: "cached child".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "cached child".to_string(),
                    response_meta: None,
                }],
                ..LlmResponse::default()
            }),
        },
        MockCall {
            stream_events: Vec::new(),
            response: Ok(LlmResponse {
                full_text: "done".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "done".to_string(),
                    response_meta: None,
                }],
                ..LlmResponse::default()
            }),
        },
    ]);
    let tools: Arc<dyn crate::ToolProvider> = Arc::new(ChildSessionTool);
    let mut runtime = runtime_with_plugins_and_tools(Vec::new(), tools, transport).await;
    let sink = RecordingSink::default();

    runtime
        .stream_turn(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "run child".to_string(),
                }],
                image_blobs: HashMap::new(),
                user_input: None,
                mode: None,
                mode_turn_options: None,
            },
            &sink,
            CancellationToken::new(),
        )
        .await
        .expect("parent turn");

    let events = sink.snapshot();
    let child_usage_event = events
        .clone()
        .into_iter()
        .find_map(|event| match event {
            SessionEvent::ChildTokenUsage {
                usage, cumulative, ..
            } => Some((usage, cumulative)),
            _ => None,
        })
        .unwrap_or_else(|| panic!("child token usage event missing from {events:?}"));
    assert_eq!(child_usage_event.0.input_tokens, 0);
    assert_eq!(child_usage_event.0.output_tokens, 0);
    assert_eq!(child_usage_event.0.cached_input_tokens, 9);
    assert_eq!(child_usage_event.0.reasoning_tokens, 0);
    assert_eq!(child_usage_event.1.cached_input_tokens, 9);

    let usage = runtime.usage_report();
    assert_eq!(usage.by_source["subagent"].input_tokens, 0);
    assert_eq!(usage.by_source["subagent"].output_tokens, 0);
    assert_eq!(usage.by_source["subagent"].cached_input_tokens, 9);
    assert_eq!(usage.by_source["subagent"].reasoning_tokens, 0);
}

#[test]
fn assembler_prefers_final_message() {
    let mut assembler = TurnAssembler::default();
    assembler.push(&SessionEvent::TextDelta {
        content: "stream".to_string(),
    });
    assembler.push(&SessionEvent::Message {
        text: "final".to_string(),
        kind: "final".to_string(),
    });
    assembler.push(&SessionEvent::Done);
    let out = assembler.finish(
        default_state().export_state(),
        false,
        None,
        &SanitizerPolicy::default(),
        &TerminationPolicy::default(),
    );
    assert!(matches!(
        &out.outcome,
        TurnOutcome::Finished(_) | TurnOutcome::Handoff { .. }
    ));
    assert!(matches!(
        &out.outcome,
        TurnOutcome::Finished(TurnFinish::AssistantMessage { .. })
    ));
    assert_eq!(out.assistant_output.safe_text, "final");
    assert_eq!(out.assistant_output.raw_text, "final");
    assert_eq!(out.assistant_output.state, OutputState::Usable);
    assert!(!out.has_plugin_visible_output);
}

#[test]
fn assembler_falls_back_to_last_assistant_message_when_stream_output_is_empty() {
    let mut state = default_state();
    append_message(
        &mut state,
        Message {
            id: "m0".to_string(),
            role: MessageRole::Assistant,
            parts: vec![Part {
                id: "m0.p0".to_string(),
                kind: PartKind::Prose,
                content: "stored".to_string(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                tool_item_id: None,
                tool_signature: None,
                prune_state: PruneState::Intact,
                reasoning_meta: None,
                response_meta: None,
            }]
            .into(),
            user_input: None,
            origin: None,
        },
    );
    let mut assembler = TurnAssembler::default();
    assembler.push(&SessionEvent::Done);
    let out = assembler.finish(
        state.export_state(),
        false,
        None,
        &SanitizerPolicy::default(),
        &TerminationPolicy::default(),
    );
    assert!(matches!(
        &out.outcome,
        TurnOutcome::Finished(_) | TurnOutcome::Handoff { .. }
    ));
    assert!(matches!(
        &out.outcome,
        TurnOutcome::Finished(TurnFinish::AssistantMessage { .. })
    ));
    assert_eq!(out.assistant_output.safe_text, "stored");
    assert_eq!(out.assistant_output.raw_text, "stored");
    assert_eq!(out.assistant_output.state, OutputState::Usable);
}

#[test]
fn interrupted_assembler_does_not_reuse_assistant_before_latest_user_input() {
    let mut state = default_state();
    append_message(
        &mut state,
        Message {
            id: "a0".to_string(),
            role: MessageRole::Assistant,
            parts: vec![Part {
                id: "a0.p0".to_string(),
                kind: PartKind::Prose,
                content: "previous assistant answer".to_string(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                tool_item_id: None,
                tool_signature: None,
                prune_state: PruneState::Intact,
                reasoning_meta: None,
                response_meta: None,
            }]
            .into(),
            user_input: None,
            origin: None,
        },
    );
    append_message(
        &mut state,
        Message {
            id: "u1".to_string(),
            role: MessageRole::User,
            parts: vec![Part {
                id: "u1.p0".to_string(),
                kind: PartKind::Text,
                content: "new prompt".to_string(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                tool_item_id: None,
                tool_signature: None,
                prune_state: PruneState::Intact,
                reasoning_meta: None,
                response_meta: None,
            }]
            .into(),
            user_input: Some(crate::UserInputProvenance {
                display_text: "new prompt".to_string(),
                effective_text: "new prompt".to_string(),
                transforms: Vec::new(),
            }),
            origin: None,
        },
    );

    let out = TurnAssembler::default().finish(
        state.export_state(),
        true,
        None,
        &SanitizerPolicy::default(),
        &TerminationPolicy::default(),
    );

    assert!(matches!(
        &out.outcome,
        TurnOutcome::Stopped(TurnStop::Cancelled)
    ));
    assert!(out.assistant_output.safe_text.is_empty());
    assert!(out.assistant_output.raw_text.is_empty());
}

#[test]
fn assembler_prefers_state_output_when_streamed_text_is_a_truncated_prefix() {
    let mut state = default_state();
    append_message(
        &mut state,
        Message {
            id: "m0".to_string(),
            role: MessageRole::Assistant,
            parts: vec![Part {
                id: "m0.p0".to_string(),
                kind: PartKind::Prose,
                content: "You graduated with a degree in Business Administration.".to_string(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                tool_item_id: None,
                tool_signature: None,
                prune_state: PruneState::Intact,
                reasoning_meta: None,
                response_meta: None,
            }]
            .into(),
            user_input: None,
            origin: None,
        },
    );
    let mut assembler = TurnAssembler::default();
    assembler.push(&SessionEvent::TextDelta {
        content: "You graduated with a degree in Business".to_string(),
    });
    assembler.push(&SessionEvent::Done);
    let out = assembler.finish(
        state.export_state(),
        false,
        None,
        &SanitizerPolicy::default(),
        &TerminationPolicy::default(),
    );
    assert_eq!(
        out.assistant_output.safe_text,
        "You graduated with a degree in Business Administration."
    );
    assert_eq!(
        out.assistant_output.raw_text,
        "You graduated with a degree in Business Administration."
    );
    assert_eq!(out.assistant_output.state, OutputState::Usable);
}

#[test]
fn assembler_state_output_excludes_tool_call_payload() {
    // Regression: codex commits an assistant message containing a prose
    // part followed by a tool-call part whose `content` is the raw JSON
    // arguments. On interrupt the assembler falls back to the last
    // assistant message's parts; concatenating EVERY part's content
    // leaks the tool-call JSON into safe_text and the UI then renders it
    // as a literal AssistantText block. Only Text/Prose/Image parts
    // should appear in safe_text.
    let mut state = default_state();
    append_message(
        &mut state,
        Message {
            id: "m0".to_string(),
            role: MessageRole::Assistant,
            parts: vec![
                Part {
                    id: "m0.p0".to_string(),
                    kind: PartKind::Prose,
                    content: "Searching for the relevant code.".to_string(),
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    tool_item_id: None,
                    tool_signature: None,
                    prune_state: PruneState::Intact,
                    reasoning_meta: None,
                    response_meta: None,
                },
                Part {
                    id: "m0.p1".to_string(),
                    kind: PartKind::ToolCall,
                    content:
                        "{\"tool_calls\":[{\"tool\":\"grep\",\"parameters\":{\"query\":\"x\"}}]}"
                            .to_string(),
                    attachment: None,
                    tool_call_id: Some("tc1".to_string()),
                    tool_name: Some("batch".to_string()),
                    tool_item_id: None,
                    tool_signature: None,
                    prune_state: PruneState::Intact,
                    reasoning_meta: None,
                    response_meta: None,
                },
            ]
            .into(),
            user_input: None,
            origin: None,
        },
    );
    let assembler = TurnAssembler::default();
    let out = assembler.finish(
        state.export_state(),
        true,
        None,
        &SanitizerPolicy::default(),
        &TerminationPolicy::default(),
    );
    assert!(matches!(
        &out.outcome,
        TurnOutcome::Stopped(TurnStop::Cancelled)
    ));
    assert_eq!(
        out.assistant_output.safe_text,
        "Searching for the relevant code."
    );
    assert!(!out.assistant_output.raw_text.contains("tool_calls"));
}

#[test]
fn assembler_marks_tool_failure() {
    let mut assembler = TurnAssembler::default();
    assembler.push(&SessionEvent::ToolCall {
        call_id: Some("tc1".to_string()),
        name: "x".to_string(),
        args: serde_json::json!({}),
        result: serde_json::json!({"error": true}),
        success: false,
        duration_ms: 1,
    });
    assembler.push(&SessionEvent::Error {
        message: "tool failed".to_string(),
        envelope: None,
    });
    assembler.push(&SessionEvent::Done);
    let out = assembler.finish(
        default_state().export_state(),
        false,
        None,
        &SanitizerPolicy::default(),
        &TerminationPolicy::default(),
    );
    assert!(matches!(&out.outcome, TurnOutcome::Stopped(_)));
    assert!(matches!(
        &out.outcome,
        TurnOutcome::Stopped(TurnStop::ToolFailure)
    ));
    assert_eq!(out.tool_calls.len(), 1);
}

#[test]
fn assembler_marks_missing_done_as_failure() {
    let mut assembler = TurnAssembler::default();
    assembler.push(&SessionEvent::TextDelta {
        content: "partial".to_string(),
    });
    let out = assembler.finish(
        default_state().export_state(),
        false,
        None,
        &SanitizerPolicy::default(),
        &TerminationPolicy::default(),
    );
    assert!(matches!(&out.outcome, TurnOutcome::Stopped(_)));
    assert!(matches!(
        &out.outcome,
        TurnOutcome::Stopped(TurnStop::RuntimeError)
    ));
}

#[test]
fn assembler_detects_max_turn_message() {
    let mut state = default_state();
    append_message(
        &mut state,
        Message {
            id: "m0".to_string(),
            role: MessageRole::System,
            parts: vec![Part {
                id: "m0.p0".to_string(),
                kind: PartKind::Text,
                content: "Turn limit reached (5).".to_string(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                tool_item_id: None,
                tool_signature: None,
                prune_state: PruneState::Intact,
                reasoning_meta: None,
                response_meta: None,
            }]
            .into(),
            user_input: None,
            origin: None,
        },
    );
    let mut assembler = TurnAssembler::default();
    assembler.push(&SessionEvent::Done);
    let out = assembler.finish(
        state.export_state(),
        false,
        None,
        &SanitizerPolicy::default(),
        &TerminationPolicy::default(),
    );
    assert!(matches!(
        &out.outcome,
        TurnOutcome::Stopped(TurnStop::MaxTurns)
    ));
}

#[test]
fn assembler_tracks_plugin_panel_output() {
    let mut assembler = TurnAssembler::default();
    assembler.push(&SessionEvent::PluginEvent {
        plugin_id: "demo".to_string(),
        event: crate::PluginSurfaceEvent::PanelUpsert {
            key: "panel:1".to_string(),
            title: "TASK BOARD".to_string(),
            content: "1. Inspect\n2. Patch".to_string(),
        },
    });
    assembler.push(&SessionEvent::Done);
    let out = assembler.finish(
        default_state().export_state(),
        false,
        None,
        &SanitizerPolicy::default(),
        &TerminationPolicy::default(),
    );
    assert!(matches!(
        &out.outcome,
        TurnOutcome::Finished(_) | TurnOutcome::Handoff { .. }
    ));
    assert!(out.has_plugin_visible_output);
}

#[test]
fn output_state_empty_output() {
    assert_eq!(classify_output_state("", "", &[]), OutputState::EmptyOutput);
}

#[test]
fn output_state_traceback_only() {
    let raw = "Runtime error: Traceback (most recent call last):\nFile \"rlm_1.py\", line 2, in <module>\nNameError: name 'now' is not defined";
    assert_eq!(
        classify_output_state(raw, "", &[]),
        OutputState::TracebackOnly
    );
}

#[test]
fn output_state_recovered_from_error() {
    let issues = vec![TurnIssue {
        kind: "runtime".to_string(),
        code: Some("example".to_string()),
        message: "something failed".to_string(),
        raw: None,
    }];
    assert_eq!(
        classify_output_state("raw", "usable", &issues),
        OutputState::RecoveredFromError
    );
}

#[test]
fn normalize_items_resolves_relative_paths_with_base_dir() {
    let tmp = tempfile::tempdir().expect("tmpdir");
    let file_path = tmp.path().join("a.txt");
    std::fs::write(&file_path, "x").expect("write");
    let dir_path = tmp.path().join("sub");
    std::fs::create_dir_all(&dir_path).expect("mkdir");

    let items = vec![
        InputItem::FileRef {
            path: "a.txt".to_string(),
        },
        InputItem::DirRef {
            path: "sub".to_string(),
        },
    ];
    let resolver = DefaultPathResolver;
    let out = normalize_input_items(
        &items,
        &HashMap::new(),
        tmp.path(),
        &resolver,
        &crate::InMemoryAttachmentStore::new(),
    )
    .expect("normalized");
    assert_eq!(out.len(), 1);
    match &out[0] {
        NormalizedItem::Text(text) => {
            assert!(text.contains("[file:"));
            assert!(text.contains("[directory:"));
        }
        _ => panic!("expected merged text item"),
    }
}

#[tokio::test]
async fn standard_runtime_assembles_stream_only_text_response() {
    let transport = mock_provider(vec![MockCall {
        stream_events: vec![
            LlmStreamEvent::Delta("What time ".to_string()),
            LlmStreamEvent::Part(LlmOutputPart::Text {
                text: "is it?".to_string(),
                response_meta: None,
            }),
            LlmStreamEvent::Usage(LlmUsage {
                input_tokens: 11,
                output_tokens: 4,
                cached_input_tokens: 0,
                reasoning_tokens: 0,
            }),
        ],
        response: Ok(LlmResponse {
            full_text: "What time is it?".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "What time is it?".to_string(),
                response_meta: None,
            }],
            ..LlmResponse::default()
        }),
    }]);
    let mut runtime = standard_runtime_with_transport(transport).await;
    let sink = RecordingSink::default();

    let turn = runtime
        .stream_turn(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "hi".to_string(),
                }],
                image_blobs: HashMap::new(),
                user_input: None,
                mode: None,
                mode_turn_options: None,
            },
            &sink,
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    assert!(matches!(
        &turn.outcome,
        TurnOutcome::Finished(_) | TurnOutcome::Handoff { .. }
    ));
    assert!(matches!(
        &turn.outcome,
        TurnOutcome::Finished(TurnFinish::AssistantMessage { .. })
    ));
    assert_eq!(turn.assistant_output.safe_text, "What time is it?");

    let streamed_text: String = sink
        .snapshot()
        .into_iter()
        .filter_map(|event| match event {
            SessionEvent::TextDelta { content } => Some(content),
            _ => None,
        })
        .collect();
    assert_eq!(streamed_text, "What time is it?");
}

#[tokio::test]
async fn standard_runtime_recovers_streamed_text_when_final_response_is_empty() {
    let expected =
        "I’m continuing with a type-safety cleanup now: replace the remaining raw JSON paths.";
    let transport = mock_provider(vec![MockCall {
        stream_events: vec![
            LlmStreamEvent::Delta("I’m continuing with a type-safety cleanup now: ".to_string()),
            LlmStreamEvent::Part(LlmOutputPart::Text {
                text: "replace the remaining raw JSON paths.".to_string(),
                response_meta: None,
            }),
        ],
        response: Ok(LlmResponse::default()),
    }]);
    let mut runtime = standard_runtime_with_transport(transport).await;
    let sink = RecordingSink::default();

    let turn = runtime
        .stream_turn(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "continue".to_string(),
                }],
                image_blobs: HashMap::new(),
                user_input: None,
                mode: None,
                mode_turn_options: None,
            },
            &sink,
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    assert!(matches!(
        &turn.outcome,
        TurnOutcome::Finished(_) | TurnOutcome::Handoff { .. }
    ));
    assert!(matches!(
        &turn.outcome,
        TurnOutcome::Finished(TurnFinish::AssistantMessage { .. })
    ));
    assert_eq!(turn.assistant_output.safe_text, expected);
    assert!(turn.errors.is_empty());

    let streamed_text: String = sink
        .snapshot()
        .into_iter()
        .filter_map(|event| match event {
            SessionEvent::TextDelta { content } => Some(content),
            _ => None,
        })
        .collect();
    assert_eq!(streamed_text, expected);
}

#[tokio::test]
async fn standard_runtime_cancels_in_flight_tool_calls_when_token_fires() {
    // Model emits one tool call that would sleep for 10s; we cancel the turn
    // and expect run_tool_calls to tear down promptly (< 2s), either via
    // JoinSet::abort_all or via the tool observing the cancellation token.
    let transport = mock_provider(vec![
        MockCall {
            stream_events: vec![
                LlmStreamEvent::Part(LlmOutputPart::ToolCall {
                    call_id: "slow-1".to_string(),
                    tool_name: "slow_tool".to_string(),
                    input_json: "{}".to_string(),
                    item_id: None,
                    signature: None,
                }),
                LlmStreamEvent::Usage(LlmUsage {
                    input_tokens: 10,
                    output_tokens: 1,
                    cached_input_tokens: 0,
                    reasoning_tokens: 0,
                }),
            ],
            response: Ok(LlmResponse::default()),
        },
        // Extra call not expected to happen — provided as a safety net in case
        // the turn machine makes a second LLM call before noticing cancel.
        MockCall {
            stream_events: Vec::new(),
            response: Ok(LlmResponse {
                full_text: "stopped".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "stopped".to_string(),
                    response_meta: None,
                }],
                ..LlmResponse::default()
            }),
        },
    ]);
    let observed_cancel = Arc::new(AtomicBool::new(false));
    let tools: Arc<dyn crate::ToolProvider> = Arc::new(SlowTool {
        observed_cancel: Arc::clone(&observed_cancel),
    });
    let mut runtime = runtime_with_plugins_and_tools(Vec::new(), tools, transport).await;
    let cancel = CancellationToken::new();
    let cancel_trigger = cancel.clone();
    tokio::spawn(async move {
        // Give the turn time to spawn the slow tool before we cancel.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        cancel_trigger.cancel();
    });

    let start = std::time::Instant::now();
    let _ = runtime
        .run_turn_assembled(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "trigger slow tool".to_string(),
                }],
                image_blobs: HashMap::new(),
                user_input: None,
                mode: None,
                mode_turn_options: None,
            },
            cancel,
        )
        .await;
    let elapsed = start.elapsed();

    assert!(
        elapsed < std::time::Duration::from_secs(2),
        "turn cancellation did not tear down in-flight tool call quickly: elapsed={elapsed:?}"
    );
    // The tool either saw the cancellation token and returned, or its future
    // was aborted by the JoinSet. Either outcome is acceptable — what matters
    // is the prompt return above. We still assert cooperative observation as a
    // stronger signal that the token is now plumbed through to tool context.
    assert!(
        observed_cancel.load(Ordering::SeqCst),
        "slow tool did not observe cancellation token through ToolExecutionContext"
    );
}

#[tokio::test]
async fn standard_runtime_executes_streamed_tool_call_when_final_response_is_empty() {
    let transport = mock_provider(vec![
        MockCall {
            stream_events: vec![
                LlmStreamEvent::Part(LlmOutputPart::ToolCall {
                    call_id: "tool-1".to_string(),
                    tool_name: "echo_tool".to_string(),
                    input_json: r#"{"value":"sample"}"#.to_string(),
                    item_id: None,
                    signature: None,
                }),
                LlmStreamEvent::Usage(LlmUsage {
                    input_tokens: 12,
                    output_tokens: 3,
                    cached_input_tokens: 0,
                    reasoning_tokens: 0,
                }),
            ],
            response: Ok(LlmResponse::default()),
        },
        MockCall {
            stream_events: Vec::new(),
            response: Ok(LlmResponse {
                full_text: "done".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "done".to_string(),
                    response_meta: None,
                }],
                ..LlmResponse::default()
            }),
        },
    ]);
    let tools: Arc<dyn crate::ToolProvider> = Arc::new(EchoTool);
    let mut runtime = runtime_with_plugins_and_tools(Vec::new(), tools, transport).await;

    let turn = runtime
        .run_turn_assembled(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "run the tool".to_string(),
                }],
                image_blobs: HashMap::new(),
                user_input: None,
                mode: None,
                mode_turn_options: None,
            },
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    assert_eq!(turn.assistant_output.safe_text, "done");
    assert_eq!(active_tool_calls(&turn.state).len(), 1);
    assert_eq!(
        active_tool_calls(&turn.state)[0].call_id.as_deref(),
        Some("tool-1")
    );
    assert_eq!(
        active_tool_calls(&turn.state)[0].result,
        serde_json::json!({
            "payload": "raw:sample"
        })
    );
}

#[tokio::test]
async fn standard_runtime_preserves_part_boundaries_when_response_is_not_streamed() {
    let transport = mock_provider(vec![MockCall {
        stream_events: vec![],
        response: Ok(LlmResponse {
            full_text: "Intro paragraph.\n\n## Heading".to_string(),
            parts: vec![
                LlmOutputPart::Text {
                    text: "Intro paragraph.".to_string(),
                    response_meta: None,
                },
                LlmOutputPart::Text {
                    text: "## Heading".to_string(),
                    response_meta: None,
                },
            ],
            ..LlmResponse::default()
        }),
    }]);
    let mut runtime = standard_runtime_with_transport(transport).await;
    let sink = RecordingSink::default();

    let turn = runtime
        .stream_turn(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "hi".to_string(),
                }],
                image_blobs: HashMap::new(),
                user_input: None,
                mode: None,
                mode_turn_options: None,
            },
            &sink,
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    assert_eq!(
        turn.assistant_output.safe_text,
        "Intro paragraph.\n\n## Heading"
    );

    let streamed_text: String = sink
        .snapshot()
        .into_iter()
        .filter_map(|event| match event {
            SessionEvent::TextDelta { content } => Some(content),
            _ => None,
        })
        .collect();
    assert_eq!(streamed_text, "Intro paragraph.\n\n## Heading");
}

#[cfg(feature = "tool-impls")]
#[tokio::test(flavor = "multi_thread")]
async fn runtime_session_manager_forwards_user_prompts_when_available() {
    let transport = mock_provider(Vec::new());
    let runtime = rlm_mode_with_transport(transport).await;
    let prompt_bridge = HostPromptBridge::new();
    let manager = runtime
        .runtime_session_manager_with_prompt_bridge(Some(prompt_bridge.clone()))
        .expect("manager");
    let (prompt_tx, mut prompt_rx) = tokio::sync::mpsc::unbounded_channel::<PendingPrompt>();
    prompt_bridge.set_sender(prompt_tx);

    let prompt_task = tokio::spawn(async move {
        let prompt = prompt_rx.recv().await.expect("prompt");
        assert_eq!(prompt.request.question, "Pick one");
        assert_eq!(
            prompt.request.options,
            vec!["works".to_string(), "done".to_string()]
        );
        prompt
            .response_tx
            .send(crate::PromptResponse::Single {
                selection: "done".to_string(),
                note: None,
            })
            .expect("prompt response");
    });

    let answer = manager
        .prompt_user(crate::PromptRequest::single(
            "Pick one",
            vec!["works".to_string(), "done".to_string()],
        ))
        .await
        .expect("prompt answer");

    prompt_bridge.clear_sender();
    prompt_task.await.expect("prompt task");
    assert_eq!(
        answer,
        crate::PromptResponse::Single {
            selection: "done".to_string(),
            note: None,
        }
    );
}

#[tokio::test]
async fn standard_runtime_uses_streamed_usage_when_final_usage_missing() {
    let transport = mock_provider(vec![MockCall {
        stream_events: vec![
            LlmStreamEvent::Delta("Hi".to_string()),
            LlmStreamEvent::Usage(LlmUsage {
                input_tokens: 9,
                output_tokens: 3,
                cached_input_tokens: 2,
                reasoning_tokens: 0,
            }),
        ],
        response: Ok(LlmResponse {
            full_text: "Hi".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "Hi".to_string(),
                response_meta: None,
            }],
            usage: LlmUsage::default(),
            ..LlmResponse::default()
        }),
    }]);
    let mut runtime = standard_runtime_with_transport(transport).await;

    let turn = runtime
        .run_turn_assembled(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "hello".to_string(),
                }],
                image_blobs: HashMap::new(),
                user_input: None,
                mode: None,
                mode_turn_options: None,
            },
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    assert_eq!(turn.token_usage.input_tokens, 9);
    assert_eq!(turn.token_usage.output_tokens, 3);
    assert_eq!(turn.token_usage.cached_input_tokens, 2);
}

#[tokio::test]
async fn standard_runtime_prefers_final_usage_over_streamed_usage() {
    let transport = mock_provider(vec![MockCall {
        stream_events: vec![
            LlmStreamEvent::Delta("Hi".to_string()),
            LlmStreamEvent::Usage(LlmUsage {
                input_tokens: 9,
                output_tokens: 3,
                cached_input_tokens: 2,
                reasoning_tokens: 0,
            }),
        ],
        response: Ok(LlmResponse {
            full_text: "Hi".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "Hi".to_string(),
                response_meta: None,
            }],
            usage: LlmUsage {
                input_tokens: 12,
                output_tokens: 4,
                cached_input_tokens: 1,
                reasoning_tokens: 0,
            },
            ..LlmResponse::default()
        }),
    }]);
    let mut runtime = standard_runtime_with_transport(transport).await;

    let turn = runtime
        .run_turn_assembled(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "hello".to_string(),
                }],
                image_blobs: HashMap::new(),
                user_input: None,
                mode: None,
                mode_turn_options: None,
            },
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    assert_eq!(turn.token_usage.input_tokens, 12);
    assert_eq!(turn.token_usage.output_tokens, 4);
    assert_eq!(turn.token_usage.cached_input_tokens, 1);
}

#[tokio::test]
async fn standard_runtime_trace_records_stream_event_entries() {
    let transport = mock_provider(vec![MockCall {
        stream_events: vec![
            LlmStreamEvent::Delta("Hello ".to_string()),
            LlmStreamEvent::Part(LlmOutputPart::Text {
                text: "world".to_string(),
                response_meta: None,
            }),
            LlmStreamEvent::Usage(LlmUsage {
                input_tokens: 10,
                output_tokens: 2,
                cached_input_tokens: 0,
                reasoning_tokens: 0,
            }),
        ],
        response: Ok(LlmResponse {
            full_text: "Hello world".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "Hello world".to_string(),
                response_meta: None,
            }],
            ..LlmResponse::default()
        }),
    }]);
    let trace_path = std::env::temp_dir().join(format!(
        "lash-standard-trace-{}-{}.jsonl",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    let mut runtime = standard_runtime_with_transport_and_host(
        transport,
        test_host_config_with_trace_path_and_stream_events(trace_path.clone()),
    )
    .await;

    let turn = runtime
        .run_turn_assembled(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "hello".to_string(),
                }],
                image_blobs: HashMap::new(),
                user_input: None,
                mode: None,
                mode_turn_options: None,
            },
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    assert!(matches!(
        &turn.outcome,
        TurnOutcome::Finished(_) | TurnOutcome::Handoff { .. }
    ));

    let logged = std::fs::read_to_string(&trace_path).expect("read trace");
    let entries = logged
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("json log entry"))
        .collect::<Vec<_>>();

    assert!(
        entries
            .iter()
            .any(|entry| entry.get("type").and_then(|v| v.as_str())
                == Some("runtime_stream_event")
                && entry
                    .get("event")
                    .and_then(|payload| payload.get("event_name"))
                    .and_then(|v| v.as_str())
                    == Some("delta")
                && entry
                    .get("event")
                    .and_then(|payload| payload.get("raw_text"))
                    .and_then(|v| v.as_str())
                    == Some("Hello ")),
        "expected delta stream event in trace: {entries:?}"
    );
    assert!(
        entries
            .iter()
            .any(|entry| entry.get("type").and_then(|v| v.as_str())
                == Some("runtime_stream_event")
                && entry
                    .get("event")
                    .and_then(|payload| payload.get("event_name"))
                    .and_then(|v| v.as_str())
                    == Some("text_part")
                && entry
                    .get("event")
                    .and_then(|payload| payload.get("raw_text"))
                    .and_then(|v| v.as_str())
                    == Some("world")),
        "expected text_part stream event in trace: {entries:?}"
    );
    assert!(
        entries
            .iter()
            .any(|entry| entry.get("type").and_then(|v| v.as_str()) == Some("llm_call_completed")),
        "expected final llm trace entry in trace: {entries:?}"
    );
    let response_entry = entries
        .iter()
        .find(|entry| entry.get("type").and_then(|v| v.as_str()) == Some("llm_call_completed"))
        .expect("completed llm call entry");
    let stream_summary = response_entry
        .get("stream_summary")
        .and_then(|value| value.as_object())
        .expect("stream summary");
    assert_eq!(
        stream_summary
            .get("text_delta_count")
            .and_then(|value| value.as_u64()),
        Some(2)
    );
    assert_eq!(
        stream_summary
            .get("visible_chunk_count")
            .and_then(|value| value.as_u64()),
        Some(2)
    );
    assert_eq!(
        stream_summary
            .get("max_visible_chunk_chars")
            .and_then(|value| value.as_u64()),
        Some(6)
    );
    let avg_chunk_chars = stream_summary
        .get("avg_visible_chunk_chars")
        .and_then(|value| value.as_f64())
        .expect("avg visible chunk chars");
    assert!((avg_chunk_chars - 5.5).abs() < f64::EPSILON);
    assert!(
        stream_summary
            .get("first_visible_token_latency_ms")
            .is_some_and(|value| !value.is_null())
    );
    assert!(
        stream_summary
            .get("stream_duration_ms")
            .is_some_and(|value| !value.is_null())
    );

    let _ = std::fs::remove_file(&trace_path);
}

#[tokio::test]
async fn extended_runtime_trace_records_provider_stream_events() {
    let transport = TestProvider::builder()
        .kind("mock")
        .default_model("mock-model")
        .requires_streaming(true)
        .complete(|req| async move {
            if let Some(tx) = req.provider_trace.as_ref() {
                tx.send(LlmProviderTraceEvent {
                    provider: "codex",
                    event_name: "response.output_item.done".to_string(),
                    raw: serde_json::json!({
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": { "id": "msg_1" }
                    })
                    .to_string(),
                });
                tx.send(LlmProviderTraceEvent {
                    provider: "codex",
                    event_name: "response.output_item.done".to_string(),
                    raw: serde_json::json!({
                        "type": "response.output_item.done",
                        "output_index": 1,
                        "item": { "id": "msg_2" }
                    })
                    .to_string(),
                });
            }
            Ok(LlmResponse {
                full_text: "Hello".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "Hello".to_string(),
                    response_meta: None,
                }],
                ..LlmResponse::default()
            })
        })
        .build();
    let trace_path = std::env::temp_dir().join(format!(
        "lash-provider-trace-{}-{}.jsonl",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    let mut runtime = standard_runtime_with_transport_and_host(
        transport,
        test_host_config_with_trace_path_and_stream_events(trace_path.clone()),
    )
    .await;

    let turn = runtime
        .run_turn_assembled(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "hello".to_string(),
                }],
                image_blobs: HashMap::new(),
                user_input: None,
                mode: None,
                mode_turn_options: None,
            },
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    assert!(matches!(
        &turn.outcome,
        TurnOutcome::Finished(_) | TurnOutcome::Handoff { .. }
    ));

    let logged = std::fs::read_to_string(&trace_path).expect("read trace");
    let entries = logged
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("json log entry"))
        .collect::<Vec<_>>();
    let provider_events = entries
        .iter()
        .filter(|entry| entry.get("type").and_then(|v| v.as_str()) == Some("provider_stream_event"))
        .collect::<Vec<_>>();
    assert_eq!(
        provider_events.len(),
        2,
        "provider trace entries: {entries:?}"
    );
    assert_eq!(provider_events[0]["event"]["item_id"], "msg_1");
    assert_eq!(provider_events[0]["event"]["output_index"], 0);
    assert_eq!(provider_events[1]["event"]["item_id"], "msg_2");
    assert_eq!(
        provider_events[1]["event"]["raw_json"]["item"]["id"],
        "msg_2"
    );
    assert!(
        provider_events[1]["event"]["raw_sha256"]
            .as_str()
            .is_some_and(|hash| !hash.is_empty())
    );

    let _ = std::fs::remove_file(&trace_path);
}

#[tokio::test]
async fn standard_runtime_trace_omits_stream_event_entries_by_default() {
    let transport = mock_provider(vec![MockCall {
        stream_events: vec![
            LlmStreamEvent::Delta("Hello ".to_string()),
            LlmStreamEvent::Part(LlmOutputPart::Text {
                text: "world".to_string(),
                response_meta: None,
            }),
        ],
        response: Ok(LlmResponse {
            full_text: "Hello world".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "Hello world".to_string(),
                response_meta: None,
            }],
            ..LlmResponse::default()
        }),
    }]);
    let trace_path = std::env::temp_dir().join(format!(
        "lash-standard-trace-summary-{}-{}.jsonl",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    let mut runtime = standard_runtime_with_transport_and_host(
        transport,
        test_host_config_with_trace_path(trace_path.clone()),
    )
    .await;

    let turn = runtime
        .run_turn_assembled(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "hello".to_string(),
                }],
                image_blobs: HashMap::new(),
                user_input: None,
                mode: None,
                mode_turn_options: None,
            },
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    assert!(matches!(
        &turn.outcome,
        TurnOutcome::Finished(_) | TurnOutcome::Handoff { .. }
    ));

    let logged = std::fs::read_to_string(&trace_path).expect("read trace");
    let entries = logged
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("json log entry"))
        .collect::<Vec<_>>();

    assert!(
        !entries.iter().any(
            |entry| entry.get("type").and_then(|v| v.as_str()) == Some("runtime_stream_event")
        ),
        "stream event entries should be opt-in: {entries:?}"
    );
    let response_entry = entries
        .iter()
        .find(|entry| entry.get("type").and_then(|v| v.as_str()) == Some("llm_call_completed"))
        .expect("completed llm call entry");
    assert!(
        response_entry
            .get("stream_summary")
            .is_some_and(|value| !value.is_null()),
        "stream summary should remain in completed LLM trace: {response_entry:?}"
    );

    let _ = std::fs::remove_file(&trace_path);
}

#[tokio::test]
async fn standard_runtime_trace_records_failed_llm_calls() {
    let transport = mock_provider(vec![MockCall {
        stream_events: Vec::new(),
        response: Err(crate::llm::transport::LlmTransportError::new(
            "HTTP request failed: builder error",
        )
        .with_code("builder")
        .with_raw("transport raw body")
        .with_request_body("{\"model\":\"mock-model\"}")),
    }]);
    let trace_path = std::env::temp_dir().join(format!(
        "lash-standard-trace-error-{}-{}.jsonl",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    let mut runtime = standard_runtime_with_transport_and_host(
        transport,
        test_host_config_with_trace_path(trace_path.clone()),
    )
    .await;

    let turn = runtime
        .run_turn_assembled(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "hello".to_string(),
                }],
                image_blobs: HashMap::new(),
                user_input: None,
                mode: None,
                mode_turn_options: None,
            },
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    assert!(matches!(&turn.outcome, TurnOutcome::Stopped(_)));
    assert_eq!(turn.errors.len(), 1);
    assert_eq!(turn.errors[0].raw.as_deref(), Some("transport raw body"));

    let logged = std::fs::read_to_string(&trace_path).expect("read trace");
    let entries = logged
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("json log entry"))
        .collect::<Vec<_>>();
    let error_entry = entries
        .iter()
        .find(|entry| entry.get("type").and_then(|v| v.as_str()) == Some("llm_call_failed"))
        .expect("llm error entry");
    assert_eq!(
        error_entry["error"]["message"].as_str(),
        Some("HTTP request failed: builder error")
    );
    assert_eq!(error_entry["error"]["code"].as_str(), Some("builder"));
    assert_eq!(
        error_entry["error"]["raw"].as_str(),
        Some("transport raw body")
    );
    let request_entry = entries
        .iter()
        .find(|entry| entry.get("type").and_then(|v| v.as_str()) == Some("llm_call_started"))
        .expect("llm request entry");
    assert_eq!(
        request_entry["request"]["model"].as_str(),
        Some("mock-model")
    );
}

#[test]
fn normalize_prompt_usage_uses_input_tokens_for_openai_compatible() {
    let usage = TokenUsage {
        input_tokens: 80,
        output_tokens: 0,
        cached_input_tokens: 20,
        reasoning_tokens: 0,
    };
    let stub = mock_provider(Vec::new()).into_handle();
    let prompt_usage = normalize_prompt_usage(&stub, &usage).expect("prompt usage");
    assert_eq!(prompt_usage.prompt_context_tokens, 80);
    assert_eq!(prompt_usage.context_budget_tokens, 80);
}

#[tokio::test]
async fn tool_result_projectors_split_state_model_and_history_views() {
    let committed_results = Arc::new(tokio::sync::Mutex::new(Vec::<(
        serde_json::Value,
        serde_json::Value,
    )>::new()));
    let committed_results_hook = Arc::clone(&committed_results);
    let plugin = Arc::new(RuntimeTestPluginFactory {
        build: Arc::new(move |_| {
            let committed_results = Arc::clone(&committed_results_hook);
            Ok(Arc::new(RuntimeTestPlugin {
                before_turn: None,
                checkpoint: None,
                tool_result_projectors: vec![
                    (
                        crate::ToolResultProjectionHook::BeforeState,
                        Arc::new(|mut ctx| {
                            Box::pin(async move {
                                ctx.result.result = serde_json::json!("state projection");
                                Ok(ctx.result)
                            })
                        }),
                    ),
                    (
                        crate::ToolResultProjectionHook::BeforeModel,
                        Arc::new(|mut ctx| {
                            Box::pin(async move {
                                ctx.result.result = serde_json::json!("model projection");
                                Ok(ctx.result)
                            })
                        }),
                    ),
                    (
                        crate::ToolResultProjectionHook::BeforeHistory,
                        Arc::new(|mut ctx| {
                            Box::pin(async move {
                                ctx.result.result = serde_json::json!("history projection");
                                Ok(ctx.result)
                            })
                        }),
                    ),
                ],
                runtime_event: Some(Arc::new(move |event| {
                    let committed_results = Arc::clone(&committed_results);
                    Box::pin(async move {
                        if let crate::plugin::PluginRuntimeEvent::TurnCommitted(turn) = event {
                            committed_results.lock().await.push((
                                turn.tool_calls
                                    .first()
                                    .map(|call| call.result.clone())
                                    .unwrap_or(serde_json::Value::Null),
                                turn.state
                                    .read_model()
                                    .tool_calls
                                    .first()
                                    .map(|call| call.result.clone())
                                    .unwrap_or(serde_json::Value::Null),
                            ));
                        }
                        Ok(())
                    })
                })),
                external_registrar: None,
            }))
        }),
    });
    let transport = mock_provider(vec![
        MockCall {
            stream_events: Vec::new(),
            response: Ok(LlmResponse {
                parts: vec![
                    LlmOutputPart::Text {
                        text: "checking tool".to_string(),
                        response_meta: None,
                    },
                    LlmOutputPart::ToolCall {
                        call_id: "tool-1".to_string(),
                        tool_name: "echo_tool".to_string(),
                        input_json: r#"{"value":"sample"}"#.to_string(),
                        item_id: None,
                        signature: None,
                    },
                ],
                ..LlmResponse::default()
            }),
        },
        MockCall {
            stream_events: Vec::new(),
            response: Ok(LlmResponse {
                full_text: "done".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "done".to_string(),
                    response_meta: None,
                }],
                ..LlmResponse::default()
            }),
        },
    ]);
    let tools: Arc<dyn crate::ToolProvider> = Arc::new(EchoTool);
    let mut runtime = runtime_with_plugins_and_tools(vec![plugin], tools, transport).await;

    let turn = runtime
        .run_turn_assembled(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "run the tool".to_string(),
                }],
                image_blobs: HashMap::new(),
                user_input: None,
                mode: None,
                mode_turn_options: None,
            },
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    assert!(
        active_conversation_messages(&turn.state)
            .iter()
            .any(|message| {
                message.parts.iter().any(|part| {
                    part.content.contains("model projection")
                        && matches!(part.kind, PartKind::ToolResult)
                })
            })
    );
    let committed = committed_results.lock().await;
    assert_eq!(
        committed.as_slice(),
        &[(
            serde_json::json!("history projection"),
            serde_json::json!("history projection"),
        )]
    );
    assert_eq!(active_tool_calls(&turn.state).len(), 1);
    assert_eq!(
        active_tool_calls(&turn.state)[0].call_id.as_deref(),
        Some("tool-1")
    );
    assert_eq!(turn.tool_calls.len(), 1);
    assert_eq!(turn.tool_calls[0].call_id.as_deref(), Some("tool-1"));
    assert_eq!(
        active_tool_calls(&turn.state)[0].result,
        serde_json::json!("state projection")
    );
    assert_eq!(
        turn.tool_calls[0].result,
        serde_json::json!("state projection")
    );
}

#[tokio::test]
async fn completed_turns_are_persisted_for_custom_runtime_store() {
    let transport = mock_provider(vec![MockCall {
        stream_events: vec![LlmStreamEvent::Delta("Stored answer".to_string())],
        response: Ok(LlmResponse {
            full_text: "Stored answer".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "Stored answer".to_string(),
                response_meta: None,
            }],
            usage: LlmUsage {
                input_tokens: 12,
                output_tokens: 4,
                cached_input_tokens: 1,
                reasoning_tokens: 2,
            },
            ..LlmResponse::default()
        }),
    }]);

    let store = Arc::new(RecordingStore::default());
    let plugins =
        plugin_session_with_tools("root", ExecutionMode::standard(), Arc::new(EmptyTools));
    let mut runtime = LashRuntime::from_persistent_embedded_state(
        standard_test_policy(),
        test_host_config(),
        crate::PersistentRuntimeServices::new(
            Arc::clone(&plugins),
            store.clone() as Arc<dyn crate::store::RuntimePersistence>,
        ),
        PersistedSessionState::default(),
    )
    .await
    .expect("runtime");
    runtime.policy.provider = transport.clone().into_handle();

    let _turn = runtime
        .run_turn_assembled(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "where did this go?".to_string(),
                }],
                image_blobs: HashMap::new(),
                user_input: None,
                mode: None,
                mode_turn_options: None,
            },
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    let read_model = crate::store::RuntimePersistence::load_session(
        store.as_ref(),
        crate::store::SessionReadScope::FullGraph,
    )
    .await
    .expect("load session")
    .expect("session head")
    .graph
    .read_model();
    let messages = read_model.messages.as_slice();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].role, MessageRole::User);
    assert_eq!(messages[0].parts[0].content, "where did this go?");
    assert_eq!(messages[1].role, MessageRole::Assistant);
    assert_eq!(messages[1].parts[0].content, "Stored answer");
}

#[tokio::test]
async fn park_returns_error_when_final_commit_fails() {
    let store = Arc::new(RecordingStore::default());
    store
        .save_session_head_meta(crate::SessionHeadMeta {
            session_id: "other-session".to_string(),
            ..crate::SessionHeadMeta::default()
        })
        .await;
    let plugins = plugin_session_with_tools(
        "park-session",
        ExecutionMode::standard(),
        Arc::new(EmptyTools),
    );
    let runtime = LashRuntime::from_persistent_embedded_state(
        standard_test_policy(),
        test_host_config(),
        crate::PersistentRuntimeServices::new(
            plugins,
            store as Arc<dyn crate::store::RuntimePersistence>,
        ),
        PersistedSessionState {
            session_id: "park-session".to_string(),
            policy: standard_test_policy(),
            ..PersistedSessionState::default()
        },
    )
    .await
    .expect("runtime");

    let err = match runtime.park().await {
        Ok(_) => panic!("park should fail when final persistence fails"),
        Err(err) => err,
    };

    let message = err.to_string();
    assert!(message.contains("failed to persist runtime state"));
    assert!(message.contains("other-session"));
    assert!(message.contains("park-session"));
}

#[tokio::test]
async fn completed_turns_are_persisted_in_session_graph() {
    let transport = mock_provider(vec![MockCall {
        stream_events: vec![
            LlmStreamEvent::Delta("Stored answer".to_string()),
            LlmStreamEvent::Usage(LlmUsage {
                input_tokens: 12,
                output_tokens: 4,
                cached_input_tokens: 1,
                reasoning_tokens: 2,
            }),
        ],
        response: Ok(LlmResponse {
            full_text: "Stored answer".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "Stored answer".to_string(),
                response_meta: None,
            }],
            usage: LlmUsage {
                input_tokens: 12,
                output_tokens: 4,
                cached_input_tokens: 1,
                reasoning_tokens: 2,
            },
            ..LlmResponse::default()
        }),
    }]);

    let store = Arc::new(RecordingStore::default());
    let base_provider: Arc<dyn crate::ToolProvider> = Arc::new(EmptyTools);
    let base_provider_factory = Arc::clone(&base_provider);
    let plugin_host = crate::PluginHost::new(vec![Arc::new(StaticPluginFactory::new(
        "base_tools",
        crate::PluginSpec::new().with_tool_provider(Arc::clone(&base_provider_factory)),
    ))]);
    let plugins = plugin_host
        .build_standard_session("root", None)
        .expect("plugins");
    let mut runtime = LashRuntime::from_persistent_embedded_state(
        standard_test_policy(),
        test_host_config(),
        crate::PersistentRuntimeServices::new(
            Arc::clone(&plugins),
            store.clone() as Arc<dyn crate::store::RuntimePersistence>,
        ),
        PersistedSessionState::default(),
    )
    .await
    .expect("runtime");
    runtime.policy.provider = transport.clone().into_handle();

    let _turn = runtime
        .run_turn_assembled(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "where did this go?".to_string(),
                }],
                image_blobs: HashMap::new(),
                user_input: None,
                mode: None,
                mode_turn_options: None,
            },
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    let read = crate::store::RuntimePersistence::load_session(
        store.as_ref(),
        crate::store::SessionReadScope::FullGraph,
    )
    .await
    .expect("load session")
    .expect("session read");
    let graph = read.graph;
    let read_model = graph.read_model();
    let messages = read_model.messages.as_slice();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].parts[0].content, "where did this go?");
    assert_eq!(messages[1].parts[0].content, "Stored answer");
    let _checkpoint = read.checkpoint.expect("checkpoint");
    let ledger = read.token_ledger;
    assert_eq!(ledger.len(), 1);
    assert_eq!(ledger[0].source, "turn");
    assert_eq!(ledger[0].model, standard_test_policy().model);
    assert_eq!(ledger[0].usage.input_tokens, 12);
    assert_eq!(ledger[0].usage.output_tokens, 4);
    assert_eq!(ledger[0].usage.cached_input_tokens, 1);
    assert_eq!(ledger[0].usage.reasoning_tokens, 2);
}

#[cfg(feature = "tool-impls")]
#[tokio::test]
async fn resumed_rlm_turns_refresh_turn_state_and_token_ledger() {
    let first_usage = LlmUsage {
        input_tokens: 12,
        output_tokens: 4,
        cached_input_tokens: 1,
        reasoning_tokens: 2,
    };
    let second_usage = LlmUsage {
        input_tokens: 30,
        output_tokens: 7,
        cached_input_tokens: 5,
        reasoning_tokens: 6,
    };
    let transport = mock_provider(vec![
        MockCall {
            stream_events: vec![
                LlmStreamEvent::Delta("stored".to_string()),
                LlmStreamEvent::Usage(first_usage.clone()),
            ],
            response: Ok(LlmResponse {
                full_text: "stored".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "stored".to_string(),
                    response_meta: None,
                }],
                usage: first_usage.clone(),
                ..LlmResponse::default()
            }),
        },
        MockCall {
            stream_events: vec![
                LlmStreamEvent::Delta("stored again".to_string()),
                LlmStreamEvent::Usage(second_usage.clone()),
            ],
            response: Ok(LlmResponse {
                full_text: "stored again".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "stored again".to_string(),
                    response_meta: None,
                }],
                usage: second_usage.clone(),
                ..LlmResponse::default()
            }),
        },
    ]);

    let store = Arc::new(RecordingStore::default());
    let store_trait = store.clone() as Arc<dyn crate::store::RuntimePersistence>;

    let mut runtime = rlm_mode_with_transport_and_store(transport.clone(), store_trait).await;
    let first_turn = runtime
        .run_turn_assembled(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "remember this".to_string(),
                }],
                image_blobs: HashMap::new(),
                user_input: None,
                mode: None,
                mode_turn_options: None,
            },
            CancellationToken::new(),
        )
        .await
        .expect("first turn");
    assert_eq!(first_turn.token_usage.input_tokens, 12);
    assert_eq!(first_turn.token_usage.output_tokens, 4);
    assert_eq!(first_turn.token_usage.cached_input_tokens, 1);
    assert_eq!(first_turn.token_usage.reasoning_tokens, 2);

    let resumed_read = crate::store::RuntimePersistence::load_session(
        store.as_ref(),
        crate::store::SessionReadScope::FullGraph,
    )
    .await
    .expect("load resumed session")
    .expect("resumed head");
    let resumed_checkpoint = resumed_read.checkpoint.clone().expect("resumed checkpoint");
    let mut resumed = LashRuntime::from_persistent_embedded_state(
        rlm_test_policy(),
        test_host_config(),
        crate::PersistentRuntimeServices::new_with_bridges(
            default_tool_session("root", ExecutionMode::new("rlm"), true),
            crate::TurnInjectionBridge::new(),
            crate::TurnInputInjectionBridge::new(),
            store.clone() as Arc<dyn crate::store::RuntimePersistence>,
        ),
        PersistedSessionState {
            policy: SessionPolicy {
                execution_mode: ExecutionMode::new("rlm"),
                standard_context_approach: None,
                ..Default::default()
            },
            session_graph: resumed_read.graph,
            iteration: resumed_checkpoint.turn_state.iteration,
            token_usage: resumed_checkpoint.turn_state.token_usage.clone(),
            last_prompt_usage: resumed_checkpoint.turn_state.last_prompt_usage.clone(),
            dynamic_state_ref: resumed_checkpoint.dynamic_state_ref.clone(),
            dynamic_state_generation: resumed_checkpoint
                .dynamic_state
                .as_ref()
                .map(|snapshot| snapshot.base_generation),
            dynamic_state_snapshot: resumed_checkpoint.dynamic_state.clone(),
            plugin_snapshot_ref: resumed_checkpoint.plugin_snapshot_ref.clone(),
            plugin_snapshot: resumed_checkpoint.plugin_snapshot.clone(),
            execution_state_snapshot: None,
            token_ledger: resumed_read.token_ledger.clone(),
            checkpoint_ref: resumed_read.checkpoint_ref.clone(),
            ..PersistedSessionState::default()
        },
    )
    .await
    .expect("resumed runtime");
    resumed.policy.provider = transport.clone().into_handle();

    let second_turn = resumed
        .run_turn_assembled(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "what did you store?".to_string(),
                }],
                image_blobs: HashMap::new(),
                user_input: None,
                mode: None,
                mode_turn_options: None,
            },
            CancellationToken::new(),
        )
        .await
        .expect("second turn");
    assert_eq!(second_turn.token_usage.input_tokens, 30);
    assert_eq!(second_turn.token_usage.output_tokens, 7);
    assert_eq!(second_turn.token_usage.cached_input_tokens, 5);
    assert_eq!(second_turn.token_usage.reasoning_tokens, 6);

    let read = crate::store::RuntimePersistence::load_session(
        store.as_ref(),
        crate::store::SessionReadScope::FullGraph,
    )
    .await
    .expect("load session")
    .expect("session head");
    let checkpoint = read.checkpoint.expect("checkpoint");
    let turn_state = checkpoint.turn_state;
    assert_eq!(turn_state.token_usage.input_tokens, 30);
    assert_eq!(turn_state.token_usage.output_tokens, 7);
    assert_eq!(turn_state.token_usage.cached_input_tokens, 5);
    assert_eq!(turn_state.token_usage.reasoning_tokens, 6);

    let ledger = read.token_ledger;
    assert_eq!(ledger.len(), 1);
    assert_eq!(ledger[0].source, "turn");
    assert_eq!(ledger[0].usage.input_tokens, 42);
    assert_eq!(ledger[0].usage.output_tokens, 11);
    assert_eq!(ledger[0].usage.cached_input_tokens, 6);
    assert_eq!(ledger[0].usage.reasoning_tokens, 8);
}

#[test]
fn session_usage_report_aggregates_sources_and_models() {
    let entries = vec![
        TokenLedgerEntry {
            source: "turn".to_string(),
            model: "gpt-5.4-mini".to_string(),
            usage: TokenUsage {
                input_tokens: 10,
                output_tokens: 2,
                cached_input_tokens: 3,
                reasoning_tokens: 1,
            },
        },
        TokenLedgerEntry {
            source: "observer".to_string(),
            model: "gpt-5.4-mini".to_string(),
            usage: TokenUsage {
                input_tokens: 7,
                output_tokens: 1,
                cached_input_tokens: 0,
                reasoning_tokens: 0,
            },
        },
        TokenLedgerEntry {
            source: "turn".to_string(),
            model: "gpt-5.4".to_string(),
            usage: TokenUsage {
                input_tokens: 20,
                output_tokens: 4,
                cached_input_tokens: 5,
                reasoning_tokens: 2,
            },
        },
    ];

    let report = SessionUsageReport::from_entries(&entries);

    assert_eq!(report.entry_count, 3);
    assert_eq!(report.usage.input_tokens, 37);
    assert_eq!(report.usage.output_tokens, 7);
    assert_eq!(report.usage.cached_input_tokens, 8);
    assert_eq!(report.usage.reasoning_tokens, 3);
    assert_eq!(report.usage.total_tokens, 47);
    assert_eq!(report.usage.context_total_tokens, 55);
    assert_eq!(report.by_source["turn"].input_tokens, 30);
    assert_eq!(report.by_source["observer"].output_tokens, 1);
    assert_eq!(report.by_model["gpt-5.4-mini"].input_tokens, 17);
    assert_eq!(report.by_model["gpt-5.4"].reasoning_tokens, 2);

    let delta = diff_token_ledger(
        &[TokenLedgerEntry {
            source: "turn".to_string(),
            model: "gpt-5.4-mini".to_string(),
            usage: TokenUsage {
                input_tokens: 10,
                output_tokens: 2,
                cached_input_tokens: 3,
                reasoning_tokens: 1,
            },
        }],
        &entries,
    )
    .expect("delta");
    assert_eq!(delta.len(), 2);
    assert_eq!(delta[0].source, "observer");
    assert_eq!(delta[1].model, "gpt-5.4");
}

#[tokio::test]
async fn await_background_work_waits_for_registered_jobs() {
    let runtime = standard_runtime_with_transport_and_background(mock_provider(Vec::new())).await;
    let manager = runtime.session_manager().expect("session manager");
    let observed = Arc::new(AtomicBool::new(false));
    let observed_task = Arc::clone(&observed);

    manager
        .spawn_hidden_task(
            "root",
            "test",
            Box::pin(async move {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                observed_task.store(true, Ordering::SeqCst);
                Ok(())
            }),
        )
        .await
        .expect("spawn background job");

    let mut runtime = runtime;
    runtime
        .await_background_work()
        .await
        .expect("await background work");
    assert!(observed.load(Ordering::SeqCst));
}

#[tokio::test]
async fn await_background_work_does_not_cross_runtime_sessions_with_same_logical_id() {
    let executor: Arc<dyn SessionTaskExecutor> = Arc::new(TokioSessionTaskExecutor::default());
    let runtime_one = standard_runtime_with_shared_background_executor(
        mock_provider(Vec::new()),
        Arc::clone(&executor),
    )
    .await;
    let runtime_two = standard_runtime_with_shared_background_executor(
        mock_provider(Vec::new()),
        Arc::clone(&executor),
    )
    .await;
    let manager_one = runtime_one.session_manager().expect("session manager");
    let observed = Arc::new(AtomicBool::new(false));
    let observed_task = Arc::clone(&observed);
    manager_one
        .spawn_hidden_task(
            "root",
            "test",
            Box::pin(async move {
                tokio::time::sleep(std::time::Duration::from_millis(40)).await;
                observed_task.store(true, Ordering::SeqCst);
                Ok(())
            }),
        )
        .await
        .expect("spawn background job");

    let mut runtime_two = runtime_two;
    tokio::time::timeout(
        std::time::Duration::from_millis(10),
        runtime_two.await_background_work(),
    )
    .await
    .expect("second runtime should not block on first runtime jobs")
    .expect("await background work");
    assert!(!observed.load(Ordering::SeqCst));

    let mut runtime_one = runtime_one;
    runtime_one
        .await_background_work()
        .await
        .expect("first runtime await background work");
    assert!(observed.load(Ordering::SeqCst));
}

#[cfg(feature = "tool-impls")]
#[tokio::test]
async fn environment_park_resume_preserves_active_path() {
    // Build a RuntimeEnvironment with the usual test tool factories +
    // Residency::ActivePathOnly (webserver pattern; host owns disk).
    let factories: Vec<Arc<dyn crate::PluginFactory>> = vec![
        Arc::new(crate::BuiltinToolResultProjectionPluginFactory::default()),
        Arc::new(crate::testing::FakeStandardContextApproachPluginFactory::rolling_history()),
        Arc::new(StaticPluginFactory::new(
            "shell",
            crate::PluginSpec::new()
                .with_tool_provider(Arc::new(crate::tools::StandardShell::new())),
        )),
    ];
    let plugin_host = Arc::new(crate::PluginHost::new(factories));

    let env = crate::RuntimeEnvironment::builder()
        .with_plugin_host(plugin_host)
        .with_residency(crate::Residency::ActivePathOnly)
        .build();

    // In-memory SQLite store — shared across park/resume.
    let store: Arc<dyn crate::store::RuntimePersistence> = Arc::new(RecordingStore::default());

    // Initial state: one user message. Force the active leaf to this
    // node so later reads find it.
    let mut initial_state = PersistedSessionState {
        session_id: "integration-park-resume".to_string(),
        policy: rlm_test_policy(),
        ..PersistedSessionState::default()
    };
    // Borrow the existing plugin-message helper to construct a
    // minimal Message without touching sansio's Message struct
    // directly (its shape may change).
    let first_msg = crate::session_model::plugin_message_to_message(
        &crate::PluginMessage::text(crate::session_model::MessageRole::User, "hello"),
        None,
    );
    let first_id = initial_state.session_graph.append_message(first_msg);

    // Build runtime via from_environment.
    let runtime = LashRuntime::from_environment(
        &env,
        rlm_test_policy(),
        initial_state,
        Some(Arc::clone(&store)),
    )
    .await
    .expect("runtime from environment");

    // Under ActivePathOnly, the resident graph is already trimmed to
    // the active path. The leaf node must still be there.
    assert!(runtime.state.session_graph.find_node(&first_id).is_some());

    // Park — flushes to store, returns a lightweight handle.
    let parked = runtime.park().await.expect("park");
    assert_eq!(parked.session_id(), "integration-park-resume");

    // Resume — loads from the same store.
    let resumed = LashRuntime::resume(parked, &env).await.expect("resume");
    assert_eq!(resumed.state.session_id, "integration-park-resume");
    // Active path preserved through park/resume.
    assert!(resumed.state.session_graph.find_node(&first_id).is_some());

    // Host-driven cleanup primitives: orphaned_node_ids() + vacuum().
    // No orphans in this test (only one node on active path), so both
    // paths return empty / zero.
    let orphans = resumed.orphaned_node_ids().await.expect("orphans");
    assert!(orphans.is_empty());
    let report = store.vacuum().await.expect("vacuum");
    assert_eq!(report.removed_node_count, 0);
}
