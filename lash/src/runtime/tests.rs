async fn drain_standard_stream_queue(
    event_tx: &mpsc::Sender<SessionEvent>,
    llm_stream_rx: &mut tokio::sync::mpsc::UnboundedReceiver<LlmStreamEvent>,
    text_streamed: &mut bool,
    streamed_usage: &mut LlmUsage,
) {
    while let Ok(stream_event) = llm_stream_rx.try_recv() {
        match stream_event {
            LlmStreamEvent::Delta(delta) => {
                if !delta.is_empty() {
                    *text_streamed = true;
                    crate::session_model::send_event(
                        event_tx,
                        SessionEvent::TextDelta { content: delta },
                    )
                    .await;
                }
            }
            LlmStreamEvent::Part(LlmOutputPart::Text { text }) => {
                if !text.is_empty() {
                    *text_streamed = true;
                    crate::session_model::send_event(
                        event_tx,
                        SessionEvent::TextDelta { content: text },
                    )
                    .await;
                }
            }
            LlmStreamEvent::Part(LlmOutputPart::ToolCall { .. }) => {}
            LlmStreamEvent::Usage(usage) => *streamed_usage = usage,
        }
    }
}

use super::*;
use serde_json::json;
use sha2::Digest;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::llm::transport::LlmTransport;
use crate::llm::transport::LlmTransportError;
use crate::llm::types::{LlmRequest, LlmUsage};
use crate::plugin::StaticPluginFactory;
use crate::provider::Provider;
use crate::store::RuntimeStore;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

fn default_state() -> PersistedSessionState {
    PersistedSessionState::default()
}

trait ProjectionState {
    fn projected_messages(&self) -> &[Message];
    fn projected_tool_calls(&self) -> &[ToolCallRecord];
}

impl ProjectionState for SessionStateEnvelope {
    fn projected_messages(&self) -> &[Message] {
        self.projected_messages()
    }

    fn projected_tool_calls(&self) -> &[ToolCallRecord] {
        self.projected_tool_calls()
    }
}

impl ProjectionState for PersistedSessionState {
    fn projected_messages(&self) -> &[Message] {
        self.projected_messages()
    }

    fn projected_tool_calls(&self) -> &[ToolCallRecord] {
        self.projected_tool_calls()
    }
}

trait ProjectionStateMut: ProjectionState {
    #[cfg(feature = "sqlite-store")]
    fn replace_projection(&mut self, messages: &[Message], tool_calls: &[ToolCallRecord]);
    fn append_message(&mut self, message: Message);
}

impl ProjectionStateMut for SessionStateEnvelope {
    #[cfg(feature = "sqlite-store")]
    fn replace_projection(&mut self, messages: &[Message], tool_calls: &[ToolCallRecord]) {
        self.replace_projection(messages, tool_calls);
    }

    fn append_message(&mut self, message: Message) {
        self.session_graph.append_message(message);
    }
}

impl ProjectionStateMut for PersistedSessionState {
    #[cfg(feature = "sqlite-store")]
    fn replace_projection(&mut self, messages: &[Message], tool_calls: &[ToolCallRecord]) {
        self.replace_projection(messages, tool_calls);
    }

    fn append_message(&mut self, message: Message) {
        self.session_graph.append_message(message);
    }
}

fn projected_messages(state: &impl ProjectionState) -> &[Message] {
    state.projected_messages()
}

fn projected_tool_calls(state: &impl ProjectionState) -> &[ToolCallRecord] {
    state.projected_tool_calls()
}

#[cfg(feature = "sqlite-store")]
fn set_projection(
    state: &mut impl ProjectionStateMut,
    messages: &[Message],
    tool_calls: &[ToolCallRecord],
) {
    state.replace_projection(messages, tool_calls);
}

fn append_message(state: &mut impl ProjectionStateMut, message: Message) {
    state.append_message(message);
}

#[cfg(feature = "sqlite-store")]
fn text_message(id: &str, role: MessageRole, content: &str) -> Message {
    Message {
        id: id.to_string(),
        role,
        parts: vec![Part {
            id: format!("{id}.p0"),
            kind: PartKind::Text,
            content: content.to_string(),
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            prune_state: PruneState::Intact,
        }],
        user_input: None,
        origin: None,
    }
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
    blobs: Mutex<HashMap<String, Vec<u8>>>,
    session_head_meta: Mutex<Option<crate::SessionHeadMeta>>,
    session_graph: Mutex<crate::SessionGraph>,
    live_resume: Mutex<Option<crate::LiveResumeSnapshot>>,
    usage_deltas: Mutex<Vec<crate::TokenLedgerEntry>>,
}

#[async_trait::async_trait]
impl crate::store::RuntimeStore for RecordingStore {
    async fn put_blob(&self, content: &[u8]) -> crate::BlobRef {
        let hash = format!("{:x}", sha2::Sha256::digest(content));
        self.blobs
            .lock()
            .expect("lock blobs")
            .insert(hash.clone(), content.to_vec());
        crate::BlobRef(hash)
    }

    async fn get_blob(&self, blob_ref: &crate::BlobRef) -> Option<Vec<u8>> {
        self.blobs
            .lock()
            .expect("lock blobs")
            .get(blob_ref.as_str())
            .cloned()
    }

    async fn append_usage_deltas(&self, entries: &[crate::TokenLedgerEntry]) {
        self.usage_deltas
            .lock()
            .expect("lock usage deltas")
            .extend(entries.iter().cloned());
    }

    async fn load_usage_deltas(&self) -> Vec<crate::TokenLedgerEntry> {
        self.usage_deltas.lock().expect("lock usage deltas").clone()
    }

    async fn save_session_head_meta(&self, meta: crate::SessionHeadMeta) {
        *self.session_head_meta.lock().expect("lock store") = Some(meta);
    }

    async fn load_session_head_meta(&self) -> Option<crate::SessionHeadMeta> {
        self.session_head_meta.lock().expect("lock store").clone()
    }

    async fn replace_session_graph(&self, graph: &crate::SessionGraph) {
        *self.session_graph.lock().expect("lock graph") = graph.clone();
    }

    async fn append_session_graph_nodes(&self, nodes: &[crate::SessionNodeRecord]) {
        self.session_graph
            .lock()
            .expect("lock graph")
            .extend_node_records(nodes.iter().cloned());
    }

    async fn load_session_graph(&self) -> crate::SessionGraph {
        self.session_graph.lock().expect("lock graph").clone()
    }

    async fn save_live_resume(&self, snapshot: crate::LiveResumeSnapshot) {
        *self.live_resume.lock().expect("lock live resume") = Some(snapshot);
    }

    async fn load_live_resume(&self) -> Option<crate::LiveResumeSnapshot> {
        self.live_resume.lock().expect("lock live resume").clone()
    }

    async fn clear_live_resume(&self) {
        self.live_resume.lock().expect("lock live resume").take();
    }

    async fn save_session_meta(&self, _meta: crate::store::SessionMeta) {}

    async fn load_session_meta(&self) -> Option<crate::store::SessionMeta> {
        None
    }
}

struct MockCall {
    stream_events: Vec<LlmStreamEvent>,
    response: Result<LlmResponse, LlmTransportError>,
}

#[derive(Clone)]
struct MockTransport {
    calls: Arc<Mutex<Vec<MockCall>>>,
}

impl MockTransport {
    fn new(calls: Vec<MockCall>) -> Self {
        Self {
            calls: Arc::new(Mutex::new(calls)),
        }
    }
}

#[async_trait::async_trait]
impl LlmTransport for MockTransport {
    fn default_root_model(&self) -> &'static str {
        "mock-model"
    }

    fn default_agent_model(&self, _tier: &str) -> Option<crate::llm::types::ModelSelection> {
        None
    }

    fn requires_streaming(&self) -> bool {
        true
    }

    fn normalize_model(&self, model: &str) -> String {
        model.to_string()
    }

    fn context_lookup_model(&self, model: &str) -> String {
        model.to_string()
    }

    async fn ensure_ready(&self, _provider: &mut Provider) -> Result<bool, LlmTransportError> {
        Ok(false)
    }

    async fn complete(
        &self,
        _provider: &mut Provider,
        req: LlmRequest,
    ) -> Result<LlmResponse, LlmTransportError> {
        let call = self.calls.lock().expect("lock calls").remove(0);
        if let Some(tx) = req.stream_events.as_ref() {
            for event in &call.stream_events {
                tx.send(event.clone());
            }
        }
        call.response
    }
}

fn standard_test_policy() -> SessionPolicy {
    SessionPolicy {
        execution_mode: ExecutionMode::Standard,
        provider: Provider::OpenAiGeneric {
            api_key: "test-key".to_string(),
            base_url: "https://example.invalid/v1".to_string(),
            options: crate::provider::ProviderOptions::default(),
        },
        model: "mock-model".to_string(),
        max_context_tokens: Some(200_000),
        ..SessionPolicy::default()
    }
}

fn test_host_config() -> EmbeddedRuntimeHost {
    EmbeddedRuntimeHost::new(RuntimeCoreConfig::default())
}

fn test_host_config_with_llm_log_path(path: PathBuf) -> EmbeddedRuntimeHost {
    EmbeddedRuntimeHost::new(RuntimeCoreConfig::default().with_llm_log_path(Some(path)))
}

#[cfg(feature = "sqlite-store")]
#[derive(Clone, Default)]
struct RecordingSessionStoreFactory {
    stores: Arc<StdMutex<Vec<Arc<crate::store::Store>>>>,
}

#[cfg(feature = "sqlite-store")]
impl RecordingSessionStoreFactory {
    fn stores(&self) -> Vec<Arc<crate::store::Store>> {
        self.stores.lock().expect("store factory").clone()
    }
}

#[cfg(feature = "sqlite-store")]
impl SessionStoreFactory for RecordingSessionStoreFactory {
    fn create_store(
        &self,
        request: &SessionStoreCreateRequest,
    ) -> Result<Arc<dyn crate::store::RuntimeStore>, String> {
        let store = Arc::new(crate::store::Store::memory().map_err(|err| err.to_string())?);
        store.save_session_meta(crate::SessionMeta {
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
        Ok(store as Arc<dyn crate::store::RuntimeStore>)
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
        .build_session(session_id, mode, crate::ContextApproach::default(), None)
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
        Arc::new(crate::BuiltinRollingHistoryPluginFactory::default()),
        Arc::new(crate::BuiltinObservationalMemoryPluginFactory),
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
        factories.push(Arc::new(StaticPluginFactory::new(
            "wait",
            crate::PluginSpec::new().with_tool_provider(Arc::new(crate::tools::WaitTool::new())),
        )));
        factories.push(Arc::new(StaticPluginFactory::new(
            "show_snippet_to_user",
            crate::PluginSpec::new()
                .with_tool_provider(Arc::new(crate::tools::ShowSnippetToUser::new())),
        )));
    }
    crate::PluginHost::new(factories)
        .build_session(session_id, mode, crate::ContextApproach::default(), None)
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

async fn standard_runtime_with_transport(transport: MockTransport) -> LashRuntime {
    let tools: Arc<dyn crate::ToolProvider> = Arc::new(EmptyTools);
    let mut runtime = LashRuntime::from_embedded_state(
        standard_test_policy(),
        test_host_config(),
        crate::RuntimeServices::new(plugin_session_with_tools(
            "root",
            ExecutionMode::Standard,
            tools,
        )),
        PersistedSessionState::default(),
    )
    .await
    .expect("runtime");
    runtime.llm_factory = Arc::new(move |_| Box::new(transport.clone()));
    runtime
}

#[test]
fn plugin_host_rejects_observational_memory_without_supporting_plugin() {
    let host = crate::PluginHost::new(vec![Arc::new(
        crate::BuiltinRollingHistoryPluginFactory::default(),
    )]);
    let result = host.build_session(
        "root",
        ExecutionMode::Standard,
        crate::ContextApproach::ObservationalMemory(crate::ObservationalMemoryConfig::default()),
        None,
    );
    let err = match result {
        Ok(_) => panic!("OM should require supporting plugin"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains(
            "context approach `observational_memory` requires a supporting plugin factory"
        ),
        "unexpected error: {err}"
    );
}

async fn standard_runtime_with_transport_and_background(transport: MockTransport) -> LashRuntime {
    let tools: Arc<dyn crate::ToolProvider> = Arc::new(EmptyTools);
    let host = BackgroundRuntimeHost::new(
        test_host_config(),
        Arc::new(TokioBackgroundExecutor::default()),
    );
    let mut runtime = LashRuntime::from_background_state(
        standard_test_policy(),
        host,
        crate::RuntimeServices::new(plugin_session_with_tools(
            "root",
            ExecutionMode::Standard,
            tools,
        )),
        PersistedSessionState::default(),
    )
    .await
    .expect("runtime");
    runtime.llm_factory = Arc::new(move |_| Box::new(transport.clone()));
    runtime
}

async fn standard_runtime_with_shared_background_executor(
    transport: MockTransport,
    executor: Arc<dyn BackgroundExecutor>,
) -> LashRuntime {
    let tools: Arc<dyn crate::ToolProvider> = Arc::new(EmptyTools);
    let host = BackgroundRuntimeHost::new(test_host_config(), executor);
    let mut runtime = LashRuntime::from_background_state(
        standard_test_policy(),
        host,
        crate::RuntimeServices::new(plugin_session_with_tools(
            "root",
            ExecutionMode::Standard,
            tools,
        )),
        PersistedSessionState::default(),
    )
    .await
    .expect("runtime");
    runtime.llm_factory = Arc::new(move |_| Box::new(transport.clone()));
    runtime
}

#[cfg(feature = "sqlite-store")]
async fn om_runtime_with_transport_and_background(
    transport: MockTransport,
    config: crate::ObservationalMemoryConfig,
) -> (LashRuntime, Arc<crate::store::Store>) {
    let tools: Arc<dyn crate::ToolProvider> = Arc::new(EmptyTools);
    let host = BackgroundRuntimeHost::new(
        test_host_config(),
        Arc::new(TokioBackgroundExecutor::default()),
    );
    let plugin_host = crate::PluginHost::new(vec![
        Arc::new(crate::BuiltinObservationalMemoryPluginFactory),
        Arc::new(StaticPluginFactory::new(
            "test_tools",
            crate::PluginSpec::new().with_tool_provider(Arc::clone(&tools)),
        )),
    ]);
    let plugins = plugin_host
        .build_session(
            "root",
            ExecutionMode::Standard,
            crate::ContextApproach::ObservationalMemory(config.clone()),
            None,
        )
        .expect("plugins");
    let store = Arc::new(crate::store::Store::memory().expect("store"));
    let mut runtime = LashRuntime::from_persistent_background_state(
        SessionPolicy {
            execution_mode: ExecutionMode::Standard,
            provider: Provider::OpenAiGeneric {
                api_key: "test-key".to_string(),
                base_url: "https://example.invalid/v1".to_string(),
                options: crate::provider::ProviderOptions::default(),
            },
            model: "mock-model".to_string(),
            max_context_tokens: Some(200_000),
            context_approach: crate::ContextApproach::ObservationalMemory(config),
            ..SessionPolicy::default()
        },
        host,
        crate::PersistentRuntimeServices::new(
            plugins,
            store.clone() as Arc<dyn crate::store::RuntimeStore>,
        ),
        PersistedSessionState::default(),
    )
    .await
    .expect("runtime");
    runtime.llm_factory = Arc::new(move |_| Box::new(transport.clone()));
    (runtime, store)
}

async fn standard_runtime_with_transport_and_host(
    transport: MockTransport,
    host: EmbeddedRuntimeHost,
) -> LashRuntime {
    let tools: Arc<dyn crate::ToolProvider> = Arc::new(EmptyTools);
    let mut runtime = LashRuntime::from_embedded_state(
        standard_test_policy(),
        host,
        crate::RuntimeServices::new(plugin_session_with_tools(
            "root",
            ExecutionMode::Standard,
            tools,
        )),
        PersistedSessionState::default(),
    )
    .await
    .expect("runtime");
    runtime.llm_factory = Arc::new(move |_| Box::new(transport.clone()));
    runtime
}

#[tokio::test]
async fn runtime_requires_explicit_max_context_tokens() {
    let tools: Arc<dyn crate::ToolProvider> = Arc::new(EmptyTools);
    let result = LashRuntime::from_embedded_state(
        SessionPolicy {
            execution_mode: ExecutionMode::Standard,
            provider: Provider::OpenAiGeneric {
                api_key: "test-key".to_string(),
                base_url: "https://example.invalid/v1".to_string(),
                options: crate::provider::ProviderOptions::default(),
            },
            model: "mock-model".to_string(),
            max_context_tokens: None,
            ..SessionPolicy::default()
        },
        test_host_config(),
        crate::RuntimeServices::new(plugin_session_with_tools(
            "root",
            ExecutionMode::Standard,
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
        execution_mode: ExecutionMode::Rlm,
        provider: Provider::OpenAiGeneric {
            api_key: "test-key".to_string(),
            base_url: "https://example.invalid/v1".to_string(),
            options: crate::provider::ProviderOptions::default(),
        },
        model: "mock-model".to_string(),
        max_context_tokens: Some(200_000),
        ..SessionPolicy::default()
    }
}

#[cfg(feature = "tool-impls")]
async fn rlm_runtime_with_transport(transport: MockTransport) -> LashRuntime {
    let plugins = default_tool_session("root", ExecutionMode::Rlm, true);
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
                execution_mode: ExecutionMode::Rlm,
                ..Default::default()
            },
            ..SessionStateEnvelope::default()
        }),
    )
    .await
    .expect("runtime");
    runtime.llm_factory = Arc::new(move |_| Box::new(transport.clone()));
    runtime
}

#[cfg(feature = "tool-impls")]
async fn rlm_runtime_with_transport_and_store(
    transport: MockTransport,
    store: Arc<dyn crate::store::RuntimeStore>,
) -> LashRuntime {
    let plugins = default_tool_session("root", ExecutionMode::Rlm, true);
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
                execution_mode: ExecutionMode::Rlm,
                ..Default::default()
            },
            ..SessionStateEnvelope::default()
        }),
    )
    .await
    .expect("runtime");
    runtime.llm_factory = Arc::new(move |_| Box::new(transport.clone()));
    runtime
}

#[cfg(feature = "tool-impls")]
#[tokio::test]
async fn active_tool_catalog_uses_runtime_execution_mode() {
    let runtime = LashRuntime::from_embedded_state(
        rlm_test_policy(),
        test_host_config(),
        crate::RuntimeServices::new(default_tool_session("root", ExecutionMode::Rlm, false)),
        PersistedSessionState::from_state(SessionStateEnvelope {
            policy: SessionPolicy {
                execution_mode: ExecutionMode::Rlm,
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
    assert!(names.contains(&"write_stdin"));
    assert!(!names.contains(&"shell_wait"));
    assert!(!names.contains(&"shell_read"));
}

async fn standard_runtime_with_bridge(
    transport: MockTransport,
    turn_injection_bridge: crate::TurnInjectionBridge,
) -> LashRuntime {
    let tools: Arc<dyn crate::ToolProvider> = Arc::new(EmptyTools);
    let mut runtime = LashRuntime::from_embedded_state(
        standard_test_policy(),
        test_host_config(),
        crate::RuntimeServices::new_with_bridges(
            plugin_session_with_tools("root", ExecutionMode::Standard, tools),
            turn_injection_bridge,
            crate::TurnInputInjectionBridge::new(),
        ),
        PersistedSessionState::default(),
    )
    .await
    .expect("runtime");
    runtime.llm_factory = Arc::new(move |_| Box::new(transport.clone()));
    runtime
}

async fn standard_runtime_with_input_bridge(
    transport: MockTransport,
    turn_input_injection_bridge: crate::TurnInputInjectionBridge,
) -> LashRuntime {
    let tools: Arc<dyn crate::ToolProvider> = Arc::new(EmptyTools);
    let mut runtime = LashRuntime::from_embedded_state(
        standard_test_policy(),
        test_host_config(),
        crate::RuntimeServices::new_with_bridges(
            plugin_session_with_tools("root", ExecutionMode::Standard, tools),
            crate::TurnInjectionBridge::new(),
            turn_input_injection_bridge,
        ),
        PersistedSessionState::default(),
    )
    .await
    .expect("runtime");
    runtime.llm_factory = Arc::new(move |_| Box::new(transport.clone()));
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
    transport: MockTransport,
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
    transport: MockTransport,
) -> LashRuntime {
    runtime_with_plugins_and_tools_and_host(plugins, tools, transport, test_host_config()).await
}

async fn runtime_with_plugins_and_tools_and_host(
    plugins: Vec<Arc<dyn crate::PluginFactory>>,
    tools: Arc<dyn crate::ToolProvider>,
    transport: MockTransport,
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
    runtime.llm_factory = Arc::new(move |_| Box::new(transport.clone()));
    runtime
}

struct EchoTool;

#[async_trait::async_trait]
impl crate::ToolProvider for EchoTool {
    fn definitions(&self) -> Vec<crate::ToolDefinition> {
        vec![crate::ToolDefinition {
            name: "echo_tool".to_string(),
            description: "Return a tool payload".to_string(),
            params: vec![crate::ToolParam::typed("value", "str")],
            returns: "json".to_string(),
            examples: vec![],
            enabled: true,
            injected: true,
            input_schema_override: None,
            output_schema_override: None,
        }]
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
    let transport = MockTransport::new(Vec::new());
    let mut runtime = runtime_with_plugins(vec![plugin], transport).await;

    runtime
        .update_session_config(
            Some(crate::Provider::Codex {
                access_token: "tok".into(),
                refresh_token: "ref".into(),
                expires_at: u64::MAX,
                account_id: None,
                options: crate::provider::ProviderOptions::default(),
            }),
            Some("gpt-5.4".to_string()),
            Some(None),
            Some(123_456),
        )
        .await;

    let changes = observed.lock().await;
    assert_eq!(changes.len(), 1);
    let (previous, current) = &changes[0];
    assert_eq!(
        previous.provider.kind(),
        crate::provider::ProviderKind::OpenAiGeneric
    );
    assert_eq!(
        current.provider.kind(),
        crate::provider::ProviderKind::Codex
    );
    assert_eq!(current.model, "gpt-5.4");
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
    let transport = MockTransport::new(Vec::new());
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
            },
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    assert_eq!(turn.status, TurnStatus::Failed);
    assert_eq!(turn.done_reason, DoneReason::RuntimeError);
    assert!(turn.errors.iter().any(|issue| issue.kind == "plugin"));
    assert!(projected_messages(&turn.state).iter().any(|message| {
        message
            .parts
            .iter()
            .any(|part| part.content.contains("plugin preface"))
    }));
}

#[tokio::test]
async fn normal_turn_preserves_user_input_provenance_in_state() {
    let transport = MockTransport::new(vec![MockCall {
        stream_events: Vec::new(),
        response: Ok(LlmResponse {
            full_text: "Done".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "Done".to_string(),
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
            },
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    let user_message = turn
        .state
        .projected_messages()
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
    let transport = MockTransport::new(vec![
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
            },
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    assert_eq!(turn.status, TurnStatus::Failed);
    assert_eq!(turn.done_reason, DoneReason::RuntimeError);
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
    let transport = MockTransport::new(vec![
        MockCall {
            stream_events: Vec::new(),
            response: Ok(LlmResponse {
                full_text: "First answer.".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "First answer.".to_string(),
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
            },
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    assert!(projected_messages(&turn.state).iter().any(|message| {
        message.role == MessageRole::Assistant
            && message
                .parts
                .iter()
                .any(|part| part.content.contains("Second answer."))
    }));
    assert!(projected_messages(&turn.state).iter().all(|message| {
        !(message.role == MessageRole::User
            && message
                .parts
                .iter()
                .any(|part| part.content == "one more thing"))
    }));
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
                        mime: "image/png".to_string(),
                        url: crate::session_model::message::data_url_for_bytes(
                            "image/png",
                            &[9, 8, 7],
                        ),
                        filename: None,
                    }),
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: crate::PruneState::Intact,
                },
                crate::Part {
                    id: String::new(),
                    kind: crate::PartKind::Text,
                    content: "see image".to_string(),
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: crate::PruneState::Intact,
                },
            ],
            images: Vec::new(),
            user_input: None,
        }])
        .expect("enqueue");
    let transport = MockTransport::new(vec![
        MockCall {
            stream_events: Vec::new(),
            response: Ok(LlmResponse {
                full_text: "First answer.".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "First answer.".to_string(),
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
            },
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    assert!(projected_messages(&turn.state).iter().any(|message| {
        message.role == MessageRole::User
            && message
                .parts
                .iter()
                .any(|part| matches!(part.kind, PartKind::Image) && part.attachment.is_some())
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
                tool_result_projectors: Vec::new(),
                runtime_event: None,
                external_registrar: None,
            }))
        }),
    });
    let transport = MockTransport::new(vec![
        MockCall {
            stream_events: Vec::new(),
            response: Ok(LlmResponse {
                full_text: "First answer.".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "First answer.".to_string(),
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
            },
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    assert!(projected_messages(&turn.state).iter().any(|message| {
        message.role == MessageRole::System
            && message
                .parts
                .iter()
                .any(|part| part.content == "checkpoint injected")
    }));
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

    let transport = MockTransport::new(vec![MockCall {
        stream_events: Vec::new(),
        response: Ok(LlmResponse {
            full_text: "answer".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "answer".to_string(),
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

    let projected = projected_messages(&assembled.state);
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
                                                "message_count": snapshot.projected_messages().len(),
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
    let transport = MockTransport::new(Vec::new());
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
                prune_state: PruneState::Intact,
            }],
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
    let transport = MockTransport::new(vec![MockCall {
        stream_events: vec![
            LlmStreamEvent::Delta("child ".to_string()),
            LlmStreamEvent::Part(LlmOutputPart::Text {
                text: "session".to_string(),
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

#[cfg(feature = "sqlite-store")]
#[tokio::test]
async fn session_manager_persists_child_sessions_in_separate_store() {
    let factory = RecordingSessionStoreFactory::default();
    let host = test_host_config().with_session_store_factory(Arc::new(factory.clone()));
    let mut runtime = runtime_with_plugins_and_tools_and_host(
        Vec::new(),
        Arc::new(EmptyTools),
        MockTransport::new(Vec::new()),
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
                prune_state: PruneState::Intact,
            }],
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
            context_surface: crate::SessionContextSurface::default(),
            mode_extras: crate::ModeExtras::default(),
            usage_source: None,
        })
        .await
        .expect("child session");

    assert_eq!(handle.session_id, "child-store");
    let stores = factory.stores();
    assert_eq!(stores.len(), 1);
    let meta = stores[0].load_session_meta().expect("session meta");
    assert_eq!(meta.session_id, "child-store");
    assert_eq!(meta.parent_session_id.as_deref(), Some("root"));
    let head = stores[0].load_session_head().expect("session head");
    let graph = head.graph;
    let messages = graph.project_messages();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].parts[0].content, "parent hello");
    let checkpoint = head
        .checkpoint_ref
        .as_ref()
        .and_then(|blob_ref| stores[0].get_checkpoint(blob_ref))
        .expect("checkpoint");
    let turn_state = checkpoint.turn_state;
    assert_eq!(turn_state.iteration, 3);
}

#[tokio::test]
async fn session_manager_rejects_duplicate_child_session_ids() {
    let runtime = runtime_with_plugins(Vec::new(), MockTransport::new(Vec::new())).await;
    let manager = runtime.session_manager().expect("session manager");
    manager
        .create_session(crate::SessionCreateRequest {
            session_id: Some("child".to_string()),
            parent_session_id: None,
            start: crate::SessionStartPoint::Empty,
            policy: None,
            plugin_mode: crate::SessionPluginMode::InheritCurrent,
            initial_nodes: Vec::new(),
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
            context_surface: crate::SessionContextSurface::default(),
            mode_extras: crate::ModeExtras::default(),
            usage_source: None,
        })
        .await
        .expect_err("duplicate child session should fail");
    assert!(err.to_string().contains("already exists"));
}

struct MemoryProbeTool;

#[async_trait::async_trait]
impl crate::ToolProvider for MemoryProbeTool {
    fn definitions(&self) -> Vec<crate::ToolDefinition> {
        vec![crate::ToolDefinition {
            name: "memory_probe".to_string(),
            description: "probe".to_string(),
            params: Vec::new(),
            returns: "str".to_string(),
            examples: Vec::new(),
            enabled: true,
            injected: true,
            input_schema_override: None,
            output_schema_override: None,
        }]
    }

    async fn execute(&self, _name: &str, _args: &serde_json::Value) -> crate::ToolResult {
        crate::ToolResult::ok(json!("ok"))
    }
}

struct ChildSessionTool;

#[async_trait::async_trait]
impl crate::ToolProvider for ChildSessionTool {
    fn definitions(&self) -> Vec<crate::ToolDefinition> {
        vec![crate::ToolDefinition {
            name: "spawn_child".to_string(),
            description: "spawn a child session".to_string(),
            params: Vec::new(),
            returns: "record".to_string(),
            examples: Vec::new(),
            enabled: true,
            injected: true,
            input_schema_override: None,
            output_schema_override: None,
        }]
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
                session_id: Some("delegate-child".to_string()),
                parent_session_id: Some(context.session_id.clone()),
                start: crate::SessionStartPoint::Empty,
                policy: None,
                plugin_mode: crate::SessionPluginMode::InheritCurrent,
                initial_nodes: Vec::new(),
                context_surface: crate::SessionContextSurface::default(),
                mode_extras: crate::ModeExtras::default(),
                usage_source: Some("delegate".to_string()),
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
    let runtime = runtime_with_plugins(Vec::new(), MockTransport::new(Vec::new())).await;
    let manager = runtime.session_manager().expect("session manager");
    let handle = manager
        .create_session(crate::SessionCreateRequest {
            session_id: Some("memory-child".to_string()),
            parent_session_id: None,
            start: crate::SessionStartPoint::Empty,
            policy: None,
            plugin_mode: crate::SessionPluginMode::Fresh,
            initial_nodes: Vec::new(),
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
async fn parent_turn_receives_live_child_token_usage_events() {
    let transport = MockTransport::new(vec![
        MockCall {
            stream_events: vec![
                LlmStreamEvent::Part(LlmOutputPart::ToolCall {
                    call_id: "tool-1".to_string(),
                    tool_name: "spawn_child".to_string(),
                    input_json: "{}".to_string(),
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
            },
            &sink,
            CancellationToken::new(),
        )
        .await
        .expect("parent turn");

    assert_eq!(turn.status, TurnStatus::Completed);
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
    assert_eq!(child_usage_event.0, "delegate-child");
    assert_eq!(child_usage_event.1, "delegate");
    assert_eq!(child_usage_event.2, "mock-model");
    assert_eq!(child_usage_event.3.input_tokens, 7);
    assert_eq!(child_usage_event.3.output_tokens, 2);
    assert_eq!(child_usage_event.3.cached_input_tokens, 4);
    assert_eq!(child_usage_event.3.reasoning_tokens, 1);
    assert_eq!(child_usage_event.4.cached_input_tokens, 4);

    let usage = runtime.usage_report();
    assert_eq!(usage.by_source["delegate"].input_tokens, 7);
    assert_eq!(usage.by_source["delegate"].output_tokens, 2);
    assert_eq!(usage.by_source["delegate"].cached_input_tokens, 4);
    assert_eq!(usage.by_source["delegate"].reasoning_tokens, 1);
}

#[tokio::test]
async fn parent_turn_keeps_cached_only_child_usage_live() {
    let transport = MockTransport::new(vec![
        MockCall {
            stream_events: vec![
                LlmStreamEvent::Part(LlmOutputPart::ToolCall {
                    call_id: "tool-1".to_string(),
                    tool_name: "spawn_child".to_string(),
                    input_json: "{}".to_string(),
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
    assert_eq!(usage.by_source["delegate"].input_tokens, 0);
    assert_eq!(usage.by_source["delegate"].output_tokens, 0);
    assert_eq!(usage.by_source["delegate"].cached_input_tokens, 9);
    assert_eq!(usage.by_source["delegate"].reasoning_tokens, 0);
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
    assert_eq!(out.status, TurnStatus::Completed);
    assert_eq!(out.done_reason, DoneReason::ModelStop);
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
                prune_state: PruneState::Intact,
            }],
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
    assert_eq!(out.status, TurnStatus::Completed);
    assert_eq!(out.done_reason, DoneReason::ModelStop);
    assert_eq!(out.assistant_output.safe_text, "stored");
    assert_eq!(out.assistant_output.raw_text, "stored");
    assert_eq!(out.assistant_output.state, OutputState::Usable);
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
                prune_state: PruneState::Intact,
            }],
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
    assert_eq!(out.status, TurnStatus::Failed);
    assert_eq!(out.done_reason, DoneReason::ToolFailure);
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
    assert_eq!(out.status, TurnStatus::Failed);
    assert_eq!(out.done_reason, DoneReason::RuntimeError);
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
                prune_state: PruneState::Intact,
            }],
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
    assert_eq!(out.status, TurnStatus::Completed);
    assert_eq!(out.done_reason, DoneReason::MaxTurns);
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
    assert_eq!(out.status, TurnStatus::Completed);
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
    let out =
        normalize_input_items(&items, &HashMap::new(), tmp.path(), &resolver).expect("normalized");
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
    let transport = MockTransport::new(vec![MockCall {
        stream_events: vec![
            LlmStreamEvent::Delta("What time ".to_string()),
            LlmStreamEvent::Part(LlmOutputPart::Text {
                text: "is it?".to_string(),
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
            },
            &sink,
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    assert_eq!(turn.status, TurnStatus::Completed);
    assert_eq!(turn.done_reason, DoneReason::ModelStop);
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
    let transport = MockTransport::new(vec![MockCall {
        stream_events: vec![
            LlmStreamEvent::Delta("I’m continuing with a type-safety cleanup now: ".to_string()),
            LlmStreamEvent::Part(LlmOutputPart::Text {
                text: "replace the remaining raw JSON paths.".to_string(),
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
            },
            &sink,
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    assert_eq!(turn.status, TurnStatus::Completed);
    assert_eq!(turn.done_reason, DoneReason::ModelStop);
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
async fn standard_runtime_executes_streamed_tool_call_when_final_response_is_empty() {
    let transport = MockTransport::new(vec![
        MockCall {
            stream_events: vec![
                LlmStreamEvent::Part(LlmOutputPart::ToolCall {
                    call_id: "tool-1".to_string(),
                    tool_name: "echo_tool".to_string(),
                    input_json: r#"{"value":"sample"}"#.to_string(),
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
            },
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    assert_eq!(turn.assistant_output.safe_text, "done");
    assert_eq!(projected_tool_calls(&turn.state).len(), 1);
    assert_eq!(
        projected_tool_calls(&turn.state)[0].call_id.as_deref(),
        Some("tool-1")
    );
    assert_eq!(
        projected_tool_calls(&turn.state)[0].result,
        serde_json::json!({
            "payload": "raw:sample"
        })
    );
}

#[tokio::test]
async fn standard_runtime_preserves_part_boundaries_when_response_is_not_streamed() {
    let transport = MockTransport::new(vec![MockCall {
        stream_events: vec![],
        response: Ok(LlmResponse {
            full_text: "Intro paragraph.\n\n## Heading".to_string(),
            parts: vec![
                LlmOutputPart::Text {
                    text: "Intro paragraph.".to_string(),
                },
                LlmOutputPart::Text {
                    text: "## Heading".to_string(),
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
    let transport = MockTransport::new(Vec::new());
    let runtime = rlm_runtime_with_transport(transport).await;
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
    let transport = MockTransport::new(vec![MockCall {
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
    let transport = MockTransport::new(vec![MockCall {
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
async fn standard_runtime_debug_log_records_stream_event_entries() {
    let transport = MockTransport::new(vec![MockCall {
        stream_events: vec![
            LlmStreamEvent::Delta("Hello ".to_string()),
            LlmStreamEvent::Part(LlmOutputPart::Text {
                text: "world".to_string(),
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
            }],
            ..LlmResponse::default()
        }),
    }]);
    let log_path = std::env::temp_dir().join(format!(
        "lash-standard-debug-log-{}-{}.jsonl",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    let mut runtime = standard_runtime_with_transport_and_host(
        transport,
        test_host_config_with_llm_log_path(log_path.clone()),
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
            },
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    assert_eq!(turn.status, TurnStatus::Completed);

    let logged = std::fs::read_to_string(&log_path).expect("read log");
    let entries = logged
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("json log entry"))
        .collect::<Vec<_>>();

    assert!(
        entries
            .iter()
            .any(
                |entry| entry.get("kind").and_then(|v| v.as_str()) == Some("stream_event")
                    && entry.get("event").and_then(|v| v.as_str()) == Some("delta")
                    && entry.get("raw_text").and_then(|v| v.as_str()) == Some("Hello ")
            ),
        "expected delta stream event in log: {entries:?}"
    );
    assert!(
        entries
            .iter()
            .any(
                |entry| entry.get("kind").and_then(|v| v.as_str()) == Some("stream_event")
                    && entry.get("event").and_then(|v| v.as_str()) == Some("text_part")
                    && entry.get("raw_text").and_then(|v| v.as_str()) == Some("world")
            ),
        "expected text_part stream event in log: {entries:?}"
    );
    assert!(
        entries.iter().any(|entry| entry.get("response").is_some()),
        "expected final debug log entry in log: {entries:?}"
    );
    let response_entry = entries
        .iter()
        .find(|entry| entry.get("response").is_some())
        .expect("final response entry");
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

    let _ = std::fs::remove_file(&log_path);
}

#[tokio::test]
async fn standard_runtime_debug_log_records_failed_llm_calls() {
    let transport = MockTransport::new(vec![MockCall {
        stream_events: Vec::new(),
        response: Err(crate::llm::transport::LlmTransportError::new(
            "HTTP request failed: builder error",
        )
        .with_code("builder")
        .with_raw("transport raw body")
        .with_request_body("{\"model\":\"mock-model\"}")),
    }]);
    let log_path = std::env::temp_dir().join(format!(
        "lash-standard-debug-log-error-{}-{}.jsonl",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    let mut runtime = standard_runtime_with_transport_and_host(
        transport,
        test_host_config_with_llm_log_path(log_path.clone()),
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
            },
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    assert_eq!(turn.status, TurnStatus::Failed);
    assert_eq!(turn.errors.len(), 1);
    assert_eq!(turn.errors[0].raw.as_deref(), Some("transport raw body"));

    let logged = std::fs::read_to_string(&log_path).expect("read log");
    let entries = logged
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("json log entry"))
        .collect::<Vec<_>>();
    let error_entry = entries
        .iter()
        .find(|entry| entry.get("kind").and_then(|v| v.as_str()) == Some("llm_error"))
        .expect("llm error entry");
    assert_eq!(
        error_entry["error"]["message"].as_str(),
        Some("HTTP request failed: builder error")
    );
    assert_eq!(error_entry["error"]["code"].as_str(), Some("builder"));
    assert_eq!(error_entry["raw"].as_str(), Some("transport raw body"));
    assert_eq!(
        error_entry["request"].as_str(),
        Some("{\"model\":\"mock-model\"}")
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
    let prompt_usage = normalize_prompt_usage(
        &Provider::OpenAiGeneric {
            api_key: "key".into(),
            base_url: "https://example.invalid/v1".into(),
            options: crate::provider::ProviderOptions::default(),
        },
        &usage,
    )
    .expect("prompt usage");
    assert_eq!(prompt_usage.prompt_context_tokens, 80);
    assert_eq!(prompt_usage.context_budget_tokens, 80);
}

#[cfg(feature = "sqlite-store")]
#[tokio::test]
async fn history_plugin_compacts_using_previous_prompt_usage_across_turns() {
    let transport = MockTransport::new(vec![MockCall {
        stream_events: Vec::new(),
        response: Ok(LlmResponse {
            full_text: "compacted".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "compacted".to_string(),
            }],
            usage: LlmUsage {
                input_tokens: 20,
                output_tokens: 5,
                cached_input_tokens: 0,
                reasoning_tokens: 0,
            },
            ..LlmResponse::default()
        }),
    }]);
    let store = Arc::new(crate::store::Store::memory().expect("store"));
    let base_provider: Arc<dyn crate::ToolProvider> = Arc::new(EmptyTools);
    let base_provider_factory = Arc::clone(&base_provider);
    let plugin_host = crate::PluginHost::new(vec![
        Arc::new(crate::BuiltinRollingHistoryPluginFactory::default()),
        Arc::new(StaticPluginFactory::new(
            "base_tools",
            crate::PluginSpec::new().with_tool_provider(Arc::clone(&base_provider_factory)),
        )),
    ]);
    let plugins = plugin_host
        .build_standard_session("root", None)
        .expect("plugins");
    let mut runtime = LashRuntime::from_persistent_embedded_state(
        standard_test_policy(),
        test_host_config(),
        crate::PersistentRuntimeServices::new(
            Arc::clone(&plugins),
            store.clone() as Arc<dyn crate::store::RuntimeStore>,
        ),
        PersistedSessionState::default(),
    )
    .await
    .expect("runtime");
    runtime.llm_factory = Arc::new(move |_| Box::new(transport.clone()));
    let history = vec![
        text_message("u1", MessageRole::User, &"oldest user".repeat(20)),
        text_message("a1", MessageRole::Assistant, &"oldest assistant".repeat(20)),
        text_message("u2", MessageRole::User, &"older user".repeat(20)),
        text_message("a2", MessageRole::Assistant, &"older assistant".repeat(20)),
        text_message("u3", MessageRole::User, &"recent user".repeat(20)),
        text_message("a3", MessageRole::Assistant, &"recent assistant".repeat(20)),
        text_message("u4", MessageRole::User, &"latest user".repeat(20)),
        text_message("a4", MessageRole::Assistant, &"latest assistant".repeat(20)),
    ];
    let mut state = PersistedSessionState {
        session_id: "root".to_string(),
        policy: SessionPolicy {
            execution_mode: ExecutionMode::Standard,
            ..runtime.policy.clone()
        },
        iteration: 4,
        token_usage: TokenUsage::default(),
        last_prompt_usage: Some(PromptUsage {
            prompt_context_tokens: 70,
            input_tokens: 70,
            cached_input_tokens: 0,
            context_budget_tokens: 70,
        }),
        ..PersistedSessionState::default()
    };
    set_projection(&mut state, &history, &[]);
    runtime.set_persisted_state(state);
    runtime.policy.max_context_tokens = Some(100);
    runtime.state.policy.max_context_tokens = Some(100);

    let turn = runtime
        .run_turn_assembled(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "new request".to_string(),
                }],
                image_blobs: HashMap::new(),
                user_input: None,
                mode: None,
            },
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    assert!(!projected_messages(&turn.state).iter().any(|message| {
        message
            .parts
            .iter()
            .any(|part| part.content.contains("oldest user"))
    }));
    assert!(projected_messages(&turn.state).iter().any(|message| {
        message
            .parts
            .iter()
            .any(|part| part.content.contains("new request"))
    }));
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
                                    .projected_tool_calls()
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
    let transport = MockTransport::new(vec![
        MockCall {
            stream_events: Vec::new(),
            response: Ok(LlmResponse {
                parts: vec![
                    LlmOutputPart::Text {
                        text: "checking tool".to_string(),
                    },
                    LlmOutputPart::ToolCall {
                        call_id: "tool-1".to_string(),
                        tool_name: "echo_tool".to_string(),
                        input_json: r#"{"value":"sample"}"#.to_string(),
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
            },
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    assert!(projected_messages(&turn.state).iter().any(|message| {
        message.parts.iter().any(|part| {
            part.content.contains("model projection") && matches!(part.kind, PartKind::ToolResult)
        })
    }));
    let committed = committed_results.lock().await;
    assert_eq!(
        committed.as_slice(),
        &[(
            serde_json::json!("history projection"),
            serde_json::json!("history projection"),
        )]
    );
    assert_eq!(projected_tool_calls(&turn.state).len(), 1);
    assert_eq!(
        projected_tool_calls(&turn.state)[0].call_id.as_deref(),
        Some("tool-1")
    );
    assert_eq!(turn.tool_calls.len(), 1);
    assert_eq!(turn.tool_calls[0].call_id.as_deref(), Some("tool-1"));
    assert_eq!(
        projected_tool_calls(&turn.state)[0].result,
        serde_json::json!("state projection")
    );
    assert_eq!(
        turn.tool_calls[0].result,
        serde_json::json!("state projection")
    );
}

#[tokio::test]
async fn completed_turns_are_persisted_for_custom_runtime_store() {
    let transport = MockTransport::new(vec![MockCall {
        stream_events: vec![LlmStreamEvent::Delta("Stored answer".to_string())],
        response: Ok(LlmResponse {
            full_text: "Stored answer".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "Stored answer".to_string(),
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
    let plugins = plugin_session_with_tools("root", ExecutionMode::Standard, Arc::new(EmptyTools));
    let mut runtime = LashRuntime::from_persistent_embedded_state(
        standard_test_policy(),
        test_host_config(),
        crate::PersistentRuntimeServices::new(
            Arc::clone(&plugins),
            store.clone() as Arc<dyn crate::store::RuntimeStore>,
        ),
        PersistedSessionState::default(),
    )
    .await
    .expect("runtime");
    runtime.llm_factory = Arc::new(move |_| Box::new(transport.clone()));

    let _turn = runtime
        .run_turn_assembled(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "where did this go?".to_string(),
                }],
                image_blobs: HashMap::new(),
                user_input: None,
                mode: None,
            },
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    let messages = store
        .load_session_head()
        .await
        .expect("session head")
        .graph
        .project_messages();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].role, MessageRole::User);
    assert_eq!(messages[0].parts[0].content, "where did this go?");
    assert_eq!(messages[1].role, MessageRole::Assistant);
    assert_eq!(messages[1].parts[0].content, "Stored answer");
}

#[cfg(feature = "sqlite-store")]
#[tokio::test]
async fn completed_turns_are_persisted_in_session_graph() {
    let transport = MockTransport::new(vec![MockCall {
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

    let store = Arc::new(crate::store::Store::memory().expect("store"));
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
            store.clone() as Arc<dyn crate::store::RuntimeStore>,
        ),
        PersistedSessionState::default(),
    )
    .await
    .expect("runtime");
    runtime.llm_factory = Arc::new(move |_| Box::new(transport.clone()));

    let _turn = runtime
        .run_turn_assembled(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "where did this go?".to_string(),
                }],
                image_blobs: HashMap::new(),
                user_input: None,
                mode: None,
            },
            CancellationToken::new(),
        )
        .await
        .expect("turn");

    let head = store.load_session_head().expect("session head");
    let graph = head.graph;
    let messages = graph.project_messages();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].parts[0].content, "where did this go?");
    assert_eq!(messages[1].parts[0].content, "Stored answer");
    let _checkpoint = head
        .checkpoint_ref
        .as_ref()
        .and_then(|blob_ref| store.get_checkpoint(blob_ref))
        .expect("checkpoint");
    let ledger = head.token_ledger;
    assert_eq!(ledger.len(), 1);
    assert_eq!(ledger[0].source, "turn");
    assert_eq!(ledger[0].model, standard_test_policy().model);
    assert_eq!(ledger[0].usage.input_tokens, 12);
    assert_eq!(ledger[0].usage.output_tokens, 4);
    assert_eq!(ledger[0].usage.cached_input_tokens, 1);
    assert_eq!(ledger[0].usage.reasoning_tokens, 2);
}

#[cfg(all(feature = "sqlite-store", feature = "tool-impls"))]
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
    let transport = MockTransport::new(vec![
        MockCall {
            stream_events: vec![
                LlmStreamEvent::Delta("stored".to_string()),
                LlmStreamEvent::Usage(first_usage.clone()),
            ],
            response: Ok(LlmResponse {
                full_text: "stored".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "stored".to_string(),
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
                }],
                usage: second_usage.clone(),
                ..LlmResponse::default()
            }),
        },
    ]);

    let store = Arc::new(crate::store::Store::memory().expect("store"));
    let store_trait = store.clone() as Arc<dyn crate::store::RuntimeStore>;

    let mut runtime = rlm_runtime_with_transport_and_store(transport.clone(), store_trait).await;
    let first_turn = runtime
        .run_turn_assembled(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "remember this".to_string(),
                }],
                image_blobs: HashMap::new(),
                user_input: None,
                mode: None,
            },
            CancellationToken::new(),
        )
        .await
        .expect("first turn");
    assert_eq!(first_turn.token_usage.input_tokens, 12);
    assert_eq!(first_turn.token_usage.output_tokens, 4);
    assert_eq!(first_turn.token_usage.cached_input_tokens, 1);
    assert_eq!(first_turn.token_usage.reasoning_tokens, 2);

    let resumed_head = store.load_session_head().expect("resumed head");
    let resumed_checkpoint = resumed_head
        .checkpoint_ref
        .as_ref()
        .and_then(|blob_ref| store.get_checkpoint(blob_ref))
        .expect("resumed checkpoint");
    let mut resumed = LashRuntime::from_persistent_embedded_state(
        rlm_test_policy(),
        test_host_config(),
        crate::PersistentRuntimeServices::new_with_bridges(
            default_tool_session("root", ExecutionMode::Rlm, true),
            crate::TurnInjectionBridge::new(),
            crate::TurnInputInjectionBridge::new(),
            store.clone() as Arc<dyn crate::store::RuntimeStore>,
        ),
        PersistedSessionState {
            policy: SessionPolicy {
                execution_mode: ExecutionMode::Rlm,
                ..Default::default()
            },
            session_graph: resumed_head.graph,
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
            token_ledger: resumed_head.token_ledger.clone(),
            checkpoint_ref: resumed_head.checkpoint_ref.clone(),
            ..PersistedSessionState::default()
        },
    )
    .await
    .expect("resumed runtime");
    resumed.llm_factory = Arc::new(move |_| Box::new(transport.clone()));

    let second_turn = resumed
        .run_turn_assembled(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "what did you store?".to_string(),
                }],
                image_blobs: HashMap::new(),
                user_input: None,
                mode: None,
            },
            CancellationToken::new(),
        )
        .await
        .expect("second turn");
    assert_eq!(second_turn.token_usage.input_tokens, 30);
    assert_eq!(second_turn.token_usage.output_tokens, 7);
    assert_eq!(second_turn.token_usage.cached_input_tokens, 5);
    assert_eq!(second_turn.token_usage.reasoning_tokens, 6);

    let head = store.load_session_head().expect("session head");
    let checkpoint = head
        .checkpoint_ref
        .as_ref()
        .and_then(|blob_ref| store.get_checkpoint(blob_ref))
        .expect("checkpoint");
    let turn_state = checkpoint.turn_state;
    assert_eq!(turn_state.token_usage.input_tokens, 30);
    assert_eq!(turn_state.token_usage.output_tokens, 7);
    assert_eq!(turn_state.token_usage.cached_input_tokens, 5);
    assert_eq!(turn_state.token_usage.reasoning_tokens, 6);

    let ledger = head.token_ledger;
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
    let runtime =
        standard_runtime_with_transport_and_background(MockTransport::new(Vec::new())).await;
    let manager = runtime.session_manager().expect("session manager");
    let observed = Arc::new(AtomicBool::new(false));
    let observed_task = Arc::clone(&observed);

    manager
        .spawn_background_job(
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
    let executor: Arc<dyn BackgroundExecutor> = Arc::new(TokioBackgroundExecutor::default());
    let runtime_one = standard_runtime_with_shared_background_executor(
        MockTransport::new(Vec::new()),
        Arc::clone(&executor),
    )
    .await;
    let runtime_two = standard_runtime_with_shared_background_executor(
        MockTransport::new(Vec::new()),
        Arc::clone(&executor),
    )
    .await;
    let manager_one = runtime_one.session_manager().expect("session manager");
    let observed = Arc::new(AtomicBool::new(false));
    let observed_task = Arc::clone(&observed);
    manager_one
        .spawn_background_job(
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

#[cfg(feature = "sqlite-store")]
#[tokio::test]
async fn observational_memory_background_work_appends_buffered_nodes() {
    let transport = MockTransport::new(vec![
        MockCall {
            stream_events: vec![LlmStreamEvent::Delta("stored".to_string())],
            response: Ok(LlmResponse {
                full_text: "stored".to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "stored".to_string(),
                }],
                usage: LlmUsage {
                    input_tokens: 100,
                    output_tokens: 1,
                    cached_input_tokens: 0,
                    reasoning_tokens: 0,
                },
                ..LlmResponse::default()
            }),
        },
        MockCall {
            stream_events: Vec::new(),
            response: Ok(LlmResponse {
                full_text: "<observations>\nDate: Apr 14, 2026\n* 🔴 User stated they graduated with a degree in Business Administration.\n</observations>\n<current-task>\nPrimary: retain user memory\n</current-task>\n<suggested-response>\nAcknowledge storage.\n</suggested-response>"
                    .to_string(),
                parts: vec![LlmOutputPart::Text {
                    text: "<observations>\nDate: Apr 14, 2026\n* 🔴 User stated they graduated with a degree in Business Administration.\n</observations>\n<current-task>\nPrimary: retain user memory\n</current-task>\n<suggested-response>\nAcknowledge storage.\n</suggested-response>"
                        .to_string(),
                }],
                usage: LlmUsage {
                    input_tokens: 80,
                    output_tokens: 40,
                    cached_input_tokens: 0,
                    reasoning_tokens: 0,
                },
                ..LlmResponse::default()
            }),
        },
    ]);
    let (mut runtime, store) = om_runtime_with_transport_and_background(
        transport.clone(),
        crate::ObservationalMemoryConfig {
            observation_message_tokens: 40,
            observation_buffer_tokens: 20,
            observation_block_after_tokens: 60,
            observation_max_tokens_per_batch: 200,
            previous_observer_tokens: 50,
            reflection_observation_tokens: 10_000,
            reflection_buffer_activation_bps: 5_000,
            reflection_block_after_tokens: 12_000,
        },
    )
    .await;

    runtime
        .stream_turn(
            TurnInput {
                items: vec![InputItem::Text {
                    text: "I graduated with a degree in Business Administration, please remember that."
                        .to_string(),
                }],
                image_blobs: HashMap::new(),
                user_input: None,
                mode: None,
            },
            &NoopEventSink,
            CancellationToken::new(),
        )
        .await
        .expect("turn");
    runtime
        .await_background_work()
        .await
        .expect("await background work");

    let graph = runtime.export_state().session_graph;
    let usage_sources = runtime
        .usage_report()
        .by_source
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    let plugin_types = graph
        .active_path_nodes()
        .into_iter()
        .filter_map(|node| node.plugin().map(|(kind, _)| kind.to_string()))
        .collect::<Vec<_>>();
    let persisted_plugin_types = store
        .load_session_head()
        .expect("head")
        .graph
        .active_path_nodes()
        .into_iter()
        .filter_map(|node| node.plugin().map(|(kind, _)| kind.to_string()))
        .collect::<Vec<_>>();
    assert!(
        plugin_types
            .iter()
            .any(|kind| kind == "lash.context.observational_memory.buffered_observation"),
        "expected OM buffered observation node, got runtime={plugin_types:?}; persisted={persisted_plugin_types:?}; usage_sources={usage_sources:?}; remaining_calls={}",
        transport.calls.lock().expect("transport calls").len()
    );
}

#[cfg(feature = "sqlite-store")]
#[tokio::test]
async fn history_plugin_compacts_messages_when_model_change_shrinks_context_window() {
    let transport = MockTransport::new(vec![MockCall {
        stream_events: Vec::new(),
        response: Ok(LlmResponse {
            full_text: "compacted summary".to_string(),
            parts: vec![LlmOutputPart::Text {
                text: "compacted summary".to_string(),
            }],
            usage: LlmUsage {
                input_tokens: 20,
                output_tokens: 5,
                cached_input_tokens: 0,
                reasoning_tokens: 0,
            },
            ..LlmResponse::default()
        }),
    }]);
    let store = Arc::new(crate::store::Store::memory().expect("store"));
    let base_provider: Arc<dyn crate::ToolProvider> = Arc::new(EmptyTools);
    let base_provider_factory = Arc::clone(&base_provider);
    let plugin_host = crate::PluginHost::new(vec![
        Arc::new(crate::BuiltinRollingHistoryPluginFactory::default()),
        Arc::new(StaticPluginFactory::new(
            "base_tools",
            crate::PluginSpec::new().with_tool_provider(Arc::clone(&base_provider_factory)),
        )),
    ]);
    let plugins = plugin_host
        .build_standard_session("root", None)
        .expect("plugins");
    let mut runtime = LashRuntime::from_persistent_embedded_state(
        standard_test_policy(),
        test_host_config(),
        crate::PersistentRuntimeServices::new(
            Arc::clone(&plugins),
            store.clone() as Arc<dyn crate::store::RuntimeStore>,
        ),
        {
            let history = vec![
                text_message("u1", MessageRole::User, &"oldest user".repeat(20)),
                text_message("a1", MessageRole::Assistant, &"oldest assistant".repeat(20)),
                text_message("u2", MessageRole::User, &"older user".repeat(20)),
                text_message("a2", MessageRole::Assistant, &"older assistant".repeat(20)),
                text_message("u3", MessageRole::User, &"recent user".repeat(20)),
                text_message("a3", MessageRole::Assistant, &"recent assistant".repeat(20)),
                text_message("u4", MessageRole::User, &"latest user".repeat(20)),
                text_message("a4", MessageRole::Assistant, &"latest assistant".repeat(20)),
            ];
            let mut state = PersistedSessionState {
                session_id: "root".to_string(),
                policy: SessionPolicy {
                    execution_mode: ExecutionMode::Standard,
                    ..Default::default()
                },
                iteration: 4,
                token_usage: TokenUsage::default(),
                last_prompt_usage: Some(PromptUsage {
                    prompt_context_tokens: 70_000,
                    input_tokens: 70_000,
                    cached_input_tokens: 0,
                    context_budget_tokens: 70_000,
                }),
                ..PersistedSessionState::default()
            };
            set_projection(&mut state, &history, &[]);
            state
        },
    )
    .await
    .expect("runtime");
    runtime.llm_factory = Arc::new(move |_| Box::new(transport.clone()));

    runtime
        .update_session_config(
            Some(crate::Provider::GoogleOAuth {
                access_token: "tok".into(),
                refresh_token: "ref".into(),
                expires_at: u64::MAX,
                project_id: None,
                options: crate::provider::ProviderOptions::default(),
            }),
            Some("gemini-2.5-flash-image".to_string()),
            None,
            Some(32_768),
        )
        .await;

    assert!(!projected_messages(&runtime.state).iter().any(|message| {
        message
            .parts
            .iter()
            .any(|part| part.content.contains("oldest user"))
    }));
    assert_eq!(runtime.session_policy().max_context_tokens, Some(32_768));
}

#[tokio::test]
async fn drain_standard_stream_queue_forwards_prequeued_text() {
    let (event_tx, mut event_rx) = mpsc::channel(8);
    let (llm_stream_tx, mut llm_stream_rx) =
        tokio::sync::mpsc::unbounded_channel::<LlmStreamEvent>();
    llm_stream_tx
        .send(LlmStreamEvent::Delta("Hello".to_string()))
        .expect("delta");
    llm_stream_tx
        .send(LlmStreamEvent::Part(LlmOutputPart::Text {
            text: " there".to_string(),
        }))
        .expect("part");
    drop(llm_stream_tx);

    let mut text_streamed = false;
    let mut streamed_usage = LlmUsage::default();
    drain_standard_stream_queue(
        &event_tx,
        &mut llm_stream_rx,
        &mut text_streamed,
        &mut streamed_usage,
    )
    .await;
    drop(event_tx);

    let mut streamed_text = String::new();
    while let Some(event) = event_rx.recv().await {
        if let SessionEvent::TextDelta { content } = event {
            streamed_text.push_str(&content);
        }
    }

    assert!(text_streamed);
    assert_eq!(streamed_text, "Hello there");
}

#[tokio::test]
async fn set_state_syncs_runtime_policy_with_restored_state_policy() {
    let plugins = crate::PluginHost::new(Vec::new())
        .build_standard_session("root", None)
        .expect("plugins");
    let mut runtime = LashRuntime::from_embedded_state(
        standard_test_policy(),
        test_host_config(),
        crate::RuntimeServices::new(plugins),
        PersistedSessionState::default(),
    )
    .await
    .expect("runtime");

    let mut state = runtime.state.clone();
    state.policy.model = "restored-model".to_string();
    state.policy.execution_mode = ExecutionMode::Rlm;
    runtime.set_persisted_state(state);

    assert_eq!(runtime.state.policy.model, "restored-model");
    assert_eq!(runtime.policy.model, "restored-model");
    assert_eq!(runtime.state.policy.execution_mode, ExecutionMode::Rlm);
    assert_eq!(runtime.policy.execution_mode, ExecutionMode::Rlm);
}
