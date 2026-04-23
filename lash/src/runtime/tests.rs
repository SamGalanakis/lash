use super::*;
use serde_json::json;
use sha2::Digest;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::llm::transport::LlmTransportError;
use crate::llm::types::LlmUsage;
use crate::plugin::StaticPluginFactory;
use crate::store::RuntimeStore;
use crate::testing::TestProvider;
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
    fn append_message(&mut self, message: Message);
}

impl ProjectionStateMut for SessionStateEnvelope {
    fn append_message(&mut self, message: Message) {
        self.session_graph.append_message(message);
    }
}

impl ProjectionStateMut for PersistedSessionState {
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

fn append_message(state: &mut impl ProjectionStateMut, message: Message) {
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
    blobs: Mutex<HashMap<String, Vec<u8>>>,
    session_head_meta: Mutex<Option<crate::SessionHeadMeta>>,
    session_graph: Mutex<crate::SessionGraph>,
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

    async fn save_session_meta(&self, _meta: crate::store::SessionMeta) {}

    async fn load_session_meta(&self) -> Option<crate::store::SessionMeta> {
        None
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
        execution_mode: ExecutionMode::Standard,
        provider: mock_provider(Vec::new()).into_handle(),
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
        Arc::new(crate::testing::FakeContextApproachPluginFactory::rolling_history()),
        Arc::new(crate::testing::FakeContextApproachPluginFactory::observational_memory()),
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

async fn standard_runtime_with_transport(transport: TestProvider) -> LashRuntime {
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
    runtime.policy.provider = transport.clone().into_handle();
    runtime
}

#[test]
fn plugin_host_rejects_observational_memory_without_supporting_plugin() {
    let host = crate::PluginHost::new(vec![Arc::new(
        crate::testing::FakeContextApproachPluginFactory::rolling_history(),
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
            ExecutionMode::Standard,
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
            ExecutionMode::Standard,
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
            ExecutionMode::Standard,
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
            execution_mode: ExecutionMode::Standard,
            provider: mock_provider(Vec::new()).into_handle(),
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
        provider: mock_provider(Vec::new()).into_handle(),
        model: "mock-model".to_string(),
        max_context_tokens: Some(200_000),
        ..SessionPolicy::default()
    }
}

#[cfg(feature = "tool-impls")]
async fn rlm_runtime_with_transport(transport: TestProvider) -> LashRuntime {
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
    runtime.policy.provider = transport.clone().into_handle();
    runtime
}

#[cfg(all(feature = "sqlite-store", feature = "tool-impls"))]
async fn rlm_runtime_with_transport_and_store(
    transport: TestProvider,
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
    runtime.policy.provider = transport.clone().into_handle();
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
    transport: TestProvider,
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
            plugin_session_with_tools("root", ExecutionMode::Standard, tools),
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
        vec![crate::ToolDefinition {
            name: "echo_tool".to_string(),
            description: "Return a tool payload".to_string(),
            params: vec![crate::ToolParam::typed("value", "str")],
            returns: "json".to_string(),
            examples: vec![],
            availability: crate::ToolAvailabilityConfig::documented(),
            activation: crate::ToolActivation::Always,
            availability_override: None,
            input_schema_override: None,
            output_schema_override: None,
            execution_mode: crate::ToolExecutionMode::Parallel,
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

/// Tool that sleeps for 10 seconds unless its future is aborted or the
/// execution-context cancellation token fires. Used to verify that turn
/// cancellation unwinds in-flight tool tasks promptly.
struct SlowTool {
    observed_cancel: Arc<AtomicBool>,
}

#[async_trait::async_trait]
impl crate::ToolProvider for SlowTool {
    fn definitions(&self) -> Vec<crate::ToolDefinition> {
        vec![crate::ToolDefinition {
            name: "slow_tool".to_string(),
            description: "Sleep for a long time; respects cancellation.".to_string(),
            params: vec![],
            returns: "json".to_string(),
            examples: vec![],
            availability: crate::ToolAvailabilityConfig::documented(),
            activation: crate::ToolActivation::Always,
            availability_override: None,
            execution_mode: crate::ToolExecutionMode::Parallel,
            input_schema_override: None,
            output_schema_override: None,
        }]
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
                rlm_termination_override: None,
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
    let transport = mock_provider(vec![MockCall {
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
                rlm_termination_override: None,
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
                rlm_termination_override: None,
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
    let transport = mock_provider(vec![
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
                rlm_termination_override: None,
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
                    tool_item_id: None,
                    tool_signature: None,
                    prune_state: crate::PruneState::Intact,
                    reasoning_meta: None,
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
                rlm_termination_override: None,
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
    let transport = mock_provider(vec![
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
                rlm_termination_override: None,
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

    let transport = mock_provider(vec![MockCall {
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
                rlm_termination_override: None,
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
    let transport = mock_provider(vec![MockCall {
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
                rlm_termination_override: None,
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
            availability: crate::ToolAvailabilityConfig::documented(),
            activation: crate::ToolActivation::Always,
            availability_override: None,
            input_schema_override: None,
            output_schema_override: None,
            execution_mode: crate::ToolExecutionMode::Parallel,
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
            availability: crate::ToolAvailabilityConfig::documented(),
            activation: crate::ToolActivation::Always,
            availability_override: None,
            input_schema_override: None,
            output_schema_override: None,
            execution_mode: crate::ToolExecutionMode::Parallel,
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
                session_id: Some("subagent-child".to_string()),
                parent_session_id: Some(context.session_id.clone()),
                start: crate::SessionStartPoint::Empty,
                policy: None,
                plugin_mode: crate::SessionPluginMode::InheritCurrent,
                initial_nodes: Vec::new(),
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
                    rlm_termination_override: None,
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
                rlm_termination_override: None,
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
                rlm_termination_override: None,
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
                tool_item_id: None,
                tool_signature: None,
                prune_state: PruneState::Intact,
                reasoning_meta: None,
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
                tool_item_id: None,
                tool_signature: None,
                prune_state: PruneState::Intact,
                reasoning_meta: None,
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
                },
            ],
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
    assert_eq!(out.status, TurnStatus::Interrupted);
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
                tool_item_id: None,
                tool_signature: None,
                prune_state: PruneState::Intact,
                reasoning_meta: None,
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
    let transport = mock_provider(vec![MockCall {
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
                rlm_termination_override: None,
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
    let transport = mock_provider(vec![MockCall {
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
                rlm_termination_override: None,
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
                rlm_termination_override: None,
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
                rlm_termination_override: None,
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
    let transport = mock_provider(vec![MockCall {
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
                rlm_termination_override: None,
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
                rlm_termination_override: None,
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
                rlm_termination_override: None,
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
    let transport = mock_provider(vec![MockCall {
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
                rlm_termination_override: None,
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
    let transport = mock_provider(vec![MockCall {
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
                rlm_termination_override: None,
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
    let stub = mock_provider(Vec::new());
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
    let transport = mock_provider(vec![
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
                rlm_termination_override: None,
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
    let transport = mock_provider(vec![MockCall {
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
                rlm_termination_override: None,
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
                rlm_termination_override: None,
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
                rlm_termination_override: None,
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
                rlm_termination_override: None,
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

#[cfg(all(feature = "sqlite-store", feature = "tool-impls"))]
#[tokio::test]
async fn environment_park_resume_preserves_active_path() {
    // Build a RuntimeEnvironment with the usual test tool factories +
    // Residency::ActivePathOnly (webserver pattern; host owns disk).
    let factories: Vec<Arc<dyn crate::PluginFactory>> = vec![
        Arc::new(crate::BuiltinToolResultProjectionPluginFactory::default()),
        Arc::new(crate::testing::FakeContextApproachPluginFactory::rolling_history()),
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
    let store: Arc<dyn crate::store::RuntimeStore> =
        Arc::new(crate::store::Store::memory().expect("in-memory store"));

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
    let report = store.vacuum().await;
    assert_eq!(report.removed_node_count, 0);
}
