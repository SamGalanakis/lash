//! In-tree test fixtures shared across the lash crate's test modules.
//!
//! Cuts down on per-test-module `MockSessionManager` boilerplate by
//! providing a configurable mock implementation plus a couple of small
//! builders for common policy / turn fixtures.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use crate::llm::transport::LlmTransportError;
use crate::llm::types::{LlmRequest, LlmResponse};
use crate::plugin::{PluginError, SessionCreateRequest, SessionHandle, SessionSnapshot};
use crate::provider::{
    AgentModelSelection, ProviderComponents, ProviderHandle, ProviderModelPolicy, ProviderState,
    ProviderTransport, VariantRequestConfig,
};
use crate::session_model::{ConversationRecord, SessionEventRecord};
use crate::{
    AssembledTurn, AssistantOutput, ExecutionMode, ExecutionSummary, OutputState,
    PersistedSessionState, ProcessRegistry, ProviderOptions, SessionPolicy, SessionStateEnvelope,
    TokenUsage, TurnFinish, TurnInput, TurnOutcome, TurnStop,
};

type CompletionFuture =
    Pin<Box<dyn Future<Output = Result<LlmResponse, LlmTransportError>> + Send>>;
type CompletionFn = dyn Fn(LlmRequest) -> CompletionFuture + Send + Sync;
type SupportedVariantsFn = dyn Fn(&str) -> &'static [&'static str] + Send + Sync;
type DefaultVariantFn = dyn Fn(&str) -> Option<&'static str> + Send + Sync;
type RequestVariantConfigFn = dyn Fn(&str, &str) -> Option<VariantRequestConfig> + Send + Sync;
type DefaultAgentModelFn = dyn Fn(&str) -> Option<AgentModelSelection> + Send + Sync;
type SerializeConfigFn = dyn Fn() -> serde_json::Value + Send + Sync;

fn no_supported_variants(_model: &str) -> &'static [&'static str] {
    &[]
}

fn no_default_variant(_model: &str) -> Option<&'static str> {
    None
}

fn no_request_variant_config(_model: &str, _variant: &str) -> Option<VariantRequestConfig> {
    None
}

fn no_default_agent_model(_tier: &str) -> Option<AgentModelSelection> {
    None
}

fn empty_provider_config() -> serde_json::Value {
    serde_json::Value::Object(Default::default())
}

/// Configurable provider fixture used by lash's own tests and shared
/// with downstream plugin crates through `lash_core::testing`.
#[derive(Clone)]
pub struct TestProvider {
    kind: &'static str,
    default_model: String,
    supported_variants: Arc<SupportedVariantsFn>,
    default_model_variant: Arc<DefaultVariantFn>,
    request_variant_config: Arc<RequestVariantConfigFn>,
    default_agent_model: Arc<DefaultAgentModelFn>,
    requires_streaming: bool,
    options: ProviderOptions,
    serialize_config: Arc<SerializeConfigFn>,
    complete: Arc<CompletionFn>,
}

impl std::fmt::Debug for TestProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TestProvider")
            .field("kind", &self.kind)
            .field("default_model", &self.default_model)
            .field("requires_streaming", &self.requires_streaming)
            .field("options", &self.options)
            .finish_non_exhaustive()
    }
}

impl Default for TestProvider {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl TestProvider {
    pub fn builder() -> TestProviderBuilder {
        TestProviderBuilder::new()
    }

    pub fn into_handle(self) -> ProviderHandle {
        let model_policy: Arc<dyn ProviderModelPolicy> = Arc::new(self.clone());
        ProviderHandle::new(ProviderComponents::shared(self, model_policy))
    }
}

pub struct TestProviderBuilder {
    provider: TestProvider,
}

impl TestProviderBuilder {
    pub fn new() -> Self {
        Self {
            provider: TestProvider {
                kind: "test",
                default_model: "mock-model".to_string(),
                supported_variants: Arc::new(no_supported_variants),
                default_model_variant: Arc::new(no_default_variant),
                request_variant_config: Arc::new(no_request_variant_config),
                default_agent_model: Arc::new(no_default_agent_model),
                requires_streaming: false,
                options: ProviderOptions::default(),
                serialize_config: Arc::new(empty_provider_config),
                complete: Arc::new(|_request| {
                    Box::pin(async {
                        Err(LlmTransportError::new(
                            "TestProvider::complete was called without a test completion handler",
                        ))
                    })
                }),
            },
        }
    }

    pub fn kind(mut self, kind: &'static str) -> Self {
        self.provider.kind = kind;
        self
    }

    pub fn default_model(mut self, model: impl Into<String>) -> Self {
        self.provider.default_model = model.into();
        self
    }

    pub fn supported_variants<F>(mut self, supported_variants: F) -> Self
    where
        F: Fn(&str) -> &'static [&'static str] + Send + Sync + 'static,
    {
        self.provider.supported_variants = Arc::new(supported_variants);
        self
    }

    pub fn default_model_variant<F>(mut self, default_model_variant: F) -> Self
    where
        F: Fn(&str) -> Option<&'static str> + Send + Sync + 'static,
    {
        self.provider.default_model_variant = Arc::new(default_model_variant);
        self
    }

    pub fn request_variant_config<F>(mut self, request_variant_config: F) -> Self
    where
        F: Fn(&str, &str) -> Option<VariantRequestConfig> + Send + Sync + 'static,
    {
        self.provider.request_variant_config = Arc::new(request_variant_config);
        self
    }

    pub fn default_agent_model<F>(mut self, default_agent_model: F) -> Self
    where
        F: Fn(&str) -> Option<AgentModelSelection> + Send + Sync + 'static,
    {
        self.provider.default_agent_model = Arc::new(default_agent_model);
        self
    }

    pub fn requires_streaming(mut self, requires_streaming: bool) -> Self {
        self.provider.requires_streaming = requires_streaming;
        self
    }

    pub fn options(mut self, options: ProviderOptions) -> Self {
        self.provider.options = options;
        self
    }

    pub fn serialize_config<F>(mut self, serialize_config: F) -> Self
    where
        F: Fn() -> serde_json::Value + Send + Sync + 'static,
    {
        self.provider.serialize_config = Arc::new(serialize_config);
        self
    }

    pub fn complete<F, Fut>(mut self, complete: F) -> Self
    where
        F: Fn(LlmRequest) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<LlmResponse, LlmTransportError>> + Send + 'static,
    {
        self.provider.complete = Arc::new(move |request| Box::pin(complete(request)));
        self
    }

    pub fn complete_error(mut self, message: impl Into<String>) -> Self {
        let message = Arc::new(message.into());
        self.provider.complete = Arc::new(move |_request| {
            let message = Arc::clone(&message);
            Box::pin(async move { Err(LlmTransportError::new(message.as_str())) })
        });
        self
    }

    pub fn build(self) -> TestProvider {
        self.provider
    }
}

impl Default for TestProviderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ProviderState for TestProvider {
    fn kind(&self) -> &'static str {
        self.kind
    }

    fn options(&self) -> ProviderOptions {
        self.options.clone()
    }

    fn set_options(&mut self, options: ProviderOptions) {
        self.options = options;
    }

    fn serialize_config(&self) -> serde_json::Value {
        (self.serialize_config)()
    }

    fn clone_boxed(&self) -> Box<dyn ProviderState> {
        Box::new(self.clone())
    }
}

#[async_trait::async_trait]
impl ProviderTransport for TestProvider {
    async fn complete(&mut self, request: LlmRequest) -> Result<LlmResponse, LlmTransportError> {
        (self.complete)(request).await
    }

    fn requires_streaming(&self) -> bool {
        self.requires_streaming
    }

    fn clone_boxed(&self) -> Box<dyn ProviderTransport> {
        Box::new(self.clone())
    }
}

impl ProviderModelPolicy for TestProvider {
    fn default_agent_model(&self, tier: &str) -> Option<AgentModelSelection> {
        (self.default_agent_model)(tier)
    }

    fn default_model(&self) -> &str {
        &self.default_model
    }

    fn supported_variants(&self, model: &str) -> &'static [&'static str] {
        (self.supported_variants)(model)
    }

    fn default_model_variant(&self, model: &str) -> Option<&'static str> {
        (self.default_model_variant)(model)
    }

    fn request_variant_config(&self, model: &str, variant: &str) -> Option<VariantRequestConfig> {
        (self.request_variant_config)(model, variant)
    }
}

/// Build a `SessionPolicy` populated with the canonical stub provider
/// + model used by lash's in-tree tests.
pub fn mock_session_policy() -> SessionPolicy {
    SessionPolicy {
        provider: TestProvider::builder()
            .kind("stub")
            .default_model("mock-model")
            .complete_error(
                "TestProvider::complete was called; tests must supply a real provider or mock",
            )
            .build()
            .into_handle(),
        model: "mock-model".to_string(),
        execution_mode: ExecutionMode::standard(),
        ..Default::default()
    }
}

/// A `ToolContext` backed by a default [`MockSessionManager`], suitable for
/// unit-testing a `ToolProvider` in isolation. Use [`mock_tool_context_with_host`]
/// when the tool under test interacts with host state and needs a configured
/// `MockSessionManager` (or another `RuntimeSessionHost` implementation).
pub fn mock_tool_context() -> crate::ToolContext<'static> {
    mock_tool_context_with_host(Arc::new(MockSessionManager::default()))
}

/// Like [`mock_tool_context`], but lets the caller supply the host. Useful
/// when a tool reads from the host (snapshots, tool state, lifecycle hooks)
/// and the test wants to assert against captured interactions.
pub fn mock_tool_context_with_host(
    host: Arc<dyn crate::plugin::RuntimeSessionHost>,
) -> crate::ToolContext<'static> {
    mock_tool_context_with_host_and_direct_completions(
        host,
        crate::DirectCompletionClient::unavailable(
            "direct completions are unavailable in this test context",
        ),
    )
}

pub fn mock_tool_context_with_host_and_direct_completions(
    host: Arc<dyn crate::plugin::RuntimeSessionHost>,
    direct_completions: crate::DirectCompletionClient<'static>,
) -> crate::ToolContext<'static> {
    crate::tool_provider::ToolContext::__for_testing(
        "test-session".to_string(),
        host,
        crate::TurnContext::new(),
        Arc::new(crate::InMemoryAttachmentStore::new()),
        direct_completions,
        None,
    )
}

/// Convenience helper for the common tool-test shape: build a
/// [`mock_tool_context`], wrap `name` + `args` in a `ToolCall`, and `await`
/// the provider's `execute`. Use this for unit tests that don't need to
/// inspect host interactions; call `mock_tool_context()` directly and
/// construct `ToolCall` manually for more involved scenarios.
pub async fn run_tool<P>(tool: &P, name: &str, args: &serde_json::Value) -> crate::ToolResult
where
    P: crate::ToolProvider + ?Sized,
{
    let context = mock_tool_context();
    tool.execute(crate::ToolCall {
        name,
        args,
        context: &context,
        progress: None,
    })
    .await
}

/// Build an empty `AssembledTurn` whose assistant text is `summary`.
pub fn mock_assembled_turn(session_id: &str, summary: &str) -> AssembledTurn {
    AssembledTurn {
        state: SessionStateEnvelope {
            session_id: session_id.to_string(),
            policy: SessionPolicy {
                execution_mode: ExecutionMode::standard(),
                ..Default::default()
            },
            ..Default::default()
        },
        outcome: TurnOutcome::Finished(TurnFinish::AssistantMessage {
            text: summary.to_string(),
        }),
        assistant_output: AssistantOutput {
            safe_text: summary.to_string(),
            raw_text: summary.to_string(),
            state: OutputState::Usable,
        },
        execution: ExecutionSummary {
            mode: ExecutionMode::standard(),
            had_tool_calls: false,
            had_code_execution: false,
        },
        token_usage: TokenUsage::default(),
        children_usage: Vec::new(),
        tool_calls: Vec::new(),
        errors: Vec::new(),
    }
}

/// Configurable mock for host capability traits. Tests override
/// the snapshot, tool catalog, and turn outcome via the builder
/// methods; mutations (`create_session`, `close_session`)
/// are recorded so tests can assert against them.
pub struct MockSessionManager {
    pub snapshot: SessionSnapshot,
    pub tool_catalog: Vec<serde_json::Value>,
    pub turn: AssembledTurn,
    pub tool_registry: Option<crate::ToolRegistry>,
    pub process_registry: Arc<crate::LocalProcessRegistry>,
    pub created: Mutex<Vec<SessionCreateRequest>>,
    pub closed: Mutex<Vec<String>>,
}

impl Default for MockSessionManager {
    fn default() -> Self {
        Self {
            snapshot: PersistedSessionState::default(),
            tool_catalog: Vec::new(),
            turn: mock_assembled_turn("root", ""),
            tool_registry: None,
            process_registry: Arc::new(crate::LocalProcessRegistry::default()),
            created: Mutex::new(Vec::new()),
            closed: Mutex::new(Vec::new()),
        }
    }
}

impl MockSessionManager {
    #[allow(dead_code)]
    pub fn with_snapshot(mut self, snapshot: SessionSnapshot) -> Self {
        self.snapshot = snapshot;
        self
    }

    pub fn with_tool_catalog(mut self, catalog: Vec<serde_json::Value>) -> Self {
        self.tool_catalog = catalog;
        self
    }

    pub fn with_turn(mut self, turn: AssembledTurn) -> Self {
        self.turn = turn;
        self
    }

    #[allow(dead_code)]
    pub fn with_tool_registry(mut self, tool_registry: crate::ToolRegistry) -> Self {
        self.tool_registry = Some(tool_registry);
        self
    }

    /// Snapshot of the requests captured by `create_session`. Panics if
    /// the lock is poisoned (a panic from another test thread).
    pub fn created_snapshot(&self) -> Vec<SessionCreateRequest> {
        self.created.lock().expect("created lock").clone()
    }
}

#[async_trait::async_trait]
impl crate::plugin::RuntimeSessionHost for MockSessionManager {
    async fn snapshot_current(&self) -> Result<SessionSnapshot, PluginError> {
        Ok(self.snapshot.clone())
    }

    async fn snapshot_session(&self, _session_id: &str) -> Result<SessionSnapshot, PluginError> {
        Ok(self.snapshot.clone())
    }
    async fn tool_catalog(&self, _session_id: &str) -> Result<Vec<serde_json::Value>, PluginError> {
        Ok(self.tool_catalog.clone())
    }
    async fn tool_state(&self, _session_id: &str) -> Result<crate::ToolState, PluginError> {
        self.tool_registry
            .as_ref()
            .map(crate::ToolRegistry::export_state)
            .ok_or_else(|| {
                PluginError::Session("tool state is unavailable in this session".to_string())
            })
    }

    async fn apply_tool_state(
        &self,
        _session_id: &str,
        snapshot: crate::ToolState,
    ) -> Result<u64, PluginError> {
        let Some(tool_registry) = self.tool_registry.as_ref() else {
            return Err(PluginError::Session(
                "tool state mutation is unavailable in this session".to_string(),
            ));
        };
        tool_registry
            .apply_state(snapshot)
            .map_err(|err| PluginError::Session(err.to_string()))
    }
    async fn create_session(
        &self,
        request: SessionCreateRequest,
    ) -> Result<SessionHandle, PluginError> {
        self.created
            .lock()
            .expect("created lock")
            .push(request.clone());
        Ok(SessionHandle {
            session_id: request
                .session_id
                .clone()
                .unwrap_or_else(|| "child".to_string()),
            parent_session_id: request.relation.parent_session_id().map(ToOwned::to_owned),
            policy: request.policy.unwrap_or_else(mock_session_policy),
        })
    }

    async fn close_session(&self, session_id: &str) -> Result<(), PluginError> {
        self.closed
            .lock()
            .expect("closed lock")
            .push(session_id.to_string());
        Ok(())
    }
    async fn start_turn(
        &self,
        _session_id: &str,
        _input: TurnInput,
    ) -> Result<AssembledTurn, PluginError> {
        Ok(self.turn.clone())
    }

    async fn start_process(
        &self,
        _session_id: &str,
        registration: crate::ProcessRegistration,
        descriptor: Option<crate::ProcessHandleDescriptor>,
        _execution_context: crate::ProcessExecutionContext,
    ) -> Result<crate::ProcessRecord, PluginError> {
        let id = registration.id.clone();
        self.process_registry.register_process(registration).await?;
        if let Some(descriptor) = descriptor {
            self.process_registry
                .grant_handle(_session_id, &id, descriptor)
                .await?;
        }
        self.process_registry
            .complete_process(
                &id,
                crate::ProcessAwaitOutput::from_tool_output(crate::ToolCallOutput::success(
                    serde_json::json!({
                        "state": "completed"
                    }),
                )),
            )
            .await
    }

    async fn await_process(
        &self,
        process_id: &str,
    ) -> Result<crate::ProcessAwaitOutput, PluginError> {
        self.process_registry.await_process(process_id).await
    }

    async fn list_process_handles(
        &self,
        session_id: &str,
    ) -> Result<Vec<crate::ProcessHandleGrantEntry>, PluginError> {
        self.process_registry.list_handle_grants(session_id).await
    }

    async fn validate_process_handles_visible(
        &self,
        session_id: &str,
        handle_ids: &[String],
    ) -> Result<(), PluginError> {
        let visible = self
            .process_registry
            .list_handle_grants(session_id)
            .await?
            .into_iter()
            .map(|(grant, _)| grant.process_id)
            .collect::<std::collections::HashSet<_>>();
        if let Some(missing) = handle_ids.iter().find(|id| !visible.contains(*id)) {
            return Err(PluginError::Session(format!(
                "process handle `{missing}` is not live or visible in this session"
            )));
        }
        Ok(())
    }

    async fn transfer_process_handles(
        &self,
        from_session_id: &str,
        to_session_id: &str,
        handle_ids: &[String],
    ) -> Result<(), PluginError> {
        self.process_registry
            .transfer_handle_grants(from_session_id, to_session_id, handle_ids)
            .await
    }

    async fn cancel_unreferenced_process_handles(
        &self,
        session_id: &str,
        keep_handle_ids: &[String],
    ) -> Result<Vec<crate::ProcessRecord>, PluginError> {
        let keep = keep_handle_ids
            .iter()
            .cloned()
            .collect::<std::collections::HashSet<_>>();
        let grants = self.process_registry.list_handle_grants(session_id).await?;
        let mut cancelled = Vec::new();

        for (grant, record) in grants {
            if keep.contains(&grant.process_id) {
                continue;
            }
            self.process_registry
                .revoke_handle(session_id, &grant.process_id)
                .await?;
            if record.is_terminal()
                || !self
                    .process_registry
                    .handle_grants_for_process(&grant.process_id)
                    .await?
                    .is_empty()
            {
                continue;
            }
            cancelled.push(
                crate::InlineRuntimeEffectController::default()
                    .request_process_cancel(
                        self.process_registry.clone(),
                        &grant.process_id,
                        Some("unreferenced by test".to_string()),
                    )
                    .await?,
            );
        }

        Ok(cancelled)
    }

    async fn cancel_process(
        &self,
        _session_id: &str,
        process_id: &str,
    ) -> Result<crate::ProcessRecord, PluginError> {
        crate::InlineRuntimeEffectController::default()
            .request_process_cancel(
                self.process_registry.clone(),
                process_id,
                Some("requested by test".to_string()),
            )
            .await
    }
}
// ─────────────────────────────────────────────────────────────────────
// Minimal in-tree plugin fake advertising support for a given
// `StandardContextApproachKind`. Lash tests use this instead of pulling in
// `lash-plugin-rolling-history` / `lash-plugin-observational-memory`
// as dev-deps, which would create a dev-dep cycle.
// ─────────────────────────────────────────────────────────────────────

use crate::plugin::{PluginFactory, PluginSessionContext, PluginSpec, SessionPlugin};
use crate::standard_context_approach::StandardContextApproachKind;

pub struct FakeStandardContextApproachPluginFactory {
    id: &'static str,
    approaches: &'static [StandardContextApproachKind],
}

impl FakeStandardContextApproachPluginFactory {
    pub fn rolling_history() -> Self {
        Self {
            id: "fake_rolling_history",
            approaches: &[StandardContextApproachKind::RollingHistory],
        }
    }

    pub fn observational_memory() -> Self {
        Self {
            id: "fake_observational_memory",
            approaches: &[StandardContextApproachKind::ObservationalMemory],
        }
    }
}

impl PluginFactory for FakeStandardContextApproachPluginFactory {
    fn id(&self) -> &'static str {
        self.id
    }

    fn supported_standard_context_approaches(&self) -> &'static [StandardContextApproachKind] {
        self.approaches
    }

    fn build(
        &self,
        ctx: &PluginSessionContext,
    ) -> Result<Arc<dyn SessionPlugin>, crate::plugin::PluginError> {
        crate::plugin::StaticPluginFactory::new(self.id, PluginSpec::new()).build(ctx)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Test mode plugin fakes.
//
// Exposed publicly under the `testing` feature so downstream plugin
// crates (e.g. `lash-plugin-plan-mode`) can wire minimal fake mode
// plugins into their integration tests without depending on the real
// `lash-mode-standard` / `lash-mode-rlm` crates (which would create a
// dev-dep cycle through the plugin crates those modes already
// include).
// ─────────────────────────────────────────────────────────────────────
pub use test_mode_fakes::test_mode_factories;

mod test_mode_fakes {
    use std::sync::Arc;

    use async_trait::async_trait;

    use super::*;
    use crate::plugin::{
        ModeProtocolDriverPlugin, ModeRuntimeContext, ModeSessionContext, ModeSessionPlugin,
        PluginFactory, PluginRegistrar, PluginSessionContext, SessionPlugin,
    };
    use crate::sansio::{
        CompletedToolCall, ProtocolDriverHandle, WaitingExecState, WaitingLlmState,
    };
    use crate::{
        DriverAction, DriverContextView, ExecResponse, ModeBuildInput, ModeConfig, ModePreamble,
    };
    use lash_sansio::llm::types::LlmResponse;

    /// Factories that register minimal fake mode plugins for lash's own
    /// unit tests and downstream plugin crate integration tests.
    /// Production callers embed the real `lash-mode-standard` /
    /// `lash-mode-rlm` crates instead.
    pub fn test_mode_factories() -> Vec<Arc<dyn PluginFactory>> {
        vec![
            Arc::new(crate::BuiltinProcessControlsPluginFactory::new()),
            Arc::new(crate::BuiltinMonitorToolPluginFactory::new()),
            Arc::new(TestModeFactory {
                id: "mode_standard",
                mode: ExecutionMode::standard(),
            }),
            Arc::new(TestModeFactory {
                id: "mode_rlm",
                mode: ExecutionMode::new("rlm"),
            }),
        ]
    }

    struct TestModeFactory {
        id: &'static str,
        mode: ExecutionMode,
    }

    impl PluginFactory for TestModeFactory {
        fn id(&self) -> &'static str {
            self.id
        }

        fn build(&self, ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
            Ok(Arc::new(TestModePlugin {
                id: self.id,
                active: ctx.execution_mode == self.mode,
                mode: self.mode.clone(),
            }))
        }
    }

    struct TestModePlugin {
        id: &'static str,
        active: bool,
        mode: ExecutionMode,
    }

    impl SessionPlugin for TestModePlugin {
        fn id(&self) -> &'static str {
            self.id
        }

        fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
            if !self.active {
                return Ok(());
            }
            reg.mode().session(Arc::new(TestModeSession {
                mode: self.mode.clone(),
            }))?;
            if self.mode == ExecutionMode::standard() {
                reg.mode().native_tools(Arc::new(TestModeNativeTools))?;
            }
            reg.mode().protocol_driver(Arc::new(TestProtocolDriver {
                mode: self.mode.clone(),
            }))?;
            Ok(())
        }
    }

    struct TestModeSession {
        mode: ExecutionMode,
    }

    #[async_trait]
    impl ModeSessionPlugin for TestModeSession {
        async fn initialize_session(
            &self,
            _ctx: ModeSessionContext<'_>,
        ) -> Result<(), crate::SessionError> {
            Ok(())
        }

        fn configure_runtime_from_request(
            &self,
            mut ctx: ModeRuntimeContext<'_>,
            request: &crate::SessionCreateRequest,
        ) {
            if self.mode == ExecutionMode::new("rlm")
                && let Ok(Some(termination)) = request
                    .mode_extras
                    .decode::<serde_json::Value>(&ExecutionMode::new("rlm"))
                && let Some(termination) = termination.get("termination").cloned()
                && let Ok(options) =
                    crate::ModeTurnOptions::typed(ExecutionMode::new("rlm"), termination)
            {
                ctx.set_mode_turn_options(options);
            }
        }
    }

    struct TestModeNativeTools;

    #[async_trait]
    impl crate::plugin::ModeNativeToolsPlugin for TestModeNativeTools {
        fn tool_manifests(&self) -> Vec<crate::ToolManifest> {
            vec![test_batch_tool_definition().manifest()]
        }

        fn resolve_contract(&self, name: &str) -> Option<Arc<crate::ToolContract>> {
            (name == "batch").then(|| Arc::new(test_batch_tool_definition().contract()))
        }

        async fn execute(
            &self,
            context: &crate::tool_dispatch::ToolDispatchContext<'_>,
            name: &str,
            args: &serde_json::Value,
            progress: Option<&crate::ProgressSender>,
        ) -> Option<crate::ToolResult> {
            match name {
                "batch" => Some(execute_test_batch(context, args, progress).await),
                _ => None,
            }
        }
    }

    /// Minimal `batch` tool definition used by lash's own tests. Mirrors
    /// the schema from `lash_mode_standard::batch::batch_tool_definition`,
    /// but lives here so lash's tests don't need a dev-dep on
    /// `lash-mode-standard`.
    fn test_batch_tool_definition() -> crate::ToolDefinition {
        crate::ToolDefinition::raw(
            "tool:batch",
            "batch",
            "Execute up to 25 independent tool calls concurrently.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "tool_calls": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 25,
                        "items": {
                            "type": "object",
                            "properties": {
                                "tool": { "type": "string" },
                                "parameters": { "type": "object", "additionalProperties": true }
                            },
                            "required": ["tool", "parameters"],
                            "additionalProperties": false
                        }
                    }
                },
                "required": ["tool_calls"],
                "additionalProperties": false,
            }),
            serde_json::json!({ "type": "object", "additionalProperties": true }),
        )
        .with_discovery(crate::ToolDiscoveryMetadata {
            namespace: Some("runtime".to_string()),
            aliases: vec!["parallel_tools".to_string()],
        })
        .with_execution_mode(crate::ToolExecutionMode::Parallel)
    }

    /// Minimal batch executor used by lash's own tests (mirrors the
    /// behavior of `lash-mode-standard`'s `execute_batch_tool_call`).
    async fn execute_test_batch(
        context: &crate::tool_dispatch::ToolDispatchContext<'_>,
        args: &serde_json::Value,
        progress: Option<&crate::ProgressSender>,
    ) -> crate::ToolResult {
        use crate::tool_dispatch::{ParallelToolCallSpec, dispatch_parallel_tool_calls};

        const MAX: usize = 25;
        let Some(raw_calls) = args.get("tool_calls").and_then(|v| v.as_array()) else {
            return crate::ToolResult::err_fmt("Missing required parameter: tool_calls");
        };
        if raw_calls.is_empty() {
            return crate::ToolResult::err_fmt("Invalid tool_calls: expected at least one call");
        }

        let mut results = Vec::new();
        let mut parallel_specs = Vec::new();
        for (index, item) in raw_calls.iter().enumerate().take(MAX) {
            let Some(obj) = item.as_object() else {
                return crate::ToolResult::err_fmt(format_args!(
                    "Invalid tool_calls[{index}]: expected object with tool and parameters"
                ));
            };
            let Some(tool) = obj
                .get("tool")
                .and_then(|v| v.as_str())
                .map(str::trim)
                .filter(|t| !t.is_empty())
            else {
                return crate::ToolResult::err_fmt(format_args!(
                    "Invalid tool_calls[{index}].tool: expected non-empty string"
                ));
            };
            if tool == "batch" {
                results.push(serde_json::json!({
                    "index": index,
                    "tool": tool,
                    "success": false,
                    "duration_ms": 0,
                    "error": "Tool 'batch' is not allowed inside batch",
                }));
                continue;
            }
            let parameters = obj
                .get("parameters")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({}));
            parallel_specs.push(ParallelToolCallSpec {
                index,
                tool_name: tool.to_string(),
                args: parameters,
            });
        }

        let outcomes =
            dispatch_parallel_tool_calls(Arc::new(context.clone()), parallel_specs, progress).await;
        for outcome in outcomes {
            let mut record = serde_json::Map::new();
            record.insert("index".to_string(), serde_json::json!(outcome.index));
            record.insert("tool".to_string(), serde_json::json!(outcome.record.tool));
            record.insert(
                "success".to_string(),
                serde_json::json!(outcome.record.output.is_success()),
            );
            record.insert(
                "duration_ms".to_string(),
                serde_json::json!(outcome.record.duration_ms),
            );
            record.insert(
                if outcome.record.output.is_success() {
                    "result"
                } else {
                    "error"
                }
                .to_string(),
                outcome.record.output.value_for_projection(),
            );
            results.push(serde_json::Value::Object(record));
        }

        for overflow_index in MAX..raw_calls.len() {
            results.push(serde_json::json!({
                "index": overflow_index,
                "tool": raw_calls
                    .get(overflow_index)
                    .and_then(|item| item.get("tool"))
                    .and_then(|value| value.as_str())
                    .unwrap_or("unknown"),
                "success": false,
                "duration_ms": 0,
                "error": "Maximum of 25 tool calls allowed in batch",
            }));
        }

        results.sort_by_key(|r| {
            r.get("index")
                .and_then(|value| value.as_u64())
                .unwrap_or(u64::MAX)
        });
        crate::ToolResult::ok(serde_json::json!({ "results": results }))
    }

    struct TestProtocolDriver {
        mode: ExecutionMode,
    }

    impl ModeProtocolDriverPlugin for TestProtocolDriver {
        fn mode_id(&self) -> &str {
            self.mode.plugin_id()
        }

        fn build_preamble(&self, input: ModeBuildInput) -> ModePreamble {
            let tool_names = input.tool_surface.tool_names();
            let tool_names_fingerprint = input.tool_surface.tool_names_fingerprint();
            ModePreamble {
                config: ModeConfig::chat(
                    Arc::new(TestDriver),
                    false,
                    Arc::new(test_turn_limit_final_message),
                ),
                tool_specs: input.tool_surface.model_tool_specs(),
                tool_names,
                tool_names_fingerprint,
                omitted_tool_count: 0,
                execution_prompt: Arc::from(""),
                prompt_contributions: input.extra_prompt_contributions,
            }
        }
    }

    fn test_turn_limit_final_message(message_id: String, max_turns: usize) -> crate::Message {
        crate::Message {
            id: message_id.clone(),
            role: crate::MessageRole::System,
            parts: crate::shared_parts(vec![crate::Part {
                id: format!("{message_id}.p0"),
                kind: crate::PartKind::Error,
                content: format!("Turn limit reached ({max_turns}) before a final test response."),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                tool_replay: None,
                prune_state: crate::PruneState::Intact,
                reasoning_meta: None,
                response_meta: None,
            }]),
            origin: None,
        }
    }

    /// Minimal Standard-style driver used by lash's own test suite. Mirrors
    /// the parts of the real `lash-mode-standard::StandardDriver` that
    /// production tests depend on: extract tool calls + assistant text from
    /// the LLM response, append the assistant message, dispatch tools, and
    /// finish-checkpoint when there are no tools. Reasoning parts are
    /// surfaced but without the interleave ordering the real driver uses —
    /// no test asserts that ordering.
    struct TestDriver;

    impl ProtocolDriverHandle<crate::HostModeProtocol> for TestDriver {
        fn prepare_mode_iteration(&self, ctx: DriverContextView<'_>) -> Vec<DriverAction> {
            vec![DriverAction::StartLlm {
                request: ctx.project_llm_request(true),
                driver_state: None,
            }]
        }

        fn handle_llm_success(
            &self,
            ctx: DriverContextView<'_>,
            _waiting: WaitingLlmState<crate::HostModeProtocol>,
            llm_response: LlmResponse,
            text_streamed: bool,
        ) -> Vec<DriverAction> {
            use crate::sansio::{CheckpointResumeAction, PendingToolCall};
            use crate::session_model::fresh_message_id;
            use crate::{
                CheckpointKind, Message, MessageRole, Part, PartKind, PruneState, SessionEvent,
            };
            use lash_sansio::llm::types::LlmOutputPart;
            use lash_sansio::session_model::make_error_event;

            let parts = crate::normalized_response_parts(&llm_response);
            let mut assistant_text = String::new();
            let mut tool_calls: Vec<(
                String,
                String,
                String,
                Option<lash_sansio::llm::types::ProviderReplayMeta>,
            )> = Vec::new();
            let mut actions = Vec::new();

            for part in parts {
                match part {
                    LlmOutputPart::Text { text, .. } => {
                        if !text.is_empty() {
                            let previous_len = assistant_text.len();
                            crate::append_assistant_text_part(&mut assistant_text, &text);
                            if !text_streamed {
                                actions.push(DriverAction::Emit(SessionEvent::TextDelta {
                                    content: assistant_text[previous_len..].to_string(),
                                }));
                            }
                        }
                    }
                    LlmOutputPart::Reasoning { .. } => {}
                    LlmOutputPart::ToolCall {
                        call_id,
                        tool_name,
                        input_json,
                        replay,
                    } => {
                        tool_calls.push((call_id, tool_name, input_json, replay));
                    }
                }
            }

            actions.push(DriverAction::Emit(SessionEvent::LlmResponse {
                mode_iteration: ctx.mode_iteration(),
                content: assistant_text.clone(),
                duration_ms: 0,
            }));

            if tool_calls.is_empty() {
                if assistant_text.trim().is_empty() {
                    actions.push(DriverAction::Emit(make_error_event(
                        "llm_provider",
                        Some("empty_response"),
                        "Model returned no assistant text or tool calls.",
                        None,
                    )));
                    actions.push(DriverAction::Finish(TurnOutcome::Stopped(
                        TurnStop::ProviderError,
                    )));
                    return actions;
                }
                let asst_id = fresh_message_id();
                let outcome_text = assistant_text.clone();
                let parts_out = vec![Part {
                    id: format!("{asst_id}.p0"),
                    kind: PartKind::Prose,
                    content: assistant_text,
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    tool_replay: None,
                    prune_state: PruneState::Intact,
                    reasoning_meta: None,
                    response_meta: None,
                }];
                actions.push(DriverAction::AppendEvents(vec![
                    SessionEventRecord::Conversation(ConversationRecord::from_message(Message {
                        id: asst_id,
                        role: MessageRole::Assistant,
                        parts: lash_sansio::shared_parts(parts_out),
                        origin: None,
                    })),
                ]));
                actions.push(DriverAction::StartCheckpoint {
                    checkpoint: CheckpointKind::BeforeCompletion,
                    on_empty: CheckpointResumeAction::Finish(TurnOutcome::Finished(
                        TurnFinish::AssistantMessage { text: outcome_text },
                    )),
                });
                return actions;
            }

            let asst_id = fresh_message_id();
            let mut assistant_parts = Vec::new();
            if !assistant_text.trim().is_empty() {
                assistant_parts.push(Part {
                    id: format!("{}.p{}", asst_id, assistant_parts.len()),
                    kind: PartKind::Prose,
                    content: assistant_text,
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    tool_replay: None,
                    prune_state: PruneState::Intact,
                    reasoning_meta: None,
                    response_meta: None,
                });
            }
            let mut calls = Vec::new();
            for (call_id, tool_name, input_json, replay) in tool_calls {
                assistant_parts.push(Part {
                    id: format!("{}.p{}", asst_id, assistant_parts.len()),
                    kind: PartKind::ToolCall,
                    content: input_json.clone(),
                    attachment: None,
                    tool_call_id: Some(call_id.clone()),
                    tool_name: Some(tool_name.clone()),
                    tool_replay: replay.clone(),
                    prune_state: PruneState::Intact,
                    reasoning_meta: None,
                    response_meta: None,
                });
                let args = serde_json::from_str::<serde_json::Value>(&input_json)
                    .unwrap_or_else(|_| serde_json::json!({}));
                calls.push(PendingToolCall {
                    call_id,
                    tool_name,
                    args,
                    replay,
                });
            }
            if !assistant_parts.is_empty() {
                actions.push(DriverAction::AppendEvents(vec![
                    SessionEventRecord::Conversation(ConversationRecord::from_message(Message {
                        id: asst_id,
                        role: MessageRole::Assistant,
                        parts: lash_sansio::shared_parts(assistant_parts),
                        origin: None,
                    })),
                ]));
            }
            actions.push(DriverAction::StartTools { calls });
            actions
        }

        fn handle_tool_results(
            &self,
            ctx: DriverContextView<'_>,
            completed: Vec<CompletedToolCall>,
        ) -> Vec<DriverAction> {
            use crate::sansio::CheckpointResumeAction;
            use crate::session_model::fresh_message_id;
            use crate::{
                CheckpointKind, Message, MessageRole, Part, PartKind, PruneState, SessionEvent,
            };
            use lash_sansio::session_model::reassign_part_ids;
            let mut actions = Vec::new();
            let mut result_parts = Vec::new();
            let mut terminal_outcome = None;
            for outcome in completed {
                if terminal_outcome.is_none() && outcome.output.is_success() {
                    terminal_outcome = match outcome.output.control.as_ref() {
                        Some(crate::ToolControl::Handoff { session_id })
                            if !session_id.trim().is_empty() =>
                        {
                            Some(TurnOutcome::Handoff {
                                session_id: session_id.clone(),
                            })
                        }
                        Some(crate::ToolControl::Finish { value }) => {
                            Some(TurnOutcome::Finished(TurnFinish::ToolValue {
                                tool_name: outcome.tool_name.clone(),
                                value: value.to_json_value(),
                            }))
                        }
                        Some(crate::ToolControl::Fail { failure }) => {
                            Some(TurnOutcome::Stopped(TurnStop::ToolError {
                                tool_name: outcome.tool_name.clone(),
                                value: failure.to_json_value(),
                            }))
                        }
                        _ => None,
                    };
                }
                for part in &outcome.model_return.parts {
                    match part {
                        lash_sansio::ModelToolReturnPart::Text(content) => {
                            if content.is_empty() {
                                continue;
                            }
                            result_parts.push(Part {
                                id: String::new(),
                                kind: PartKind::ToolResult,
                                content: content.clone(),
                                attachment: None,
                                tool_call_id: Some(outcome.call_id.clone()),
                                tool_name: Some(outcome.tool_name.clone()),
                                tool_replay: None,
                                prune_state: PruneState::Intact,
                                reasoning_meta: None,
                                response_meta: None,
                            });
                        }
                        lash_sansio::ModelToolReturnPart::Attachment(reference) => {
                            result_parts.push(Part {
                                id: String::new(),
                                kind: PartKind::Image,
                                content: String::new(),
                                attachment: Some(lash_sansio::PartAttachment {
                                    reference: reference.clone(),
                                }),
                                tool_call_id: Some(outcome.call_id.clone()),
                                tool_name: Some(outcome.tool_name.clone()),
                                tool_replay: None,
                                prune_state: PruneState::Intact,
                                reasoning_meta: None,
                                response_meta: None,
                            });
                        }
                    }
                }
            }
            if !result_parts.is_empty() {
                let user_id = fresh_message_id();
                reassign_part_ids(&user_id, &mut result_parts);
                actions.push(DriverAction::AppendEvents(vec![
                    SessionEventRecord::Conversation(ConversationRecord::from_message(Message {
                        id: user_id,
                        role: MessageRole::User,
                        parts: lash_sansio::shared_parts(result_parts),
                        origin: None,
                    })),
                ]));
            }
            if let Some(outcome) = terminal_outcome {
                actions.push(DriverAction::Finish(outcome));
                return actions;
            }
            actions.push(DriverAction::AdvanceModeIteration);
            let next_mode_iteration = ctx.mode_iteration() + 1;
            if let Some(max_turns) = ctx.max_turns()
                && next_mode_iteration >= ctx.mode_run_offset() + max_turns
            {
                let message_id = fresh_message_id();
                actions.push(DriverAction::AppendEvents(vec![
                    SessionEventRecord::Conversation(ConversationRecord::from_message(
                        test_turn_limit_final_message(message_id, max_turns),
                    )),
                ]));
                actions.push(DriverAction::Finish(TurnOutcome::Stopped(
                    TurnStop::MaxTurns,
                )));
                let _ = SessionEvent::Done;
                return actions;
            }
            actions.push(DriverAction::StartCheckpoint {
                checkpoint: CheckpointKind::AfterWork,
                on_empty: CheckpointResumeAction::PrepareIteration,
            });
            actions
        }

        fn handle_exec_result(
            &self,
            _ctx: DriverContextView<'_>,
            _waiting: WaitingExecState<crate::HostModeProtocol>,
            _result: Result<ExecResponse, String>,
        ) -> Vec<DriverAction> {
            Vec::new()
        }
    }
} // mod test_mode_fakes
