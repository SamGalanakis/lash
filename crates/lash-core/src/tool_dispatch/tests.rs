use super::*;
use crate::plugin::{PluginHost, PluginSession, StaticPluginFactory};
use crate::runtime::RuntimeEffectControllerHandle;
use crate::{
    ToolCall, ToolCallOutcome, ToolContext, ToolProvider, ToolResult, ToolRetryDisposition,
    ToolRetryPolicy, ToolScheduling,
};
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use tokio::sync::{Barrier, mpsc};
use tokio::time::{Duration, timeout};

type ExecutionWindow = (&'static str, Instant, Instant);
type SharedExecutionWindows = Arc<std::sync::Mutex<Vec<ExecutionWindow>>>;
type AttemptObservation = (u32, u32, Option<String>);
type SharedAttemptObservations = Arc<std::sync::Mutex<Vec<AttemptObservation>>>;

fn test_tool(name: &str, scheduling: ToolScheduling) -> crate::ToolDefinition {
    crate::ToolDefinition::raw(
        format!("tool:{name}"),
        name,
        "",
        crate::ToolDefinition::default_input_schema(),
        json!({ "type": "string" }),
    )
    .with_scheduling(scheduling)
}

fn beta_tool() -> crate::ToolDefinition {
    crate::ToolDefinition::raw(
        "tool:beta",
        "beta",
        "",
        json!({
            "type": "object",
            "properties": {
                "value": { "type": "string" }
            },
            "required": ["value"],
            "additionalProperties": false
        }),
        json!({ "type": "string" }),
    )
    .with_scheduling(ToolScheduling::Parallel)
}

fn named_beta_tool(name: &str) -> crate::ToolDefinition {
    crate::ToolDefinition::raw(
        format!("tool:{name}"),
        name,
        "",
        json!({
            "type": "object",
            "properties": {
                "value": { "type": "string" }
            },
            "required": ["value"],
            "additionalProperties": false
        }),
        json!({ "type": "string" }),
    )
    .with_scheduling(ToolScheduling::Parallel)
}

fn manifests(definitions: Vec<crate::ToolDefinition>) -> Vec<crate::ToolManifest> {
    definitions
        .into_iter()
        .map(|tool| tool.manifest())
        .collect()
}

fn contract_from(
    definitions: Vec<crate::ToolDefinition>,
    name: &str,
) -> Option<Arc<crate::ToolContract>> {
    definitions
        .into_iter()
        .find(|tool| tool.name() == name)
        .map(|tool| Arc::new(tool.contract()))
}

#[derive(Clone)]
struct ScheduledProbe {
    index: usize,
    name: &'static str,
    scheduling: ToolScheduling,
    delay: Duration,
}

#[tokio::test]
async fn scheduler_runs_parallel_bucket_then_serial_and_preserves_order() {
    let windows: SharedExecutionWindows = Arc::new(std::sync::Mutex::new(Vec::new()));
    let probes = vec![
        ScheduledProbe {
            index: 0,
            name: "parallel_slow",
            scheduling: ToolScheduling::Parallel,
            delay: Duration::from_millis(40),
        },
        ScheduledProbe {
            index: 1,
            name: "serial",
            scheduling: ToolScheduling::Serial,
            delay: Duration::from_millis(10),
        },
        ScheduledProbe {
            index: 2,
            name: "parallel_fast",
            scheduling: ToolScheduling::Parallel,
            delay: Duration::from_millis(5),
        },
    ];

    let outputs = schedule_tool_batch(probes, |probe| probe.index, |probe| probe.scheduling, {
        let windows = Arc::clone(&windows);
        move |probe| {
            let windows = Arc::clone(&windows);
            async move {
                let start = Instant::now();
                tokio::time::sleep(probe.delay).await;
                let end = Instant::now();
                windows
                    .lock()
                    .expect("windows")
                    .push((probe.name, start, end));
                probe.name
            }
        }
    })
    .await;

    assert_eq!(outputs, ["parallel_slow", "serial", "parallel_fast"]);

    let recorded = windows.lock().expect("windows").clone();
    let parallel_slow = recorded
        .iter()
        .find(|(name, _, _)| *name == "parallel_slow")
        .expect("parallel_slow");
    let parallel_fast = recorded
        .iter()
        .find(|(name, _, _)| *name == "parallel_fast")
        .expect("parallel_fast");
    let serial = recorded
        .iter()
        .find(|(name, _, _)| *name == "serial")
        .expect("serial");

    assert!(
        parallel_fast.1 < parallel_slow.2,
        "parallel tools should overlap even when completion order differs"
    );
    assert!(
        serial.1 >= parallel_slow.2 && serial.1 >= parallel_fast.2,
        "serial tool should start after the parallel bucket completes"
    );
}

struct MockTools;

#[async_trait::async_trait]
impl ToolProvider for MockTools {
    fn tool_manifests(&self) -> Vec<crate::ToolManifest> {
        manifests(vec![
            test_tool("alpha", ToolScheduling::Parallel),
            beta_tool(),
        ])
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<crate::ToolContract>> {
        contract_from(
            vec![test_tool("alpha", ToolScheduling::Parallel), beta_tool()],
            name,
        )
    }

    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        match call.name {
            "alpha" => ToolResult::ok(json!("alpha")),
            "beta" => {
                if call.args.get("value").and_then(|value| value.as_str()) == Some("fail") {
                    ToolResult::err_fmt("beta failed")
                } else {
                    ToolResult::ok(json!(
                        call.args.get("value").cloned().unwrap_or(json!(null))
                    ))
                }
            }
            other => ToolResult::err_fmt(format!("Unknown tool: {other}")),
        }
    }
}

struct ParallelProbeTools {
    barrier: Arc<Barrier>,
    started: Arc<AtomicUsize>,
}

#[derive(Clone, Copy)]
enum PendingProbeMode {
    MissingKey,
    PendingWithKey,
    Done,
}

#[derive(Clone)]
struct PendingProbeTools {
    definition: crate::ToolDefinition,
    attempts: Arc<AtomicUsize>,
    mode: PendingProbeMode,
}

#[async_trait::async_trait]
impl ToolProvider for PendingProbeTools {
    fn tool_manifests(&self) -> Vec<crate::ToolManifest> {
        manifests(vec![self.definition.clone()])
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<crate::ToolContract>> {
        (name == self.definition.name()).then(|| Arc::new(self.definition.contract()))
    }

    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        self.attempts.fetch_add(1, Ordering::SeqCst);
        match self.mode {
            PendingProbeMode::MissingKey => ToolResult::pending(crate::PendingCompletion::new()),
            PendingProbeMode::PendingWithKey => {
                call.context.completion_key().await.expect("completion key");
                ToolResult::pending(crate::PendingCompletion::new())
            }
            PendingProbeMode::Done => ToolResult::ok(json!({ "done": true })),
        }
    }
}

#[async_trait::async_trait]
impl ToolProvider for ParallelProbeTools {
    fn tool_manifests(&self) -> Vec<crate::ToolManifest> {
        manifests(vec![
            test_tool("probe_a", ToolScheduling::Parallel),
            test_tool("probe_b", ToolScheduling::Parallel),
        ])
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<crate::ToolContract>> {
        contract_from(
            vec![
                test_tool("probe_a", ToolScheduling::Parallel),
                test_tool("probe_b", ToolScheduling::Parallel),
            ],
            name,
        )
    }

    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        self.started.fetch_add(1, Ordering::SeqCst);
        let waited = timeout(Duration::from_millis(100), self.barrier.wait()).await;
        match waited {
            Ok(_) => ToolResult::ok(json!(call.name)),
            Err(_) => ToolResult::err_fmt(format!("{} did not overlap with peer", call.name)),
        }
    }
}

struct StrictMcpTools {
    executed: Arc<AtomicUsize>,
}

#[async_trait::async_trait]
impl ToolProvider for StrictMcpTools {
    fn tool_manifests(&self) -> Vec<crate::ToolManifest> {
        manifests(vec![strict_mcp_tool_definition()])
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<crate::ToolContract>> {
        (name == "mcp__appworld__venmo_show_transactions")
            .then(|| Arc::new(strict_mcp_tool_definition().contract()))
    }

    async fn execute(&self, _call: ToolCall<'_>) -> ToolResult {
        self.executed.fetch_add(1, Ordering::SeqCst);
        ToolResult::ok(json!({ "executed": true }))
    }
}

fn strict_mcp_tool_definition() -> crate::ToolDefinition {
    crate::ToolDefinition::raw(
        "tool:mcp__appworld__venmo_show_transactions",
        "mcp__appworld__venmo_show_transactions",
        "Show Venmo transactions",
        json!({
            "type": "object",
            "properties": {
                "min_created_at": { "type": "string" },
                "max_created_at": { "type": "string" },
                "limit": { "type": "integer", "maximum": 100 }
            },
            "required": ["limit"]
        }),
        json!({ "type": "object", "additionalProperties": true }),
    )
}

struct ProjectionPolicyTools;

#[async_trait::async_trait]
impl ToolProvider for ProjectionPolicyTools {
    fn tool_manifests(&self) -> Vec<crate::ToolManifest> {
        manifests(vec![projection_policy_tool_definition()])
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<crate::ToolContract>> {
        (name == "seedy").then(|| Arc::new(projection_policy_tool_definition().contract()))
    }

    async fn execute(&self, _call: ToolCall<'_>) -> ToolResult {
        ToolResult::ok(json!("ok"))
    }
}

fn projection_policy_tool_definition() -> crate::ToolDefinition {
    crate::ToolDefinition::raw(
        "tool:seedy",
        "seedy",
        "Seed-aware",
        crate::ToolDefinition::default_input_schema(),
        json!({ "type": "string" }),
    )
    .with_argument_projection(
        crate::ToolArgumentProjectionPolicy::preserve_projected_refs_in_field("seed"),
    )
}

fn strict_mcp_dispatch_context(executed: Arc<AtomicUsize>) -> ToolDispatchContext<'static> {
    let (event_tx, _event_rx) = mpsc::channel(8);
    let plugins = test_plugins(Arc::new(StrictMcpTools { executed }));
    let tools = plugins.tools();
    let tool_catalog = plugins
        .resolved_tool_catalog("session")
        .expect("tool catalog");
    ToolDispatchContext {
        plugins,
        tools,
        tool_catalog,
        sessions: Arc::new(MockSessionManager::default()),
        session_lifecycle: Arc::new(MockSessionManager::default()),
        session_graph: Arc::new(MockSessionManager::default()),
        processes: Arc::new(crate::UnavailableProcessService),
        process_cancel_ability: Arc::new(crate::DefaultProcessCancelAbility),
        trigger_router: None,
        effect_controller: RuntimeEffectControllerHandle::shared(Arc::new(
            crate::InlineRuntimeEffectController,
        )),
        direct_completions: crate::DirectCompletionClient::unavailable(
            "direct completions are unavailable in this test context",
        ),
        parent_invocation: None,
        execution_env_spec: crate::ProcessExecutionEnvSpec::new(
            crate::PluginOptions::default(),
            crate::SessionPolicy::default(),
        ),
        session_id: "session".to_string(),
        agent_frame_id: String::new(),
        event_tx,
        checkpoint_messages: crate::tool_dispatch::CheckpointMessageBuffer::default(),
        trigger_outcomes: crate::tool_dispatch::ToolTriggerOutcomeBuffer::default(),
        attachment_store: Arc::new(crate::InMemoryAttachmentStore::new()),
        turn_context: crate::TurnContext::default(),
        clock: std::sync::Arc::new(crate::SystemClock),
    }
}

fn test_plugins(provider: Arc<dyn ToolProvider>) -> Arc<PluginSession> {
    PluginHost::new(vec![Arc::new(StaticPluginFactory::new(
        "test_tools",
        crate::PluginSpec::new().with_tool_provider(Arc::clone(&provider)),
    ))])
    .build_session("root", None)
    .expect("plugin session")
}

use crate::testing::MockSessionManager;

fn dispatch_context() -> ToolDispatchContext<'static> {
    let (event_tx, _event_rx) = mpsc::channel(8);
    let plugins = test_plugins(Arc::new(MockTools));
    let tools = plugins.tools();
    let tool_catalog = plugins
        .resolved_tool_catalog("session")
        .expect("tool catalog");
    ToolDispatchContext {
        plugins,
        tools,
        tool_catalog,
        sessions: Arc::new(MockSessionManager::default()),
        session_lifecycle: Arc::new(MockSessionManager::default()),
        session_graph: Arc::new(MockSessionManager::default()),
        processes: Arc::new(crate::UnavailableProcessService),
        process_cancel_ability: Arc::new(crate::DefaultProcessCancelAbility),
        trigger_router: None,
        effect_controller: RuntimeEffectControllerHandle::shared(Arc::new(
            crate::InlineRuntimeEffectController,
        )),
        direct_completions: crate::DirectCompletionClient::unavailable(
            "direct completions are unavailable in this test context",
        ),
        parent_invocation: None,
        execution_env_spec: crate::ProcessExecutionEnvSpec::new(
            crate::PluginOptions::default(),
            crate::SessionPolicy::default(),
        ),
        session_id: "session".to_string(),
        agent_frame_id: String::new(),
        event_tx,
        checkpoint_messages: crate::tool_dispatch::CheckpointMessageBuffer::default(),
        trigger_outcomes: crate::tool_dispatch::ToolTriggerOutcomeBuffer::default(),
        attachment_store: Arc::new(crate::InMemoryAttachmentStore::new()),
        turn_context: crate::TurnContext::default(),
        clock: std::sync::Arc::new(crate::SystemClock),
    }
}

fn projection_policy_dispatch_context(
    captured: Arc<std::sync::Mutex<Option<crate::ToolArgumentProjectionPolicy>>>,
) -> ToolDispatchContext<'static> {
    let (event_tx, _event_rx) = mpsc::channel(8);
    let provider: Arc<dyn ToolProvider> = Arc::new(ProjectionPolicyTools);
    let hook_captured = Arc::clone(&captured);
    let hook: crate::plugin::BeforeToolCallHook = Arc::new(move |ctx| {
        let hook_captured = Arc::clone(&hook_captured);
        Box::pin(async move {
            *hook_captured.lock().expect("captured policy") = Some(ctx.argument_projection.clone());
            Ok(Vec::new())
        })
    });
    let plugins = PluginHost::new(vec![Arc::new(StaticPluginFactory::new(
        "projection_policy_tools",
        crate::PluginSpec::new()
            .with_tool_provider(Arc::clone(&provider))
            .with_before_tool_call(hook),
    ))])
    .build_session("root", None)
    .expect("plugin session");
    let tools = plugins.tools();
    let tool_catalog = plugins
        .resolved_tool_catalog("session")
        .expect("tool catalog");
    ToolDispatchContext {
        plugins,
        tools,
        tool_catalog,
        sessions: Arc::new(MockSessionManager::default()),
        session_lifecycle: Arc::new(MockSessionManager::default()),
        session_graph: Arc::new(MockSessionManager::default()),
        processes: Arc::new(crate::UnavailableProcessService),
        process_cancel_ability: Arc::new(crate::DefaultProcessCancelAbility),
        trigger_router: None,
        effect_controller: RuntimeEffectControllerHandle::shared(Arc::new(
            crate::InlineRuntimeEffectController,
        )),
        direct_completions: crate::DirectCompletionClient::unavailable(
            "direct completions are unavailable in this test context",
        ),
        parent_invocation: None,
        execution_env_spec: crate::ProcessExecutionEnvSpec::new(
            crate::PluginOptions::default(),
            crate::SessionPolicy::default(),
        ),
        session_id: "session".to_string(),
        agent_frame_id: String::new(),
        event_tx,
        checkpoint_messages: crate::tool_dispatch::CheckpointMessageBuffer::default(),
        trigger_outcomes: crate::tool_dispatch::ToolTriggerOutcomeBuffer::default(),
        attachment_store: Arc::new(crate::InMemoryAttachmentStore::new()),
        turn_context: crate::TurnContext::default(),
        clock: std::sync::Arc::new(crate::SystemClock),
    }
}

struct CountingContractTools {
    contracts_resolved: Arc<AtomicUsize>,
    executed: Arc<AtomicUsize>,
}

struct ExactDispatchTools {
    contracts_resolved: Arc<AtomicUsize>,
    executed: Arc<AtomicUsize>,
    contract_available: bool,
    observed_execution_bindings: Option<Arc<std::sync::Mutex<Vec<serde_json::Value>>>>,
}

struct HiddenDispatchTools {
    contracts_resolved: Arc<AtomicUsize>,
    executed: Arc<AtomicUsize>,
}

struct RetryProbeTools {
    definition: crate::ToolDefinition,
    attempts: Arc<AtomicUsize>,
    successes_after: usize,
    cancel_on_first: bool,
    observed_attempts: SharedAttemptObservations,
    retry_after_ms: Option<u64>,
}

#[async_trait::async_trait]
impl ToolProvider for CountingContractTools {
    fn tool_manifests(&self) -> Vec<crate::ToolManifest> {
        manifests(vec![beta_tool()])
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<crate::ToolContract>> {
        self.contracts_resolved.fetch_add(1, Ordering::SeqCst);
        (name == "beta").then(|| Arc::new(beta_tool().contract()))
    }

    async fn execute(&self, _call: ToolCall<'_>) -> ToolResult {
        self.executed.fetch_add(1, Ordering::SeqCst);
        ToolResult::ok(json!("ok"))
    }
}

#[async_trait::async_trait]
impl ToolProvider for ExactDispatchTools {
    fn tool_manifests(&self) -> Vec<crate::ToolManifest> {
        Vec::new()
    }

    fn resolve_manifest(&self, name: &str) -> Option<crate::ToolManifest> {
        (name == "host_only").then(|| named_beta_tool("host_only").manifest())
    }

    fn resolve_manifest_by_id(&self, id: &crate::ToolId) -> Option<crate::ToolManifest> {
        (id == &crate::ToolId::from("tool:host_only"))
            .then(|| named_beta_tool("host_only").manifest())
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<crate::ToolContract>> {
        self.contracts_resolved.fetch_add(1, Ordering::SeqCst);
        (self.contract_available && name == "host_only")
            .then(|| Arc::new(named_beta_tool("host_only").contract()))
    }

    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        self.executed.fetch_add(1, Ordering::SeqCst);
        if let Some(bindings) = &self.observed_execution_bindings {
            bindings
                .lock()
                .expect("execution bindings")
                .push(call.context.tool_execution_binding().clone());
        }
        ToolResult::ok(json!("host"))
    }

    async fn prepare_granted_tool_call(
        &self,
        _grant: &crate::ToolExecutionGrant,
        call: crate::ToolPrepareCall<'_>,
    ) -> Result<crate::PreparedToolCall, ToolResult> {
        Ok(crate::PreparedToolCall::identity(
            call.tool_id,
            call.pending,
        ))
    }

    async fn execute_granted(
        &self,
        grant: &crate::ToolExecutionGrant,
        args: &serde_json::Value,
        context: &crate::ToolContext<'_>,
        progress: Option<&crate::ProgressSender>,
    ) -> ToolResult {
        self.execute_by_id(&grant.manifest.id, args, context, progress)
            .await
    }
}

#[async_trait::async_trait]
impl ToolProvider for HiddenDispatchTools {
    fn tool_manifests(&self) -> Vec<crate::ToolManifest> {
        manifests(vec![named_beta_tool("hidden")])
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<crate::ToolContract>> {
        self.contracts_resolved.fetch_add(1, Ordering::SeqCst);
        (name == "hidden").then(|| Arc::new(named_beta_tool("hidden").contract()))
    }

    async fn execute(&self, _call: ToolCall<'_>) -> ToolResult {
        self.executed.fetch_add(1, Ordering::SeqCst);
        ToolResult::ok(json!("hidden"))
    }
}

#[async_trait::async_trait]
impl ToolProvider for RetryProbeTools {
    fn tool_manifests(&self) -> Vec<crate::ToolManifest> {
        manifests(vec![self.definition.clone()])
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<crate::ToolContract>> {
        (name == self.definition.name()).then(|| Arc::new(self.definition.contract()))
    }

    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        self.observed_attempts.lock().expect("attempts").push((
            call.context.attempt_number(),
            call.context.max_attempts(),
            call.context.replay_key().map(str::to_string),
        ));
        let attempt_index = self.attempts.fetch_add(1, Ordering::SeqCst) + 1;
        if self.cancel_on_first {
            return ToolResult::cancelled("cancelled");
        }
        if attempt_index >= self.successes_after {
            return ToolResult::ok(json!({ "attempt": attempt_index }));
        }
        ToolResult::retryable_failure(
            crate::ToolFailureClass::External,
            "transient",
            "transient failure",
            self.retry_after_ms,
        )
    }
}

fn lazy_contract_dispatch_context(
    contracts_resolved: Arc<AtomicUsize>,
    executed: Arc<AtomicUsize>,
) -> ToolDispatchContext<'static> {
    let (event_tx, _event_rx) = mpsc::channel(8);
    let provider: Arc<dyn ToolProvider> = Arc::new(CountingContractTools {
        contracts_resolved,
        executed,
    });
    let tools = Arc::clone(&provider);
    let tool_catalog = Arc::new(crate::ToolCatalog::from_tools(
        provider.tool_manifests(),
        BTreeMap::new(),
    ));
    ToolDispatchContext {
        plugins: test_plugins(provider),
        tools,
        tool_catalog,
        sessions: Arc::new(MockSessionManager::default()),
        session_lifecycle: Arc::new(MockSessionManager::default()),
        session_graph: Arc::new(MockSessionManager::default()),
        processes: Arc::new(crate::UnavailableProcessService),
        process_cancel_ability: Arc::new(crate::DefaultProcessCancelAbility),
        trigger_router: None,
        effect_controller: RuntimeEffectControllerHandle::shared(Arc::new(
            crate::InlineRuntimeEffectController,
        )),
        direct_completions: crate::DirectCompletionClient::unavailable(
            "direct completions are unavailable in this test context",
        ),
        parent_invocation: None,
        execution_env_spec: crate::ProcessExecutionEnvSpec::new(
            crate::PluginOptions::default(),
            crate::SessionPolicy::default(),
        ),
        session_id: "session".to_string(),
        agent_frame_id: String::new(),
        event_tx,
        checkpoint_messages: crate::tool_dispatch::CheckpointMessageBuffer::default(),
        trigger_outcomes: crate::tool_dispatch::ToolTriggerOutcomeBuffer::default(),
        attachment_store: Arc::new(crate::InMemoryAttachmentStore::new()),
        turn_context: crate::TurnContext::default(),
        clock: std::sync::Arc::new(crate::SystemClock),
    }
}

/// Build a dispatch context where the provider's tool is authority-hidden,
/// so it is removed from the Tool Catalog (non-membership) and rejected before
/// contract resolution.
fn hidden_member_dispatch_context(provider: Arc<dyn ToolProvider>) -> ToolDispatchContext<'static> {
    let (event_tx, _event_rx) = mpsc::channel(8);
    let mut tool_access = crate::SessionToolAccess::default();
    tool_access.hidden_tools.insert("hidden".to_string());
    let plugins = PluginHost::new(vec![Arc::new(StaticPluginFactory::new(
        "test_tools",
        crate::PluginSpec::new().with_tool_provider(Arc::clone(&provider)),
    ))])
    .build_session_with_parent(
        "root",
        None,
        None,
        crate::plugin::SessionAuthorityContext {
            tool_access,
            ..Default::default()
        },
    )
    .expect("plugin session");
    let tools = plugins.tools();
    let tool_catalog = plugins
        .resolved_tool_catalog("session")
        .expect("tool catalog");
    ToolDispatchContext {
        plugins,
        tools,
        tool_catalog,
        sessions: Arc::new(MockSessionManager::default()),
        session_lifecycle: Arc::new(MockSessionManager::default()),
        session_graph: Arc::new(MockSessionManager::default()),
        processes: Arc::new(crate::UnavailableProcessService),
        process_cancel_ability: Arc::new(crate::DefaultProcessCancelAbility),
        trigger_router: None,
        effect_controller: RuntimeEffectControllerHandle::shared(Arc::new(
            crate::InlineRuntimeEffectController,
        )),
        direct_completions: crate::DirectCompletionClient::unavailable(
            "direct completions are unavailable in this test context",
        ),
        parent_invocation: None,
        execution_env_spec: crate::ProcessExecutionEnvSpec::new(
            crate::PluginOptions::default(),
            crate::SessionPolicy::default(),
        ),
        session_id: "session".to_string(),
        agent_frame_id: String::new(),
        event_tx,
        checkpoint_messages: crate::tool_dispatch::CheckpointMessageBuffer::default(),
        trigger_outcomes: crate::tool_dispatch::ToolTriggerOutcomeBuffer::default(),
        attachment_store: Arc::new(crate::InMemoryAttachmentStore::new()),
        turn_context: crate::TurnContext::default(),
        clock: std::sync::Arc::new(crate::SystemClock),
    }
}

fn exact_dispatch_context(provider: Arc<dyn ToolProvider>) -> ToolDispatchContext<'static> {
    let (event_tx, _event_rx) = mpsc::channel(8);
    let plugins = test_plugins(Arc::clone(&provider));
    let tools = plugins.tools();
    let tool_catalog = plugins
        .resolved_tool_catalog("session")
        .expect("tool catalog");
    ToolDispatchContext {
        plugins,
        tools,
        tool_catalog,
        sessions: Arc::new(MockSessionManager::default()),
        session_lifecycle: Arc::new(MockSessionManager::default()),
        session_graph: Arc::new(MockSessionManager::default()),
        processes: Arc::new(crate::UnavailableProcessService),
        process_cancel_ability: Arc::new(crate::DefaultProcessCancelAbility),
        trigger_router: None,
        effect_controller: RuntimeEffectControllerHandle::shared(Arc::new(
            crate::InlineRuntimeEffectController,
        )),
        direct_completions: crate::DirectCompletionClient::unavailable(
            "direct completions are unavailable in this test context",
        ),
        parent_invocation: None,
        execution_env_spec: crate::ProcessExecutionEnvSpec::new(
            crate::PluginOptions::default(),
            crate::SessionPolicy::default(),
        ),
        session_id: "session".to_string(),
        agent_frame_id: String::new(),
        event_tx,
        checkpoint_messages: crate::tool_dispatch::CheckpointMessageBuffer::default(),
        trigger_outcomes: crate::tool_dispatch::ToolTriggerOutcomeBuffer::default(),
        attachment_store: Arc::new(crate::InMemoryAttachmentStore::new()),
        turn_context: crate::TurnContext::default(),
        clock: std::sync::Arc::new(crate::SystemClock),
    }
}

fn retry_tool(name: &str, retry_policy: ToolRetryPolicy) -> crate::ToolDefinition {
    named_beta_tool(name)
        .with_scheduling(ToolScheduling::Parallel)
        .with_retry_policy(retry_policy)
}

fn retry_dispatch_context(
    retry_policy: ToolRetryPolicy,
    attempts: Arc<AtomicUsize>,
    successes_after: usize,
    cancel_on_first: bool,
    observed_attempts: SharedAttemptObservations,
) -> ToolDispatchContext<'static> {
    exact_dispatch_context(Arc::new(RetryProbeTools {
        definition: retry_tool("retry_probe", retry_policy),
        attempts,
        successes_after,
        cancel_on_first,
        observed_attempts,
        retry_after_ms: Some(0),
    }))
}

fn pending_probe_tool(retry_policy: ToolRetryPolicy) -> crate::ToolDefinition {
    named_beta_tool("pending_probe")
        .with_scheduling(ToolScheduling::Parallel)
        .with_retry_policy(retry_policy)
}

fn pending_dispatch_context(
    mode: PendingProbeMode,
    attempts: Arc<AtomicUsize>,
    after_calls: Option<Arc<AtomicUsize>>,
    retry_policy: ToolRetryPolicy,
) -> ToolDispatchContext<'static> {
    let (event_tx, _event_rx) = mpsc::channel(8);
    let provider: Arc<dyn ToolProvider> = Arc::new(PendingProbeTools {
        definition: pending_probe_tool(retry_policy),
        attempts,
        mode,
    });
    let mut spec = crate::PluginSpec::new().with_tool_provider(Arc::clone(&provider));
    if let Some(after_calls) = after_calls {
        let hook: crate::plugin::AfterToolCallHook = Arc::new(move |_ctx| {
            let after_calls = Arc::clone(&after_calls);
            Box::pin(async move {
                after_calls.fetch_add(1, Ordering::SeqCst);
                Ok(Vec::new())
            })
        });
        spec = spec.with_after_tool_call(hook);
    }
    let plugins = PluginHost::new(vec![Arc::new(StaticPluginFactory::new(
        "pending_probe_tools",
        spec,
    ))])
    .build_session("root", None)
    .expect("plugin session");
    let tools = plugins.tools();
    let tool_catalog = plugins
        .resolved_tool_catalog("session")
        .expect("tool catalog");
    ToolDispatchContext {
        plugins,
        tools,
        tool_catalog,
        sessions: Arc::new(MockSessionManager::default()),
        session_lifecycle: Arc::new(MockSessionManager::default()),
        session_graph: Arc::new(MockSessionManager::default()),
        processes: Arc::new(crate::UnavailableProcessService),
        process_cancel_ability: Arc::new(crate::DefaultProcessCancelAbility),
        trigger_router: None,
        effect_controller: RuntimeEffectControllerHandle::shared(Arc::new(
            crate::InlineRuntimeEffectController,
        )),
        direct_completions: crate::DirectCompletionClient::unavailable(
            "direct completions are unavailable in this test context",
        ),
        parent_invocation: None,
        execution_env_spec: crate::ProcessExecutionEnvSpec::new(
            crate::PluginOptions::default(),
            crate::SessionPolicy::default(),
        ),
        session_id: "session".to_string(),
        agent_frame_id: String::new(),
        event_tx,
        checkpoint_messages: crate::tool_dispatch::CheckpointMessageBuffer::default(),
        trigger_outcomes: crate::tool_dispatch::ToolTriggerOutcomeBuffer::default(),
        attachment_store: Arc::new(crate::InMemoryAttachmentStore::new()),
        turn_context: crate::TurnContext::default(),
        clock: std::sync::Arc::new(crate::SystemClock),
    }
}

fn pending_prepared_call() -> crate::PreparedToolCall {
    crate::PreparedToolCall::from_parts(
        "pending-call",
        "tool:pending_probe",
        "pending_probe",
        json!({ "value": "runtime perf benchmark ok" }),
        None,
        serde_json::Value::Null,
    )
}

fn tool_context_for_prepared<'run>(
    context: &ToolDispatchContext<'run>,
    prepared: &crate::PreparedToolCall,
) -> ToolContext<'run> {
    ToolContext::from_dispatch(Arc::new(context.clone()))
        .prepared_call(prepared)
        .build()
}

fn parallel_dispatch_context(
    barrier: Arc<Barrier>,
    started: Arc<AtomicUsize>,
) -> ToolDispatchContext<'static> {
    let (event_tx, _event_rx) = mpsc::channel(8);
    let plugins = test_plugins(Arc::new(ParallelProbeTools { barrier, started }));
    let tools = plugins.tools();
    let tool_catalog = plugins
        .resolved_tool_catalog("session")
        .expect("tool catalog");
    ToolDispatchContext {
        plugins,
        tools,
        tool_catalog,
        sessions: Arc::new(MockSessionManager::default()),
        session_lifecycle: Arc::new(MockSessionManager::default()),
        session_graph: Arc::new(MockSessionManager::default()),
        processes: Arc::new(crate::UnavailableProcessService),
        process_cancel_ability: Arc::new(crate::DefaultProcessCancelAbility),
        trigger_router: None,
        effect_controller: RuntimeEffectControllerHandle::shared(Arc::new(
            crate::InlineRuntimeEffectController,
        )),
        direct_completions: crate::DirectCompletionClient::unavailable(
            "direct completions are unavailable in this test context",
        ),
        parent_invocation: None,
        execution_env_spec: crate::ProcessExecutionEnvSpec::new(
            crate::PluginOptions::default(),
            crate::SessionPolicy::default(),
        ),
        session_id: "session".to_string(),
        agent_frame_id: String::new(),
        event_tx,
        checkpoint_messages: crate::tool_dispatch::CheckpointMessageBuffer::default(),
        trigger_outcomes: crate::tool_dispatch::ToolTriggerOutcomeBuffer::default(),
        attachment_store: Arc::new(crate::InMemoryAttachmentStore::new()),
        turn_context: crate::TurnContext::default(),
        clock: std::sync::Arc::new(crate::SystemClock),
    }
}

#[tokio::test]
async fn dispatch_rejects_invalid_args_before_provider_execution() {
    let outcome =
        dispatch_tool_call(&dispatch_context(), "beta".to_string(), json!({}), None).await;

    assert!(!outcome.record.output.is_success());
    assert_eq!(
        outcome.record.output.value_for_projection()["message"],
        json!("value: required property missing")
    );
}

#[tokio::test]
async fn dispatch_resolves_contract_only_for_called_tool_before_execution() {
    let contracts_resolved = Arc::new(AtomicUsize::new(0));
    let executed = Arc::new(AtomicUsize::new(0));
    let outcome = dispatch_tool_call(
        &lazy_contract_dispatch_context(Arc::clone(&contracts_resolved), Arc::clone(&executed)),
        "beta".to_string(),
        json!({ "value": "ok" }),
        None,
    )
    .await;

    assert!(outcome.record.output.is_success());
    assert_eq!(contracts_resolved.load(Ordering::SeqCst), 1);
    assert_eq!(executed.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn pending_tool_without_completion_key_is_runtime_failure() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let context = pending_dispatch_context(
        PendingProbeMode::MissingKey,
        Arc::clone(&attempts),
        None,
        ToolRetryPolicy::Never,
    );
    let prepared = pending_prepared_call();
    let tool_context = tool_context_for_prepared(&context, &prepared);

    let launch = dispatch_prepared_tool_call_launch_with_execution_context(
        &context,
        prepared,
        None,
        tool_context,
    )
    .await;

    let ToolCallLaunch::Done(outcome) = launch else {
        panic!("missing completion key must fail launch synchronously");
    };
    assert_eq!(attempts.load(Ordering::SeqCst), 1);
    let ToolCallOutcome::Failure(failure) = &outcome.record.output.outcome else {
        panic!("expected failure output");
    };
    assert_eq!(failure.code, "pending_tool_missing_completion_key");
}

#[tokio::test]
async fn retry_policy_stops_after_pending_launch() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let context = pending_dispatch_context(
        PendingProbeMode::PendingWithKey,
        Arc::clone(&attempts),
        None,
        ToolRetryPolicy::safe(5, 0, 0),
    );
    let prepared = pending_prepared_call();
    let tool_context = tool_context_for_prepared(&context, &prepared);

    let launch = dispatch_prepared_tool_call_launch_with_execution_context(
        &context,
        prepared,
        None,
        tool_context,
    )
    .await;

    let ToolCallLaunch::Pending(pending) = launch else {
        panic!("tool should launch pending");
    };
    assert_eq!(attempts.load(Ordering::SeqCst), 1);
    assert_eq!(pending.tool_name, "pending_probe");
    assert_eq!(
        pending.key.wait,
        crate::AwaitEventWaitIdentity::tool_completion("pending-call")
    );
}

#[tokio::test]
async fn after_tool_hook_runs_only_for_completed_tool_results() {
    let after_calls = Arc::new(AtomicUsize::new(0));
    let pending_attempts = Arc::new(AtomicUsize::new(0));
    let pending_context = pending_dispatch_context(
        PendingProbeMode::PendingWithKey,
        pending_attempts,
        Some(Arc::clone(&after_calls)),
        ToolRetryPolicy::Never,
    );
    let prepared = pending_prepared_call();
    let tool_context = tool_context_for_prepared(&pending_context, &prepared);

    let launch = dispatch_prepared_tool_call_launch_with_execution_context(
        &pending_context,
        prepared,
        None,
        tool_context,
    )
    .await;

    assert!(matches!(launch, ToolCallLaunch::Pending(_)));
    assert_eq!(
        after_calls.load(Ordering::SeqCst),
        0,
        "launch-time Pending is not a completed tool result"
    );

    let done_attempts = Arc::new(AtomicUsize::new(0));
    let done_context = pending_dispatch_context(
        PendingProbeMode::Done,
        done_attempts,
        Some(Arc::clone(&after_calls)),
        ToolRetryPolicy::Never,
    );
    let prepared = pending_prepared_call();
    let tool_context = tool_context_for_prepared(&done_context, &prepared);

    let launch = dispatch_prepared_tool_call_launch_with_execution_context(
        &done_context,
        prepared,
        None,
        tool_context,
    )
    .await;

    assert!(matches!(launch, ToolCallLaunch::Done(_)));
    assert_eq!(after_calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn before_tool_hook_receives_resolved_argument_projection_policy() {
    let captured = Arc::new(std::sync::Mutex::new(None));
    let outcome = dispatch_tool_call(
        &projection_policy_dispatch_context(Arc::clone(&captured)),
        "seedy".to_string(),
        json!({}),
        None,
    )
    .await;

    assert!(outcome.record.output.is_success());
    assert_eq!(
        captured.lock().expect("captured policy").clone(),
        Some(crate::ToolArgumentProjectionPolicy::preserve_projected_refs_in_field("seed"))
    );
}

#[tokio::test]
async fn dispatch_rejects_non_catalog_tool_before_provider_resolution() {
    let contracts_resolved = Arc::new(AtomicUsize::new(0));
    let executed = Arc::new(AtomicUsize::new(0));
    let provider: Arc<dyn ToolProvider> = Arc::new(ExactDispatchTools {
        contracts_resolved: Arc::clone(&contracts_resolved),
        executed: Arc::clone(&executed),
        contract_available: true,
        observed_execution_bindings: None,
    });
    let outcome = dispatch_tool_call(
        &exact_dispatch_context(provider),
        "host_only".to_string(),
        json!({ "value": "ok" }),
        None,
    )
    .await;

    assert!(!outcome.record.output.is_success());
    assert_eq!(
        outcome.record.output.value_for_projection()["message"],
        json!("Tool is unavailable in this session")
    );
    assert_eq!(contracts_resolved.load(Ordering::SeqCst), 0);
    assert_eq!(executed.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn non_catalog_tool_is_rejected_before_contract_resolution() {
    let contracts_resolved = Arc::new(AtomicUsize::new(0));
    let executed = Arc::new(AtomicUsize::new(0));
    let provider: Arc<dyn ToolProvider> = Arc::new(ExactDispatchTools {
        contracts_resolved: Arc::clone(&contracts_resolved),
        executed: Arc::clone(&executed),
        contract_available: false,
        observed_execution_bindings: None,
    });
    let outcome = dispatch_tool_call(
        &exact_dispatch_context(provider),
        "host_only".to_string(),
        json!({ "value": "ok" }),
        None,
    )
    .await;

    assert!(!outcome.record.output.is_success());
    assert_eq!(
        outcome.record.output.value_for_projection()["message"],
        json!("Tool is unavailable in this session")
    );
    assert_eq!(contracts_resolved.load(Ordering::SeqCst), 0);
    assert_eq!(executed.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn explicit_execution_grant_runs_non_catalog_tool_with_binding() {
    let contracts_resolved = Arc::new(AtomicUsize::new(0));
    let executed = Arc::new(AtomicUsize::new(0));
    let observed_execution_bindings = Arc::new(std::sync::Mutex::new(Vec::new()));
    let provider: Arc<dyn ToolProvider> = Arc::new(ExactDispatchTools {
        contracts_resolved: Arc::clone(&contracts_resolved),
        executed: Arc::clone(&executed),
        contract_available: false,
        observed_execution_bindings: Some(Arc::clone(&observed_execution_bindings)),
    });
    let context = exact_dispatch_context(provider);
    let grant = crate::ToolExecutionGrant::from_definition(named_beta_tool("host_only"))
        .with_source_id(crate::PLUGIN_TOOL_SOURCE_ID)
        .with_execution_binding(json!({ "kind": "test", "route": "deferred" }));
    let pending = crate::sansio::PendingToolCall {
        call_id: "grant-call".to_string(),
        tool_name: "host_only".to_string(),
        args: json!({ "value": "ok" }),
        replay: None,
    };
    let prepared = match prepare_granted_tool_call_with_context(
        &context,
        &grant,
        pending,
        Some("grant-call".to_string()),
    )
    .await
    {
        ToolPreparationOutcome::Prepared(prepared) => prepared,
        ToolPreparationOutcome::Completed(outcome) => {
            panic!("grant should prepare, got {:?}", outcome.record.output)
        }
    };
    let tool_context = ToolContext::from_dispatch(Arc::new(context.clone()))
        .prepared_call(&prepared)
        .tool_execution_binding(grant.execution_binding.clone())
        .build();
    let launch = dispatch_granted_prepared_tool_call_launch_with_execution_context(
        &context,
        &grant,
        prepared,
        None,
        tool_context,
    )
    .await;
    let ToolCallLaunch::Done(outcome) = launch else {
        panic!("grant call should complete");
    };

    assert!(outcome.record.output.is_success());
    assert_eq!(outcome.record.output.value_for_projection(), json!("host"));
    assert_eq!(contracts_resolved.load(Ordering::SeqCst), 0);
    assert_eq!(executed.load(Ordering::SeqCst), 1);
    assert_eq!(
        *observed_execution_bindings
            .lock()
            .expect("execution bindings"),
        vec![json!({ "kind": "test", "route": "deferred" })]
    );
}

#[tokio::test]
async fn dispatch_rejects_hidden_tool_before_contract_resolution() {
    let contracts_resolved = Arc::new(AtomicUsize::new(0));
    let executed = Arc::new(AtomicUsize::new(0));
    let provider: Arc<dyn ToolProvider> = Arc::new(HiddenDispatchTools {
        contracts_resolved: Arc::clone(&contracts_resolved),
        executed: Arc::clone(&executed),
    });
    let outcome = dispatch_tool_call(
        &hidden_member_dispatch_context(provider),
        "hidden".to_string(),
        json!({ "value": "ok" }),
        None,
    )
    .await;

    assert!(!outcome.record.output.is_success());
    assert_eq!(
        outcome.record.output.value_for_projection()["message"],
        json!("Tool is unavailable in this session")
    );
    assert_eq!(contracts_resolved.load(Ordering::SeqCst), 0);
    assert_eq!(executed.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn dispatch_rejects_unknown_mcp_args_before_provider_execution() {
    let executed = Arc::new(AtomicUsize::new(0));
    let outcome = dispatch_tool_call(
        &strict_mcp_dispatch_context(Arc::clone(&executed)),
        "mcp__appworld__venmo_show_transactions".to_string(),
        json!({
            "min_datetime": "2024-01-01T00:00:00Z",
            "limit": 20
        }),
        None,
    )
    .await;

    assert!(!outcome.record.output.is_success());
    assert_eq!(
        outcome.record.output.value_for_projection()["message"],
        json!("min_datetime: unexpected property")
    );
    assert_eq!(executed.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn default_retry_policy_never_retries_safe_failures() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
    let outcome = dispatch_tool_call(
        &retry_dispatch_context(
            ToolRetryPolicy::Never,
            Arc::clone(&attempts),
            usize::MAX,
            false,
            Arc::clone(&observed),
        ),
        "retry_probe".to_string(),
        json!({ "value": "ok" }),
        None,
    )
    .await;

    assert!(!outcome.record.output.is_success());
    assert_eq!(attempts.load(Ordering::SeqCst), 1);
    assert_eq!(observed.lock().expect("observed")[0].0, 1);
}

#[tokio::test]
async fn safe_retry_policy_retries_safe_failure_and_stops_on_success() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
    let outcome = dispatch_tool_call(
        &retry_dispatch_context(
            ToolRetryPolicy::safe(3, 0, 0),
            Arc::clone(&attempts),
            2,
            false,
            Arc::clone(&observed),
        ),
        "retry_probe".to_string(),
        json!({ "value": "ok" }),
        None,
    )
    .await;

    assert!(outcome.record.output.is_success());
    assert_eq!(attempts.load(Ordering::SeqCst), 2);
    assert_eq!(
        observed
            .lock()
            .expect("observed")
            .iter()
            .map(|(attempt, max, _)| (*attempt, *max))
            .collect::<Vec<_>>(),
        vec![(1, 3), (2, 3)]
    );
}

#[derive(Default)]
struct SleepRecordingEffectController {
    sleeps: Arc<std::sync::Mutex<Vec<crate::RuntimeInvocation>>>,
}

#[async_trait::async_trait]
impl crate::RuntimeEffectController for SleepRecordingEffectController {
    async fn execute_effect(
        &self,
        envelope: crate::RuntimeEffectEnvelope,
        _local_executor: crate::RuntimeEffectLocalExecutor<'_>,
    ) -> Result<crate::RuntimeEffectOutcome, crate::RuntimeEffectControllerError> {
        self.sleeps
            .lock()
            .expect("sleep records")
            .push(envelope.invocation);
        Ok(crate::RuntimeEffectOutcome::Sleep)
    }
}

struct FailingSleepEffectController;

#[async_trait::async_trait]
impl crate::RuntimeEffectController for FailingSleepEffectController {
    async fn execute_effect(
        &self,
        envelope: crate::RuntimeEffectEnvelope,
        _local_executor: crate::RuntimeEffectLocalExecutor<'_>,
    ) -> Result<crate::RuntimeEffectOutcome, crate::RuntimeEffectControllerError> {
        Err(crate::RuntimeEffectControllerError::new(
            "test_sleep_rejected",
            format!("rejected {}", envelope.command.kind().as_str()),
        ))
    }
}

#[tokio::test]
async fn retry_delay_crosses_effect_controller_as_sleep_effect() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
    let recorder = Arc::new(SleepRecordingEffectController::default());
    let mut context = exact_dispatch_context(Arc::new(RetryProbeTools {
        definition: retry_tool("retry_probe", ToolRetryPolicy::safe(3, 25, 25)),
        attempts: Arc::clone(&attempts),
        successes_after: 2,
        cancel_on_first: false,
        observed_attempts: Arc::clone(&observed),
        retry_after_ms: Some(25),
    }));
    context.effect_controller = RuntimeEffectControllerHandle::shared(recorder.clone());
    let tool_context = ToolContext::from_dispatch(Arc::new(context.clone()))
        .tool_call_id("call-1".to_string())
        .build();

    let outcome = dispatch_tool_call_with_execution_context(
        &context,
        "retry_probe".to_string(),
        json!({ "value": "ok" }),
        None,
        tool_context,
    )
    .await;

    assert!(outcome.record.output.is_success());
    let sleeps = recorder.sleeps.lock().expect("sleep records");
    assert_eq!(sleeps.len(), 1);
    assert_eq!(
        sleeps[0].effect_kind(),
        Some(crate::RuntimeEffectKind::Sleep)
    );
    assert_eq!(
        sleeps[0].replay_key(),
        Some("lash-tool:session:call-1:retry_probe:attempt:1:sleep")
    );
}

#[tokio::test]
async fn retry_sleep_controller_rejection_returns_explicit_tool_failure() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut context = exact_dispatch_context(Arc::new(RetryProbeTools {
        definition: retry_tool("retry_probe", ToolRetryPolicy::safe(3, 25, 25)),
        attempts: Arc::clone(&attempts),
        successes_after: 2,
        cancel_on_first: false,
        observed_attempts: Arc::clone(&observed),
        retry_after_ms: Some(25),
    }));
    context.effect_controller =
        RuntimeEffectControllerHandle::shared(Arc::new(FailingSleepEffectController));
    let tool_context = ToolContext::from_dispatch(Arc::new(context.clone()))
        .tool_call_id("call-1".to_string())
        .build();

    let outcome = dispatch_tool_call_with_execution_context(
        &context,
        "retry_probe".to_string(),
        json!({ "value": "ok" }),
        None,
        tool_context,
    )
    .await;

    assert_eq!(attempts.load(Ordering::SeqCst), 1);
    let ToolCallOutcome::Failure(failure) = outcome.record.output.outcome else {
        panic!("expected failure");
    };
    assert_eq!(failure.code, "tool_retry_sleep_failed");
}

#[tokio::test]
async fn safe_retry_policy_marks_exhausted_after_final_attempt() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
    let outcome = dispatch_tool_call(
        &retry_dispatch_context(
            ToolRetryPolicy::safe(2, 0, 0),
            Arc::clone(&attempts),
            usize::MAX,
            false,
            Arc::clone(&observed),
        ),
        "retry_probe".to_string(),
        json!({ "value": "ok" }),
        None,
    )
    .await;

    assert!(!outcome.record.output.is_success());
    assert_eq!(attempts.load(Ordering::SeqCst), 2);
    let ToolCallOutcome::Failure(failure) = outcome.record.output.outcome else {
        panic!("expected failure");
    };
    assert_eq!(
        failure.retry,
        ToolRetryDisposition::Exhausted { attempts: 2 }
    );
}

#[tokio::test]
async fn cancellation_stops_retry_immediately() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
    let outcome = dispatch_tool_call(
        &retry_dispatch_context(
            ToolRetryPolicy::safe(3, 0, 0),
            Arc::clone(&attempts),
            usize::MAX,
            true,
            Arc::clone(&observed),
        ),
        "retry_probe".to_string(),
        json!({ "value": "ok" }),
        None,
    )
    .await;

    assert!(!outcome.record.output.is_success());
    assert_eq!(attempts.load(Ordering::SeqCst), 1);
    assert!(matches!(
        outcome.record.output.outcome,
        ToolCallOutcome::Cancelled(_)
    ));
}

#[tokio::test]
async fn retry_context_has_stable_replay_key_across_attempts() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
    let context = retry_dispatch_context(
        ToolRetryPolicy::safe(3, 0, 0),
        Arc::clone(&attempts),
        3,
        false,
        Arc::clone(&observed),
    );
    let tool_context = ToolContext::from_dispatch(Arc::new(context.clone()))
        .tool_call_id("call-1".to_string())
        .build();
    let outcome = dispatch_tool_call_with_execution_context(
        &context,
        "retry_probe".to_string(),
        json!({ "value": "ok" }),
        None,
        tool_context,
    )
    .await;

    assert!(outcome.record.output.is_success());
    let observed = observed.lock().expect("observed");
    assert_eq!(observed.len(), 3);
    assert_eq!(
        observed
            .iter()
            .map(|(attempt, max, _)| (*attempt, *max))
            .collect::<Vec<_>>(),
        vec![(1, 3), (2, 3), (3, 3)]
    );
    let keys = observed
        .iter()
        .map(|(_, _, key)| key.clone())
        .collect::<Vec<_>>();
    assert!(keys.iter().all(|key| key == &keys[0]));
    assert_eq!(
        keys[0].as_deref(),
        Some("lash-tool:session:call-1:retry_probe")
    );
}

#[tokio::test]
async fn idempotent_retry_policy_requires_stable_key() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
    let outcome = dispatch_tool_call(
        &retry_dispatch_context(
            ToolRetryPolicy::idempotent(3, 0, 0),
            Arc::clone(&attempts),
            usize::MAX,
            false,
            Arc::clone(&observed),
        ),
        "retry_probe".to_string(),
        json!({ "value": "ok" }),
        None,
    )
    .await;

    assert!(!outcome.record.output.is_success());
    assert_eq!(attempts.load(Ordering::SeqCst), 1);
    assert_eq!(observed.lock().expect("observed")[0].1, 1);
}

#[tokio::test]
async fn batch_returns_explicit_errors_without_runtime_execution_context() {
    let outcome = dispatch_tool_call(
        &dispatch_context(),
        "batch".to_string(),
        json!({
            "tool_calls": [
                {"tool": "alpha", "parameters": {}},
                {"tool": "beta", "parameters": {"value": "ok"}},
                {"tool": "beta", "parameters": {"value": "fail"}}
            ]
        }),
        None,
    )
    .await;

    assert!(outcome.record.output.is_success());
    assert_eq!(outcome.record.tool, "batch");
    let value = outcome.record.output.value_for_projection();
    let results = value
        .get("results")
        .and_then(|value| value.as_array())
        .expect("results");
    assert_eq!(results.len(), 3);
    assert_eq!(
        results
            .iter()
            .filter(|item| item.get("success").and_then(|value| value.as_bool()) == Some(false))
            .count(),
        3
    );
    assert_eq!(results[0].get("tool"), Some(&json!("tool:alpha")));
    assert_eq!(
        results[0]
            .get("error")
            .and_then(|value| value.get("message"))
            .and_then(|value| value.as_str()),
        Some("tool batch dispatch is unavailable outside runtime execution")
    );
}

#[tokio::test]
async fn batch_rejects_nested_batch_as_partial_failure() {
    let outcome = dispatch_tool_call(
        &dispatch_context(),
        "batch".to_string(),
        json!({
            "tool_calls": [
                {"tool": "batch", "parameters": {"tool_calls": []}}
            ]
        }),
        None,
    )
    .await;

    assert!(outcome.record.output.is_success());
    let value = outcome.record.output.value_for_projection();
    let first = value
        .get("results")
        .and_then(|value| value.as_array())
        .and_then(|items| items.first())
        .expect("first result");
    assert_eq!(
        first.get("error"),
        Some(&json!("Tool 'batch' is not allowed inside batch"))
    );
}

#[tokio::test]
async fn batch_marks_overflow_calls_as_failures() {
    let tool_calls = (0..26)
        .map(|_| json!({"tool": "alpha", "parameters": {}}))
        .collect::<Vec<_>>();

    let outcome = dispatch_tool_call(
        &dispatch_context(),
        "batch".to_string(),
        json!({ "tool_calls": tool_calls }),
        None,
    )
    .await;

    assert!(!outcome.record.output.is_success());
    let value = outcome.record.output.value_for_projection();
    let error = value
        .get("message")
        .and_then(|value| value.as_str())
        .expect("string error result");
    assert!(
        error.contains("tool_calls") && error.contains("items <= 25"),
        "{error}",
    );
}

#[tokio::test]
async fn batch_does_not_run_child_tools_without_runtime_execution_context() {
    let barrier = Arc::new(Barrier::new(2));
    let started = Arc::new(AtomicUsize::new(0));
    let outcome = dispatch_tool_call(
        &parallel_dispatch_context(Arc::clone(&barrier), Arc::clone(&started)),
        "batch".to_string(),
        json!({
            "tool_calls": [
                {"tool": "probe_a", "parameters": {}},
                {"tool": "probe_b", "parameters": {}}
            ]
        }),
        None,
    )
    .await;

    assert!(outcome.record.output.is_success());
    assert_eq!(started.load(Ordering::SeqCst), 0);
    let value = outcome.record.output.value_for_projection();
    let results = value
        .get("results")
        .and_then(|value| value.as_array())
        .expect("results");
    assert_eq!(results.len(), 2);
    assert!(
        results
            .iter()
            .all(|item| item.get("success").and_then(|value| value.as_bool()) == Some(false))
    );
}

/// A tool provider whose tools are marked [`ToolScheduling::Serial`]
/// and log (start, end) instants around a sleep into a shared `Mutex`.
struct SerialProbeTools {
    /// (tool_name, start_instant, end_instant)
    log: Arc<std::sync::Mutex<Vec<(String, Instant, Instant)>>>,
}

#[async_trait::async_trait]
impl ToolProvider for SerialProbeTools {
    fn tool_manifests(&self) -> Vec<crate::ToolManifest> {
        manifests(vec![
            test_tool("serial_a", ToolScheduling::Serial),
            test_tool("serial_b", ToolScheduling::Serial),
        ])
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<crate::ToolContract>> {
        contract_from(
            vec![
                test_tool("serial_a", ToolScheduling::Serial),
                test_tool("serial_b", ToolScheduling::Serial),
            ],
            name,
        )
    }

    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        let start = Instant::now();
        // Sleep long enough that if the two tools *were* dispatched
        // concurrently, their windows would overlap by a detectable
        // margin.
        tokio::time::sleep(Duration::from_millis(40)).await;
        let end = Instant::now();
        self.log
            .lock()
            .expect("serial probe log")
            .push((call.name.to_string(), start, end));
        ToolResult::ok(json!(call.name))
    }
}

fn serial_dispatch_context(
    log: Arc<std::sync::Mutex<Vec<(String, Instant, Instant)>>>,
) -> ToolDispatchContext<'static> {
    let (event_tx, _event_rx) = mpsc::channel(8);
    let plugins = test_plugins(Arc::new(SerialProbeTools { log }));
    let tools = plugins.tools();
    let tool_catalog = plugins
        .resolved_tool_catalog("session")
        .expect("tool catalog");
    ToolDispatchContext {
        plugins,
        tools,
        tool_catalog,
        sessions: Arc::new(MockSessionManager::default()),
        session_lifecycle: Arc::new(MockSessionManager::default()),
        session_graph: Arc::new(MockSessionManager::default()),
        processes: Arc::new(crate::UnavailableProcessService),
        process_cancel_ability: Arc::new(crate::DefaultProcessCancelAbility),
        trigger_router: None,
        effect_controller: RuntimeEffectControllerHandle::shared(Arc::new(
            crate::InlineRuntimeEffectController,
        )),
        direct_completions: crate::DirectCompletionClient::unavailable(
            "direct completions are unavailable in this test context",
        ),
        parent_invocation: None,
        execution_env_spec: crate::ProcessExecutionEnvSpec::new(
            crate::PluginOptions::default(),
            crate::SessionPolicy::default(),
        ),
        session_id: "session".to_string(),
        agent_frame_id: String::new(),
        event_tx,
        checkpoint_messages: crate::tool_dispatch::CheckpointMessageBuffer::default(),
        trigger_outcomes: crate::tool_dispatch::ToolTriggerOutcomeBuffer::default(),
        attachment_store: Arc::new(crate::InMemoryAttachmentStore::new()),
        turn_context: crate::TurnContext::default(),
        clock: std::sync::Arc::new(crate::SystemClock),
    }
}

/// Two Serial tools in the same batch must not interleave: the second
/// call's start instant must be at or after the first call's end
/// instant.
#[tokio::test]
async fn serial_tools_do_not_interleave() {
    let log: Arc<std::sync::Mutex<Vec<(String, Instant, Instant)>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    let context = Arc::new(serial_dispatch_context(Arc::clone(&log)));

    let specs = vec![
        ParallelToolCallSpec {
            index: 0,
            tool_name: "serial_a".to_string(),
            args: json!({}),
        },
        ParallelToolCallSpec {
            index: 1,
            tool_name: "serial_b".to_string(),
            args: json!({}),
        },
    ];

    let outcomes = dispatch_parallel_tool_calls(context, specs, None).await;

    assert_eq!(outcomes.len(), 2);
    assert!(
        outcomes
            .iter()
            .all(|outcome| outcome.record.output.is_success())
    );
    // Outcomes are sorted by original index.
    assert_eq!(outcomes[0].index, 0);
    assert_eq!(outcomes[1].index, 1);
    assert_eq!(outcomes[0].record.tool, "serial_a");
    assert_eq!(outcomes[1].record.tool, "serial_b");

    let entries = log.lock().expect("log").clone();
    assert_eq!(entries.len(), 2, "both serial tools must have executed");
    // Sort entries by start time so we compare the first-to-run vs
    // second-to-run regardless of which tool happened to go first.
    let mut sorted = entries;
    sorted.sort_by_key(|(_, start, _)| *start);
    let (first_name, _first_start, first_end) = &sorted[0];
    let (second_name, second_start, _second_end) = &sorted[1];
    assert_ne!(first_name, second_name, "both tools should have run");
    assert!(
        second_start >= first_end,
        "serial tool ranges must not overlap: first ended at {:?}, second started at {:?}",
        first_end,
        second_start,
    );
}

struct SerialRetryProbeTools {
    log: Arc<std::sync::Mutex<Vec<(String, Instant, Instant)>>>,
    attempts_a: Arc<AtomicUsize>,
    attempts_b: Arc<AtomicUsize>,
}

#[async_trait::async_trait]
impl ToolProvider for SerialRetryProbeTools {
    fn tool_manifests(&self) -> Vec<crate::ToolManifest> {
        manifests(vec![
            test_tool("serial_retry_a", ToolScheduling::Serial)
                .with_retry_policy(ToolRetryPolicy::safe(2, 0, 0)),
            test_tool("serial_retry_b", ToolScheduling::Serial)
                .with_retry_policy(ToolRetryPolicy::safe(2, 0, 0)),
        ])
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<crate::ToolContract>> {
        contract_from(
            vec![
                test_tool("serial_retry_a", ToolScheduling::Serial)
                    .with_retry_policy(ToolRetryPolicy::safe(2, 0, 0)),
                test_tool("serial_retry_b", ToolScheduling::Serial)
                    .with_retry_policy(ToolRetryPolicy::safe(2, 0, 0)),
            ],
            name,
        )
    }

    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        let start = Instant::now();
        tokio::time::sleep(Duration::from_millis(20)).await;
        let end = Instant::now();
        self.log
            .lock()
            .expect("serial retry log")
            .push((call.name.to_string(), start, end));

        let attempt = match call.name {
            "serial_retry_a" => self.attempts_a.fetch_add(1, Ordering::SeqCst) + 1,
            "serial_retry_b" => self.attempts_b.fetch_add(1, Ordering::SeqCst) + 1,
            _ => 1,
        };
        if attempt == 1 {
            ToolResult::retryable_failure(
                crate::ToolFailureClass::External,
                "transient",
                "transient failure",
                Some(0),
            )
        } else {
            ToolResult::ok(json!(call.name))
        }
    }
}

#[tokio::test]
async fn serial_tool_retries_do_not_overlap_other_serial_calls() {
    let log = Arc::new(std::sync::Mutex::new(Vec::new()));
    let attempts_a = Arc::new(AtomicUsize::new(0));
    let attempts_b = Arc::new(AtomicUsize::new(0));
    let provider = Arc::new(SerialRetryProbeTools {
        log: Arc::clone(&log),
        attempts_a: Arc::clone(&attempts_a),
        attempts_b: Arc::clone(&attempts_b),
    });
    let (event_tx, _event_rx) = mpsc::channel(8);
    let plugins = test_plugins(provider);
    let tools = plugins.tools();
    let tool_catalog = plugins
        .resolved_tool_catalog("session")
        .expect("tool catalog");
    let context = Arc::new(ToolDispatchContext {
        plugins,
        tools,
        tool_catalog,
        sessions: Arc::new(MockSessionManager::default()),
        session_lifecycle: Arc::new(MockSessionManager::default()),
        session_graph: Arc::new(MockSessionManager::default()),
        processes: Arc::new(crate::UnavailableProcessService),
        process_cancel_ability: Arc::new(crate::DefaultProcessCancelAbility),
        trigger_router: None,
        effect_controller: RuntimeEffectControllerHandle::shared(Arc::new(
            crate::InlineRuntimeEffectController,
        )),
        direct_completions: crate::DirectCompletionClient::unavailable(
            "direct completions are unavailable in this test context",
        ),
        parent_invocation: None,
        execution_env_spec: crate::ProcessExecutionEnvSpec::new(
            crate::PluginOptions::default(),
            crate::SessionPolicy::default(),
        ),
        session_id: "session".to_string(),
        agent_frame_id: String::new(),
        event_tx,
        checkpoint_messages: crate::tool_dispatch::CheckpointMessageBuffer::default(),
        trigger_outcomes: crate::tool_dispatch::ToolTriggerOutcomeBuffer::default(),
        attachment_store: Arc::new(crate::InMemoryAttachmentStore::new()),
        turn_context: crate::TurnContext::default(),
        clock: std::sync::Arc::new(crate::SystemClock),
    });

    let outcomes = dispatch_parallel_tool_calls(
        context,
        vec![
            ParallelToolCallSpec {
                index: 0,
                tool_name: "serial_retry_a".to_string(),
                args: json!({}),
            },
            ParallelToolCallSpec {
                index: 1,
                tool_name: "serial_retry_b".to_string(),
                args: json!({}),
            },
        ],
        None,
    )
    .await;

    assert!(
        outcomes
            .iter()
            .all(|outcome| outcome.record.output.is_success())
    );
    assert_eq!(attempts_a.load(Ordering::SeqCst), 2);
    assert_eq!(attempts_b.load(Ordering::SeqCst), 2);

    let mut entries = log.lock().expect("serial retry log").clone();
    entries.sort_by_key(|(_, start, _)| *start);
    assert_eq!(entries.len(), 4);
    for window in entries.windows(2) {
        assert!(
            window[1].1 >= window[0].2,
            "serial retry windows must not overlap: {:?} then {:?}",
            window[0],
            window[1],
        );
    }
}

/// When a batch contains a mix of parallel and serial tools, the
/// parallel-safe ones should still run concurrently with each other
/// (verified via a Barrier), and the serial one should run separately
/// without interleaving with any parallel peer's window.
#[tokio::test]
async fn mixed_batch_runs_parallel_tools_concurrently_and_serial_alone() {
    struct MixedTools {
        barrier: Arc<Barrier>,
        serial_window: Arc<std::sync::Mutex<Option<(Instant, Instant)>>>,
        parallel_windows: Arc<std::sync::Mutex<Vec<(String, Instant, Instant)>>>,
    }

    #[async_trait::async_trait]
    impl ToolProvider for MixedTools {
        fn tool_manifests(&self) -> Vec<crate::ToolManifest> {
            manifests(vec![
                test_tool("par_a", ToolScheduling::Parallel),
                test_tool("par_b", ToolScheduling::Parallel),
                test_tool("ser", ToolScheduling::Serial),
            ])
        }

        fn resolve_contract(&self, name: &str) -> Option<Arc<crate::ToolContract>> {
            contract_from(
                vec![
                    test_tool("par_a", ToolScheduling::Parallel),
                    test_tool("par_b", ToolScheduling::Parallel),
                    test_tool("ser", ToolScheduling::Serial),
                ],
                name,
            )
        }

        async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
            let name = call.name;
            if name == "ser" {
                let start = Instant::now();
                tokio::time::sleep(Duration::from_millis(30)).await;
                let end = Instant::now();
                *self.serial_window.lock().expect("serial window") = Some((start, end));
                ToolResult::ok(json!(name))
            } else {
                let start = Instant::now();
                // Block until both parallel tools have reached this
                // barrier — proves they're running concurrently.
                let waited = timeout(Duration::from_millis(200), self.barrier.wait()).await;
                let end = Instant::now();
                self.parallel_windows
                    .lock()
                    .expect("parallel windows")
                    .push((name.to_string(), start, end));
                match waited {
                    Ok(_) => ToolResult::ok(json!(name)),
                    Err(_) => ToolResult::err_fmt(format!("{name} did not overlap with peer")),
                }
            }
        }
    }

    let barrier = Arc::new(Barrier::new(2));
    let serial_window = Arc::new(std::sync::Mutex::new(None));
    let parallel_windows = Arc::new(std::sync::Mutex::new(Vec::new()));
    let (event_tx, _event_rx) = mpsc::channel(8);
    let provider = Arc::new(MixedTools {
        barrier: Arc::clone(&barrier),
        serial_window: Arc::clone(&serial_window),
        parallel_windows: Arc::clone(&parallel_windows),
    });
    let plugins = test_plugins(provider);
    let tools = plugins.tools();
    let tool_catalog = plugins
        .resolved_tool_catalog("session")
        .expect("tool catalog");
    let context = Arc::new(ToolDispatchContext {
        plugins,
        tools,
        tool_catalog,
        sessions: Arc::new(MockSessionManager::default()),
        session_lifecycle: Arc::new(MockSessionManager::default()),
        session_graph: Arc::new(MockSessionManager::default()),
        processes: Arc::new(crate::UnavailableProcessService),
        process_cancel_ability: Arc::new(crate::DefaultProcessCancelAbility),
        trigger_router: None,
        effect_controller: RuntimeEffectControllerHandle::shared(Arc::new(
            crate::InlineRuntimeEffectController,
        )),
        direct_completions: crate::DirectCompletionClient::unavailable(
            "direct completions are unavailable in this test context",
        ),
        parent_invocation: None,
        execution_env_spec: crate::ProcessExecutionEnvSpec::new(
            crate::PluginOptions::default(),
            crate::SessionPolicy::default(),
        ),
        session_id: "session".to_string(),
        agent_frame_id: String::new(),
        event_tx,
        checkpoint_messages: crate::tool_dispatch::CheckpointMessageBuffer::default(),
        trigger_outcomes: crate::tool_dispatch::ToolTriggerOutcomeBuffer::default(),
        attachment_store: Arc::new(crate::InMemoryAttachmentStore::new()),
        turn_context: crate::TurnContext::default(),
        clock: std::sync::Arc::new(crate::SystemClock),
    });

    let specs = vec![
        ParallelToolCallSpec {
            index: 0,
            tool_name: "par_a".to_string(),
            args: json!({}),
        },
        ParallelToolCallSpec {
            index: 1,
            tool_name: "ser".to_string(),
            args: json!({}),
        },
        ParallelToolCallSpec {
            index: 2,
            tool_name: "par_b".to_string(),
            args: json!({}),
        },
    ];

    let outcomes = dispatch_parallel_tool_calls(context, specs, None).await;

    assert_eq!(outcomes.len(), 3);
    assert!(
        outcomes
            .iter()
            .all(|outcome| outcome.record.output.is_success()),
        "all tools should succeed: {:?}",
        outcomes
            .iter()
            .map(|outcome| (&outcome.record.tool, outcome.record.output.is_success()))
            .collect::<Vec<_>>()
    );

    let pw = parallel_windows.lock().expect("parallel windows");
    assert_eq!(pw.len(), 2);
    let sw = serial_window
        .lock()
        .expect("serial window")
        .expect("serial window recorded");

    // The serial tool's window must not overlap either parallel
    // tool's window (Option A: serial runs after parallel).
    for (name, p_start, p_end) in pw.iter() {
        assert!(
            sw.0 >= *p_end || sw.1 <= *p_start,
            "serial window {:?} overlaps parallel window {} {:?}..{:?}",
            sw,
            name,
            p_start,
            p_end,
        );
    }
}
