use std::sync::Arc;

use async_trait::async_trait;
use lash::advanced::{AdvancedLashCoreBuilder, RuntimeCoreConfig};
use lash::direct::{
    DirectLlmClient, DirectLlmError, DirectRequest, LlmAttachment, LlmEventSender, LlmOutputPart,
    LlmResponse, LlmUsage, TokenUsage,
};
use lash::model_info::{
    ModelCatalog, ModelInfo, ResolvedModelSpec, bundled_models_dev_snapshot,
};
use lash::persistence::{
    GcReport, GraphCommitDelta, PersistedSessionRead, PersistedSessionState, RuntimeCommit,
    RuntimeCommitResult, RuntimePersistence, SessionCheckpoint, SessionMeta, SessionNodeRecord,
    SessionReadScope, StoreError, TokenLedgerEntry, VacuumReport,
    load_persisted_session_state, load_persisted_session_state_active_path,
};
use lash::plugins::{
    AfterToolCallHook, BeforeToolCallHook, PluginDirective, PluginHost, PluginSpec,
    PluginSpecBuilder, PluginSpecFactory, ToolCallHookContext, ToolResultHookContext,
};
use lash::provider::{
    ProviderRateLimitPolicy, ProviderReliability, ProviderReliabilityBuilder, ProviderRetryPolicy,
    ProviderTimeoutPolicy,
};
use lash::tools::{ToolActivation, ToolDiscoveryMetadata, ToolOutputContract};

struct FacadeStore;

#[async_trait]
impl RuntimePersistence for FacadeStore {
    async fn load_session(
        &self,
        _scope: SessionReadScope,
    ) -> Result<Option<PersistedSessionRead>, StoreError> {
        Ok(None)
    }

    async fn load_node(
        &self,
        _node_id: &str,
    ) -> Result<Option<SessionNodeRecord>, StoreError> {
        Ok(None)
    }

    async fn commit_runtime_state(
        &self,
        commit: RuntimeCommit,
    ) -> Result<RuntimeCommitResult, StoreError> {
        let manifest = SessionCheckpoint {
            turn_state: commit.checkpoint.turn_state,
            tool_state_ref: commit.checkpoint.tool_state_ref,
            plugin_snapshot_ref: commit.checkpoint.plugin_snapshot_ref,
            plugin_snapshot_revision: commit.checkpoint.plugin_snapshot_revision,
            execution_state_ref: commit.checkpoint.execution_state_ref,
        };
        Ok(RuntimeCommitResult {
            head_revision: commit.expected_head_revision.unwrap_or_default() + 1,
            checkpoint_ref: "checkpoint".to_string().into(),
            manifest,
        })
    }

    async fn save_session_meta(&self, _meta: SessionMeta) -> Result<(), StoreError> {
        Ok(())
    }

    async fn load_session_meta(&self) -> Result<Option<SessionMeta>, StoreError> {
        Ok(None)
    }

    async fn tombstone_nodes(&self, _ids: &[String]) -> Result<(), StoreError> {
        Ok(())
    }

    async fn vacuum(&self) -> Result<VacuumReport, StoreError> {
        Ok(VacuumReport::default())
    }

    async fn gc_unreachable(&self) -> Result<GcReport, StoreError> {
        Ok(GcReport::default())
    }
}

fn persistence_types_are_nameable(
    graph: GraphCommitDelta,
    ledger: Vec<TokenLedgerEntry>,
) -> RuntimeCommit {
    RuntimeCommit {
        session_id: "facade".to_string(),
        expected_head_revision: Some(0),
        config: Default::default(),
        graph,
        checkpoint: Default::default(),
        usage_deltas: ledger,
    }
}

fn plugin_types_are_nameable() -> PluginHost {
    let before: BeforeToolCallHook = Arc::new(|ctx: ToolCallHookContext| {
        Box::pin(async move {
            Ok(vec![PluginDirective::ReplaceToolArgs {
                args: ctx.args,
            }])
        })
    });
    let after: AfterToolCallHook = Arc::new(|ctx: ToolResultHookContext| {
        Box::pin(async move {
            Ok(vec![PluginDirective::short_circuit(ctx.result)])
        })
    });
    let builder: PluginSpecBuilder = Arc::new(move |_ctx| {
        Ok(PluginSpec::new()
            .with_before_tool_call(Arc::clone(&before))
            .with_after_tool_call(Arc::clone(&after)))
    });
    PluginHost::new(vec![Arc::new(PluginSpecFactory::new("facade", builder))])
}

async fn direct_response_type_is_nameable(
    client: &mut DirectLlmClient,
    request: DirectRequest,
) -> Result<LlmResponse, DirectLlmError> {
    client.complete(request).await
}

fn direct_payload_types_are_nameable(
    attachment: LlmAttachment,
    event_sender: LlmEventSender,
    output: LlmOutputPart,
    usage: LlmUsage,
    token_usage: TokenUsage,
) {
    let _ = (attachment, event_sender, output, usage, token_usage);
}

fn advanced_builder_accepts_runtime_core_config(
    builder: AdvancedLashCoreBuilder,
    config: RuntimeCoreConfig,
) -> AdvancedLashCoreBuilder {
    builder.runtime_core_config(config)
}

fn tool_contract_types_are_nameable(
    activation: ToolActivation,
    contract: ToolOutputContract,
    discovery: ToolDiscoveryMetadata,
) {
    let _ = (activation, contract, discovery);
}

fn provider_reliability_types_are_nameable(
    reliability: ProviderReliability,
    builder: ProviderReliabilityBuilder,
    retry: ProviderRetryPolicy,
    timeouts: ProviderTimeoutPolicy,
    rate_limits: ProviderRateLimitPolicy,
) {
    let _ = (reliability, builder, retry, timeouts, rate_limits);
}

fn model_info_types_are_nameable(
    catalog: ModelCatalog,
    info: ModelInfo,
    spec: ResolvedModelSpec,
) {
    let _ = (catalog, info, spec, bundled_models_dev_snapshot());
}

async fn persistence_load_helpers_are_nameable(
    store: &dyn RuntimePersistence,
) -> Result<Option<PersistedSessionState>, StoreError> {
    let _ = load_persisted_session_state_active_path(store, None).await?;
    load_persisted_session_state(store).await
}

fn assert_store_object(_: Arc<dyn RuntimePersistence>) {}

fn main() {
    assert_store_object(Arc::new(FacadeStore));
    let _ = persistence_types_are_nameable(
        GraphCommitDelta::Unchanged { leaf_node_id: None },
        Vec::new(),
    );
    let _ = plugin_types_are_nameable();
    let _ = direct_response_type_is_nameable;
    let _ = direct_payload_types_are_nameable;
    let _ = advanced_builder_accepts_runtime_core_config;
    let _ = tool_contract_types_are_nameable;
    let _ = provider_reliability_types_are_nameable;
    let _ = model_info_types_are_nameable;
    let _ = persistence_load_helpers_are_nameable;
}
