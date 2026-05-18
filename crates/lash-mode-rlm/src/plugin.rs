use std::collections::BTreeSet;
use std::sync::{Arc, Mutex, RwLock};

use lash_core::llm::types::{
    LlmContentBlock, LlmJsonSchema, LlmMessage, LlmOutputSpec, LlmRequest, LlmRole, LlmToolChoice,
};
use lash_core::plugin::{
    CheckpointHookContext, HistoryError, ModeBeforeLlmCallContext, ModeLlmCallAction,
    ModeProtocolDriverPlugin, ModeRuntimeContext, ModeSessionContext, ModeSessionPlugin,
    PluginDirective, PluginError, PluginFactory, PluginRegistrar, PluginSessionContext,
    SessionPlugin, ToolCallHookContext, TurnContextTransform, TurnTransformContext,
};
use lash_core::session_model::context::PreparedContext;
use lash_core::{
    CheckpointKind, ExecutionMode, ModeBuildInput, ModePreamble, SessionError,
    ToolOutputBudgetConfig,
};
use lash_rlm_types::{RlmCreateExtras, RlmGlobalsPatchPluginBody, RlmModeEvent};

use crate::driver::{RlmProjectorConfig, SharedPromptUsage, build_rlm_preamble};
use crate::executor::{RlmExecutionState, execute_code};
use crate::projected_bindings::{
    ProjectionRegistry, ProjectionResolver, RLM_TURN_INPUT_PLUGIN_ID, RlmProjectedBindings,
    RlmProjectionExtension,
};
use crate::rlm_support::BoundVariablesCache;
#[cfg(test)]
use crate::rlm_support::format_budget_suffix;
use crate::stream_mask;

const BUDGET_WARNING_STATUS: &str = "rlm_context_budget_warning";

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct RlmModePluginConfig {
    pub observe_projection: ToolOutputBudgetConfig,
    #[serde(default)]
    pub prompt_features: crate::protocol::RlmPromptFeatures,
    #[serde(default = "default_max_output_chars")]
    pub max_output_chars: usize,
    #[serde(default = "default_continue_as_soft_warn_tokens")]
    pub continue_as_soft_warn_tokens: Option<usize>,
    #[serde(default = "default_continue_as_forced_fallback_tokens")]
    pub continue_as_forced_fallback_tokens: Option<usize>,
}

fn default_max_output_chars() -> usize {
    10_000
}

fn default_continue_as_soft_warn_tokens() -> Option<usize> {
    Some(100_000)
}

fn default_continue_as_forced_fallback_tokens() -> Option<usize> {
    Some(140_000)
}

impl Default for RlmModePluginConfig {
    fn default() -> Self {
        Self {
            observe_projection: ToolOutputBudgetConfig::default(),
            prompt_features: crate::protocol::RlmPromptFeatures::default(),
            max_output_chars: default_max_output_chars(),
            continue_as_soft_warn_tokens: default_continue_as_soft_warn_tokens(),
            continue_as_forced_fallback_tokens: default_continue_as_forced_fallback_tokens(),
        }
    }
}

impl RlmModePluginConfig {
    pub fn validate(&self) -> Result<(), String> {
        if let (Some(soft), Some(forced)) = (
            self.continue_as_soft_warn_tokens,
            self.continue_as_forced_fallback_tokens,
        ) && forced < soft
        {
            return Err(format!(
                "continue_as_forced_fallback_tokens ({forced}) must be greater than or equal to continue_as_soft_warn_tokens ({soft})"
            ));
        }
        Ok(())
    }
}

pub struct BuiltinRlmModePluginFactory {
    config: RlmModePluginConfig,
    projection_resolver: Arc<dyn ProjectionResolver>,
}

impl BuiltinRlmModePluginFactory {
    pub fn new(config: RlmModePluginConfig) -> Self {
        Self {
            config,
            projection_resolver: Arc::new(ProjectionRegistry::default()),
        }
    }

    pub fn with_projection_resolver(
        mut self,
        projection_resolver: Arc<dyn ProjectionResolver>,
    ) -> Self {
        self.projection_resolver = projection_resolver;
        self
    }
}

impl Default for BuiltinRlmModePluginFactory {
    fn default() -> Self {
        Self::new(RlmModePluginConfig::default())
    }
}

impl PluginFactory for BuiltinRlmModePluginFactory {
    fn id(&self) -> &'static str {
        "mode_rlm"
    }

    fn build(&self, ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(RlmModePlugin {
            active: ctx.execution_mode == ExecutionMode::new("rlm"),
            config: self.config.clone(),
            projection_resolver: Arc::clone(&self.projection_resolver),
            last_prompt_usage: Arc::new(RwLock::new(None)),
        }))
    }
}

struct RlmModePlugin {
    active: bool,
    config: RlmModePluginConfig,
    projection_resolver: Arc<dyn ProjectionResolver>,
    last_prompt_usage: SharedPromptUsage,
}

impl SessionPlugin for RlmModePlugin {
    fn id(&self) -> &'static str {
        "mode_rlm"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        if !self.active {
            return Ok(());
        }
        let mode_session = Arc::new(
            RlmModeSession::new(self.config.clone(), Arc::clone(&self.projection_resolver))
                .map_err(|err| PluginError::Session(err.to_string()))?,
        );
        reg.mode().session(mode_session.clone())?;
        reg.mode().protocol_driver(Arc::new(RlmProtocolDriver {
            config: self.config.clone(),
            last_prompt_usage: Arc::clone(&self.last_prompt_usage),
        }))?;
        reg.tools()
            .provider(Arc::new(crate::control_tools::RlmControlToolsProvider))?;
        reg.tool_calls().before(Arc::new(|ctx| {
            Box::pin(async move { normalize_projected_tool_args(ctx) })
        }));

        let bound_vars_cache = Arc::new(BoundVariablesCache::new());
        let bound_vars_hook: lash_core::plugin::PromptContributor = Arc::new(move |ctx| {
            let cache = Arc::clone(&bound_vars_cache);
            Box::pin(async move { Ok(cache.contributions(&ctx)) })
        });
        reg.prompt().contribute(bound_vars_hook);

        let projected_session = mode_session.clone();
        reg.prompt().contribute(Arc::new(move |ctx| {
            let session = projected_session.clone();
            Box::pin(async move {
                let mut contributions = session.projected_binding_prompt_contributions().await;
                if let Some(extension) = ctx
                    .turn_context
                    .plugin_input::<RlmProjectionExtension>(RLM_TURN_INPUT_PLUGIN_ID)
                {
                    contributions.extend(RlmProjectionExtension::prompt_contributions_for(
                        &extension.bindings,
                    ));
                }
                Ok(contributions)
            })
        }));

        // Per-turn `prompt_usage` is captured here and passed to the
        // projector via a shared cell so the budget line can ride in the
        // volatile turn-tail message instead of poisoning the cached
        // system prefix.
        reg.history().prepare_turn(
            10,
            Arc::new(BudgetUsageObserver {
                cell: Arc::clone(&self.last_prompt_usage),
            }),
        );

        let warn_session = mode_session.clone();
        reg.turn().checkpoint(Arc::new(move |ctx| {
            let session = warn_session.clone();
            Box::pin(async move { session.soft_warn_directives(ctx) })
        }));

        stream_mask::register_stream_mask(reg)?;
        Ok(())
    }
}

struct RlmProtocolDriver {
    config: RlmModePluginConfig,
    last_prompt_usage: SharedPromptUsage,
}

impl ModeProtocolDriverPlugin for RlmProtocolDriver {
    fn mode_id(&self) -> &str {
        "rlm"
    }

    fn build_preamble(&self, input: ModeBuildInput) -> ModePreamble {
        build_rlm_preamble(
            input,
            RlmProjectorConfig {
                max_output_chars: self.config.max_output_chars,
                max_budget_tokens: self.config.continue_as_soft_warn_tokens,
                last_prompt_usage: Arc::clone(&self.last_prompt_usage),
                prompt_features: self.config.prompt_features,
            },
        )
    }
}

fn normalize_projected_tool_args(
    ctx: ToolCallHookContext,
) -> Result<Vec<PluginDirective>, PluginError> {
    let original = ctx.args;
    let normalized = crate::projection_codec::normalize_tool_args_for_projection(
        original.clone(),
        &ctx.argument_projection,
    );
    if normalized == original {
        Ok(Vec::new())
    } else {
        Ok(vec![PluginDirective::ReplaceToolArgs { args: normalized }])
    }
}

struct BudgetUsageObserver {
    cell: SharedPromptUsage,
}

#[async_trait::async_trait]
impl TurnContextTransform for BudgetUsageObserver {
    fn id(&self) -> &'static str {
        "rlm.budget_usage_observer"
    }

    async fn transform(
        &self,
        ctx: &TurnTransformContext,
        input: PreparedContext,
    ) -> Result<PreparedContext, HistoryError> {
        if let Ok(mut guard) = self.cell.write() {
            *guard = ctx.prompt_usage.clone();
        }
        Ok(input)
    }
}

struct RlmModeSession {
    config: RlmModePluginConfig,
    projection_resolver: Arc<dyn ProjectionResolver>,
    warned_at_threshold: Mutex<bool>,
    execution: tokio::sync::Mutex<Option<RlmExecutionState>>,
    session_projected_bindings: tokio::sync::Mutex<RlmProjectedBindings>,
}

impl RlmModeSession {
    fn new(
        config: RlmModePluginConfig,
        projection_resolver: Arc<dyn ProjectionResolver>,
    ) -> Result<Self, SessionError> {
        config
            .validate()
            .map_err(|err| SessionError::Protocol(err.to_string()))?;
        Ok(Self {
            execution: tokio::sync::Mutex::new(Some(RlmExecutionState::new(
                config.observe_projection.clone(),
            )?)),
            projection_resolver,
            config,
            warned_at_threshold: Mutex::new(false),
            session_projected_bindings: tokio::sync::Mutex::new(RlmProjectedBindings::new()),
        })
    }

    async fn projected_binding_prompt_contributions(&self) -> Vec<lash_core::PromptContribution> {
        let bindings = self.session_projected_bindings.lock().await;
        RlmProjectionExtension::prompt_contributions_for(&bindings)
    }

    async fn protected_projected_binding_names(&self) -> BTreeSet<String> {
        self.session_projected_bindings
            .lock()
            .await
            .names()
            .collect()
    }

    fn soft_warn_directives(
        &self,
        ctx: CheckpointHookContext,
    ) -> Result<Vec<PluginDirective>, PluginError> {
        if ctx.checkpoint != CheckpointKind::AfterWork {
            return Ok(Vec::new());
        }
        let Some(threshold) = self.config.continue_as_soft_warn_tokens else {
            return Ok(Vec::new());
        };
        let used = ctx.state.token_usage().total().max(0) as usize;
        if used == 0 {
            return Ok(Vec::new());
        }
        if used < threshold {
            return Ok(Vec::new());
        }
        let mut warned = self
            .warned_at_threshold
            .lock()
            .map_err(|_| PluginError::Session("rlm soft-warning state poisoned".to_string()))?;
        if *warned {
            return Ok(Vec::new());
        }
        *warned = true;
        Ok(vec![PluginDirective::emit_runtime_events(vec![
            lash_core::PluginRuntimeEvent::Status {
                key: BUDGET_WARNING_STATUS.to_string(),
                label: "context budget".to_string(),
                detail: Some(format!(
                    "{used} tokens used; warn at {threshold}; choose handoff path"
                )),
            },
        ])])
    }
}

#[async_trait::async_trait]
impl ModeSessionPlugin for RlmModeSession {
    async fn initialize_session(&self, _ctx: ModeSessionContext<'_>) -> Result<(), SessionError> {
        Ok(())
    }

    async fn restore_session(
        &self,
        _ctx: ModeSessionContext<'_>,
        state: &lash_core::runtime::PersistedSessionState,
    ) -> Result<(), SessionError> {
        let mut execution = self.execution.lock().await;
        let execution = execution
            .as_mut()
            .ok_or_else(|| SessionError::Protocol("RLM execution state is busy".to_string()))?;
        let protected_names = self.protected_projected_binding_names().await;
        if let Some(snapshot) = state.execution_state_snapshot().map(|bytes| bytes.to_vec()) {
            execution.restore_execution_state(&snapshot)?;
            execution.prune_protected_globals(&protected_names);
        }
        for event in state.read_view().active_events() {
            if let lash_core::SessionEventRecord::Mode(event) = event
                && let Some(event) = crate::decode_rlm_mode_event(event)
            {
                self.apply_seed_or_globals_event(execution, event, &protected_names)
                    .await?;
            }
        }
        Ok(())
    }

    async fn append_session_nodes(
        &self,
        _ctx: ModeSessionContext<'_>,
        nodes: &[lash_core::SessionAppendNode],
    ) -> Result<(), SessionError> {
        let mut execution = self.execution.lock().await;
        let execution = execution
            .as_mut()
            .ok_or_else(|| SessionError::Protocol("RLM execution state is busy".to_string()))?;
        let protected_names = self.protected_projected_binding_names().await;
        execution.prune_protected_globals(&protected_names);
        for node in nodes {
            if let lash_core::SessionAppendNode::ModeEvent { event } = node
                && let Some(event) = crate::decode_rlm_mode_event(event)
            {
                self.apply_seed_or_globals_event(execution, event, &protected_names)
                    .await?;
            }
        }
        Ok(())
    }

    async fn execute_code(
        &self,
        ctx: lash_core::ModeExecutionContext<'_>,
        request: lash_core::ExecRequest,
    ) -> Result<lash_core::ExecResponse, SessionError> {
        let session_projected_bindings = self.session_projected_bindings.lock().await.clone();
        let mut guard = self.execution.lock().await;
        let state = guard
            .take()
            .ok_or_else(|| SessionError::Protocol("RLM execution state is busy".to_string()))?;

        let result = execute_code(
            state,
            ctx,
            request,
            session_projected_bindings,
            Arc::clone(&self.projection_resolver),
        )
        .await;
        match result {
            Ok((state, response)) => {
                *guard = Some(state);
                Ok(response)
            }
            Err(err) => {
                *guard = Some(RlmExecutionState::new(
                    self.config.observe_projection.clone(),
                )?);
                Err(err)
            }
        }
    }

    async fn apply_session_extension(
        &self,
        extension: lash_core::ModeSessionExtensionHandle,
    ) -> Result<(), SessionError> {
        let extension = extension
            .as_any()
            .downcast_ref::<RlmProjectionExtension>()
            .ok_or_else(|| {
                SessionError::Protocol(
                    "RLM mode received an unsupported session extension".to_string(),
                )
            })?;
        reject_reserved_projected_binding_names(&extension.bindings)?;
        let mut guard = self.session_projected_bindings.lock().await;
        let merged = guard
            .clone()
            .merge(extension.bindings.clone())
            .map_err(|err| SessionError::Protocol(err.to_string()))?;
        *guard = merged;
        Ok(())
    }

    async fn validate_turn_extension(
        &self,
        extension: &lash_core::ModeTurnExtensionHandle,
    ) -> Result<(), SessionError> {
        let extension = extension
            .as_any()
            .downcast_ref::<RlmProjectionExtension>()
            .ok_or_else(|| {
                SessionError::Protocol(
                    "RLM mode received an unsupported turn extension".to_string(),
                )
            })?;
        reject_reserved_projected_binding_names(&extension.bindings)?;
        self.session_projected_bindings
            .lock()
            .await
            .clone()
            .merge(extension.bindings.clone())
            .map(|_| ())
            .map_err(|err| SessionError::Protocol(err.to_string()))
    }

    fn execution_state_dirty(&self) -> bool {
        self.execution
            .try_lock()
            .map(|execution| {
                execution
                    .as_ref()
                    .map(|execution| execution.execution_state_dirty())
                    .unwrap_or(true)
            })
            .unwrap_or(true)
    }

    async fn snapshot_execution_state(
        &self,
        _ctx: ModeSessionContext<'_>,
    ) -> Result<Option<Vec<u8>>, SessionError> {
        self.execution
            .lock()
            .await
            .as_mut()
            .ok_or_else(|| SessionError::Protocol("RLM execution state is busy".to_string()))?
            .snapshot_execution_state()
    }

    async fn restore_execution_state(
        &self,
        _ctx: ModeSessionContext<'_>,
        data: &[u8],
    ) -> Result<(), SessionError> {
        self.execution
            .lock()
            .await
            .as_mut()
            .ok_or_else(|| SessionError::Protocol("RLM execution state is busy".to_string()))?
            .restore_execution_state(data)
    }

    async fn before_llm_call(
        &self,
        ctx: ModeBeforeLlmCallContext,
        request: &LlmRequest,
    ) -> Result<Option<ModeLlmCallAction>, PluginError> {
        let Some(threshold) = self.config.continue_as_forced_fallback_tokens else {
            return Ok(None);
        };
        let Some(usage) = ctx.latest_prompt_usage.as_ref() else {
            return Ok(None);
        };
        if usage.context_budget_tokens < threshold {
            return Ok(None);
        }

        let observed_tokens = usage.context_budget_tokens;
        let args = forced_continue_as_args(&ctx, request, threshold, observed_tokens).await?;
        let seed = crate::RlmSeed::from_tool_args(&args)
            .map_err(|err| PluginError::Session(format!("forced continue_as {err}")))?;
        let referenced_handles =
            crate::control_tools::collect_seed_async_handle_ids(args.get("seed"));
        let referenced_handles_vec = referenced_handles.into_iter().collect::<Vec<_>>();
        ctx.host
            .validate_async_handles_visible(&ctx.session_id, &referenced_handles_vec)
            .await
            .map_err(|err| {
                PluginError::Session(format!(
                    "forced continue_as async handle validation failed: {err}"
                ))
            })?;
        let task = args
            .get("task")
            .and_then(serde_json::Value::as_str)
            .filter(|task| !task.trim().is_empty())
            .ok_or_else(|| {
                PluginError::Session(
                    "forced continue_as fallback returned missing or empty `task`".to_string(),
                )
            })?
            .to_string();

        let metadata = serde_json::Map::from_iter([
            (
                "trigger".to_string(),
                serde_json::Value::String("context_budget_tokens".to_string()),
            ),
            ("threshold_tokens".to_string(), serde_json::json!(threshold)),
            (
                "observed_tokens".to_string(),
                serde_json::json!(observed_tokens),
            ),
            (
                "source".to_string(),
                serde_json::Value::String("rlm_forced_fallback".to_string()),
            ),
        ]);
        let current_snapshot = ctx
            .host
            .snapshot_session(&ctx.session_id)
            .await
            .map_err(|err| {
                PluginError::Session(format!("failed to snapshot current session: {err}"))
            })?;
        let session_id = crate::control_tools::create_continue_as_successor(
            ctx.host.as_ref(),
            &ctx.session_id,
            current_snapshot,
            task,
            seed,
            crate::control_tools::ContinueAsHandoff {
                relation_reason: "continue_as_forced_context_fallback".to_string(),
                usage_source: "continue_as_forced_context_fallback".to_string(),
                metadata,
            },
        )
        .await
        .map_err(PluginError::Session)?;
        if let Err(err) = ctx
            .host
            .transfer_async_handles(&ctx.session_id, &session_id, &referenced_handles_vec)
            .await
        {
            let _ = ctx.host.close_session(&session_id).await;
            return Err(PluginError::Session(format!(
                "forced continue_as async handle transfer failed: {err}"
            )));
        }
        if let Err(err) = ctx
            .host
            .cancel_unreferenced_async_handles(&ctx.session_id, &referenced_handles_vec)
            .await
        {
            let _ = ctx.host.close_session(&session_id).await;
            return Err(PluginError::Session(format!(
                "forced continue_as async handle cleanup failed after successor creation: {err}"
            )));
        }
        Ok(Some(ModeLlmCallAction::Handoff { session_id }))
    }

    fn configure_runtime_from_request(
        &self,
        mut ctx: ModeRuntimeContext<'_>,
        request: &lash_core::SessionCreateRequest,
    ) {
        if let Ok(Some(extras)) = request
            .mode_extras
            .decode::<RlmCreateExtras>(&ExecutionMode::new("rlm"))
        {
            if let Ok(options) =
                lash_core::ModeTurnOptions::typed(ExecutionMode::new("rlm"), extras.termination)
            {
                ctx.set_mode_turn_options(options);
            }
        }
    }
}

impl RlmModeSession {
    async fn apply_seed_or_globals_event(
        &self,
        execution: &mut RlmExecutionState,
        event: RlmModeEvent,
        protected_names: &BTreeSet<String>,
    ) -> Result<(), SessionError> {
        match event {
            RlmModeEvent::RlmGlobalsPatch(patch) => {
                execution.patch_globals(&patch, protected_names)?;
            }
            RlmModeEvent::RlmSeed(seed) => {
                if !seed.projected.is_empty() {
                    self.install_initial_projected_seed(seed.projected)?;
                }
                if !seed.globals.is_empty() {
                    execution.patch_globals(
                        &RlmGlobalsPatchPluginBody {
                            set_default: seed.globals,
                        },
                        protected_names,
                    )?;
                }
            }
            RlmModeEvent::RlmTrajectoryEntry(_) | RlmModeEvent::RlmDiagnostic(_) => {}
        }
        Ok(())
    }

    /// Apply a projected-binding seed at session-creation time. Called from
    /// `configure_runtime_from_request` (sync); session_projected_bindings is
    /// guaranteed to be uncontended at this point because the session has just
    /// been constructed and no async path has touched it yet, so a sync
    /// `try_lock` is safe. We deliberately fail loudly on either an
    /// already-held lock or a duplicate-name merge so the bug shows up at the
    /// session boundary instead of silently producing a partial child.
    fn install_initial_projected_seed(
        &self,
        snapshot: lash_rlm_types::RlmProjectedSeedSnapshot,
    ) -> Result<(), SessionError> {
        let bindings = match RlmProjectedBindings::from_snapshot(&snapshot) {
            Ok(bindings) => bindings,
            Err(err) => {
                return Err(SessionError::Protocol(format!(
                    "rlm projected seed snapshot rejected: {err}"
                )));
            }
        };
        reject_reserved_projected_binding_names(&bindings)?;
        let mut guard = match self.session_projected_bindings.try_lock() {
            Ok(guard) => guard,
            Err(_) => return Err(SessionError::Protocol(
                "rlm projected seed snapshot could not be installed because session bindings were contended".to_string(),
            )),
        };
        let merged = guard
            .clone()
            .merge(bindings)
            .map_err(|err| SessionError::Protocol(err.to_string()))?;
        *guard = merged;
        Ok(())
    }
}

fn reject_reserved_projected_binding_names(
    bindings: &RlmProjectedBindings,
) -> Result<(), SessionError> {
    if bindings.names().any(|name| name == "history") {
        return Err(SessionError::Protocol(
            "`history` is reserved as an RLM built-in binding".to_string(),
        ));
    }
    Ok(())
}

async fn forced_continue_as_args(
    ctx: &ModeBeforeLlmCallContext,
    request: &LlmRequest,
    threshold: usize,
    observed_tokens: usize,
) -> Result<serde_json::Value, PluginError> {
    let mut fallback_request = request.clone();
    fallback_request.tools = Arc::new(Vec::new());
    fallback_request.tool_choice = LlmToolChoice::None;
    fallback_request.output_spec = Some(LlmOutputSpec::JsonSchema(LlmJsonSchema {
        name: "continue_as".to_string(),
        schema: crate::control_tools::continue_as_input_schema(),
        strict: true,
    }));
    fallback_request.stream_events = None;
    fallback_request.provider_trace = None;
    fallback_request.messages.push(LlmMessage::new(
        LlmRole::User,
        vec![LlmContentBlock::Text {
            text: forced_continue_as_instruction(threshold, observed_tokens).into(),
            response_meta: None,
            cache_breakpoint: false,
        }],
    ));
    let completion = ctx
        .host
        .direct_llm_completion(fallback_request, "continue_as_forced_context_fallback")
        .await
        .map_err(|err| {
            PluginError::Session(format!(
                "forced continue_as fallback LLM call failed: {err}"
            ))
        })?;
    parse_forced_continue_as_args(&completion.response.full_text)
}

fn forced_continue_as_instruction(threshold: usize, observed_tokens: usize) -> String {
    format!(
        "Context budget is above the forced continuation threshold ({observed_tokens} observed, threshold {threshold}). Produce fresh-context continuation arguments as JSON matching the provided schema. Set out the task at hand in `task`. Put only necessary state in `seed`, including only live async handles the successor must await. Live handles omitted from `seed` are not carried forward. Leave bulky logs, transcripts, raw command output, and repeated context behind. Prefer variable names, file paths, projected references, and compact summaries over copying large values. Omit `seed` when no extra state is needed."
    )
}

fn parse_forced_continue_as_args(text: &str) -> Result<serde_json::Value, PluginError> {
    let value: serde_json::Value = serde_json::from_str(text.trim()).map_err(|err| {
        PluginError::Session(format!(
            "forced continue_as fallback returned invalid JSON: {err}"
        ))
    })?;
    let validator =
        jsonschema::JSONSchema::compile(&crate::control_tools::continue_as_input_schema())
            .map_err(|err| {
                PluginError::Session(format!(
                    "failed to compile forced continue_as fallback schema: {err}"
                ))
            })?;
    if let Err(errors) = validator.validate(&value) {
        let messages = errors.map(|err| err.to_string()).collect::<Vec<_>>();
        return Err(PluginError::Session(format!(
            "forced continue_as fallback returned schema-invalid JSON: {}",
            messages.join("; ")
        )));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NoopPromptManager;

    #[async_trait::async_trait]
    impl lash_core::plugin::runtime_host::RuntimeSessionHost for NoopPromptManager {
        async fn snapshot_current(
            &self,
        ) -> Result<lash_core::PersistedSessionState, lash_core::plugin::PluginError> {
            Err(lash_core::plugin::PluginError::Session(
                "not used".to_string(),
            ))
        }

        async fn snapshot_session(
            &self,
            _session_id: &str,
        ) -> Result<lash_core::PersistedSessionState, lash_core::plugin::PluginError> {
            Err(lash_core::plugin::PluginError::Session(
                "not used".to_string(),
            ))
        }
        async fn tool_catalog(
            &self,
            _session_id: &str,
        ) -> Result<Vec<serde_json::Value>, lash_core::plugin::PluginError> {
            Ok(Vec::new())
        }

        async fn create_session(
            &self,
            _request: lash_core::SessionCreateRequest,
        ) -> Result<lash_core::SessionHandle, lash_core::plugin::PluginError> {
            Err(lash_core::plugin::PluginError::Session(
                "not used".to_string(),
            ))
        }

        async fn close_session(
            &self,
            _session_id: &str,
        ) -> Result<(), lash_core::plugin::PluginError> {
            Ok(())
        }
    }
    fn prompt_usage(context_budget_tokens: usize) -> lash_core::PromptUsage {
        lash_core::PromptUsage {
            prompt_context_tokens: context_budget_tokens,
            input_tokens: context_budget_tokens,
            cached_input_tokens: 0,
            context_budget_tokens,
        }
    }

    #[test]
    fn rlm_config_defaults_budget_thresholds() {
        let config = RlmModePluginConfig::default();

        assert_eq!(config.continue_as_soft_warn_tokens, Some(100_000));
        assert_eq!(config.continue_as_forced_fallback_tokens, Some(140_000));
        config.validate().expect("default config should validate");
    }

    #[test]
    fn rlm_config_validation_accepts_equal_and_disabled_thresholds() {
        RlmModePluginConfig {
            continue_as_soft_warn_tokens: Some(100_000),
            continue_as_forced_fallback_tokens: Some(100_000),
            ..Default::default()
        }
        .validate()
        .expect("equal thresholds are allowed");

        RlmModePluginConfig {
            continue_as_soft_warn_tokens: None,
            continue_as_forced_fallback_tokens: Some(80_000),
            ..Default::default()
        }
        .validate()
        .expect("soft warning can be disabled independently");

        RlmModePluginConfig {
            continue_as_soft_warn_tokens: Some(100_000),
            continue_as_forced_fallback_tokens: None,
            ..Default::default()
        }
        .validate()
        .expect("forced fallback can be disabled independently");
    }

    #[test]
    fn rlm_config_validation_rejects_forced_below_soft_warn() {
        let err = RlmModePluginConfig {
            continue_as_soft_warn_tokens: Some(100_000),
            continue_as_forced_fallback_tokens: Some(99_999),
            ..Default::default()
        }
        .validate()
        .expect_err("forced threshold below soft warn should fail");

        assert!(err.contains("continue_as_forced_fallback_tokens"));
    }

    #[test]
    fn forced_fallback_accepts_omitted_seed() {
        let parsed = parse_forced_continue_as_args(r#"{"task":"continue from compact state"}"#)
            .expect("valid fallback args");

        assert_eq!(
            parsed.get("task").and_then(serde_json::Value::as_str),
            Some("continue from compact state")
        );
        assert!(parsed.get("seed").is_none());
    }

    #[test]
    fn forced_fallback_rejects_schema_extra_fields() {
        let err = parse_forced_continue_as_args(
            r#"{"task":"continue from compact state","unexpected":true}"#,
        )
        .expect_err("extra properties should fail");

        assert!(err.to_string().contains("schema-invalid"));
    }

    fn projected(value: serde_json::Value) -> serde_json::Value {
        serde_json::json!({ "__projected__": value })
    }

    fn received_tool_args(
        policy: lash_core::ToolArgumentProjectionPolicy,
        args: serde_json::Value,
    ) -> serde_json::Value {
        crate::projection_codec::normalize_tool_args_for_projection(args, &policy)
    }

    fn materializing_args(args: serde_json::Value) -> serde_json::Value {
        received_tool_args(
            lash_core::ToolArgumentProjectionPolicy::MaterializeProjectedValues,
            args,
        )
    }

    fn seed_preserving_args(args: serde_json::Value) -> serde_json::Value {
        received_tool_args(
            lash_core::ToolArgumentProjectionPolicy::preserve_projected_refs_in_field("seed"),
            args,
        )
    }

    fn classify_received_seed(received: &serde_json::Value) -> crate::RlmSeed {
        crate::RlmSeed::from_tool_args(received).expect("seed should classify")
    }

    #[test]
    fn projected_tool_arg_normalization_materializes_ordinary_tools_recursively() {
        let args = serde_json::json!({
            "path": projected(serde_json::json!("/tmp/projected.txt")),
            "nested": {
                "items": [
                    projected(serde_json::json!("a")),
                    { "plain": projected(serde_json::json!(true)) }
                ]
            }
        });

        let normalized = materializing_args(args);

        assert_eq!(
            normalized,
            serde_json::json!({
                "path": "/tmp/projected.txt",
                "nested": {
                    "items": [
                        "a",
                        { "plain": true }
                    ]
                }
            })
        );
    }

    #[test]
    fn projected_tool_arg_normalization_preserves_seed_roots_for_projection_aware_tools() {
        let args = serde_json::json!({
            "task": projected(serde_json::json!("inspect the file")),
            "capability": "explore",
            "seed": {
                "projected_root": projected(serde_json::json!("carry-over")),
                "computed_record": {
                    "field": projected(serde_json::json!("materialize me"))
                }
            }
        });

        let normalized = seed_preserving_args(args);

        assert_eq!(
            normalized,
            serde_json::json!({
                "task": "inspect the file",
                "capability": "explore",
                "seed": {
                    "projected_root": { "__projected__": "carry-over" },
                    "computed_record": {
                        "field": "materialize me"
                    }
                }
            })
        );
    }

    #[test]
    fn projected_tool_arg_normalization_preserves_continue_as_seed_roots() {
        let args = serde_json::json!({
            "task": projected(serde_json::json!("continue")),
            "seed": {
                "problem": projected(serde_json::json!({ "prompt": "large prompt" }))
            }
        });

        let normalized = seed_preserving_args(args);
        let seed = classify_received_seed(&normalized);

        assert_eq!(
            normalized.get("task").and_then(serde_json::Value::as_str),
            Some("continue")
        );
        assert_eq!(
            seed.projected.entries.as_slice(),
            &[(
                "problem".to_string(),
                serde_json::json!({ "prompt": "large prompt" })
            )]
        );
        assert!(seed.globals.is_empty());
    }

    #[test]
    fn ordinary_tool_receives_non_projected_input_without_materialization() {
        let args = serde_json::json!({
            "query": "plain",
            "options": { "limit": 3, "exact": true }
        });

        let received = materializing_args(args.clone());

        assert_eq!(received, args);
    }

    #[test]
    fn ordinary_tool_receives_projected_input_materialized_as_plain_json() {
        let received = materializing_args(serde_json::json!({
            "query": projected(serde_json::json!("lazy query")),
            "options": {
                "limit": projected(serde_json::json!(3)),
                "filters": [
                    projected(serde_json::json!("rust")),
                    "tests"
                ]
            }
        }));

        assert_eq!(
            received,
            serde_json::json!({
                "query": "lazy query",
                "options": {
                    "limit": 3,
                    "filters": ["rust", "tests"]
                }
            })
        );
    }

    #[test]
    fn projection_aware_tool_receives_non_projected_seed_as_plain_input() {
        let received = seed_preserving_args(serde_json::json!({
            "task": "continue from facts",
            "capability": "explore",
            "seed": {
                "facts": { "count": 2 },
                "label": "plain"
            }
        }));

        let seed = classify_received_seed(&received);

        assert!(seed.projected.is_empty());
        assert_eq!(
            seed.globals,
            serde_json::Map::from_iter([
                ("facts".to_string(), serde_json::json!({ "count": 2 })),
                ("label".to_string(), serde_json::json!("plain")),
            ])
        );
    }

    #[test]
    fn projection_aware_tool_receives_projected_seed_roots_without_materializing_them() {
        let received = seed_preserving_args(serde_json::json!({
            "task": projected(serde_json::json!("continue from projected context")),
            "capability": "explore",
            "seed": {
                "problem": projected(serde_json::json!("large parent context")),
                "computed": {
                    "summary": projected(serde_json::json!("short summary"))
                }
            }
        }));

        let seed = classify_received_seed(&received);

        assert_eq!(
            received.get("task").and_then(serde_json::Value::as_str),
            Some("continue from projected context")
        );
        assert_eq!(
            seed.projected.entries.as_slice(),
            &[(
                "problem".to_string(),
                serde_json::json!("large parent context")
            )]
        );
        assert_eq!(
            seed.globals,
            serde_json::Map::from_iter([(
                "computed".to_string(),
                serde_json::json!({ "summary": "short summary" })
            )])
        );
    }

    #[test]
    fn projection_policy_cutover_has_no_name_based_projection_checks() {
        let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let plugin_src = std::fs::read_to_string(manifest_dir.join("src/plugin.rs"))
            .expect("read plugin source");
        let executor_src = std::fs::read_to_string(manifest_dir.join("src/executor.rs"))
            .expect("read executor source");
        let host_bridge_src =
            std::fs::read_to_string(manifest_dir.join("src/executor/host_bridge.rs"))
                .expect("read host bridge source");
        let projected_bindings_src =
            std::fs::read_to_string(manifest_dir.join("src/projected_bindings.rs"))
                .expect("read projected bindings source");

        let old_hook_call = ["normalize_tool_args_for_projection", "(&ctx.tool_name"].concat();
        let old_name_match = [
            "matches!",
            "(tool_name, ",
            "\"continue_as\" | \"spawn_agent\")",
        ]
        .concat();
        let old_invalid_ref = [
            "ProjectedBindingError::duplicate(",
            "\"invalid_projection_ref\")",
        ]
        .concat();

        assert!(!plugin_src.contains(&old_hook_call));
        assert!(!executor_src.contains(&old_name_match));
        assert!(!host_bridge_src.contains(&old_name_match));
        assert!(!projected_bindings_src.contains(&old_invalid_ref));
    }

    #[test]
    fn budget_prompt_contribution_below_advisory_floor_emits_status_only() {
        // 23% of the configured handoff threshold — emit the status line so
        // the model has continuous context-size awareness, but no
        // `continue_as` nag.
        let usage = prompt_usage(47_213);
        let content = format_budget_suffix(0, Some(&usage), Some(200_000))
            .expect("budget suffix should render");

        assert!(content.contains("Tokens: 47213 · handoff threshold: 200000 (23%)"));
        assert!(content.contains("Turn:"));
        assert!(!content.contains("Look for a clean handoff point"));
        assert!(!content.contains("Budget tight"));
        assert!(!content.contains("Past the handoff threshold"));
    }

    #[tokio::test]
    async fn session_projection_extension_rejects_duplicate_names() {
        let session = RlmModeSession::new(
            RlmModePluginConfig::default(),
            Arc::new(ProjectionRegistry::default()),
        )
        .expect("session should build");
        session
            .apply_session_extension(crate::rlm_session_projection_extension(
                RlmProjectedBindings::new()
                    .bind_json("current_query", serde_json::json!("first"))
                    .expect("first bind"),
            ))
            .await
            .expect("first projection");

        let duplicate = session
            .apply_session_extension(crate::rlm_session_projection_extension(
                RlmProjectedBindings::new()
                    .bind_json("current_query", serde_json::json!("second"))
                    .expect("second bind"),
            ))
            .await;
        let Err(err) = duplicate else {
            panic!("duplicate session projection should fail");
        };
        assert!(err.to_string().contains("current_query"));
    }

    #[tokio::test]
    async fn session_projection_prompt_contribution_lists_names() {
        let session = RlmModeSession::new(
            RlmModePluginConfig::default(),
            Arc::new(ProjectionRegistry::default()),
        )
        .expect("session should build");
        session
            .apply_session_extension(crate::rlm_session_projection_extension(
                RlmProjectedBindings::new()
                    .bind_json("current_query", serde_json::json!("first"))
                    .expect("bind"),
            ))
            .await
            .expect("projection");

        let contributions = session.projected_binding_prompt_contributions().await;
        assert_eq!(contributions.len(), 1);
        assert!(contributions[0].content.contains("`current_query`"));
        assert!(contributions[0].content.contains("Readonly: true"));
    }

    #[test]
    fn budget_prompt_contribution_advisory_tier_60_to_89_pct() {
        // 75% of threshold — advisory: scout for a clean handoff point.
        let usage = prompt_usage(75_000);
        let content = format_budget_suffix(0, Some(&usage), Some(100_000))
            .expect("budget suffix should render");

        assert!(content.contains("Tokens: 75000 · handoff threshold: 100000 (75%)"));
        assert!(content.contains("Look for a clean handoff point"));
        assert!(!content.contains("Budget tight"));
        assert!(!content.contains("Past the handoff threshold"));
    }

    #[test]
    fn budget_prompt_contribution_tight_tier_90_to_99_pct() {
        // 95% of threshold — tight: wrap the current step, then continue_as.
        let usage = prompt_usage(95_000);
        let content = format_budget_suffix(0, Some(&usage), Some(100_000))
            .expect("budget suffix should render");

        assert!(content.contains("Tokens: 95000 · handoff threshold: 100000 (95%)"));
        assert!(content.contains("Budget tight"));
        assert!(content.contains("`continue_as`"));
        assert!(!content.contains("Past the handoff threshold"));
        assert!(!content.contains("Look for a clean handoff point"));
    }

    #[test]
    fn budget_prompt_contribution_over_threshold_forces_handoff() {
        let usage = prompt_usage(120_292);
        let content = format_budget_suffix(0, Some(&usage), Some(100_000))
            .expect("budget suffix should render");

        assert!(content.contains("Tokens: 120292 · handoff threshold: 100000 (120%)"));
        assert!(content.contains("Past the handoff threshold"));
        assert!(content.contains("End this block with `continue_as` now"));
        assert!(content.contains("do not call `submit`"));
        assert!(content.contains("`task` + `seed`"));
    }

    #[test]
    fn soft_budget_warning_emits_surface_event_not_user_message() {
        let session = RlmModeSession::new(
            RlmModePluginConfig {
                continue_as_soft_warn_tokens: Some(100_000),
                ..Default::default()
            },
            Arc::new(ProjectionRegistry::default()),
        )
        .expect("rlm mode session");
        let state = lash_core::SessionStateEnvelope {
            token_usage: lash_core::TokenUsage {
                input_tokens: 120_292,
                ..Default::default()
            },
            ..Default::default()
        };
        let directives = session
            .soft_warn_directives(lash_core::plugin::CheckpointHookContext {
                session_id: "root".to_string(),
                checkpoint: lash_core::CheckpointKind::AfterWork,
                state: lash_core::SessionReadView::from_exported_state(&state),
                host: std::sync::Arc::new(NoopPromptManager),
            })
            .expect("warning directives");

        assert_eq!(directives.len(), 1);
        let lash_core::plugin::PluginDirective::EmitRuntimeEvents { events } = &directives[0]
        else {
            panic!("budget warning must be a runtime event, not an injected message");
        };
        assert_eq!(events.len(), 1);
        let lash_core::PluginRuntimeEvent::Status { key, label, detail } = &events[0] else {
            panic!("budget warning should use a typed status runtime event");
        };
        assert_eq!(key, BUDGET_WARNING_STATUS);
        assert_eq!(label, "context budget");
        assert!(detail.as_deref().is_some_and(|text| {
            text.contains("120292 tokens used") && text.contains("choose handoff path")
        }));
    }

    #[test]
    fn budget_prompt_contribution_omits_without_configured_budget() {
        let usage = prompt_usage(47_213);

        assert!(format_budget_suffix(0, Some(&usage), None).is_none());
    }

    #[test]
    fn budget_prompt_contribution_omits_without_used_tokens() {
        let usage = prompt_usage(0);

        assert!(format_budget_suffix(0, Some(&usage), Some(200_000)).is_none());
    }
}
