use crate::support::*;

pub use lash_core::{AssistantOutput, TurnIssue};

/// The two internal event sinks threaded through the turn-execution helpers.
///
/// `events` is the raw `SessionEvent` stream (the lower-level escape hatch,
/// reachable from app code only via [`TurnBuilder::advanced`]); `turn_events`
/// is the semantic [`TurnActivity`] stream used by the primary builder API.
/// Bundling them keeps the internal turn fns to a single sink parameter.
#[derive(Clone, Copy, Default)]
pub(crate) struct TurnSinks<'a> {
    events: Option<&'a dyn EventSink>,
    turn_events: Option<&'a dyn TurnActivitySink>,
}

impl<'a> TurnSinks<'a> {
    pub(crate) fn turn(events: &'a dyn TurnActivitySink) -> Self {
        Self {
            events: None,
            turn_events: Some(events),
        }
    }

    pub(crate) fn session(events: &'a dyn EventSink) -> Self {
        Self {
            events: Some(events),
            turn_events: None,
        }
    }

    fn events(&self) -> Option<&'a dyn EventSink> {
        self.events
    }

    fn turn_events(&self) -> Option<&'a dyn TurnActivitySink> {
        self.turn_events
    }
}

pub struct TurnBuilder {
    pub(crate) runtime: RuntimeHandle,
    pub(crate) effect_host: Arc<dyn EffectHost>,
    pub(crate) active_plugins: Vec<ActivePluginBinding>,
    pub(crate) input: TurnInput,
    pub(crate) cancel: CancellationToken,
    pub(crate) protocol_turn_options: Option<ProtocolTurnOptions>,
    pub(crate) provider: Option<ProviderHandle>,
    pub(crate) model: Option<ModelSpec>,
    pub(crate) turn_id: Option<String>,
}

impl TurnBuilder {
    pub fn cancel(mut self, cancel: CancellationToken) -> Self {
        self.cancel = cancel;
        self
    }

    pub fn protocol_turn_options(mut self, options: ProtocolTurnOptions) -> Self {
        self.protocol_turn_options = Some(options);
        self
    }

    pub fn provider(mut self, provider: ProviderHandle) -> Self {
        self.provider = Some(provider);
        self
    }

    pub fn model(mut self, model: ModelSpec) -> Self {
        self.model = Some(model);
        self
    }

    pub fn turn_id(mut self, id: impl Into<String>) -> Self {
        self.turn_id = Some(id.into());
        self
    }

    pub fn prompt_template(mut self, template: PromptTemplate) -> Self {
        self.input.turn_context.set_prompt_template(template);
        self
    }

    pub fn prompt_contribution(mut self, contribution: PromptContribution) -> Self {
        self.input
            .turn_context
            .add_prompt_contribution(contribution);
        self
    }

    pub fn replace_prompt_slot(
        mut self,
        slot: PromptSlot,
        contributions: impl IntoIterator<Item = PromptContribution>,
    ) -> Self {
        self.input
            .turn_context
            .replace_prompt_slot(slot, contributions);
        self
    }

    pub fn clear_prompt_slot(mut self, slot: PromptSlot) -> Self {
        self.input.turn_context.clear_prompt_slot(slot);
        self
    }

    pub fn prompt_layer(mut self, layer: PromptLayer) -> Self {
        self.input.turn_context.set_prompt_layer(layer);
        self
    }

    /// Attach typed per-turn input for an activated plugin binding.
    ///
    /// This is the generic primitive. Plugin crates should usually wrap it in a
    /// domain extension trait such as `.with_tone(tone)` or `.with_board(board)`
    /// so application code stays typed in its own vocabulary.
    pub fn with_plugin_input<P: PluginBinding>(mut self, input: P::Input) -> Self {
        self.input.turn_context.insert_plugin_input(P::ID, input);
        self
    }

    pub fn effects(self, controller: &dyn RuntimeEffectController) -> ScopedTurnBuilder<'_> {
        ScopedTurnBuilder {
            builder: self,
            controller,
        }
    }

    pub async fn run(self) -> Result<TurnOutput> {
        let collector = RunActivityCollector::default();
        let result = self.stream_to(&collector).await?;
        Ok(TurnOutput {
            result,
            activities: collector.into_activities(),
        })
    }

    pub async fn stream_to(self, events: &dyn TurnActivitySink) -> Result<TurnResult> {
        let effect_host = Arc::clone(&self.effect_host);
        reject_configured_durable_effect_host(effect_host.as_ref(), "turn")?;
        self.stream_to_with_effect_host(events, effect_host.as_ref())
            .await
    }

    pub fn stream(self) -> Result<TurnStream> {
        let effect_host = Arc::clone(&self.effect_host);
        reject_configured_durable_effect_host(effect_host.as_ref(), "turn stream")?;
        self.stream_with_effect_host(effect_host.as_ref())
    }

    /// Access lower-level turn execution that bypasses the semantic
    /// [`TurnActivity`] tier.
    pub fn advanced(self) -> AdvancedTurn {
        AdvancedTurn { builder: self }
    }

    fn resolved_turn_id(&self) -> String {
        self.turn_id
            .clone()
            .or_else(|| self.input.trace_turn_id.clone())
            .unwrap_or_else(fresh_turn_id)
    }

    fn turn_scope(&self, turn_id: &str) -> lash_core::EffectScope {
        lash_core::EffectScope::turn(self.runtime.observe().session_id(), turn_id)
    }

    pub(crate) fn prepare(
        mut self,
        trace_turn_id: Option<String>,
    ) -> Result<(RuntimeHandle, TurnInput, CancellationToken)> {
        if let Some(options) = self.protocol_turn_options {
            self.input.protocol_turn_options = Some(options);
        }
        if let Some(provider) = self.provider {
            self.input.turn_context.set_provider(provider);
        }
        if let Some(model) = self.model {
            self.input.turn_context.set_model(model);
        }
        if let Some(trace_turn_id) = trace_turn_id {
            self.input.trace_turn_id = Some(trace_turn_id);
        }
        validate_required_plugin_inputs(&self.active_plugins, &self.input)?;
        Ok((self.runtime, self.input, self.cancel))
    }

    async fn stream_to_with_effect_host(
        self,
        events: &dyn TurnActivitySink,
        effect_host: &dyn EffectHost,
    ) -> Result<TurnResult> {
        let turn_id = self.resolved_turn_id();
        let scoped_effect_controller = effect_host.scoped(self.turn_scope(&turn_id))?;
        self.stream_to_with_scope(events, scoped_effect_controller, Some(turn_id))
            .await
    }

    async fn stream_to_with_effect_controller(
        self,
        events: &dyn TurnActivitySink,
        controller: &dyn RuntimeEffectController,
    ) -> Result<TurnResult> {
        let turn_id = self.resolved_turn_id();
        let scoped_effect_controller =
            ScopedEffectController::borrowed(controller, self.turn_scope(&turn_id))?;
        self.stream_to_with_scope(events, scoped_effect_controller, Some(turn_id))
            .await
    }

    async fn stream_to_with_scope(
        self,
        events: &dyn TurnActivitySink,
        scoped_effect_controller: ScopedEffectController<'_>,
        trace_turn_id: Option<String>,
    ) -> Result<TurnResult> {
        let (runtime, input, cancel) = self.prepare(trace_turn_id)?;
        stream_prepared_turn(
            &runtime,
            input,
            TurnSinks::turn(events),
            scoped_effect_controller,
            cancel,
        )
        .await
    }

    fn stream_with_effect_host(self, effect_host: &dyn EffectHost) -> Result<TurnStream> {
        let turn_id = self.resolved_turn_id();
        let scoped_effect_controller = effect_host
            .scoped_static(self.turn_scope(&turn_id))?
            .ok_or(EmbedError::StaticTurnStreamRequiresStaticEffectHost)?;
        self.stream_with_scope(scoped_effect_controller, Some(turn_id))
    }

    fn stream_with_scope(
        self,
        scoped_effect_controller: ScopedEffectController<'static>,
        trace_turn_id: Option<String>,
    ) -> Result<TurnStream> {
        let (runtime, input, cancel) = self.prepare(trace_turn_id)?;
        let (tx, rx) = mpsc::channel(64);
        let sink = ChannelTurnActivitySink { tx };
        let completion = tokio::spawn(async move {
            stream_prepared_turn(
                &runtime,
                input,
                TurnSinks::turn(&sink),
                scoped_effect_controller,
                cancel,
            )
            .await
        });
        Ok(TurnStream {
            activities: rx,
            completion,
        })
    }
}

pub struct ScopedTurnBuilder<'run> {
    builder: TurnBuilder,
    controller: &'run dyn RuntimeEffectController,
}

impl<'run> ScopedTurnBuilder<'run> {
    pub fn cancel(mut self, cancel: CancellationToken) -> Self {
        self.builder = self.builder.cancel(cancel);
        self
    }

    pub fn protocol_turn_options(mut self, options: ProtocolTurnOptions) -> Self {
        self.builder = self.builder.protocol_turn_options(options);
        self
    }

    pub fn provider(mut self, provider: ProviderHandle) -> Self {
        self.builder = self.builder.provider(provider);
        self
    }

    pub fn model(mut self, model: ModelSpec) -> Self {
        self.builder = self.builder.model(model);
        self
    }

    pub fn turn_id(mut self, id: impl Into<String>) -> Self {
        self.builder = self.builder.turn_id(id);
        self
    }

    pub fn prompt_template(mut self, template: PromptTemplate) -> Self {
        self.builder = self.builder.prompt_template(template);
        self
    }

    pub fn prompt_contribution(mut self, contribution: PromptContribution) -> Self {
        self.builder = self.builder.prompt_contribution(contribution);
        self
    }

    pub fn replace_prompt_slot(
        mut self,
        slot: PromptSlot,
        contributions: impl IntoIterator<Item = PromptContribution>,
    ) -> Self {
        self.builder = self.builder.replace_prompt_slot(slot, contributions);
        self
    }

    pub fn clear_prompt_slot(mut self, slot: PromptSlot) -> Self {
        self.builder = self.builder.clear_prompt_slot(slot);
        self
    }

    pub fn prompt_layer(mut self, layer: PromptLayer) -> Self {
        self.builder = self.builder.prompt_layer(layer);
        self
    }

    pub fn with_plugin_input<P: PluginBinding>(mut self, input: P::Input) -> Self {
        self.builder = self.builder.with_plugin_input::<P>(input);
        self
    }

    pub async fn run(self) -> Result<TurnOutput> {
        let collector = RunActivityCollector::default();
        let result = self.stream_to(&collector).await?;
        Ok(TurnOutput {
            result,
            activities: collector.into_activities(),
        })
    }

    pub async fn stream_to(self, events: &dyn TurnActivitySink) -> Result<TurnResult> {
        self.builder
            .stream_to_with_effect_controller(events, self.controller)
            .await
    }
}

/// Lower-level turn execution that exposes the raw `SessionEvent` stream.
///
/// Reachable via [`TurnBuilder::advanced`]. Most applications should use
/// [`TurnBuilder::stream_to`] for semantic turn activity; benchmarks and
/// diagnostics use this when they need the same session-event stream as the
/// lower-level runtime trace.
pub struct AdvancedTurn {
    builder: TurnBuilder,
}

impl AdvancedTurn {
    pub async fn run_with_scope(
        self,
        scoped_effect_controller: ScopedEffectController<'_>,
    ) -> Result<TurnOutput> {
        let collector = RunActivityCollector::default();
        let result = self
            .stream_to_with_scope(&collector, scoped_effect_controller)
            .await?;
        Ok(TurnOutput {
            result,
            activities: collector.into_activities(),
        })
    }

    pub async fn collect_with_scope(
        self,
        events: &dyn TurnActivitySink,
        scoped_effect_controller: ScopedEffectController<'_>,
    ) -> Result<TurnOutput> {
        let collector = RunActivityCollector::default();
        let fanout = BorrowedTurnActivityFanout {
            live: events,
            collector: &collector,
        };
        let result = self
            .stream_to_with_scope(&fanout, scoped_effect_controller)
            .await?;
        Ok(TurnOutput {
            result,
            activities: collector.into_activities(),
        })
    }

    pub async fn stream_to_with_scope(
        self,
        events: &dyn TurnActivitySink,
        scoped_effect_controller: ScopedEffectController<'_>,
    ) -> Result<TurnResult> {
        let trace_turn_id = trace_turn_id_for_scope(&self.builder, &scoped_effect_controller);
        self.builder
            .stream_to_with_scope(events, scoped_effect_controller, trace_turn_id)
            .await
    }

    pub fn stream_with_scope(
        self,
        scoped_effect_controller: ScopedEffectController<'static>,
    ) -> Result<TurnStream> {
        let trace_turn_id = trace_turn_id_for_scope(&self.builder, &scoped_effect_controller);
        self.builder
            .stream_with_scope(scoped_effect_controller, trace_turn_id)
    }

    /// Run the turn while sending raw session events to `events`.
    pub async fn collect_session_events_with_scope(
        self,
        events: &dyn EventSink,
        scoped_effect_controller: ScopedEffectController<'_>,
    ) -> Result<TurnResult> {
        let trace_turn_id = trace_turn_id_for_scope(&self.builder, &scoped_effect_controller);
        let (runtime, input, cancel) = self.builder.prepare(trace_turn_id)?;
        stream_prepared_turn(
            &runtime,
            input,
            TurnSinks::session(events),
            scoped_effect_controller,
            cancel,
        )
        .await
    }
}

pub struct TurnStream {
    activities: mpsc::Receiver<Result<TurnActivity>>,
    completion: JoinHandle<Result<TurnResult>>,
}

impl TurnStream {
    pub async fn next_activity(&mut self) -> Option<Result<TurnActivity>> {
        self.activities.recv().await
    }

    pub async fn finish(self) -> Result<TurnResult> {
        self.completion.await.map_err(|err| {
            EmbedError::Runtime(lash_core::RuntimeError::new(
                RuntimeErrorCode::TurnStreamJoin,
                format!("turn stream task failed: {err}"),
            ))
        })?
    }
}

pub struct QueuedTurnBuilder {
    pub(crate) runtime: RuntimeHandle,
    pub(crate) effect_host: Arc<dyn EffectHost>,
    pub(crate) cancel: CancellationToken,
    pub(crate) batch_ids: Vec<String>,
    pub(crate) drain_id: Option<String>,
}

impl QueuedTurnBuilder {
    pub fn cancel(mut self, cancel: CancellationToken) -> Self {
        self.cancel = cancel;
        self
    }

    pub fn batch_ids(mut self, batch_ids: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.batch_ids = batch_ids.into_iter().map(Into::into).collect();
        self
    }

    pub fn drain_id(mut self, drain_id: impl Into<String>) -> Self {
        self.drain_id = Some(drain_id.into());
        self
    }

    pub fn effects(self, controller: &dyn RuntimeEffectController) -> ScopedQueuedTurnBuilder<'_> {
        ScopedQueuedTurnBuilder {
            builder: self,
            controller,
        }
    }

    pub async fn run(self) -> Result<Option<TurnOutput>> {
        let collector = RunActivityCollector::default();
        let Some(result) = self.stream_to(&collector).await? else {
            return Ok(None);
        };
        Ok(Some(TurnOutput {
            result,
            activities: collector.into_activities(),
        }))
    }

    pub async fn stream_to(self, events: &dyn TurnActivitySink) -> Result<Option<TurnResult>> {
        let effect_host = Arc::clone(&self.effect_host);
        reject_configured_durable_effect_host(effect_host.as_ref(), "queued turn")?;
        self.stream_to_with_effect_host(events, effect_host.as_ref())
            .await
    }

    fn resolved_drain_id(&self) -> String {
        self.drain_id
            .clone()
            .or_else(|| self.batch_ids.first().cloned())
            .unwrap_or_else(fresh_queue_drain_id)
    }

    async fn stream_to_with_effect_host(
        self,
        events: &dyn TurnActivitySink,
        effect_host: &dyn EffectHost,
    ) -> Result<Option<TurnResult>> {
        let drain_id = self.resolved_drain_id();
        let scope =
            lash_core::EffectScope::queue_drain(self.runtime.observe().session_id(), drain_id);
        let scoped_effect_controller = effect_host.scoped(scope)?;
        self.stream_to_with_scope(events, scoped_effect_controller)
            .await
    }

    async fn stream_to_with_effect_controller(
        self,
        events: &dyn TurnActivitySink,
        controller: &dyn RuntimeEffectController,
    ) -> Result<Option<TurnResult>> {
        let drain_id = self.resolved_drain_id();
        let scope =
            lash_core::EffectScope::queue_drain(self.runtime.observe().session_id(), drain_id);
        let scoped_effect_controller = ScopedEffectController::borrowed(controller, scope)?;
        self.stream_to_with_scope(events, scoped_effect_controller)
            .await
    }

    async fn stream_to_with_scope(
        self,
        events: &dyn TurnActivitySink,
        scoped_effect_controller: ScopedEffectController<'_>,
    ) -> Result<Option<TurnResult>> {
        let Self {
            runtime,
            effect_host: _,
            cancel,
            batch_ids,
            drain_id: _,
        } = self;
        stream_next_queued_prepared_turn(
            &runtime,
            TurnSinks::turn(events),
            scoped_effect_controller,
            cancel,
            &batch_ids,
        )
        .await
    }
}

pub struct ScopedQueuedTurnBuilder<'run> {
    builder: QueuedTurnBuilder,
    controller: &'run dyn RuntimeEffectController,
}

impl<'run> ScopedQueuedTurnBuilder<'run> {
    pub fn cancel(mut self, cancel: CancellationToken) -> Self {
        self.builder = self.builder.cancel(cancel);
        self
    }

    pub fn batch_ids(mut self, batch_ids: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.builder = self.builder.batch_ids(batch_ids);
        self
    }

    pub fn drain_id(mut self, drain_id: impl Into<String>) -> Self {
        self.builder = self.builder.drain_id(drain_id);
        self
    }

    pub async fn run(self) -> Result<Option<TurnOutput>> {
        let collector = RunActivityCollector::default();
        let Some(result) = self.stream_to(&collector).await? else {
            return Ok(None);
        };
        Ok(Some(TurnOutput {
            result,
            activities: collector.into_activities(),
        }))
    }

    pub async fn stream_to(self, events: &dyn TurnActivitySink) -> Result<Option<TurnResult>> {
        self.builder
            .stream_to_with_effect_controller(events, self.controller)
            .await
    }
}

fn fresh_turn_id() -> String {
    lash_core::TurnActivityId::fresh().0
}

fn fresh_queue_drain_id() -> String {
    format!("queue-drain:{}", fresh_turn_id())
}

fn trace_turn_id_for_scope(
    builder: &TurnBuilder,
    scoped_effect_controller: &ScopedEffectController<'_>,
) -> Option<String> {
    if scoped_effect_controller
        .effect_scope()
        .validates_turn_trace_id()
    {
        Some(
            builder
                .turn_id
                .clone()
                .unwrap_or_else(|| scoped_effect_controller.scope_id().to_string()),
        )
    } else {
        builder
            .turn_id
            .clone()
            .or_else(|| builder.input.trace_turn_id.clone())
    }
}

fn reject_configured_durable_effect_host(
    effect_host: &dyn EffectHost,
    operation: &'static str,
) -> Result<()> {
    if effect_host.durability_tier() == DurabilityTier::Durable {
        return Err(EmbedError::DurableEffectHostRequiresHandlerContext { operation });
    }
    Ok(())
}

pub(crate) async fn stream_next_queued_prepared_turn(
    runtime: &RuntimeHandle,
    sinks: TurnSinks<'_>,
    scoped_effect_controller: ScopedEffectController<'_>,
    cancel: CancellationToken,
    batch_ids: &[String],
) -> Result<Option<TurnResult>> {
    let turn = Box::pin(stream_next_queued_prepared_assembled(
        runtime,
        sinks,
        scoped_effect_controller,
        cancel,
        batch_ids,
    ))
    .await?;
    Ok(turn.map(TurnResult::from_assembled))
}

pub(crate) async fn stream_next_queued_prepared_assembled(
    runtime: &RuntimeHandle,
    sinks: TurnSinks<'_>,
    scoped_effect_controller: ScopedEffectController<'_>,
    cancel: CancellationToken,
    batch_ids: &[String],
) -> Result<Option<AssembledTurn>> {
    let writer_handle = runtime.writer();
    let mut writer = writer_handle.lock().await;
    let observation_sink = SessionObservationTurnActivitySink {
        runtime: runtime.clone(),
        live: sinks.turn_events(),
    };
    let opts = turn_options(
        sinks.events(),
        &observation_sink,
        scoped_effect_controller,
        cancel,
    );
    let turn = if batch_ids.is_empty() {
        writer.stream_next_queued_work(opts).await?
    } else {
        writer.stream_selected_queued_work(opts, batch_ids).await?
    };
    runtime.publish_from(&writer);
    Ok(turn)
}

fn turn_options<'a>(
    events: Option<&'a dyn EventSink>,
    turn_events: &'a dyn TurnActivitySink,
    scoped_effect_controller: ScopedEffectController<'a>,
    cancel: CancellationToken,
) -> lash_core::TurnOptions<'a> {
    let mut opts = lash_core::TurnOptions::new(cancel, scoped_effect_controller);
    if let Some(events) = events {
        opts = opts.with_events(events);
    }
    opts.with_turn_events(turn_events)
}

struct SessionObservationTurnActivitySink<'a> {
    runtime: RuntimeHandle,
    live: Option<&'a dyn TurnActivitySink>,
}

#[async_trait]
impl TurnActivitySink for SessionObservationTurnActivitySink<'_> {
    fn is_noop(&self) -> bool {
        false
    }

    async fn emit(&self, activity: TurnActivity) {
        self.runtime.record_turn_activity(activity.clone());
        if let Some(live) = self.live {
            live.emit(activity).await;
        }
    }
}

struct ChannelTurnActivitySink {
    tx: mpsc::Sender<Result<TurnActivity>>,
}

#[async_trait]
impl TurnActivitySink for ChannelTurnActivitySink {
    async fn emit(&self, activity: TurnActivity) {
        let _ = self.tx.send(Ok(activity)).await;
    }
}
fn validate_required_plugin_inputs(
    active_plugins: &[ActivePluginBinding],
    input: &TurnInput,
) -> Result<()> {
    for plugin in active_plugins {
        if plugin.requires_turn_input && !input.turn_context.has_plugin_input(plugin.id) {
            return Err(EmbedError::MissingPluginTurnInput {
                plugin_id: plugin.id,
            });
        }
    }
    Ok(())
}

pub(crate) async fn stream_prepared_turn(
    runtime: &RuntimeHandle,
    input: TurnInput,
    sinks: TurnSinks<'_>,
    scoped_effect_controller: ScopedEffectController<'_>,
    cancel: CancellationToken,
) -> Result<TurnResult> {
    let turn = Box::pin(stream_prepared_assembled(
        runtime,
        input,
        sinks,
        scoped_effect_controller,
        cancel,
    ))
    .await?;
    Ok(TurnResult::from_assembled(turn))
}

pub(crate) async fn stream_prepared_assembled(
    runtime: &RuntimeHandle,
    input: TurnInput,
    sinks: TurnSinks<'_>,
    scoped_effect_controller: ScopedEffectController<'_>,
    cancel: CancellationToken,
) -> Result<AssembledTurn> {
    let turn = Box::pin(stream_prepared_agent_frame_run(
        runtime,
        input,
        sinks,
        scoped_effect_controller,
        cancel,
    ))
    .await?;
    turn.into_final_turn().ok_or_else(|| {
        EmbedError::Runtime(lash_core::RuntimeError::new(
            RuntimeErrorCode::EmptyAgentFrameRun,
            "runtime completed without an assembled turn",
        ))
    })
}

pub(crate) async fn stream_prepared_agent_frame_run(
    runtime: &RuntimeHandle,
    input: TurnInput,
    sinks: TurnSinks<'_>,
    scoped_effect_controller: ScopedEffectController<'_>,
    cancel: CancellationToken,
) -> Result<lash_core::AgentFrameRun> {
    let writer_handle = runtime.writer();
    let mut writer = writer_handle.lock().await;
    if let Some(extension) = input.protocol_extension.as_ref() {
        writer
            .validate_protocol_turn_extension(extension)
            .await
            .map_err(EmbedError::Session)?;
    }
    let observation_sink = SessionObservationTurnActivitySink {
        runtime: runtime.clone(),
        live: sinks.turn_events(),
    };
    let turn = Box::pin(writer.stream_turn_with_agent_frames(
        input,
        turn_options(
            sinks.events(),
            &observation_sink,
            scoped_effect_controller,
            cancel,
        ),
    ))
    .await?;
    runtime.publish_from(&writer);
    Ok(turn)
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TurnResult {
    pub state: SessionSnapshot,
    pub outcome: TurnOutcome,
    pub assistant_output: AssistantOutput,
    /// Parent's own LLM tokens for this turn. Does **not** include child
    /// sessions; see [`children_usage`](Self::children_usage) and
    /// [`total_usage`](Self::total_usage).
    pub usage: TokenUsage,
    /// Per-`(source, model)` ledger entries for child sessions whose LLM
    /// calls completed during this turn (subagents, compaction, observers,
    /// etc.). Empty unless the turn spawned children.
    #[serde(default)]
    pub children_usage: Vec<TokenLedgerEntry>,
    pub tool_calls: Vec<ToolCallRecord>,
    pub execution: ExecutionSummary,
    pub errors: Vec<TurnIssue>,
}

impl TurnResult {
    fn from_assembled(turn: lash_core::AssembledTurn) -> Self {
        Self {
            state: turn.state,
            outcome: turn.outcome,
            assistant_output: turn.assistant_output,
            usage: turn.token_usage,
            children_usage: turn.children_usage,
            tool_calls: turn.tool_calls,
            execution: turn.execution,
            errors: turn.errors,
        }
    }

    /// Sum of parent's own LLM tokens and every child session's LLM tokens
    /// for this turn.
    pub fn total_usage(&self) -> TokenUsage {
        let mut total = self.usage.clone();
        for entry in &self.children_usage {
            total.add(&entry.usage);
        }
        total
    }

    pub fn assistant_message(&self) -> Option<&str> {
        match &self.outcome {
            TurnOutcome::Finished(lash_core::TurnFinish::AssistantMessage { text }) => Some(text),
            _ => None,
        }
    }

    pub fn submitted_value(&self) -> Option<&serde_json::Value> {
        match &self.outcome {
            TurnOutcome::Finished(lash_core::TurnFinish::SubmittedValue { value }) => Some(value),
            _ => None,
        }
    }

    pub fn tool_value(&self) -> Option<(&str, &serde_json::Value)> {
        match &self.outcome {
            TurnOutcome::Finished(lash_core::TurnFinish::ToolValue { tool_name, value }) => {
                Some((tool_name.as_str(), value))
            }
            _ => None,
        }
    }

    pub fn is_success(&self) -> bool {
        matches!(
            self.outcome,
            TurnOutcome::Finished(_) | TurnOutcome::AgentFrameSwitch { .. }
        )
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TurnOutput {
    pub result: TurnResult,
    pub activities: Vec<TurnActivity>,
}

impl TurnOutput {
    pub fn assistant_message(&self) -> Option<&str> {
        self.result.assistant_message()
    }

    pub fn submitted_value(&self) -> Option<&serde_json::Value> {
        self.result.submitted_value()
    }

    pub fn tool_value(&self) -> Option<(&str, &serde_json::Value)> {
        self.result.tool_value()
    }

    pub fn is_success(&self) -> bool {
        self.result.is_success()
    }
}

struct BorrowedTurnActivityFanout<'a> {
    live: &'a dyn TurnActivitySink,
    collector: &'a RunActivityCollector,
}

#[async_trait]
impl TurnActivitySink for BorrowedTurnActivityFanout<'_> {
    async fn emit(&self, activity: TurnActivity) {
        self.live.emit(activity.clone()).await;
        self.collector.emit(activity).await;
    }
}

#[derive(Default)]
pub(crate) struct RunActivityCollector {
    activities: Arc<StdMutex<Vec<TurnActivity>>>,
}

impl RunActivityCollector {
    fn into_activities(self) -> Vec<TurnActivity> {
        self.activities
            .lock()
            .expect("run activity collector lock")
            .clone()
    }

    #[cfg(test)]
    pub(crate) fn snapshot(&self) -> Vec<TurnActivity> {
        self.activities
            .lock()
            .expect("run activity collector lock")
            .clone()
    }
}

#[async_trait]
impl TurnActivitySink for RunActivityCollector {
    async fn emit(&self, activity: TurnActivity) {
        self.activities
            .lock()
            .expect("run activity collector lock")
            .push(activity);
    }
}

pub struct TurnActivityFanout {
    sinks: Vec<Arc<dyn TurnActivitySink>>,
}

impl TurnActivityFanout {
    pub fn new(sinks: impl IntoIterator<Item = Arc<dyn TurnActivitySink>>) -> Self {
        Self {
            sinks: sinks.into_iter().collect(),
        }
    }
}

#[async_trait]
impl TurnActivitySink for TurnActivityFanout {
    async fn emit(&self, activity: TurnActivity) {
        for sink in &self.sinks {
            sink.emit(activity.clone()).await;
        }
    }
}

pub fn message_text(message: &Message) -> String {
    message
        .parts
        .iter()
        .map(|part| part.content.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn message_role(message: &Message) -> &'static str {
    match message.role {
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::System => "system",
        MessageRole::Event => "event",
    }
}
