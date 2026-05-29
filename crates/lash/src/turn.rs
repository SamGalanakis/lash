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
    pub(crate) active_plugins: Vec<ActivePluginBinding>,
    pub(crate) input: TurnInput,
    pub(crate) cancel: CancellationToken,
    pub(crate) protocol_turn_options: Option<ProtocolTurnOptions>,
    pub(crate) provider: Option<ProviderHandle>,
    pub(crate) model: Option<lash_core::ModelSpec>,
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

    pub fn model(mut self, model: lash_core::ModelSpec) -> Self {
        self.model = Some(model);
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

    pub fn require_submit(self) -> Result<Self> {
        self.rlm_termination(lash_rlm_types::RlmTermination::SubmitRequired { schema: None })
    }

    pub fn require_submit_schema(self, schema: serde_json::Value) -> Result<Self> {
        self.rlm_termination(lash_rlm_types::RlmTermination::SubmitRequired {
            schema: Some(schema),
        })
    }

    pub fn allow_prose_or_submit(self) -> Result<Self> {
        self.rlm_termination(lash_rlm_types::RlmTermination::ProseOrSubmit)
    }

    fn rlm_termination(self, termination: lash_rlm_types::RlmTermination) -> Result<Self> {
        Ok(self.protocol_turn_options(ProtocolTurnOptions::typed(termination)?))
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

    pub async fn run(self) -> Result<TurnOutput> {
        let collector = RunActivityCollector::default();
        let result = self.stream(&collector).await?;
        Ok(TurnOutput {
            result,
            activities: collector.into_activities(),
        })
    }

    pub async fn run_with_effect_scope(
        self,
        effect_scope: RuntimeEffectControllerScope<'_>,
    ) -> Result<TurnOutput> {
        let collector = RunActivityCollector::default();
        let result = self
            .stream_with_effect_scope(&collector, effect_scope)
            .await?;
        Ok(TurnOutput {
            result,
            activities: collector.into_activities(),
        })
    }

    pub async fn collect_with(self, events: &dyn TurnActivitySink) -> Result<TurnOutput> {
        let collector = RunActivityCollector::default();
        let fanout = BorrowedTurnActivityFanout {
            live: events,
            collector: &collector,
        };
        let result = self.stream(&fanout).await?;
        Ok(TurnOutput {
            result,
            activities: collector.into_activities(),
        })
    }

    /// Access lower-level turn execution that bypasses the semantic
    /// [`TurnActivity`] tier.
    pub fn advanced(self) -> AdvancedTurn {
        AdvancedTurn { builder: self }
    }

    pub(crate) fn prepare(mut self) -> Result<(RuntimeHandle, TurnInput, CancellationToken)> {
        if let Some(options) = self.protocol_turn_options {
            self.input.protocol_turn_options = Some(options);
        }
        if let Some(provider) = self.provider {
            self.input.turn_context.set_provider(provider);
        }
        if let Some(model) = self.model {
            self.input.turn_context.set_model(model);
        }
        validate_required_plugin_inputs(&self.active_plugins, &self.input)?;
        Ok((self.runtime, self.input, self.cancel))
    }

    pub async fn stream(self, events: &dyn TurnActivitySink) -> Result<TurnResult> {
        let (runtime, input, cancel) = self.prepare()?;
        stream_prepared_turn(&runtime, input, TurnSinks::turn(events), None, cancel).await
    }

    pub async fn stream_with_effect_scope(
        self,
        events: &dyn TurnActivitySink,
        effect_scope: RuntimeEffectControllerScope<'_>,
    ) -> Result<TurnResult> {
        let (runtime, input, cancel) = self.prepare()?;
        stream_prepared_turn(
            &runtime,
            input,
            TurnSinks::turn(events),
            Some(effect_scope),
            cancel,
        )
        .await
    }

    pub fn into_stream(self) -> Result<TurnStream> {
        let (runtime, input, cancel) = self.prepare()?;
        let (tx, rx) = mpsc::channel(64);
        let sink = ChannelTurnActivitySink { tx };
        let completion = tokio::spawn(async move {
            stream_prepared_turn(&runtime, input, TurnSinks::turn(&sink), None, cancel).await
        });
        Ok(TurnStream {
            activities: rx,
            completion,
        })
    }
}

/// Lower-level turn execution that exposes the raw `SessionEvent` stream.
///
/// Reachable via [`TurnBuilder::advanced`]. Most applications should use
/// [`TurnBuilder::collect_with`] for semantic turn activity; benchmarks and
/// diagnostics use this when they need the same session-event stream as the
/// lower-level runtime trace.
pub struct AdvancedTurn {
    builder: TurnBuilder,
}

impl AdvancedTurn {
    /// Run the turn while sending raw session events to `events`.
    pub async fn collect_session_events_with(self, events: &dyn EventSink) -> Result<TurnResult> {
        let (runtime, input, cancel) = self.builder.prepare()?;
        stream_prepared_turn(&runtime, input, TurnSinks::session(events), None, cancel).await
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

pub struct ResumeTurnBuilder {
    pub(crate) runtime: RuntimeHandle,
    pub(crate) turn_id: String,
    pub(crate) cancel: CancellationToken,
}

impl ResumeTurnBuilder {
    pub fn cancel(mut self, cancel: CancellationToken) -> Self {
        self.cancel = cancel;
        self
    }

    pub async fn run(self) -> Result<TurnOutput> {
        let collector = RunActivityCollector::default();
        let result = self.stream(&collector).await?;
        Ok(TurnOutput {
            result,
            activities: collector.into_activities(),
        })
    }

    pub async fn run_with_effect_scope(
        self,
        effect_scope: RuntimeEffectControllerScope<'_>,
    ) -> Result<TurnOutput> {
        let collector = RunActivityCollector::default();
        let result = self
            .stream_with_effect_scope(&collector, effect_scope)
            .await?;
        Ok(TurnOutput {
            result,
            activities: collector.into_activities(),
        })
    }

    pub async fn collect_with(self, events: &dyn TurnActivitySink) -> Result<TurnOutput> {
        let collector = RunActivityCollector::default();
        let fanout = BorrowedTurnActivityFanout {
            live: events,
            collector: &collector,
        };
        let result = self.stream(&fanout).await?;
        Ok(TurnOutput {
            result,
            activities: collector.into_activities(),
        })
    }

    pub async fn collect_with_effect_scope(
        self,
        events: &dyn TurnActivitySink,
        effect_scope: RuntimeEffectControllerScope<'_>,
    ) -> Result<TurnOutput> {
        let collector = RunActivityCollector::default();
        let fanout = BorrowedTurnActivityFanout {
            live: events,
            collector: &collector,
        };
        let result = self.stream_with_effect_scope(&fanout, effect_scope).await?;
        Ok(TurnOutput {
            result,
            activities: collector.into_activities(),
        })
    }

    pub async fn stream(self, events: &dyn TurnActivitySink) -> Result<TurnResult> {
        let assembled = resume_prepared_assembled(
            &self.runtime,
            &self.turn_id,
            TurnSinks::turn(events),
            None,
            self.cancel,
        )
        .await?;
        Ok(TurnResult::from_assembled(assembled))
    }

    pub async fn stream_with_effect_scope(
        self,
        events: &dyn TurnActivitySink,
        effect_scope: RuntimeEffectControllerScope<'_>,
    ) -> Result<TurnResult> {
        let assembled = resume_prepared_assembled(
            &self.runtime,
            &self.turn_id,
            TurnSinks::turn(events),
            Some(effect_scope),
            self.cancel,
        )
        .await?;
        Ok(TurnResult::from_assembled(assembled))
    }

    /// Access lower-level resume execution that bypasses the semantic
    /// [`TurnActivity`] tier.
    pub fn advanced(self) -> AdvancedResumeTurn {
        AdvancedResumeTurn { builder: self }
    }
}

/// Lower-level turn resume that exposes the raw `SessionEvent` stream.
///
/// Reachable via [`ResumeTurnBuilder::advanced`]; see [`AdvancedTurn`] for the
/// rationale.
pub struct AdvancedResumeTurn {
    builder: ResumeTurnBuilder,
}

impl AdvancedResumeTurn {
    pub async fn collect_session_events_with(self, events: &dyn EventSink) -> Result<TurnResult> {
        let assembled = resume_prepared_assembled(
            &self.builder.runtime,
            &self.builder.turn_id,
            TurnSinks::session(events),
            None,
            self.builder.cancel,
        )
        .await?;
        Ok(TurnResult::from_assembled(assembled))
    }

    pub async fn collect_session_events_with_effect_scope(
        self,
        events: &dyn EventSink,
        effect_scope: RuntimeEffectControllerScope<'_>,
    ) -> Result<TurnResult> {
        let assembled = resume_prepared_assembled(
            &self.builder.runtime,
            &self.builder.turn_id,
            TurnSinks::session(events),
            Some(effect_scope),
            self.builder.cancel,
        )
        .await?;
        Ok(TurnResult::from_assembled(assembled))
    }
}

pub struct QueuedTurnBuilder {
    pub(crate) runtime: RuntimeHandle,
    pub(crate) cancel: CancellationToken,
}

impl QueuedTurnBuilder {
    pub fn cancel(mut self, cancel: CancellationToken) -> Self {
        self.cancel = cancel;
        self
    }

    pub async fn run(self) -> Result<Option<TurnOutput>> {
        let collector = RunActivityCollector::default();
        let Some(result) = self.stream(&collector).await? else {
            return Ok(None);
        };
        Ok(Some(TurnOutput {
            result,
            activities: collector.into_activities(),
        }))
    }

    pub async fn stream(self, events: &dyn TurnActivitySink) -> Result<Option<TurnResult>> {
        stream_next_queued_prepared_turn(&self.runtime, TurnSinks::turn(events), self.cancel).await
    }
}

pub(crate) async fn resume_prepared_assembled(
    runtime: &RuntimeHandle,
    turn_id: &str,
    sinks: TurnSinks<'_>,
    effect_scope: Option<RuntimeEffectControllerScope<'_>>,
    cancel: CancellationToken,
) -> Result<AssembledTurn> {
    let writer_handle = runtime.writer();
    let mut writer = writer_handle.lock().await;
    let turn = writer
        .resume_turn(turn_id, turn_options(sinks, effect_scope, cancel))
        .await?;
    runtime.publish_from(&writer);
    Ok(turn)
}

pub(crate) async fn stream_next_queued_prepared_turn(
    runtime: &RuntimeHandle,
    sinks: TurnSinks<'_>,
    cancel: CancellationToken,
) -> Result<Option<TurnResult>> {
    let turn = Box::pin(stream_next_queued_prepared_assembled(
        runtime, sinks, cancel,
    ))
    .await?;
    Ok(turn.map(TurnResult::from_assembled))
}

pub(crate) async fn stream_next_queued_prepared_assembled(
    runtime: &RuntimeHandle,
    sinks: TurnSinks<'_>,
    cancel: CancellationToken,
) -> Result<Option<AssembledTurn>> {
    let writer_handle = runtime.writer();
    let mut writer = writer_handle.lock().await;
    let turn = writer
        .stream_next_queued_work(turn_options(sinks, None, cancel))
        .await?;
    runtime.publish_from(&writer);
    Ok(turn)
}

fn turn_options<'a>(
    sinks: TurnSinks<'a>,
    effect_scope: Option<RuntimeEffectControllerScope<'a>>,
    cancel: CancellationToken,
) -> lash_core::TurnOptions<'a> {
    let mut opts = lash_core::TurnOptions::new(cancel);
    if let Some(events) = sinks.events() {
        opts = opts.with_events(events);
    }
    if let Some(turn_events) = sinks.turn_events() {
        opts = opts.with_turn_events(turn_events);
    }
    if let Some(effect_scope) = effect_scope {
        opts = opts.with_effect_scope(effect_scope);
    }
    opts
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
    effect_scope: Option<RuntimeEffectControllerScope<'_>>,
    cancel: CancellationToken,
) -> Result<TurnResult> {
    let turn = Box::pin(stream_prepared_assembled(
        runtime,
        input,
        sinks,
        effect_scope,
        cancel,
    ))
    .await?;
    Ok(TurnResult::from_assembled(turn))
}

pub(crate) async fn stream_prepared_assembled(
    runtime: &RuntimeHandle,
    input: TurnInput,
    sinks: TurnSinks<'_>,
    effect_scope: Option<RuntimeEffectControllerScope<'_>>,
    cancel: CancellationToken,
) -> Result<AssembledTurn> {
    let turn = Box::pin(stream_prepared_agent_frame_run(
        runtime,
        input,
        sinks,
        effect_scope,
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
    effect_scope: Option<RuntimeEffectControllerScope<'_>>,
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
    let turn = Box::pin(
        writer.stream_turn_with_agent_frames(input, turn_options(sinks, effect_scope, cancel)),
    )
    .await?;
    runtime.publish_from(&writer);
    Ok(turn)
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TurnResult {
    pub state: SessionStateEnvelope,
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
