use crate::support::*;

pub struct TurnBuilder {
    pub(crate) runtime: RuntimeHandle,
    pub(crate) active_plugins: Vec<ActivePluginBinding>,
    pub(crate) input: TurnInput,
    pub(crate) cancel: CancellationToken,
    pub(crate) mode_turn_options: Option<ModeTurnOptions>,
    pub(crate) provider: Option<ProviderHandle>,
    pub(crate) model: Option<ModelSelection>,
}

impl TurnBuilder {
    pub fn cancel(mut self, cancel: CancellationToken) -> Self {
        self.cancel = cancel;
        self
    }

    pub fn mode_turn_options(mut self, options: ModeTurnOptions) -> Self {
        self.mode_turn_options = Some(options);
        self
    }

    pub fn provider(mut self, provider: ProviderHandle) -> Self {
        self.provider = Some(provider);
        self
    }

    pub fn model(mut self, model: impl Into<String>, variant: Option<String>) -> Self {
        self.model = Some(ModelSelection::new(model, variant));
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
        Ok(self.mode_turn_options(ModeTurnOptions::typed(
            ExecutionMode::new("rlm"),
            termination,
        )?))
    }

    /// Attach typed per-turn context for an activated plugin binding.
    ///
    /// This is the generic primitive. Plugin crates should usually wrap it in a
    /// domain extension trait such as `.with_tone(tone)` or `.with_board(board)`
    /// so application code stays typed in its own vocabulary.
    pub fn with_plugin_context<P: PluginBinding>(mut self, context: P::TurnContext) -> Self {
        self.input
            .turn_context
            .insert_plugin_context(P::ID, context);
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

    pub(crate) fn prepare(mut self) -> Result<(RuntimeHandle, TurnInput, CancellationToken)> {
        if let Some(options) = self.mode_turn_options {
            self.input.mode_turn_options = Some(options);
        }
        if let Some(provider) = self.provider {
            self.input.turn_context.set_provider(provider);
        }
        if let Some(model) = self.model {
            self.input
                .turn_context
                .set_model(model.model, model.variant);
        }
        validate_required_plugin_contexts(&self.active_plugins, &self.input)?;
        Ok((self.runtime, self.input, self.cancel))
    }

    pub async fn stream(self, events: &dyn TurnActivitySink) -> Result<TurnResult> {
        let (runtime, input, cancel) = self.prepare()?;
        stream_prepared_turn(&runtime, input, None, Some(events), cancel).await
    }

    pub fn into_stream(self) -> Result<TurnStream> {
        let (runtime, input, cancel) = self.prepare()?;
        let (tx, rx) = mpsc::channel(64);
        let sink = ChannelTurnActivitySink { tx };
        let completion = tokio::spawn(async move {
            stream_prepared_turn(&runtime, input, None, Some(&sink), cancel).await
        });
        Ok(TurnStream {
            activities: rx,
            completion,
        })
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
            EmbedError::Runtime(lash::RuntimeError {
                code: "turn_stream_join".to_string(),
                message: format!("turn stream task failed: {err}"),
            })
        })?
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
fn validate_required_plugin_contexts(
    active_plugins: &[ActivePluginBinding],
    input: &TurnInput,
) -> Result<()> {
    for plugin in active_plugins {
        if plugin.requires_turn_context && !input.turn_context.has_plugin_context(plugin.id) {
            return Err(EmbedError::MissingPluginTurnContext {
                plugin_id: plugin.id,
            });
        }
    }
    Ok(())
}

pub(crate) async fn stream_prepared_turn(
    runtime: &RuntimeHandle,
    input: TurnInput,
    session_events: Option<&dyn EventSink>,
    turn_events: Option<&dyn TurnActivitySink>,
    cancel: CancellationToken,
) -> Result<TurnResult> {
    let turn =
        stream_prepared_assembled(runtime, input, session_events, turn_events, cancel).await?;
    Ok(TurnResult::from_assembled(turn))
}

pub(crate) async fn stream_prepared_assembled(
    runtime: &RuntimeHandle,
    input: TurnInput,
    session_events: Option<&dyn EventSink>,
    turn_events: Option<&dyn TurnActivitySink>,
    cancel: CancellationToken,
) -> Result<AssembledTurn> {
    let writer_handle = runtime.writer();
    let mut writer = writer_handle.lock().await;
    if let Some(extension) = input.mode_extension.as_ref() {
        writer
            .validate_mode_turn_extension(extension)
            .await
            .map_err(EmbedError::Session)?;
    }
    let turn = match (session_events, turn_events) {
        (Some(session_events), Some(turn_events)) => {
            writer
                .stream_turn_with_events_following_handoffs(
                    input,
                    session_events,
                    turn_events,
                    cancel,
                )
                .await?
        }
        (Some(session_events), None) => {
            writer
                .stream_turn_following_handoffs(input, session_events, cancel)
                .await?
        }
        (None, Some(turn_events)) => {
            writer
                .stream_turn_events_following_handoffs(input, turn_events, cancel)
                .await?
        }
        (None, None) => {
            writer
                .stream_turn_events_following_handoffs(input, &lash::NoopTurnActivitySink, cancel)
                .await?
        }
    };
    runtime.publish_from(&writer);
    turn.into_final_turn().ok_or_else(|| {
        EmbedError::Runtime(lash::RuntimeError {
            code: "empty_followed_turn".to_string(),
            message: "runtime completed without an assembled turn".to_string(),
        })
    })
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TurnResult {
    pub outcome: TurnOutcome,
    pub usage: TokenUsage,
    pub errors: Vec<TurnIssue>,
}

impl TurnResult {
    fn from_assembled(turn: lash::AssembledTurn) -> Self {
        Self {
            outcome: turn.outcome,
            usage: turn.token_usage,
            errors: turn.errors,
        }
    }

    pub fn assistant_message(&self) -> Option<&str> {
        match &self.outcome {
            TurnOutcome::Finished(lash::TurnFinish::AssistantMessage { text }) => Some(text),
            _ => None,
        }
    }

    pub fn submitted_value(&self) -> Option<&serde_json::Value> {
        match &self.outcome {
            TurnOutcome::Finished(lash::TurnFinish::SubmittedValue { value }) => Some(value),
            _ => None,
        }
    }

    pub fn tool_value(&self) -> Option<(&str, &serde_json::Value)> {
        match &self.outcome {
            TurnOutcome::Finished(lash::TurnFinish::ToolValue { tool_name, value }) => {
                Some((tool_name.as_str(), value))
            }
            _ => None,
        }
    }

    pub fn is_success(&self) -> bool {
        matches!(
            self.outcome,
            TurnOutcome::Finished(_) | TurnOutcome::Handoff { .. }
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

    pub fn assistant_messages(&self) -> Vec<String> {
        let mut messages: Vec<String> = Vec::new();
        let mut current_id: Option<TurnActivityId> = None;
        for activity in &self.activities {
            let TurnEvent::AssistantProseDelta { text } = &activity.event else {
                continue;
            };
            if current_id.as_ref() == Some(&activity.correlation_id) {
                if let Some(current) = messages.last_mut() {
                    current.push_str(text);
                }
                continue;
            }
            current_id = Some(activity.correlation_id.clone());
            messages.push(text.clone());
        }
        messages
    }

    pub fn assistant_transcript_text(&self) -> String {
        self.assistant_messages().concat()
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
    }
}
