use super::*;

pub(in crate::runtime::session_manager) struct ChannelEventSink {
    pub(in crate::runtime::session_manager) tx: mpsc::Sender<SessionEvent>,
    pub(in crate::runtime::session_manager) live_usage: Option<LiveChildUsageForwarder>,
}

#[derive(Clone)]
pub(in crate::runtime::session_manager) struct LiveChildUsageForwarder {
    pub(in crate::runtime::session_manager) turn_id: String,
    pub(in crate::runtime::session_manager) session_id: String,
    pub(in crate::runtime::session_manager) source: String,
    pub(in crate::runtime::session_manager) model: String,
    pub(in crate::runtime::session_manager) token_ledger:
        Arc<std::sync::Mutex<Vec<TokenLedgerEntry>>>,
    pub(in crate::runtime::session_manager) child_turn_live_usage:
        Arc<std::sync::Mutex<HashMap<String, TokenUsage>>>,
    pub(in crate::runtime::session_manager) relay: Option<ChildUsageEventRelay>,
}

#[derive(Clone, Default)]
pub(in crate::runtime) struct ChildUsageEventRelay {
    tx: Arc<StdMutex<Option<mpsc::Sender<RuntimeStreamEvent>>>>,
}

impl ChildUsageEventRelay {
    pub(in crate::runtime) fn new(tx: mpsc::Sender<RuntimeStreamEvent>) -> Self {
        Self {
            tx: Arc::new(StdMutex::new(Some(tx))),
        }
    }

    pub(in crate::runtime) fn clear(&self) {
        self.tx.lock().expect("child usage relay lock").take();
    }

    async fn emit(&self, event: SessionEvent) {
        let tx = self.tx.lock().expect("child usage relay lock").clone();
        if let Some(tx) = tx
            && !tx.is_closed()
        {
            let _ = tx.send(RuntimeStreamEvent::Session(event)).await;
        }
    }
}

impl UsageCapability {
    pub(in crate::runtime::session_manager) fn record_token_usage(
        &self,
        source: &str,
        model: &str,
        usage: &TokenUsage,
    ) {
        record_token_usage_shared(&self.token_ledger, source, model, usage);
    }

    pub(in crate::runtime::session_manager) fn drain_token_ledger(&self) -> Vec<TokenLedgerEntry> {
        let mut ledger = self.token_ledger.lock().expect("token ledger lock");
        std::mem::take(&mut *ledger)
    }

    pub(in crate::runtime::session_manager) fn merge_drained_token_ledger(
        &self,
        state: &mut SessionSnapshot,
    ) -> Vec<TokenLedgerEntry> {
        let drained = self.drain_token_ledger();
        for entry in drained.iter().cloned() {
            merge_ledger_entry(&mut state.token_ledger, entry);
        }
        drained
    }

    pub(in crate::runtime::session_manager) async fn persist_current_usage_ledger(
        &self,
        current: &CurrentSessionCapability,
    ) -> Result<(), crate::PluginError> {
        if !self.persist_to_store {
            return Ok(());
        }
        let Some(store) = &current.store else {
            return Ok(());
        };
        let mut state = current.current_snapshot_for_store_write().await;
        let drained = self.drain_token_ledger();
        if drained.is_empty() {
            return Ok(());
        }
        for entry in drained.iter().cloned() {
            merge_ledger_entry(&mut state.token_ledger, entry);
        }
        let commit = crate::store::RuntimeCommit::persisted_state(&state, &drained);
        match store.commit_runtime_state(commit).await {
            Ok(result) => state.apply_persisted_commit_result(result),
            Err(err) => tracing::warn!("failed to persist current usage ledger: {err}"),
        }
        Ok(())
    }
}

fn usage_has_any_tokens(usage: &TokenUsage) -> bool {
    usage.input_tokens != 0
        || usage.output_tokens != 0
        || usage.cached_input_tokens != 0
        || usage.reasoning_tokens != 0
}

pub(in crate::runtime::session_manager) fn record_token_usage_shared(
    token_ledger: &Arc<std::sync::Mutex<Vec<TokenLedgerEntry>>>,
    source: &str,
    model: &str,
    usage: &TokenUsage,
) {
    if !usage_has_any_tokens(usage) {
        return;
    }
    let mut ledger = token_ledger.lock().expect("token ledger lock");
    if let Some(entry) = ledger
        .iter_mut()
        .find(|entry| entry.source == source && entry.model == model)
    {
        entry.usage.input_tokens += usage.input_tokens;
        entry.usage.output_tokens += usage.output_tokens;
        entry.usage.cached_input_tokens += usage.cached_input_tokens;
        entry.usage.reasoning_tokens += usage.reasoning_tokens;
    } else {
        ledger.push(TokenLedgerEntry {
            source: source.to_string(),
            model: model.to_string(),
            usage: usage.clone(),
        });
    }
}

pub(in crate::runtime::session_manager) fn subtract_usage(
    reported_total: &TokenUsage,
    final_total: &TokenUsage,
) -> Option<TokenUsage> {
    let delta = TokenUsage {
        input_tokens: final_total
            .input_tokens
            .saturating_sub(reported_total.input_tokens),
        output_tokens: final_total
            .output_tokens
            .saturating_sub(reported_total.output_tokens),
        cached_input_tokens: final_total
            .cached_input_tokens
            .saturating_sub(reported_total.cached_input_tokens),
        reasoning_tokens: final_total
            .reasoning_tokens
            .saturating_sub(reported_total.reasoning_tokens),
    };
    usage_has_any_tokens(&delta).then_some(delta)
}

impl LiveChildUsageForwarder {
    async fn relay_token_usage(
        &self,
        mode_iteration: usize,
        _usage: &TokenUsage,
        cumulative_usage: &TokenUsage,
    ) {
        let (delta, cumulative) = {
            let mut live_usage = self
                .child_turn_live_usage
                .lock()
                .expect("child turn live usage lock");
            let reported = live_usage.entry(self.turn_id.clone()).or_default();
            let Some(delta) = subtract_usage(reported, cumulative_usage) else {
                return;
            };
            *reported = cumulative_usage.clone();
            (delta, reported.clone())
        };
        record_token_usage_shared(&self.token_ledger, &self.source, &self.model, &delta);
        if let Some(relay) = &self.relay {
            relay
                .emit(SessionEvent::ChildTokenUsage {
                    session_id: self.session_id.clone(),
                    source: self.source.clone(),
                    model: self.model.clone(),
                    mode_iteration,
                    usage: delta,
                    cumulative,
                })
                .await;
        }
    }
}

#[async_trait::async_trait]
impl EventSink for ChannelEventSink {
    async fn emit(&self, event: SessionEvent) {
        if let SessionEvent::TokenUsage {
            mode_iteration,
            usage,
            cumulative,
        } = &event
            && let Some(live_usage) = &self.live_usage
        {
            live_usage
                .relay_token_usage(*mode_iteration, usage, cumulative)
                .await;
        }
        if !self.tx.is_closed() {
            let _ = self.tx.send(event).await;
        }
    }
}
