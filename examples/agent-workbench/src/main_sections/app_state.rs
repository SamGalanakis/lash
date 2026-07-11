impl AppState {
    fn current_session_id(&self) -> String {
        self.session_ids.current()
    }

    fn selected_model(&self) -> ModelSelection {
        self.selected_model
            .lock()
            .expect("selected model lock")
            .clone()
    }

    fn set_selected_model(&self, model: ModelSelection) {
        *self.selected_model.lock().expect("selected model lock") = model;
    }

    fn settings(&self) -> Settings {
        let selected_model = self.selected_model();
        Settings {
            model: selected_model.model,
            model_variant: selected_model.model_variant,
            web_configured: self.web_configured,
            model_variants: vec!["", "low", "medium", "high"],
            session_id: self.current_session_id(),
        }
    }

    fn messages_snapshot(&self) -> Vec<ChatMessage> {
        self.messages.lock().expect("messages lock").clone()
    }

    fn trace(&self, name: &str, payload: Value) {
        emit_workbench_trace(
            &self.trace_sink,
            Some(self.current_session_id()),
            name,
            payload,
        );
    }

    fn publish(&self, item: StreamItem) {
        let _ = self.event_tx.send(item);
    }

    fn track_restate_invocation(
        &self,
        session_id: &str,
        turn_id: &str,
        invocation_id: lash_restate::RestateInvocationId,
    ) {
        // In-memory cancellation aid only. Restate owns in-flight workflow
        // replay, and Lash owns the turn commit; the workbench does not
        // persist a submitted/running work-item lifecycle table.
        self.active_restate_invocations
            .insert(session_id, turn_id, invocation_id);
    }

    fn clear_restate_invocation_on_drop(
        &self,
        session_id: &str,
        turn_id: &str,
    ) -> ActiveRestateInvocationGuard {
        self.active_restate_invocations.guard(session_id, turn_id)
    }

    /// Cancel every active Restate-backed turn invocation for `session_id`;
    /// returns how many cancellation requests were accepted. Missing entries
    /// mean this process has no live cancellation handle, not that the turn
    /// lifecycle is stored somewhere else.
    async fn cancel_turns_for_session(&self, session_id: &str) -> Result<usize, AppError> {
        let active = self.active_restate_invocations.for_session(session_id);
        let admin = lash_restate::RestateAdminClient::new(
            lash_restate::RestateConnection::with_client(
            self.restate_admin_url.clone(),
                self.restate_http.clone(),
            ),
        );
        let mut cancelled = 0;
        for (turn_id, invocation_id) in active {
            admin
                .cancel_invocation(&invocation_id)
                .await
                .map_err(|err| AppError::internal(err.to_string()))?;
            cancelled += 1;
            self.trace(
                "turn.restate.cancel_requested",
                json!({
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "invocation_id": invocation_id.as_str(),
                }),
            );
        }
        if cancelled > 0 {
            self.publish(StreamItem::Done);
        }
        Ok(cancelled)
    }

    fn push_message(&self, role: impl Into<String>, text: impl Into<String>) -> ChatMessage {
        let message = ChatMessage {
            id: uuid::Uuid::new_v4().to_string(),
            role: role.into(),
            text: text.into(),
            at: Utc::now().to_rfc3339(),
        };
        self.messages
            .lock()
            .expect("messages lock")
            .push(message.clone());
        self.publish(StreamItem::Message {
            message: message.clone(),
        });
        message
    }
}

fn emit_workbench_trace(
    sink: &Option<Arc<dyn TraceSink>>,
    session_id: Option<String>,
    name: &str,
    payload: Value,
) {
    let Some(sink) = sink else {
        return;
    };
    let context = session_id
        .map(|session_id| TraceContext::default().for_session(session_id))
        .unwrap_or_default();
    let record = TraceRecord::new(
        context,
        TraceEvent::Custom {
            name: format!("agent_workbench.{name}"),
            payload,
        },
    );
    if let Err(err) = sink.append(&record) {
        eprintln!("warning: failed to append agent-workbench trace event `{name}`: {err}");
    }
}

fn trace_work_item(item: &WorkItem) -> Value {
    json!({
        "process_id": item.process.process_id.clone(),
        "graph_key": item.process.graph_key.clone(),
        "kind": item.kind.clone(),
        "label": item.label.clone(),
        "status_label": item.process.status_label.clone(),
        "terminal": item.process.terminal,
        "created_at_ms": item.process.created_at_ms,
        "updated_at_ms": item.process.updated_at_ms,
        "input": item.process.input.clone(),
        "events": item.events.iter().map(|event| {
            json!({
                "sequence": event.sequence,
                "event_type": event.event_type.clone(),
                "occurred_at_ms": event.occurred_at_ms,
                "payload": event.payload.clone(),
            })
        }).collect::<Vec<_>>(),
    })
}

#[derive(Clone, Debug)]
struct WorkbenchSessionIds {
    current: Arc<Mutex<String>>,
}

impl WorkbenchSessionIds {
    fn fresh() -> Self {
        Self {
            current: Arc::new(Mutex::new(new_session_id())),
        }
    }

    fn current(&self) -> String {
        self.current.lock().expect("session id lock").clone()
    }

    fn rotate(&self) -> (String, String) {
        let mut current = self.current.lock().expect("session id lock");
        let old = current.clone();
        let new = new_session_id();
        *current = new.clone();
        (old, new)
    }
}

fn new_session_id() -> String {
    format!("{SESSION_ID_PREFIX}-{}", uuid::Uuid::new_v4().simple())
}

fn model_spec_for_request(
    state: &AppState,
    model: Option<&str>,
    model_variant: Option<&str>,
) -> Result<lash::ModelSpec, AppError> {
    let selected_model = state.selected_model();
    let model = model
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(selected_model.model.as_str())
        .to_string();
    let model_variant = model_variant_for_request(&selected_model, model_variant);
    lash::ModelSpec::from_token_limits(
        model,
        model_variant
            .map(lash::provider::ReasoningSelection::Effort)
            .unwrap_or_default(),
        DEFAULT_CONTEXT_WINDOW_TOKENS,
        None,
    )
        .map(with_workbench_model_capability)
        .map_err(AppError::bad_request)
}

fn model_variant_for_request(
    selected_model: &ModelSelection,
    model_variant: Option<&str>,
) -> Option<String> {
    match model_variant {
        Some(value) => {
            let value = value.trim();
            if value.is_empty() {
                None
            } else {
                Some(value.to_string())
            }
        }
        None => selected_model.model_variant.clone(),
    }
}

fn model_spec_from_selection(selection: ModelSelection) -> lash::ModelSpec {
    lash::ModelSpec::from_token_limits(
        selection.model,
        selection
            .model_variant
            .map(lash::provider::ReasoningSelection::Effort)
            .unwrap_or_default(),
        DEFAULT_CONTEXT_WINDOW_TOKENS,
        None,
    )
    .expect("workbench model selection should use a valid token limit")
    .with_capability(workbench_model_capability())
}

fn with_workbench_model_capability(model: lash::ModelSpec) -> lash::ModelSpec {
    model.with_capability(workbench_model_capability())
}

fn workbench_model_capability() -> lash::provider::ModelCapability {
    lash::provider::ModelCapability {
        reasoning: Some(lash::provider::ReasoningCapability {
            efforts: ["low", "medium", "high"]
                .into_iter()
                .map(String::from)
                .collect(),
            default_effort: Some("medium".to_string()),
            aliases: Default::default(),
            encoding: lash::provider::ReasoningEncoding::Effort,
            disable: None,
            mandatory: false,
        }),
    }
}

async fn apply_model_selection_to_session(
    state: &AppState,
    session: &lash::LashSession,
    model: lash::ModelSpec,
    reason: &str,
) -> Result<(), AppError> {
    state.set_selected_model(ModelSelection::from_spec(&model));
    session
        .configure(lash::SessionConfigPatch {
            model: Some(model.clone()),
            ..lash::SessionConfigPatch::default()
        })
        .await
        .map_err(AppError::internal)?;
    state.trace(
        "model_selection.applied",
        json!({
            "reason": reason,
            "model": serde_json::to_value(&model).unwrap_or(Value::Null),
        }),
    );
    Ok(())
}

fn assistant_text_for_display(output: &TurnResult, streamed_prose: &str) -> String {
    let terminal = output.final_value().map(terminal_value_text).or_else(|| {
        output
            .tool_value()
            .map(|(_tool_name, value)| terminal_value_text(value))
    });
    let assistant = (!streamed_prose.trim().is_empty())
        .then(|| streamed_prose.to_string())
        .or_else(|| {
            output
                .assistant_message()
                .filter(|text| !text.trim().is_empty())
                .map(str::to_string)
        });
    combine_assistant_display_parts(assistant, terminal)
}

fn combine_assistant_display_parts(assistant: Option<String>, terminal: Option<String>) -> String {
    let assistant = assistant.filter(|text| !text.trim().is_empty());
    let terminal = terminal.filter(|text| !text.trim().is_empty());
    match (assistant, terminal) {
        (Some(assistant), Some(terminal)) if assistant.trim() == terminal.trim() => assistant,
        (Some(assistant), Some(terminal)) => format!("{}\n\n{}", assistant.trim_end(), terminal),
        (Some(assistant), None) => assistant,
        (None, Some(terminal)) => terminal,
        (None, None) => String::new(),
    }
}

fn terminal_value_text(value: &Value) -> String {
    value
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(|| value.to_string())
}

fn compact_payload(value: Value) -> Value {
    match value {
        Value::String(text) if text.len() > 1_200 => Value::String(truncate_chars(&text, 1_200)),
        Value::Array(items) => {
            Value::Array(items.into_iter().take(12).map(compact_payload).collect())
        }
        Value::Object(map) => Value::Object(
            map.into_iter()
                .map(|(key, value)| (key, compact_payload(value)))
                .collect(),
        ),
        other => other,
    }
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    format!("{}...", text.chars().take(max_chars).collect::<String>())
}

fn work_item_from_observed(item: lash::process::ObservedWorkItem) -> WorkItem {
    WorkItem {
        process: work_process_from_observed(item.process),
        descriptor: item.descriptor,
        events: item
            .events
            .into_iter()
            .map(work_event_from_observed)
            .collect(),
        kind: item.kind,
        label: item.label,
    }
}

fn work_process_from_observed(process: lash::process::ObservedProcess) -> WorkProcess {
    WorkProcess {
        process_id: process.process_id,
        graph_key: process.graph_key,
        lifecycle: process.lifecycle,
        status_label: process.status_label,
        terminal: process.terminal,
        error: process.error,
        created_at_ms: process.created_at_ms,
        updated_at_ms: process.updated_at_ms,
        input: compact_payload(serde_json::to_value(process.input).unwrap_or(Value::Null)),
        external_ref: process
            .external_ref
            .and_then(|value| serde_json::to_value(value).ok())
            .map(compact_payload),
        child_session_id: process.child_session_id,
        label: process.label,
    }
}

fn work_event_from_observed(event: lash::process::ObservedProcessEvent) -> WorkEvent {
    WorkEvent {
        sequence: event.sequence,
        event_type: event.event_type,
        occurred_at_ms: event.occurred_at_ms,
        payload: compact_payload(event.payload),
    }
}

#[derive(Debug)]
struct AppError {
    status: StatusCode,
    message: String,
}

impl AppError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: message.into(),
        }
    }

    fn internal(message: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.to_string(),
        }
    }

    fn gateway_timeout(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::GATEWAY_TIMEOUT,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for AppError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for AppError {}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(json!({
                "error": self.message,
            })),
        )
            .into_response()
    }
}
