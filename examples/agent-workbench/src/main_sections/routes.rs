async fn healthz() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "service": "agent-workbench", "status": "ok" }))
}

async fn index() -> Html<&'static str> {
    Html(ui::INDEX_HTML)
}

async fn app_state(State(state): State<AppState>) -> Result<Json<StateSnapshot>, AppError> {
    let session = state
        .core
        .session(state.current_session_id())
        .open()
        .await
        .map_err(AppError::internal)?;
    let mut messages: Vec<_> = session
        .read_view()
        .messages()
        .iter()
        .map(chat_message_from_committed)
        .collect();
    let pending_turn_inputs = session
        .pending_turn_inputs()
        .await
        .map_err(AppError::internal)?;
    session.close().await.map_err(AppError::internal)?;
    let active_turns = state.active_turns.for_session(&state.current_session_id());
    messages.extend(active_turns.iter().filter_map(|address| {
        state
            .active_turns
            .prompt_for(&address.session_id, &address.turn_id)
            .map(|text| ChatMessage {
                id: format!("workbench-active-prompt:{}", address.turn_id),
                role: "user".to_string(),
                text,
                at: String::new(),
            })
    }));
    Ok(Json(StateSnapshot {
        settings: state.settings(),
        messages,
        active_turns,
        pending_turn_inputs,
    }))
}

fn chat_message_from_committed(message: &lash::messages::Message) -> ChatMessage {
    ChatMessage {
        id: message.id.clone(),
        role: lash::message_role(message).to_string(),
        text: lash::message_text(message),
        // The durable session graph records ordering but not a presentation
        // timestamp. The workbench does not render this field, so keep the
        // established wire shape without fabricating a time during resume.
        at: String::new(),
    }
}

async fn session_events(
    State(state): State<AppState>,
    Query(query): Query<EventsQuery>,
) -> Result<Response, AppError> {
    let session = state
        .core
        .session(state.current_session_id())
        .open()
        .await
        .map_err(AppError::internal)?;
    let observable = session.observe();
    let cursor = match query.cursor.as_deref().filter(|cursor| !cursor.trim().is_empty()) {
        Some(cursor) => serde_json::from_value::<SessionCursor>(json!(cursor))
            .map_err(|err| AppError::bad_request(format!("invalid session cursor: {err}")))?,
        None => observable.current_observation().cursor,
    };
    let mut product_events = state.event_tx.subscribe();
    let (tx, rx) = mpsc::channel::<StreamItem>(64);
    let observation_tx = tx.clone();
    tokio::spawn(async move {
        forward_session_observations(session, cursor, observation_tx).await;
    });
    tokio::spawn(async move {
        loop {
            match product_events.recv().await {
                Ok(item) => {
                    if tx.send(item).await.is_err() {
                        break;
                    }
                }
                Err(broadcast::error::RecvError::Lagged(count)) => {
                    let _ = tx
                        .send(StreamItem::Error {
                            message: format!("event stream skipped {count} updates"),
                        })
                        .await;
                }
                Err(broadcast::error::RecvError::Closed) => break,
            }
        }
    });
    Ok(ndjson_response(rx))
}

async fn send_turn(
    State(state): State<AppState>,
    Json(request): Json<TurnRequest>,
) -> Result<Json<CommandAccepted>, AppError> {
    let text = request.text.trim().to_string();
    if text.is_empty() {
        return Err(AppError::bad_request("message text is required"));
    }
    let turn_model = model_spec_for_request(
        &state,
        request.model.as_deref(),
        request.model_variant.as_deref(),
    )?;
    state.trace(
        "api.turn.request",
        json!({
            "text": text.clone(),
            "model": serde_json::to_value(&turn_model).unwrap_or(Value::Null),
        }),
    );
    state.push_message("user", text.clone());
    state.set_selected_model(ModelSelection::from_spec(&turn_model));
    let turn_id = format!("workbench-turn-{}", uuid::Uuid::new_v4());
    let session_id = state.current_session_id();
    state.track_turn_prompt(&session_id, &turn_id, text.clone());
    if let Err(err) = restate::submit_user_turn(
        &state,
        restate::WorkbenchTurnWorkflowRequest {
            turn_id: turn_id.clone(),
            session_id: session_id.clone(),
            text,
            model: ModelSelection::from_spec(&turn_model),
        },
    )
    .await
    {
        state.active_turns.remove(&session_id, &turn_id);
        return Err(err);
    }
    Ok(Json(CommandAccepted { accepted: true }))
}

async fn enqueue_turn_input(
    State(state): State<AppState>,
    Json(request): Json<TurnInputRequest>,
) -> Result<Json<TurnInputReceipt>, AppError> {
    let text = request.text.trim().to_string();
    if text.is_empty() {
        return Err(AppError::bad_request("message text is required"));
    }
    let session_id = state.current_session_id();
    let ingress = match request.ingress {
        TurnInputIngressRequest::ActiveTurn => {
            let active = state.active_turns.for_session(&session_id);
            let [address] = active.as_slice() else {
                return Err(AppError::conflict(
                    "inject now requires exactly one running turn",
                ));
            };
            lash::persistence::TurnInputIngress::active_turn(
                address.turn_id.clone(),
                lash::persistence::TurnInputCheckpointBoundary::AfterWork,
            )
        }
        TurnInputIngressRequest::NextTurn => lash::persistence::TurnInputIngress::next_turn(),
    };
    let source_id = format!("workbench-turn-input-{}", uuid::Uuid::new_v4());
    let pending = state
        .core
        .enqueue_turn_input(
            session_id,
            lash::TurnInput::text(text.clone()),
            ingress,
            Some(source_id),
        )
        .await
        .map_err(AppError::internal)?;
    reject_if_active_turn_settled(&state, &pending).await?;
    let receipt = TurnInputReceipt {
        accepted: true,
        input_id: pending.input_id.clone(),
        ingress: pending.ingress.clone(),
        state: pending.state,
        text,
    };
    state.trace(
        "turn_input.enqueued",
        serde_json::to_value(&receipt).unwrap_or(Value::Null),
    );
    state.publish(StreamItem::TurnInput {
        receipt: receipt.clone(),
    });
    Ok(Json(receipt))
}

async fn reject_if_active_turn_settled(
    state: &AppState,
    pending: &lash::PendingTurnInput,
) -> Result<(), AppError> {
    let Some(turn_id) = pending.ingress.active_turn_id() else {
        return Ok(());
    };
    if state.active_turns.contains(&pending.session_id, turn_id) {
        return Ok(());
    }

    let session = state
        .core
        .session(pending.session_id.clone())
        .open()
        .await
        .map_err(AppError::runtime)?;
    let outcome = session
        .cancel_pending_turn_input(&pending.input_id)
        .await
        .map_err(AppError::runtime)?;
    match outcome {
        lash::PendingTurnInputCancelOutcome::Cancelled(_)
        | lash::PendingTurnInputCancelOutcome::AlreadyCancelled(_) => Err(AppError::conflict(
            "the running turn settled before the input could be injected",
        )),
        lash::PendingTurnInputCancelOutcome::AlreadyClaimed { .. }
        | lash::PendingTurnInputCancelOutcome::AlreadyCompleted(_) => Ok(()),
        lash::PendingTurnInputCancelOutcome::NotFound => Err(AppError::internal(format!(
            "active-turn input `{}` disappeared during settle reconciliation",
            pending.input_id
        ))),
    }
}

async fn button_trigger(
    State(state): State<AppState>,
    Json(request): Json<ButtonEventRequest>,
) -> Result<Json<CommandAccepted>, AppError> {
    let turn_model = model_spec_for_request(
        &state,
        request.model.as_deref(),
        request.model_variant.as_deref(),
    )?;
    let model = ModelSelection::from_spec(&turn_model);
    state.set_selected_model(model.clone());
    state.trace(
        "api.button_trigger.request",
        json!({
            "button": request.button,
            "model": serde_json::to_value(&turn_model).unwrap_or(Value::Null),
        }),
    );
    let pressed_at = Utc::now().to_rfc3339();
    state.push_message(
        "event",
        format!("{} button trigger occurrence", request.button.lower()),
    );
    restate::submit_button_trigger(
        &state,
        restate::WorkbenchButtonTriggerWorkflowRequest {
            operation_id: format!("workbench-button-{}", uuid::Uuid::new_v4()),
            session_id: state.current_session_id(),
            button: request.button,
            model,
            pressed_at,
        },
    )
    .await?;
    Ok(Json(CommandAccepted { accepted: true }))
}

async fn list_accounts(State(state): State<AppState>) -> Json<Vec<mail::AccountSummary>> {
    Json(state.mail_world.account_summaries())
}

async fn add_account(
    State(state): State<AppState>,
    Json(request): Json<AddAccountRequest>,
) -> Result<Json<mail::AccountSummary>, AppError> {
    let summary = state
        .mail_world
        .add_account(&request.name)
        .map_err(AppError::bad_request)?;
    state.trace(
        "api.accounts.add",
        json!({ "slug": summary.slug, "authority": summary.authority }),
    );
    enqueue_tool_catalog_refresh(&state, "account_added").await?;
    state.push_message(
        "event",
        format!("connected mock account `{}`", summary.authority),
    );
    Ok(Json(summary))
}

async fn delete_account(
    AxumPath(slug): AxumPath<String>,
    State(state): State<AppState>,
) -> Result<Json<CommandAccepted>, AppError> {
    state
        .mail_world
        .remove_account(&slug)
        .map_err(AppError::not_found)?;
    state.trace("api.accounts.remove", json!({ "slug": slug }));
    enqueue_tool_catalog_refresh(&state, "account_removed").await?;
    state.push_message("event", format!("removed mock account `inbox.{slug}`"));
    Ok(Json(CommandAccepted { accepted: true }))
}

async fn delete_message(
    AxumPath((slug, id)): AxumPath<(String, String)>,
    State(state): State<AppState>,
) -> Result<Json<CommandAccepted>, AppError> {
    state
        .mail_world
        .remove_message(&slug, &id)
        .map_err(AppError::not_found)?;
    state.trace(
        "api.accounts.message.delete",
        json!({ "account": slug, "id": id }),
    );
    Ok(Json(CommandAccepted { accepted: true }))
}

async fn account_inbox(
    AxumPath(slug): AxumPath<String>,
    State(state): State<AppState>,
) -> Result<Json<Vec<mail::MailMessage>>, AppError> {
    let inbox = state.mail_world.inbox(&slug).map_err(AppError::not_found)?;
    Ok(Json(inbox))
}

/// Enqueue a durable tool-catalog refresh for the chat session.
///
/// The enqueue asks the host-owned queued-work driver to submit a Restate
/// workflow for the batch; that workflow drains it with a durable handler
/// context and the runtime commits the refreshed surface to the SQLite session store.
/// Nothing here executes effects in the foreground — the workbench runs
/// Restate + SQLite only.
async fn enqueue_tool_catalog_refresh(
    state: &AppState,
    reason: &str,
) -> Result<lash::SessionCommandReceipt, AppError> {
    let session_id = state.current_session_id();
    let session = state
        .core
        .session(session_id.clone())
        .open()
        .await
        .map_err(AppError::internal)?;
    let receipt = session
        .commands()
        .refresh_tool_catalog(
            reason,
            format!(
                "workbench-refresh-tool-catalog:{}:{}:{}",
                session_id,
                reason,
                uuid::Uuid::new_v4()
            ),
        )
        .await
        .map_err(AppError::internal)?;
    session.close().await.map_err(AppError::internal)?;
    state.trace(
        "mail.tool_catalog.refresh_enqueued",
        json!({
            "reason": reason,
            "session_id": session_id,
            "command_batch_id": receipt.batch_id,
            "command_source_key": receipt.source_key,
        }),
    );
    Ok(receipt)
}

async fn inject_message(
    AxumPath(slug): AxumPath<String>,
    State(state): State<AppState>,
    Json(request): Json<InjectMessageRequest>,
) -> Result<Json<CommandAccepted>, AppError> {
    let turn_model = model_spec_for_request(
        &state,
        request.model.as_deref(),
        request.model_variant.as_deref(),
    )?;
    let model = ModelSelection::from_spec(&turn_model);
    state.set_selected_model(model.clone());
    let delivered = state
        .mail_world
        .deliver(
            &slug,
            request.title.as_deref().unwrap_or_default(),
            request.text.as_deref().unwrap_or_default(),
        )
        .map_err(AppError::not_found)?;
    let message = delivered.message;
    let delivery = delivered.delivery;
    state.trace(
        "api.accounts.inject",
        json!({ "account": slug, "title": message.title }),
    );
    state.push_message(
        "event",
        format!("message delivered to `inbox.{}`: {}", slug, message.title),
    );
    restate::submit_mail_received(
        &state,
        restate::WorkbenchMailReceivedWorkflowRequest {
            operation_id: format!("workbench-mail-{}", uuid::Uuid::new_v4()),
            session_id: state.current_session_id(),
            model,
            delivery,
        },
    )
    .await?;
    Ok(Json(CommandAccepted { accepted: true }))
}

async fn cancel_turn(State(state): State<AppState>) -> Result<Json<TurnCancelResponse>, AppError> {
    let session_id = state.current_session_id();
    let cancellations = state.cancel_turns_for_session(&session_id).await?;
    state.trace(
        "api.turn.cancel",
        json!({ "session_id": session_id, "cancellations": cancellations }),
    );
    Ok(Json(TurnCancelResponse {
        accepted: !cancellations.is_empty(),
        cancellations,
    }))
}

async fn reset_chat(State(state): State<AppState>) -> Result<Json<StateSnapshot>, AppError> {
    let old_session_id = state.current_session_id();
    restate::cancel_cron_jobs_for_session(&state, &old_session_id, "reset").await?;
    restate::submit_session_delete(
        &state,
        restate::WorkbenchSessionDeleteWorkflowRequest {
            operation_id: format!("workbench-delete-{}", uuid::Uuid::new_v4()),
            session_id: old_session_id.clone(),
        },
    )
    .await?;
    let (rotated_old, _) = state.session_ids.rotate();
    if rotated_old != old_session_id {
        eprintln!(
            "warning: workbench session changed during reset; deleted {old_session_id}, rotated {rotated_old}"
        );
    }
    let new_session_id = state.current_session_id();
    state.trace(
        "api.reset",
        json!({
            "old_session_id": old_session_id,
            "new_session_id": new_session_id.clone(),
        }),
    );
    let session = state
        .core
        .session(new_session_id)
        .open()
        .await
        .map_err(AppError::internal)?;
    let selected_model = model_spec_from_selection(state.selected_model());
    session
        .configure(lash::SessionConfigPatch {
            model: Some(selected_model),
            ..lash::SessionConfigPatch::default()
        })
        .await
        .map_err(AppError::internal)?;
    state.messages.lock().expect("messages lock").clear();
    state.lashlang_execution.clear();
    state.mail_world.clear();
    Ok(Json(StateSnapshot {
        settings: state.settings(),
        messages: Vec::new(),
        active_turns: Vec::new(),
        pending_turn_inputs: Vec::new(),
    }))
}

async fn list_work(State(state): State<AppState>) -> Result<Json<Vec<WorkItem>>, AppError> {
    let work = state
        .process_observer
        .snapshot_all(&lash::process::ProcessListFilter {
            status: lash::process::ProcessStatusFilter::Any,
            ..lash::process::ProcessListFilter::default()
        })
        .await
        .map_err(AppError::internal)?
        .into_iter()
        .map(work_item_from_observed)
        .collect::<Vec<_>>();
    state.trace(
        "api.work.response",
        json!({
            "count": work.len(),
            "items": work.iter().map(trace_work_item).collect::<Vec<_>>(),
        }),
    );
    Ok(Json(work))
}

async fn cancel_work(
    AxumPath(process_id): AxumPath<String>,
    State(state): State<AppState>,
) -> Result<Json<ProcessCancelAccepted>, AppError> {
    let process = state
        .process_observer
        .process(&process_id)
        .await
        .ok_or_else(|| AppError::not_found(format!("unknown process `{process_id}`")))?;
    if process.terminal {
        return Err(AppError::conflict(format!(
            "process `{process_id}` is already terminal"
        )));
    }
    let operation_id = format!("workbench-process-cancel-{}", uuid::Uuid::new_v4());
    restate::submit_process_cancel(
        &state,
        restate::WorkbenchProcessCancelWorkflowRequest {
            operation_id: operation_id.clone(),
            process_id: process_id.clone(),
        },
    )
    .await?;
    state.trace(
        "api.work.cancel_submitted",
        json!({
            "operation_id": operation_id,
            "process_id": process_id,
        }),
    );
    Ok(Json(ProcessCancelAccepted {
        accepted: true,
        operation_id,
        process_id,
    }))
}

/// Wait for one durable work item to reach a terminal state, then return its
/// outcome and the authoritative event log.
///
/// This is the host-facing "wait for the work item" flow. It routes through
/// [`ProcessWorkDriver::await_terminal`](lash::process::ProcessWorkDriver::await_terminal)
/// (ADR 0016) — the Restate ingress attach, never a store poll loop — and bounds
/// the wait with `tokio::time::timeout` so a still-running or unknown-to-this-pod
/// process cannot pin the request. On terminal it reconciles from `events_after`
/// (ADR 0017): the durable log is the truth; the best-effort event sink is only
/// freshness and may have dropped events.
async fn await_work(
    AxumPath(process_id): AxumPath<String>,
    State(state): State<AppState>,
) -> Result<Json<WorkAwaitResult>, AppError> {
    const AWAIT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);
    let outcome = match tokio::time::timeout(
        AWAIT_TIMEOUT,
        state.process_work_driver.await_terminal(&process_id),
    )
    .await
    {
        Ok(Ok(outcome)) => outcome,
        Ok(Err(err)) => return Err(AppError::internal(err)),
        Err(_elapsed) => {
            return Err(AppError::gateway_timeout(format!(
                "timed out waiting for process `{process_id}` to terminate"
            )));
        }
    };
    let events: Vec<WorkAwaitEvent> = state
        .process_work_driver
        .process_registry()
        .events_after(&process_id, 0)
        .await
        .map_err(AppError::internal)?
        .into_iter()
        .map(|event| WorkAwaitEvent {
            sequence: event.sequence,
            event_type: event.event_type,
        })
        .collect();
    state.trace(
        "api.work.await",
        json!({
            "process_id": process_id,
            "terminal_state": format!("{:?}", outcome.terminal_state()),
            "event_count": events.len(),
        }),
    );
    Ok(Json(WorkAwaitResult {
        process_id,
        outcome,
        events,
    }))
}

async fn list_lashlang_graphs(
    State(state): State<AppState>,
) -> Result<Json<execution_graphs::LashlangGraphIndex>, AppError> {
    let index = execution_graphs::index_for_session(
        &state.process_observer,
        &state.current_session_id(),
        state.lashlang_execution.graphs(),
    )
    .await?;
    Ok(Json(index))
}

async fn lashlang_graph(
    AxumPath(graph_key): AxumPath<String>,
    State(state): State<AppState>,
) -> Result<Json<TraceLashlangGraph>, AppError> {
    let graph = execution_graphs::visible_graph_by_key(
        &state.process_observer,
        &state.current_session_id(),
        state.lashlang_execution.graphs(),
        &graph_key,
    )
    .await?;
    Ok(Json(graph))
}

fn ndjson_response(rx: mpsc::Receiver<StreamItem>) -> Response {
    let stream = ReceiverStream::new(rx).map(|item| {
        let mut line = serde_json::to_string(&item).unwrap_or_else(|err| {
            json!({
                "type": "error",
                "message": err.to_string(),
            })
            .to_string()
        });
        line.push('\n');
        Ok::<Bytes, Infallible>(Bytes::from(line))
    });

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/x-ndjson; charset=utf-8")
        .header(header::CACHE_CONTROL, "no-store")
        .body(Body::from_stream(stream))
        .expect("valid streaming response")
}

async fn forward_session_observations(
    session: lash::LashSession,
    cursor: SessionCursor,
    tx: mpsc::Sender<StreamItem>,
) {
    if tx
        .send(StreamItem::ReplayCursor {
            cursor: cursor.to_string(),
        })
        .await
        .is_err()
    {
        return;
    }
    let mut stream = match session
        .observe()
        .subscribe_and_recover_remote(RemoteSessionCursor::new(cursor.to_string()))
    {
        Ok(stream) => stream,
        Err(err) => {
            let _ = tx
                .send(StreamItem::Error {
                    message: err.to_string(),
                })
                .await;
            return;
        }
    };
    while let Some(item) = stream.next().await {
        match item {
            Ok(RemoteSessionObservationStreamItem::Event(event)) => {
                if tx
                    .send(StreamItem::Observation {
                        event: Box::new(event),
                    })
                    .await
                    .is_err()
                {
                    break;
                }
            }
            Ok(RemoteSessionObservationStreamItem::Gap { observation, gap }) => {
                if tx
                    .send(StreamItem::ReplayGap {
                        observation: Box::new(observation),
                        gap: Box::new(gap),
                    })
                    .await
                    .is_err()
                {
                    break;
                }
            }
            Err(err) => {
                let _ = tx
                    .send(StreamItem::Error {
                        message: err.to_string(),
                    })
                    .await;
                break;
            }
        }
    }
}

#[derive(Default)]
struct TurnStreamState {
    assistant_prose: Vec<TurnStreamProseChunk>,
}

struct TurnStreamProseChunk {
    correlation_id: lash::TurnActivityId,
    text: String,
}

impl TurnStreamState {
    fn assistant_prose(&self) -> String {
        self.assistant_prose
            .iter()
            .map(|chunk| chunk.text.as_str())
            .collect()
    }
}

struct ChannelTurnEvents {
    turn_state: Arc<Mutex<TurnStreamState>>,
}

#[async_trait]
impl TurnActivitySink for ChannelTurnEvents {
    async fn emit(&self, activity: TurnActivity) {
        let mut turn_state = self.turn_state.lock().expect("turn state lock");
        match activity.event {
            TurnEvent::AssistantProseDelta { text } => {
                turn_state.assistant_prose.push(TurnStreamProseChunk {
                    correlation_id: activity.correlation_id,
                    text,
                });
            }
            TurnEvent::ModelAttemptReset {
                assistant_prose_correlation_ids,
                ..
            } => {
                turn_state.assistant_prose.retain(|chunk| {
                    !assistant_prose_correlation_ids.contains(&chunk.correlation_id)
                });
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod turn_stream_state_tests {
    use super::*;

    #[tokio::test]
    async fn workbench_turn_stream_state_retracts_only_superseded_prose() {
        let turn_state = Arc::new(Mutex::new(TurnStreamState::default()));
        let sink = ChannelTurnEvents {
            turn_state: Arc::clone(&turn_state),
        };
        sink.emit(TurnActivity::new(
            lash::TurnActivityId::new("prior"),
            TurnEvent::AssistantProseDelta {
                text: "kept ".to_string(),
            },
        ))
        .await;
        sink.emit(TurnActivity::new(
            lash::TurnActivityId::new("failed"),
            TurnEvent::AssistantProseDelta {
                text: "discarded ".to_string(),
            },
        ))
        .await;
        sink.emit(TurnActivity::independent(
            TurnEvent::ModelAttemptReset {
                assistant_prose_correlation_ids: vec![lash::TurnActivityId::new("failed")],
                reasoning_correlation_ids: Vec::new(),
            },
        ))
        .await;
        sink.emit(TurnActivity::new(
            lash::TurnActivityId::new("successful"),
            TurnEvent::AssistantProseDelta {
                text: "answer".to_string(),
            },
        ))
        .await;

        assert_eq!(
            turn_state
                .lock()
                .expect("turn state lock")
                .assistant_prose(),
            "kept answer"
        );
    }
}

pub(crate) async fn enqueue_button_trigger_command(
    state: &AppState,
    button: ButtonChoice,
    pressed_at: &str,
    operation_id: &str,
    scoped_effect_controller: lash::runtime::ScopedEffectController<'_>,
) -> AnyhowResult<lash::triggers::TriggerEmitReport> {
    let payload = json!({
        "pressed_at": pressed_at,
        "button": button.as_str(),
        "message": format!("user pressed the {} button", button.lower()),
    });
    let source_key = lash::triggers::empty_trigger_source_key(BUTTON_TRIGGER_SOURCE_TYPE)
        .context("button source key")?;
    state.trace(
        "trigger.emit",
        json!({
            "resource_type": BUTTON_TRIGGER_RESOURCE,
            "alias": BUTTON_TRIGGER_ALIAS,
            "event": BUTTON_TRIGGER_EVENT,
            "source_type": BUTTON_TRIGGER_SOURCE_TYPE,
            "source_key": source_key,
            "payload": payload.clone(),
        }),
    );
    state
        .core
        .triggers()
        .emit(
            lash::triggers::TriggerOccurrenceRequest::new(
                BUTTON_TRIGGER_SOURCE_TYPE,
                source_key,
                payload,
                format!("workbench-button-trigger:{operation_id}"),
            )
            .with_source(json!({})),
            scoped_effect_controller,
        )
        .await
        .context("emit button trigger occurrence")
}

pub(crate) async fn enqueue_mail_received_trigger_command(
    state: &AppState,
    message: &mail::MailDelivery,
    operation_id: &str,
    scoped_effect_controller: lash::runtime::ScopedEffectController<'_>,
) -> AnyhowResult<lash::triggers::TriggerEmitReport> {
    let payload = json!({
        "account": message.account,
        "title": message.title,
        "text": message.text,
    });
    let source_key = lash::triggers::empty_trigger_source_key(MAIL_RECEIVED_SOURCE_TYPE)
        .context("mail source key")?;
    state.trace(
        "trigger.emit",
        json!({
            "resource_type": MAIL_EVENT_RESOURCE,
            "alias": MAIL_EVENT_ALIAS,
            "event": MAIL_EVENT_EVENT,
            "source_type": MAIL_RECEIVED_SOURCE_TYPE,
            "source_key": source_key,
            "payload": payload.clone(),
        }),
    );
    state
        .core
        .triggers()
        .emit(
            lash::triggers::TriggerOccurrenceRequest::new(
                MAIL_RECEIVED_SOURCE_TYPE,
                source_key,
                payload,
                format!("workbench-mail-trigger:{operation_id}"),
            )
            .with_source(json!({})),
            scoped_effect_controller,
        )
        .await
        .context("emit mail received trigger occurrence")
}

fn workbench_lashlang_abilities() -> lashlang::LashlangAbilities {
    lashlang::LashlangAbilities::default()
        .with_processes()
        .with_sleep()
        .with_process_signals()
        .with_triggers()
}
