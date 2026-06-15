async fn healthz() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "service": "agent-workbench", "status": "ok" }))
}

async fn index() -> Html<&'static str> {
    Html(ui::INDEX_HTML)
}

async fn app_state(State(state): State<AppState>) -> Json<StateSnapshot> {
    Json(StateSnapshot {
        settings: state.settings(),
        messages: state.messages_snapshot(),
        timeline: state.timeline_snapshot(),
    })
}

async fn session_events(State(state): State<AppState>) -> Response {
    let mut events = state.event_tx.subscribe();
    let (tx, rx) = mpsc::channel::<StreamItem>(64);
    tokio::spawn(async move {
        loop {
            match events.recv().await {
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
    ndjson_response(rx)
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
    restate::submit_user_turn(
        &state,
        restate::WorkbenchTurnWorkflowRequest {
            turn_id,
            session_id: state.current_session_id(),
            text,
            model: ModelSelection::from_spec(&turn_model),
        },
    )
    .await?;
    Ok(Json(CommandAccepted { accepted: true }))
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
    state.push_message("event", format!("{} button event", request.button.lower()));
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
/// The enqueue pokes the queued-work runner, which submits a Restate workflow
/// for the batch; that workflow drains it with a durable handler context and
/// the runtime commits the refreshed surface to the SQLite session store.
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
        .rlm()
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

async fn cancel_turn(State(state): State<AppState>) -> Result<Json<CommandAccepted>, AppError> {
    let session_id = state.current_session_id();
    let cancelled = state.cancel_turns_for_session(&session_id);
    state.trace(
        "api.turn.cancel",
        json!({ "session_id": session_id, "cancelled": cancelled }),
    );
    Ok(Json(CommandAccepted {
        accepted: cancelled > 0,
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
        .rlm()
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
    state.timeline.lock().expect("timeline lock").clear();
    state.lashlang_execution.clear();
    state.mail_world.clear();
    Ok(Json(StateSnapshot {
        settings: state.settings(),
        messages: Vec::new(),
        timeline: Vec::new(),
    }))
}

async fn list_work(State(state): State<AppState>) -> Result<Json<Vec<WorkItem>>, AppError> {
    let session_id = state.current_session_id();
    let snapshot = state
        .process_observer
        .snapshot_for_session(session_id)
        .await
        .map_err(AppError::internal)?;
    let work = snapshot
        .items
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

#[derive(Default)]
struct TurnStreamState {
    assistant_prose: String,
}

struct ChannelTurnEvents {
    state: AppState,
    turn_state: Arc<Mutex<TurnStreamState>>,
}

#[async_trait]
impl TurnActivitySink for ChannelTurnEvents {
    async fn emit(&self, activity: TurnActivity) {
        if let TurnEvent::AssistantProseDelta { text } = &activity.event {
            self.turn_state
                .lock()
                .expect("turn state lock")
                .assistant_prose
                .push_str(text);
        }
        self.state.publish(StreamItem::event(activity));
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

fn button_trigger_event_type() -> lashlang::NamedDataType {
    lashlang::NamedDataType::object(
        "ui.button.Pressed",
        vec![
            lashlang::TypeField {
                name: "button".into(),
                ty: lashlang::TypeExpr::Union(vec![
                    lashlang::TypeExpr::Enum(vec!["Red".into()]),
                    lashlang::TypeExpr::Enum(vec!["Blue".into()]),
                ]),
                optional: false,
            },
            lashlang::TypeField {
                name: "message".into(),
                ty: lashlang::TypeExpr::Str,
                optional: false,
            },
            lashlang::TypeField {
                name: "pressed_at".into(),
                ty: lashlang::TypeExpr::Str,
                optional: false,
            },
        ],
    )
    .expect("valid button trigger event type")
}

fn workbench_lashlang_abilities() -> lashlang::LashlangAbilities {
    lashlang::LashlangAbilities::default()
        .with_processes()
        .with_sleep()
        .with_process_signals()
        .with_triggers()
}
