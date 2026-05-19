use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use axum::Json;
use axum::body::Body;
use axum::extract::{Path as AxumPath, State};
use axum::http::{StatusCode, header};
use axum::response::{Html, Response};
use bytes::Bytes;
use lash::{TurnActivity, TurnActivitySink, TurnEvent, TurnInput, TurnOutput};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

use crate::board::BoardState;
use crate::db::{ChatMessage, ChatModelSelection, ChatSummary};
#[cfg(feature = "restate")]
use crate::restate::send_message_restate;
#[cfg(feature = "restate")]
use crate::state::AgentServiceDurability;
use crate::state::{AppError, AppResult, AppStateData};
use crate::ui::INDEX_HTML;

#[derive(Debug, Deserialize)]
pub(crate) struct CreateChatRequest {
    title: Option<String>,
    model: Option<String>,
    model_variant: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct UpdateChatModelRequest {
    model: String,
    model_variant: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SendMessageRequest {
    text: String,
    board: BoardState,
    model: Option<String>,
    model_variant: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct AppSettings {
    default_model: String,
    default_model_variant: Option<String>,
    model_variants: Vec<&'static str>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum StreamItem {
    Event { event: Box<TurnActivity> },
    Message { message: ChatMessage },
    Error { message: String },
    Done,
}

impl StreamItem {
    fn event(event: TurnActivity) -> Self {
        Self::Event {
            event: Box::new(event),
        }
    }

    pub(crate) fn is_done(&self) -> bool {
        matches!(self, Self::Done)
    }
}

pub(crate) async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

pub(crate) async fn list_chats(
    State(state): State<AppStateData>,
) -> AppResult<Json<Vec<ChatSummary>>> {
    state.with_db(|db| db.list_chats()).await.map(Json)
}

pub(crate) async fn settings(State(state): State<AppStateData>) -> Json<AppSettings> {
    Json(AppSettings {
        default_model: state.default_model().to_string(),
        default_model_variant: state.default_model_variant().map(str::to_string),
        model_variants: vec!["low", "medium", "high"],
    })
}

pub(crate) async fn create_chat(
    State(state): State<AppStateData>,
    Json(request): Json<CreateChatRequest>,
) -> AppResult<Json<ChatSummary>> {
    let title = request
        .title
        .as_deref()
        .map(str::trim)
        .filter(|title| !title.is_empty())
        .unwrap_or("New chat");
    let title = title.to_string();
    let selection = normalize_model_selection(
        request.model.as_deref(),
        request.model_variant.as_deref(),
        state.default_model(),
        state.default_model_variant(),
    )?;
    state
        .with_db(move |db| {
            db.create_chat(&title, &selection.model, selection.model_variant.as_deref())
        })
        .await
        .map(Json)
}

pub(crate) async fn update_chat_model(
    State(state): State<AppStateData>,
    AxumPath(chat_id): AxumPath<String>,
    Json(request): Json<UpdateChatModelRequest>,
) -> AppResult<Json<ChatSummary>> {
    let selection =
        normalize_optional_model_selection(Some(&request.model), request.model_variant.as_deref())?
            .ok_or_else(|| AppError::bad_request("model is required"))?;
    state
        .with_db(move |db| {
            db.update_chat_model(
                &chat_id,
                &selection.model,
                selection.model_variant.as_deref(),
            )
        })
        .await
        .map(Json)
}

pub(crate) async fn list_messages(
    State(state): State<AppStateData>,
    AxumPath(chat_id): AxumPath<String>,
) -> AppResult<Json<Vec<ChatMessage>>> {
    state
        .with_db(move |db| {
            db.require_chat(&chat_id)?;
            db.list_messages(&chat_id)
        })
        .await
        .map(Json)
}

pub(crate) async fn send_message(
    State(state): State<AppStateData>,
    AxumPath(chat_id): AxumPath<String>,
    Json(request): Json<SendMessageRequest>,
) -> AppResult<Response> {
    let text = request.text.trim().to_string();
    if text.is_empty() {
        return Err(AppError::bad_request("message text is required"));
    }
    let request_model = normalize_optional_model_selection(
        request.model.as_deref(),
        request.model_variant.as_deref(),
    )?;

    // The board is app-owned state. The user message keeps a snapshot for UI
    // replay, while tools read and mutate the canonical copy in the database.
    let user_board = request.board.clone();
    let (user_message, model_selection) = state
        .with_db({
            let chat_id = chat_id.clone();
            let text = text.clone();
            let user_board = user_board.clone();
            move |db| {
                db.require_chat(&chat_id)?;
                if let Some(selection) = request_model {
                    db.update_chat_model(
                        &chat_id,
                        &selection.model,
                        selection.model_variant.as_deref(),
                    )?;
                }
                let model_selection = db.chat_model_selection(&chat_id)?;
                db.maybe_title_from_first_message(&chat_id, &text)?;
                db.upsert_chat_board(&chat_id, &user_board)?;
                let message = db.insert_message_with_payload(
                    &chat_id,
                    "user",
                    &text,
                    Some(json!({ "board": user_board })),
                )?;
                Ok((message, model_selection))
            }
        })
        .await?;

    #[cfg(feature = "restate")]
    if state.durability() == AgentServiceDurability::Restate {
        return send_message_restate(state, chat_id, text, user_message, model_selection).await;
    }

    let session = state.open_session(&chat_id).await?;
    let (tx, rx) = mpsc::channel::<StreamItem>(64);
    let run_state = state.clone();
    tokio::spawn(async move {
        let _ = tx
            .send(StreamItem::Message {
                message: user_message,
            })
            .await;
        let turn_state = Arc::new(Mutex::new(TurnPersistenceState::default()));
        let ui_events = ChannelTurnEvents::streaming(
            tx.clone(),
            run_state.clone(),
            chat_id.clone(),
            Arc::clone(&turn_state),
        );
        let turn = session
            .turn(TurnInput::text(text))
            .model(model_selection.model, model_selection.model_variant)
            .require_submit();
        let turn = match turn {
            Ok(turn) => turn.collect_with(&ui_events).await,
            Err(err) => Err(err),
        };
        match turn {
            Ok(output) => {
                let assistant_text = assistant_text_for_persistence(
                    &output,
                    turn_state
                        .lock()
                        .expect("turn state lock")
                        .assistant_prose(),
                );
                let inserted = run_state
                    .with_db({
                        let chat_id = chat_id.clone();
                        move |db| db.insert_message(&chat_id, "assistant", &assistant_text)
                    })
                    .await;
                match inserted {
                    Ok(message) => {
                        let _ = tx.send(StreamItem::Message { message }).await;
                    }
                    Err(err) => {
                        let _ = tx
                            .send(StreamItem::Error {
                                message: err.message,
                            })
                            .await;
                    }
                }
            }
            Err(err) => {
                let _ = tx
                    .send(StreamItem::Error {
                        message: err.to_string(),
                    })
                    .await;
            }
        }
        let _ = tx.send(StreamItem::Done).await;
    });

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

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/x-ndjson; charset=utf-8")
        .header(header::CACHE_CONTROL, "no-store")
        .body(Body::from_stream(stream))
        .expect("valid streaming response"))
}

pub(crate) struct ChannelTurnEvents {
    tx: Option<mpsc::Sender<StreamItem>>,
    state: AppStateData,
    chat_id: String,
    turn_id: Option<String>,
    turn_state: Arc<Mutex<TurnPersistenceState>>,
}

#[derive(Default)]
pub(crate) struct TurnPersistenceState {
    reasoning: Option<(i64, String)>,
    assistant_prose: String,
    code: Option<String>,
    code_message: Option<i64>,
    tools: HashMap<String, i64>,
}

impl TurnPersistenceState {
    pub(crate) fn assistant_prose(&self) -> &str {
        &self.assistant_prose
    }
}

impl ChannelTurnEvents {
    pub(crate) fn streaming(
        tx: mpsc::Sender<StreamItem>,
        state: AppStateData,
        chat_id: String,
        turn_state: Arc<Mutex<TurnPersistenceState>>,
    ) -> Self {
        Self {
            tx: Some(tx),
            state,
            chat_id,
            turn_id: None,
            turn_state,
        }
    }

    #[cfg(feature = "restate")]
    pub(crate) fn outbox(
        state: AppStateData,
        chat_id: String,
        turn_id: String,
        turn_state: Arc<Mutex<TurnPersistenceState>>,
    ) -> Self {
        Self {
            tx: None,
            state,
            chat_id,
            turn_id: Some(turn_id),
            turn_state,
        }
    }

    async fn emit_item(&self, item: StreamItem) {
        if let Some(turn_id) = self.turn_id.clone() {
            let item_for_db = item.clone();
            if let Err(err) = self
                .state
                .with_db(move |db| {
                    let is_done = item_for_db.is_done();
                    db.insert_turn_event(&turn_id, &item_for_db, is_done)
                })
                .await
                && let Some(tx) = &self.tx
            {
                let _ = tx
                    .send(StreamItem::Error {
                        message: err.message,
                    })
                    .await;
            }
        }
        if let Some(tx) = &self.tx {
            let _ = tx.send(item).await;
        }
    }

    async fn handle(&self, activity: TurnActivity) {
        let event = &activity.event;
        if let TurnEvent::AssistantProseDelta { text } = &event {
            self.turn_state
                .lock()
                .expect("turn state lock")
                .assistant_prose
                .push_str(text);
            let _ = self.emit_item(StreamItem::event(activity)).await;
            return;
        }
        // Keep persisted message order tied to event start order. The browser
        // only renders completed code/tool rows, but reload should still
        // reconstruct "thinking -> lashlang -> tools -> assistant".
        if let TurnEvent::ReasoningDelta { text } = &event {
            let update = {
                let mut state = self.turn_state.lock().expect("turn state lock");
                match state.reasoning.as_mut() {
                    Some((id, existing)) => {
                        existing.push_str(text);
                        Some((*id, existing.clone(), false))
                    }
                    None => Some((0, text.clone(), true)),
                }
            };
            if let Some((id, reasoning, insert)) = update {
                let result = if insert {
                    self.state
                        .with_db({
                            let chat_id = self.chat_id.clone();
                            let reasoning = reasoning.clone();
                            move |db| db.insert_reasoning(&chat_id, &reasoning)
                        })
                        .await
                        .map(|message| {
                            self.turn_state.lock().expect("turn state lock").reasoning =
                                Some((message.id, reasoning));
                        })
                } else {
                    self.state
                        .with_db({
                            let reasoning = reasoning.clone();
                            move |db| db.update_reasoning(id, &reasoning)
                        })
                        .await
                        .map(|_| ())
                };
                if let Err(err) = result {
                    self.emit_item(StreamItem::Error {
                        message: err.message,
                    })
                    .await;
                }
            }
            let _ = self.emit_item(StreamItem::event(activity)).await;
            return;
        }
        if let TurnEvent::CodeBlockStarted { code, .. } = &event {
            self.turn_state.lock().expect("turn state lock").code = Some(code.clone());
            match self
                .state
                .with_db({
                    let chat_id = self.chat_id.clone();
                    let event = event.clone();
                    let code = code.clone();
                    move |db| db.insert_code_block(&chat_id, event, Some(code))
                })
                .await
            {
                Ok(message) => {
                    self.turn_state
                        .lock()
                        .expect("turn state lock")
                        .code_message = Some(message.id);
                }
                Err(err) => {
                    self.emit_item(StreamItem::Error {
                        message: err.message,
                    })
                    .await;
                }
            }
            let _ = self.emit_item(StreamItem::event(activity)).await;
            return;
        }
        if matches!(&event, TurnEvent::ToolCallStarted { .. }) {
            match self
                .state
                .with_db({
                    let chat_id = self.chat_id.clone();
                    let event = event.clone();
                    move |db| db.insert_tool_call(&chat_id, event)
                })
                .await
            {
                Ok(message) => {
                    self.turn_state
                        .lock()
                        .expect("turn state lock")
                        .tools
                        .insert(activity.correlation_id.0.clone(), message.id);
                }
                Err(err) => {
                    self.emit_item(StreamItem::Error {
                        message: err.message,
                    })
                    .await;
                }
            }
            let _ = self.emit_item(StreamItem::event(activity)).await;
            return;
        }
        if matches!(&event, TurnEvent::ToolCallCompleted { .. }) {
            let existing = self
                .turn_state
                .lock()
                .expect("turn state lock")
                .tools
                .remove(&activity.correlation_id.0);
            let result = self
                .state
                .with_db({
                    let chat_id = self.chat_id.clone();
                    let event = event.clone();
                    move |db| {
                        if let Some(id) = existing {
                            db.update_tool_call(id, event)
                        } else {
                            db.insert_tool_call(&chat_id, event)
                        }
                    }
                })
                .await;
            if let Err(err) = result {
                self.emit_item(StreamItem::Error {
                    message: err.message,
                })
                .await;
            }
            let _ = self.emit_item(StreamItem::event(activity)).await;
            return;
        }
        if matches!(&event, TurnEvent::CodeBlockCompleted { .. }) {
            let (code, existing) = {
                let mut state = self.turn_state.lock().expect("turn state lock");
                (state.code.take(), state.code_message.take())
            };
            let result = self
                .state
                .with_db({
                    let chat_id = self.chat_id.clone();
                    let event = event.clone();
                    move |db| {
                        if let Some(id) = existing {
                            db.update_code_block(id, event, code)
                        } else {
                            db.insert_code_block(&chat_id, event, code)
                        }
                    }
                })
                .await;
            if let Err(err) = result {
                self.emit_item(StreamItem::Error {
                    message: err.message,
                })
                .await;
            }
            let _ = self.emit_item(StreamItem::event(activity)).await;
            return;
        }
        if matches!(
            &event,
            TurnEvent::SubmittedValue { .. } | TurnEvent::ToolValue { .. }
        ) {
            let _ = self.emit_item(StreamItem::event(activity)).await;
            return;
        }
        let _ = self.emit_item(StreamItem::event(activity)).await;
    }
}

#[async_trait]
impl TurnActivitySink for ChannelTurnEvents {
    async fn emit(&self, activity: TurnActivity) {
        self.handle(activity).await;
    }
}

fn normalize_model_selection(
    model: Option<&str>,
    model_variant: Option<&str>,
    default_model: &str,
    default_model_variant: Option<&str>,
) -> AppResult<ChatModelSelection> {
    let model = model
        .map(str::trim)
        .filter(|model| !model.is_empty())
        .unwrap_or(default_model)
        .to_string();
    let model_variant = normalize_model_variant(model_variant).or_else(|| {
        default_model_variant
            .map(str::trim)
            .filter(|variant| !variant.is_empty())
            .map(str::to_string)
    });
    if model.trim().is_empty() {
        return Err(AppError::bad_request("model is required"));
    }
    Ok(ChatModelSelection {
        model,
        model_variant,
    })
}

fn normalize_optional_model_selection(
    model: Option<&str>,
    model_variant: Option<&str>,
) -> AppResult<Option<ChatModelSelection>> {
    let Some(model) = model.map(str::trim).filter(|model| !model.is_empty()) else {
        return Ok(None);
    };
    Ok(Some(ChatModelSelection {
        model: model.to_string(),
        model_variant: normalize_model_variant(model_variant),
    }))
}

fn normalize_model_variant(model_variant: Option<&str>) -> Option<String> {
    model_variant
        .map(str::trim)
        .filter(|variant| !variant.is_empty())
        .map(str::to_string)
}

pub(crate) fn assistant_text_for_persistence(output: &TurnOutput, streamed_prose: &str) -> String {
    if let Some(value) = output.submitted_value() {
        return terminal_value_text(value);
    }
    if let Some((_tool_name, value)) = output.tool_value() {
        return terminal_value_text(value);
    }
    output
        .assistant_message()
        .filter(|text| !text.trim().is_empty())
        .unwrap_or(streamed_prose)
        .to_string()
}

fn terminal_value_text(value: &serde_json::Value) -> String {
    value
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(|| value.to_string())
}
