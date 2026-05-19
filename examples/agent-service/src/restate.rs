#![cfg(feature = "restate")]

use std::convert::Infallible;
use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{StatusCode, header};
use axum::response::Response;
use bytes::Bytes;
use lash::advanced::RuntimeErrorCode;
use lash::{TurnInput, TurnOutput};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::db::{ChatMessage, ChatModelSelection};
use crate::routes::{
    ChannelTurnEvents, StreamItem, TurnPersistenceState, assistant_text_for_persistence,
};
use crate::state::{AppError, AppResult, AppStateData};

#[cfg(feature = "restate")]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct AgentServiceTurnWorkflowRequest {
    turn_id: String,
    chat_id: String,
    text: String,
    model: String,
    model_variant: Option<String>,
}

impl AgentServiceTurnWorkflowRequest {
    fn new(
        turn_id: String,
        chat_id: String,
        text: String,
        model: String,
        model_variant: Option<String>,
    ) -> Self {
        Self {
            turn_id,
            chat_id,
            text,
            model,
            model_variant,
        }
    }
}

#[cfg(feature = "restate")]
#[restate_sdk::workflow]
pub(crate) trait AgentServiceTurnWorkflow {
    async fn run(
        request: restate_sdk::serde::Json<AgentServiceTurnWorkflowRequest>,
    ) -> restate_sdk::errors::HandlerResult<restate_sdk::serde::Json<()>>;
}

#[cfg(feature = "restate")]
pub(crate) struct AgentServiceTurnWorkflowImpl {
    state: AppStateData,
}

impl AgentServiceTurnWorkflowImpl {
    pub(crate) fn new(state: AppStateData) -> Self {
        Self { state }
    }
}

#[cfg(feature = "restate")]
impl AgentServiceTurnWorkflow for AgentServiceTurnWorkflowImpl {
    async fn run(
        &self,
        ctx: restate_sdk::prelude::WorkflowContext<'_>,
        restate_sdk::serde::Json(request): restate_sdk::serde::Json<
            AgentServiceTurnWorkflowRequest,
        >,
    ) -> restate_sdk::errors::HandlerResult<restate_sdk::serde::Json<()>> {
        let controller = lash_restate::RestateRuntimeEffectController::new(ctx);
        run_restate_chat_turn_and_persist(self.state.clone(), request, &controller)
            .await
            .map_err(restate_sdk::errors::TerminalError::from_error)?;
        Ok(restate_sdk::serde::Json(()))
    }
}

#[cfg(feature = "restate")]
pub(crate) async fn send_message_restate(
    state: AppStateData,
    chat_id: String,
    text: String,
    user_message: ChatMessage,
    model_selection: ChatModelSelection,
) -> AppResult<Response> {
    let turn_id = uuid::Uuid::new_v4().to_string();
    state
        .with_db({
            let turn_id = turn_id.clone();
            let user_message = user_message.clone();
            move |db| {
                let item = StreamItem::Message {
                    message: user_message,
                };
                db.insert_turn_event(&turn_id, &item, item.is_done())
            }
        })
        .await?;

    let request = AgentServiceTurnWorkflowRequest::new(
        turn_id.clone(),
        chat_id,
        text,
        model_selection.model,
        model_selection.model_variant,
    );
    let ingress = state
        .restate_ingress_url()
        .map(str::to_string)
        .ok_or_else(|| AppError::internal("Restate ingress URL is not configured"))?;
    let url = format!(
        "{}/AgentServiceTurnWorkflow/{}/run/send",
        ingress.trim_end_matches('/'),
        turn_id
    );
    let response = state
        .restate_http()
        .post(url)
        .json(&request)
        .send()
        .await
        .map_err(|err| AppError::internal(format!("Restate workflow submit failed: {err}")))?;
    if !response.status().is_success() {
        return Err(AppError::internal(format!(
            "Restate workflow submit failed with status {}",
            response.status()
        )));
    }

    stream_turn_outbox(state, turn_id).await
}

#[cfg(feature = "restate")]
async fn stream_turn_outbox(state: AppStateData, turn_id: String) -> AppResult<Response> {
    let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(64);
    tokio::spawn(async move {
        let mut last_id = 0_i64;
        loop {
            match state
                .with_db({
                    let turn_id = turn_id.clone();
                    move |db| db.list_turn_events_after(&turn_id, last_id)
                })
                .await
            {
                Ok(events) => {
                    let mut done = false;
                    for event in events {
                        last_id = event.id;
                        if event.is_done {
                            done = true;
                        }
                        let mut line = event.item_json;
                        line.push('\n');
                        if tx.send(Ok(Bytes::from(line))).await.is_err() {
                            return;
                        }
                    }
                    if done {
                        return;
                    }
                }
                Err(err) => {
                    let mut line = json!({
                        "type": "error",
                        "message": err.message,
                    })
                    .to_string();
                    line.push('\n');
                    let _ = tx.send(Ok(Bytes::from(line))).await;
                    return;
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        }
    });
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/x-ndjson; charset=utf-8")
        .header(header::CACHE_CONTROL, "no-store")
        .body(Body::from_stream(ReceiverStream::new(rx)))
        .expect("valid streaming response"))
}

#[cfg(feature = "restate")]
async fn run_restate_chat_turn_and_persist(
    state: AppStateData,
    request: AgentServiceTurnWorkflowRequest,
    controller: &lash_restate::RestateRuntimeEffectController<
        '_,
        restate_sdk::prelude::WorkflowContext<'_>,
    >,
) -> AppResult<()> {
    let session = state.open_session(&request.chat_id).await?;
    let turn_state = Arc::new(Mutex::new(TurnPersistenceState::default()));
    let ui_events = ChannelTurnEvents::outbox(
        state.clone(),
        request.chat_id.clone(),
        request.turn_id.clone(),
        Arc::clone(&turn_state),
    );

    let resume_scope = controller
        .effect_scope(&request.turn_id)
        .map_err(|err| AppError::internal(err.to_string()))?;
    let resume = session
        .resume_turn(request.turn_id.clone())
        .stream_with_effect_scope(&ui_events, resume_scope)
        .await;
    let output = route_restate_resume_or_start_fresh(resume, || async {
        let fresh_scope = controller
            .effect_scope(&request.turn_id)
            .map_err(|err| AppError::internal(err.to_string()))?;
        let mut input = TurnInput::text(request.text.clone());
        input.trace_turn_id = Some(request.turn_id.clone());
        let turn = session
            .turn(input)
            .model(request.model.clone(), request.model_variant.clone())
            .require_submit();
        Ok(turn?
            .stream_with_effect_scope(&ui_events, fresh_scope)
            .await?)
    })
    .await;

    match output {
        Ok(output) => {
            let assistant_text = assistant_text_for_persistence(
                &TurnOutput {
                    result: output,
                    activities: Vec::new(),
                },
                turn_state
                    .lock()
                    .expect("turn state lock")
                    .assistant_prose(),
            );
            let message = state
                .with_db({
                    let chat_id = request.chat_id.clone();
                    move |db| db.insert_message(&chat_id, "assistant", &assistant_text)
                })
                .await?;
            state
                .with_db({
                    let turn_id = request.turn_id.clone();
                    move |db| {
                        let item = StreamItem::Message { message };
                        db.insert_turn_event(&turn_id, &item, item.is_done())
                    }
                })
                .await?;
        }
        Err(err) => {
            state
                .with_db({
                    let turn_id = request.turn_id.clone();
                    let message = err.to_string();
                    move |db| {
                        let item = StreamItem::Error { message };
                        db.insert_turn_event(&turn_id, &item, item.is_done())
                    }
                })
                .await?;
        }
    }
    state
        .with_db({
            let turn_id = request.turn_id;
            move |db| {
                let item = StreamItem::Done;
                db.insert_turn_event(&turn_id, &item, item.is_done())
            }
        })
        .await?;
    Ok(())
}

#[cfg(feature = "restate")]
async fn route_restate_resume_or_start_fresh<Output, Fresh, FreshFuture>(
    resume: lash::Result<Output>,
    fresh: Fresh,
) -> AppResult<Output>
where
    Fresh: FnOnce() -> FreshFuture,
    FreshFuture: std::future::Future<Output = AppResult<Output>>,
{
    match resume {
        Ok(output) => Ok(output),
        Err(err) if is_runtime_turn_checkpoint_missing(&err) => fresh().await,
        Err(err) => Err(err.into()),
    }
}

#[cfg(feature = "restate")]
fn is_runtime_turn_checkpoint_missing(err: &lash::EmbedError) -> bool {
    matches!(
        err,
        lash::EmbedError::Runtime(runtime)
            if runtime.is_code(RuntimeErrorCode::RuntimeTurnCheckpointMissing)
    )
}

#[cfg(all(test, feature = "restate"))]
mod restate_tests {
    use std::net::SocketAddr;
    use std::path::Path;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::board::BoardState;
    use crate::db::AppDb;
    use crate::state::AgentServiceDurability;
    use lash::advanced::LocalBackgroundTaskRegistry;
    use lash::{LashCore, ModeId, ModePreset};
    use lash_core::LlmOutputPart;
    use lash_core::llm::types::LlmResponse;

    fn runtime_error(code: RuntimeErrorCode) -> lash::EmbedError {
        lash::advanced::RuntimeError::new(code, "test resume failure").into()
    }

    #[tokio::test]
    async fn checkpoint_missing_resume_starts_fresh_scoped_turn() {
        let fresh_calls = Arc::new(AtomicUsize::new(0));
        let fresh_calls_for_closure = Arc::clone(&fresh_calls);

        let output = route_restate_resume_or_start_fresh(
            Err(runtime_error(
                RuntimeErrorCode::RuntimeTurnCheckpointMissing,
            )),
            move || async move {
                fresh_calls_for_closure.fetch_add(1, Ordering::SeqCst);
                Ok("fresh")
            },
        )
        .await
        .expect("fresh fallback");

        assert_eq!(output, "fresh");
        assert_eq!(fresh_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn non_checkpoint_resume_error_does_not_start_fresh_turn() {
        let fresh_calls = Arc::new(AtomicUsize::new(0));
        let fresh_calls_for_closure = Arc::clone(&fresh_calls);

        let err = route_restate_resume_or_start_fresh(
            Err(runtime_error(RuntimeErrorCode::RuntimeTurnCheckpointLoad)),
            move || async move {
                fresh_calls_for_closure.fetch_add(1, Ordering::SeqCst);
                Ok("fresh")
            },
        )
        .await
        .expect_err("resume error should propagate");

        assert_eq!(fresh_calls.load(Ordering::SeqCst), 0);
        assert!(err.message.contains("test resume failure"));
    }

    #[tokio::test]
    #[ignore = "requires a running Restate server; set RESTATE_INGRESS_URL and run with --ignored"]
    async fn live_restate_ingress_runs_agent_turn_workflow_end_to_end() {
        let Some(ingress_url) = std::env::var("RESTATE_INGRESS_URL").ok() else {
            eprintln!("skipping live Restate E2E: RESTATE_INGRESS_URL is not set");
            return;
        };
        let admin_url = std::env::var("RESTATE_ADMIN_URL")
            .unwrap_or_else(|_| "http://127.0.0.1:9070".to_string());
        let bind_addr: SocketAddr = std::env::var("AGENT_SERVICE_E2E_ENDPOINT_BIND")
            .unwrap_or_else(|_| "127.0.0.1:19080".to_string())
            .parse()
            .expect("valid AGENT_SERVICE_E2E_ENDPOINT_BIND");
        let endpoint_url = std::env::var("AGENT_SERVICE_E2E_ENDPOINT_URL")
            .unwrap_or_else(|_| format!("http://{bind_addr}"));

        let temp = tempfile::tempdir().expect("tempdir");
        let state = live_restate_test_state(temp.path());
        let listener = tokio::net::TcpListener::bind(bind_addr)
            .await
            .expect("bind Restate endpoint");
        let local_addr = listener.local_addr().expect("endpoint local addr");
        let local_probe_addr = if local_addr.ip().is_unspecified() {
            SocketAddr::from(([127, 0, 0, 1], local_addr.port()))
        } else {
            local_addr
        };
        let endpoint = restate_sdk::endpoint::Endpoint::builder()
            .bind(AgentServiceTurnWorkflowImpl::new(state.clone()).serve())
            .build();
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        let server = tokio::spawn(async move {
            restate_sdk::http_server::HttpServer::new(endpoint)
                .serve_with_cancel(listener, async {
                    let _ = shutdown_rx.await;
                })
                .await;
        });

        wait_for_endpoint_socket(local_probe_addr).await;
        register_restate_deployment(&admin_url, &endpoint_url).await;

        let chat = state
            .with_db(|db| db.create_chat("Restate E2E", "mock-model", None))
            .await
            .expect("create chat");
        state
            .with_db({
                let chat_id = chat.id.clone();
                move |db| {
                    db.upsert_chat_board(
                        &chat_id,
                        &BoardState {
                            cells: vec![None; 9],
                            turn: "O".to_string(),
                        },
                    )
                }
            })
            .await
            .expect("seed board");
        let turn_id = format!("agent-service-e2e-{}", uuid::Uuid::new_v4());
        let request = AgentServiceTurnWorkflowRequest::new(
            turn_id.clone(),
            chat.id.clone(),
            "play the next move".to_string(),
            "mock-model".to_string(),
            None,
        );
        let submit = state
            .restate_http()
            .post(format!(
                "{}/AgentServiceTurnWorkflow/{}/run/send",
                ingress_url.trim_end_matches('/'),
                turn_id
            ))
            .json(&request)
            .send()
            .await
            .expect("submit workflow through Restate ingress");
        assert!(
            submit.status().is_success(),
            "Restate workflow submit failed: {}",
            submit.status()
        );

        wait_for_turn_done(&state, &turn_id).await;
        let outbox_events = state
            .with_db({
                let turn_id = turn_id.clone();
                move |db| db.list_turn_events_after(&turn_id, 0)
            })
            .await
            .expect("list turn outbox events");
        let messages = state
            .with_db({
                let chat_id = chat.id.clone();
                move |db| db.list_messages(&chat_id)
            })
            .await
            .expect("list messages");
        assert!(
            messages.iter().any(|message| message.role == "assistant"
                && message.text.contains("done via Restate E2E")),
            "assistant message was not persisted through Restate workflow; messages={messages:?}; outbox={outbox_events:?}"
        );

        let _ = shutdown_tx.send(());
        server.await.expect("endpoint server task");
    }

    fn live_restate_test_state(data_dir: &Path) -> AppStateData {
        let provider = lash_core::testing::TestProvider::builder()
            .kind("mock-provider")
            .default_model("mock-model")
            .complete(|_request| async {
                Ok(LlmResponse {
                    full_text: "```lashlang\nsubmit \"done via Restate E2E\"\n```".to_string(),
                    parts: vec![LlmOutputPart::Text {
                        text: "```lashlang\nsubmit \"done via Restate E2E\"\n```".to_string(),
                        response_meta: None,
                    }],
                    ..LlmResponse::default()
                })
            })
            .build()
            .into_handle();
        let core = LashCore::builder()
            .install_mode(ModePreset::rlm())
            .default_mode(ModeId::rlm())
            .provider(provider)
            .model("mock-model", None)
            .max_context_tokens(200_000)
            .store_factory(Arc::new(lash_sqlite_store::SqliteSessionStoreFactory::new(
                data_dir.join("lash-sessions"),
            )))
            .attachment_store(Arc::new(lash::persistence::FileAttachmentStore::new(
                data_dir.join("attachments"),
            )))
            .advanced()
            .background_task_registry(Arc::new(LocalBackgroundTaskRegistry::default()))
            .build()
            .expect("build test core");
        AppStateData::new(
            core,
            AppDb::open(&data_dir.join("app.db")).expect("open app db"),
            "mock-model".to_string(),
            None,
            AgentServiceDurability::Restate,
            std::env::var("RESTATE_INGRESS_URL").ok(),
        )
    }

    async fn wait_for_endpoint_socket(addr: SocketAddr) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            if tokio::net::TcpStream::connect(addr).await.is_ok() {
                return;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "Restate endpoint did not open a TCP listener at {addr}"
            );
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
    }

    async fn register_restate_deployment(admin_url: &str, endpoint_url: &str) {
        let client = reqwest::Client::builder()
            .http2_prior_knowledge()
            .build()
            .expect("build Restate admin client");
        let response = client
            .post(format!("{}/deployments", admin_url.trim_end_matches('/')))
            .json(&json!({
                "uri": endpoint_url,
                "force": true,
                "breaking": true,
            }))
            .send()
            .await
            .expect("register deployment with Restate admin API");
        assert!(
            response.status().is_success(),
            "Restate deployment registration failed: {} {}",
            response.status(),
            response.text().await.unwrap_or_default()
        );
    }

    async fn wait_for_turn_done(state: &AppStateData, turn_id: &str) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        loop {
            let events = state
                .with_db({
                    let turn_id = turn_id.to_string();
                    move |db| db.list_turn_events_after(&turn_id, 0)
                })
                .await
                .expect("list turn events");
            if events.iter().any(|event| event.is_done) {
                return;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "timed out waiting for Restate workflow turn outbox to finish"
            );
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        }
    }
}
