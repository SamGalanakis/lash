//! HTTP + WebSocket transport for the lash app-server.
//!
//! Exposes the same JSON-RPC protocol as the stdio transport, but over:
//! - **`GET /api/ws`** — WebSocket for bidirectional JSON-RPC (requests + notifications)
//! - **`POST /api/rpc`** — HTTP endpoint for one-shot JSON-RPC requests
//! - **`GET /api/health`** — Health check
//!
//! This mirrors the Node.js bridge in `lash-web/server/index.js` so the
//! existing React frontend can connect directly to the Rust server.

use std::net::SocketAddr;
use std::time::Duration;

use axum::extract::ws::{Message, WebSocket};
use axum::extract::{State, WebSocketUpgrade};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::{SinkExt, StreamExt};
use tokio::sync::{broadcast, mpsc};
use tokio_util::sync::CancellationToken;
use tower_http::cors::CorsLayer;

use super::protocol::RawMessage;
use super::transport::{MessageWriter, ServerMessage};

#[derive(Clone)]
struct AppState {
    inbound_tx: mpsc::Sender<RawMessage>,
    writer: MessageWriter,
}

/// Start the HTTP/WebSocket server. Blocks until the server shuts down.
pub async fn serve(
    addr: SocketAddr,
    writer: MessageWriter,
    inbound_tx: mpsc::Sender<RawMessage>,
    shutdown: CancellationToken,
) -> anyhow::Result<()> {
    let state = AppState { inbound_tx, writer };

    let app = Router::new()
        .route("/api/health", get(health_handler))
        .route("/api/rpc", post(rpc_handler))
        .route("/api/ws", get(ws_handler))
        .layer(CorsLayer::permissive())
        .with_state(state);

    eprintln!("lash app-server (HTTP)");
    eprintln!("  listening on: http://{addr}");
    eprintln!("  WebSocket:    ws://{addr}/api/ws");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(async move { shutdown.cancelled().await })
        .await?;
    Ok(())
}

// ─── Handlers ───

async fn health_handler() -> impl IntoResponse {
    Json(serde_json::json!({ "ok": true }))
}

/// HTTP JSON-RPC endpoint: send a request, wait for the matching response.
async fn rpc_handler(
    State(state): State<AppState>,
    Json(msg): Json<RawMessage>,
) -> impl IntoResponse {
    let request_id = msg.id.clone();

    if request_id.is_none() {
        let _ = state.inbound_tx.send(msg).await;
        return Json(serde_json::json!({}));
    }

    let request_id = request_id.unwrap();

    // Subscribe before sending so we don't miss the response
    let mut rx = state.writer.subscribe();

    if state.inbound_tx.send(msg).await.is_err() {
        return Json(serde_json::json!({
            "error": { "code": -32603, "message": "Server unavailable" }
        }));
    }

    let timeout_result = tokio::time::timeout(Duration::from_secs(120), async {
        loop {
            match rx.recv().await {
                Ok(ServerMessage::Response(resp)) if resp.id == request_id => {
                    return Some(resp);
                }
                Ok(_) => continue,
                Err(broadcast::error::RecvError::Closed) => return None,
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
            }
        }
    });

    match timeout_result.await {
        Ok(Some(resp)) => Json(serde_json::to_value(resp).unwrap_or_default()),
        Ok(None) => Json(serde_json::json!({
            "error": { "code": -32603, "message": "Server closed" }
        })),
        Err(_) => Json(serde_json::json!({
            "error": { "code": -32603, "message": "Request timed out" }
        })),
    }
}

async fn ws_handler(
    State(state): State<AppState>,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_ws_connection(socket, state))
}

async fn handle_ws_connection(socket: WebSocket, state: AppState) {
    tracing::info!("WebSocket client connected");
    let (mut ws_tx, mut ws_rx) = socket.split();
    let mut broadcast_rx = state.writer.subscribe();
    let inbound_tx = state.inbound_tx.clone();

    let outbound = tokio::spawn(async move {
        loop {
            match broadcast_rx.recv().await {
                Ok(msg) => {
                    let json = match &msg {
                        ServerMessage::Response(r) => serde_json::to_string(r),
                        ServerMessage::Notification(n) => serde_json::to_string(n),
                    };
                    match json {
                        Ok(text) => {
                            if ws_tx.send(Message::Text(text.into())).await.is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            tracing::error!("failed to serialize WS message: {e}");
                        }
                    }
                }
                Err(broadcast::error::RecvError::Closed) => break,
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!("WS client lagged by {n} messages");
                }
            }
        }
    });

    let inbound = tokio::spawn(async move {
        while let Some(Ok(msg)) = ws_rx.next().await {
            match msg {
                Message::Text(text) => {
                    match serde_json::from_str::<RawMessage>(&text) {
                        Ok(raw) => {
                            if inbound_tx.send(raw).await.is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            tracing::warn!("failed to parse WS JSON-RPC message: {e}");
                        }
                    }
                }
                Message::Close(_) => break,
                _ => {}
            }
        }
    });

    tokio::select! {
        _ = outbound => {}
        _ = inbound => {}
    }

    tracing::info!("WebSocket client disconnected");
}
