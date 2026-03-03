//! Transport layer for the lash app-server.
//!
//! Provides a transport-agnostic [`MessageWriter`] backed by a broadcast channel,
//! with concrete transport implementations for stdio and HTTP/WebSocket.

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::{broadcast, mpsc};

use super::protocol::{JsonRpcNotification, JsonRpcResponse, RawMessage};

/// Outbound server message (response or notification).
#[derive(Clone, Debug)]
pub enum ServerMessage {
    Response(JsonRpcResponse),
    Notification(JsonRpcNotification),
}

/// Transport-agnostic message writer.
///
/// Routes outbound JSON-RPC messages through a broadcast channel so that
/// multiple consumers (stdio writer, WebSocket clients, HTTP response waiters)
/// can each receive a copy.
#[derive(Clone)]
pub struct MessageWriter {
    tx: broadcast::Sender<ServerMessage>,
}

impl MessageWriter {
    pub fn new(tx: broadcast::Sender<ServerMessage>) -> Self {
        Self { tx }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<ServerMessage> {
        self.tx.subscribe()
    }

    pub async fn send_response(&self, response: &JsonRpcResponse) {
        let _ = self.tx.send(ServerMessage::Response(response.clone()));
    }

    pub async fn send_notification(&self, notification: &JsonRpcNotification) {
        let _ = self.tx.send(ServerMessage::Notification(notification.clone()));
    }
}

/// Read JSON-RPC messages from stdin as JSONL.
pub async fn read_stdin(tx: mpsc::Sender<RawMessage>) {
    let stdin = tokio::io::stdin();
    let reader = BufReader::new(stdin);
    let mut lines = reader.lines();

    loop {
        let line = match lines.next_line().await {
            Ok(Some(line)) => line,
            Ok(None) => break,
            Err(e) => {
                tracing::warn!("stdin read error: {e}");
                break;
            }
        };

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        match serde_json::from_str::<RawMessage>(trimmed) {
            Ok(msg) => {
                if tx.send(msg).await.is_err() {
                    break;
                }
            }
            Err(e) => {
                tracing::warn!("failed to parse incoming JSON-RPC message: {e}");
            }
        }
    }
}

/// Set up the stdio transport: stdin reader → inbound channel, broadcast → stdout writer.
///
/// Returns a [`MessageWriter`] that the server loop uses for outbound messages.
pub fn start_stdio_transport(inbound_tx: mpsc::Sender<RawMessage>) -> MessageWriter {
    let (broadcast_tx, broadcast_rx) = broadcast::channel(1024);
    let writer = MessageWriter::new(broadcast_tx);

    tokio::spawn(read_stdin(inbound_tx));
    tokio::spawn(drain_to_stdout(broadcast_rx));

    writer
}

async fn drain_to_stdout(mut rx: broadcast::Receiver<ServerMessage>) {
    let mut stdout = tokio::io::stdout();
    loop {
        match rx.recv().await {
            Ok(msg) => {
                let json = match &msg {
                    ServerMessage::Response(r) => serde_json::to_vec(r),
                    ServerMessage::Notification(n) => serde_json::to_vec(n),
                };
                match json {
                    Ok(mut buf) => {
                        buf.push(b'\n');
                        if let Err(e) = stdout.write_all(&buf).await {
                            tracing::error!("stdout write error: {e}");
                            break;
                        }
                        let _ = stdout.flush().await;
                    }
                    Err(e) => {
                        tracing::error!("failed to serialize outbound message: {e}");
                    }
                }
            }
            Err(broadcast::error::RecvError::Closed) => break,
            Err(broadcast::error::RecvError::Lagged(n)) => {
                tracing::warn!("stdout writer lagged by {n} messages");
            }
        }
    }
}
