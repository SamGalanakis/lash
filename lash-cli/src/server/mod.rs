//! `lash app-server`: JSON-RPC server for driving lash from rich clients.
//!
//! Adapted from the Codex app-server protocol. Supports bidirectional
//! communication using JSON-RPC 2.0 messages (with `"jsonrpc":"2.0"` omitted)
//! over stdio (newline-delimited JSON) or HTTP/WebSocket.
//!
//! # Lifecycle
//!
//! 1. Client opens stdio transport or connects via WebSocket.
//! 2. Client sends `initialize` with `clientInfo`, then emits `initialized`.
//! 3. Client calls `thread/start` or `thread/resume` to open a session.
//! 4. Client calls `turn/start` to send input; server streams `item/*` events.
//! 5. Server emits `turn/completed` when the agent finishes.

pub mod handler;
pub mod http;
#[allow(dead_code)]
pub mod protocol;
pub mod transport;

use std::net::SocketAddr;

use lash_core::provider::LashConfig;
use lash_core::runtime::RuntimeConfig;
use lash_core::{AgentEvent, AssembledTurn, Provider, RuntimeEngine, RuntimeError};
use tokio::sync::{broadcast, mpsc};

use handler::ServerHandler;
use protocol::*;
use transport::{MessageWriter, start_stdio_transport};

/// Active turn state managed by the server loop.
pub(crate) struct ActiveTurn {
    thread_id: String,
    turn_id: String,
    event_rx: mpsc::UnboundedReceiver<AgentEvent>,
    return_rx: tokio::sync::oneshot::Receiver<TurnReturn>,
    accumulated_text: String,
    items: Vec<ThreadItem>,
    error: Option<TurnError>,
    agent_message_item_id: Option<String>,
}

pub(crate) struct TurnReturn {
    runtime: RuntimeEngine,
    result: Result<AssembledTurn, RuntimeError>,
}

/// Run the lash app-server on stdio.
pub async fn run_server(
    config: RuntimeConfig,
    provider: Provider,
    model: String,
    lash_config: LashConfig,
) -> anyhow::Result<()> {
    tracing::info!("lash app-server starting (stdio transport)");

    let (inbound_tx, inbound_rx) = mpsc::channel(256);
    let writer = start_stdio_transport(inbound_tx);
    let handler = ServerHandler::new(writer.clone(), config, provider, model, lash_config);

    run_server_loop(writer, handler, inbound_rx).await
}

/// Run the lash app-server on HTTP/WebSocket.
pub async fn run_http_server(
    addr: SocketAddr,
    config: RuntimeConfig,
    provider: Provider,
    model: String,
    lash_config: LashConfig,
) -> anyhow::Result<()> {
    tracing::info!("lash app-server starting (HTTP transport on {addr})");

    let (broadcast_tx, _) = broadcast::channel(1024);
    let writer = MessageWriter::new(broadcast_tx);
    let (inbound_tx, inbound_rx) = mpsc::channel(256);

    let handler = ServerHandler::new(writer.clone(), config, provider, model, lash_config);

    let shutdown = tokio_util::sync::CancellationToken::new();
    let shutdown_signal = shutdown.clone();

    let http_handle = tokio::spawn({
        let shutdown = shutdown.clone();
        let http_writer = writer.clone();
        async move {
            http::serve(addr, http_writer, inbound_tx, shutdown).await
        }
    });

    tokio::select! {
        res = http_handle => {
            res??;
        }
        res = run_server_loop(writer, handler, inbound_rx) => {
            res?;
        }
        _ = tokio::signal::ctrl_c() => {
            eprintln!("\nShutting down (Ctrl-C again to force quit)...");
            shutdown_signal.cancel();

            tokio::select! {
                _ = tokio::signal::ctrl_c() => {
                    eprintln!("Force quit.");
                    std::process::exit(130);
                }
                _ = tokio::time::sleep(std::time::Duration::from_secs(3)) => {}
            }
        }
    }

    Ok(())
}

/// Transport-agnostic server loop: reads inbound messages from the channel,
/// dispatches them, and processes turn events.
async fn run_server_loop(
    writer: MessageWriter,
    mut handler: ServerHandler,
    mut inbound_rx: mpsc::Receiver<RawMessage>,
) -> anyhow::Result<()> {
    let mut active_turn: Option<ActiveTurn> = None;

    loop {
        if let Some(ref mut turn) = active_turn {
            // Order matters: check return_rx before event_rx so that when the
            // turn task finishes (dropping event_tx), we don't spin on the
            // immediately-ready None from event_rx.recv(). finish_turn() drains
            // any remaining buffered events via try_recv().
            tokio::select! {
                biased;

                msg = inbound_rx.recv() => {
                    let Some(msg) = msg else {
                        tracing::info!("inbound channel closed, shutting down");
                        return Ok(());
                    };
                    dispatch_message(&mut handler, msg).await;
                }

                return_val = &mut turn.return_rx => {
                    let turn_data = active_turn.take().unwrap();
                    match return_val {
                        Ok(ret) => {
                            finish_turn(&writer, &mut handler, turn_data, ret).await;
                        }
                        Err(_) => {
                            tracing::error!("turn task panicked");
                            finish_turn_panicked(&writer, &mut handler, turn_data).await;
                        }
                    }
                }

                event = turn.event_rx.recv() => {
                    if let Some(event) = event {
                        process_turn_event(&writer, &handler, turn, &event).await;
                    }
                }
            }
        } else {
            let Some(msg) = inbound_rx.recv().await else {
                tracing::info!("inbound channel closed, shutting down");
                break;
            };

            let maybe_turn = dispatch_message(&mut handler, msg).await;
            if let Some(turn_handle) = maybe_turn {
                active_turn = Some(turn_handle);
            }
        }
    }

    Ok(())
}

/// Dispatch a raw message to the handler. Returns an ActiveTurn if turn/start was called.
async fn dispatch_message(
    handler: &mut ServerHandler,
    msg: RawMessage,
) -> Option<ActiveTurn> {
    let method = match msg.method {
        Some(ref m) => m.clone(),
        None => {
            tracing::warn!("received message without method, ignoring");
            return None;
        }
    };

    if let Some(id) = msg.id {
        handler.handle_request(&method, id, msg.params).await
    } else {
        handler.handle_notification(&method, msg.params).await;
        None
    }
}

/// Process a single AgentEvent during an active turn, emitting JSON-RPC notifications.
async fn process_turn_event(
    writer: &MessageWriter,
    handler: &ServerHandler,
    turn: &mut ActiveTurn,
    event: &AgentEvent,
) {
    let thread_id = &turn.thread_id;
    let turn_id = &turn.turn_id;

    match event {
        AgentEvent::TextDelta { content } => {
            turn.accumulated_text.push_str(content);

            if turn.agent_message_item_id.is_none() {
                let item_id = handler.next_item_id();
                let item = ThreadItem::AgentMessage {
                    id: item_id.clone(),
                    text: String::new(),
                };
                notify(writer, handler, "item/started", serde_json::json!({ "item": item })).await;
                turn.agent_message_item_id = Some(item_id);
            }

            notify(
                writer,
                handler,
                "item/agentMessage/delta",
                serde_json::json!({
                    "threadId": thread_id,
                    "turnId": turn_id,
                    "itemId": turn.agent_message_item_id,
                    "delta": content,
                }),
            )
            .await;
        }

        AgentEvent::CodeBlock { code } => {
            finalize_agent_message(writer, handler, turn).await;

            let item_id = handler.next_item_id();
            let item = ThreadItem::CodeBlock {
                id: item_id,
                code: code.clone(),
            };
            notify(writer, handler, "item/started", serde_json::json!({ "item": item })).await;
            notify(writer, handler, "item/completed", serde_json::json!({ "item": item })).await;
            turn.items.push(item);
        }

        AgentEvent::CodeOutput { output, error } => {
            let item_id = handler.next_item_id();
            let item = ThreadItem::CodeOutput {
                id: item_id,
                output: output.clone(),
                error: error.clone(),
            };
            notify(writer, handler, "item/started", serde_json::json!({ "item": item })).await;
            notify(writer, handler, "item/completed", serde_json::json!({ "item": item })).await;
            turn.items.push(item);
        }

        AgentEvent::ToolCall {
            name,
            args,
            result,
            success,
            duration_ms,
        } => {
            let item_id = handler.next_item_id();
            let item = ThreadItem::ToolCall {
                id: item_id,
                name: name.clone(),
                args: args.clone(),
                result: Some(result.clone()),
                success: *success,
                duration_ms: Some(*duration_ms),
                status: if *success {
                    ItemStatus::Completed
                } else {
                    ItemStatus::Failed
                },
            };
            notify(writer, handler, "item/started", serde_json::json!({ "item": item })).await;
            notify(writer, handler, "item/completed", serde_json::json!({ "item": item })).await;
            turn.items.push(item);
        }

        AgentEvent::SubAgentDone {
            task,
            tool_calls,
            iterations,
            success,
            ..
        } => {
            let item_id = handler.next_item_id();
            let item = ThreadItem::SubAgentResult {
                id: item_id,
                task: task.clone(),
                success: *success,
                tool_calls: *tool_calls,
                iterations: *iterations,
            };
            notify(writer, handler, "item/started", serde_json::json!({ "item": item })).await;
            notify(writer, handler, "item/completed", serde_json::json!({ "item": item })).await;
            turn.items.push(item);
        }

        AgentEvent::RetryStatus {
            wait_seconds,
            attempt,
            max_attempts,
            reason,
        } => {
            let item_id = handler.next_item_id();
            let item = ThreadItem::RetryStatus {
                id: item_id,
                wait_seconds: *wait_seconds,
                attempt: *attempt,
                max_attempts: *max_attempts,
                reason: reason.clone(),
            };
            notify(writer, handler, "item/started", serde_json::json!({ "item": item })).await;
            turn.items.push(item);
        }

        AgentEvent::TokenUsage {
            iteration,
            usage,
            cumulative,
        } => {
            notify(
                writer,
                handler,
                "thread/tokenUsage/updated",
                serde_json::json!({
                    "threadId": thread_id,
                    "turnId": turn_id,
                    "iteration": iteration,
                    "usage": TokenUsageInfo::from(usage),
                    "cumulative": TokenUsageInfo::from(cumulative),
                }),
            )
            .await;
        }

        AgentEvent::Error {
            message, envelope, ..
        } => {
            let item_id = handler.next_item_id();
            let error_info = envelope.as_ref().map(|e| e.kind.clone());
            let item = ThreadItem::Error {
                id: item_id,
                message: message.clone(),
                error_info: error_info.clone(),
            };
            notify(writer, handler, "item/started", serde_json::json!({ "item": item })).await;
            notify(writer, handler, "item/completed", serde_json::json!({ "item": item })).await;
            turn.items.push(item);
            turn.error = Some(TurnError {
                message: message.clone(),
                error_info,
            });
        }

        AgentEvent::Prompt { response_tx, .. } => {
            let _ = response_tx.send(String::new());
        }

        AgentEvent::Message { text, kind } => {
            if kind == "final" && !text.is_empty() {
                turn.accumulated_text.clear();
                turn.accumulated_text.push_str(text);
            }
        }

        AgentEvent::Done | AgentEvent::LlmRequest { .. } | AgentEvent::LlmResponse { .. } => {}
    }
}

/// Finalize a pending agent message item (flush accumulated text).
async fn finalize_agent_message(
    writer: &MessageWriter,
    handler: &ServerHandler,
    turn: &mut ActiveTurn,
) {
    if let Some(item_id) = turn.agent_message_item_id.take() {
        let item = ThreadItem::AgentMessage {
            id: item_id,
            text: std::mem::take(&mut turn.accumulated_text),
        };
        notify(writer, handler, "item/completed", serde_json::json!({ "item": item })).await;
        turn.items.push(item);
    }
}

/// Complete a turn after the runtime task finishes.
async fn finish_turn(
    writer: &MessageWriter,
    handler: &mut ServerHandler,
    mut turn: ActiveTurn,
    ret: TurnReturn,
) {
    while let Ok(event) = turn.event_rx.try_recv() {
        process_turn_event(writer, handler, &mut turn, &event).await;
    }

    finalize_agent_message(writer, handler, &mut turn).await;

    let final_status = match ret.result {
        Ok(assembled) => {
            handler.complete_turn(&turn.thread_id, ret.runtime, assembled.state);
            if turn.error.is_some() {
                TurnStatus::Failed
            } else {
                TurnStatus::Completed
            }
        }
        Err(e) => {
            tracing::error!("turn runtime error: {e}");
            turn.error = Some(TurnError {
                message: e.to_string(),
                error_info: None,
            });
            handler.clear_active_turn(&turn.thread_id);
            TurnStatus::Failed
        }
    };

    let final_turn = Turn {
        id: turn.turn_id.clone(),
        status: final_status,
        items: turn.items,
        error: turn.error,
    };

    notify(
        writer,
        handler,
        "turn/completed",
        serde_json::json!({ "turn": final_turn }),
    )
    .await;

    notify(
        writer,
        handler,
        "thread/status/changed",
        serde_json::json!({
            "threadId": turn.thread_id,
            "status": { "type": "idle" }
        }),
    )
    .await;
}

/// Handle a turn whose spawned task panicked (return_rx was canceled).
async fn finish_turn_panicked(
    writer: &MessageWriter,
    handler: &mut ServerHandler,
    mut turn: ActiveTurn,
) {
    while let Ok(event) = turn.event_rx.try_recv() {
        process_turn_event(writer, handler, &mut turn, &event).await;
    }

    finalize_agent_message(writer, handler, &mut turn).await;
    handler.clear_active_turn(&turn.thread_id);

    let final_turn = Turn {
        id: turn.turn_id.clone(),
        status: TurnStatus::Failed,
        items: turn.items,
        error: Some(TurnError {
            message: "Internal error: turn task panicked".to_string(),
            error_info: None,
        }),
    };

    notify(
        writer,
        handler,
        "turn/completed",
        serde_json::json!({ "turn": final_turn }),
    )
    .await;

    notify(
        writer,
        handler,
        "thread/status/changed",
        serde_json::json!({
            "threadId": turn.thread_id,
            "status": { "type": "idle" }
        }),
    )
    .await;
}

/// Send a notification if the filter allows it.
async fn notify(
    writer: &MessageWriter,
    handler: &ServerHandler,
    method: &str,
    params: serde_json::Value,
) {
    if !handler.should_send_notification(method) {
        return;
    }
    writer
        .send_notification(&JsonRpcNotification {
            method: method.to_string(),
            params: Some(params),
        })
        .await;
}
