use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex as StdMutex};

use serde_json::json;
use tokio::sync::Notify;

use crate::{
    AgentEvent, InputItem, ProgressSender, SandboxMessage, ToolExecutionContext, ToolResult,
    TurnInput,
};

use super::{AgentCall, Tier};

/// A running delegated child session managed by AgentCall.
pub(super) struct RunningAgent {
    pub(super) session_id: String,
    pub(super) turn_id: String,
    pub(super) host: Arc<dyn crate::SessionManager>,
    pub(super) task: String,
    pub(super) model: String,
    pub(super) model_variant: Option<String>,
    /// Accumulated prose output from the agent (drainable).
    pub(super) buffer: Arc<StdMutex<String>>,
    /// Final result once the agent completes.
    pub(super) result: Arc<StdMutex<Option<serde_json::Value>>>,
    /// Notified when the agent finishes.
    pub(super) done_notify: Arc<Notify>,
    /// Ensures the child session is only closed once.
    pub(super) session_closed: Arc<AtomicBool>,
}

async fn close_session_once(
    host: &Arc<dyn crate::SessionManager>,
    session_id: &str,
    session_closed: &Arc<AtomicBool>,
) {
    if session_closed
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok()
    {
        let _ = host.close_session(session_id).await;
    }
}

impl AgentCall {
    /// Spawn a child session in the background and return its handle ID.
    pub(super) async fn spawn_agent(
        &self,
        args: &serde_json::Value,
        context: &ToolExecutionContext,
    ) -> ToolResult {
        let prompt = args
            .get("prompt")
            .and_then(|v| v.as_str())
            .unwrap_or_default();

        if prompt.is_empty() {
            return ToolResult::err(json!("Missing required parameter: prompt"));
        }

        let intelligence = args
            .get("intelligence")
            .and_then(|v| v.as_str())
            .unwrap_or_default();

        let tier = match Tier::from_str(intelligence) {
            Some(t) => t,
            None => {
                return ToolResult::err(json!(
                    "Missing or invalid 'intelligence' parameter: must be \"low\", \"medium\", or \"high\""
                ));
            }
        };

        let schema = args
            .get("schema")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let agent_id = uuid::Uuid::new_v4().to_string();
        let session = match context
            .host
            .create_session(self.build_create_request(agent_id, &tier))
            .await
        {
            Ok(session) => session,
            Err(err) => {
                return ToolResult::err_fmt(format_args!(
                    "Failed to create delegate session: {err}"
                ));
            }
        };
        let agent_execution_mode = session.policy.execution_mode;

        let user_content = if let Some(ref schema_str) = schema {
            let model_name = serde_json::from_str::<serde_json::Value>(schema_str)
                .ok()
                .and_then(|v| v.get("title").and_then(|t| t.as_str()).map(String::from))
                .unwrap_or_else(|| "Result".to_string());
            match agent_execution_mode {
                crate::ExecutionMode::Repl => format!(
                    "{prompt}\n\nWhen you are done, reply in plain text with a single JSON object matching this schema exactly:\n{schema_str}\n\nDo not wrap the final JSON in `<repl>`, markdown fences, or extra commentary. The object should represent a `{model_name}`."
                ),
                crate::ExecutionMode::Standard => format!(
                    "{prompt}\n\nReturn your final answer as a single JSON object matching this schema exactly:\n{schema_str}\n\nDo not wrap it in markdown fences or extra commentary."
                ),
            }
        } else {
            prompt.to_string()
        };

        let mut turn = match context
            .host
            .start_turn_stream(
                &session.session_id,
                TurnInput {
                    items: vec![InputItem::Text { text: user_content }],
                    image_blobs: HashMap::new(),
                    mode: None,
                },
            )
            .await
        {
            Ok(turn) => turn,
            Err(err) => {
                let _ = context.host.close_session(&session.session_id).await;
                return ToolResult::err_fmt(format_args!(
                    "Failed to start delegate session: {err}"
                ));
            }
        };

        let buffer = Arc::new(StdMutex::new(String::new()));
        let context_chunks = Arc::new(StdMutex::new(Vec::new()));
        let result: Arc<StdMutex<Option<serde_json::Value>>> = Arc::new(StdMutex::new(None));
        let done_notify = Arc::new(Notify::new());
        let session_closed = Arc::new(AtomicBool::new(false));

        let buf_clone = buffer.clone();
        let ctx_clone = context_chunks.clone();
        let res_clone = result.clone();
        let done_clone = done_notify.clone();
        let session_closed_clone = session_closed.clone();
        let host = Arc::clone(&context.host);
        let turn_id = turn.turn_id.clone();
        let session_id = session.session_id.clone();
        let task = prompt.to_string();

        tokio::spawn(async move {
            let mut final_message: Option<String> = None;
            let mut current_prose = String::new();

            while let Some(event) = turn.events.recv().await {
                match event {
                    AgentEvent::TextDelta { content } => {
                        current_prose.push_str(&content);
                        buf_clone.lock().unwrap().push_str(&content);
                    }
                    AgentEvent::Message { text, kind } => {
                        if kind == "final" {
                            final_message = Some(text);
                        }
                    }
                    AgentEvent::CodeBlock { .. } => {
                        let trimmed = current_prose.trim().to_string();
                        if !trimmed.is_empty() {
                            ctx_clone.lock().unwrap().push(trimmed);
                        }
                        current_prose.clear();
                    }
                    _ => {}
                }
            }

            let assembled = host.await_turn(&turn_id).await;
            let mut result_json = match assembled {
                Ok(turn) => {
                    let result_text = if let Some(msg) = final_message {
                        msg
                    } else if !current_prose.trim().is_empty() {
                        current_prose.trim().to_string()
                    } else if !turn.assistant_output.raw_text.trim().is_empty() {
                        turn.assistant_output.raw_text.trim().to_string()
                    } else {
                        ctx_clone.lock().unwrap().join("\n\n")
                    };
                    let status = match turn.status {
                        crate::TurnStatus::Completed => "completed",
                        crate::TurnStatus::Interrupted => "interrupted",
                        crate::TurnStatus::Failed => "failed",
                    };

                    // Build child tool call summaries for the UI.
                    let all_tool_calls = &turn.state.tool_calls;
                    let tool_call_summaries: Vec<serde_json::Value> = all_tool_calls
                        .iter()
                        .map(|tc| {
                            json!({
                                "tool": tc.tool,
                                "success": tc.success,
                                "duration_ms": tc.duration_ms,
                            })
                        })
                        .collect();

                    json!({
                        "result": result_text,
                        "status": status,
                        "_sub_agent": {
                            "task": task,
                            "iterations": turn.state.iteration,
                            "tool_calls": all_tool_calls.len(),
                            "tool_call_details": tool_call_summaries,
                            "model": turn.state.policy.model,
                            "model_variant": turn.state.policy.model_variant,
                            "token_usage": {
                                "input_tokens": turn.token_usage.input_tokens,
                                "output_tokens": turn.token_usage.output_tokens,
                                "cached_input_tokens": turn.token_usage.cached_input_tokens,
                                "reasoning_tokens": turn.token_usage.reasoning_tokens,
                                "total_tokens": turn.token_usage.total(),
                            },
                        },
                    })
                }
                Err(err) => json!({
                    "result": "",
                    "status": "failed",
                    "error": err.to_string(),
                }),
            };

            if result_json
                .get("result")
                .and_then(|value| value.as_str())
                .is_some_and(|value| value.is_empty())
                && !current_prose.trim().is_empty()
            {
                result_json["result"] = json!(current_prose.trim());
            }

            close_session_once(&host, &session_id, &session_closed_clone).await;
            *res_clone.lock().unwrap() = Some(result_json);
            done_clone.notify_waiters();
        });

        self.agents.lock().unwrap().insert(
            session.session_id.clone(),
            RunningAgent {
                session_id: session.session_id.clone(),
                turn_id: turn.turn_id,
                host: Arc::clone(&context.host),
                task: prompt.to_string(),
                model: session.policy.model.clone(),
                model_variant: session.policy.model_variant.clone(),
                buffer,
                result,
                done_notify,
                session_closed,
            },
        );

        ToolResult::ok(json!({
            "__handle__": "agent",
            "id": session.session_id,
            "model": session.policy.model,
            "model_variant": session.policy.model_variant,
            "execution_mode": session.policy.execution_mode,
        }))
    }

    /// Wait for agent completion and return the result.
    pub(super) async fn agent_result(
        &self,
        id: &str,
        timeout: Option<f64>,
        progress: Option<&ProgressSender>,
    ) -> ToolResult {
        let (
            result_arc,
            done_notify,
            buffer,
            session_id,
            turn_id,
            host,
            task,
            model,
            model_variant,
            session_closed,
        ) = {
            let agents = self.agents.lock().unwrap();
            match agents.get(id) {
                Some(a) => (
                    a.result.clone(),
                    a.done_notify.clone(),
                    a.buffer.clone(),
                    a.session_id.clone(),
                    a.turn_id.clone(),
                    Arc::clone(&a.host),
                    a.task.clone(),
                    a.model.clone(),
                    a.model_variant.clone(),
                    a.session_closed.clone(),
                ),
                None => return ToolResult::err_fmt(format_args!("No agent with id: {id}")),
            }
        };

        let deadline =
            timeout.map(|t| tokio::time::Instant::now() + std::time::Duration::from_secs_f64(t));

        let mut sent_len = 0usize;
        let mut sent_start = false;

        loop {
            let done = result_arc.lock().unwrap().is_some();

            if let Some(tx) = progress {
                if !sent_start {
                    let _ = tx.send(SandboxMessage {
                        text: json!({
                            "name": "delegate",
                            "task": task.clone(),
                            "model": model.clone(),
                            "model_variant": model_variant.clone(),
                            "id": session_id.clone(),
                        })
                        .to_string(),
                        kind: "delegate_start".into(),
                    });
                    sent_start = true;
                }
                let buf = buffer.lock().unwrap();
                if buf.len() > sent_len {
                    let new_chunk = &buf[sent_len..];
                    let _ = tx.send(SandboxMessage {
                        text: new_chunk.to_string(),
                        kind: "tool_output".into(),
                    });
                    sent_len = buf.len();
                }
            }

            if done {
                break;
            }

            if let Some(dl) = deadline
                && tokio::time::Instant::now() >= dl
            {
                let _ = host.cancel_turn(&turn_id).await;
                let _ =
                    tokio::time::timeout(std::time::Duration::from_secs(5), done_notify.notified())
                        .await;
                close_session_once(&host, &session_id, &session_closed).await;
                return ToolResult::err(json!("Agent timed out"));
            }

            tokio::select! {
                _ = done_notify.notified() => {}
                _ = tokio::time::sleep(std::time::Duration::from_millis(50)) => {}
            }
        }

        let result = result_arc.lock().unwrap().clone().unwrap_or(json!(null));
        ToolResult::ok(result)
    }

    /// Cancel a running agent.
    pub(super) async fn agent_kill(&self, id: &str) -> ToolResult {
        let (turn_id, session_id, done_notify, host, session_closed) = {
            let agents = self.agents.lock().unwrap();
            match agents.get(id) {
                Some(a) => (
                    a.turn_id.clone(),
                    a.session_id.clone(),
                    a.done_notify.clone(),
                    Arc::clone(&a.host),
                    a.session_closed.clone(),
                ),
                None => return ToolResult::err_fmt(format_args!("No agent with id: {id}")),
            }
        };

        let _ = host.cancel_turn(&turn_id).await;
        let _ =
            tokio::time::timeout(std::time::Duration::from_secs(5), done_notify.notified()).await;
        close_session_once(&host, &session_id, &session_closed).await;
        ToolResult::ok(json!(null))
    }
}
