use std::collections::HashMap;
use std::sync::{Arc, Mutex as StdMutex};

use serde_json::json;
use tokio::sync::Notify;

use lash::{
    AgentEvent, InputItem, ProgressSender, SandboxMessage, ToolExecutionContext, ToolResult,
    TurnInput,
};

use super::{DelegateTools, policy::Tier};

pub(super) struct RunningDelegate {
    pub(super) session_id: String,
    pub(super) parent_session_id: Option<String>,
    pub(super) turn_id: String,
    pub(super) host: Arc<dyn lash::SessionManager>,
    pub(super) task: String,
    pub(super) model: String,
    pub(super) model_variant: Option<String>,
    pub(super) buffer: Arc<StdMutex<String>>,
    pub(super) result: Arc<StdMutex<Option<serde_json::Value>>>,
    pub(super) done_notify: Arc<Notify>,
}

impl DelegateTools {
    pub(super) async fn spawn_agent(
        &self,
        args: &serde_json::Value,
        context: &ToolExecutionContext,
    ) -> ToolResult {
        let prompt = args
            .get("prompt")
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        if prompt.is_empty() {
            return ToolResult::err(json!("Missing required parameter: prompt"));
        }

        let intelligence = args
            .get("intelligence")
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        let tier = match Tier::from_str(intelligence) {
            Some(tier) => tier,
            None => {
                return ToolResult::err(json!(
                    "Missing or invalid 'intelligence' parameter: must be \"low\", \"medium\", or \"high\""
                ));
            }
        };

        let schema = args
            .get("schema")
            .and_then(|value| value.as_str())
            .map(str::to_string);

        let agent_id = uuid::Uuid::new_v4().to_string();
        let session = match context
            .host
            .create_session(self.build_create_request(agent_id, context.session_id.clone(), &tier))
            .await
        {
            Ok(session) => session,
            Err(err) => {
                return ToolResult::err_fmt(format_args!("Failed to create child session: {err}"));
            }
        };
        let agent_execution_mode = session.policy.execution_mode;

        let user_content = if let Some(schema_str) = schema {
            let model_name = serde_json::from_str::<serde_json::Value>(&schema_str)
                .ok()
                .and_then(|value| {
                    value
                        .get("title")
                        .and_then(|title| title.as_str())
                        .map(str::to_string)
                })
                .unwrap_or_else(|| "Result".to_string());
            match agent_execution_mode {
                lash::ExecutionMode::Repl => format!(
                    "{prompt}\n\nWhen you are done, reply in plain text with a single JSON object matching this schema exactly:\n{schema_str}\n\nDo not wrap the final JSON in `<repl>`, markdown fences, or extra commentary. The object should represent a `{model_name}`."
                ),
                lash::ExecutionMode::Standard => format!(
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
                return ToolResult::err_fmt(format_args!("Failed to start child session: {err}"));
            }
        };

        let buffer = Arc::new(StdMutex::new(String::new()));
        let context_chunks = Arc::new(StdMutex::new(Vec::new()));
        let result = Arc::new(StdMutex::new(None));
        let done_notify = Arc::new(Notify::new());

        let buffer_clone = Arc::clone(&buffer);
        let context_chunks_clone = Arc::clone(&context_chunks);
        let result_clone = Arc::clone(&result);
        let done_notify_clone = Arc::clone(&done_notify);
        let host = Arc::clone(&context.host);
        let turn_id = turn.turn_id.clone();
        let session_id = session.session_id.clone();
        let parent_session_id = session.parent_session_id.clone();
        let task = prompt.to_string();

        tokio::spawn(async move {
            let mut final_message: Option<String> = None;
            let mut current_prose = String::new();

            while let Some(event) = turn.events.recv().await {
                match event {
                    AgentEvent::TextDelta { content } => {
                        current_prose.push_str(&content);
                        buffer_clone.lock().unwrap().push_str(&content);
                    }
                    AgentEvent::Message { text, kind } => {
                        if kind == "final" {
                            final_message = Some(text);
                        }
                    }
                    AgentEvent::CodeBlock { .. } => {
                        let trimmed = current_prose.trim().to_string();
                        if !trimmed.is_empty() {
                            context_chunks_clone.lock().unwrap().push(trimmed);
                        }
                        current_prose.clear();
                    }
                    _ => {}
                }
            }

            let assembled = host.await_turn(&turn_id).await;
            let mut result_json = match assembled {
                Ok(turn) => {
                    let result_text = if let Some(message) = final_message {
                        message
                    } else if !current_prose.trim().is_empty() {
                        current_prose.trim().to_string()
                    } else if !turn.assistant_output.raw_text.trim().is_empty() {
                        turn.assistant_output.raw_text.trim().to_string()
                    } else {
                        context_chunks_clone.lock().unwrap().join("\n\n")
                    };
                    let status = match turn.status {
                        lash::TurnStatus::Completed => "completed",
                        lash::TurnStatus::Interrupted => "interrupted",
                        lash::TurnStatus::Failed => "failed",
                    };
                    let tool_call_summaries = turn
                        .state
                        .tool_calls
                        .iter()
                        .map(|tool_call| {
                            json!({
                                "tool": tool_call.tool,
                                "success": tool_call.success,
                                "duration_ms": tool_call.duration_ms,
                            })
                        })
                        .collect::<Vec<_>>();

                    json!({
                        "result": result_text,
                        "status": status,
                        "session": {
                            "id": session_id,
                            "parent_session_id": parent_session_id,
                            "task": task,
                            "iterations": turn.state.iteration,
                            "tool_calls": turn.state.tool_calls.len(),
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

            *result_clone.lock().unwrap() = Some(result_json);
            done_notify_clone.notify_waiters();
        });

        self.delegates.lock().unwrap().insert(
            session.session_id.clone(),
            RunningDelegate {
                session_id: session.session_id.clone(),
                parent_session_id: session.parent_session_id.clone(),
                turn_id: turn.turn_id,
                host: Arc::clone(&context.host),
                task: prompt.to_string(),
                model: session.policy.model.clone(),
                model_variant: session.policy.model_variant.clone(),
                buffer,
                result,
                done_notify,
            },
        );

        ToolResult::ok(json!({
            "__handle__": "agent",
            "id": session.session_id,
            "parent_session_id": session.parent_session_id,
            "model": session.policy.model,
            "model_variant": session.policy.model_variant,
            "execution_mode": session.policy.execution_mode,
        }))
    }

    pub(super) async fn agent_result(
        &self,
        id: &str,
        timeout: Option<f64>,
        progress: Option<&ProgressSender>,
    ) -> ToolResult {
        let (
            result,
            done_notify,
            buffer,
            session_id,
            parent_session_id,
            turn_id,
            host,
            task,
            model,
            model_variant,
        ) = {
            let delegates = self.delegates.lock().unwrap();
            match delegates.get(id) {
                Some(delegate) => (
                    Arc::clone(&delegate.result),
                    Arc::clone(&delegate.done_notify),
                    Arc::clone(&delegate.buffer),
                    delegate.session_id.clone(),
                    delegate.parent_session_id.clone(),
                    delegate.turn_id.clone(),
                    Arc::clone(&delegate.host),
                    delegate.task.clone(),
                    delegate.model.clone(),
                    delegate.model_variant.clone(),
                ),
                None => return ToolResult::err_fmt(format_args!("No agent with id: {id}")),
            }
        };

        let deadline = timeout
            .map(|value| tokio::time::Instant::now() + std::time::Duration::from_secs_f64(value));
        let mut sent_len = 0usize;
        let mut sent_start = false;

        loop {
            let done = result.lock().unwrap().is_some();

            if let Some(progress) = progress {
                if !sent_start {
                    let _ = progress.send(SandboxMessage {
                        text: json!({
                            "name": "delegate",
                            "task": task.clone(),
                            "model": model.clone(),
                            "model_variant": model_variant.clone(),
                            "id": session_id.clone(),
                            "parent_session_id": parent_session_id.clone(),
                        })
                        .to_string(),
                        kind: "delegate_start".into(),
                    });
                    sent_start = true;
                }

                let buffer = buffer.lock().unwrap();
                if buffer.len() > sent_len {
                    let new_chunk = &buffer[sent_len..];
                    let _ = progress.send(SandboxMessage {
                        text: new_chunk.to_string(),
                        kind: "tool_output".into(),
                    });
                    sent_len = buffer.len();
                }
            }

            if done {
                break;
            }

            if let Some(deadline) = deadline
                && tokio::time::Instant::now() >= deadline
            {
                let _ = host.cancel_turn(&turn_id).await;
                let _ =
                    tokio::time::timeout(std::time::Duration::from_secs(5), done_notify.notified())
                        .await;
                return ToolResult::err(json!("Agent timed out"));
            }

            tokio::select! {
                _ = done_notify.notified() => {}
                _ = tokio::time::sleep(std::time::Duration::from_millis(50)) => {}
            }
        }

        let result = result.lock().unwrap();
        ToolResult::ok(result.clone().unwrap_or(serde_json::Value::Null))
    }

    pub(super) async fn agent_kill(&self, id: &str) -> ToolResult {
        let (turn_id, done_notify, host) = {
            let delegates = self.delegates.lock().unwrap();
            match delegates.get(id) {
                Some(delegate) => (
                    delegate.turn_id.clone(),
                    Arc::clone(&delegate.done_notify),
                    Arc::clone(&delegate.host),
                ),
                None => return ToolResult::err_fmt(format_args!("No agent with id: {id}")),
            }
        };

        let _ = host.cancel_turn(&turn_id).await;
        let _ =
            tokio::time::timeout(std::time::Duration::from_secs(5), done_notify.notified()).await;
        ToolResult::ok(json!(null))
    }
}
