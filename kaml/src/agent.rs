use std::collections::HashMap;

use tokio::sync::mpsc;

use crate::baml_client::ClientRegistry;
use crate::baml_client::async_client::B;
pub use crate::baml_client::types::ChatMsg;
use crate::session::Session;
use crate::ToolDefinition;

/// Configuration for the agent loop.
#[derive(Clone)]
pub struct AgentConfig {
    /// Model identifier (e.g. "anthropic/claude-sonnet-4-5")
    pub model: String,
    /// API key for the LLM provider
    pub api_key: String,
    /// Base URL for the API (defaults to OpenRouter)
    pub base_url: String,
    /// Maximum CodeAct iterations per run (default: 15)
    pub max_iterations: usize,
    /// Maximum total character budget for context truncation
    pub max_context_chars: usize,
    /// When true, use SubAgentStep prompt instead of CodeActStep
    pub sub_agent: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            model: "anthropic/claude-sonnet-4-5".to_string(),
            api_key: String::new(),
            base_url: "https://openrouter.ai/api/v1".to_string(),
            max_iterations: 15,
            max_context_chars: 400_000,
            sub_agent: false,
        }
    }
}

/// Events emitted during an agent run.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "type")]
pub enum AgentEvent {
    #[serde(rename = "text_delta")]
    TextDelta { content: String },
    #[serde(rename = "tool_call")]
    ToolCall {
        name: String,
        args: serde_json::Value,
        result: serde_json::Value,
        success: bool,
        duration_ms: u64,
    },
    #[serde(rename = "code_block")]
    CodeBlock { code: String },
    #[serde(rename = "code_output")]
    CodeOutput {
        output: String,
        error: Option<String>,
    },
    #[serde(rename = "message")]
    Message { text: String, kind: String },
    #[serde(rename = "llm_request")]
    LlmRequest {
        iteration: usize,
        message_count: usize,
        tool_list: String,
    },
    #[serde(rename = "llm_response")]
    LlmResponse {
        iteration: usize,
        content: String,
        duration_ms: u64,
    },
    #[serde(rename = "done")]
    Done,
    #[serde(rename = "error")]
    Error { message: String },
}

/// Timeout for waiting on LLM streaming chunks.
const LLM_STREAM_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);
/// Max retries for rate-limited or empty LLM responses.
const LLM_MAX_RETRIES: usize = 3;
/// Delays between retries (exponential backoff).
const LLM_RETRY_DELAYS: [std::time::Duration; 3] = [
    std::time::Duration::from_secs(2),
    std::time::Duration::from_secs(5),
    std::time::Duration::from_secs(10),
];

/// CodeAct agent: LLM writes Python code, REPL executes, output feeds back.
pub struct Agent {
    session: Session,
    config: AgentConfig,
}

impl Agent {
    pub fn new(session: Session, config: AgentConfig) -> Self {
        Self { session, config }
    }

    pub fn set_model(&mut self, model: String) {
        self.config.model = model;
    }

    /// Run the agent loop with user messages, emitting events.
    /// Returns the updated message history (including assistant/system feedback).
    pub async fn run(
        &mut self,
        messages: Vec<ChatMsg>,
        event_tx: mpsc::Sender<AgentEvent>,
    ) -> Vec<ChatMsg> {
        macro_rules! emit {
            ($event:expr) => {
                if !event_tx.is_closed() {
                    let _ = event_tx.send($event).await;
                }
            };
        }

        // Build ClientRegistry
        let mut cr = ClientRegistry::new();
        cr.add_llm_client(
            "DefaultClient",
            "openai-generic",
            HashMap::from([
                (
                    "base_url".into(),
                    serde_json::json!(&self.config.base_url),
                ),
                ("api_key".into(), serde_json::json!(&self.config.api_key)),
                ("model".into(), serde_json::json!(&self.config.model)),
                ("temperature".into(), serde_json::json!(0)),
                ("max_tokens".into(), serde_json::json!(16384)),
            ]),
        );
        cr.set_primary_client("DefaultClient");

        // Generate tool docs and context
        let tool_list = ToolDefinition::format_tool_docs(&self.session.tools().definitions());
        let context = build_context();

        let mut msgs = messages;

        for iteration in 0..self.config.max_iterations {
            // Truncate context to stay within budget
            truncate_messages(&mut msgs, self.config.max_context_chars);

            emit!(AgentEvent::LlmRequest {
                iteration,
                message_count: msgs.len(),
                tool_list: tool_list.clone(),
            });

            // LLM call with retry on transient API errors (rate limits, 5xx)
            let full_text = 'llm_retry: {
                let mut last_error = None;
                for attempt in 0..=LLM_MAX_RETRIES {
                    if attempt > 0 {
                        let delay = LLM_RETRY_DELAYS[attempt - 1];
                        tracing::warn!(
                            "Retrying LLM call (attempt {}/{}) after {}s",
                            attempt + 1,
                            LLM_MAX_RETRIES + 1,
                            delay.as_secs()
                        );
                        emit!(AgentEvent::Error {
                            message: format!(
                                "Retrying in {}s (attempt {}/{})...",
                                delay.as_secs(),
                                attempt + 1,
                                LLM_MAX_RETRIES + 1,
                            ),
                        });
                        tokio::time::sleep(delay).await;
                    }

                    let llm_start = std::time::Instant::now();

                    let mut call = match if self.config.sub_agent {
                        B.SubAgentStep
                            .with_client_registry(&cr)
                            .stream(&msgs, &tool_list, &context)
                    } else {
                        B.CodeActStep
                            .with_client_registry(&cr)
                            .stream(&msgs, &tool_list, &context)
                    } {
                        Ok(c) => c,
                        Err(e) => {
                            let msg = format!("{}", e);
                            if is_retryable(&msg) && attempt < LLM_MAX_RETRIES {
                                last_error = Some(msg);
                                continue;
                            }
                            emit!(AgentEvent::Error {
                                message: format!("LLM error: {}", e),
                            });
                            break 'llm_retry String::new();
                        }
                    };

                    // Stream LLM tokens incrementally
                    let mut stream_error = None;
                    let mut prev_len = 0usize;
                    loop {
                        match tokio::time::timeout(LLM_STREAM_TIMEOUT, call.next()).await {
                            Err(_timeout) => {
                                emit!(AgentEvent::Error {
                                    message: "LLM response timed out".to_string(),
                                });
                                emit!(AgentEvent::Done);
                                return msgs;
                            }
                            Ok(None) => break,
                            Ok(Some(Ok(partial))) => {
                                if partial.len() > prev_len {
                                    let delta = &partial[prev_len..];
                                    emit!(AgentEvent::TextDelta {
                                        content: delta.to_string(),
                                    });
                                    prev_len = partial.len();
                                }
                            }
                            Ok(Some(Err(e))) => {
                                stream_error = Some(format!("{}", e));
                                break;
                            }
                        }
                    }

                    if let Some(err) = stream_error {
                        if is_retryable(&err) && attempt < LLM_MAX_RETRIES {
                            last_error = Some(err);
                            continue;
                        }
                        emit!(AgentEvent::Error {
                            message: format!("LLM error: {}", err),
                        });
                        emit!(AgentEvent::Done);
                        return msgs;
                    }

                    let text = match call.get_final_response().await {
                        Ok(v) => v,
                        Err(e) => {
                            let msg = format!("{}", e);
                            if is_retryable(&msg) && attempt < LLM_MAX_RETRIES {
                                last_error = Some(msg);
                                continue;
                            }
                            emit!(AgentEvent::Error {
                                message: format!("LLM error: {}", e),
                            });
                            break 'llm_retry String::new();
                        }
                    };

                    emit!(AgentEvent::LlmResponse {
                        iteration,
                        content: text.clone(),
                        duration_ms: llm_start.elapsed().as_millis() as u64,
                    });

                    break 'llm_retry text;
                }

                // All retries exhausted
                if let Some(err) = last_error {
                    emit!(AgentEvent::Error {
                        message: format!("LLM failed after {} retries: {}", LLM_MAX_RETRIES + 1, err),
                    });
                }
                String::new()
            };

            // Extract ```python code blocks
            let code = extract_code(&full_text);

            if full_text.trim().is_empty() {
                tracing::warn!("LLM returned empty response");
                emit!(AgentEvent::TextDelta {
                    content: "I didn't get a response — please try again.".to_string()
                });
                emit!(AgentEvent::Done);
                return msgs;
            }

            // No code blocks = pure prose answer, done
            if code.is_empty() {
                msgs.push(ChatMsg {
                    role: "assistant".to_string(),
                    content: full_text,
                });
                emit!(AgentEvent::Done);
                return msgs;
            }

            emit!(AgentEvent::CodeBlock { code: code.clone() });

            // Set up message forwarding for this execution
            let (msg_tx, mut msg_rx) = tokio::sync::mpsc::unbounded_channel::<crate::SandboxMessage>();
            self.session.set_message_sender(msg_tx);
            let event_tx_clone = event_tx.clone();
            let drain_handle = tokio::spawn(async move {
                while let Some(sandbox_msg) = msg_rx.recv().await {
                    if !event_tx_clone.is_closed() {
                        let _ = event_tx_clone
                            .send(AgentEvent::Message {
                                text: sandbox_msg.text,
                                kind: sandbox_msg.kind,
                            })
                            .await;
                    }
                }
            });

            // Execute via Python REPL
            let exec_result = self.session.run_code(&code).await;

            // Clean up message forwarding
            self.session.clear_message_sender();
            let _ = drain_handle.await;

            match exec_result {
                Ok(r) => {
                    // Emit tool call events
                    for tc in &r.tool_calls {
                        emit!(AgentEvent::ToolCall {
                            name: tc.tool.clone(),
                            args: tc.args.clone(),
                            result: tc.result.clone(),
                            success: tc.success,
                            duration_ms: tc.duration_ms,
                        });
                    }

                    // Emit code output
                    if !r.output.is_empty() || r.error.is_some() {
                        emit!(AgentEvent::CodeOutput {
                            output: r.output.clone(),
                            error: r.error.clone(),
                        });
                    }

                    // message(kind="final") = stop signal
                    if !r.response.is_empty() {
                        emit!(AgentEvent::TextDelta {
                            content: format!("\n\n{}", r.response),
                        });
                        msgs.push(ChatMsg {
                            role: "assistant".to_string(),
                            content: full_text.clone(),
                        });
                        emit!(AgentEvent::Done);
                        return msgs;
                    }

                    // Build labeled feedback
                    let mut feedback = format!("[Executed]\n{}", code);
                    if let Some(err) = &r.error {
                        feedback.push_str(&format!(
                            "\n\n[Error]\n{}\nFix and retry with a ```python block.",
                            err
                        ));
                    } else if !r.output.is_empty() {
                        feedback.push_str(&format!("\n\n[Output]\n{}", r.output));
                        if !r.tool_calls.is_empty() {
                            feedback.push_str(&format!(
                                "\n[{} tool call(s) executed]",
                                r.tool_calls.len()
                            ));
                        }
                    } else if !r.tool_calls.is_empty() {
                        feedback.push_str(&format!(
                            "\n\n[{} tool call(s) executed]",
                            r.tool_calls.len()
                        ));
                    } else {
                        // No output, no error, no tool calls
                        msgs.push(ChatMsg {
                            role: "assistant".to_string(),
                            content: full_text.clone(),
                        });
                        emit!(AgentEvent::Done);
                        return msgs;
                    }

                    msgs.push(ChatMsg {
                        role: "assistant".to_string(),
                        content: full_text.clone(),
                    });
                    msgs.push(ChatMsg {
                        role: "system".to_string(),
                        content: feedback,
                    });
                }
                Err(e) => {
                    let err_text = format!("Execution error: {}", e);
                    emit!(AgentEvent::CodeOutput {
                        output: String::new(),
                        error: Some(err_text.clone()),
                    });

                    msgs.push(ChatMsg {
                        role: "assistant".to_string(),
                        content: full_text.clone(),
                    });
                    msgs.push(ChatMsg {
                        role: "system".to_string(),
                        content: format!(
                            "[Executed]\n{}\n\n[Error]\n{}\nFix and retry with a ```python block.",
                            code, err_text
                        ),
                    });
                }
            }

            if iteration == self.config.max_iterations - 1 {
                emit!(AgentEvent::Error {
                    message: "Max iterations reached".to_string(),
                });
            }
        }

        emit!(AgentEvent::Done);
        msgs
    }

    /// Get the inner session (for pooling, snapshot, etc.)
    pub fn into_session(self) -> Session {
        self.session
    }
}

/// Build environment context string for the system prompt.
fn build_context() -> String {
    match std::env::current_dir() {
        Ok(cwd) => format!("Working directory: {}", cwd.display()),
        Err(_) => String::new(),
    }
}

/// Check if an LLM error is retryable (rate limits, server errors).
fn is_retryable(error: &str) -> bool {
    let lower = error.to_lowercase();
    lower.contains("429")
        || lower.contains("rate")
        || lower.contains("503")
        || lower.contains("502")
        || lower.contains("overloaded")
        || lower.contains("temporarily")
}

/// Extract ```python fenced code blocks from LLM text.
fn extract_code(text: &str) -> String {
    let mut blocks = Vec::new();
    let mut in_block = false;
    let mut current = String::new();

    for line in text.lines() {
        if line.trim_start().starts_with("```python") {
            in_block = true;
            current.clear();
        } else if line.trim_start().starts_with("```") && in_block {
            in_block = false;
            if !current.trim().is_empty() {
                blocks.push(current.clone());
            }
        } else if in_block {
            current.push_str(line);
            current.push('\n');
        }
    }

    if !blocks.is_empty() {
        return blocks.join("\n");
    }

    String::new()
}

/// Truncate messages to fit within the context budget.
fn truncate_messages(messages: &mut Vec<ChatMsg>, budget: usize) {
    let total: usize = messages.iter().map(|m| m.content.len()).sum();
    if total <= budget {
        return;
    }
    if messages.len() <= 5 {
        return;
    }
    let keep_end = 4;
    let keep_start = 1;
    while messages.len() > keep_start + keep_end {
        let total: usize = messages.iter().map(|m| m.content.len()).sum();
        if total <= budget {
            break;
        }
        messages.remove(keep_start);
    }
}
