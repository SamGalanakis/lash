mod exec;
pub mod message;

use std::path::PathBuf;
use std::sync::Arc;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::ToolDefinition;
use crate::baml_client::ClientRegistry;
use crate::baml_client::async_client::B;
use crate::baml_client::new_image_from_base64;
pub use crate::baml_client::types::{ChatMsg, Image};
use crate::provider::Provider;
use crate::session::Session;
use crate::store::Store;

pub use message::{
    Message, MessageRole, Part, PartKind, PruneState, messages_from_chat, messages_to_chat,
};

use exec::{ExecAccumulator, execute_and_collect};

/// Send an event to the channel if it's still open.
pub(crate) async fn send_event(tx: &mpsc::Sender<AgentEvent>, event: AgentEvent) {
    if !tx.is_closed() {
        let _ = tx.send(event).await;
    }
}

// ─── Token tracking ───

/// Token usage statistics from an LLM call.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct TokenUsage {
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cached_input_tokens: i64,
}

impl TokenUsage {
    pub fn total(&self) -> i64 {
        self.input_tokens + self.output_tokens
    }
    pub fn add(&mut self, other: &TokenUsage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.cached_input_tokens += other.cached_input_tokens;
    }
}

/// Configuration for the agent loop.
#[derive(Clone)]
pub struct AgentConfig {
    /// Model identifier (e.g. "anthropic/claude-sonnet-4-5")
    pub model: String,
    /// LLM provider (Claude OAuth or OpenRouter)
    pub provider: Provider,
    /// Maximum total character budget for context truncation
    pub max_context_chars: usize,
    /// When true, use SubAgentStep prompt instead of CodeActStep
    pub sub_agent: bool,
    /// Optional reasoning effort level (e.g. "medium", "high") for Codex
    pub reasoning_effort: Option<String>,
    /// Optional turn limit. None = unlimited (default for root agent).
    pub max_turns: Option<usize>,
    /// When true, include Soul principles in the system prompt.
    pub include_soul: bool,
    /// When set, append raw LLM request/response JSON to this file (debug logging).
    pub llm_log_path: Option<PathBuf>,
    /// When true, use headless prompt (no ask(), no TUI references).
    pub headless: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            model: "anthropic/claude-sonnet-4-5".to_string(),
            provider: Provider::OpenRouter {
                api_key: String::new(),
                base_url: "https://openrouter.ai/api/v1".to_string(),
            },
            max_context_chars: 400_000,
            sub_agent: false,
            reasoning_effort: None,
            max_turns: None,
            include_soul: false,
            llm_log_path: None,
            headless: false,
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
    #[serde(rename = "token_usage")]
    TokenUsage {
        iteration: usize,
        usage: TokenUsage,
        cumulative: TokenUsage,
    },
    #[serde(rename = "sub_agent_done")]
    SubAgentDone {
        task: String,
        usage: TokenUsage,
        tool_calls: usize,
        iterations: usize,
        success: bool,
    },
    #[serde(rename = "done")]
    Done,
    #[serde(rename = "error")]
    Error { message: String },
    #[serde(rename = "prompt")]
    Prompt {
        question: String,
        options: Vec<String>,
        #[serde(skip)]
        response_tx: std::sync::mpsc::Sender<String>,
    },
}

/// Timeout for waiting on LLM streaming chunks.
const LLM_STREAM_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);
/// Abort LLM response if it exceeds this length (likely degenerate output).
const LLM_MAX_RESPONSE_CHARS: usize = 50_000;
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
    agent_id: String,
    session: Session,
    config: AgentConfig,
    store: Arc<Store>,
}

impl Agent {
    pub fn new(
        session: Session,
        config: AgentConfig,
        store: Arc<Store>,
        agent_id: Option<String>,
    ) -> Self {
        let agent_id = agent_id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        Self {
            agent_id,
            session,
            config,
            store,
        }
    }

    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }

    /// Snapshot the Python session (dill + scratch files).
    pub async fn snapshot(&mut self) -> Option<Vec<u8>> {
        match self.session.snapshot().await {
            Ok(data) => Some(data),
            Err(e) => {
                tracing::warn!("Session snapshot failed: {}", e);
                None
            }
        }
    }

    /// Restore the Python session from a snapshot blob.
    pub async fn restore(&mut self, data: &[u8]) -> Result<(), crate::SessionError> {
        self.session.restore(data).await
    }

    pub fn set_model(&mut self, model: String) {
        self.config.model = model;
    }

    /// Reset the underlying Python session (clear namespace).
    pub async fn reset_session(&mut self) -> Result<(), crate::SessionError> {
        self.session.reset().await
    }

    /// Run the agent loop with structured messages, emitting events.
    /// Returns the updated message history and the final iteration counter.
    pub async fn run(
        &mut self,
        messages: Vec<Message>,
        images: Vec<Vec<u8>>,
        event_tx: mpsc::Sender<AgentEvent>,
        cancel: CancellationToken,
        run_offset: usize,
    ) -> (Vec<Message>, usize) {
        macro_rules! emit {
            ($event:expr) => {
                send_event(&event_tx, $event).await
            };
        }

        // Refresh tokens if needed before starting
        match self.config.provider.ensure_fresh().await {
            Ok(true) => {
                let _ = crate::provider::save_provider(&self.config.provider);
            }
            Err(e) => {
                emit!(AgentEvent::Error {
                    message: format!("Token refresh failed: {}", e),
                });
                emit!(AgentEvent::Done);
                return (messages, run_offset);
            }
            _ => {}
        }

        // Build ClientRegistry
        let model = self.config.provider.resolve_model(&self.config.model);
        let mut cr = ClientRegistry::new();
        cr.add_llm_client(
            "DefaultClient",
            self.config.provider.baml_provider(),
            self.config
                .provider
                .baml_options(&model, self.config.reasoning_effort.as_deref()),
        );
        cr.set_primary_client("DefaultClient");

        // Generate tool docs, context, and dynamic prompt sections
        let all_tools = self.session.tools().definitions();
        let visible: Vec<_> = all_tools.iter().filter(|t| !t.hidden).cloned().collect();
        let tool_list = ToolDefinition::format_tool_docs(&visible);
        let context = build_context();
        let tool_names: Vec<String> = visible.iter().map(|t| t.name.clone()).collect();
        let loader = crate::instructions::InstructionLoader::new();
        let project_instructions = loader.system_instructions().to_string();

        // Convert raw PNG bytes to BAML images
        use base64::Engine;
        let user_images: Vec<Image> = images
            .iter()
            .map(|png_bytes| {
                let b64 = base64::engine::general_purpose::STANDARD.encode(png_bytes);
                new_image_from_base64(&b64, Some("image/png"))
            })
            .collect();

        // Create collector for token tracking
        let collector = crate::baml_client::new_collector(&self.agent_id);
        let mut cumulative_usage = TokenUsage::default();

        let mut msgs = messages;
        let mut iteration: usize = run_offset;
        let mut tool_images: Vec<Image> = Vec::new();
        let mut max_steps_final = false;

        loop {
            if cancel.is_cancelled() {
                self.snapshot_to_store(None, &msgs, iteration, &cumulative_usage)
                    .await;
                emit!(AgentEvent::Done);
                return (msgs, iteration);
            }

            if max_steps_final {
                self.store.mark_agent_done(&self.agent_id);
                emit!(AgentEvent::Done);
                return (msgs, iteration);
            }

            // Rolling window: collapse old turns out of the prompt
            const HISTORY_WINDOW: usize = 6;
            // Remove any previous history note
            msgs.retain(|m| m.id != "history_note");
            // Keep first msg (user request) + last HISTORY_WINDOW pairs
            if msgs.len() > 1 + HISTORY_WINDOW * 2 {
                let keep_start = 1;
                let remove_end = msgs.len() - HISTORY_WINDOW * 2;
                let collapsed_count = (remove_end - keep_start) / 2;
                msgs.drain(keep_start..remove_end);
                let note = format!(
                    "[{} earlier turns collapsed into `_history` (accessible in your REPL). \
                     Use `_history.search(\"pattern\")` to find past results, \
                     `_history[i]` for a specific turn, \
                     `_history.summary()` for an overview.]",
                    collapsed_count
                );
                msgs.insert(
                    keep_start,
                    Message {
                        id: "history_note".to_string(),
                        role: MessageRole::System,
                        parts: vec![Part {
                            id: "history_note.p0".to_string(),
                            kind: PartKind::Text,
                            content: note,
                            prune_state: PruneState::Intact,
                        }],
                    },
                );
            }

            let chat_msgs = messages_to_chat(&msgs);

            emit!(AgentEvent::LlmRequest {
                iteration,
                message_count: msgs.len(),
                tool_list: tool_list.clone(),
            });

            // Set up message forwarding for incremental execution
            let (msg_tx, mut msg_rx) =
                tokio::sync::mpsc::unbounded_channel::<crate::SandboxMessage>();
            self.session.set_message_sender(msg_tx);
            let event_tx_clone = event_tx.clone();
            let drain_handle = tokio::spawn(async move {
                while let Some(sandbox_msg) = msg_rx.recv().await {
                    if sandbox_msg.kind == "final" {
                        continue;
                    }
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

            let (prompt_tx, mut prompt_rx) =
                tokio::sync::mpsc::unbounded_channel::<crate::session::UserPrompt>();
            self.session.set_prompt_sender(prompt_tx);
            let event_tx_prompt = event_tx.clone();
            let prompt_drain_handle = tokio::spawn(async move {
                while let Some(user_prompt) = prompt_rx.recv().await {
                    if !event_tx_prompt.is_closed() {
                        let _ = event_tx_prompt
                            .send(AgentEvent::Prompt {
                                question: user_prompt.question,
                                options: user_prompt.options,
                                response_tx: user_prompt.response_tx,
                            })
                            .await;
                    }
                }
            });

            // Execution state for incremental execution
            let mut acc = ExecAccumulator::new();
            let mut stop_marker_code: Option<String> = None;

            // LLM call with retry on transient API errors
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

                    let all_images: Vec<Image> = user_images
                        .iter()
                        .chain(tool_images.iter())
                        .cloned()
                        .collect();
                    tool_images.clear();

                    let include_soul = if self.config.sub_agent {
                        self.config.include_soul
                    } else {
                        true
                    };
                    let mut call = match if self.config.sub_agent {
                        B.SubAgentStep
                            .with_client_registry(&cr)
                            .with_collector(&collector)
                            .stream(
                                &chat_msgs,
                                &tool_list,
                                &context,
                                &tool_names,
                                &project_instructions,
                                &all_images,
                                include_soul,
                                self.config.headless,
                            )
                    } else {
                        B.CodeActStep
                            .with_client_registry(&cr)
                            .with_collector(&collector)
                            .stream(
                                &chat_msgs,
                                &tool_list,
                                &context,
                                &tool_names,
                                &project_instructions,
                                &all_images,
                                include_soul,
                                self.config.headless,
                            )
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

                    // Stream LLM tokens with incremental execution
                    let mut stream_error = None;
                    let mut prev_len = 0usize;
                    let mut line_buffer = String::new();
                    loop {
                        if cancel.is_cancelled() {
                            self.session.clear_message_sender();
                            self.session.clear_prompt_sender();
                            self.snapshot_to_store(None, &msgs, iteration, &cumulative_usage)
                                .await;
                            emit!(AgentEvent::Done);
                            return (msgs, iteration);
                        }
                        match tokio::time::timeout(LLM_STREAM_TIMEOUT, call.next()).await {
                            Err(_timeout) => {
                                self.session.clear_message_sender();
                                self.session.clear_prompt_sender();
                                emit!(AgentEvent::Error {
                                    message: "LLM response timed out".to_string(),
                                });
                                emit!(AgentEvent::Done);
                                return (msgs, iteration);
                            }
                            Ok(None) => {
                                break;
                            }
                            Ok(Some(Ok(partial))) => {
                                if partial.len() > prev_len {
                                    let delta = &partial[prev_len..];
                                    emit!(AgentEvent::TextDelta {
                                        content: delta.to_string(),
                                    });

                                    line_buffer.push_str(delta);

                                    // Stop marker: execute accumulated code, break to new iteration
                                    if let Some(turn_pos) = find_stop_marker(&line_buffer) {
                                        let before = line_buffer[..turn_pos].to_string();
                                        line_buffer.clear();
                                        stop_marker_code = Some(before.clone());
                                        if !before.trim().is_empty() && !acc.had_failure {
                                            execute_and_collect(
                                                &mut self.session,
                                                &before,
                                                &mut acc,
                                                &event_tx,
                                            )
                                            .await;
                                        }
                                        break;
                                    }

                                    // Incremental execution on complete statements
                                    if delta.contains('\n') && !acc.had_failure {
                                        loop {
                                            let boundaries =
                                                find_candidate_boundaries(&line_buffer);
                                            let mut executed_any = false;
                                            for &pos in boundaries.iter().rev() {
                                                let prefix = &line_buffer[..pos];
                                                if prefix.trim().is_empty() {
                                                    continue;
                                                }
                                                if self
                                                    .session
                                                    .check_complete(prefix)
                                                    .unwrap_or(false)
                                                {
                                                    let ready = line_buffer[..pos].to_string();
                                                    line_buffer = line_buffer[pos..].to_string();
                                                    execute_and_collect(
                                                        &mut self.session,
                                                        &ready,
                                                        &mut acc,
                                                        &event_tx,
                                                    )
                                                    .await;
                                                    executed_any = true;
                                                    break; // re-scan remaining buffer
                                                }
                                            }
                                            if !executed_any {
                                                break;
                                            }
                                        }
                                    }

                                    prev_len = partial.len();
                                }
                                if prev_len > LLM_MAX_RESPONSE_CHARS {
                                    tracing::warn!(
                                        "LLM response exceeded {} chars, aborting",
                                        LLM_MAX_RESPONSE_CHARS
                                    );
                                    self.session.clear_message_sender();
                                    self.session.clear_prompt_sender();
                                    emit!(AgentEvent::Error {
                                        message: format!(
                                            "Response exceeded {} chars — likely degenerate output, aborting",
                                            LLM_MAX_RESPONSE_CHARS
                                        ),
                                    });
                                    emit!(AgentEvent::Done);
                                    return (msgs, iteration);
                                }
                            }
                            Ok(Some(Err(e))) => {
                                stream_error = Some(format!("{}", e));
                                break;
                            }
                        }
                    }

                    // Stop marker hit — drop stream, log usage
                    if stop_marker_code.is_some() {
                        drop(call);
                        let code = stop_marker_code.as_deref().unwrap_or("");
                        emit!(AgentEvent::LlmResponse {
                            iteration,
                            content: code.to_string(),
                            duration_ms: llm_start.elapsed().as_millis() as u64,
                        });

                        collect_usage(
                            &collector,
                            &mut cumulative_usage,
                            iteration,
                            code,
                            &self.config,
                            &self.agent_id,
                            &event_tx,
                        )
                        .await;

                        break 'llm_retry code.to_string();
                    }

                    if let Some(err) = stream_error {
                        if !acc.tool_calls.is_empty()
                            || !acc.combined_output.is_empty()
                            || acc.had_failure
                        {
                            emit!(AgentEvent::Error {
                                message: format!(
                                    "LLM stream error (after partial execution): {}",
                                    err
                                ),
                            });
                            break 'llm_retry String::new();
                        }
                        if is_retryable(&err) && attempt < LLM_MAX_RETRIES {
                            last_error = Some(err);
                            continue;
                        }
                        self.session.clear_message_sender();
                        self.session.clear_prompt_sender();
                        emit!(AgentEvent::Error {
                            message: format!("LLM error: {}", err),
                        });
                        emit!(AgentEvent::Done);
                        return (msgs, iteration);
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

                    collect_usage(
                        &collector,
                        &mut cumulative_usage,
                        iteration,
                        &text,
                        &self.config,
                        &self.agent_id,
                        &event_tx,
                    )
                    .await;

                    // Execute remaining buffer
                    if !line_buffer.trim().is_empty() && !acc.had_failure {
                        execute_and_collect(&mut self.session, &line_buffer, &mut acc, &event_tx)
                            .await;
                    }

                    break 'llm_retry text;
                }

                // All retries exhausted
                if let Some(err) = last_error {
                    emit!(AgentEvent::Error {
                        message: format!(
                            "LLM failed after {} retries: {}",
                            LLM_MAX_RETRIES + 1,
                            err
                        ),
                    });
                }
                String::new()
            };

            // Clean up message and prompt forwarding
            self.session.clear_message_sender();
            self.session.clear_prompt_sender();
            let _ = drain_handle.await;
            let _ = prompt_drain_handle.await;

            let executed_text = stop_marker_code.unwrap_or(full_text);

            // Check for empty response with no execution
            if executed_text.trim().is_empty()
                && acc.tool_calls.is_empty()
                && acc.combined_output.is_empty()
                && !acc.had_failure
            {
                tracing::warn!("LLM returned empty response");
                emit!(AgentEvent::Error {
                    message: "I didn't get a response — please try again.".to_string()
                });
                emit!(AgentEvent::Done);
                return (msgs, iteration);
            }

            // Convert tool images to BAML Image for the next LLM turn
            for img in &acc.images {
                let b64 = base64::engine::general_purpose::STANDARD.encode(&img.data);
                tool_images.push(new_image_from_base64(&b64, Some(&img.mime)));
            }

            // respond() = stop signal
            if !acc.final_response.is_empty() {
                emit!(AgentEvent::Message {
                    text: acc.final_response.clone(),
                    kind: "final".to_string(),
                });
                let mid = format!("m{}", msgs.len());
                msgs.push(Message {
                    id: mid.clone(),
                    role: MessageRole::Assistant,
                    parts: vec![Part {
                        id: format!("{}.p0", mid),
                        kind: PartKind::Code,
                        content: executed_text.clone(),
                        prune_state: PruneState::Intact,
                    }],
                });
                self.store.mark_agent_done(&self.agent_id);
                emit!(AgentEvent::Done);
                return (msgs, iteration);
            }

            // Inject turn data into _history
            {
                let tool_calls_json: Vec<serde_json::Value> = acc
                    .tool_calls
                    .iter()
                    .map(|tc| {
                        serde_json::json!({
                            "tool": tc.tool,
                            "args": tc.args,
                            "result": tc.result,
                            "success": tc.success,
                            "duration_ms": tc.duration_ms,
                        })
                    })
                    .collect();
                let turn_json = serde_json::json!({
                    "index": iteration,
                    "code": executed_text,
                    "output": acc.combined_output,
                    "error": acc.exec_error,
                    "tool_calls": tool_calls_json,
                });
                let json_str = turn_json
                    .to_string()
                    .replace('\\', "\\\\")
                    .replace('\'', "\\'");
                let inject_code = format!("_history._add_turn('{}')", json_str);
                let _ = self.session.run_code(&inject_code).await;
            }

            // Build structured feedback parts from accumulated execution state
            {
                let mut feedback_parts = vec![Part {
                    id: String::new(),
                    kind: PartKind::Code,
                    content: executed_text.clone(),
                    prune_state: PruneState::Intact,
                }];

                let has_output = !acc.combined_output.is_empty();
                let has_tool_calls = !acc.tool_calls.is_empty();

                if has_output {
                    let mut output_text = acc.combined_output;
                    if has_tool_calls {
                        output_text.push_str(&format!(
                            "\n[{} tool call(s) executed]",
                            acc.tool_calls.len()
                        ));
                    }
                    feedback_parts.push(Part {
                        id: String::new(),
                        kind: PartKind::Output,
                        content: output_text,
                        prune_state: PruneState::Intact,
                    });
                } else if has_tool_calls {
                    feedback_parts.push(Part {
                        id: String::new(),
                        kind: PartKind::Output,
                        content: format!("[{} tool call(s) executed]", acc.tool_calls.len()),
                        prune_state: PruneState::Intact,
                    });
                }
                if let Some(err) = &acc.exec_error {
                    feedback_parts.push(Part {
                        id: String::new(),
                        kind: PartKind::Error,
                        content: format!("{}\nFix and retry.", err),
                        prune_state: PruneState::Intact,
                    });
                } else if !has_output && !has_tool_calls {
                    let mid = format!("m{}", msgs.len());
                    msgs.push(Message {
                        id: mid.clone(),
                        role: MessageRole::Assistant,
                        parts: vec![Part {
                            id: format!("{}.p0", mid),
                            kind: PartKind::Code,
                            content: executed_text.clone(),
                            prune_state: PruneState::Intact,
                        }],
                    });
                    self.store.mark_agent_done(&self.agent_id);
                    emit!(AgentEvent::Done);
                    return (msgs, iteration);
                }

                // Push assistant message
                let asst_id = format!("m{}", msgs.len());
                msgs.push(Message {
                    id: asst_id.clone(),
                    role: MessageRole::Assistant,
                    parts: vec![Part {
                        id: format!("{}.p0", asst_id),
                        kind: PartKind::Code,
                        content: executed_text.clone(),
                        prune_state: PruneState::Intact,
                    }],
                });

                // Push system feedback with typed parts
                let sys_id = format!("m{}", msgs.len());
                for (idx, part) in feedback_parts.iter_mut().enumerate() {
                    part.id = format!("{}.p{}", sys_id, idx);
                }
                msgs.push(Message {
                    id: sys_id,
                    role: MessageRole::System,
                    parts: feedback_parts,
                });
            }

            iteration += 1;
            if let Some(max) = self.config.max_turns
                && iteration >= run_offset + max
            {
                let sys_id = format!("m{}", msgs.len());
                msgs.push(Message {
                    id: sys_id.clone(),
                    role: MessageRole::System,
                    parts: vec![Part {
                        id: format!("{}.p0", sys_id),
                        kind: PartKind::Text,
                        content: format!(
                            "Turn limit reached ({max}). You MUST call respond() now with:\n\
                                1. Summary of what you accomplished\n\
                                2. List of remaining tasks not yet completed\n\
                                3. Recommended next steps\n\
                                Do NOT make any more tool calls. Call respond() immediately."
                        ),
                        prune_state: PruneState::Intact,
                    }],
                });
                max_steps_final = true;
            }
        }
    }

    /// Snapshot agent state to the Store on cancellation.
    async fn snapshot_to_store(
        &mut self,
        parent_id: Option<&str>,
        msgs: &[Message],
        iteration: usize,
        cumulative_usage: &TokenUsage,
    ) {
        let dill_blob = self.snapshot().await;
        let msgs_json = serde_json::to_string(msgs).unwrap_or_else(|_| "[]".to_string());
        self.store.save_agent_state(
            &self.agent_id,
            parent_id,
            "active",
            &msgs_json,
            iteration as i64,
            "{}",
            dill_blob.as_deref(),
            cumulative_usage.input_tokens,
            cumulative_usage.output_tokens,
            cumulative_usage.cached_input_tokens,
        );
    }

    /// Get the inner session (for pooling, snapshot, etc.)
    pub fn into_session(self) -> Session {
        self.session
    }
}

/// Log raw LLM request/response to a debug file (if configured).
fn log_llm_debug(
    log_path: &Option<PathBuf>,
    agent_id: &str,
    iteration: usize,
    usage: &TokenUsage,
    request_body: Option<String>,
    response_text: &str,
) {
    let Some(path) = log_path else { return };

    let entry = serde_json::json!({
        "turn": iteration,
        "ts": chrono::Utc::now().to_rfc3339(),
        "agent_id": agent_id,
        "request": request_body,
        "response": response_text,
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cached_input_tokens": usage.cached_input_tokens,
        }
    });

    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        use std::io::Write;
        let _ = writeln!(f, "{}", entry);
    }
}

/// Read token usage from the collector, emit a TokenUsage event, and log debug info.
/// Called after both stop-marker and normal LLM completion paths.
async fn collect_usage(
    collector: &baml::Collector,
    cumulative_usage: &mut TokenUsage,
    iteration: usize,
    response_text: &str,
    config: &AgentConfig,
    agent_id: &str,
    event_tx: &mpsc::Sender<AgentEvent>,
) {
    if let Some(log) = collector.last() {
        let u = log.usage();
        let usage = TokenUsage {
            input_tokens: u.input_tokens(),
            output_tokens: u.output_tokens(),
            cached_input_tokens: u.cached_input_tokens().unwrap_or(0),
        };
        cumulative_usage.add(&usage);
        send_event(
            event_tx,
            AgentEvent::TokenUsage {
                iteration,
                usage: usage.clone(),
                cumulative: cumulative_usage.clone(),
            },
        )
        .await;

        let request_body = log
            .selected_call()
            .and_then(|c| c.http_request())
            .and_then(|r| r.body().text().ok());
        log_llm_debug(
            &config.llm_log_path,
            agent_id,
            iteration,
            &usage,
            request_body,
            response_text,
        );
    }
}

/// Build environment context string for the system prompt.
fn build_context() -> String {
    let mut parts = Vec::new();

    if let Ok(cwd) = std::env::current_dir() {
        parts.push(format!("Working directory: {}", cwd.display()));

        let git_dir = cwd.join(".git");
        if git_dir.exists() {
            parts.push("Git repository: yes".into());
        }

        if let Ok(entries) = std::fs::read_dir(&cwd) {
            let mut names: Vec<String> = entries
                .filter_map(|e| e.ok())
                .map(|e| {
                    let name = e.file_name().to_string_lossy().to_string();
                    if e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                        format!("{}/", name)
                    } else {
                        name
                    }
                })
                .filter(|n| !n.starts_with('.'))
                .collect();
            names.sort();
            if !names.is_empty() {
                parts.push(format!("Top-level entries: {}", names.join(", ")));
            }
        }
    }

    parts.join("\n")
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

/// Find a stop marker (`observe()` or `<turn>`) in the buffer.
fn find_stop_marker(buf: &str) -> Option<usize> {
    let mut offset = 0;
    for line in buf.split('\n') {
        let trimmed = line.trim();
        if trimmed == "observe()" || trimmed == "<turn>" {
            return Some(offset);
        }
        offset += line.len() + 1;
    }
    None
}

/// Find byte offsets of candidate statement boundaries in code.
fn find_candidate_boundaries(code: &str) -> Vec<usize> {
    let mut boundaries = Vec::new();
    for (i, _) in code.match_indices('\n') {
        if let Some(c) = code[i + 1..].chars().next()
            && !c.is_whitespace()
            && c != '#'
        {
            boundaries.push(i + 1);
        }
    }
    boundaries
}
