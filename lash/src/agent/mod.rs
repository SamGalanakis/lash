mod exec;
pub mod message;

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::ToolDefinition;
use crate::baml_client::ClientRegistry;
use crate::baml_client::async_client::B;
use crate::baml_client::new_image_from_base64;
pub use crate::baml_client::types::{ChatMsg, Image};
use crate::instructions::{FsInstructionSource, InstructionSource};
use crate::provider::Provider;
use crate::session::Session;

pub use message::{Message, MessageRole, Part, PartKind, PruneState, messages_to_chat};

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
    /// Model identifier (e.g. "anthropic/claude-sonnet-4.6")
    pub model: String,
    /// LLM provider (OpenRouter, Claude OAuth, Codex, or Google OAuth)
    pub provider: Provider,
    /// Override for context window size (tokens). If None, looked up from model_info.
    pub max_context_tokens: Option<usize>,
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
    /// Custom preamble (identity/role). If None, uses the default lash preamble.
    pub preamble: Option<String>,
    /// Custom soul (personality principles). If None, uses the default Soul.
    /// Set to Some("") to disable soul entirely.
    pub soul: Option<String>,
    /// Host-provided instruction source (filesystem by default).
    pub instruction_source: Arc<dyn InstructionSource>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            model: "anthropic/claude-sonnet-4.6".to_string(),
            provider: Provider::OpenRouter {
                api_key: String::new(),
                base_url: "https://openrouter.ai/api/v1".to_string(),
            },
            max_context_tokens: None,
            sub_agent: false,
            reasoning_effort: None,
            max_turns: None,
            include_soul: false,
            llm_log_path: None,
            headless: false,
            preamble: None,
            soul: None,
            instruction_source: Arc::new(FsInstructionSource::new()),
        }
    }
}

/// Standardized error payload surfaced to hosts/UI.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ErrorEnvelope {
    /// Broad category (e.g. llm_provider, token_refresh, runtime, input_validation).
    pub kind: String,
    /// Optional machine-friendly error code.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    /// Human-friendly message safe to show in UI.
    pub user_message: String,
    /// Optional raw/source error text (possibly sanitized/truncated).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<String>,
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
    Error {
        message: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        envelope: Option<ErrorEnvelope>,
    },
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
/// Max retries for rate-limited or empty LLM responses.
const LLM_MAX_RETRIES: usize = 3;
/// Delays between retries (exponential backoff).
const LLM_RETRY_DELAYS: [std::time::Duration; 3] = [
    std::time::Duration::from_secs(2),
    std::time::Duration::from_secs(5),
    std::time::Duration::from_secs(10),
];

fn make_error_event(
    kind: &str,
    code: Option<&str>,
    user_message: impl Into<String>,
    raw: Option<String>,
) -> AgentEvent {
    let user_message = user_message.into();
    AgentEvent::Error {
        message: user_message.clone(),
        envelope: Some(ErrorEnvelope {
            kind: kind.to_string(),
            code: code.map(str::to_string),
            user_message,
            raw: raw.map(|s| truncate_raw_error(s.trim())),
        }),
    }
}

fn truncate_raw_error(s: &str) -> String {
    const MAX_RAW: usize = 4000;
    if s.len() <= MAX_RAW {
        return s.to_string();
    }
    let keep = MAX_RAW / 2;
    let omitted = s.len() - MAX_RAW;
    format!(
        "{}\n\n... ({omitted} chars omitted) ...\n\n{}",
        &s[..keep],
        &s[s.len() - keep..]
    )
}

/// CodeAct agent: LLM writes Python code, REPL executes, output feeds back.
pub struct Agent {
    agent_id: String,
    session: Session,
    config: AgentConfig,
}

impl Agent {
    pub fn new(session: Session, config: AgentConfig, agent_id: Option<String>) -> Self {
        let agent_id = agent_id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        Self {
            agent_id,
            session,
            config,
        }
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

    pub fn set_provider(&mut self, provider: Provider) {
        self.config.provider = provider;
    }

    pub fn set_reasoning_effort(&mut self, reasoning_effort: Option<String>) {
        self.config.reasoning_effort = reasoning_effort;
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
                emit!(make_error_event(
                    "token_refresh",
                    Some("refresh_failed"),
                    format!("Token refresh failed: {}", e),
                    Some(e.to_string()),
                ));
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
        let prompt_tools: Vec<_> = visible
            .iter()
            .filter(|t| t.inject_into_prompt)
            .cloned()
            .collect();
        let mut tool_list = ToolDefinition::format_tool_docs(&prompt_tools);
        let omitted_tool_count = visible.iter().filter(|t| !t.inject_into_prompt).count();
        if omitted_tool_count > 0 {
            let note = format!(
                "\n\n- **Note:** {omitted_tool_count} additional tool(s) are available but omitted from this prompt for brevity. Use `list_tools()` / `find_tools(...)` to discover them, then call via `T.<tool>(...)`."
            );
            tool_list.push_str(&note);
        }
        let context = build_context();
        let tool_names: Vec<String> = prompt_tools.iter().map(|t| t.name.clone()).collect();
        let instruction_source = Arc::clone(&self.config.instruction_source);
        let project_instructions = instruction_source.system_instructions();

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
        let mut last_input_tokens: usize = 0;

        let max_context = self
            .config
            .max_context_tokens
            .or_else(|| {
                self.config
                    .provider
                    .context_window(&self.config.model)
                    .map(|v| v as usize)
            })
            .unwrap_or_else(|| {
                eprintln!(
                    "Warning: unknown context window for model '{}', defaulting to 200k",
                    self.config.model
                );
                200_000
            });

        let mut msgs = messages;
        let mut iteration: usize = run_offset;
        let mut tool_images: Vec<Image> = Vec::new();
        let mut max_steps_final = false;
        let mut has_history = false;
        let session_start = std::time::Instant::now();

        loop {
            if cancel.is_cancelled() {
                emit!(AgentEvent::Done);
                return (msgs, iteration);
            }

            // Rolling window: when context exceeds 60% of the model's token limit,
            // collapse old turns to make room. Always preserve all User messages.
            // Uses actual input_tokens from the last API call for accurate measurement,
            // with char-based estimation (~4 chars/token) as proportional weights.
            msgs.retain(|m| m.id != "history_note");
            {
                let token_budget = max_context * 60 / 100;
                let needs_pruning = last_input_tokens > token_budget;

                // `keep_from` is the start index of the retained tail.
                // Default to 1 (= keep everything after the initial system message).
                // If we are not pruning this turn, [1..keep_from) is empty and no collapse happens.
                let mut keep_from = 1usize;
                if needs_pruning {
                    keep_from = msgs.len();
                    // Estimate how many chars to keep: scale by (budget / actual) ratio
                    let total_chars: usize = msgs.iter().map(|m| m.char_count()).sum();
                    let target_chars = total_chars * token_budget / last_input_tokens.max(1);

                    // Walk backwards, accumulating char counts as proportional weights
                    let mut tail_chars = 0usize;
                    for i in (1..msgs.len()).rev() {
                        let cost = msgs[i].char_count();
                        if tail_chars + cost > target_chars {
                            break;
                        }
                        tail_chars += cost;
                        keep_from = i;
                    }
                }
                // Collect User messages from the pruned region [1..keep_from)
                // so they're preserved in the context.
                let mut preserved_user_msgs: Vec<Message> = Vec::new();
                if keep_from > 1 {
                    for msg in &msgs[1..keep_from] {
                        if msg.role == MessageRole::User {
                            preserved_user_msgs.push(msg.clone());
                        }
                    }
                }
                // Count collapsed turns (assistant+system pairs, minus preserved users)
                let collapsed_count = if keep_from > 1 {
                    (keep_from - 1 - preserved_user_msgs.len()) / 2
                } else {
                    0
                };
                if collapsed_count > 0 {
                    has_history = true;
                    // Drain the pruned region
                    msgs.drain(1..keep_from);
                    // Re-insert preserved user messages, then the history note
                    let elapsed = session_start.elapsed();
                    let elapsed_str = if elapsed.as_secs() >= 3600 {
                        format!(
                            "{}h {}m",
                            elapsed.as_secs() / 3600,
                            (elapsed.as_secs() % 3600) / 60
                        )
                    } else if elapsed.as_secs() >= 60 {
                        format!("{}m {}s", elapsed.as_secs() / 60, elapsed.as_secs() % 60)
                    } else {
                        format!("{}s", elapsed.as_secs())
                    };
                    let note = format!(
                        "[Turn {}, {} elapsed. {} earlier turn(s) dropped from context (not summarized). \
                         Use `_history` to access them at full fidelity:\n \
                         `_history.user_messages()` -- what the user asked\n \
                         `_history.find(\"query\", mode=\"hybrid\")` -- find past results\n \
                         `_history[i]` -- specific turn]",
                        iteration, elapsed_str, collapsed_count
                    );
                    msgs.insert(
                        1,
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
                    // Insert preserved user messages right after the history note
                    for (idx, user_msg) in preserved_user_msgs.into_iter().enumerate() {
                        msgs.insert(2 + idx, user_msg);
                    }
                }
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
                    // Explicit allowlist for sandbox message kinds rendered in the TUI.
                    // Unknown kinds are dropped by default so random tool/runtime chatter
                    // never leaks into user-visible output.
                    match sandbox_msg.kind.as_str() {
                        "final" => continue,
                        "tool_output" => {}
                        other => {
                            tracing::debug!("dropping unsupported sandbox message kind: {other}");
                            continue;
                        }
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

            // Execution state for fence-aware parsing
            let mut acc = ExecAccumulator::new();
            let mut response = String::new();
            let mut prose_parts: Vec<String> = Vec::new();
            let mut code_parts: Vec<String> = Vec::new();
            // Fence parser state
            let mut in_code_fence = false;
            let mut current_prose = String::new();
            let mut current_code = String::new();
            let mut last_line_start = 0usize; // byte offset of current incomplete line
            let mut code_executed = false;

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
                        emit!(make_error_event(
                            "llm_provider",
                            Some("retrying"),
                            format!(
                                "Retrying in {}s (attempt {}/{})...",
                                delay.as_secs(),
                                attempt + 1,
                                LLM_MAX_RETRIES + 1,
                            ),
                            None,
                        ));
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
                    let preamble = self.config.preamble.clone().unwrap_or_default();
                    let soul = self.config.soul.clone().unwrap_or_default();
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
                                has_history,
                                &preamble,
                                &soul,
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
                                has_history,
                                &preamble,
                                &soul,
                            )
                    } {
                        Ok(c) => c,
                        Err(e) => {
                            let msg = format!("{}", e);
                            if is_retryable(&msg) && attempt < LLM_MAX_RETRIES {
                                last_error = Some(msg);
                                continue;
                            }
                            let mut env = sanitize_llm_error(&msg);
                            let http = last_http_call_summary(&collector)
                                .map(|s| format!("\n{}", s))
                                .unwrap_or_default();
                            let display = format!("LLM error: {}{}", env.user_message, http);
                            env.user_message = display.clone();
                            emit!(AgentEvent::Error {
                                message: display,
                                envelope: Some(env),
                            });
                            break 'llm_retry String::new();
                        }
                    };

                    // Stream LLM tokens with fence-aware parsing
                    let mut stream_error = None;
                    loop {
                        if cancel.is_cancelled() {
                            self.session.clear_message_sender();
                            self.session.clear_prompt_sender();
                            let _ = collect_usage(
                                &collector,
                                &mut cumulative_usage,
                                iteration,
                                &response,
                                &self.config,
                                &self.agent_id,
                                &event_tx,
                            )
                            .await;
                            emit!(AgentEvent::Done);
                            return (msgs, iteration);
                        }
                        match tokio::time::timeout(LLM_STREAM_TIMEOUT, call.next()).await {
                            Err(_timeout) => {
                                self.session.clear_message_sender();
                                self.session.clear_prompt_sender();
                                let _ = collect_usage(
                                    &collector,
                                    &mut cumulative_usage,
                                    iteration,
                                    &response,
                                    &self.config,
                                    &self.agent_id,
                                    &event_tx,
                                )
                                .await;
                                emit!(make_error_event(
                                    "llm_provider",
                                    Some("timeout"),
                                    "LLM response timed out",
                                    None,
                                ));
                                emit!(AgentEvent::Done);
                                return (msgs, iteration);
                            }
                            Ok(None) => {
                                break;
                            }
                            Ok(Some(Ok(partial))) => {
                                if partial.len() > response.len() {
                                    let delta = &partial[response.len()..];
                                    response.push_str(delta);

                                    // Process complete lines from the response buffer
                                    while let Some(nl) = response[last_line_start..].find('\n') {
                                        let line_end = last_line_start + nl;
                                        let line = response[last_line_start..line_end].to_string();
                                        last_line_start = line_end + 1;

                                        if !in_code_fence {
                                            // Check for opening fence
                                            let trimmed = line.trim();
                                            if trimmed == "<code>" {
                                                // Flush accumulated prose
                                                let prose = current_prose.trim().to_string();
                                                if !prose.is_empty() {
                                                    prose_parts.push(prose);
                                                }
                                                current_prose.clear();
                                                in_code_fence = true;
                                                current_code.clear();
                                            } else {
                                                // Prose line — emit as TextDelta
                                                if !current_prose.is_empty() {
                                                    current_prose.push('\n');
                                                }
                                                current_prose.push_str(&line);
                                                emit!(AgentEvent::TextDelta {
                                                    content: format!("{}\n", line),
                                                });
                                            }
                                        } else {
                                            // Inside code fence
                                            let trimmed = line.trim();
                                            if trimmed == "</code>" {
                                                // Closing fence — execute the code block
                                                in_code_fence = false;
                                                let code = current_code.clone();
                                                if !code.trim().is_empty() {
                                                    code_parts.push(code.clone());
                                                    emit!(AgentEvent::CodeBlock {
                                                        code: code.clone(),
                                                    });
                                                    if !acc.had_failure {
                                                        execute_and_collect(
                                                            &mut self.session,
                                                            &code,
                                                            &mut acc,
                                                            &event_tx,
                                                        )
                                                        .await;
                                                    }
                                                    code_executed = true;
                                                    break; // break line-processing loop
                                                }
                                                current_code.clear();
                                            } else {
                                                // Code line — accumulate (no TextDelta)
                                                if !current_code.is_empty() {
                                                    current_code.push('\n');
                                                }
                                                current_code.push_str(&line);
                                            }
                                        }
                                    }
                                    if code_executed {
                                        break; // break stream loop
                                    }
                                }
                            }
                            Ok(Some(Err(e))) => {
                                stream_error = Some(format!("{}", e));
                                break;
                            }
                        }
                    }

                    // Code block detected mid-stream: drop stream, feed results back
                    if code_executed {
                        emit!(AgentEvent::LlmResponse {
                            iteration,
                            content: response.clone(),
                            duration_ms: llm_start.elapsed().as_millis() as u64,
                        });
                        break 'llm_retry response.clone();
                    }

                    if let Some(err) = stream_error {
                        if !acc.tool_calls.is_empty()
                            || !acc.combined_output.is_empty()
                            || acc.had_failure
                        {
                            emit!(make_error_event(
                                "llm_provider",
                                Some("stream_error_partial"),
                                format!("LLM stream error (after partial execution): {}", err),
                                Some(err),
                            ));
                            break 'llm_retry response.clone();
                        }
                        if is_retryable(&err) && attempt < LLM_MAX_RETRIES {
                            last_error = Some(err);
                            continue;
                        }
                        self.session.clear_message_sender();
                        self.session.clear_prompt_sender();
                        let _ = collect_usage(
                            &collector,
                            &mut cumulative_usage,
                            iteration,
                            &response,
                            &self.config,
                            &self.agent_id,
                            &event_tx,
                        )
                        .await;
                        let mut env = sanitize_llm_error(&err);
                        let http = last_http_call_summary(&collector)
                            .map(|s| format!("\n{}", s))
                            .unwrap_or_default();
                        let display = format!("LLM error: {}{}", env.user_message, http);
                        env.user_message = display.clone();
                        emit!(AgentEvent::Error {
                            message: display,
                            envelope: Some(env),
                        });
                        emit!(AgentEvent::Done);
                        return (msgs, iteration);
                    }

                    // Finalize the BAML stream (for collector/usage tracking).
                    match call.get_final_response().await {
                        Ok(_) => {}
                        Err(e) => tracing::warn!("get_final_response: {}", e),
                    };

                    emit!(AgentEvent::LlmResponse {
                        iteration,
                        content: response.clone(),
                        duration_ms: llm_start.elapsed().as_millis() as u64,
                    });

                    // Handle any remaining content after stream ends
                    // If we were in a code fence (unclosed), execute remaining code
                    if in_code_fence && !current_code.trim().is_empty() {
                        code_parts.push(current_code.clone());
                        emit!(AgentEvent::CodeBlock {
                            code: current_code.clone(),
                        });
                        if !acc.had_failure {
                            execute_and_collect(
                                &mut self.session,
                                &current_code,
                                &mut acc,
                                &event_tx,
                            )
                            .await;
                        }
                        current_code.clear();
                        in_code_fence = false;
                    }

                    // Flush any remaining prose
                    let remaining_prose = current_prose.trim().to_string();
                    if !remaining_prose.is_empty() {
                        prose_parts.push(remaining_prose);
                    }
                    // Also handle any trailing incomplete line as prose
                    if last_line_start < response.len() && !in_code_fence {
                        let trailing = response[last_line_start..].trim().to_string();
                        if !trailing.is_empty() {
                            emit!(AgentEvent::TextDelta {
                                content: trailing.clone(),
                            });
                            if let Some(last) = prose_parts.last_mut() {
                                last.push('\n');
                                last.push_str(&trailing);
                            } else {
                                prose_parts.push(trailing);
                            }
                        }
                    }

                    break 'llm_retry response.clone();
                }

                // All retries exhausted
                if let Some(err) = last_error {
                    emit!(make_error_event(
                        "llm_provider",
                        Some("retries_exhausted"),
                        format!("LLM failed after {} retries: {}", LLM_MAX_RETRIES + 1, err),
                        Some(err),
                    ));
                }
                String::new()
            };

            // Clean up message and prompt forwarding
            self.session.clear_message_sender();
            self.session.clear_prompt_sender();
            let _ = drain_handle.await;
            let _ = prompt_drain_handle.await;

            // Collect token usage for all paths that exit the retry block
            // (normal completion, stream error with partial exec, retries exhausted)
            last_input_tokens = collect_usage(
                &collector,
                &mut cumulative_usage,
                iteration,
                &full_text,
                &self.config,
                &self.agent_id,
                &event_tx,
            )
            .await;

            // For mid-stream break: flush any remaining prose
            if code_executed {
                let remaining_prose = current_prose.trim().to_string();
                if !remaining_prose.is_empty() {
                    prose_parts.push(remaining_prose);
                }
            }

            let executed_text = full_text;
            let has_code = !code_parts.is_empty();

            // Check for empty response with no execution
            if executed_text.trim().is_empty()
                && acc.tool_calls.is_empty()
                && acc.combined_output.is_empty()
                && !acc.had_failure
            {
                tracing::warn!("LLM returned empty response");
                emit!(make_error_event(
                    "llm_provider",
                    Some("empty_response"),
                    "I didn't get a response — please try again.",
                    None,
                ));
                emit!(AgentEvent::Done);
                return (msgs, iteration);
            }

            // Convert tool images to BAML Image for the next LLM turn
            for img in &acc.images {
                let b64 = base64::engine::general_purpose::STANDARD.encode(&img.data);
                tool_images.push(new_image_from_base64(&b64, Some(&img.mime)));
            }

            // done() = stop signal
            if !acc.final_response.is_empty() {
                emit!(AgentEvent::Message {
                    text: acc.final_response.clone(),
                    kind: "final".to_string(),
                });
                let mid = format!("m{}", msgs.len());
                let asst_parts = build_assistant_parts(&mid, &prose_parts, &code_parts);
                msgs.push(Message {
                    id: mid.clone(),
                    role: MessageRole::Assistant,
                    parts: asst_parts,
                });
                emit!(AgentEvent::Done);
                return (msgs, iteration);
            }

            // Inject turn data into _history (skip for mid-stream code breaks)
            if !code_executed {
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
                // Find the most recent user message for this turn
                let user_msg = msgs
                    .iter()
                    .rev()
                    .find(|m| m.role == MessageRole::User)
                    .map(|m| {
                        m.parts
                            .iter()
                            .map(|p| p.content.as_str())
                            .collect::<Vec<_>>()
                            .join("\n")
                    })
                    .unwrap_or_default();
                let turn_json = serde_json::json!({
                    "index": iteration,
                    "user_message": user_msg,
                    "prose": prose_parts.join("\n\n"),
                    "code": code_parts.join("\n"),
                    "output": acc.combined_output,
                    "error": acc.exec_error,
                    "tool_calls": tool_calls_json,
                });
                let json_str = turn_json
                    .to_string()
                    .replace('\\', "\\\\")
                    .replace('\'', "\\'");
                let inject_code = format!(
                    "_history._add_turn('{}'); _mem._set_turn({})",
                    json_str, iteration
                );
                let _ = self.session.run_code(&inject_code).await;
            }

            // Build structured feedback parts from accumulated execution state
            {
                let has_output = !acc.combined_output.is_empty();
                let has_tool_calls = !acc.tool_calls.is_empty();

                // Pure prose response with no code execution — turn ends
                if !has_code && !has_output && !has_tool_calls && !acc.had_failure {
                    let mid = format!("m{}", msgs.len());
                    let asst_parts = build_assistant_parts(&mid, &prose_parts, &code_parts);
                    msgs.push(Message {
                        id: mid.clone(),
                        role: MessageRole::Assistant,
                        parts: asst_parts,
                    });
                    emit!(AgentEvent::Done);
                    return (msgs, iteration);
                }

                // Build feedback: code blocks wrapped in fences, then output/error
                let mut feedback_parts: Vec<Part> = Vec::new();
                for code in &code_parts {
                    feedback_parts.push(Part {
                        id: String::new(),
                        kind: PartKind::Code,
                        content: code.clone(),
                        prune_state: PruneState::Intact,
                    });
                }

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
                }

                // Push assistant message with prose+code parts
                let asst_id = format!("m{}", msgs.len());
                let asst_parts = build_assistant_parts(&asst_id, &prose_parts, &code_parts);
                msgs.push(Message {
                    id: asst_id.clone(),
                    role: MessageRole::Assistant,
                    parts: asst_parts,
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

                // Inject any newly discovered context-aware instructions from file reads.
                let context_text =
                    resolve_context_instructions(instruction_source.as_ref(), &acc.tool_calls);
                if !context_text.is_empty() {
                    let instruction_id = format!("m{}", msgs.len());
                    msgs.push(Message {
                        id: instruction_id.clone(),
                        role: MessageRole::System,
                        parts: vec![Part {
                            id: format!("{}.p0", instruction_id),
                            kind: PartKind::Text,
                            content: context_text,
                            prune_state: PruneState::Intact,
                        }],
                    });
                }
            }

            iteration += 1;
            // Agent had its grace turn after the turn-limit message — force return
            if max_steps_final {
                emit!(AgentEvent::Done);
                return (msgs, iteration);
            }
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
                            "Turn limit reached ({max}). You MUST call done() now with:\n\
                                1. Summary of what you accomplished\n\
                                2. List of remaining tasks not yet completed\n\
                                3. Recommended next steps\n\
                                Do NOT make any more tool calls. Call done() immediately."
                        ),
                        prune_state: PruneState::Intact,
                    }],
                });
                max_steps_final = true;
            }
        }
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

/// Best-effort HTTP call summary from the latest selected LLM call.
/// Excludes sensitive headers/body; intended for transport debugging.
fn last_http_call_summary(collector: &baml::Collector) -> Option<String> {
    let log = collector.last()?;
    let call = log.selected_call()?;
    let req = call.http_request()?;
    let method = req.method();
    let url = req.url();
    Some(format!("HTTP {} {}", method, url))
}

/// Trim noisy transport payloads and build a standardized provider-error envelope.
fn sanitize_llm_error(raw: &str) -> ErrorEnvelope {
    let trimmed = raw.trim();
    let lower = trimmed.to_ascii_lowercase();
    let user_message = if let Some(idx) = lower.find("<html") {
        let head = trimmed[..idx]
            .trim_end_matches(|c: char| c == ',' || c.is_whitespace())
            .trim();
        if head.is_empty() {
            "request failed (HTML error body omitted)".to_string()
        } else {
            format!("{head} [HTML body omitted]")
        }
    } else {
        trimmed.to_string()
    };
    ErrorEnvelope {
        kind: "llm_provider".to_string(),
        code: infer_provider_error_code(&lower).map(str::to_string),
        user_message,
        raw: Some(truncate_raw_error(trimmed)),
    }
}

fn infer_provider_error_code(lower_msg: &str) -> Option<&'static str> {
    if lower_msg.contains("timed out") || lower_msg.contains("timeout") {
        return Some("timeout");
    }
    if lower_msg.contains("429") || lower_msg.contains("rate limit") {
        return Some("rate_limited");
    }
    if lower_msg.contains("401") || lower_msg.contains("unauthorized") {
        return Some("unauthorized");
    }
    if lower_msg.contains("403") || lower_msg.contains("forbidden") {
        return Some("forbidden");
    }
    if lower_msg.contains("400") || lower_msg.contains("bad request") {
        return Some("bad_request");
    }
    if lower_msg.contains("404") || lower_msg.contains("not found") {
        return Some("not_found");
    }
    if lower_msg.contains("500")
        || lower_msg.contains("502")
        || lower_msg.contains("503")
        || lower_msg.contains("504")
        || lower_msg.contains("internal error")
    {
        return Some("server_error");
    }
    None
}

/// Read token usage from the collector, emit a TokenUsage event, and log debug info.
/// Called after both stop-marker and normal LLM completion paths.
/// Returns the input_tokens from this call (the actual context size seen by the API).
async fn collect_usage(
    collector: &baml::Collector,
    cumulative_usage: &mut TokenUsage,
    iteration: usize,
    response_text: &str,
    config: &AgentConfig,
    agent_id: &str,
    event_tx: &mpsc::Sender<AgentEvent>,
) -> usize {
    if let Some(log) = collector.last() {
        let u = log.usage();
        let usage = TokenUsage {
            input_tokens: u.input_tokens(),
            output_tokens: u.output_tokens(),
            cached_input_tokens: u.cached_input_tokens().unwrap_or(0),
        };
        let this_input = usage.input_tokens as usize;
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
        this_input
    } else {
        0
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

/// Build alternating Prose/Code parts for an assistant message.
fn build_assistant_parts(msg_id: &str, prose_parts: &[String], code_parts: &[String]) -> Vec<Part> {
    let mut parts = Vec::new();
    let mut idx = 0;
    let mut prose_iter = prose_parts.iter();
    let mut code_iter = code_parts.iter();

    // Interleave prose and code parts in order they appeared
    // Pattern: prose0, code0, prose1, code1, ...
    // (first prose may be empty if response starts with code)
    loop {
        let prose = prose_iter.next();
        let code = code_iter.next();
        if prose.is_none() && code.is_none() {
            break;
        }
        if let Some(p) = prose
            && !p.is_empty()
        {
            parts.push(Part {
                id: format!("{}.p{}", msg_id, idx),
                kind: PartKind::Prose,
                content: p.clone(),
                prune_state: PruneState::Intact,
            });
            idx += 1;
        }
        if let Some(c) = code
            && !c.is_empty()
        {
            parts.push(Part {
                id: format!("{}.p{}", msg_id, idx),
                kind: PartKind::Code,
                content: c.clone(),
                prune_state: PruneState::Intact,
            });
            idx += 1;
        }
    }

    // If nothing was added (shouldn't happen), add an empty prose part
    if parts.is_empty() {
        parts.push(Part {
            id: format!("{}.p0", msg_id),
            kind: PartKind::Prose,
            content: String::new(),
            prune_state: PruneState::Intact,
        });
    }

    parts
}

/// Resolve and aggregate context-aware instructions discovered during this turn.
/// We currently trigger this on successful `read_file` tool calls with a `path` argument.
fn resolve_context_instructions(
    source: &dyn InstructionSource,
    tool_calls: &[crate::ToolCallRecord],
) -> String {
    let mut seen_paths = HashSet::new();
    let read_paths: Vec<String> = tool_calls
        .iter()
        .filter(|tc| tc.success && tc.tool == "read_file")
        .filter_map(|tc| tc.args.get("path").and_then(|v| v.as_str()))
        .filter(|p| !p.is_empty())
        .filter(|p| seen_paths.insert((*p).to_string()))
        .map(str::to_string)
        .collect();
    source.context_instructions_for_reads(&read_paths)
}
