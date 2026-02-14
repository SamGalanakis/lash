use std::sync::Arc;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::store::Store;
use crate::baml_client::ClientRegistry;
use crate::baml_client::TypeBuilder;
use crate::baml_client::async_client::B;
use crate::baml_client::new_image_from_base64;
pub use crate::baml_client::types::{ChatMsg, Image};
use crate::baml_client::types::{
    PartForTriage, PartRole, Union3DeleteDecisionOrKeepDecisionOrSummarizeDecision,
};
use crate::provider::Provider;
use crate::session::Session;
use crate::ToolDefinition;

// ─── Structured message types for context-aware pruning ───

/// A structured message with typed parts for context management.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Message {
    pub id: String,
    pub role: MessageRole,
    pub parts: Vec<Part>,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum MessageRole {
    User,
    Assistant,
    System,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Part {
    /// e.g. "m3.p0"
    pub id: String,
    pub kind: PartKind,
    pub content: String,
    pub prune_state: PruneState,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum PartKind {
    Text,
    Code,
    Output,
    Error,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum PruneState {
    Intact,
    Deleted {
        breadcrumb: String,
        archive_hash: String,
    },
    Summarized {
        summary: String,
        archive_hash: String,
    },
}

impl Part {
    fn render(&self) -> String {
        match &self.prune_state {
            PruneState::Intact => self.content.clone(),
            PruneState::Deleted {
                breadcrumb,
                archive_hash,
            } => format!("[pruned:{} — {}]", archive_hash, breadcrumb),
            PruneState::Summarized {
                summary,
                archive_hash,
            } => format!("[SUMMARY of original {}]\n{}", archive_hash, summary),
        }
    }
}

impl Message {
    /// Convert structured message to ChatMsg for LLM prompt.
    pub fn to_chat_msg(&self) -> ChatMsg {
        let content = self
            .parts
            .iter()
            .map(|p| p.render())
            .collect::<Vec<_>>()
            .join("\n\n");
        ChatMsg {
            role: match self.role {
                MessageRole::User => "user",
                MessageRole::Assistant => "assistant",
                MessageRole::System => "system",
            }
            .to_string(),
            content,
        }
    }

    /// Build a Message from a ChatMsg (for backward compat with session loading).
    /// Parses the structured feedback format for system messages.
    pub fn from_chat_msg(msg: &ChatMsg, id: String) -> Self {
        let role = match msg.role.as_str() {
            "user" => MessageRole::User,
            "assistant" => MessageRole::Assistant,
            _ => MessageRole::System,
        };

        // Parse system messages with [Executed]/[Output]/[Error] sections
        if role == MessageRole::System {
            let mut parts = Vec::new();
            let content = &msg.content;

            // Try to split on known section markers
            let sections = parse_system_sections(content);
            for (idx, (kind, text)) in sections.into_iter().enumerate() {
                parts.push(Part {
                    id: format!("{}.p{}", id, idx),
                    kind,
                    content: text,
                    prune_state: PruneState::Intact,
                });
            }

            if parts.is_empty() {
                parts.push(Part {
                    id: format!("{}.p0", id),
                    kind: PartKind::Text,
                    content: content.clone(),
                    prune_state: PruneState::Intact,
                });
            }

            return Message { id, role, parts };
        }

        // User and assistant messages: single Text part
        Message {
            id: id.clone(),
            role,
            parts: vec![Part {
                id: format!("{}.p0", id),
                kind: if role == MessageRole::Assistant {
                    PartKind::Code
                } else {
                    PartKind::Text
                },
                content: msg.content.clone(),
                prune_state: PruneState::Intact,
            }],
        }
    }

    /// Total character count of all parts (rendered).
    pub fn char_count(&self) -> usize {
        self.parts.iter().map(|p| p.render().len()).sum()
    }
}

/// Parse system feedback into typed sections.
fn parse_system_sections(content: &str) -> Vec<(PartKind, String)> {
    let mut sections = Vec::new();
    let mut current_kind = None;
    let mut current_text = String::new();

    for line in content.lines() {
        if line == "[Executed]" {
            if let Some(kind) = current_kind.take() {
                sections.push((kind, current_text.trim().to_string()));
                current_text.clear();
            }
            current_kind = Some(PartKind::Code);
        } else if line == "[Output]" {
            if let Some(kind) = current_kind.take() {
                sections.push((kind, current_text.trim().to_string()));
                current_text.clear();
            }
            current_kind = Some(PartKind::Output);
        } else if line == "[Error]" {
            if let Some(kind) = current_kind.take() {
                sections.push((kind, current_text.trim().to_string()));
                current_text.clear();
            }
            current_kind = Some(PartKind::Error);
        } else {
            if current_kind.is_none() {
                current_kind = Some(PartKind::Text);
            }
            if !current_text.is_empty() {
                current_text.push('\n');
            }
            current_text.push_str(line);
        }
    }

    if let Some(kind) = current_kind {
        let text = current_text.trim().to_string();
        if !text.is_empty() {
            sections.push((kind, text));
        }
    }

    sections
}

/// Convert Vec<ChatMsg> to Vec<Message> (for loading legacy sessions).
pub fn messages_from_chat(msgs: &[ChatMsg]) -> Vec<Message> {
    msgs.iter()
        .enumerate()
        .map(|(i, m)| Message::from_chat_msg(m, format!("m{}", i)))
        .collect()
}

/// Convert Vec<Message> to Vec<ChatMsg> (for LLM calls).
pub fn messages_to_chat(msgs: &[Message]) -> Vec<ChatMsg> {
    msgs.iter().map(|m| m.to_chat_msg()).collect()
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
    pub fn new(session: Session, config: AgentConfig, store: Arc<Store>, agent_id: Option<String>) -> Self {
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
    /// Returns None if snapshotting fails.
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
    /// Returns the updated message history (including assistant/system feedback).
    /// The `cancel` token can be used to interrupt the agent gracefully.
    /// `images` are base64-encoded PNG bytes to attach to the user message.
    pub async fn run(
        &mut self,
        messages: Vec<Message>,
        images: Vec<Vec<u8>>,
        event_tx: mpsc::Sender<AgentEvent>,
        cancel: CancellationToken,
    ) -> Vec<Message> {
        macro_rules! emit {
            ($event:expr) => {
                if !event_tx.is_closed() {
                    let _ = event_tx.send($event).await;
                }
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
                return messages;
            }
            _ => {}
        }

        // Build ClientRegistry
        let model = self.config.provider.resolve_model(&self.config.model);
        let mut cr = ClientRegistry::new();
        cr.add_llm_client(
            "DefaultClient",
            self.config.provider.baml_provider(),
            self.config.provider.baml_options(&model, self.config.reasoning_effort.as_deref()),
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
        let mut iteration: usize = 0;
        let mut tool_images: Vec<Image> = Vec::new();

        loop {
            // Check for cancellation at the start of each iteration
            if cancel.is_cancelled() {
                self.snapshot_to_store(None, &msgs, iteration, &cumulative_usage).await;
                emit!(AgentEvent::Done);
                return msgs;
            }

            // Prune context to stay within budget
            prune_messages(&mut msgs, self.config.max_context_chars, &self.store, &cr).await;

            // Convert structured messages to ChatMsg for LLM
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
            let mut all_tool_calls: Vec<crate::ToolCallRecord> = Vec::new();
            let mut exec_images: Vec<crate::ToolImage> = Vec::new();
            let mut final_response = String::new();
            let mut combined_output = String::new();
            let mut exec_error: Option<String> = None;
            let mut had_exec_failure = false;

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

                    // Combine user-provided images with any tool-returned images (e.g. read_file on PNG)
                    let all_images: Vec<Image> = user_images.iter().chain(tool_images.iter()).cloned().collect();
                    tool_images.clear();

                    let mut call = match if self.config.sub_agent {
                        B.SubAgentStep
                            .with_client_registry(&cr)
                            .with_collector(&collector)
                            .stream(&chat_msgs, &tool_list, &context, &tool_names, &project_instructions, &all_images)
                    } else {
                        B.CodeActStep
                            .with_client_registry(&cr)
                            .with_collector(&collector)
                            .stream(&chat_msgs, &tool_list, &context, &tool_names, &project_instructions, &all_images)
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
                        // Check cancellation during streaming
                        if cancel.is_cancelled() {
                            self.session.clear_message_sender();
                            self.session.clear_prompt_sender();
                            self.snapshot_to_store(None, &msgs, iteration, &cumulative_usage).await;
                            emit!(AgentEvent::Done);
                            return msgs;
                        }
                        match tokio::time::timeout(LLM_STREAM_TIMEOUT, call.next()).await {
                            Err(_timeout) => {
                                self.session.clear_message_sender();
                                self.session.clear_prompt_sender();
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

                                    // Accumulate for incremental execution
                                    line_buffer.push_str(delta);
                                    if delta.contains('\n') && !had_exec_failure {
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
                                                    let ready =
                                                        line_buffer[..pos].to_string();
                                                    line_buffer =
                                                        line_buffer[pos..].to_string();
                                                    emit!(AgentEvent::CodeBlock {
                                                        code: ready.clone(),
                                                    });
                                                    match self
                                                        .session
                                                        .run_code(&ready)
                                                        .await
                                                    {
                                                        Ok(r) => {
                                                            for tc in &r.tool_calls {
                                                                emit!(AgentEvent::ToolCall {
                                                                    name: tc.tool.clone(),
                                                                    args: tc.args.clone(),
                                                                    result: tc.result.clone(),
                                                                    success: tc.success,
                                                                    duration_ms: tc.duration_ms,
                                                                });
                                                                if tc.tool == "delegate_task"
                                                                    || tc.tool
                                                                        == "delegate_search"
                                                                    || tc.tool
                                                                        == "delegate_deep"
                                                                {
                                                                    if let Some(sub) =
                                                                        tc.result
                                                                            .get("_sub_agent")
                                                                    {
                                                                        let task = sub.get("task").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                                                        let usage: TokenUsage = sub.get("usage").and_then(|v| serde_json::from_value(v.clone()).ok()).unwrap_or_default();
                                                                        let tc_count = sub.get("tool_calls").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                                                                        let iters = sub.get("iterations").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                                                                        emit!(AgentEvent::SubAgentDone {
                                                                            task,
                                                                            usage,
                                                                            tool_calls: tc_count,
                                                                            iterations: iters,
                                                                            success: tc.success,
                                                                        });
                                                                    }
                                                                }
                                                            }
                                                            if !r.output.is_empty()
                                                                || r.error.is_some()
                                                            {
                                                                emit!(AgentEvent::CodeOutput {
                                                                    output: r.output.clone(),
                                                                    error: r.error.clone(),
                                                                });
                                                            }
                                                            all_tool_calls
                                                                .extend(r.tool_calls);
                                                            exec_images.extend(r.images);
                                                            if !r.output.is_empty() {
                                                                combined_output
                                                                    .push_str(&r.output);
                                                            }
                                                            if !r.response.is_empty() {
                                                                final_response = r.response;
                                                            }
                                                            if r.error.is_some() {
                                                                exec_error = r.error;
                                                                had_exec_failure = true;
                                                            }
                                                        }
                                                        Err(e) => {
                                                            emit!(AgentEvent::CodeOutput {
                                                                output: String::new(),
                                                                error: Some(format!(
                                                                    "{}",
                                                                    e
                                                                )),
                                                            });
                                                            exec_error =
                                                                Some(format!("{}", e));
                                                            had_exec_failure = true;
                                                        }
                                                    }
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
                                    tracing::warn!("LLM response exceeded {} chars, aborting", LLM_MAX_RESPONSE_CHARS);
                                    self.session.clear_message_sender();
                                    self.session.clear_prompt_sender();
                                    emit!(AgentEvent::Error {
                                        message: format!("Response exceeded {} chars — likely degenerate output, aborting", LLM_MAX_RESPONSE_CHARS),
                                    });
                                    emit!(AgentEvent::Done);
                                    return msgs;
                                }
                            }
                            Ok(Some(Err(e))) => {
                                stream_error = Some(format!("{}", e));
                                break;
                            }
                        }
                    }

                    if let Some(err) = stream_error {
                        // Don't retry if code was already executed — side effects can't be undone
                        if !all_tool_calls.is_empty()
                            || !combined_output.is_empty()
                            || had_exec_failure
                        {
                            emit!(AgentEvent::Error {
                                message: format!("LLM stream error (after partial execution): {}", err),
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

                    // Read token usage from collector
                    if let Some(log) = collector.last() {
                        let u = log.usage();
                        let usage = TokenUsage {
                            input_tokens: u.input_tokens(),
                            output_tokens: u.output_tokens(),
                            cached_input_tokens: u.cached_input_tokens().unwrap_or(0),
                        };
                        cumulative_usage.add(&usage);
                        emit!(AgentEvent::TokenUsage {
                            iteration,
                            usage,
                            cumulative: cumulative_usage.clone(),
                        });
                    }

                    // Execute remaining buffer
                    if !line_buffer.trim().is_empty() && !had_exec_failure {
                        emit!(AgentEvent::CodeBlock {
                            code: line_buffer.clone(),
                        });
                        match self.session.run_code(&line_buffer).await {
                            Ok(r) => {
                                for tc in &r.tool_calls {
                                    emit!(AgentEvent::ToolCall {
                                        name: tc.tool.clone(),
                                        args: tc.args.clone(),
                                        result: tc.result.clone(),
                                        success: tc.success,
                                        duration_ms: tc.duration_ms,
                                    });
                                    if tc.tool == "delegate_task"
                                        || tc.tool == "delegate_search"
                                        || tc.tool == "delegate_deep"
                                    {
                                        if let Some(sub) = tc.result.get("_sub_agent") {
                                            let task = sub.get("task").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                            let usage: TokenUsage = sub.get("usage").and_then(|v| serde_json::from_value(v.clone()).ok()).unwrap_or_default();
                                            let tc_count = sub.get("tool_calls").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                                            let iters = sub.get("iterations").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                                            emit!(AgentEvent::SubAgentDone {
                                                task,
                                                usage,
                                                tool_calls: tc_count,
                                                iterations: iters,
                                                success: tc.success,
                                            });
                                        }
                                    }
                                }
                                if !r.output.is_empty() || r.error.is_some() {
                                    emit!(AgentEvent::CodeOutput {
                                        output: r.output.clone(),
                                        error: r.error.clone(),
                                    });
                                }
                                all_tool_calls.extend(r.tool_calls);
                                exec_images.extend(r.images);
                                if !r.output.is_empty() {
                                    combined_output.push_str(&r.output);
                                }
                                if !r.response.is_empty() {
                                    final_response = r.response;
                                }
                                if r.error.is_some() {
                                    exec_error = r.error;
                                }
                            }
                            Err(e) => {
                                emit!(AgentEvent::CodeOutput {
                                    output: String::new(),
                                    error: Some(format!("{}", e)),
                                });
                                exec_error = Some(format!("{}", e));
                            }
                        }
                    }

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

            // Clean up message and prompt forwarding
            self.session.clear_message_sender();
            self.session.clear_prompt_sender();
            let _ = drain_handle.await;
            let _ = prompt_drain_handle.await;

            // Check for empty response with no execution
            if full_text.trim().is_empty()
                && all_tool_calls.is_empty()
                && combined_output.is_empty()
                && !had_exec_failure
            {
                tracing::warn!("LLM returned empty response");
                emit!(AgentEvent::Error {
                    message: "I didn't get a response — please try again.".to_string()
                });
                emit!(AgentEvent::Done);
                return msgs;
            }

            // Convert tool images to BAML Image for the next LLM turn
            for img in &exec_images {
                let b64 = base64::engine::general_purpose::STANDARD.encode(&img.data);
                tool_images.push(new_image_from_base64(&b64, Some(&img.mime)));
            }

            // respond() = stop signal
            if !final_response.is_empty() {
                emit!(AgentEvent::Message {
                    text: final_response.clone(),
                    kind: "final".to_string(),
                });
                let mid = format!("m{}", msgs.len());
                msgs.push(Message {
                    id: mid.clone(),
                    role: MessageRole::Assistant,
                    parts: vec![Part {
                        id: format!("{}.p0", mid),
                        kind: PartKind::Code,
                        content: full_text.clone(),
                        prune_state: PruneState::Intact,
                    }],
                });
                self.store.mark_agent_done(&self.agent_id);
                emit!(AgentEvent::Done);
                return msgs;
            }

            // Build structured feedback parts from accumulated execution state
            {
                let mut feedback_parts = vec![Part {
                    id: String::new(),
                    kind: PartKind::Code,
                    content: full_text.clone(),
                    prune_state: PruneState::Intact,
                }];

                if let Some(err) = &exec_error {
                    feedback_parts.push(Part {
                        id: String::new(),
                        kind: PartKind::Error,
                        content: format!("{}\nFix and retry.", err),
                        prune_state: PruneState::Intact,
                    });
                } else if !combined_output.is_empty() {
                    let mut output_text = combined_output;
                    if !all_tool_calls.is_empty() {
                        output_text.push_str(&format!(
                            "\n[{} tool call(s) executed]",
                            all_tool_calls.len()
                        ));
                    }
                    feedback_parts.push(Part {
                        id: String::new(),
                        kind: PartKind::Output,
                        content: output_text,
                        prune_state: PruneState::Intact,
                    });
                } else if !all_tool_calls.is_empty() {
                    feedback_parts.push(Part {
                        id: String::new(),
                        kind: PartKind::Output,
                        content: format!(
                            "[{} tool call(s) executed]",
                            all_tool_calls.len()
                        ),
                        prune_state: PruneState::Intact,
                    });
                } else {
                    // No output, no error, no tool calls
                    let mid = format!("m{}", msgs.len());
                    msgs.push(Message {
                        id: mid.clone(),
                        role: MessageRole::Assistant,
                        parts: vec![Part {
                            id: format!("{}.p0", mid),
                            kind: PartKind::Code,
                            content: full_text.clone(),
                            prune_state: PruneState::Intact,
                        }],
                    });
                    self.store.mark_agent_done(&self.agent_id);
                    emit!(AgentEvent::Done);
                    return msgs;
                }

                // Push assistant message
                let asst_id = format!("m{}", msgs.len());
                msgs.push(Message {
                    id: asst_id.clone(),
                    role: MessageRole::Assistant,
                    parts: vec![Part {
                        id: format!("{}.p0", asst_id),
                        kind: PartKind::Code,
                        content: full_text.clone(),
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
            if let Some(max) = self.config.max_turns {
                if iteration >= max {
                    emit!(AgentEvent::Error {
                        message: format!("Turn limit reached ({max})"),
                    });
                    self.store.mark_agent_done(&self.agent_id);
                    emit!(AgentEvent::Done);
                    return msgs;
                }
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

/// Build environment context string for the system prompt.
fn build_context() -> String {
    let mut parts = Vec::new();

    if let Ok(cwd) = std::env::current_dir() {
        parts.push(format!("Working directory: {}", cwd.display()));

        // Detect git repo
        let git_dir = cwd.join(".git");
        if git_dir.exists() {
            parts.push("Git repository: yes".into());
        }

        // List top-level entries so the model knows what exists
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

/// Find byte offsets of candidate statement boundaries in code.
/// A candidate is a line starting with a non-whitespace, non-`#` character.
/// Skips position 0 (nothing before it to split off).
fn find_candidate_boundaries(code: &str) -> Vec<usize> {
    let mut boundaries = Vec::new();
    for (i, _) in code.match_indices('\n') {
        if let Some(c) = code[i + 1..].chars().next() {
            if !c.is_whitespace() && c != '#' {
                boundaries.push(i + 1);
            }
        }
    }
    boundaries
}

/// Context-aware pruning: uses LLM triage to classify parts as keep/delete/summarize.
/// Falls back to dropping oldest messages if still over hard budget.
async fn prune_messages(
    messages: &mut Vec<Message>,
    budget: usize,
    store: &Arc<Store>,
    cr: &ClientRegistry,
) {
    let total: usize = messages.iter().map(|m| m.char_count()).sum();
    if total <= budget {
        return;
    }
    if messages.len() <= 5 {
        return;
    }

    const PROTECT_TAIL: usize = 4;
    const PROTECT_CHARS: usize = 40_000;

    // 1. Collect prunable parts (skip first msg, skip last PROTECT_TAIL)
    // Walk backward to protect recent content first
    let mut prunable: Vec<(usize, usize, String)> = vec![]; // (msg_idx, part_idx, part_id)
    let mut protected_chars = 0usize;

    let last_prunable = messages.len().saturating_sub(PROTECT_TAIL);

    for msg_idx in (1..last_prunable).rev() {
        for (part_idx, part) in messages[msg_idx].parts.iter().enumerate().rev() {
            if !matches!(part.prune_state, PruneState::Intact) {
                continue;
            }
            // Only prune Output and Error parts (leave Text/Code alone)
            if !matches!(part.kind, PartKind::Output | PartKind::Error) {
                continue;
            }
            if protected_chars < PROTECT_CHARS {
                protected_chars += part.content.len();
                continue; // still in protection window
            }
            prunable.push((msg_idx, part_idx, part.id.clone()));
        }
    }

    if prunable.is_empty() {
        // Nothing to triage — fall back to dropping oldest messages
        fallback_drop(messages, budget);
        return;
    }

    // 2. Build TypeBuilder with dynamic PartId enum
    let tb = TypeBuilder::new();
    match tb.add_enum("PartId") {
        Ok(part_id_enum) => {
            for (_, _, id) in &prunable {
                if let Err(e) = part_id_enum.add_value(id) {
                    tracing::warn!("Failed to add PartId value {}: {}", id, e);
                }
            }
        }
        Err(e) => {
            tracing::warn!("Failed to create PartId enum: {}", e);
            fallback_drop(messages, budget);
            return;
        }
    }

    // 3. Build PartForTriage list
    let triage_parts: Vec<PartForTriage> = prunable
        .iter()
        .map(|(msg_idx, part_idx, _)| {
            let part = &messages[*msg_idx].parts[*part_idx];
            // Truncate very large content to keep the triage prompt manageable
            let content = if part.content.len() > 2000 {
                format!("{}...[truncated, {} total chars]", &part.content[..2000], part.content.len())
            } else {
                part.content.clone()
            };
            PartForTriage {
                id: crate::baml_client::types::PartId::Placeholder, // will be overridden by TypeBuilder
                role: match messages[*msg_idx].role {
                    MessageRole::Assistant => PartRole::Assistant,
                    _ => PartRole::System,
                },
                content,
            }
        })
        .collect();

    // 4. Get task context from first user message
    let task_context = messages
        .iter()
        .find(|m| m.role == MessageRole::User)
        .and_then(|m| m.parts.first())
        .map(|p| {
            if p.content.len() > 500 {
                p.content[..500].to_string()
            } else {
                p.content.clone()
            }
        })
        .unwrap_or_default();

    // 5. Call TriageParts
    let decisions = match B
        .TriageParts
        .with_client_registry(cr)
        .with_type_builder(&tb)
        .call(&triage_parts, &task_context)  // Note: TriageParts is internal, not user-facing — no collector needed
        .await
    {
        Ok(d) => d,
        Err(e) => {
            tracing::warn!("TriageParts call failed: {}, falling back to drop", e);
            fallback_drop(messages, budget);
            return;
        }
    };

    // 6. Apply decisions
    for decision in decisions {
        let (part_id_str, action) = match &decision {
            Union3DeleteDecisionOrKeepDecisionOrSummarizeDecision::KeepDecision(k) => {
                (format!("{}", k.id), "keep")
            }
            Union3DeleteDecisionOrKeepDecisionOrSummarizeDecision::DeleteDecision(d) => {
                (format!("{}", d.id), "delete")
            }
            Union3DeleteDecisionOrKeepDecisionOrSummarizeDecision::SummarizeDecision(s) => {
                (format!("{}", s.id), "summarize")
            }
        };

        // Find the matching part by looking up the part ID in our prunable list
        let Some(&(msg_idx, part_idx, _)) = prunable
            .iter()
            .find(|(_, _, id)| *id == part_id_str)
        else {
            tracing::warn!("Triage returned unknown part_id: {} (action: {})", part_id_str, action);
            continue;
        };

        let part = &mut messages[msg_idx].parts[part_idx];
        match decision {
            Union3DeleteDecisionOrKeepDecisionOrSummarizeDecision::KeepDecision(_) => {
                // No change
            }
            Union3DeleteDecisionOrKeepDecisionOrSummarizeDecision::DeleteDecision(d) => {
                let hash = store.store_archive(&part.content);
                part.prune_state = PruneState::Deleted {
                    breadcrumb: d.breadcrumb,
                    archive_hash: hash,
                };
            }
            Union3DeleteDecisionOrKeepDecisionOrSummarizeDecision::SummarizeDecision(s) => {
                let hash = store.store_archive(&part.content);
                part.prune_state = PruneState::Summarized {
                    summary: s.summary,
                    archive_hash: hash,
                };
            }
        }
    }

    // 7. If still over budget, fall back to dropping oldest
    let new_total: usize = messages.iter().map(|m| m.char_count()).sum();
    if new_total > budget {
        fallback_drop(messages, budget);
    }
}

/// Simple fallback: drop oldest messages (keeping first and last 4).
fn fallback_drop(messages: &mut Vec<Message>, budget: usize) {
    let keep_end = 4;
    let keep_start = 1;
    while messages.len() > keep_start + keep_end {
        let total: usize = messages.iter().map(|m| m.char_count()).sum();
        if total <= budget {
            break;
        }
        messages.remove(keep_start);
    }
}
