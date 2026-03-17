pub(crate) mod context;
pub(crate) mod exec;
pub mod message;
pub(crate) mod prompt;

use tokio::sync::mpsc;

use crate::ContextStrategy;
use crate::ExecutionMode;
use crate::PromptContext;
use crate::ToolDefinition;
use crate::llm::factory::adapter_for;
use crate::llm::types::{LlmStreamEvent, LlmToolSpec};
use crate::plugin::PromptContribution;
use crate::plugin::{CheckpointKind, PluginMessage, PluginSurfaceEvent};
use crate::provider::{OPENAI_GENERIC_DEFAULT_BASE_URL, Provider};
use crate::session::Session;

pub use message::{
    Message, MessageRole, Part, PartKind, PruneState, render_prompt, render_transcript_prompt,
};

pub use prompt::{
    DefaultPromptRenderer, PromptOverrideMode, PromptRenderer, PromptSectionName,
    PromptSectionOverride, default_prompt_renderer,
};

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
    #[serde(default)]
    pub reasoning_tokens: i64,
}

impl TokenUsage {
    pub fn total(&self) -> i64 {
        self.input_tokens + self.output_tokens + self.reasoning_tokens
    }
    pub fn context_total(&self) -> i64 {
        self.total() + self.cached_input_tokens
    }
    pub fn add(&mut self, other: &TokenUsage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.cached_input_tokens += other.cached_input_tokens;
        self.reasoning_tokens += other.reasoning_tokens;
    }
}

/// Resolved session policy for a running agent/session.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SessionPolicy {
    /// Model identifier (e.g. "anthropic/claude-sonnet-4.6")
    pub model: String,
    /// LLM provider (OpenAI-generic, Claude OAuth, Codex, or Google OAuth)
    pub provider: Provider,
    /// Explicit context window size (tokens) supplied by the host.
    pub max_context_tokens: Option<usize>,
    /// When true, use SubAgentStep prompt instead of CodeActStep
    pub sub_agent: bool,
    /// Optional provider-native model variant (e.g. "high", "max", "xhigh").
    pub model_variant: Option<String>,
    /// Optional override model for the recall-agent child session.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recall_agent_model: Option<String>,
    /// Optional host session ID propagated to provider request metadata.
    pub session_id: Option<String>,
    /// Optional turn limit. None = unlimited (default for root agent).
    pub max_turns: Option<usize>,
    /// When true, include Soul principles in the system prompt.
    pub include_soul: bool,
    /// Execution backend for turns.
    pub execution_mode: ExecutionMode,
    /// Strategy for selecting/rendering prior context into the next turn.
    pub context_strategy: ContextStrategy,
}

impl Default for SessionPolicy {
    fn default() -> Self {
        Self {
            model: "anthropic/claude-sonnet-4.6".to_string(),
            provider: Provider::OpenAiGeneric {
                api_key: String::new(),
                base_url: OPENAI_GENERIC_DEFAULT_BASE_URL.to_string(),
                options: crate::provider::ProviderOptions::default(),
            },
            max_context_tokens: None,
            sub_agent: false,
            model_variant: None,
            recall_agent_model: None,
            session_id: None,
            max_turns: None,
            include_soul: false,
            execution_mode: crate::default_execution_mode(),
            context_strategy: crate::default_context_strategy(),
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

/// Low-level events emitted during an agent run.
/// These are intentionally mode-specific; hosts that want a stable cross-mode result should
/// consume `AssembledTurn` from the runtime layer instead.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "type")]
pub enum AgentEvent {
    #[serde(rename = "text_delta")]
    TextDelta { content: String },
    #[serde(rename = "tool_call")]
    ToolCall {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        call_id: Option<String>,
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
    #[serde(rename = "retry_status")]
    RetryStatus {
        wait_seconds: u64,
        attempt: usize,
        max_attempts: usize,
        reason: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        envelope: Option<ErrorEnvelope>,
    },
    #[serde(rename = "injected_messages_committed")]
    InjectedMessagesCommitted {
        messages: Vec<PluginMessage>,
        checkpoint: CheckpointKind,
    },
    #[serde(rename = "plugin_event")]
    PluginEvent {
        plugin_id: String,
        event: PluginSurfaceEvent,
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

/// Max retries for rate-limited or empty LLM responses.
pub(crate) const LLM_MAX_RETRIES: usize = 3;
/// Delays between retries (exponential backoff).
pub(crate) const LLM_RETRY_DELAYS: [std::time::Duration; 3] = [
    std::time::Duration::from_secs(2),
    std::time::Duration::from_secs(5),
    std::time::Duration::from_secs(10),
];

pub(crate) struct TurnTerminationPolicyState {
    max_steps_final: bool,
}

impl TurnTerminationPolicyState {
    pub(crate) fn new() -> Self {
        Self {
            max_steps_final: false,
        }
    }

    pub(crate) fn should_force_exit_after_grace_turn(&self) -> bool {
        self.max_steps_final
    }

    pub(crate) fn maybe_schedule_turn_limit_final(
        &mut self,
        iteration: usize,
        run_offset: usize,
        max_turns: Option<usize>,
        msgs: &mut Vec<Message>,
    ) {
        let Some(max) = max_turns else { return };
        if iteration < run_offset + max {
            return;
        }
        let sys_id = format!("m{}", msgs.len());
        msgs.push(Message {
            id: sys_id.clone(),
            role: MessageRole::System,
            parts: vec![Part {
                id: format!("{}.p0", sys_id),
                kind: PartKind::Text,
                content: format!(
                    "Turn limit reached ({max}). You MUST reply in plain prose now containing:\n\
                        1. Summary of what you accomplished\n\
                        2. List of remaining tasks not yet completed\n\
                        3. Recommended next steps\n\
                        Do NOT make any more tool calls and do NOT emit `<repl>`."
                ),
                tool_call_id: None,
                tool_name: None,
                prune_state: PruneState::Intact,
            }],
            origin: None,
        });
        self.max_steps_final = true;
    }
}

pub(crate) fn make_error_envelope(
    kind: &str,
    code: Option<&str>,
    user_message: impl Into<String>,
    raw: Option<String>,
) -> ErrorEnvelope {
    let user_message = user_message.into();
    ErrorEnvelope {
        kind: kind.to_string(),
        code: code.map(str::to_string),
        user_message,
        raw: raw.map(|s| truncate_raw_error(s.trim())),
    }
}

pub(crate) fn make_error_event(
    kind: &str,
    code: Option<&str>,
    user_message: impl Into<String>,
    raw: Option<String>,
) -> AgentEvent {
    let user_message = user_message.into();
    AgentEvent::Error {
        message: user_message.clone(),
        envelope: Some(make_error_envelope(kind, code, user_message, raw)),
    }
}

pub(crate) fn truncate_raw_error(s: &str) -> String {
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

pub(crate) fn is_malformed_assistant_output(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return false;
    }

    if matches!(
        trimmed,
        "<" | "</" | "<repl" | "</repl" | "<repl>" | "</repl>"
    ) {
        return true;
    }

    // Broken or partial REPL tags should never be surfaced as final user text.
    if (trimmed.starts_with("<repl") && !trimmed.contains("</repl>"))
        || trimmed.starts_with("</repl")
        || trimmed.ends_with("<repl")
        || trimmed.ends_with("</repl")
    {
        return true;
    }

    // Catch short dangling tag-like fragments such as "<", "<re", "</r".
    if trimmed.starts_with('<') && !trimmed.contains('>') && trimmed.len() <= 16 {
        let looks_like_fragment = trimmed
            .chars()
            .skip(1)
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '/' || ch == '_' || ch == '-');
        if looks_like_fragment {
            return true;
        }
    }

    false
}

pub(crate) struct ExecutionPreamble {
    pub(crate) model: String,
    pub(crate) tool_specs: Vec<LlmToolSpec>,
    pub(crate) prompt: PromptContext,
}

pub(crate) fn transport_stream_events(
    provider: &Provider,
    requested: Option<tokio::sync::mpsc::UnboundedSender<LlmStreamEvent>>,
) -> Option<tokio::sync::mpsc::UnboundedSender<LlmStreamEvent>> {
    if requested.is_some() {
        return requested;
    }

    let llm = adapter_for(provider);
    if llm.requires_streaming() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<LlmStreamEvent>();
        drop(rx);
        Some(tx)
    } else {
        None
    }
}

pub(crate) fn build_execution_preamble(
    session: &Session,
    policy: &SessionPolicy,
    mode: ExecutionMode,
    model: String,
) -> ExecutionPreamble {
    let session_id = policy.session_id.as_deref().unwrap_or("root");
    let surface = session.execution_surface(session_id, mode);
    let enabled_tools = surface.enabled_tools();
    let (tool_list, omitted_tool_count) = if matches!(mode, ExecutionMode::Repl) {
        let prompt_tools = surface.prompt_tools();
        let mut tool_list = ToolDefinition::format_tool_docs(&prompt_tools);
        let omitted_tool_count = count_prompt_omitted_tools(&enabled_tools);
        for note in &surface.tool_list_notes {
            tool_list.push_str("\n\n");
            tool_list.push_str(note);
        }
        (tool_list, omitted_tool_count)
    } else {
        (String::new(), 0)
    };
    let tool_specs = if matches!(mode, ExecutionMode::Standard) {
        enabled_tools
            .iter()
            .map(|tool| {
                let model_tool = tool.model_tool();
                LlmToolSpec {
                    name: model_tool.name,
                    description: model_tool.description,
                    input_schema: model_tool.input_schema,
                    output_schema: model_tool.output_schema,
                }
            })
            .collect()
    } else {
        Vec::new()
    };
    let tool_names: Vec<String> = enabled_tools.iter().map(|t| t.name.clone()).collect();
    let can_write = tool_names.iter().any(|name| name == "apply_patch");
    let prompt = PromptContext {
        mode,
        tool_list,
        tool_names,
        omitted_tool_count,
        is_subagent: policy.sub_agent,
        can_write,
        include_soul: if policy.sub_agent {
            policy.include_soul
        } else {
            true
        },
        contributions: Vec::new(),
    };

    ExecutionPreamble {
        model,
        tool_specs,
        prompt,
    }
}

pub(crate) fn finalize_prompt_context(
    mut prompt: PromptContext,
    plugin_prompt_contributions: Vec<PromptContribution>,
) -> PromptContext {
    prompt.contributions.extend(plugin_prompt_contributions);
    prompt
}

fn count_prompt_omitted_tools(all_tools: &[crate::ToolDefinition]) -> usize {
    all_tools
        .iter()
        .filter(|t| t.enabled)
        .filter(|t| !t.injected)
        .count()
}

#[cfg(test)]
mod execution_preamble_tests {
    use super::*;

    #[test]
    fn omitted_tool_count_ignores_hidden_tools_when_everything_else_is_prompted() {
        let defs = vec![
            crate::ToolDefinition {
                name: "read_file".into(),
                description: "Read file".into(),
                params: vec![],
                returns: "str".into(),
                examples: vec![],
                enabled: true,
                injected: true,
            },
            crate::ToolDefinition {
                name: "search_tools".into(),
                description: "Discover tools".into(),
                params: vec![],
                returns: "list".into(),
                examples: vec![],
                enabled: true,
                injected: false,
            },
        ];

        assert_eq!(count_prompt_omitted_tools(&defs), 1);
    }
}

pub(crate) fn format_tool_result_content(success: bool, result: &serde_json::Value) -> String {
    if success {
        match result {
            serde_json::Value::String(text) => text.clone(),
            other => serde_json::to_string(other).unwrap_or_else(|_| "null".to_string()),
        }
    } else {
        match result {
            serde_json::Value::String(text) => {
                if text.is_empty() {
                    "[Tool execution failed]".to_string()
                } else {
                    format!("[Tool execution failed]\n{text}")
                }
            }
            other => serde_json::to_string(&serde_json::json!({ "error": other }))
                .unwrap_or_else(|_| "{\"error\":\"tool execution failed\"}".to_string()),
        }
    }
}

/// Build environment context string for the system prompt.
/// Build alternating Prose/Code parts for an assistant message.
pub(crate) fn build_assistant_parts(
    msg_id: &str,
    prose_parts: &[String],
    code_parts: &[String],
) -> Vec<Part> {
    let mut parts = Vec::new();
    let mut idx = 0;
    let mut prose_iter = prose_parts.iter();
    let mut code_iter = code_parts.iter();

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
                tool_call_id: None,
                tool_name: None,
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
                tool_call_id: None,
                tool_name: None,
                prune_state: PruneState::Intact,
            });
            idx += 1;
        }
    }

    if parts.is_empty() {
        parts.push(Part {
            id: format!("{}.p0", msg_id),
            kind: PartKind::Prose,
            content: String::new(),
            tool_call_id: None,
            tool_name: None,
            prune_state: PruneState::Intact,
        });
    }

    parts
}

pub(crate) struct FenceLineParse {
    pub(crate) prose_delta: String,
    pub(crate) codes_to_execute: Vec<String>,
}

pub(crate) fn append_line_segment(target: &mut String, segment: &str, line_started: &mut bool) {
    if segment.is_empty() {
        return;
    }
    if !*line_started {
        if !target.is_empty() {
            target.push('\n');
        }
        *line_started = true;
    }
    target.push_str(segment);
}

/// Parse one streamed line, accepting `<repl>` / `</repl>` tags inline as well as on
/// standalone lines. Returns prose to emit as a text delta and any completed code
/// blocks encountered on that line.
pub(crate) fn parse_fence_line(
    line: &str,
    in_code_fence: &mut bool,
    current_prose: &mut String,
    current_code: &mut String,
    prose_parts: &mut Vec<String>,
) -> FenceLineParse {
    const OPEN_TAG: &str = "<repl>";
    const CLOSE_TAG: &str = "</repl>";

    let mut out = FenceLineParse {
        prose_delta: String::new(),
        codes_to_execute: Vec::new(),
    };

    let mut remaining = line;
    let mut prose_started_this_line = false;
    let mut code_started_this_line = false;

    if *in_code_fence && line.is_empty() && !current_code.is_empty() {
        current_code.push('\n');
        return out;
    }

    loop {
        if !*in_code_fence {
            if let Some(idx) = remaining.find(OPEN_TAG) {
                let before = &remaining[..idx];
                append_line_segment(current_prose, before, &mut prose_started_this_line);
                out.prose_delta.push_str(before);

                let prose = current_prose.trim().to_string();
                if !prose.is_empty() {
                    prose_parts.push(prose);
                }
                current_prose.clear();
                *in_code_fence = true;
                current_code.clear();
                code_started_this_line = false;
                remaining = &remaining[idx + OPEN_TAG.len()..];
                continue;
            }

            append_line_segment(current_prose, remaining, &mut prose_started_this_line);
            out.prose_delta.push_str(remaining);
            break;
        }

        if let Some(idx) = remaining.find(CLOSE_TAG) {
            let before = &remaining[..idx];
            append_line_segment(current_code, before, &mut code_started_this_line);
            *in_code_fence = false;
            let code = std::mem::take(current_code);
            if !code.trim().is_empty() {
                out.codes_to_execute.push(code);
            }
            break;
        }

        append_line_segment(current_code, remaining, &mut code_started_this_line);
        break;
    }

    out
}

#[cfg(test)]
mod tests {
    use super::{format_tool_result_content, is_malformed_assistant_output, parse_fence_line};

    #[test]
    fn parses_inline_open_and_close_tags() {
        let mut in_code_fence = false;
        let mut current_prose = String::new();
        let mut current_code = String::new();
        let mut prose_parts = Vec::new();

        let out = parse_fence_line(
            "preface <repl>print('x')</repl>",
            &mut in_code_fence,
            &mut current_prose,
            &mut current_code,
            &mut prose_parts,
        );

        assert_eq!(out.prose_delta, "preface ");
        assert_eq!(out.codes_to_execute, vec!["print('x')"]);
        assert_eq!(prose_parts, vec!["preface"]);
        assert!(!in_code_fence);
        assert!(current_prose.is_empty());
        assert!(current_code.is_empty());
    }

    #[test]
    fn parses_inline_open_tag_without_close() {
        let mut in_code_fence = false;
        let mut current_prose = String::new();
        let mut current_code = String::new();
        let mut prose_parts = Vec::new();

        let out = parse_fence_line(
            "note<repl>x = 1",
            &mut in_code_fence,
            &mut current_prose,
            &mut current_code,
            &mut prose_parts,
        );

        assert_eq!(out.prose_delta, "note");
        assert!(out.codes_to_execute.is_empty());
        assert_eq!(prose_parts, vec!["note"]);
        assert!(in_code_fence);
        assert_eq!(current_code, "x = 1");
    }

    #[test]
    fn close_tag_executes_accumulated_multiline_code() {
        let mut in_code_fence = true;
        let mut current_prose = String::new();
        let mut current_code = "x = 1".to_string();
        let mut prose_parts = Vec::new();

        let out = parse_fence_line(
            "print(x)</repl> trailing text",
            &mut in_code_fence,
            &mut current_prose,
            &mut current_code,
            &mut prose_parts,
        );

        assert_eq!(out.codes_to_execute, vec!["x = 1\nprint(x)"]);
        assert!(out.prose_delta.is_empty());
        assert!(!in_code_fence);
        assert!(current_code.is_empty());
    }

    #[test]
    fn ignores_second_inline_block_after_first_closes() {
        let mut in_code_fence = false;
        let mut current_prose = String::new();
        let mut current_code = String::new();
        let mut prose_parts = Vec::new();

        let out = parse_fence_line(
            "<repl>a=1</repl><repl>b=2</repl>",
            &mut in_code_fence,
            &mut current_prose,
            &mut current_code,
            &mut prose_parts,
        );

        assert!(out.prose_delta.is_empty());
        assert_eq!(out.codes_to_execute, vec!["a=1"]);
        assert!(!in_code_fence);
    }

    #[test]
    fn preserves_blank_lines_inside_code_block() {
        let mut in_code_fence = true;
        let mut current_prose = String::new();
        let mut current_code = String::new();
        let mut prose_parts = Vec::new();

        let _ = parse_fence_line(
            "a = 1",
            &mut in_code_fence,
            &mut current_prose,
            &mut current_code,
            &mut prose_parts,
        );
        let _ = parse_fence_line(
            "",
            &mut in_code_fence,
            &mut current_prose,
            &mut current_code,
            &mut prose_parts,
        );
        let out = parse_fence_line(
            "b = 2</repl>",
            &mut in_code_fence,
            &mut current_prose,
            &mut current_code,
            &mut prose_parts,
        );

        assert_eq!(out.codes_to_execute, vec!["a = 1\n\nb = 2"]);
        assert!(!in_code_fence);
    }

    #[test]
    fn malformed_output_detector_flags_partial_repl_fragments() {
        assert!(is_malformed_assistant_output("<"));
        assert!(is_malformed_assistant_output("<re"));
        assert!(is_malformed_assistant_output("</r"));
        assert!(is_malformed_assistant_output("<repl"));
        assert!(is_malformed_assistant_output("</repl"));
        assert!(is_malformed_assistant_output("<repl>"));
        assert!(!is_malformed_assistant_output("All good here."));
        assert!(!is_malformed_assistant_output("<tag>"));
        assert!(!is_malformed_assistant_output(
            "Use `<repl>` tags only when needed."
        ));
    }

    #[test]
    fn formats_successful_tool_results_without_wrapper() {
        assert_eq!(
            format_tool_result_content(true, &serde_json::json!("done")),
            "done"
        );
        assert_eq!(
            format_tool_result_content(true, &serde_json::json!({"count": 2})),
            r#"{"count":2}"#
        );
    }

    #[test]
    fn formats_failed_tool_results_as_explicit_errors() {
        assert_eq!(
            format_tool_result_content(false, &serde_json::json!("permission denied")),
            "[Tool execution failed]\npermission denied"
        );
        assert_eq!(
            format_tool_result_content(false, &serde_json::json!({"code": "ENOENT"})),
            r#"{"error":{"code":"ENOENT"}}"#
        );
    }
}
