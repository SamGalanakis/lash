//! Turn stream assembly and output classification.
//!
//! Extracted from `runtime/mod.rs`. These types are crate-internal — sibling
//! modules (`turn_driver.rs`, `tests.rs`) import them via `use super::*` in
//! `mod.rs`. No public API is exposed here.

use std::time::Instant;

use serde_json::json;

use crate::ToolCallRecord;
use crate::llm::types::{LlmOutputPart, LlmResponse, LlmUsage};
use crate::session_model::{MessageRole, PartKind, SessionEvent, TokenUsage};
use crate::{TurnFinish, TurnOutcome, TurnStop};

use super::state::SessionStateEnvelope;
use super::{
    AssembledTurn, AssistantOutput, ExecutionSummary, OutputState, SanitizerPolicy,
    TerminationPolicy, TurnActivity, TurnActivityId, TurnEvent, TurnIssue,
};

#[derive(Clone, Debug, Default)]
pub(super) struct StandardStreamAccumulator {
    pub(super) parts: Vec<LlmOutputPart>,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct LlmStreamDebugState {
    pub(super) started_at: Instant,
    pub(super) sequence: u64,
    pub(super) summary: LlmStreamSummary,
}

#[derive(Clone, Copy)]
pub(super) struct LlmDebugText<'a> {
    pub(super) raw: Option<&'a str>,
    pub(super) visible: Option<&'a str>,
}

#[derive(Clone, Copy)]
pub(super) struct LlmDebugToolCall<'a> {
    pub(super) call_id: &'a str,
    pub(super) tool_name: &'a str,
    pub(super) input_json: &'a str,
}

#[derive(Clone, Copy)]
pub(super) struct LlmStreamEventLog<'a> {
    pub(super) mode_iteration: usize,
    pub(super) event_type: &'a str,
    pub(super) text: LlmDebugText<'a>,
    pub(super) item_id: Option<&'a str>,
    pub(super) usage: Option<&'a LlmUsage>,
    pub(super) tool_call: Option<LlmDebugToolCall<'a>>,
}

pub(super) struct StandardStreamState<'a> {
    pub(super) text_streamed: &'a mut bool,
    pub(super) streamed_usage: &'a mut LlmUsage,
    pub(super) stream_accumulator: &'a mut StandardStreamAccumulator,
    pub(super) debug: &'a mut LlmStreamDebugState,
    pub(super) mode_iteration: usize,
    pub(super) assistant_prose_correlation: &'a mut Option<crate::TurnActivityId>,
    pub(super) reasoning_correlation: &'a mut Option<crate::TurnActivityId>,
    /// Set to `true` by `forward_standard_stream_event` when a plugin
    /// stream hook has raised `AssistantStreamTransform.abort_stream`.
    /// The LLM runner checks this after each stream event and
    /// short-circuits the select loop, synthesizing a response from the
    /// already-streamed parts.
    pub(super) abort_requested: &'a mut bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct LlmStreamSummary {
    pub(super) first_visible_token_latency_ms: Option<u64>,
    pub(super) last_visible_chunk_latency_ms: Option<u64>,
    pub(super) text_delta_count: u64,
    pub(super) visible_chunk_count: u64,
    pub(super) total_visible_chars: u64,
    pub(super) max_visible_chunk_chars: u64,
}

impl LlmStreamDebugState {
    pub(super) fn new() -> Self {
        Self {
            started_at: Instant::now(),
            sequence: 0,
            summary: LlmStreamSummary::default(),
        }
    }

    pub(super) fn next_sequence(&mut self) -> u64 {
        let sequence = self.sequence;
        self.sequence += 1;
        sequence
    }

    pub(super) fn elapsed_ms(&self) -> u64 {
        self.started_at.elapsed().as_millis() as u64
    }
}

impl LlmStreamSummary {
    pub(super) fn record_text_chunk(&mut self, visible_text: Option<&str>, elapsed_ms: u64) {
        self.text_delta_count += 1;

        let visible_chars = visible_text
            .map(|text| text.chars().count() as u64)
            .unwrap_or(0);
        if visible_chars == 0 {
            return;
        }

        if self.first_visible_token_latency_ms.is_none() {
            self.first_visible_token_latency_ms = Some(elapsed_ms);
        }
        self.last_visible_chunk_latency_ms = Some(elapsed_ms);
        self.visible_chunk_count += 1;
        self.total_visible_chars += visible_chars;
        self.max_visible_chunk_chars = self.max_visible_chunk_chars.max(visible_chars);
    }

    pub(super) fn to_json(self) -> serde_json::Value {
        let avg_visible_chunk_chars = if self.visible_chunk_count == 0 {
            None
        } else {
            Some(self.total_visible_chars as f64 / self.visible_chunk_count as f64)
        };
        let stream_duration_ms = match (
            self.first_visible_token_latency_ms,
            self.last_visible_chunk_latency_ms,
        ) {
            (Some(first), Some(last)) => Some(last.saturating_sub(first)),
            _ => None,
        };
        json!({
            "first_visible_token_latency_ms": self.first_visible_token_latency_ms,
            "stream_duration_ms": stream_duration_ms,
            "text_delta_count": self.text_delta_count,
            "visible_chunk_count": self.visible_chunk_count,
            "avg_visible_chunk_chars": avg_visible_chunk_chars,
            "max_visible_chunk_chars": if self.visible_chunk_count == 0 {
                serde_json::Value::Null
            } else {
                serde_json::Value::from(self.max_visible_chunk_chars)
            },
        })
    }
}

impl StandardStreamAccumulator {
    pub(super) fn push_text(&mut self, piece: &str) {
        if piece.is_empty() {
            return;
        }
        match self.parts.last_mut() {
            Some(LlmOutputPart::Text { text, .. }) => append_stream_piece(text, piece),
            _ => self.parts.push(LlmOutputPart::Text {
                text: piece.to_string(),
                response_meta: None,
            }),
        }
    }

    pub(super) fn push_tool_call(
        &mut self,
        call_id: String,
        tool_name: String,
        input_json: String,
        item_id: Option<String>,
        signature: Option<String>,
    ) {
        self.parts.push(LlmOutputPart::ToolCall {
            call_id,
            tool_name,
            input_json,
            item_id,
            signature,
        });
    }

    pub(super) fn push_reasoning(
        &mut self,
        text: String,
        item_id: Option<String>,
        summary: Vec<String>,
        encrypted_content: Option<String>,
    ) {
        let has_roundtrip_payload = encrypted_content.is_some();
        if let Some(LlmOutputPart::Reasoning {
            text: existing,
            signature: None,
            redacted: false,
            item_id: existing_item_id,
            encrypted_content: existing_encrypted_content,
            summary: existing_summary,
        }) = self.parts.last_mut()
            && existing_item_id.is_none()
            && existing_encrypted_content.is_none()
            && existing_summary.is_empty()
            && item_id.is_none()
            && encrypted_content.is_none()
            && summary.is_empty()
        {
            append_stream_piece(existing, &text);
            return;
        }
        if let Some(LlmOutputPart::Reasoning {
            text: existing,
            signature: existing_signature,
            redacted: existing_redacted,
            item_id: existing_item_id,
            encrypted_content: existing_encrypted_content,
            summary: existing_summary,
        }) = self.parts.last_mut()
            && has_roundtrip_payload
            && existing_encrypted_content.is_none()
            && !existing.trim().is_empty()
            && (text.trim().is_empty() || text.contains(existing.as_str()))
        {
            if !text.trim().is_empty() && text != *existing {
                *existing = text;
            }
            *existing_signature = None;
            *existing_redacted = false;
            *existing_item_id = item_id;
            *existing_encrypted_content = encrypted_content;
            *existing_summary = summary;
            return;
        }
        if let Some(LlmOutputPart::Reasoning {
            text: existing,
            signature: existing_signature,
            redacted: existing_redacted,
            item_id: existing_item_id,
            encrypted_content: existing_encrypted_content,
            summary: existing_summary,
        }) = self.parts.last_mut()
            && !has_roundtrip_payload
            && existing_encrypted_content.is_some()
            && !text.trim().is_empty()
            && existing.trim().is_empty()
        {
            append_stream_piece(existing, &text);
            *existing_signature = None;
            *existing_redacted = false;
            if existing_item_id.is_none() {
                *existing_item_id = item_id;
            }
            if existing_summary.is_empty() {
                *existing_summary = summary;
            }
            return;
        }
        self.parts.push(LlmOutputPart::Reasoning {
            text,
            signature: None,
            redacted: false,
            item_id,
            encrypted_content,
            summary,
        });
    }

    pub(super) fn is_empty(&self) -> bool {
        !self.parts.iter().any(|part| match part {
            LlmOutputPart::Text { text, .. } => !text.is_empty(),
            LlmOutputPart::Reasoning { .. } => true,
            LlmOutputPart::ToolCall { .. } => true,
        })
    }

    pub(super) fn full_text(&self) -> String {
        let mut full_text = String::new();
        for part in &self.parts {
            if let LlmOutputPart::Text { text, .. } = part {
                full_text.push_str(text);
            }
        }
        full_text
    }

    pub(super) fn apply_to_response(&self, response: &mut LlmResponse) {
        if self.is_empty() {
            return;
        }
        if response.parts.is_empty() {
            response.parts = self.parts.clone();
            if response.full_text.is_empty() {
                response.full_text = self.full_text();
            }
            return;
        }

        if response_contains_accumulated_parts(response, &self.parts) {
            return;
        }

        response.parts = reconcile_accumulated_parts(&self.parts, &response.parts);
        if response.full_text.is_empty() {
            response.full_text = response_parts_text(&response.parts);
        }
    }
}

fn response_contains_accumulated_parts(
    response: &LlmResponse,
    accumulated_parts: &[LlmOutputPart],
) -> bool {
    accumulated_parts
        .iter()
        .filter(|part| part_has_visible_or_tool_content(part))
        .all(|part| response_contains_part(response, part))
}

fn response_contains_part(response: &LlmResponse, part: &LlmOutputPart) -> bool {
    match part {
        LlmOutputPart::Text { text, .. } => {
            text.trim().is_empty()
                || response.full_text.contains(text)
                || response.parts.iter().any(|candidate| {
                    matches!(candidate, LlmOutputPart::Text { text: candidate, .. } if candidate.contains(text))
                })
        }
        LlmOutputPart::Reasoning { text, item_id, .. } => {
            text.trim().is_empty()
                || response.parts.iter().any(|candidate| match candidate {
                    LlmOutputPart::Reasoning {
                        text: candidate,
                        item_id: candidate_id,
                        ..
                    } => {
                        candidate.contains(text)
                            || (item_id.is_some()
                                && candidate_id.is_some()
                                && item_id == candidate_id
                                && !candidate.trim().is_empty())
                    }
                    _ => false,
                })
        }
        LlmOutputPart::ToolCall {
            call_id, item_id, ..
        } => response.parts.iter().any(|candidate| match candidate {
            LlmOutputPart::ToolCall {
                call_id: candidate_call_id,
                item_id: candidate_item_id,
                ..
            } => {
                candidate_call_id == call_id
                    || (item_id.is_some() && candidate_item_id.is_some() && item_id == candidate_item_id)
            }
            _ => false,
        }),
    }
}

fn reconcile_accumulated_parts(
    accumulated_parts: &[LlmOutputPart],
    final_parts: &[LlmOutputPart],
) -> Vec<LlmOutputPart> {
    let mut out = accumulated_parts.to_vec();
    for final_part in final_parts {
        match final_part {
            LlmOutputPart::ToolCall { .. } => {
                if let Some(existing) = out
                    .iter_mut()
                    .find(|candidate| tool_calls_match(candidate, final_part))
                {
                    *existing = final_part.clone();
                } else {
                    out.push(final_part.clone());
                }
            }
            LlmOutputPart::Reasoning { .. } => {
                if !out.iter().any(|candidate| reasoning_matches(candidate, final_part)) {
                    out.push(final_part.clone());
                }
            }
            LlmOutputPart::Text { text, .. } => {
                if !text.trim().is_empty()
                    && !out
                        .iter()
                        .any(|candidate| matches!(candidate, LlmOutputPart::Text { text: candidate, .. } if candidate.contains(text)))
                {
                    out.push(final_part.clone());
                }
            }
        }
    }
    out
}

fn part_has_visible_or_tool_content(part: &LlmOutputPart) -> bool {
    match part {
        LlmOutputPart::Text { text, .. } => !text.trim().is_empty(),
        LlmOutputPart::Reasoning {
            text,
            encrypted_content,
            signature,
            ..
        } => !text.trim().is_empty() || encrypted_content.is_some() || signature.is_some(),
        LlmOutputPart::ToolCall { .. } => true,
    }
}

fn tool_calls_match(candidate: &LlmOutputPart, expected: &LlmOutputPart) -> bool {
    match (candidate, expected) {
        (
            LlmOutputPart::ToolCall {
                call_id, item_id, ..
            },
            LlmOutputPart::ToolCall {
                call_id: expected_call_id,
                item_id: expected_item_id,
                ..
            },
        ) => {
            call_id == expected_call_id
                || (item_id.is_some() && expected_item_id.is_some() && item_id == expected_item_id)
        }
        _ => false,
    }
}

fn reasoning_matches(candidate: &LlmOutputPart, expected: &LlmOutputPart) -> bool {
    match (candidate, expected) {
        (
            LlmOutputPart::Reasoning {
                text,
                item_id,
                encrypted_content,
                ..
            },
            LlmOutputPart::Reasoning {
                text: expected_text,
                item_id: expected_item_id,
                encrypted_content: expected_encrypted_content,
                ..
            },
        ) => {
            (!text.trim().is_empty()
                && !expected_text.trim().is_empty()
                && (text.contains(expected_text) || expected_text.contains(text)))
                || (item_id.is_some() && expected_item_id.is_some() && item_id == expected_item_id)
                || (encrypted_content.is_some()
                    && expected_encrypted_content.is_some()
                    && encrypted_content == expected_encrypted_content)
        }
        _ => false,
    }
}

fn response_parts_text(parts: &[LlmOutputPart]) -> String {
    let mut full_text = String::new();
    for part in parts {
        if let LlmOutputPart::Text { text, .. } = part {
            full_text.push_str(text);
        }
    }
    full_text
}

fn append_stream_piece(full: &mut String, piece: &str) {
    if piece.is_empty() {
        return;
    }
    if piece.starts_with(full.as_str()) {
        full.push_str(&piece[full.len()..]);
    } else {
        full.push_str(piece);
    }
}

pub(super) struct TurnAssembler {
    pub(super) final_message: Option<String>,
    pub(super) current_assistant_prose_id: Option<TurnActivityId>,
    pub(super) current_assistant_prose: Option<String>,
    pub(super) text_deltas: String,
    pub(super) capture_text_deltas: bool,
    pub(super) tool_calls: Vec<ToolCallRecord>,
    pub(super) token_usage: TokenUsage,
    pub(super) last_llm_usage: Option<TokenUsage>,
    pub(super) issues: Vec<TurnIssue>,
    pub(super) saw_done: bool,
    pub(super) saw_tool_failure: bool,
    pub(super) outcome: Option<TurnOutcome>,
}

impl Default for TurnAssembler {
    fn default() -> Self {
        Self::new(true)
    }
}

impl TurnAssembler {
    pub(super) fn new(capture_text_deltas: bool) -> Self {
        Self {
            final_message: None,
            current_assistant_prose_id: None,
            current_assistant_prose: None,
            text_deltas: String::new(),
            capture_text_deltas,
            tool_calls: Vec::new(),
            token_usage: TokenUsage::default(),
            last_llm_usage: None,
            issues: Vec::new(),
            saw_done: false,
            saw_tool_failure: false,
            outcome: None,
        }
    }

    pub(super) fn push_turn_activity(&mut self, activity: &TurnActivity) {
        let TurnEvent::AssistantProseDelta { text } = &activity.event else {
            return;
        };
        if self.current_assistant_prose_id.as_ref() == Some(&activity.correlation_id) {
            if let Some(current) = self.current_assistant_prose.as_mut() {
                current.push_str(text);
            }
            return;
        }
        self.current_assistant_prose_id = Some(activity.correlation_id.clone());
        self.current_assistant_prose = Some(text.clone());
    }

    pub(super) fn push(&mut self, event: &SessionEvent) {
        match event {
            SessionEvent::TextDelta { content } if self.capture_text_deltas => {
                self.text_deltas.push_str(content);
            }
            SessionEvent::ToolCall {
                call_id,
                name,
                args,
                result,
                success,
                duration_ms,
            } => {
                self.tool_calls.push(ToolCallRecord {
                    call_id: call_id.clone(),
                    tool: name.clone(),
                    args: args.clone(),
                    result: result.clone(),
                    success: *success,
                    duration_ms: *duration_ms,
                    control: None,
                });
                if !success {
                    self.saw_tool_failure = true;
                }
            }
            SessionEvent::Message { text, kind } if kind == "final" => {
                self.final_message = Some(text.clone());
            }
            SessionEvent::TokenUsage {
                usage, cumulative, ..
            } => {
                self.token_usage = cumulative.clone();
                self.last_llm_usage = Some(usage.clone());
            }
            SessionEvent::Error { message, envelope } => {
                let (kind, code, raw) = if let Some(envelope) = envelope {
                    (
                        envelope.kind.clone(),
                        envelope.code.clone(),
                        envelope.raw.clone(),
                    )
                } else {
                    ("runtime".to_string(), None, None)
                };
                self.issues.push(TurnIssue {
                    kind,
                    code,
                    message: message.clone(),
                    raw,
                });
            }
            SessionEvent::Done => {
                self.saw_done = true;
            }
            SessionEvent::TurnOutcome { outcome } => {
                self.outcome = Some(outcome.clone());
            }
            _ => {}
        }
    }

    pub(super) fn finish(
        mut self,
        state: SessionStateEnvelope,
        interrupted: bool,
        force_runtime_error: Option<TurnIssue>,
        sanitizer: &SanitizerPolicy,
        termination: &TerminationPolicy,
    ) -> AssembledTurn {
        let mut issues = self.issues;
        if let Some(issue) = force_runtime_error {
            issues.push(issue);
        }
        let read_model = state.read_model();
        let max_turn_reached = read_model.messages.iter().rev().take(8).any(|msg| {
            msg.role == MessageRole::System
                && msg
                    .parts
                    .iter()
                    .any(|part| part.content.contains("Turn limit reached ("))
        });

        let raw_output = if let Some(final_message) = self.final_message {
            final_message
        } else if let Some(assistant_prose) = self.current_assistant_prose {
            assistant_prose
        } else {
            let streamed = self.text_deltas.trim().to_string();
            let state_output = fallback_assistant_output_from_state(&state);
            if streamed.is_empty()
                || (!state_output.is_empty()
                    && state_output.len() >= streamed.len()
                    && state_output.starts_with(&streamed))
            {
                state_output
            } else {
                streamed
            }
        };
        let safe_output = sanitize_assistant_output(raw_output.clone(), sanitizer);
        let output_state = classify_output_state(&raw_output, &safe_output, &issues);

        let outcome = if interrupted {
            TurnOutcome::Stopped(TurnStop::Cancelled)
        } else if let Some(outcome) = self.outcome.take() {
            match outcome {
                TurnOutcome::Finished(TurnFinish::AssistantMessage { .. }) => {
                    TurnOutcome::Finished(TurnFinish::AssistantMessage {
                        text: safe_output.clone(),
                    })
                }
                outcome => outcome,
            }
        } else if !self.saw_done && termination.treat_missing_done_as_failure {
            TurnOutcome::Stopped(TurnStop::RuntimeError)
        } else if !issues.is_empty() {
            if self.saw_tool_failure {
                TurnOutcome::Stopped(TurnStop::ToolFailure)
            } else {
                TurnOutcome::Stopped(TurnStop::RuntimeError)
            }
        } else if max_turn_reached {
            TurnOutcome::Stopped(TurnStop::MaxTurns)
        } else {
            TurnOutcome::Finished(TurnFinish::AssistantMessage {
                text: safe_output.clone(),
            })
        };

        AssembledTurn {
            execution: ExecutionSummary {
                mode: state.policy.execution_mode.clone(),
                had_tool_calls: !self.tool_calls.is_empty(),
                had_code_execution: false,
            },
            state,
            outcome,
            assistant_output: AssistantOutput {
                safe_text: safe_output,
                raw_text: raw_output,
                state: output_state,
            },
            token_usage: self.token_usage,
            tool_calls: self.tool_calls,
            errors: issues,
        }
    }

    pub(super) fn last_llm_usage(&self) -> Option<&TokenUsage> {
        self.last_llm_usage.as_ref()
    }
}

pub(super) fn fallback_assistant_output_from_state(state: &SessionStateEnvelope) -> String {
    let read_model = state.read_model();
    let messages = read_model.messages.as_slice();
    let latest_user_message_idx = messages
        .iter()
        .rposition(|message| matches!(message.role, MessageRole::User));
    let search_messages = latest_user_message_idx
        .map(|idx| &messages[idx.saturating_add(1)..])
        .unwrap_or(messages);
    search_messages
        .iter()
        .rev()
        .find(|message| message.role == MessageRole::Assistant)
        .map(|message| {
            message
                .parts
                .iter()
                .filter(|part| {
                    matches!(
                        part.kind,
                        PartKind::Text | PartKind::Prose | PartKind::Image
                    )
                })
                .map(|part| part.content.as_str())
                .collect::<String>()
        })
        .unwrap_or_default()
}

pub(super) fn sanitize_assistant_output(text: String, _policy: &SanitizerPolicy) -> String {
    text.lines()
        .map(str::trim_end)
        .collect::<Vec<_>>()
        .join(
            "
",
        )
        .trim()
        .to_string()
}

pub(super) fn classify_output_state(
    raw_text: &str,
    safe_text: &str,
    issues: &[TurnIssue],
) -> OutputState {
    if safe_text.is_empty() && raw_text.is_empty() {
        return OutputState::EmptyOutput;
    }
    if safe_text.is_empty() && contains_traceback_only(raw_text) {
        return OutputState::TracebackOnly;
    }
    if !issues.is_empty() && !safe_text.is_empty() {
        return OutputState::RecoveredFromError;
    }
    OutputState::Usable
}

fn contains_traceback_only(raw_text: &str) -> bool {
    if raw_text.is_empty() {
        return false;
    }
    let has_traceback = raw_text.contains("Traceback (most recent call last)")
        || raw_text.lines().any(|line| {
            let trimmed = line.trim();
            trimmed.starts_with("Runtime error:")
                || trimmed.starts_with("NameError:")
                || trimmed.starts_with("TypeError:")
                || trimmed.starts_with("ValueError:")
                || trimmed.starts_with("KeyError:")
                || trimmed.starts_with("AttributeError:")
                || trimmed.starts_with("SyntaxError:")
                || trimmed.starts_with("ImportError:")
                || trimmed.starts_with("ModuleNotFoundError:")
        });
    if !has_traceback {
        return false;
    }
    // If no alphabetic prose besides traceback/exception formatting, treat as traceback-only.
    !raw_text.lines().any(|line| {
        let trimmed = line.trim();
        if trimmed.is_empty()
            || trimmed.starts_with("Traceback")
            || trimmed.starts_with("File ")
            || trimmed.starts_with("Runtime error:")
        {
            return false;
        }
        !trimmed.contains(':')
    })
}
