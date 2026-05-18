//! Mode-agnostic types and helpers used by protocol drivers. The
//! concrete mode drivers and their prompts live in
//! the mode crates crates; this module only
//! exposes the shared surface:
//!
//! - [`ModeConfig`], [`ModePreamble`], [`ModeBuildInput`] — the
//!   per-turn configuration driver-plugins populate.
//! - A small helper layer (`normalized_response_parts`, `reasoning_part`,
//!   `append_assistant_text_part`, `turn_limit_exhausted_message`) that
//!   mode drivers reuse for building assistant
//!   messages.

use std::sync::Arc;

use crate::PromptFingerprint;
use crate::llm::types::{LlmOutputPart, LlmResponse, LlmToolSpec, ProviderReasoningReplay};
use crate::sansio::{
    ChatContextProjector, ContextProjector, ModeProtocol, ProtocolDriverHandle, UnitModeProtocol,
};
use crate::session_model::{Message, MessageRole, Part, PartKind, PruneState, shared_parts};
use crate::{ExecutionMode, PromptContribution, ToolSurface};

#[derive(Clone)]
pub struct ModeConfig<M: ModeProtocol = UnitModeProtocol> {
    pub protocol: Arc<dyn ProtocolDriverHandle<M>>,
    pub projector: Arc<dyn ContextProjector<M>>,
    pub sync_execution_surface: bool,
}

impl<M: ModeProtocol> ModeConfig<M> {
    pub fn chat(protocol: Arc<dyn ProtocolDriverHandle<M>>, sync_execution_surface: bool) -> Self {
        Self {
            protocol,
            projector: Arc::new(ChatContextProjector),
            sync_execution_surface,
        }
    }
}

#[derive(Clone)]
pub struct ModePreamble<M: ModeProtocol = UnitModeProtocol> {
    pub config: ModeConfig<M>,
    pub tool_specs: Arc<Vec<LlmToolSpec>>,
    pub tool_names: Arc<Vec<String>>,
    pub tool_names_fingerprint: PromptFingerprint,
    pub omitted_tool_count: usize,
    pub execution_prompt: Arc<str>,
    pub prompt_contributions: Vec<PromptContribution>,
}

#[derive(Clone, Debug)]
pub struct ModeBuildInput {
    pub mode: ExecutionMode,
    pub tool_surface: std::sync::Arc<ToolSurface>,
    pub extra_prompt_contributions: Vec<PromptContribution>,
}

/// Convert a raw `LlmResponse` into a stream of `LlmOutputPart`s that
/// downstream code can iterate. When the response only carries
/// `full_text` (provider didn't populate `parts`), synthesize a single
/// `Text` part.
pub fn normalized_response_parts(llm_response: &LlmResponse) -> Vec<LlmOutputPart> {
    if llm_response.parts.is_empty() && !llm_response.full_text.is_empty() {
        vec![LlmOutputPart::Text {
            text: llm_response.full_text.clone(),
            response_meta: None,
        }]
    } else {
        llm_response.parts.clone()
    }
}

/// Build a Reasoning `Part` from a reasoning item. `meta` is Some when
/// the item carries provider replay metadata; None for display-only
/// summaries.
pub fn reasoning_part(
    asst_id: &str,
    index: usize,
    text: String,
    meta: Option<ProviderReasoningReplay>,
) -> Part {
    Part {
        id: format!("{asst_id}.p{index}"),
        kind: PartKind::Reasoning,
        content: text,
        attachment: None,
        tool_call_id: None,
        tool_name: None,
        tool_replay: None,
        prune_state: PruneState::Intact,
        reasoning_meta: meta,
        response_meta: None,
    }
}

/// Append a streamed text part to the running assistant text, inserting
/// the right number of blank lines so consecutive parts don't glue
/// together.
pub fn append_assistant_text_part(out: &mut String, next: &str) {
    if out.is_empty() {
        out.push_str(next);
        return;
    }

    let prev_trailing_newlines = out.chars().rev().take_while(|ch| *ch == '\n').count();
    let next_leading_newlines = next.chars().take_while(|ch| *ch == '\n').count();
    let total_boundary_newlines = prev_trailing_newlines + next_leading_newlines;
    if total_boundary_newlines < 2 {
        out.push_str(&"\n".repeat(2 - total_boundary_newlines));
    }

    out.push_str(next);
}

/// System-level "turn limit reached" message that both protocol
/// drivers append when their mode_iteration count exceeds `max_turns`.
pub fn turn_limit_exhausted_message(message_id: impl Into<String>, max_turns: usize) -> Message {
    let id = message_id.into();
    Message {
        id: id.clone(),
        role: MessageRole::System,
        parts: shared_parts(vec![Part {
            id: format!("{id}.p0"),
            kind: PartKind::Error,
            content: format!("Turn limit reached ({max_turns}) before a final assistant response."),
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            tool_replay: None,
            prune_state: PruneState::Intact,
            reasoning_meta: None,
            response_meta: None,
        }]),
        origin: None,
    }
}
