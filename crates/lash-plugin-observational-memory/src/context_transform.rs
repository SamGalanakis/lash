use async_trait::async_trait;
use std::borrow::Cow;

use lash_core::PreparedContext;
use lash_core::plugin::{HistoryError, TurnContextTransform, TurnTransformContext};
use lash_core::{Message, MessageOrigin, MessageRole, Part, PartKind};

use crate::ObservationalMemoryConfig;
use crate::constants::{
    OBSERVATION_CONTEXT_INSTRUCTIONS, OBSERVATION_CONTEXT_PROMPT, OBSERVATION_CONTINUATION_HINT,
    OBSERVATIONAL_MEMORY_PLUGIN_ID,
};
use crate::graph_state::{
    active_unobserved_message_nodes, approx_message_nodes_tokens, approx_token_count,
    build_graph_state,
};
use crate::host::OmRuntimeHost;
use crate::model::ActiveMemoryState;
use crate::transitions::maybe_advance_memory_state;

pub(crate) struct ObservationalMemoryTransform {
    config: ObservationalMemoryConfig,
}

impl ObservationalMemoryTransform {
    pub(crate) fn new(config: ObservationalMemoryConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl TurnContextTransform for ObservationalMemoryTransform {
    fn id(&self) -> &'static str {
        "observational_memory.prepare_turn"
    }

    async fn transform(
        &self,
        ctx: &TurnTransformContext,
        input: PreparedContext,
    ) -> Result<PreparedContext, HistoryError> {
        let graph = ctx.state.session_graph();
        let om_state = build_graph_state(graph);
        let pending_message_tokens = approx_message_nodes_tokens(&active_unobserved_message_nodes(
            graph,
            om_state
                .active
                .as_ref()
                .and_then(|state| state.observed_through_message_id.as_deref()),
        ));
        let active_observation_tokens = om_state
            .active
            .as_ref()
            .map(|state| approx_token_count(&state.observations))
            .unwrap_or(0);

        let should_advance_memory = pending_message_tokens
            >= self.config.observation_message_tokens
            || active_observation_tokens >= self.config.reflection_observation_tokens;
        let graph = if should_advance_memory {
            let mut graph = ctx.sessions.snapshot_current().await?.session_graph;
            graph = maybe_advance_memory_state(
                &self.config,
                &OmRuntimeHost::new(
                    &ctx.session_id,
                    &ctx.session_graph,
                    ctx.direct_completions.clone(),
                ),
                ctx.state.policy(),
                graph,
            )
            .await?;
            Cow::Owned(graph)
        } else {
            Cow::Borrowed(graph)
        };

        let Some(active) = build_graph_state(graph.as_ref()).active else {
            return Ok(input);
        };
        if active.observations.trim().is_empty()
            && active.current_task.is_none()
            && active.suggested_response.is_none()
        {
            return Ok(input);
        }

        let input_messages = input.messages.as_slice();
        let prefix_len = input_messages
            .iter()
            .take_while(|message| matches!(message.role, MessageRole::System))
            .count();
        let tail_start = memory_tail_start(
            input_messages,
            prefix_len,
            active.observed_through_message_id.as_deref(),
        );

        let mut messages = Vec::new();
        messages.extend_from_slice(&input_messages[..prefix_len]);
        messages.extend(build_memory_context_messages(&active));
        messages.extend_from_slice(&input_messages[tail_start..]);

        // Wrap as base + fresh render cache so the per-iteration chat
        // projector reuses one rendered prompt across LLM iterations
        // within this turn (OM only runs once per turn). Push-driven
        // additions during the turn land in the delta and re-render
        // each iteration as before.
        let base = std::sync::Arc::new(messages);
        let cache = std::sync::Arc::new(lash_core::BaseRenderCache::new());
        Ok(PreparedContext {
            messages: lash_core::MessageSequence::from_base(base).with_base_render_cache(cache),
            ..input
        })
    }
}

/// First index of the post-prefix "tail" the memory context should re-emit.
///
/// The system prefix (`input_messages[..prefix_len]`) is always re-emitted
/// ahead of the memory block, so the tail must begin at or after `prefix_len`.
/// The marker (`observed_through_message_id`) points at the last message the
/// memory summarized; the tail is everything after it. But if that marker
/// resolves *inside* the prefix, an unclamped `idx + 1 < prefix_len` would make
/// the tail slice re-include prefix messages, duplicating them in the prompt —
/// hence the `.max(prefix_len)` clamp.
fn memory_tail_start(
    input_messages: &[Message],
    prefix_len: usize,
    observed_through_message_id: Option<&str>,
) -> usize {
    observed_through_message_id
        .and_then(|message_id| {
            input_messages
                .iter()
                .position(|message| message.id == message_id)
                .map(|idx| idx + 1)
        })
        .unwrap_or(prefix_len)
        .max(prefix_len)
}

fn build_memory_context_messages(active: &ActiveMemoryState) -> Vec<Message> {
    let mut messages = Vec::new();
    messages.push(plugin_message(
        "om-memory-system",
        MessageRole::System,
        format!("{OBSERVATION_CONTEXT_PROMPT}\n\n{OBSERVATION_CONTEXT_INSTRUCTIONS}"),
    ));

    let mut memory_block = String::from("<observations>\n");
    memory_block.push_str(active.observations.trim());
    memory_block.push_str("\n</observations>");
    if let Some(current_task) = &active.current_task {
        memory_block.push_str(&format!(
            "\n\n<current-task>\n{}\n</current-task>",
            current_task.trim()
        ));
    }
    if let Some(suggested_response) = &active.suggested_response {
        memory_block.push_str(&format!(
            "\n\n<suggested-response>\n{}\n</suggested-response>",
            suggested_response.trim()
        ));
    }
    messages.push(plugin_message(
        "om-memory-block",
        MessageRole::System,
        memory_block,
    ));
    messages.push(plugin_message(
        "om-memory-reminder",
        MessageRole::User,
        format!("<system-reminder>{OBSERVATION_CONTINUATION_HINT}</system-reminder>"),
    ));
    messages
}

fn plugin_message(id: &str, role: MessageRole, content: String) -> Message {
    Message {
        id: id.to_string(),
        role,
        parts: lash_core::shared_parts(vec![Part {
            id: format!("{id}.p0"),
            kind: PartKind::Prose,
            content,
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            tool_replay: None,
            prune_state: lash_core::PruneState::Intact,
            reasoning_meta: None,
            response_meta: None,
        }]),
        origin: Some(MessageOrigin::Plugin {
            plugin_id: OBSERVATIONAL_MEMORY_PLUGIN_ID.to_string(),
            transient: true,
        }),
    }
}

#[cfg(test)]
mod tail_start_tests {
    use super::*;

    fn msg(id: &str, role: MessageRole) -> Message {
        plugin_message(id, role, "x".to_string())
    }

    #[test]
    fn marker_after_prefix_starts_tail_right_after_marker() {
        let messages = [
            msg("s0", MessageRole::System),
            msg("u1", MessageRole::User),
            msg("a2", MessageRole::Assistant),
            msg("u3", MessageRole::User),
        ];
        // prefix_len = 1 (one leading system message); marker is u1 (idx 1).
        assert_eq!(memory_tail_start(&messages, 1, Some("u1")), 2);
    }

    #[test]
    fn marker_inside_prefix_does_not_re_include_prefix() {
        // Two leading system messages; marker resolves to the first one (idx 0),
        // so the naive tail_start would be 1 < prefix_len(2) and the tail slice
        // would re-emit the second system message already in the prefix. Clamp
        // keeps the tail at prefix_len.
        let messages = [
            msg("s0", MessageRole::System),
            msg("s1", MessageRole::System),
            msg("u2", MessageRole::User),
        ];
        assert_eq!(memory_tail_start(&messages, 2, Some("s0")), 2);
    }

    #[test]
    fn missing_marker_falls_back_to_prefix_len() {
        let messages = [msg("s0", MessageRole::System), msg("u1", MessageRole::User)];
        assert_eq!(memory_tail_start(&messages, 1, Some("nonexistent")), 1);
        assert_eq!(memory_tail_start(&messages, 1, None), 1);
    }
}
