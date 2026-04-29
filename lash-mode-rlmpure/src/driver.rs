use std::sync::{Arc, Mutex};

use lash::llm::types::{LlmMessage, LlmRole, LlmToolChoice};
use lash::sansio::ContextProjector;
use lash::{
    LlmRequest, ModeBuildInput, ModeConfig, ModePreamble, ProjectorContext, PromptContribution,
    head_tail_truncate,
    session_model::{ConversationRecord, Message, MessageRole, PartKind, SessionEventRecord},
};
use lash_rlm_types::{RlmModeEvent, RlmTrajectoryEntry};

pub fn rlmpure_execution_section() -> String {
    lash_mode_rlm::rlm_execution_section()
}

#[derive(Clone, Debug)]
pub struct RlmpureProjectorConfig {
    pub max_output_chars: usize,
}

impl Default for RlmpureProjectorConfig {
    fn default() -> Self {
        Self {
            max_output_chars: 10_000,
        }
    }
}

pub fn build_rlmpure_preamble(
    input: ModeBuildInput,
    config: RlmpureProjectorConfig,
) -> ModePreamble {
    let omitted_tool_count = input.tool_surface.omitted_tool_count();
    let mut prompt_contributions = Vec::new();

    let tool_docs = input.tool_surface.prompt_tool_docs();
    if !tool_docs.trim().is_empty() {
        prompt_contributions.push(PromptContribution::execution("Available Tools", tool_docs));
    }
    if omitted_tool_count > 0 {
        prompt_contributions.push(PromptContribution::guidance(
            "Tool Discovery",
            "Use `discover_tools` to inspect additional discoverable tools omitted from Available Tools. If a result is marked loadable but not callable, call `load_tools(names=[...])`; the runtime refreshes the surface for the next REPL step.",
        ));
    }
    prompt_contributions.extend(input.extra_prompt_contributions);

    ModePreamble {
        config: ModeConfig {
            protocol: Arc::new(lash_mode_rlm::RlmDriver),
            projector: Arc::new(RlmpureContextProjector {
                max_output_chars: config.max_output_chars,
                cache: Mutex::new(None),
            }),
            sync_execution_surface: true,
        },
        tool_specs: Arc::new(Vec::new()),
        tool_names: input.tool_surface.tool_names(),
        omitted_tool_count,
        execution_prompt: rlmpure_execution_section(),
        prompt_contributions,
    }
}

struct RlmpureContextProjector {
    max_output_chars: usize,
    /// Memoizes the rendered REPL-history string across LLM iterations
    /// within a single turn. Events are append-only within a turn (only
    /// `apply_actions` mutates `TurnMachine.events`), so we can extend
    /// the cached prefix instead of re-walking every event each
    /// iteration. The projector is rebuilt per turn via
    /// `build_rlmpure_preamble`, so the cache scope matches turn scope
    /// without explicit invalidation.
    cache: Mutex<Option<TrajectoryCache>>,
}

#[derive(Default)]
struct TrajectoryCache {
    rendered: String,
    processed_events: usize,
    step_index: usize,
}

impl ContextProjector<lash::HostModeProtocol> for RlmpureContextProjector {
    fn project(&self, ctx: ProjectorContext<'_>) -> LlmRequest {
        let task_context = self.format_task_context(ctx.events, ctx.messages.as_slice());
        let repl_history = self.format_repl_history(ctx.events);
        let user_prompt = format!(
            "Task\n{task_context}\n\nREPL history\n{repl_history}\n\nIteration\n{}\n\nFinalization\nCall `submit <value>` from lashlang when the task is complete. Do not answer in prose without a lashlang block.",
            ctx.iteration + 1
        );

        let mut messages = Vec::new();
        if !ctx.config.system_prompt.trim().is_empty() {
            messages.push(LlmMessage::text(
                LlmRole::System,
                std::sync::Arc::clone(&ctx.config.system_prompt),
            ));
        }
        messages.push(LlmMessage::text(LlmRole::User, user_prompt));

        LlmRequest {
            model: ctx.config.model.clone(),
            messages,
            attachments: Vec::new(),
            tools: Arc::new(Vec::new()),
            tool_choice: LlmToolChoice::None,
            model_variant: ctx.config.model_variant.clone(),
            session_id: ctx.config.run_session_id.clone(),
            output_spec: None,
            stream_events: None,
        }
    }
}

impl RlmpureContextProjector {
    fn format_task_context(&self, events: &[SessionEventRecord], messages: &[Message]) -> String {
        let mut rendered = String::new();
        let mut user_message_index = 0usize;
        for event in events {
            let SessionEventRecord::Conversation(record) = event else {
                continue;
            };
            if record.role != MessageRole::User
                || matches!(record.origin, Some(lash::MessageOrigin::Plugin { .. }))
            {
                continue;
            }
            let Some(text) = conversation_text(record) else {
                continue;
            };
            user_message_index += 1;
            if !rendered.is_empty() {
                rendered.push_str("\n\n");
            }
            append_user_message(
                &mut rendered,
                user_message_index,
                &text,
                self.max_output_chars,
            );
        }
        if rendered.is_empty() {
            lash_mode_rlm::build_task_context(messages)
        } else {
            rendered
        }
    }

    fn format_repl_history(&self, events: &[SessionEventRecord]) -> String {
        let mut guard = self.cache.lock().expect("rlmpure trajectory cache lock");
        let cache = guard.get_or_insert_with(TrajectoryCache::default);
        // Events grow append-only within a turn. If the slice shrunk
        // (shouldn't happen during a turn, but be safe), reset and
        // rebuild from scratch.
        if events.len() < cache.processed_events {
            *cache = TrajectoryCache::default();
        }
        for event in &events[cache.processed_events..] {
            if let SessionEventRecord::Mode(event) = event
                && let Some(RlmModeEvent::RlmTrajectoryEntry(entry)) = event.rlm_event()
            {
                cache.step_index += 1;
                if !cache.rendered.is_empty() {
                    cache.rendered.push_str("\n\n");
                }
                append_repl_step(
                    &mut cache.rendered,
                    cache.step_index,
                    &entry,
                    self.max_output_chars,
                );
            }
        }
        cache.processed_events = events.len();
        if cache.rendered.is_empty() {
            "You have not interacted with the lashlang REPL yet.".to_string()
        } else {
            cache.rendered.clone()
        }
    }
}

fn conversation_text(record: &ConversationRecord) -> Option<String> {
    let chunks = record
        .parts
        .iter()
        .filter(|part| matches!(part.kind, PartKind::Text | PartKind::Prose))
        .map(|part| part.content.trim())
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    (!chunks.is_empty()).then(|| chunks.join("\n\n"))
}

fn append_user_message(out: &mut String, index: usize, text: &str, max_output_chars: usize) {
    use std::fmt::Write as _;
    let (preview, raw_len) = head_tail_truncate(text, max_output_chars);
    let availability = if raw_len > max_output_chars {
        format!(", available as `user_input_{index}`")
    } else {
        String::new()
    };
    let _ = write!(
        out,
        "=== Message {index} ===\nUser ({raw_len} chars{availability}):\n{preview}"
    );
}

fn append_repl_step(
    out: &mut String,
    index: usize,
    entry: &RlmTrajectoryEntry,
    max_output_chars: usize,
) {
    let (preview, raw_len) = head_tail_truncate(&entry.output, max_output_chars);
    let reasoning = reasoning_without_first_fence(&entry.reasoning)
        .trim()
        .to_string();
    use std::fmt::Write as _;
    let _ = write!(
        out,
        "=== Step {index} ===\nReasoning:\n{}\n\nCode:\n```lashlang\n{}\n```\n\nObservation ({raw_len} chars):\n{}",
        if reasoning.is_empty() {
            "(none)"
        } else {
            &reasoning
        },
        entry.code.trim(),
        preview
    );
    if let Some(error) = &entry.error {
        out.push_str("\n\nError:\n");
        out.push_str(error);
    }
    if let Some(final_output) = &entry.final_output {
        out.push_str("\n\nFinal output:\n");
        out.push_str(
            &serde_json::to_string_pretty(final_output)
                .unwrap_or_else(|_| final_output.to_string()),
        );
    }
}

fn reasoning_without_first_fence(text: &str) -> String {
    let Some(open_rel) = text.find("```") else {
        return text.to_string();
    };
    let after_open = open_rel + 3;
    let rest = &text[after_open..];
    let Some(lang_end_rel) = rest.find('\n') else {
        return text[..open_rel].to_string();
    };
    let lang = rest[..lang_end_rel].trim();
    if !matches!(lang, "lashlang" | "rlm" | "lash") {
        return text.to_string();
    }
    let body_start = after_open + lang_end_rel + 1;
    let close = text[body_start..]
        .find("```")
        .map(|rel| body_start + rel)
        .unwrap_or(text.len());
    let after_close = (close + 3).min(text.len());
    let mut out = String::new();
    out.push_str(text[..open_rel].trim_end());
    let tail = text[after_close..].trim_start();
    if !tail.is_empty() {
        if !out.is_empty() {
            out.push_str("\n\n");
        }
        out.push_str(tail);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use lash::session_model::{ConversationRecord, Part, PruneState};

    fn user_event(id: &str, text: &str) -> SessionEventRecord {
        SessionEventRecord::Conversation(ConversationRecord {
            id: id.to_string(),
            role: MessageRole::User,
            parts: vec![Part {
                id: format!("{id}.p0"),
                kind: PartKind::Text,
                content: text.to_string(),
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                tool_item_id: None,
                tool_signature: None,
                prune_state: PruneState::Intact,
                reasoning_meta: None,
            }]
            .into(),
            user_input: None,
            origin: None,
        })
    }

    fn step_event(iteration: usize, code: &str, output: &str) -> SessionEventRecord {
        SessionEventRecord::Mode(lash::ModeEvent::rlm(RlmModeEvent::RlmTrajectoryEntry(
            RlmTrajectoryEntry {
                id: format!("rlm_step_{iteration}"),
                iteration,
                reasoning: "thinking".to_string(),
                code: code.to_string(),
                output: output.to_string(),
                observations: Vec::new(),
                tool_calls: Vec::new(),
                error: None,
                final_output: None,
                output_raw_len: output.chars().count(),
            },
        )))
    }

    fn projector(max_output_chars: usize) -> RlmpureContextProjector {
        RlmpureContextProjector {
            max_output_chars,
            cache: Mutex::new(None),
        }
    }

    #[test]
    fn task_context_renders_user_messages_separately_from_repl_history() {
        let projector = projector(100);
        let events = [
            user_event("u1", "first"),
            step_event(0, "print 1", "1"),
            user_event("u2", "second"),
            step_event(1, "print 2", "2"),
        ];
        let task_context = projector.format_task_context(&events, &[]);
        let history = projector.format_repl_history(&events);

        assert!(task_context.contains("=== Message 1 ===\nUser (5 chars):\nfirst"));
        assert!(task_context.contains("=== Message 2 ===\nUser (6 chars):\nsecond"));
        assert!(history.contains("=== Step 1 ==="));
        assert!(history.contains("=== Step 2 ==="));
        assert!(!history.contains("=== Message 1 ==="));
        assert!(!history.contains("=== Message 2 ==="));
        assert!(!history.contains("Inputs"));
        assert!(!history.contains("omitted]"));
    }

    #[test]
    fn long_user_message_gets_binding_hint() {
        let projector = projector(10);
        let task_context =
            projector.format_task_context(&[user_event("u1", "abcdefghijklmnopqrstuvwxyz")], &[]);

        assert!(task_context.contains("available as `user_input_1`"));
        assert!(task_context.contains("... (16 characters omitted) ..."));
    }

    #[test]
    fn incremental_render_extends_cached_prefix_on_subsequent_calls() {
        let projector = projector(100);
        let initial = projector
            .format_repl_history(&[user_event("u1", "first"), step_event(0, "print 1", "1")]);
        assert!(!initial.contains("=== Message 1 ==="));
        assert!(initial.contains("=== Step 1 ==="));

        let extended = projector.format_repl_history(&[
            user_event("u1", "first"),
            step_event(0, "print 1", "1"),
            user_event("u2", "second"),
            step_event(1, "print 2", "2"),
        ]);
        assert!(extended.starts_with(&initial));
        assert!(!extended.contains("=== Message 2 ==="));
        assert!(extended.contains("=== Step 2 ==="));
    }
}
