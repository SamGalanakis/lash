use std::sync::Arc;

use lash::llm::types::{LlmMessage, LlmRole, LlmToolChoice};
use lash::sansio::ContextProjector;
use lash::session_model::{ConversationRecord, MessageRole, PartKind, SessionEventRecord};
use lash::{
    LlmRequest, ModeBuildInput, ModeConfig, ModePreamble, ProjectorContext, PromptContribution,
    head_tail_truncate,
};
use lash_rlm_types::{RlmModeEvent, RlmTrajectoryEntry};

/// Trajectory-shaped RLM prompt. The protocol driver is the existing RLM
/// lashlang executor; this mode differs by presenting history as a
/// compact REPL trajectory rather than as chat-shaped context.
pub const RLMPURE_EXECUTION_SECTION: &str = r#"You have access to a persistent lashlang REPL. Write a small amount of reasoning plus exactly one fenced `lashlang` block when you need to act. The block is executed immediately; its `print` output becomes the next observation, and variables persist across later blocks in this turn.

Available inside lashlang:
- `print <expr>` — inspect a value; output appears as the next observation.
- `submit <expr>` — final answer; ends the run.
- `call tool_name { ... }` — call any tool under Available Tools. Use `(call tool_name { ... })?` for fail-fast unwrapping.
- Persistent variables — values assigned in one block are available in later blocks.

Execution discipline:
1. EXPLORE FIRST when the answer depends on data you haven't seen: inspect file snippets, tool outputs, types, and assumptions before finalizing. For trivial questions or known-shape passthroughs, skip straight to `submit`.
2. ITERATE on non-trivial work: small focused blocks; observe; decide the next step. Don't try to solve everything in one block.
3. INSPECT BEFORE SUBMITTING WHEN THE OUTPUT IS UNCERTAIN: if a tool result could be empty, malformed, or need transformation, `print` it in one block and `submit` on a later block. Combining `call` + `submit` in one block is fine when the result is direct passthrough or you're returning a fixed answer.
4. VERIFY ON SURPRISES: if a result looks empty, off, or error-shaped, investigate before `submit`.
5. MINIMIZE RETYPING: keep exact long values in variables and compute from them instead of copying.
6. KEEP OUTPUTS TARGETED: large `print` output is capped; print slices or selected fields.

Only the first ` ```lashlang ` fenced block is executed. Trailing prose or later fences are ignored after that block closes.
"#;

const RLMPURE_FINALIZATION_SECTION: &str = r#"When the task is complete, call `submit <value>` from lashlang. Strings are rendered as prose; records/lists are rendered as JSON unless a required schema is active."#;

pub fn rlmpure_execution_section() -> String {
    format!(
        "{RLMPURE_EXECUTION_SECTION}\n\n{}\n\n{RLMPURE_FINALIZATION_SECTION}",
        lash_mode_rlm::LASHLANG_LANGUAGE_REFERENCE
    )
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
}

impl ContextProjector<lash::HostModeProtocol> for RlmpureContextProjector {
    fn project(&self, ctx: ProjectorContext<'_>) -> LlmRequest {
        let repl_history = format_repl_history(ctx.events, self.max_output_chars);
        let user_prompt = format!(
            "REPL history\n{repl_history}\n\nIteration\n{}\n\nNext prediction\nProduce reasoning plus one fenced `lashlang` block. Use `print` for observations or `submit` when the final output is ready.",
            ctx.iteration + 1
        );

        let mut messages = Vec::new();
        if !ctx.config.system_prompt.trim().is_empty() {
            messages.push(LlmMessage::text(
                LlmRole::System,
                ctx.config.system_prompt.as_str().to_owned(),
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

fn format_repl_history(events: &[SessionEventRecord], max_output_chars: usize) -> String {
    let mut out = String::new();
    let mut user_message_index = 0usize;
    let mut step_index = 0usize;
    for event in events {
        match event {
            SessionEventRecord::Conversation(record) if record.role == MessageRole::User => {
                let Some(text) = conversation_text(record) else {
                    continue;
                };
                user_message_index += 1;
                if !out.is_empty() {
                    out.push_str("\n\n");
                }
                append_user_message(&mut out, user_message_index, &text, max_output_chars);
            }
            SessionEventRecord::Mode(event) => {
                if let Some(RlmModeEvent::RlmTrajectoryEntry(entry)) = event.rlm_event() {
                    step_index += 1;
                    if !out.is_empty() {
                        out.push_str("\n\n");
                    }
                    append_repl_step(&mut out, step_index, &entry, max_output_chars);
                }
            }
            _ => {}
        }
    }
    if out.is_empty() {
        "No user messages or REPL interactions yet.".to_string()
    } else {
        out
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
            }],
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

    #[test]
    fn repl_history_interleaves_user_messages_and_steps() {
        let history = format_repl_history(
            &[
                user_event("u1", "first"),
                step_event(0, "print 1", "1"),
                user_event("u2", "second"),
                step_event(1, "print 2", "2"),
            ],
            100,
        );

        assert!(history.contains("=== Message 1 ===\nUser (5 chars):\nfirst"));
        assert!(history.contains("=== Step 1 ==="));
        assert!(history.contains("=== Message 2 ===\nUser (6 chars):\nsecond"));
        assert!(history.contains("=== Step 2 ==="));
        assert!(history.find("=== Message 1 ===") < history.find("=== Step 1 ==="));
        assert!(history.find("=== Step 1 ===") < history.find("=== Message 2 ==="));
        assert!(history.find("=== Message 2 ===") < history.find("=== Step 2 ==="));
        assert!(!history.contains("Inputs"));
        assert!(!history.contains("omitted]"));
    }

    #[test]
    fn long_user_message_gets_binding_hint() {
        let history = format_repl_history(&[user_event("u1", "abcdefghijklmnopqrstuvwxyz")], 10);

        assert!(history.contains("available as `user_input_1`"));
        assert!(history.contains("... (16 characters omitted) ..."));
    }
}
