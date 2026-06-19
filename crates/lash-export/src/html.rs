//! Self-contained HTML renderer for a loaded session.
//!
//! Visual language follows `docs/design-language.html` — sodium/ash/chalk
//! on warm-black, glyphs (`●`, `■`, `┊`, `•`, `×`, `◆`) for entry kinds,
//! `mock-bar` strip for the session header. Adds debug affordances on
//! top: a vertical spine minimap, sticky filter chips + search, per-entry
//! anchors, copy buttons, expand-all toggle, and keyboard navigation
//! (`j/k`, `e`, `c`, `/`, `Esc`, `Home/End`).
//!
//! Fonts load from Google Fonts when online (matching design-language.html);
//! offline the page falls back to system mono and stays fully readable.
//! All CSS and JS are inlined — drop the file anywhere.
//!
mod assets;
mod entries;
mod escaping;
mod prompt;
mod session;
mod stats;
mod tree;
mod view_model;

pub use session::render;
pub use tree::render_tree;

#[cfg(test)]
mod tests {
    use super::prompt::compute_prompt_insertions;
    use super::*;
    use crate::LoadedSession;
    use crate::trace::{LlmCallUsage, LlmPromptSnapshot, RequestMessage};
    use lash_core::session_model::{Part, PartKind, PruneState, shared_parts};
    use lash_core::{ChronologicalEntry, ChronologicalPayload};
    use lash_rlm_types::RlmTrajectoryEntry;
    use std::path::PathBuf;

    fn prompt_snapshot(protocol_iteration: u64, text: &str) -> LlmPromptSnapshot {
        LlmPromptSnapshot {
            session_id: Some("root".to_string()),
            turn_index: Some(1),
            protocol_iteration: Some(protocol_iteration),
            llm_call_id: Some(format!("root:1:{protocol_iteration}:0")),
            caused_by: None,
            timestamp: None,
            model: Some("gpt-test".to_string()),
            model_variant: None,
            system_text: "You are lash.".to_string(),
            system_chars: 13,
            system_hash: "abc123".to_string(),
            message_count: 2,
            total_chars: 13 + text.chars().count(),
            request_messages: vec![RequestMessage {
                role: "user".to_string(),
                text: text.to_string(),
                chars: text.chars().count(),
            }],
            request_chars: text.chars().count(),
            request_hash: text.to_string(),
            usage: None,
        }
    }

    fn user_message(id: &str, text: &str) -> lash_core::session_model::Message {
        lash_core::session_model::Message {
            id: id.to_string(),
            role: lash_core::session_model::MessageRole::User,
            parts: shared_parts(vec![Part {
                id: format!("{id}.p0"),
                kind: PartKind::Text,
                content: text.to_string(),
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

    fn rlm_step(protocol_iteration: usize, id: &str) -> RlmTrajectoryEntry {
        RlmTrajectoryEntry {
            id: id.to_string(),
            protocol_iteration,
            reasoning: "thinking".to_string(),
            code: "x = 1".to_string(),
            output: vec!["1".to_string()],
            images: Vec::new(),
            error: None,
            final_output: None,
        }
    }

    fn rlm_payload(step: RlmTrajectoryEntry) -> ChronologicalPayload {
        ChronologicalPayload::ProtocolEvent(lash_protocol_rlm::rlm_protocol_event(
            lash_rlm_types::RlmProtocolEvent::RlmTrajectoryEntry(step),
        ))
    }

    #[test]
    fn html_export_renders_chronological_message_and_rlm_step() {
        let session = LoadedSession {
            meta: None,
            chronological: vec![
                ChronologicalEntry {
                    index: 0,
                    payload: rlm_payload(rlm_step(0, "rlm_step_0")),
                },
                ChronologicalEntry {
                    index: 1,
                    payload: ChronologicalPayload::Message(user_message("m1", "lookup result")),
                },
            ],
            trace_path: PathBuf::from("session.trace.jsonl"),
            context_window_tokens: None,
            llm_prompts: Vec::new(),
        };

        let rendered = render(&session);
        assert!(rendered.contains("RLM step 0"));
        assert!(rendered.contains("x = 1"));
        assert!(rendered.contains("lookup result"));
        assert!(rendered.contains("chronological entries"));
    }

    #[test]
    fn repeated_rlm_trace_protocol_iterations_are_anchored_in_prompt_order() {
        let chronological = vec![
            ChronologicalEntry {
                index: 0,
                payload: ChronologicalPayload::Message(user_message("m0", "turn 1")),
            },
            ChronologicalEntry {
                index: 1,
                payload: rlm_payload(rlm_step(0, "rlm_step_0")),
            },
            ChronologicalEntry {
                index: 2,
                payload: ChronologicalPayload::Message(user_message("m1", "turn 2")),
            },
            ChronologicalEntry {
                index: 3,
                payload: rlm_payload(rlm_step(0, "rlm_step_1")),
            },
            ChronologicalEntry {
                index: 4,
                payload: ChronologicalPayload::Message(user_message("m2", "turn 3")),
            },
            ChronologicalEntry {
                index: 5,
                payload: rlm_payload(rlm_step(0, "rlm_step_2")),
            },
        ];
        let prompts = vec![
            prompt_snapshot(0, "request 1"),
            prompt_snapshot(0, "request 2"),
            prompt_snapshot(0, "request 3"),
        ];

        let insertions = compute_prompt_insertions(&chronological, &prompts);

        assert_eq!(insertions.before_index[0], vec![0]);
        assert_eq!(insertions.before_index[1], Vec::<usize>::new());
        assert_eq!(insertions.before_index[3], vec![1]);
        assert_eq!(insertions.before_index[5], vec![2]);
        assert!(insertions.trailing.is_empty());
    }

    #[test]
    fn assistant_tool_call_message_part_renders_without_detached_tool_record() {
        let tool_part = Part {
            id: "m0.p0".to_string(),
            kind: PartKind::ToolCall,
            content: r#"{"q":"x"}"#.to_string(),
            attachment: None,
            tool_call_id: Some("call_1".to_string()),
            tool_name: Some("lookup".to_string()),
            tool_replay: None,
            prune_state: PruneState::Intact,
            reasoning_meta: None,
            response_meta: None,
        };
        let assistant_msg = lash_core::session_model::Message {
            id: "m0".to_string(),
            role: lash_core::session_model::MessageRole::Assistant,
            parts: shared_parts(vec![tool_part]),
            origin: None,
        };
        let session = LoadedSession {
            meta: None,
            chronological: vec![ChronologicalEntry {
                index: 0,
                payload: ChronologicalPayload::Message(assistant_msg),
            }],
            trace_path: PathBuf::from("session.trace.jsonl"),
            context_window_tokens: None,
            llm_prompts: Vec::new(),
        };

        let rendered = render(&session);
        assert!(rendered.contains("lookup"));
        assert!(rendered.contains("call_1"));
        assert!(
            !rendered.contains("answer"),
            "detached tool result should not render"
        );
    }

    #[test]
    fn rlm_step_does_not_render_inline_hidden_tool_calls() {
        let session = LoadedSession {
            meta: None,
            chronological: vec![ChronologicalEntry {
                index: 0,
                payload: rlm_payload(RlmTrajectoryEntry {
                    code: "data = await tools.lookup({ q: \"x\" })?".to_string(),
                    output: Vec::new(),
                    ..rlm_step(0, "rlm_step_0")
                }),
            }],
            trace_path: PathBuf::from("session.trace.jsonl"),
            context_window_tokens: None,
            llm_prompts: Vec::new(),
        };

        let rendered = render(&session);
        assert!(rendered.contains("lookup"));
        assert!(!rendered.contains("rlm-tool-list"));
        assert!(!rendered.contains("entry--tool"));
    }

    #[test]
    fn provider_system_prompts_use_prompt_filter_role() {
        let session = LoadedSession {
            meta: None,
            chronological: Vec::new(),
            trace_path: PathBuf::from("session.trace.jsonl"),
            context_window_tokens: Some(100_000),
            llm_prompts: vec![LlmPromptSnapshot {
                usage: Some(LlmCallUsage {
                    input_tokens: 10_000,
                    output_tokens: 250,
                    cached_input_tokens: 7_500,
                    reasoning_tokens: 125,
                    duration_ms: Some(3000),
                }),
                ..prompt_snapshot(0, "hi")
            }],
        };

        let rendered = render(&session);
        assert!(rendered.contains("data-role=\"llm_call\""));
        assert!(rendered.contains("data-kind=\"system_prompt\""));
        assert!(rendered.contains(">llm call</button>"));
        assert!(rendered.contains("usage-chart"));
        assert!(rendered.contains("ctx 10.0%"));
        assert!(rendered.contains("75.000%"));
    }
}
