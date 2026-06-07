//! Machine-readable JSON dump of a loaded session.

use serde_json::json;

use crate::LoadedSession;

pub fn render(session: &LoadedSession) -> String {
    let meta = session.meta.as_ref().map(|meta| {
        json!({
            "session_id": meta.session_id,
            "session_name": meta.session_name,
            "created_at": meta.created_at,
            "model": meta.model,
            "cwd": meta.cwd,
            "parent_session_id": meta.parent_session_id(),
            "relation": meta.relation,
        })
    });

    let document = json!({
        "meta": meta,
        "trace_path": session.trace_path.display().to_string(),
        "context_window_tokens": session.context_window_tokens,
        "chronological": session.chronological,
        "llm_prompts": session.llm_prompts,
    });

    serde_json::to_string_pretty(&document).unwrap_or_else(|_| document.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lash_core::{
        ChronologicalEntry, ChronologicalPayload, Message, MessageRole, PruneState, shared_parts,
    };
    use std::path::PathBuf;

    #[test]
    fn json_export_uses_chronological_entries() {
        let session = LoadedSession {
            meta: None,
            chronological: vec![ChronologicalEntry {
                index: 0,
                payload: ChronologicalPayload::Message(Message {
                    id: "m1".to_string(),
                    role: MessageRole::User,
                    parts: shared_parts(vec![lash_core::Part {
                        id: "p1".to_string(),
                        kind: lash_core::PartKind::Text,
                        content: "hello".to_string(),
                        attachment: None,
                        tool_call_id: None,
                        tool_name: None,
                        tool_replay: None,
                        prune_state: PruneState::Intact,
                        reasoning_meta: None,
                        response_meta: None,
                    }]),
                    origin: None,
                }),
            }],
            trace_path: PathBuf::from("session.trace.jsonl"),
            context_window_tokens: None,
            llm_prompts: Vec::new(),
        };

        let rendered = render(&session);
        assert!(rendered.contains("\"chronological\""));
        assert!(rendered.contains("\"kind\": \"message\""));
        assert!(!rendered.contains("\"messages\""));
        assert!(!rendered.contains("\"tool_calls\""));
    }
}
