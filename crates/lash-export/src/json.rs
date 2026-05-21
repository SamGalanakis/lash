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
    use lash_core::{ChronologicalEntry, ChronologicalPayload, ToolCallRecord};
    use std::path::PathBuf;

    #[test]
    fn json_export_uses_chronological_entries() {
        let session = LoadedSession {
            meta: None,
            chronological: vec![ChronologicalEntry {
                index: 0,
                payload: ChronologicalPayload::ToolCall(ToolCallRecord {
                    call_id: Some("call_1".to_string()),
                    tool: "lookup".to_string(),
                    args: serde_json::json!({"q": "x"}),
                    output: lash_core::ToolCallOutput::success(serde_json::json!({"answer": "y"})),
                    duration_ms: 4,
                }),
            }],
            trace_path: PathBuf::from("session.trace.jsonl"),
            context_window_tokens: None,
            llm_prompts: Vec::new(),
        };

        let rendered = render(&session);
        assert!(rendered.contains("\"chronological\""));
        assert!(rendered.contains("\"kind\": \"tool_call\""));
        assert!(!rendered.contains("\"messages\""));
        assert!(!rendered.contains("\"tool_calls\""));
    }
}
