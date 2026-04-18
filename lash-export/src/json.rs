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
            "parent_session_id": meta.parent_session_id,
        })
    });

    let document = json!({
        "meta": meta,
        "messages": session.messages,
        "tool_calls": session.tool_calls,
    });

    serde_json::to_string_pretty(&document).unwrap_or_else(|_| document.to_string())
}
