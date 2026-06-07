//! Leaf renderers: self-contained HTML emitters for message parts, prose,
//! reasoning, code, images, and small id/title formatting helpers.

use super::*;
use crate::html::escaping::escape_breaks;

pub(crate) fn render_part(out: &mut String, part: &Part) {
    match &part.prune_state {
        PruneState::Cleared => {
            out.push_str("          <div class=\"part part--pruned\" data-pruned=\"cleared\"><span class=\"part-tag\">cleared</span> tool result content cleared</div>\n");
            return;
        }
        PruneState::Deleted {
            breadcrumb,
            archive_hash,
        } => {
            let _ = writeln!(
                out,
                "          <div class=\"part part--pruned\" data-pruned=\"deleted\"><span class=\"part-tag\">pruned</span> {} <span class=\"part-archive\">{}</span></div>",
                escape(breadcrumb),
                escape(archive_hash)
            );
            return;
        }
        PruneState::Summarized {
            summary,
            archive_hash,
        } => {
            let _ = writeln!(
                out,
                "          <div class=\"part part--pruned\" data-pruned=\"summarized\"><span class=\"part-tag\">summarized</span> <span class=\"part-archive\">{}</span><div class=\"part-summary\">{}</div></div>",
                escape(archive_hash),
                escape_breaks(summary)
            );
            return;
        }
        PruneState::Intact => {}
    }

    match part.kind {
        PartKind::Text | PartKind::Prose => render_prose(out, &part.content),
        PartKind::Code => render_code(out, "code", &part.content, None),
        PartKind::Output => render_code(out, "output", &part.content, Some("output")),
        PartKind::Error => render_code(out, "error", &part.content, Some("error")),
        PartKind::Image => render_image_part(out, part),
        PartKind::ToolCall => render_tool_protocol_part(out, part, "tool_call"),
        PartKind::ToolResult => render_tool_protocol_part(out, part, "tool_result"),
        PartKind::Reasoning => render_reasoning(out, &part.content),
    }
}

/// Render a prose body as markdown. Long bodies are folded inside a
/// `<details>` so a multi-kilobyte chunk doesn't dominate the timeline —
/// the expanded form is one click away.
pub(crate) fn render_prose(out: &mut String, content: &str) {
    if content.trim().is_empty() {
        return;
    }
    let total_chars = content.chars().count();
    let collapse = total_chars > 1_000;
    let body_html = crate::markdown::render(content);

    if collapse {
        let preview = first_n_chars(content, 480);
        let _ = writeln!(
            out,
            "          <details class=\"part part--prose-fold\"><summary class=\"prose-fold-bar\"><span class=\"prose-fold-tag\">prose</span><span class=\"prose-fold-size\">{}</span><span class=\"prose-fold-preview\">{}</span><span class=\"prose-fold-hint\">click to expand full text</span></summary>",
            format_count(total_chars as u64),
            escape(&preview)
        );
        out.push_str("            <div class=\"prose-body\">\n");
        out.push_str(&body_html);
        out.push_str("\n            </div>\n");
        out.push_str("          </details>\n");
    } else {
        out.push_str("          <div class=\"part part--prose-wrap\">\n");
        let _ = writeln!(
            out,
            "            <div class=\"prose-bar\"><span class=\"prose-tag\">prose</span><span class=\"prose-size\">{}</span></div>",
            format_count(total_chars as u64)
        );
        out.push_str(&body_html);
        out.push_str("\n          </div>\n");
    }
}

pub(crate) fn first_n_chars(s: &str, n: usize) -> String {
    let mut out = String::with_capacity(n.min(s.len()));
    for ch in s.chars().take(n) {
        out.push(ch);
    }
    if s.chars().count() > n {
        out.push('…');
    }
    out
}

pub(crate) fn pick_display_title(session: &LoadedSession, name: &str, id: &str) -> String {
    // If the session_name is the session_id (a UUID), or empty, look for a
    // more useful title in the transcript: the first user message's first
    // line. Some user messages have UserInputProvenance set; others don't,
    // so fall back to assembling text from Text/Prose parts.
    let name_trim = name.trim();
    let is_uuid_like = name_trim == id || looks_like_uuid(name_trim) || name_trim.is_empty();
    if is_uuid_like {
        for entry in &session.chronological {
            if let ChronologicalPayload::Message(message) = &entry.payload {
                if message.is_transient() {
                    continue;
                }
                if !matches!(message.role, MessageRole::User) {
                    continue;
                }
                if let Some(text) = first_message_text(message) {
                    return one_line_summary(&text, 110);
                }
                for part in message.parts.iter() {
                    if matches!(part.kind, PartKind::Text | PartKind::Prose)
                        && !part.content.trim().is_empty()
                    {
                        return one_line_summary(&part.content, 110);
                    }
                }
            }
        }
    }
    if name_trim.is_empty() {
        "lash session".to_string()
    } else {
        name_trim.to_string()
    }
}

fn looks_like_uuid(s: &str) -> bool {
    // 8-4-4-4-12 hex with dashes
    let bytes = s.as_bytes();
    if bytes.len() != 36 {
        return false;
    }
    for (i, b) in bytes.iter().enumerate() {
        let dash = matches!(i, 8 | 13 | 18 | 23);
        if dash {
            if *b != b'-' {
                return false;
            }
        } else if !b.is_ascii_hexdigit() {
            return false;
        }
    }
    true
}

pub(crate) fn render_reasoning(out: &mut String, content: &str) {
    if content.trim().is_empty() {
        return;
    }
    let len = content.chars().count();
    out.push_str("          <div class=\"part part--reasoning\">\n");
    out.push_str("            <span class=\"reasoning-gutter\">┊</span>\n");
    let _ = writeln!(
        out,
        "            <div class=\"reasoning-text\">{}</div>",
        escape_breaks(content)
    );
    let _ = writeln!(
        out,
        "            <span class=\"reasoning-size\">{}</span>",
        format_count(len as u64)
    );
    out.push_str("          </div>\n");
}

pub(crate) fn render_code(out: &mut String, class: &str, content: &str, badge: Option<&str>) {
    if content.is_empty() {
        return;
    }
    let len = content.chars().count();
    let big = len > 2_000;
    let badge_html = badge
        .map(|b| format!("<span class=\"code-tag code-tag--{b}\">{b}</span>"))
        .unwrap_or_default();
    let size_html = format!(
        "<span class=\"code-size\">{}</span>",
        format_count(len as u64)
    );
    let copy_html = "<button class=\"code-copy\" data-copy>copy</button>";

    if big {
        let _ = writeln!(
            out,
            "          <details class=\"part part--{class}\"><summary class=\"code-bar\">{badge}{size}<span class=\"code-hint\">click to expand</span>{copy}</summary><pre class=\"code-pre\">{body}</pre></details>",
            badge = badge_html,
            size = size_html,
            copy = copy_html,
            body = escape(content),
        );
    } else {
        let _ = writeln!(
            out,
            "          <div class=\"part part--{class}\"><div class=\"code-bar\">{badge}{size}{copy}</div><pre class=\"code-pre\">{body}</pre></div>",
            badge = badge_html,
            size = size_html,
            copy = copy_html,
            body = escape(content),
        );
    }
}

fn render_image_part(out: &mut String, part: &Part) {
    let label = part
        .attachment
        .as_ref()
        .and_then(|a| a.reference.label.clone())
        .unwrap_or_else(|| {
            if part.content.trim().is_empty() {
                "image attached".to_string()
            } else {
                part.content.clone()
            }
        });
    let aref = part
        .attachment
        .as_ref()
        .map(|a| a.reference.id.to_string())
        .unwrap_or_default();
    let _ = writeln!(
        out,
        "          <div class=\"part part--image\"><span class=\"part-tag\">image</span><span class=\"image-label\">{}</span><span class=\"image-ref\">{}</span></div>",
        escape(&label),
        escape(&aref)
    );
}

fn render_tool_protocol_part(out: &mut String, part: &Part, label: &str) {
    let name = part.tool_name.as_deref().unwrap_or("unknown");
    let call_id = part.tool_call_id.as_deref().unwrap_or("no-call-id");
    let mut body = format!("{label}: {name}\ncall_id: {call_id}");
    if !part.content.trim().is_empty() {
        body.push_str("\n\n");
        body.push_str(&part.content);
    }
    render_code(out, "code", &body, Some(label));
}
