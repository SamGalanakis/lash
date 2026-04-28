//! Self-contained HTML renderer for a loaded session.
//!
//! The output is a single HTML document with inlined CSS — no external
//! assets, safe to attach to an email or drop on a static host.

use std::collections::HashMap;
use std::fmt::Write as _;

use lash::ToolCallRecord;
use lash::session_model::{Message, MessageRole, Part, PartKind, PruneState};

use crate::LoadedSession;

pub fn render(session: &LoadedSession) -> String {
    let mut out = String::with_capacity(8 * 1024);
    let title = session
        .meta
        .as_ref()
        .map(|meta| meta.session_name.clone())
        .unwrap_or_else(|| "lash session".to_string());

    out.push_str("<!DOCTYPE html>\n");
    out.push_str("<html lang=\"en\">\n<head>\n");
    out.push_str("<meta charset=\"utf-8\">\n");
    let _ = writeln!(out, "<title>{}</title>", escape(&title));
    out.push_str("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n");
    out.push_str("<style>\n");
    out.push_str(CSS);
    out.push_str("\n</style>\n</head>\n<body>\n");

    render_header(&mut out, session);
    render_transcript(&mut out, session);

    out.push_str("</body>\n</html>\n");
    out
}

fn render_header(out: &mut String, session: &LoadedSession) {
    out.push_str("<header class=\"lash-header\">\n");
    let (name, meta_line) = if let Some(meta) = &session.meta {
        let mut parts: Vec<String> = Vec::new();
        parts.push(format!("id: {}", escape(&meta.session_id)));
        parts.push(format!("model: {}", escape(&meta.model)));
        parts.push(format!("created: {}", escape(&meta.created_at)));
        if let Some(cwd) = &meta.cwd {
            parts.push(format!("cwd: {}", escape(cwd)));
        }
        if let Some(parent) = &meta.parent_session_id {
            parts.push(format!("parent: {}", escape(parent)));
        }
        (meta.session_name.clone(), parts.join(" · "))
    } else {
        ("lash session".to_string(), String::new())
    };
    let _ = writeln!(out, "  <h1>{}</h1>", escape(&name));
    if !meta_line.is_empty() {
        let _ = writeln!(out, "  <div class=\"meta\">{}</div>", escape(&meta_line));
    }
    let _ = writeln!(
        out,
        "  <div class=\"counts\">{} messages · {} tool calls</div>",
        session.messages.len(),
        session.tool_calls.len()
    );
    out.push_str("</header>\n");
}

fn render_transcript(out: &mut String, session: &LoadedSession) {
    let tool_index = index_tool_calls(&session.tool_calls);
    out.push_str("<main class=\"transcript\">\n");
    for message in &session.messages {
        if message.is_transient() {
            continue;
        }
        render_message(out, message, &tool_index);
    }
    out.push_str("</main>\n");
}

fn index_tool_calls(records: &[ToolCallRecord]) -> HashMap<&str, &ToolCallRecord> {
    records
        .iter()
        .filter_map(|record| record.call_id.as_deref().map(|id| (id, record)))
        .collect()
}

fn render_message(
    out: &mut String,
    message: &Message,
    tool_index: &HashMap<&str, &ToolCallRecord>,
) {
    let (role_class, role_label) = match message.role {
        MessageRole::User => ("user", "User"),
        MessageRole::Assistant => ("assistant", "Assistant"),
        MessageRole::System => ("system", "System"),
    };

    let _ = writeln!(out, "  <article class=\"msg msg--{role_class}\">");
    let _ = writeln!(out, "    <div class=\"role\">{role_label}</div>");
    out.push_str("    <div class=\"parts\">\n");

    if matches!(message.role, MessageRole::User)
        && let Some(text) = message.display_user_text()
    {
        render_text_block(out, "user-text", text);
        out.push_str("    </div>\n  </article>\n");
        return;
    }

    for part in message.parts.iter() {
        render_part(out, message.role, part, tool_index);
    }

    out.push_str("    </div>\n  </article>\n");
}

fn render_part(
    out: &mut String,
    role: MessageRole,
    part: &Part,
    tool_index: &HashMap<&str, &ToolCallRecord>,
) {
    if matches!(part.prune_state, PruneState::Cleared) {
        out.push_str("      <div class=\"part part--pruned\">[tool result cleared]</div>\n");
        return;
    }
    if let PruneState::Deleted { breadcrumb, .. } = &part.prune_state {
        let _ = writeln!(
            out,
            "      <div class=\"part part--pruned\">[pruned — {}]</div>",
            escape(breadcrumb)
        );
        return;
    }

    match part.kind {
        PartKind::Text | PartKind::Prose => render_text_block(out, "text", &part.content),
        PartKind::Code => render_pre_block(out, "code", &part.content),
        PartKind::Output => render_pre_block(out, "output", &part.content),
        PartKind::Error => render_pre_block(out, "error", &part.content),
        PartKind::Image => render_image(out, part),
        PartKind::ToolCall => render_tool_call(out, part, tool_index),
        PartKind::ToolResult => {
            if !matches!(role, MessageRole::Assistant) {
                render_pre_block(out, "tool-result-inline", &part.content);
            }
        }
        // Reasoning summaries render as a muted/italic block in exported
        // HTML for parity with the TUI presentation.
        PartKind::Reasoning => render_text_block(out, "reasoning", &part.content),
    }
}

fn render_text_block(out: &mut String, class: &str, content: &str) {
    if content.trim().is_empty() {
        return;
    }
    let _ = writeln!(
        out,
        "      <div class=\"part part--{class}\">{}</div>",
        escape_with_breaks(content)
    );
}

fn render_pre_block(out: &mut String, class: &str, content: &str) {
    if content.is_empty() {
        return;
    }
    let _ = writeln!(
        out,
        "      <pre class=\"part part--{class}\">{}</pre>",
        escape(content)
    );
}

fn render_image(out: &mut String, part: &Part) {
    if let Some(attachment) = &part.attachment {
        let alt = if part.content.trim().is_empty() {
            "attached image"
        } else {
            part.content.as_str()
        };
        let _ = writeln!(
            out,
            "      <div class=\"part part--image\"><img src=\"{}\" alt=\"{}\"></div>",
            escape_attr(&attachment.url),
            escape_attr(alt)
        );
    } else {
        out.push_str("      <div class=\"part part--image\">[image]</div>\n");
    }
}

fn render_tool_call(out: &mut String, part: &Part, tool_index: &HashMap<&str, &ToolCallRecord>) {
    let tool_name = part.tool_name.as_deref().unwrap_or("tool");
    let record = part
        .tool_call_id
        .as_deref()
        .and_then(|id| tool_index.get(id).copied());

    let success = record.map(|record| record.success).unwrap_or(true);
    let duration = record
        .map(|record| format!("{} ms", record.duration_ms))
        .unwrap_or_else(|| "—".to_string());
    let status_class = if success { "ok" } else { "err" };
    let status_label = if success { "ok" } else { "error" };

    let _ = writeln!(
        out,
        "      <details class=\"part part--tool-call tool--{status_class}\">"
    );
    let _ = writeln!(
        out,
        "        <summary><span class=\"tool-name\">{}</span> <span class=\"tool-meta\">{} · {}</span></summary>",
        escape(tool_name),
        escape(&duration),
        status_label
    );

    let args_json = if part.content.trim().is_empty() || part.content.trim() == "{}" {
        record.map(|record| pretty_json(&record.args))
    } else {
        Some(pretty_or_raw(&part.content))
    };
    if let Some(args) = args_json {
        out.push_str("        <div class=\"tool-section\">\n");
        out.push_str("          <div class=\"tool-section-label\">arguments</div>\n");
        let _ = writeln!(
            out,
            "          <pre class=\"tool-args\">{}</pre>",
            escape(&args)
        );
        out.push_str("        </div>\n");
    }
    if let Some(record) = record {
        out.push_str("        <div class=\"tool-section\">\n");
        out.push_str("          <div class=\"tool-section-label\">result</div>\n");
        let _ = writeln!(
            out,
            "          <pre class=\"tool-result\">{}</pre>",
            escape(&pretty_json(&record.result))
        );
        out.push_str("        </div>\n");
    }
    out.push_str("      </details>\n");
}

fn pretty_json(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(text) => text.clone(),
        _ => serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string()),
    }
}

fn pretty_or_raw(content: &str) -> String {
    match serde_json::from_str::<serde_json::Value>(content) {
        Ok(value) => serde_json::to_string_pretty(&value).unwrap_or_else(|_| content.to_string()),
        Err(_) => content.to_string(),
    }
}

fn escape(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(ch),
        }
    }
    out
}

fn escape_attr(input: &str) -> String {
    escape(input)
}

fn escape_with_breaks(input: &str) -> String {
    let escaped = escape(input);
    escaped.replace('\n', "<br>")
}

const CSS: &str = r#"
:root {
  --bg: #0f1115;
  --fg: #e6e6e6;
  --muted: #8b93a3;
  --panel: #161922;
  --panel-2: #1c2030;
  --border: #252a38;
  --user: #5aa6ff;
  --assistant: #b68bff;
  --system: #8b93a3;
  --ok: #4caf83;
  --err: #e06b6b;
  --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  background: var(--bg);
  color: var(--fg);
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  padding: 32px 24px 96px;
  max-width: 960px;
  margin: 0 auto;
}
.lash-header {
  border-bottom: 1px solid var(--border);
  padding-bottom: 16px;
  margin-bottom: 24px;
}
.lash-header h1 {
  margin: 0 0 6px;
  font-size: 20px;
  font-weight: 600;
}
.lash-header .meta, .lash-header .counts {
  color: var(--muted);
  font-size: 12px;
  font-family: var(--mono);
}
.lash-header .counts { margin-top: 4px; }
.transcript { display: flex; flex-direction: column; gap: 18px; }
.msg {
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--panel);
  padding: 14px 16px;
}
.msg .role {
  font-size: 11px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 8px;
}
.msg--user .role { color: var(--user); }
.msg--assistant .role { color: var(--assistant); }
.msg--system { background: transparent; border-style: dashed; }
.msg--system .role { color: var(--system); }
.parts { display: flex; flex-direction: column; gap: 10px; }
.part { white-space: normal; }
.part--text, .part--user-text { color: var(--fg); }
.part--reasoning { color: var(--muted); font-style: italic; font-size: 13px; border-left: 2px solid var(--border); padding-left: 10px; margin: 6px 0; }
.part--pruned { color: var(--muted); font-style: italic; font-size: 13px; }
.part--image img { max-width: 100%; border-radius: 6px; border: 1px solid var(--border); }
pre.part {
  font-family: var(--mono);
  font-size: 13px;
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 10px 12px;
  overflow-x: auto;
  white-space: pre;
  margin: 0;
}
pre.part--error { border-color: var(--err); color: var(--err); }
pre.part--output { color: var(--muted); }
details.part--tool-call {
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0;
}
details.part--tool-call summary {
  cursor: pointer;
  padding: 8px 12px;
  display: flex;
  gap: 10px;
  align-items: baseline;
  font-family: var(--mono);
  font-size: 13px;
  list-style: none;
}
details.part--tool-call summary::-webkit-details-marker { display: none; }
details.part--tool-call summary::before {
  content: "▸";
  color: var(--muted);
  transition: transform 0.15s;
  display: inline-block;
}
details[open].part--tool-call summary::before { transform: rotate(90deg); }
.tool--ok summary .tool-name { color: var(--ok); }
.tool--err summary .tool-name { color: var(--err); }
.tool-meta { color: var(--muted); font-size: 12px; }
.tool-section { padding: 0 12px 10px; }
.tool-section-label {
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
  margin: 6px 0 4px;
}
pre.tool-args, pre.tool-result {
  font-family: var(--mono);
  font-size: 12px;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 8px 10px;
  overflow-x: auto;
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
}
"#;
