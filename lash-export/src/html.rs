//! Self-contained HTML renderer for a loaded session.
//!
//! The output is a single HTML document with inlined CSS — no external
//! assets, safe to attach to an email or drop on a static host.

use std::collections::HashSet;
use std::fmt::Write as _;

use lash::session_model::{Message, MessageRole, Part, PartKind, PruneState};
use lash::{ChronologicalPayload, RlmTrajectoryEntry, ToolCallRecord};

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
        "  <div class=\"counts\">{} chronological entries</div>",
        session.chronological.len()
    );
    out.push_str("</header>\n");
}

fn render_transcript(out: &mut String, session: &LoadedSession) {
    out.push_str("<main class=\"transcript\">\n");
    let mut seen_tool_calls = HashSet::new();
    for entry in &session.chronological {
        match &entry.payload {
            ChronologicalPayload::Message(message) => {
                if !message.is_transient() {
                    render_message(out, message);
                }
            }
            ChronologicalPayload::ToolCall(record) => {
                seen_tool_calls.insert(lash::chronological_tool_call_key(record));
                render_tool_call_record(out, record);
            }
            ChronologicalPayload::RlmStep(step) => {
                render_rlm_step(out, step, &mut seen_tool_calls);
            }
        }
    }
    out.push_str("</main>\n");
}

fn render_message(out: &mut String, message: &Message) {
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
        render_part(out, part);
    }

    out.push_str("    </div>\n  </article>\n");
}

fn render_part(out: &mut String, part: &Part) {
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
        PartKind::ToolCall => render_tool_call_part(out, part),
        PartKind::ToolResult => {}
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
            "      <div class=\"part part--image\" data-attachment-id=\"{}\">[image: {}]</div>",
            escape_attr(attachment.reference.id.as_str()),
            escape_attr(alt)
        );
    } else {
        out.push_str("      <div class=\"part part--image\">[image]</div>\n");
    }
}

fn render_tool_call_part(out: &mut String, part: &Part) {
    let tool_name = part.tool_name.as_deref().unwrap_or("tool");
    let _ = writeln!(
        out,
        "      <details class=\"part part--tool-call tool--pending\">"
    );
    let _ = writeln!(
        out,
        "        <summary><span class=\"tool-name\">{}</span> <span class=\"tool-meta\">call</span></summary>",
        escape(tool_name),
    );

    if !part.content.trim().is_empty() && part.content.trim() != "{}" {
        out.push_str("        <div class=\"tool-section\">\n");
        out.push_str("          <div class=\"tool-section-label\">arguments</div>\n");
        let _ = writeln!(
            out,
            "          <pre class=\"tool-args\">{}</pre>",
            escape(&pretty_or_raw(&part.content))
        );
        out.push_str("        </div>\n");
    }
    out.push_str("      </details>\n");
}

fn render_tool_call_record(out: &mut String, record: &ToolCallRecord) {
    let status_class = if record.success { "ok" } else { "err" };
    let status_label = if record.success { "ok" } else { "error" };
    let _ = writeln!(
        out,
        "  <article class=\"msg msg--tool\"><details class=\"part part--tool-call tool--{status_class}\" open>"
    );
    let _ = writeln!(
        out,
        "    <summary><span class=\"tool-name\">{}</span> <span class=\"tool-meta\">{} ms · {status_label}</span></summary>",
        escape(&record.tool),
        record.duration_ms
    );
    out.push_str("    <div class=\"tool-section\">\n");
    out.push_str("      <div class=\"tool-section-label\">arguments</div>\n");
    let _ = writeln!(
        out,
        "      <pre class=\"tool-args\">{}</pre>",
        escape(&pretty_json(&record.args))
    );
    out.push_str("    </div>\n");
    out.push_str("    <div class=\"tool-section\">\n");
    out.push_str("      <div class=\"tool-section-label\">result</div>\n");
    let _ = writeln!(
        out,
        "      <pre class=\"tool-result\">{}</pre>",
        escape(&pretty_json(&record.result))
    );
    out.push_str("    </div>\n");
    out.push_str("  </details></article>\n");
}

fn render_rlm_step(
    out: &mut String,
    step: &RlmTrajectoryEntry,
    seen_tool_calls: &mut HashSet<String>,
) {
    out.push_str("  <article class=\"msg msg--rlm-step\">\n");
    let _ = writeln!(
        out,
        "    <div class=\"role\">RLM step {}</div>",
        step.iteration
    );
    out.push_str("    <div class=\"parts\">\n");
    render_text_block(
        out,
        "reasoning",
        &strip_first_lashlang_fence(&step.reasoning),
    );
    render_pre_block(out, "code", &step.code);
    for observation in &step.observations {
        render_pre_block(out, "output", observation);
    }
    render_pre_block(out, "output", &step.output);
    if let Some(error) = &step.error {
        render_pre_block(out, "error", error);
    }
    if let Some(final_output) = &step.final_output {
        render_pre_block(out, "output", &pretty_json(final_output));
    }
    out.push_str("    </div>\n  </article>\n");

    for record in &step.tool_calls {
        if seen_tool_calls.insert(lash::chronological_tool_call_key(record)) {
            render_tool_call_record(out, record);
        }
    }
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

fn strip_first_lashlang_fence(text: &str) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;
    use lash::{ChronologicalEntry, ChronologicalPayload, ToolCallRecord};

    #[test]
    fn html_export_renders_chronological_tool_and_rlm_step() {
        let session = LoadedSession {
            meta: None,
            chronological: vec![
                ChronologicalEntry {
                    index: 0,
                    payload: ChronologicalPayload::RlmStep(RlmTrajectoryEntry {
                        id: "rlm_step_0".to_string(),
                        iteration: 0,
                        reasoning: "thinking".to_string(),
                        code: "x = 1".to_string(),
                        output: "1".to_string(),
                        observations: Vec::new(),
                        tool_calls: Vec::new(),
                        images: Vec::new(),
                        error: None,
                        final_output: None,
                        output_raw_len: 1,
                    }),
                },
                ChronologicalEntry {
                    index: 1,
                    payload: ChronologicalPayload::ToolCall(ToolCallRecord {
                        call_id: Some("call_1".to_string()),
                        tool: "lookup".to_string(),
                        args: serde_json::json!({"q": "x"}),
                        result: serde_json::json!({"answer": "y"}),
                        success: true,
                        duration_ms: 4,
                    }),
                },
            ],
        };

        let rendered = render(&session);
        assert!(rendered.contains("RLM step 0"));
        assert!(rendered.contains("x = 1"));
        assert!(rendered.contains("lookup"));
        assert!(rendered.contains("chronological entries"));
    }
}
