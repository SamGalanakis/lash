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
//! Tool calls are rendered exactly once. The chronological projection emits
//! both an assistant `Message` containing a `PartKind::ToolCall` part *and*
//! a separate `ChronologicalPayload::ToolCall` entry with the full record;
//! we suppress the in-message part because the standalone entry is the
//! canonical view (args + result + duration + status).

use std::collections::HashMap;
use std::fmt::Write as _;

use lash::session_model::{Message, MessageRole, Part, PartKind, PruneState};
use lash::{ChronologicalPayload, RlmTrajectoryEntry, ToolCallRecord};

use crate::LoadedSession;

pub fn render(session: &LoadedSession) -> String {
    let stats = compute_stats(session);
    let mut ctx = RenderCtx::new(&stats);

    let mut out = String::with_capacity(64 * 1024);
    let title = session
        .meta
        .as_ref()
        .map(|meta| meta.session_name.clone())
        .unwrap_or_else(|| "lash session".to_string());

    out.push_str("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n");
    out.push_str("<meta charset=\"utf-8\">\n");
    out.push_str("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n");
    let _ = writeln!(out, "<title>{} · lash trace</title>", escape(&title));
    out.push_str("<style>\n");
    out.push_str(CSS);
    out.push_str("\n</style>\n</head>\n<body>\n");

    out.push_str("<div class=\"page\"><div class=\"frame\">\n");
    write_hero(&mut out, session, &stats);
    write_session_bar(&mut out, session, &stats);
    write_controls(&mut out, &stats);
    write_body(&mut out, session, &mut ctx);
    write_footer(&mut out, &stats);
    out.push_str("</div></div>\n");

    out.push_str("<script>\n");
    out.push_str(JS);
    out.push_str("\n</script>\n");
    out.push_str("</body>\n</html>\n");
    out
}

// ─── stats ──────────────────────────────────────────────────────────────────

#[derive(Default, Debug)]
struct SessionStats {
    user_messages: usize,
    assistant_messages: usize,
    system_messages: usize,
    tool_calls_ok: usize,
    tool_calls_err: usize,
    tool_total_ms: u64,
    rlm_iterations: usize,
    rlm_errors: usize,
    pruned_parts: usize,
    cleared_parts: usize,
    total_chars: usize,
    chronological: usize,
    tool_freq: Vec<(String, usize)>,
    tool_names_set: Vec<String>,
}

fn compute_stats(session: &LoadedSession) -> SessionStats {
    let mut s = SessionStats {
        chronological: session.chronological.len(),
        ..SessionStats::default()
    };
    let mut tool_counts: HashMap<String, usize> = HashMap::new();
    let mut seen_tool_keys = std::collections::HashSet::new();

    let record_tool_call = |s: &mut SessionStats,
                            tool_counts: &mut HashMap<String, usize>,
                            record: &ToolCallRecord| {
        if record.success {
            s.tool_calls_ok += 1;
        } else {
            s.tool_calls_err += 1;
        }
        s.tool_total_ms = s.tool_total_ms.saturating_add(record.duration_ms);
        *tool_counts.entry(record.tool.clone()).or_insert(0) += 1;
    };

    for entry in &session.chronological {
        match &entry.payload {
            ChronologicalPayload::Message(message) => {
                if message.is_transient() {
                    continue;
                }
                match message.role {
                    MessageRole::User => s.user_messages += 1,
                    MessageRole::Assistant => s.assistant_messages += 1,
                    MessageRole::System => s.system_messages += 1,
                }
                for part in message.parts.iter() {
                    s.total_chars = s.total_chars.saturating_add(part.content.chars().count());
                    match &part.prune_state {
                        PruneState::Cleared => s.cleared_parts += 1,
                        PruneState::Deleted { .. } | PruneState::Summarized { .. } => {
                            s.pruned_parts += 1
                        }
                        PruneState::Intact => {}
                    }
                }
            }
            ChronologicalPayload::ToolCall(record) => {
                let key = lash::chronological_tool_call_key(record);
                if seen_tool_keys.insert(key) {
                    record_tool_call(&mut s, &mut tool_counts, record);
                }
            }
            ChronologicalPayload::RlmStep(step) => {
                s.rlm_iterations += 1;
                if step.error.is_some() {
                    s.rlm_errors += 1;
                }
                s.total_chars = s.total_chars.saturating_add(step.output.chars().count());
                s.total_chars = s.total_chars.saturating_add(step.code.chars().count());
                for record in &step.tool_calls {
                    let key = lash::chronological_tool_call_key(record);
                    if seen_tool_keys.insert(key) {
                        record_tool_call(&mut s, &mut tool_counts, record);
                    }
                }
            }
        }
    }

    let mut freq: Vec<(String, usize)> = tool_counts.into_iter().collect();
    freq.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    s.tool_names_set = freq.iter().map(|(name, _)| name.clone()).collect();
    s.tool_freq = freq;
    s
}

// ─── render context (used during walk to dedupe tool calls) ─────────────────

struct RenderCtx<'a> {
    seen_tool_keys: std::collections::HashSet<String>,
    next_index: usize,
    _stats: &'a SessionStats,
}

impl<'a> RenderCtx<'a> {
    fn new(stats: &'a SessionStats) -> Self {
        Self {
            seen_tool_keys: std::collections::HashSet::new(),
            next_index: 0,
            _stats: stats,
        }
    }
    fn next_id(&mut self) -> String {
        let n = self.next_index;
        self.next_index += 1;
        format!("e{n}")
    }
}

// ─── hero / session bar / controls / footer ─────────────────────────────────

fn write_hero(out: &mut String, session: &LoadedSession, _stats: &SessionStats) {
    let (name, model, id, created, cwd, parent) = if let Some(meta) = &session.meta {
        (
            meta.session_name.clone(),
            meta.model.clone(),
            meta.session_id.clone(),
            meta.created_at.clone(),
            meta.cwd.clone(),
            meta.parent_session_id.clone(),
        )
    } else {
        (
            "lash session".to_string(),
            String::new(),
            String::new(),
            String::new(),
            None,
            None,
        )
    };

    let display_title = pick_display_title(session, &name, &id);

    out.push_str("<header class=\"hero\">\n");
    out.push_str("  <div class=\"hero-left\">\n");
    out.push_str(
        "    <div class=\"eyebrow\">lash <span class=\"slash\">/</span> session trace</div>\n",
    );
    let _ = writeln!(
        out,
        "    <h1 class=\"hero-title\">{}</h1>",
        escape(&display_title)
    );
    out.push_str("    <div class=\"hero-meta\">\n");
    if !id.is_empty() {
        let _ = writeln!(
            out,
            "      <span class=\"meta-row\"><span class=\"meta-key\">id</span><span class=\"meta-val\">{}</span></span>",
            escape(&id)
        );
    }
    if !model.is_empty() {
        let _ = writeln!(
            out,
            "      <span class=\"meta-row\"><span class=\"meta-key\">model</span><span class=\"meta-val\">{}</span></span>",
            escape(&model)
        );
    }
    if !created.is_empty() {
        let _ = writeln!(
            out,
            "      <span class=\"meta-row\"><span class=\"meta-key\">created</span><span class=\"meta-val\">{}</span></span>",
            escape(&created)
        );
    }
    if let Some(cwd) = cwd {
        let _ = writeln!(
            out,
            "      <span class=\"meta-row\"><span class=\"meta-key\">cwd</span><span class=\"meta-val\">{}</span></span>",
            escape(&cwd)
        );
    }
    if let Some(parent) = parent {
        let _ = writeln!(
            out,
            "      <span class=\"meta-row\"><span class=\"meta-key\">parent</span><span class=\"meta-val\">{}</span></span>",
            escape(&parent)
        );
    }
    out.push_str("    </div>\n");
    out.push_str("  </div>\n");
    out.push_str("</header>\n");
}

fn write_session_bar(out: &mut String, _session: &LoadedSession, stats: &SessionStats) {
    let total_msgs = stats.user_messages + stats.assistant_messages + stats.system_messages;
    let total_calls = stats.tool_calls_ok + stats.tool_calls_err;
    let total_seconds = stats.tool_total_ms as f64 / 1000.0;
    out.push_str("<section class=\"session-bar\" aria-label=\"session statistics\">\n");
    let _ = writeln!(
        out,
        "  <div class=\"stat\"><span class=\"stat-key\">turns</span><span class=\"stat-val\">{total_msgs}</span><span class=\"stat-sub\">{u} u · {a} a · {sy} s</span></div>",
        u = stats.user_messages,
        a = stats.assistant_messages,
        sy = stats.system_messages,
    );
    // Tint just the err sub-line, not the whole stat — 526 ok / 3 err
    // shouldn't look like the whole tool-call counter is on fire.
    let err_sub_class = if stats.tool_calls_err > 0 {
        "stat-sub stat-sub--err"
    } else {
        "stat-sub"
    };
    let _ = writeln!(
        out,
        "  <div class=\"stat\"><span class=\"stat-key\">tool calls</span><span class=\"stat-val\">{total_calls}</span><span class=\"{err_sub_class}\">{ok} ok · {er} err</span></div>",
        ok = stats.tool_calls_ok,
        er = stats.tool_calls_err,
    );
    let _ = writeln!(
        out,
        "  <div class=\"stat\"><span class=\"stat-key\">tool time</span><span class=\"stat-val\">{}</span><span class=\"stat-sub\">{:.0} ms</span></div>",
        format_duration(stats.tool_total_ms),
        total_seconds * 1000.0,
    );
    let rlm_sub_class = if stats.rlm_errors > 0 {
        "stat-sub stat-sub--err"
    } else {
        "stat-sub"
    };
    let _ = writeln!(
        out,
        "  <div class=\"stat\"><span class=\"stat-key\">rlm</span><span class=\"stat-val\">{}</span><span class=\"{rlm_sub_class}\">{} err</span></div>",
        stats.rlm_iterations, stats.rlm_errors,
    );
    let _ = writeln!(
        out,
        "  <div class=\"stat\"><span class=\"stat-key\">chars</span><span class=\"stat-val\">{}</span><span class=\"stat-sub\">{} entries</span></div>",
        format_count(stats.total_chars as u64),
        stats.chronological,
    );
    if stats.pruned_parts + stats.cleared_parts > 0 {
        let _ = writeln!(
            out,
            "  <div class=\"stat stat--muted\"><span class=\"stat-key\">pruned</span><span class=\"stat-val\">{}</span><span class=\"stat-sub\">{} cleared</span></div>",
            stats.pruned_parts, stats.cleared_parts,
        );
    }
    out.push_str("</section>\n");
}

fn write_controls(out: &mut String, stats: &SessionStats) {
    out.push_str("<section class=\"controls\" aria-label=\"filters\">\n");

    out.push_str("  <div class=\"chip-row\" data-group=\"role\">\n");
    out.push_str("    <span class=\"chip-label\">show</span>\n");
    for (key, label) in [
        ("user", "user"),
        ("assistant", "assistant"),
        ("tool", "tool"),
        ("rlm", "rlm"),
        ("system", "system"),
    ] {
        let _ = writeln!(
            out,
            "    <button class=\"chip is-on\" data-filter=\"role\" data-value=\"{key}\">{label}</button>"
        );
    }
    out.push_str("  </div>\n");

    if !stats.tool_freq.is_empty() {
        out.push_str("  <div class=\"chip-row chip-row--tools\" data-group=\"tool\">\n");
        out.push_str("    <span class=\"chip-label\">tools</span>\n");
        for (name, count) in stats.tool_freq.iter().take(8) {
            let _ = writeln!(
                out,
                "    <button class=\"chip is-on\" data-filter=\"tool\" data-value=\"{name}\">{name}<span class=\"chip-count\">{count}</span></button>",
                name = escape(name),
                count = count,
            );
        }
        if stats.tool_freq.len() > 8 {
            out.push_str("    <button class=\"chip is-on\" data-filter=\"tool\" data-value=\"__other__\">other</button>\n");
        }
        out.push_str("  </div>\n");
    }

    out.push_str("  <div class=\"chip-row chip-row--toggles\">\n");
    out.push_str("    <button class=\"chip\" data-toggle=\"errors-only\" title=\"errors only (toggle)\">errors only</button>\n");
    out.push_str("    <button class=\"chip\" data-toggle=\"hide-pruned\" title=\"hide pruned/cleared parts (toggle)\">hide pruned</button>\n");
    out.push_str("    <button class=\"chip\" data-action=\"expand-all\" title=\"expand all (e)\">expand all</button>\n");
    out.push_str("    <button class=\"chip\" data-action=\"collapse-all\" title=\"collapse all (c)\">collapse all</button>\n");
    out.push_str("  </div>\n");

    out.push_str("  <div class=\"search\">\n");
    out.push_str("    <span class=\"search-glyph\">⌕</span>\n");
    out.push_str("    <input id=\"q\" type=\"search\" placeholder=\"search transcript  ( / )\" autocomplete=\"off\" spellcheck=\"false\">\n");
    out.push_str("    <span class=\"search-meta\" id=\"q-meta\"></span>\n");
    out.push_str("  </div>\n");

    out.push_str("  <div class=\"shortcuts\">\n");
    out.push_str("    <kbd>j</kbd><kbd>k</kbd> next/prev <span class=\"sep\">·</span> <kbd>e</kbd>/<kbd>c</kbd> expand/collapse <span class=\"sep\">·</span> <kbd>/</kbd> search <span class=\"sep\">·</span> <kbd>?</kbd> help\n");
    out.push_str("  </div>\n");

    out.push_str("</section>\n");
}

fn write_footer(out: &mut String, stats: &SessionStats) {
    let total = stats.chronological;
    let _ = writeln!(
        out,
        "<footer class=\"trace-foot\">{total} chronological entries · rendered by lash-export</footer>"
    );
}

// ─── body / spine + transcript ──────────────────────────────────────────────

fn write_body(out: &mut String, session: &LoadedSession, ctx: &mut RenderCtx<'_>) {
    let entries_html = render_entries(session, ctx);

    out.push_str("<div class=\"body\" id=\"body\">\n");
    out.push_str("  <aside class=\"spine\" aria-label=\"trace minimap\">\n");
    out.push_str(&entries_html.spine);
    out.push_str("  </aside>\n");
    out.push_str("  <main class=\"transcript\" id=\"transcript\">\n");
    out.push_str(&entries_html.entries);
    out.push_str("  </main>\n");
    out.push_str("</div>\n");
}

struct EntriesHtml {
    entries: String,
    spine: String,
}

fn render_entries(session: &LoadedSession, ctx: &mut RenderCtx<'_>) -> EntriesHtml {
    let mut entries = String::with_capacity(32 * 1024);
    let mut spine = String::with_capacity(2 * 1024);

    for entry in &session.chronological {
        match &entry.payload {
            ChronologicalPayload::Message(message) => {
                if message.is_transient() {
                    continue;
                }
                render_message(&mut entries, &mut spine, ctx, message);
            }
            ChronologicalPayload::ToolCall(record) => {
                let key = lash::chronological_tool_call_key(record);
                if !ctx.seen_tool_keys.insert(key) {
                    continue;
                }
                render_tool_call_entry(&mut entries, &mut spine, ctx, record, None);
            }
            ChronologicalPayload::RlmStep(step) => {
                render_rlm_step(&mut entries, &mut spine, ctx, step);
            }
        }
    }

    EntriesHtml { entries, spine }
}

// ─── messages ───────────────────────────────────────────────────────────────

fn render_message(
    out: &mut String,
    spine: &mut String,
    ctx: &mut RenderCtx<'_>,
    message: &Message,
) {
    let id = ctx.next_id();
    let (role_key, role_label, glyph) = match message.role {
        MessageRole::User => ("user", "user", "●"),
        MessageRole::Assistant => ("assistant", "assistant", "■"),
        MessageRole::System => ("system", "system", "◇"),
    };

    let user_text = if matches!(message.role, MessageRole::User) {
        message
            .display_user_text()
            .map(str::to_string)
            .filter(|t| !t.trim().is_empty())
    } else {
        None
    };

    let headline = headline_for_message(message, user_text.as_deref());
    // The body's first prose part starts with the same words the headline
    // previews; rendering both stutters. Suppress the head-row headline
    // whenever a non-empty Text/Prose part is present (or the user-input
    // provenance carries display text).
    let suppress_headline = user_text.is_some()
        || message.parts.iter().any(|p| {
            matches!(p.kind, PartKind::Text | PartKind::Prose) && !p.content.trim().is_empty()
        });
    let total_chars: usize = message
        .parts
        .iter()
        .map(|p| p.content.chars().count())
        .sum::<usize>()
        + user_text.as_deref().map(|s| s.chars().count()).unwrap_or(0);

    let search_text = build_search_text_message(message, user_text.as_deref());

    let _ = writeln!(
        out,
        "    <article class=\"entry entry--{role_key}\" id=\"{id}\" data-role=\"{role_key}\" data-kind=\"message\" data-search=\"{search}\">",
        search = escape_attr(&search_text)
    );
    out.push_str("      <div class=\"entry-rail\">\n");
    let _ = writeln!(
        out,
        "        <a class=\"entry-num\" href=\"#{id}\" title=\"permalink\">{id}</a>"
    );
    let _ = writeln!(out, "        <span class=\"entry-glyph\">{glyph}</span>");
    out.push_str("      </div>\n");
    out.push_str("      <div class=\"entry-body\">\n");
    out.push_str("        <header class=\"entry-head\">\n");
    let _ = writeln!(
        out,
        "          <span class=\"entry-tag\">{role_label}</span>"
    );
    if !suppress_headline {
        let _ = writeln!(
            out,
            "          <span class=\"entry-headline\">{}</span>",
            escape(&headline)
        );
    } else {
        // keep the row's flex layout balanced when there's no headline
        out.push_str("          <span class=\"entry-headline entry-headline--ghost\"></span>\n");
    }
    let _ = writeln!(
        out,
        "          <span class=\"entry-meta\">{}</span>",
        format_count(total_chars as u64)
    );
    out.push_str("        </header>\n");
    out.push_str("        <div class=\"entry-content\">\n");

    if let Some(text) = user_text {
        render_prose(out, &text);
    } else {
        for part in message.parts.iter() {
            render_part(out, part);
        }
    }

    out.push_str("        </div>\n");
    out.push_str("      </div>\n");
    out.push_str("    </article>\n");

    let _ = writeln!(
        spine,
        "    <a class=\"spine-tick\" href=\"#{id}\" data-spine=\"{role_key}\" title=\"{role_label} · {h}\"></a>",
        h = escape_attr(&truncate(&headline, 80)),
    );
}

fn headline_for_message(message: &Message, user_text: Option<&str>) -> String {
    if let Some(text) = user_text {
        return one_line_summary(text, 200);
    }
    // assistant / system: prefer first non-empty Text/Prose part
    for part in message.parts.iter() {
        if matches!(part.kind, PartKind::Text | PartKind::Prose) && !part.content.trim().is_empty()
        {
            return one_line_summary(&part.content, 200);
        }
    }
    // fall back to type counts
    let mut buckets: HashMap<&'static str, usize> = HashMap::new();
    for part in message.parts.iter() {
        let key = match part.kind {
            PartKind::Code => "code",
            PartKind::Output => "output",
            PartKind::Error => "error",
            PartKind::Image => "image",
            PartKind::ToolCall => "tool_call",
            PartKind::ToolResult => "tool_result",
            PartKind::Reasoning => "reasoning",
            PartKind::Text | PartKind::Prose => "text",
        };
        *buckets.entry(key).or_insert(0) += 1;
    }
    let mut summary: Vec<String> = buckets
        .into_iter()
        .map(|(k, v)| {
            if v == 1 {
                k.to_string()
            } else {
                format!("{v} {k}")
            }
        })
        .collect();
    summary.sort();
    if summary.is_empty() {
        "(empty message)".to_string()
    } else {
        summary.join(" · ")
    }
}

fn build_search_text_message(message: &Message, user_text: Option<&str>) -> String {
    let mut s = String::new();
    if let Some(t) = user_text {
        s.push_str(&t.to_lowercase());
        return s;
    }
    for part in message.parts.iter() {
        s.push_str(&part.content.to_lowercase());
        s.push('\n');
    }
    s
}

fn render_part(out: &mut String, part: &Part) {
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
        // Skip tool_call parts entirely — the separate ChronologicalPayload::ToolCall
        // entry carries the canonical record with args+result+duration.
        PartKind::ToolCall => {}
        PartKind::ToolResult => {}
        PartKind::Reasoning => render_reasoning(out, &part.content),
    }
}

/// Render a prose body. Splits the text at fenced code blocks
/// (` ```lang … ``` `) so embedded code renders as monospace `<pre>` while
/// the surrounding narrative stays in the editorial body face. Very long
/// total bodies are wrapped in a `<details>` so a single 13kb system prompt
/// doesn't eat six screens — the expanded form is still one click away.
fn render_prose(out: &mut String, content: &str) {
    if content.trim().is_empty() {
        return;
    }
    let chunks = split_prose_with_fences(content);
    let total_chars = content.chars().count();
    // Fold any prose longer than ~12 lines worth of body text. The fold
    // shows a one-line preview + a click-to-expand affordance, so a
    // multi-kilobyte system prompt no longer eats six screens before you
    // reach the next entry.
    let collapse = total_chars > 1_000;

    if collapse {
        let preview = first_n_chars(content, 480);
        let _ = writeln!(
            out,
            "          <details class=\"part part--prose-fold\"><summary class=\"prose-fold-bar\"><span class=\"prose-fold-tag\">prose</span><span class=\"prose-fold-size\">{}</span><span class=\"prose-fold-preview\">{}</span><span class=\"prose-fold-hint\">click to expand full text</span></summary>",
            format_count(total_chars as u64),
            escape(&preview)
        );
        out.push_str("            <div class=\"prose-body\">\n");
        write_prose_chunks(out, &chunks);
        out.push_str("            </div>\n");
        out.push_str("          </details>\n");
    } else {
        out.push_str("          <div class=\"part part--prose-wrap\">\n");
        write_prose_chunks(out, &chunks);
        out.push_str("          </div>\n");
    }
}

fn write_prose_chunks(out: &mut String, chunks: &[ProseChunk]) {
    for chunk in chunks {
        match chunk {
            ProseChunk::Text(text) => {
                if text.trim().is_empty() {
                    continue;
                }
                let _ = writeln!(
                    out,
                    "            <div class=\"part-prose-text\">{}</div>",
                    escape_breaks(text)
                );
            }
            ProseChunk::Code { lang, body } => {
                let len = body.chars().count();
                let label = if lang.is_empty() { "code" } else { lang };
                let _ = writeln!(
                    out,
                    "            <div class=\"part-prose-code\"><div class=\"code-bar\"><span class=\"code-tag\">{}</span><span class=\"code-size\">{}</span><button class=\"code-copy\" data-copy>copy</button></div><pre class=\"code-pre\">{}</pre></div>",
                    escape(label),
                    format_count(len as u64),
                    escape(body),
                );
            }
        }
    }
}

#[derive(Debug)]
enum ProseChunk<'a> {
    Text(&'a str),
    Code { lang: &'a str, body: &'a str },
}

fn split_prose_with_fences(input: &str) -> Vec<ProseChunk<'_>> {
    let mut out = Vec::new();
    let mut cursor = 0;
    while cursor < input.len() {
        // look for "\n```" or input starting with "```"
        let rest = &input[cursor..];
        let fence_at = if rest.starts_with("```") {
            Some(0)
        } else {
            rest.find("\n```").map(|p| p + 1)
        };
        let Some(rel) = fence_at else {
            out.push(ProseChunk::Text(rest));
            break;
        };
        let abs = cursor + rel;
        if abs > cursor {
            out.push(ProseChunk::Text(&input[cursor..abs]));
        }
        // parse fence: lang line then body until next ``` on its own line
        let after_open = abs + 3;
        let nl = input[after_open..].find('\n');
        let Some(nl_rel) = nl else {
            // unterminated fence — render rest as text
            out.push(ProseChunk::Text(&input[abs..]));
            break;
        };
        let lang = input[after_open..after_open + nl_rel].trim();
        let body_start = after_open + nl_rel + 1;
        // find closing ``` on its own line
        let close = find_closing_fence(&input[body_start..]);
        match close {
            Some((body_end_rel, after_close_rel)) => {
                let body = &input[body_start..body_start + body_end_rel];
                out.push(ProseChunk::Code { lang, body });
                cursor = body_start + after_close_rel;
            }
            None => {
                // unterminated — treat rest as code
                let body = &input[body_start..];
                out.push(ProseChunk::Code { lang, body });
                break;
            }
        }
    }
    out
}

/// Returns (offset of last char before closing fence, offset just past closing fence's newline).
fn find_closing_fence(s: &str) -> Option<(usize, usize)> {
    // closing fence is ``` at start of a line
    let mut search_from = 0;
    loop {
        let rel = s[search_from..].find("```")?;
        let abs = search_from + rel;
        let at_line_start = abs == 0 || s.as_bytes()[abs - 1] == b'\n';
        if at_line_start {
            // body ends at abs (exclusive of preceding newline if present)
            let body_end = if abs > 0 { abs - 1 } else { abs };
            // skip past the ``` and any trailing-line content + newline
            let after = abs + 3;
            let after_nl = s[after..]
                .find('\n')
                .map(|p| after + p + 1)
                .unwrap_or(s.len());
            return Some((body_end, after_nl));
        }
        search_from = abs + 3;
    }
}

fn first_n_chars(s: &str, n: usize) -> String {
    let mut out = String::with_capacity(n.min(s.len()));
    for ch in s.chars().take(n) {
        out.push(ch);
    }
    if s.chars().count() > n {
        out.push('…');
    }
    out
}

fn pick_display_title(session: &LoadedSession, name: &str, id: &str) -> String {
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
                if let Some(text) = message.display_user_text() {
                    if !text.trim().is_empty() {
                        return one_line_summary(text, 110);
                    }
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

fn call_id_short(call_id: &str) -> String {
    // Strip a `call_` prefix the LLM SDKs add, then keep first 8 chars.
    let trimmed = call_id
        .strip_prefix("call_")
        .or_else(|| call_id.strip_prefix("toolu_"))
        .unwrap_or(call_id);
    let head: String = trimmed.chars().take(8).collect();
    if trimmed.chars().count() > 8 {
        format!("{head}…")
    } else {
        head
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

fn render_reasoning(out: &mut String, content: &str) {
    if content.trim().is_empty() {
        return;
    }
    out.push_str("          <div class=\"part part--reasoning\">\n");
    out.push_str("            <span class=\"reasoning-gutter\">┊</span>\n");
    let _ = writeln!(
        out,
        "            <div class=\"reasoning-text\">{}</div>",
        escape_breaks(content)
    );
    out.push_str("          </div>\n");
}

fn render_code(out: &mut String, class: &str, content: &str, badge: Option<&str>) {
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

// ─── tool call entry ────────────────────────────────────────────────────────

fn render_tool_call_entry(
    out: &mut String,
    spine: &mut String,
    ctx: &mut RenderCtx<'_>,
    record: &ToolCallRecord,
    parent: Option<&str>,
) {
    let id = ctx.next_id();
    let (status_key, status_label) = if record.success {
        ("ok", "ok")
    } else {
        ("err", "error")
    };
    let glyph = if record.success { "•" } else { "×" };
    let summary = summarize_args(&record.args);
    let result_size = json_byte_size(&record.result);
    let args_size = json_byte_size(&record.args);

    let dur = format_duration(record.duration_ms);
    let parent_attr = parent
        .map(|p| format!(" data-parent=\"{}\"", escape_attr(p)))
        .unwrap_or_default();
    let parent_class = if parent.is_some() {
        " entry--child"
    } else {
        ""
    };

    let mut search = String::new();
    search.push_str(&record.tool.to_lowercase());
    search.push('\n');
    search.push_str(&summary.to_lowercase());
    search.push('\n');
    search.push_str(&pretty_json(&record.args).to_lowercase());
    search.push('\n');
    search.push_str(&pretty_json(&record.result).to_lowercase());

    let _ = writeln!(
        out,
        "    <article class=\"entry entry--tool entry--{status_key}{parent_class}\" id=\"{id}\" data-role=\"tool\" data-kind=\"tool_call\" data-tool=\"{tool}\" data-status=\"{status_key}\" data-search=\"{search}\"{parent_attr}>",
        tool = escape_attr(&record.tool),
        search = escape_attr(&search),
    );
    out.push_str("      <div class=\"entry-rail\">\n");
    let _ = writeln!(
        out,
        "        <a class=\"entry-num\" href=\"#{id}\" title=\"permalink\">{id}</a>"
    );
    let _ = writeln!(out, "        <span class=\"entry-glyph\">{glyph}</span>");
    out.push_str("      </div>\n");
    out.push_str("      <div class=\"entry-body\">\n");
    out.push_str("        <header class=\"entry-head\">\n");
    let _ = writeln!(
        out,
        "          <span class=\"entry-tag entry-tag--tool\">{}</span>",
        escape(&record.tool)
    );
    let _ = writeln!(
        out,
        "          <span class=\"entry-headline\">{}</span>",
        escape(&summary)
    );
    let _ = writeln!(
        out,
        "          <span class=\"entry-meta entry-meta--{status_key}\">{status_label} · {dur}</span>"
    );
    let _ = writeln!(
        out,
        "          <span class=\"entry-meta\">{}</span>",
        format_count(result_size as u64)
    );
    if let Some(call_id) = record.call_id.as_deref() {
        // Show only the leading 8 chars; full id available on hover and
        // remains in the data-search payload so / search still finds it.
        let short = call_id_short(call_id);
        let _ = writeln!(
            out,
            "          <span class=\"entry-callid\" title=\"call_id: {full}\">{short}</span>",
            full = escape_attr(call_id),
            short = escape(&short),
        );
    }
    out.push_str("        </header>\n");

    let auto_open = !record.success || (args_size + result_size) <= 4096;
    let open_attr = if auto_open { " open" } else { "" };
    let _ = writeln!(
        out,
        "        <details class=\"entry-content tool-details\"{open_attr}>"
    );
    out.push_str("          <summary class=\"tool-summary\">arguments + result</summary>\n");

    let args_str = pretty_json(&record.args);
    let result_str = pretty_json(&record.result);

    out.push_str("          <div class=\"kv\">\n");
    out.push_str("            <div class=\"kv-head\"><span class=\"kv-tag\">arguments</span>");
    let _ = writeln!(
        out,
        "<span class=\"kv-size\">{}</span><button class=\"code-copy\" data-copy>copy</button></div>",
        format_count(args_size as u64)
    );
    let _ = writeln!(
        out,
        "            <pre class=\"json\">{}</pre>",
        json_highlight(&args_str)
    );
    out.push_str("          </div>\n");

    out.push_str("          <div class=\"kv\">\n");
    out.push_str("            <div class=\"kv-head\"><span class=\"kv-tag\">result</span>");
    let _ = writeln!(
        out,
        "<span class=\"kv-size\">{}</span><button class=\"code-copy\" data-copy>copy</button></div>",
        format_count(result_size as u64)
    );
    let result_class = if record.success {
        "json"
    } else {
        "json json--err"
    };
    let _ = writeln!(
        out,
        "            <pre class=\"{result_class}\">{}</pre>",
        json_highlight(&result_str)
    );
    out.push_str("          </div>\n");

    out.push_str("        </details>\n");
    out.push_str("      </div>\n");
    out.push_str("    </article>\n");

    let _ = writeln!(
        spine,
        "    <a class=\"spine-tick spine-tick--{status_key}\" href=\"#{id}\" data-spine=\"tool\" data-status=\"{status_key}\" title=\"{tool} · {status_label} · {dur}\"></a>",
        tool = escape_attr(&record.tool),
    );
}

// ─── RLM step ───────────────────────────────────────────────────────────────

fn render_rlm_step(
    out: &mut String,
    spine: &mut String,
    ctx: &mut RenderCtx<'_>,
    step: &RlmTrajectoryEntry,
) {
    let id = ctx.next_id();
    let has_err = step.error.is_some();
    let status_key = if has_err { "err" } else { "ok" };
    let nested_calls = step.tool_calls.len();
    let output_preview = one_line_summary(&step.output, 200);
    let mut search = String::new();
    search.push_str(&step.reasoning.to_lowercase());
    search.push('\n');
    search.push_str(&step.code.to_lowercase());
    search.push('\n');
    search.push_str(&step.output.to_lowercase());
    if let Some(e) = &step.error {
        search.push('\n');
        search.push_str(&e.to_lowercase());
    }
    let total_chars = step.output.chars().count() + step.code.chars().count();

    let _ = writeln!(
        out,
        "    <article class=\"entry entry--rlm entry--{status_key}\" id=\"{id}\" data-role=\"rlm\" data-kind=\"rlm_step\" data-status=\"{status_key}\" data-search=\"{search}\">",
        search = escape_attr(&search)
    );
    out.push_str("      <div class=\"entry-rail\">\n");
    let _ = writeln!(
        out,
        "        <a class=\"entry-num\" href=\"#{id}\" title=\"permalink\">{id}</a>"
    );
    out.push_str("        <span class=\"entry-glyph\">◆</span>\n");
    out.push_str("      </div>\n");
    out.push_str("      <div class=\"entry-body\">\n");
    out.push_str("        <header class=\"entry-head\">\n");
    let _ = writeln!(
        out,
        "          <span class=\"entry-tag entry-tag--rlm\">RLM step {}</span>",
        step.iteration
    );
    let _ = writeln!(
        out,
        "          <span class=\"entry-headline\">{}</span>",
        escape(&output_preview)
    );
    if has_err {
        out.push_str("          <span class=\"entry-meta entry-meta--err\">error</span>\n");
    }
    if nested_calls > 0 {
        let _ = writeln!(
            out,
            "          <span class=\"entry-meta\">{} calls</span>",
            nested_calls
        );
    }
    let _ = writeln!(
        out,
        "          <span class=\"entry-meta\">{}</span>",
        format_count(total_chars as u64)
    );
    out.push_str("        </header>\n");
    out.push_str("        <div class=\"entry-content\">\n");

    let reasoning = strip_first_lashlang_fence(&step.reasoning);
    if !reasoning.trim().is_empty() {
        render_reasoning(out, &reasoning);
    }
    if !step.code.is_empty() {
        render_code(out, "code", &step.code, Some("lashlang"));
    }
    for observation in &step.observations {
        render_code(out, "output", observation, Some("observation"));
    }
    if !step.output.is_empty() {
        render_code(out, "output", &step.output, Some("output"));
    }
    if let Some(error) = &step.error {
        render_code(out, "error", error, Some("error"));
    }
    if let Some(final_output) = &step.final_output {
        let pretty = pretty_json(final_output);
        let _ = writeln!(
            out,
            "          <div class=\"part part--final\"><div class=\"code-bar\"><span class=\"code-tag code-tag--final\">final_output</span><span class=\"code-size\">{}</span><button class=\"code-copy\" data-copy>copy</button></div><pre class=\"json\">{}</pre></div>",
            format_count(json_byte_size(final_output) as u64),
            json_highlight(&pretty),
        );
    }

    out.push_str("        </div>\n");
    out.push_str("      </div>\n");
    out.push_str("    </article>\n");

    let _ = writeln!(
        spine,
        "    <a class=\"spine-tick spine-tick--rlm\" href=\"#{id}\" data-spine=\"rlm\" data-status=\"{status_key}\" title=\"RLM step {it}{nested}\"></a>",
        it = step.iteration,
        nested = if nested_calls > 0 {
            format!(" · {nested_calls} calls")
        } else {
            String::new()
        },
    );

    // Render nested tool calls as child entries.
    for record in &step.tool_calls {
        let key = lash::chronological_tool_call_key(record);
        if !ctx.seen_tool_keys.insert(key) {
            continue;
        }
        render_tool_call_entry(out, spine, ctx, record, Some(&id));
    }
}

// ─── helpers ────────────────────────────────────────────────────────────────

fn one_line_summary(text: &str, max_chars: usize) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return "(empty)".to_string();
    }
    let first_line: String = trimmed
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or(trimmed)
        .trim()
        .to_string();
    truncate(&first_line, max_chars)
}

fn truncate(s: &str, max_chars: usize) -> String {
    let count = s.chars().count();
    if count <= max_chars {
        return s.to_string();
    }
    let head: String = s.chars().take(max_chars.saturating_sub(1)).collect();
    format!("{head}…")
}

fn summarize_args(value: &serde_json::Value) -> String {
    use serde_json::Value;
    match value {
        Value::String(s) => format!("\"{}\"", truncate(s, 160)),
        Value::Null => "—".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::Array(arr) => {
            let preview: Vec<String> = arr.iter().take(3).map(|v| short_value(v, 40)).collect();
            let more = if arr.len() > 3 {
                format!(", … +{}", arr.len() - 3)
            } else {
                String::new()
            };
            format!("[{}{}]", preview.join(", "), more)
        }
        Value::Object(map) => {
            // priority keys to surface first
            const PRIORITY: &[&str] = &[
                "path", "file", "filename", "filepath", "command", "cmd", "shell", "url", "uri",
                "query", "q", "prompt", "name", "id", "key", "title",
            ];
            for k in PRIORITY {
                if let Some(v) = map.get(*k) {
                    return format!("{}={}", k, short_value(v, 160));
                }
            }
            // fall back to first key=value
            if let Some((k, v)) = map.iter().next() {
                if map.len() == 1 {
                    return format!("{}={}", k, short_value(v, 160));
                }
                let extra = map.len() - 1;
                return format!("{}={} +{} more", k, short_value(v, 100), extra);
            }
            "(empty)".to_string()
        }
    }
}

fn short_value(v: &serde_json::Value, max_chars: usize) -> String {
    use serde_json::Value;
    let raw = match v {
        Value::String(s) => format!("\"{s}\""),
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::Array(a) => format!("[…{} items]", a.len()),
        Value::Object(o) => format!("{{…{} keys}}", o.len()),
    };
    truncate(&raw.replace('\n', " "), max_chars)
}

fn json_byte_size(value: &serde_json::Value) -> usize {
    serde_json::to_string(value).map(|s| s.len()).unwrap_or(0)
}

fn pretty_json(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(text) => text.clone(),
        _ => serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string()),
    }
}

fn format_duration(ms: u64) -> String {
    if ms < 1000 {
        format!("{ms}ms")
    } else if ms < 60_000 {
        format!("{:.2}s", ms as f64 / 1000.0)
    } else {
        let total_s = ms / 1000;
        let m = total_s / 60;
        let s = total_s % 60;
        format!("{m}m{s:02}s")
    }
}

fn format_count(n: u64) -> String {
    if n < 1024 {
        format!("{n}b")
    } else if n < 1024 * 1024 {
        format!("{:.1}kb", n as f64 / 1024.0)
    } else if n < 1024 * 1024 * 1024 {
        format!("{:.1}mb", n as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2}gb", n as f64 / (1024.0 * 1024.0 * 1024.0))
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

/// Escape and preserve newlines. Used for prose content where line breaks
/// matter but raw HTML must not pass through. We keep `\n` as `\n` and rely
/// on `white-space: pre-wrap` in CSS, which preserves both newlines and
/// soft-wraps long lines without dropping double-newline paragraph breaks.
fn escape_breaks(input: &str) -> String {
    escape(input)
}

// ─── tiny json syntax highlighter ────────────────────────────────────────────
// Walks the already-pretty-printed JSON character-by-character. Cheap, no
// regex, safe for arbitrary input. Output is HTML with span classes:
//   .j-key, .j-str, .j-num, .j-bool, .j-null, .j-punct.
fn json_highlight(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + s.len() / 8);
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        match c {
            b'"' => {
                // find end of string (respect escapes)
                let start = i;
                i += 1;
                while i < bytes.len() {
                    if bytes[i] == b'\\' && i + 1 < bytes.len() {
                        i += 2;
                        continue;
                    }
                    if bytes[i] == b'"' {
                        i += 1;
                        break;
                    }
                    i += 1;
                }
                let raw = &s[start..i];
                // a key is a string immediately followed (after possible
                // whitespace) by ':'. peek ahead to decide the class.
                let mut j = i;
                while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b'\t') {
                    j += 1;
                }
                let class = if j < bytes.len() && bytes[j] == b':' {
                    "j-key"
                } else {
                    "j-str"
                };
                let _ = write!(out, "<span class=\"{class}\">{}</span>", escape(raw));
            }
            b'{' | b'}' | b'[' | b']' | b',' | b':' => {
                let _ = write!(
                    out,
                    "<span class=\"j-punct\">{}</span>",
                    escape(&(c as char).to_string())
                );
                i += 1;
            }
            b't' | b'f' if matches_at(bytes, i, b"true") || matches_at(bytes, i, b"false") => {
                let len = if matches_at(bytes, i, b"true") { 4 } else { 5 };
                let _ = write!(
                    out,
                    "<span class=\"j-bool\">{}</span>",
                    escape(&s[i..i + len])
                );
                i += len;
            }
            b'n' if matches_at(bytes, i, b"null") => {
                out.push_str("<span class=\"j-null\">null</span>");
                i += 4;
            }
            b'-' | b'0'..=b'9' => {
                let start = i;
                if bytes[i] == b'-' {
                    i += 1;
                }
                while i < bytes.len()
                    && (bytes[i].is_ascii_digit()
                        || bytes[i] == b'.'
                        || bytes[i] == b'e'
                        || bytes[i] == b'E'
                        || bytes[i] == b'+'
                        || bytes[i] == b'-')
                {
                    i += 1;
                }
                let _ = write!(out, "<span class=\"j-num\">{}</span>", escape(&s[start..i]));
            }
            b'\n' => {
                out.push('\n');
                i += 1;
            }
            _ => {
                // pass through (whitespace, etc.) — escape just in case
                let ch = s[i..].chars().next().unwrap_or(' ');
                let len = ch.len_utf8();
                out.push_str(&escape(&ch.to_string()));
                i += len;
            }
        }
    }
    out
}

fn matches_at(bytes: &[u8], i: usize, needle: &[u8]) -> bool {
    bytes.len() >= i + needle.len() && &bytes[i..i + needle.len()] == needle
}

// ─── CSS ────────────────────────────────────────────────────────────────────

const CSS: &str = include_str!("html_assets/style.css");
const JS: &str = include_str!("html_assets/script.js");

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

    #[test]
    fn tool_calls_are_deduped_between_message_part_and_chronological_entry() {
        // The chronological projection emits both an assistant Message
        // containing a ToolCall part and a separate ToolCall entry. The
        // canonical record (with result + duration) should render once.
        use lash::session_model::{Part, PartKind, PruneState, shared_parts};

        let tool_part = Part {
            id: "m0.p0".to_string(),
            kind: PartKind::ToolCall,
            content: r#"{"q":"x"}"#.to_string(),
            attachment: None,
            tool_call_id: Some("call_1".to_string()),
            tool_name: Some("lookup".to_string()),
            tool_item_id: None,
            tool_signature: None,
            prune_state: PruneState::Intact,
            reasoning_meta: None,
            response_meta: None,
        };
        let assistant_msg = lash::session_model::Message {
            id: "m0".to_string(),
            role: lash::session_model::MessageRole::Assistant,
            parts: shared_parts(vec![tool_part]),
            user_input: None,
            origin: None,
        };
        let session = LoadedSession {
            meta: None,
            chronological: vec![
                ChronologicalEntry {
                    index: 0,
                    payload: ChronologicalPayload::Message(assistant_msg),
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
        // A single canonical record should appear: count occurrences of the
        // tool name in entry-tag positions.
        let count = rendered.matches("entry-tag entry-tag--tool").count();
        assert_eq!(
            count, 1,
            "expected exactly one tool-call entry, got {count}\n{rendered}"
        );
        // And the result must be present (only the canonical entry has it).
        // The JSON highlighter wraps strings in spans and escapes quotes,
        // so look for the bare key text — proves the result block rendered.
        assert!(rendered.contains("answer"), "result missing");
    }
}
