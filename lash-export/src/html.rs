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
//! Tool calls are rendered from the chronological projection only. Assistant
//! `Message` parts that describe tool calls are suppressed because the
//! projection's tool-call payloads are the canonical view (args + result +
//! duration + status).

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Write as _;

use lash::session_model::{Message, MessageRole, Part, PartKind, PruneState};
use lash::{ChronologicalPayload, ToolCallRecord};
use lash_rlm_types::RlmTrajectoryEntry;

use crate::LoadedSession;
use crate::trace::{LlmCallUsage, LlmPromptSnapshot};

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
    write_usage_overview(&mut out, session);
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
    llm_calls_with_usage: usize,
    input_tokens: i64,
    output_tokens: i64,
    cached_input_tokens: i64,
    reasoning_tokens: i64,
    max_context_percent: Option<f64>,
    tool_freq: Vec<(String, usize)>,
    tool_names_set: Vec<String>,
}

fn compute_stats(session: &LoadedSession) -> SessionStats {
    let mut s = SessionStats {
        chronological: session.chronological.len(),
        ..SessionStats::default()
    };
    let mut tool_counts: HashMap<String, usize> = HashMap::new();

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
                record_tool_call(&mut s, &mut tool_counts, record);
            }
            ChronologicalPayload::RlmStep(step) => {
                s.rlm_iterations += 1;
                if step.error.is_some() {
                    s.rlm_errors += 1;
                }
                s.total_chars = s.total_chars.saturating_add(step.output_chars());
                s.total_chars = s.total_chars.saturating_add(step.code.chars().count());
                for record in &step.tool_calls {
                    record_tool_call(&mut s, &mut tool_counts, record);
                }
            }
        }
    }

    let mut freq: Vec<(String, usize)> = tool_counts.into_iter().collect();
    freq.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    s.tool_names_set = freq.iter().map(|(name, _)| name.clone()).collect();
    s.tool_freq = freq;

    for prompt in &session.llm_prompts {
        let Some(usage) = &prompt.usage else {
            continue;
        };
        s.llm_calls_with_usage += 1;
        s.input_tokens = s.input_tokens.saturating_add(usage.input_tokens);
        s.output_tokens = s.output_tokens.saturating_add(usage.output_tokens);
        s.cached_input_tokens = s
            .cached_input_tokens
            .saturating_add(usage.cached_input_tokens);
        s.reasoning_tokens = s.reasoning_tokens.saturating_add(usage.reasoning_tokens);
        if let Some(context_window) = session.context_window_tokens
            && context_window > 0
        {
            let pct = usage.input_tokens.max(0) as f64 * 100.0 / context_window as f64;
            s.max_context_percent = Some(s.max_context_percent.map_or(pct, |max| max.max(pct)));
        }
    }
    s
}

// ─── render context ─────────────────────────────────────────────────────────

struct RenderCtx<'a> {
    next_index: usize,
    stats: &'a SessionStats,
}

impl<'a> RenderCtx<'a> {
    fn new(stats: &'a SessionStats) -> Self {
        Self {
            next_index: 0,
            stats,
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
    let _ = writeln!(
        out,
        "      <span class=\"meta-row\"><span class=\"meta-key\">trace</span><span class=\"meta-val\">{}</span></span>",
        escape(&session.trace_path.display().to_string())
    );
    let _ = writeln!(
        out,
        "      <span class=\"meta-row\"><span class=\"meta-key\">llm prompts</span><span class=\"meta-val\">{}</span></span>",
        session.llm_prompts.len()
    );
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
    if stats.llm_calls_with_usage > 0 {
        let cached_pct = percent_of(stats.cached_input_tokens, stats.input_tokens)
            .map(|pct| format!("{pct:.1}% cached"))
            .unwrap_or_else(|| "cache n/a".to_string());
        let context_label = stats
            .max_context_percent
            .map(|pct| format!("max ctx {pct:.1}%"))
            .unwrap_or_else(|| "ctx n/a".to_string());
        let _ = writeln!(
            out,
            "  <div class=\"stat\"><span class=\"stat-key\">tokens</span><span class=\"stat-val\">{}</span><span class=\"stat-sub\">{} out · {cached_pct} · {context_label}</span></div>",
            format_tokens(stats.input_tokens),
            format_tokens(stats.output_tokens),
        );
    }
    if stats.pruned_parts + stats.cleared_parts > 0 {
        let _ = writeln!(
            out,
            "  <div class=\"stat stat--muted\"><span class=\"stat-key\">pruned</span><span class=\"stat-val\">{}</span><span class=\"stat-sub\">{} cleared</span></div>",
            stats.pruned_parts, stats.cleared_parts,
        );
    }
    out.push_str("</section>\n");
}

fn write_usage_overview(out: &mut String, session: &LoadedSession) {
    if session.llm_prompts.is_empty() {
        return;
    }
    out.push_str("<section class=\"usage-overview\" aria-label=\"LLM usage overview\">\n");
    out.push_str("  <div class=\"usage-overview-label\">context</div>\n");
    out.push_str("  <div class=\"usage-overview-bars\">\n");
    for (idx, prompt) in session.llm_prompts.iter().enumerate() {
        let id = prompt
            .mode_iteration
            .map(|i| format!("iter {i}"))
            .unwrap_or_else(|| format!("call {idx}"));
        let Some(usage) = prompt.usage.as_ref() else {
            let _ = writeln!(
                out,
                "    <a class=\"usage-overview-bar usage-overview-bar--missing\" href=\"#\" title=\"{}\"><span></span></a>",
                escape_attr(&format!("{id} · token usage not recorded")),
            );
            continue;
        };
        let context_pct = context_percent(usage, session.context_window_tokens);
        let width = context_pct
            .map(|pct| pct.clamp(0.0, 100.0).max(2.0))
            .unwrap_or(100.0);
        let level_class = match context_pct {
            Some(pct) if pct >= 90.0 => " usage-overview-bar--critical",
            Some(pct) if pct >= 75.0 => " usage-overview-bar--hot",
            Some(pct) if pct >= 50.0 => " usage-overview-bar--warm",
            Some(_) => "",
            None => " usage-overview-bar--unknown",
        };
        // The body renderer will assign deterministic entry ids in prompt
        // order, but other chronological entries are interleaved. Use JS to
        // bind overview bars to the final usage rows after render.
        let _ = writeln!(
            out,
            "    <a class=\"usage-overview-bar{level_class}\" href=\"#\" data-usage-index=\"{idx}\" title=\"{title}\"><span style=\"--usage-width: {width:.3}%\"></span></a>",
            title = escape_attr(&format!(
                "{id} · {}",
                usage_title(Some(usage), session.context_window_tokens)
            )),
        );
    }
    out.push_str("  </div>\n");
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
        ("prompt", "prompt"),
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
    out.push_str("  <aside class=\"usage-chart\" aria-label=\"LLM token usage chart\">\n");
    out.push_str(&entries_html.usage_chart);
    out.push_str("  </aside>\n");
    out.push_str("</div>\n");
}

struct EntriesHtml {
    entries: String,
    spine: String,
    usage_chart: String,
}

fn render_entries(session: &LoadedSession, ctx: &mut RenderCtx<'_>) -> EntriesHtml {
    let mut entries = String::with_capacity(32 * 1024);
    let mut spine = String::with_capacity(2 * 1024);
    let mut usage_chart = String::with_capacity(2 * 1024);

    let insertions = compute_prompt_insertions(&session.chronological, &session.llm_prompts);
    let mut last_hash: Option<String> = None;
    let mut first_seen: HashMap<String, PromptAnchor> = HashMap::new();

    // Identify assistant messages that just echo the prior RlmStep's submit
    // value — they're a runtime side-effect of `submit`, not a separate
    // model utterance. Suppressing avoids showing the same text twice.
    let mut suppressed_message_ids: HashSet<String> = HashSet::new();
    let mut last_final_output: Option<String> = None;
    for entry in session.chronological.iter() {
        match &entry.payload {
            ChronologicalPayload::RlmStep(step) => {
                last_final_output = step.final_output.as_ref().map(submit_value_text);
            }
            ChronologicalPayload::Message(message) => {
                if matches!(message.role, MessageRole::Assistant)
                    && let Some(prev) = last_final_output.as_deref()
                    && message_matches_text(message, prev)
                {
                    suppressed_message_ids.insert(message.id.clone());
                }
                last_final_output = None;
            }
            ChronologicalPayload::ToolCall(_) => {}
        }
    }

    let emit_prompt = |entries: &mut String,
                       spine: &mut String,
                       ctx: &mut RenderCtx<'_>,
                       last_hash: &mut Option<String>,
                       first_seen: &mut HashMap<String, PromptAnchor>,
                       usage_chart: &mut String,
                       prompt_idx: usize| {
        let prompt = &session.llm_prompts[prompt_idx];
        let anchor = first_seen.get(&prompt.system_hash).cloned();
        let id = render_system_prompt(
            entries,
            spine,
            ctx,
            prompt,
            session.context_window_tokens,
            last_hash.as_deref(),
            anchor.as_ref(),
        );
        write_usage_chart_bar(usage_chart, &id, prompt, session.context_window_tokens);
        first_seen
            .entry(prompt.system_hash.clone())
            .or_insert(PromptAnchor {
                entry_id: id,
                iter_label: prompt
                    .mode_iteration
                    .map(|i| format!("iter {i}"))
                    .unwrap_or_else(|| "first call".to_string()),
            });
        *last_hash = Some(prompt.system_hash.clone());
    };

    for (i, entry) in session.chronological.iter().enumerate() {
        for &prompt_idx in &insertions.before_index[i] {
            emit_prompt(
                &mut entries,
                &mut spine,
                ctx,
                &mut last_hash,
                &mut first_seen,
                &mut usage_chart,
                prompt_idx,
            );
        }
        match &entry.payload {
            ChronologicalPayload::Message(message) => {
                if message.is_transient() || suppressed_message_ids.contains(&message.id) {
                    continue;
                }
                render_message(&mut entries, &mut spine, ctx, message);
            }
            ChronologicalPayload::ToolCall(record) => {
                render_tool_call_entry(&mut entries, &mut spine, ctx, record, None);
            }
            ChronologicalPayload::RlmStep(step) => {
                render_rlm_step(&mut entries, &mut spine, ctx, step);
            }
        }
    }

    for &prompt_idx in &insertions.trailing {
        emit_prompt(
            &mut entries,
            &mut spine,
            ctx,
            &mut last_hash,
            &mut first_seen,
            &mut usage_chart,
            prompt_idx,
        );
    }

    EntriesHtml {
        entries,
        spine,
        usage_chart,
    }
}

#[derive(Clone)]
struct PromptAnchor {
    entry_id: String,
    iter_label: String,
}

struct PromptInsertions {
    /// `before_index[i]` lists prompt indices that should render *before*
    /// the chronological entry at index `i`.
    before_index: Vec<Vec<usize>>,
    /// Prompts with no chronological anchor — rendered at the end so a
    /// snapshot is never silently dropped.
    trailing: Vec<usize>,
}

/// Decide where each per-LLM-call system prompt belongs in the rendered
/// timeline.
///
/// Each LLM call kicks off one mode iteration. In RLM mode the chronological
/// projection emits everything that mode iteration produced — its tool calls
/// (children of the lashlang block) followed by the `RlmStep` trajectory
/// record — as a contiguous run. We want the system prompt to appear at
/// the *start* of that run, before the tool calls, since that's when the
/// model received it.
///
/// Standard mode has no `RlmStep` entries, so we positionally bind each
/// prompt to the matching assistant message instead.
fn compute_prompt_insertions(
    chronological: &[lash::ChronologicalEntry],
    prompts: &[LlmPromptSnapshot],
) -> PromptInsertions {
    let mut before_index = vec![Vec::new(); chronological.len()];
    let mut consumed = vec![false; prompts.len()];

    let has_rlm_steps = chronological
        .iter()
        .any(|e| matches!(e.payload, ChronologicalPayload::RlmStep(_)));

    if has_rlm_steps {
        let mut by_mode_iteration: HashMap<u64, VecDeque<usize>> = HashMap::new();
        for (idx, prompt) in prompts.iter().enumerate() {
            if let Some(mode_iteration) = prompt.mode_iteration {
                by_mode_iteration
                    .entry(mode_iteration)
                    .or_default()
                    .push_back(idx);
            }
        }

        // Mode iteration 0's prompt was sent to the model alongside the
        // initial user message, so anchor it to that user message
        // rather than to the mode iteration's first tool/rlm output —
        // otherwise the user message would appear *above* the prompt
        // that contextualised it.
        let mut initial_user_anchor_mode_iteration = None;
        let first_user_idx = chronological.iter().position(|e| {
            matches!(&e.payload, ChronologicalPayload::Message(m)
                if matches!(m.role, MessageRole::User) && !m.is_transient())
        });
        if let (Some(idx), Some(queue)) = (first_user_idx, by_mode_iteration.get_mut(&0))
            && let Some(pi) = queue.pop_front()
            && !consumed[pi]
        {
            before_index[idx].push(pi);
            consumed[pi] = true;
            initial_user_anchor_mode_iteration = Some(0);
        }

        let mut run_start: Option<usize> = None;
        for (i, entry) in chronological.iter().enumerate() {
            match &entry.payload {
                ChronologicalPayload::ToolCall(_) => {
                    if run_start.is_none() {
                        run_start = Some(i);
                    }
                }
                ChronologicalPayload::RlmStep(step) => {
                    let begin = run_start.unwrap_or(i);
                    let mode_iteration = step.mode_iteration as u64;
                    if initial_user_anchor_mode_iteration == Some(mode_iteration) {
                        initial_user_anchor_mode_iteration = None;
                    } else if let Some(queue) = by_mode_iteration.get_mut(&mode_iteration) {
                        while let Some(pi) = queue.pop_front() {
                            if consumed[pi] {
                                continue;
                            }
                            before_index[begin].push(pi);
                            consumed[pi] = true;
                            break;
                        }
                    }
                    run_start = None;
                }
                ChronologicalPayload::Message(_) => {
                    // Non-RLM messages (the initial user prompt or a
                    // trailing assistant message) don't open a new
                    // iteration — they sit between or after RLM runs.
                    run_start = None;
                }
            }
        }
    } else {
        // Standard mode: positionally bind to assistant messages.
        let mut cursor = 0usize;
        for (i, entry) in chronological.iter().enumerate() {
            if let ChronologicalPayload::Message(message) = &entry.payload
                && matches!(message.role, MessageRole::Assistant)
                && !message.is_transient()
            {
                while cursor < prompts.len() && consumed[cursor] {
                    cursor += 1;
                }
                if cursor < prompts.len() {
                    before_index[i].push(cursor);
                    consumed[cursor] = true;
                    cursor += 1;
                }
            }
        }
    }

    let trailing: Vec<usize> = (0..prompts.len()).filter(|i| !consumed[*i]).collect();
    PromptInsertions {
        before_index,
        trailing,
    }
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

/// Render a prose body as markdown. Long bodies are folded inside a
/// `<details>` so a multi-kilobyte chunk doesn't dominate the timeline —
/// the expanded form is one click away.
fn render_prose(out: &mut String, content: &str) {
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
                if let Some(text) = message.display_user_text()
                    && !text.trim().is_empty()
                {
                    return one_line_summary(text, 110);
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
    // Auto-open errors always; auto-open small calls only when the session
    // is short. A 500-call trace at ~5kb each becomes a 200k-pixel document
    // if everything's open by default — collapse-first, expand-on-demand
    // is the right default for sessions of that scale.
    let busy_session = ctx.stats.tool_calls_ok + ctx.stats.tool_calls_err > 30;
    let auto_open = !record.success || (!busy_session && (args_size + result_size) <= 4096);
    let open_attr = if auto_open { " open" } else { "" };

    // The whole header row IS the summary — click anywhere on it to expand.
    out.push_str("      <div class=\"entry-body\">\n");
    let _ = writeln!(
        out,
        "        <details class=\"entry-content tool-details\"{open_attr}>"
    );
    out.push_str("          <summary class=\"entry-head\">\n");
    let _ = writeln!(
        out,
        "            <span class=\"entry-tag entry-tag--tool\">{}</span>",
        escape(&record.tool)
    );
    let _ = writeln!(
        out,
        "            <span class=\"entry-headline\">{}</span>",
        escape(&summary)
    );
    let _ = writeln!(
        out,
        "            <span class=\"entry-meta entry-meta--{status_key}\">{status_label} · {dur}</span>"
    );
    let _ = writeln!(
        out,
        "            <span class=\"entry-meta\">{}</span>",
        format_count(result_size as u64)
    );
    if let Some(call_id) = record.call_id.as_deref() {
        let short = call_id_short(call_id);
        let _ = writeln!(
            out,
            "            <span class=\"entry-callid\" title=\"call_id: {full}\">{short}</span>",
            full = escape_attr(call_id),
            short = escape(&short),
        );
    }
    out.push_str("          </summary>\n");

    render_tool_call_payload(out, record);

    out.push_str("        </details>\n");
    out.push_str("      </div>\n");
    out.push_str("    </article>\n");

    let _ = writeln!(
        spine,
        "    <a class=\"spine-tick spine-tick--{status_key}\" href=\"#{id}\" data-spine=\"tool\" data-status=\"{status_key}\" title=\"{tool} · {status_label} · {dur}\"></a>",
        tool = escape_attr(&record.tool),
    );
}

fn render_tool_call_payload(out: &mut String, record: &ToolCallRecord) {
    let args_size = json_byte_size(&record.args);
    let result_size = json_byte_size(&record.result);
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
    let reasoning = strip_first_lashlang_fence(&step.reasoning);
    let has_reasoning_body = !reasoning.trim().is_empty();
    let output_preview = if has_reasoning_body {
        one_line_summary(&reasoning, 200)
    } else if let Some(out_item) = step.output.iter().find(|o| !o.trim().is_empty()) {
        one_line_summary(out_item, 200)
    } else {
        String::new()
    };
    let mut search = String::new();
    // Use the stripped reasoning so the lashlang body isn't duplicated —
    // `step.reasoning` still contains the fenced block, and `step.code` is
    // the same text extracted from it.
    search.push_str(&reasoning.to_lowercase());
    search.push('\n');
    search.push_str(&step.code.to_lowercase());
    for item in &step.output {
        search.push('\n');
        search.push_str(&item.to_lowercase());
    }
    if let Some(e) = &step.error {
        search.push('\n');
        search.push_str(&e.to_lowercase());
    }
    let total_chars = step.output_chars() + step.code.chars().count();

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
        step.mode_iteration
    );
    if has_reasoning_body {
        // Reasoning renders below as its own block — keep the layout
        // balanced without restating the same text in the head row.
        out.push_str("          <span class=\"entry-headline entry-headline--ghost\"></span>\n");
    } else {
        let _ = writeln!(
            out,
            "          <span class=\"entry-headline\">{}</span>",
            escape(&output_preview)
        );
    }
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

    if has_reasoning_body {
        render_reasoning(out, &reasoning);
    }
    if !step.code.is_empty() {
        render_code(out, "code", &step.code, Some("lashlang"));
    }
    for item in &step.output {
        render_code(out, "output", item, Some("output"));
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
    render_inline_rlm_tool_calls(out, step);

    out.push_str("        </div>\n");
    out.push_str("      </div>\n");
    out.push_str("    </article>\n");

    let _ = writeln!(
        spine,
        "    <a class=\"spine-tick spine-tick--rlm\" href=\"#{id}\" data-spine=\"rlm\" data-status=\"{status_key}\" title=\"RLM step {it}{nested}\"></a>",
        it = step.mode_iteration,
        nested = if nested_calls > 0 {
            format!(" · {nested_calls} calls")
        } else {
            String::new()
        },
    );

    // Render nested tool calls as child entries.
    for record in &step.tool_calls {
        render_tool_call_entry(out, spine, ctx, record, Some(&id));
    }
}

fn render_inline_rlm_tool_calls(out: &mut String, step: &RlmTrajectoryEntry) {
    if step.tool_calls.is_empty() {
        return;
    }
    let has_error = step.tool_calls.iter().any(|record| !record.success);
    let open_attr = if has_error { " open" } else { "" };
    let total_ms: u64 = step
        .tool_calls
        .iter()
        .map(|record| record.duration_ms)
        .sum();
    let _ = writeln!(
        out,
        "          <details class=\"rlm-tool-list\"{open_attr}>"
    );
    let _ = writeln!(
        out,
        "            <summary class=\"code-bar\"><span class=\"code-tag\">tool calls</span><span class=\"code-size\">{} · {}</span></summary>",
        step.tool_calls.len(),
        format_duration(total_ms),
    );
    out.push_str("            <div class=\"rlm-tool-stack\">\n");
    for (idx, record) in step.tool_calls.iter().enumerate() {
        render_inline_tool_call(out, idx + 1, record);
    }
    out.push_str("            </div>\n");
    out.push_str("          </details>\n");
}

fn render_inline_tool_call(out: &mut String, ordinal: usize, record: &ToolCallRecord) {
    let (status_key, status_label) = if record.success {
        ("ok", "ok")
    } else {
        ("err", "error")
    };
    let summary = summarize_args(&record.args);
    let result_size = json_byte_size(&record.result);
    let open_attr = if record.success { "" } else { " open" };
    let _ = writeln!(
        out,
        "              <details class=\"tool-details rlm-tool-details rlm-tool-details--{status_key}\"{open_attr}>"
    );
    out.push_str("                <summary class=\"tool-summary\">\n");
    let _ = writeln!(
        out,
        "                  <span class=\"entry-tag entry-tag--tool\">{}</span>",
        escape(&record.tool)
    );
    let _ = writeln!(
        out,
        "                  <span class=\"entry-headline\">#{ordinal} · {}</span>",
        escape(&summary)
    );
    let _ = writeln!(
        out,
        "                  <span class=\"entry-meta entry-meta--{status_key}\">{status_label} · {}</span>",
        format_duration(record.duration_ms)
    );
    let _ = writeln!(
        out,
        "                  <span class=\"entry-meta\">{}</span>",
        format_count(result_size as u64)
    );
    if let Some(call_id) = record.call_id.as_deref() {
        let short = call_id_short(call_id);
        let _ = writeln!(
            out,
            "                  <span class=\"entry-callid\" title=\"call_id: {full}\">{short}</span>",
            full = escape_attr(call_id),
            short = escape(&short),
        );
    }
    out.push_str("                </summary>\n");
    render_tool_call_payload(out, record);
    out.push_str("              </details>\n");
}

// ─── system prompt (per-LLM-call snapshot from trace) ───────────────────────

fn render_system_prompt(
    out: &mut String,
    spine: &mut String,
    ctx: &mut RenderCtx<'_>,
    prompt: &LlmPromptSnapshot,
    context_window_tokens: Option<u64>,
    prev_hash: Option<&str>,
    first_seen: Option<&PromptAnchor>,
) -> String {
    let id = ctx.next_id();
    let repeat_of = prev_hash.filter(|h| *h == prompt.system_hash.as_str());
    let iter_label = prompt
        .mode_iteration
        .map(|i| format!("step {i}"))
        .unwrap_or_else(|| "llm call".to_string());
    let model_label = match (prompt.model.as_deref(), prompt.model_variant.as_deref()) {
        (Some(m), Some(v)) if !v.is_empty() => format!("{m}/{v}"),
        (Some(m), _) => m.to_string(),
        _ => String::new(),
    };
    let hash_short = prompt.system_hash.chars().take(12).collect::<String>();
    let chars_label = format_count(prompt.system_chars as u64);
    let total_label = format_count(prompt.total_chars as u64);
    // Show the system block size in the headline; the right-aligned
    // entry-meta carries the *full* prompt size (system + history + new
    // user message) — that's what was actually shipped to the wire.
    let request_size_label = if prompt.total_chars > prompt.system_chars {
        format!("→ {total_label}")
    } else {
        total_label.clone()
    };
    let usage_label = prompt
        .usage
        .as_ref()
        .map(|usage| compact_usage_label(usage, context_window_tokens));
    let context_label = prompt
        .usage
        .as_ref()
        .and_then(|usage| context_percent_label(usage, context_window_tokens));

    let mut headline = String::new();
    headline.push_str(&iter_label);
    if !model_label.is_empty() {
        headline.push_str(" · ");
        headline.push_str(&model_label);
    }
    headline.push_str(" · system ");
    headline.push_str(&chars_label);
    if repeat_of.is_some() {
        headline.push_str(" · unchanged since previous call");
    }

    // Keep search text short — the prompt body is typically 30+kb and
    // duplicating it into `data-search` would balloon page weight without
    // helping anyone (the body is one click away in the <details>).
    let mut search = String::new();
    search.push_str("system prompt ");
    search.push_str(&iter_label);
    if let Some(call_id) = prompt.llm_call_id.as_deref() {
        search.push(' ');
        search.push_str(call_id);
    }
    search.push(' ');
    search.push_str(&hash_short);

    let repeat_class = if repeat_of.is_some() {
        " entry--system-repeat"
    } else {
        ""
    };

    let _ = writeln!(
        out,
        "    <article class=\"entry entry--system{repeat_class}\" id=\"{id}\" data-role=\"prompt\" data-kind=\"system_prompt\" data-search=\"{search}\">",
        search = escape_attr(&search)
    );
    out.push_str("      <div class=\"entry-rail\">\n");
    let _ = writeln!(
        out,
        "        <a class=\"entry-num\" href=\"#{id}\" title=\"permalink\">{id}</a>"
    );
    out.push_str("        <span class=\"entry-glyph\">◇</span>\n");
    out.push_str("      </div>\n");
    out.push_str("      <div class=\"entry-body\">\n");

    if repeat_of.is_some() {
        out.push_str("        <header class=\"entry-head\">\n");
        out.push_str(
            "          <span class=\"entry-tag entry-tag--system\">system prompt</span>\n",
        );
        let _ = writeln!(
            out,
            "          <span class=\"entry-headline\">{}</span>",
            escape(&headline)
        );
        if let Some(anchor) = first_seen {
            let _ = writeln!(
                out,
                "          <a class=\"entry-meta entry-meta--repeat-link\" href=\"#{anchor_id}\" title=\"jump to full prompt body at {anchor_label}\">view full at {anchor_label} →</a>",
                anchor_id = escape_attr(&anchor.entry_id),
                anchor_label = escape(&anchor.iter_label),
            );
        }
        let _ = writeln!(
            out,
            "          <span class=\"entry-meta entry-meta--repeat\" title=\"hash {full}\">↺ {short}</span>",
            full = escape_attr(&prompt.system_hash),
            short = escape(&hash_short),
        );
        let _ = writeln!(
            out,
            "          <span class=\"entry-meta\" title=\"total request size: system + history + user\">{}</span>",
            escape(&request_size_label)
        );
        if let Some(label) = &usage_label {
            let _ = writeln!(
                out,
                "          <span class=\"entry-meta entry-meta--usage\" title=\"{}\">{}</span>",
                escape_attr(&usage_title(prompt.usage.as_ref(), context_window_tokens)),
                escape(label)
            );
        }
        out.push_str("        </header>\n");
    } else {
        out.push_str("        <details class=\"entry-content prompt-details\">\n");
        out.push_str("          <summary class=\"entry-head\">\n");
        out.push_str(
            "            <span class=\"entry-tag entry-tag--system\">system prompt</span>\n",
        );
        let _ = writeln!(
            out,
            "            <span class=\"entry-headline\">{}</span>",
            escape(&headline)
        );
        let _ = writeln!(
            out,
            "            <span class=\"entry-meta\" title=\"sha hash of system text\">{}</span>",
            escape(&hash_short)
        );
        let _ = writeln!(
            out,
            "            <span class=\"entry-meta\" title=\"total request size: system + history + user\">{}</span>",
            escape(&request_size_label)
        );
        if let Some(label) = &usage_label {
            let _ = writeln!(
                out,
                "            <span class=\"entry-meta entry-meta--usage\" title=\"{}\">{}</span>",
                escape_attr(&usage_title(prompt.usage.as_ref(), context_window_tokens)),
                escape(label)
            );
        }
        out.push_str("          </summary>\n");
        out.push_str(
            "          <div class=\"prompt-body\"><div class=\"code-bar\"><span class=\"code-tag\">system</span>",
        );
        let _ = writeln!(
            out,
            "<span class=\"code-size\">{}</span><button class=\"code-copy\" data-copy>copy</button></div>",
            chars_label
        );
        out.push_str("          <div class=\"prompt-pre\">");
        out.push_str(&crate::markdown::render(&prompt.system_text));
        out.push_str("</div></div>\n");
        out.push_str("        </details>\n");
    }

    // Always render the request body — that's what changed between
    // calls. In RLM mode the runtime synthesises one growing user
    // message containing the original task plus serialised history;
    // surfacing it here is the only way to see what the model actually
    // received per call (the .db doesn't store it — the user message
    // shown in the timeline is just the original task).
    if !prompt.request_messages.is_empty() {
        let request_chars_label = format_count(prompt.request_chars as u64);
        out.push_str("        <details class=\"entry-content prompt-details prompt-request\">\n");
        out.push_str("          <summary class=\"entry-head\">\n");
        out.push_str(
            "            <span class=\"entry-tag entry-tag--request\">request body</span>\n",
        );
        let _ = writeln!(
            out,
            "            <span class=\"entry-headline\">{} non-system message{} sent at this call</span>",
            prompt.request_messages.len(),
            if prompt.request_messages.len() == 1 {
                ""
            } else {
                "s"
            }
        );
        let _ = writeln!(
            out,
            "            <span class=\"entry-meta\" title=\"sha hash of request body\">{}</span>",
            escape(&prompt.request_hash.chars().take(12).collect::<String>())
        );
        let _ = writeln!(
            out,
            "            <span class=\"entry-meta\">{}</span>",
            escape(&request_chars_label)
        );
        out.push_str("          </summary>\n");
        out.push_str("          <div class=\"prompt-body request-messages\">\n");
        for (idx, msg) in prompt.request_messages.iter().enumerate() {
            let chars_label = format_count(msg.chars as u64);
            let role_lc = msg.role.to_lowercase();
            let render_as_markdown = matches!(role_lc.as_str(), "user" | "assistant" | "system");
            let _ = writeln!(
                out,
                "            <div class=\"request-message\"><div class=\"code-bar\"><span class=\"code-tag code-tag--{role}\">{role}</span><span class=\"code-meta\">message {idx} of {total}</span><span class=\"code-size\">{size}</span><button class=\"code-copy\" data-copy>copy</button></div>",
                role = escape(&msg.role),
                idx = idx,
                total = prompt.request_messages.len(),
                size = chars_label,
            );
            if render_as_markdown {
                out.push_str("<div class=\"prompt-pre\">");
                out.push_str(&crate::markdown::render(&msg.text));
                out.push_str("</div></div>\n");
            } else {
                let _ = writeln!(
                    out,
                    "<pre class=\"code-pre\">{body}</pre></div>",
                    body = escape(&msg.text),
                );
            }
        }
        out.push_str("          </div>\n");
        out.push_str("        </details>\n");
    }

    out.push_str("      </div>\n");
    out.push_str("    </article>\n");

    let title = if repeat_of.is_some() {
        format!("system prompt · {iter_label} · unchanged")
    } else {
        format!("system prompt · {iter_label} · {chars_label}")
    };
    let title = if let Some(context_label) = context_label {
        format!("{title} · {context_label}")
    } else {
        title
    };
    let _ = writeln!(
        spine,
        "    <a class=\"spine-tick\" href=\"#{id}\" data-spine=\"prompt\" title=\"{}\"></a>",
        escape_attr(&title)
    );

    id
}

// ─── helpers ────────────────────────────────────────────────────────────────

fn submit_value_text(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn write_usage_chart_bar(
    out: &mut String,
    entry_id: &str,
    prompt: &LlmPromptSnapshot,
    context_window_tokens: Option<u64>,
) {
    let iter_label = prompt
        .mode_iteration
        .map(|i| i.to_string())
        .unwrap_or_else(|| "llm".to_string());
    let Some(usage) = prompt.usage.as_ref() else {
        let _ = writeln!(
            out,
            "    <a class=\"usage-row usage-row--missing\" href=\"#{id}\" title=\"{title}\"><span class=\"usage-row-label\">{iter}</span><span class=\"usage-track\"><span class=\"usage-bar\"></span></span><span class=\"usage-row-pct\">n/a</span></a>",
            id = escape_attr(entry_id),
            title = escape_attr(&format!("iter {iter_label} · token usage not recorded")),
            iter = escape(&iter_label),
        );
        return;
    };

    let context_pct = context_percent(usage, context_window_tokens);
    let cached_pct = percent_of(usage.cached_input_tokens, usage.input_tokens);
    let width = context_pct
        .map(|pct| {
            let clamped = pct.clamp(0.0, 100.0);
            if clamped > 0.0 { clamped.max(2.0) } else { 1.0 }
        })
        .unwrap_or_else(|| cached_pct.map(|pct| pct.clamp(2.0, 100.0)).unwrap_or(1.0));
    let level_class = match context_pct {
        Some(pct) if pct >= 90.0 => " usage-row--critical",
        Some(pct) if pct >= 75.0 => " usage-row--hot",
        Some(pct) if pct >= 50.0 => " usage-row--warm",
        Some(_) => "",
        None => " usage-row--unknown",
    };
    let pct_label = context_pct
        .map(|pct| format!("{pct:.1}%"))
        .unwrap_or_else(|| "ctx n/a".to_string());
    let title = usage_title(Some(usage), context_window_tokens);
    let _ = writeln!(
        out,
        "    <a class=\"usage-row{level_class}\" href=\"#{id}\" title=\"{title}\" data-usage-entry=\"{id}\" data-context-pct=\"{pct}\"><span class=\"usage-row-label\">{iter}</span><span class=\"usage-track\"><span class=\"usage-bar\" style=\"--usage-width: {width:.3}%\"></span></span><span class=\"usage-row-pct\">{pct_label}</span></a>",
        id = escape_attr(entry_id),
        title = escape_attr(&format!("iter {iter_label} · {title}")),
        pct = context_pct
            .map(|pct| format!("{pct:.6}"))
            .unwrap_or_default(),
        iter = escape(&iter_label),
        pct_label = escape(&pct_label),
    );
}

fn message_matches_text(message: &Message, expected: &str) -> bool {
    let expected = expected.trim();
    if expected.is_empty() {
        return false;
    }
    let collected: String = message
        .parts
        .iter()
        .filter(|p| matches!(p.kind, PartKind::Text | PartKind::Prose))
        .map(|p| p.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    collected.trim() == expected
}

fn compact_usage_label(usage: &LlmCallUsage, context_window_tokens: Option<u64>) -> String {
    let context = context_percent_label(usage, context_window_tokens)
        .unwrap_or_else(|| "ctx n/a".to_string());
    let cached = percent_of(usage.cached_input_tokens, usage.input_tokens)
        .map(|pct| format!("{pct:.1}% cached"))
        .unwrap_or_else(|| "cache n/a".to_string());
    format!("{context} · {cached}")
}

fn context_percent_label(
    usage: &LlmCallUsage,
    context_window_tokens: Option<u64>,
) -> Option<String> {
    context_percent(usage, context_window_tokens).map(|pct| format!("ctx {pct:.1}%"))
}

fn usage_title(usage: Option<&LlmCallUsage>, context_window_tokens: Option<u64>) -> String {
    let Some(usage) = usage else {
        return "token usage not recorded".to_string();
    };
    let context = match (
        context_percent(usage, context_window_tokens),
        context_window_tokens,
    ) {
        (Some(pct), Some(window)) => format!(
            "context {pct:.3}% ({input} / {window})",
            input = usage.input_tokens.max(0),
        ),
        _ => format!(
            "context n/a ({input} input)",
            input = usage.input_tokens.max(0)
        ),
    };
    let cached = percent_of(usage.cached_input_tokens, usage.input_tokens)
        .map(|pct| {
            format!(
                "cached {pct:.3}% ({cached} / {input})",
                cached = usage.cached_input_tokens.max(0),
                input = usage.input_tokens.max(0),
            )
        })
        .unwrap_or_else(|| "cached n/a".to_string());
    let duration = usage
        .duration_ms
        .map(|ms| format!(" · duration {}", format_duration(ms)))
        .unwrap_or_default();
    format!(
        "{context} · {cached} · input {input} · output {output} · reasoning {reasoning}{duration}",
        input = usage.input_tokens.max(0),
        output = usage.output_tokens.max(0),
        reasoning = usage.reasoning_tokens.max(0),
    )
}

fn context_percent(usage: &LlmCallUsage, context_window_tokens: Option<u64>) -> Option<f64> {
    let window = context_window_tokens?;
    if window == 0 {
        return None;
    }
    Some(usage.input_tokens.max(0) as f64 * 100.0 / window as f64)
}

fn percent_of(numerator: i64, denominator: i64) -> Option<f64> {
    if denominator <= 0 {
        return None;
    }
    Some(numerator.max(0) as f64 * 100.0 / denominator as f64)
}

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
            // Priority keys to surface first — action-shaped keys (the
            // WHAT) outrank location keys (the WHERE) so a grep call shows
            // its query instead of its path. Tools without an action key
            // fall through to path naturally.
            const PRIORITY: &[&str] = &[
                "query", "q", "command", "cmd", "shell", "url", "uri", "prompt", "path", "file",
                "filename", "filepath", "name", "title", "id", "key",
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

fn format_tokens(n: i64) -> String {
    let n = n.max(0) as f64;
    if n < 1_000.0 {
        format!("{}", n as u64)
    } else if n < 1_000_000.0 {
        format!("{:.1}k", n / 1_000.0)
    } else {
        format!("{:.2}m", n / 1_000_000.0)
    }
}

fn strip_first_lashlang_fence(text: &str) -> String {
    let Some(open_rel) = text.find("```") else {
        return text.to_string();
    };
    let opener_len = text.as_bytes()[open_rel..]
        .iter()
        .take_while(|&&b| b == b'`')
        .count();
    let after_open = open_rel + opener_len;
    let rest = &text[after_open..];
    let Some(lang_end_rel) = rest.find('\n') else {
        return text[..open_rel].to_string();
    };
    let lang = rest[..lang_end_rel].trim();
    if !matches!(lang, "lashlang" | "rlm" | "lash") {
        return text.to_string();
    }
    let body_start = after_open + lang_end_rel + 1;
    let body_bytes = &text.as_bytes()[body_start..];
    let mut close = text.len();
    let mut consumed = 0usize;
    let mut i = 0;
    while i < body_bytes.len() {
        if body_bytes[i] == b'`' {
            let start = i;
            while i < body_bytes.len() && body_bytes[i] == b'`' {
                i += 1;
            }
            if i - start >= opener_len {
                close = body_start + start;
                consumed = opener_len;
                break;
            }
        } else {
            i += 1;
        }
    }
    let after_close = (close + consumed).min(text.len());
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
    use crate::trace::{LlmCallUsage, LlmPromptSnapshot, RequestMessage};
    use lash::session_model::{Part, PartKind, PruneState, shared_parts};
    use lash::{ChronologicalEntry, ChronologicalPayload, ToolCallRecord};
    use std::path::PathBuf;

    fn prompt_snapshot(mode_iteration: u64, text: &str) -> LlmPromptSnapshot {
        LlmPromptSnapshot {
            turn_index: Some(1),
            mode_iteration: Some(mode_iteration),
            llm_call_id: Some(format!("root:1:{mode_iteration}:0")),
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

    fn user_message(id: &str, text: &str) -> lash::session_model::Message {
        lash::session_model::Message {
            id: id.to_string(),
            role: lash::session_model::MessageRole::User,
            parts: shared_parts(vec![Part {
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
                response_meta: None,
            }]),
            user_input: None,
            origin: None,
        }
    }

    fn rlm_step(mode_iteration: usize, id: &str) -> RlmTrajectoryEntry {
        RlmTrajectoryEntry {
            id: id.to_string(),
            mode_iteration,
            reasoning: "thinking".to_string(),
            code: "x = 1".to_string(),
            output: vec!["1".to_string()],
            tool_calls: Vec::new(),
            images: Vec::new(),
            error: None,
            final_output: None,
        }
    }

    #[test]
    fn html_export_renders_chronological_tool_and_rlm_step() {
        let session = LoadedSession {
            meta: None,
            chronological: vec![
                ChronologicalEntry {
                    index: 0,
                    payload: ChronologicalPayload::RlmStep(rlm_step(0, "rlm_step_0")),
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
            trace_path: PathBuf::from("session.trace.jsonl"),
            context_window_tokens: None,
            llm_prompts: Vec::new(),
        };

        let rendered = render(&session);
        assert!(rendered.contains("RLM step 0"));
        assert!(rendered.contains("x = 1"));
        assert!(rendered.contains("lookup"));
        assert!(rendered.contains("chronological entries"));
    }

    #[test]
    fn repeated_rlm_trace_mode_iterations_are_anchored_in_prompt_order() {
        let chronological = vec![
            ChronologicalEntry {
                index: 0,
                payload: ChronologicalPayload::Message(user_message("m0", "turn 1")),
            },
            ChronologicalEntry {
                index: 1,
                payload: ChronologicalPayload::RlmStep(rlm_step(0, "rlm_step_0")),
            },
            ChronologicalEntry {
                index: 2,
                payload: ChronologicalPayload::Message(user_message("m1", "turn 2")),
            },
            ChronologicalEntry {
                index: 3,
                payload: ChronologicalPayload::RlmStep(rlm_step(0, "rlm_step_1")),
            },
            ChronologicalEntry {
                index: 4,
                payload: ChronologicalPayload::Message(user_message("m2", "turn 3")),
            },
            ChronologicalEntry {
                index: 5,
                payload: ChronologicalPayload::RlmStep(rlm_step(0, "rlm_step_2")),
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
    fn tool_calls_are_deduped_between_message_part_and_chronological_entry() {
        // The chronological projection emits both an assistant Message
        // containing a ToolCall part and a separate ToolCall entry. The
        // canonical record (with result + duration) should render once.
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
            trace_path: PathBuf::from("session.trace.jsonl"),
            context_window_tokens: None,
            llm_prompts: Vec::new(),
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

    #[test]
    fn rlm_step_renders_inline_expandable_tool_calls() {
        let session = LoadedSession {
            meta: None,
            chronological: vec![ChronologicalEntry {
                index: 0,
                payload: ChronologicalPayload::RlmStep(RlmTrajectoryEntry {
                    code: "data = (call lookup { q: \"x\" })?".to_string(),
                    output: Vec::new(),
                    tool_calls: vec![ToolCallRecord {
                        call_id: Some("call_1".to_string()),
                        tool: "lookup".to_string(),
                        args: serde_json::json!({"q": "x"}),
                        result: serde_json::json!({"answer": "y"}),
                        success: true,
                        duration_ms: 4,
                    }],
                    ..rlm_step(0, "rlm_step_0")
                }),
            }],
            trace_path: PathBuf::from("session.trace.jsonl"),
            context_window_tokens: None,
            llm_prompts: Vec::new(),
        };

        let rendered = render(&session);
        assert!(rendered.contains("rlm-tool-list"));
        assert!(rendered.contains("tool calls"));
        assert!(rendered.contains("lookup"));
        assert!(rendered.contains("answer"));
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
        assert!(rendered.contains("data-role=\"prompt\""));
        assert!(rendered.contains("data-kind=\"system_prompt\""));
        assert!(rendered.contains(">prompt</button>"));
        assert!(rendered.contains("usage-chart"));
        assert!(rendered.contains("ctx 10.0%"));
        assert!(rendered.contains("75.000%"));
    }
}
