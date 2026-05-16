use std::fmt::Write as _;

use crate::LoadedSession;

use super::assets::{CSS, JS};
use super::entries::{pick_display_title, render_entries};
use super::escaping::{escape, escape_attr};
use super::stats::{SessionStats, compute_stats};
use super::view_model::{
    RenderCtx, context_percent, format_count, format_duration, format_tokens, percent_of,
    usage_title,
};

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
    write_run_summary(&mut out, session, &stats);
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

pub(crate) fn write_hero(out: &mut String, session: &LoadedSession, _stats: &SessionStats) {
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

fn write_run_summary(out: &mut String, session: &LoadedSession, stats: &SessionStats) {
    let llm_calls = session.llm_prompts.len();
    let total_calls = stats.tool_calls_ok + stats.tool_calls_err + stats.tool_calls_cancelled;
    let top_tools = stats
        .tool_freq
        .iter()
        .take(3)
        .map(|(name, count)| format!("{name} × {count}"))
        .collect::<Vec<_>>()
        .join(", ");
    let cost_label = if stats.est_cost_usd >= 0.01 {
        format!("${:.2}", stats.est_cost_usd)
    } else if stats.est_cost_usd > 0.0 {
        format!("${:.4}", stats.est_cost_usd)
    } else {
        "n/a".to_string()
    };
    out.push_str("<section class=\"run-summary\" aria-label=\"run summary\">\n");
    let _ = writeln!(
        out,
        "  <span><span class=\"run-summary-key\">turns</span><span class=\"run-summary-val\">{}</span></span>",
        stats.rlm_iterations.max(stats.assistant_messages),
    );
    if !top_tools.is_empty() {
        let _ = writeln!(
            out,
            "  <span><span class=\"run-summary-key\">top tools</span><span class=\"run-summary-val\">{}</span></span>",
            escape(&top_tools),
        );
    }
    let _ = writeln!(
        out,
        "  <span><span class=\"run-summary-key\">tool calls</span><span class=\"run-summary-val\">{total_calls}</span></span>"
    );
    let _ = writeln!(
        out,
        "  <span><span class=\"run-summary-key\">llm calls</span><span class=\"run-summary-val\">{llm_calls}</span></span>"
    );
    let _ = writeln!(
        out,
        "  <span><span class=\"run-summary-key\">est cost</span><span class=\"run-summary-val run-summary-cost\" title=\"estimated from per-model pricing\">{cost_label}</span></span>",
    );
    out.push_str("</section>\n");
}

fn write_session_bar(out: &mut String, _session: &LoadedSession, stats: &SessionStats) {
    let total_msgs = stats.user_messages + stats.assistant_messages + stats.system_messages;
    let total_calls = stats.tool_calls_ok + stats.tool_calls_err + stats.tool_calls_cancelled;
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
    let err_sub_class = if stats.tool_calls_err > 0 || stats.tool_calls_cancelled > 0 {
        "stat-sub stat-sub--err"
    } else {
        "stat-sub"
    };
    let _ = writeln!(
        out,
        "  <div class=\"stat\"><span class=\"stat-key\">tool calls</span><span class=\"stat-val\">{total_calls}</span><span class=\"{err_sub_class}\">{ok} ok · {er} err · {ca} cancelled</span></div>",
        ok = stats.tool_calls_ok,
        er = stats.tool_calls_err,
        ca = stats.tool_calls_cancelled,
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

pub(crate) fn write_controls(out: &mut String, stats: &SessionStats) {
    out.push_str("<section class=\"controls\" aria-label=\"filters\">\n");

    out.push_str("  <div class=\"chip-row\" data-group=\"role\">\n");
    out.push_str("    <span class=\"chip-label\">show</span>\n");
    for (key, label) in [
        ("user", "user"),
        ("assistant", "assistant"),
        ("tool", "tool"),
        ("rlm", "rlm"),
        ("llm_call", "llm call"),
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
