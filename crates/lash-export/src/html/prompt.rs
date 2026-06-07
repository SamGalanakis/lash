use std::collections::{HashMap, VecDeque};
use std::fmt::Write as _;

use lash_core::ChronologicalPayload;
use lash_core::session_model::MessageRole;

use crate::trace::LlmPromptSnapshot;

use super::chronological_rlm_step;
use super::escaping::{escape, escape_attr};
use super::view_model::{
    RenderCtx, compact_usage_label, context_percent, context_percent_label, format_count,
    format_tokens, percent_of, usage_title,
};

#[derive(Clone)]
pub(crate) struct PromptAnchor {
    pub(crate) entry_id: String,
    pub(crate) iter_label: String,
}

pub(crate) struct PromptInsertions {
    /// `before_index[i]` lists prompt indices that should render *before*
    /// the chronological entry at index `i`.
    pub(crate) before_index: Vec<Vec<usize>>,
    /// Prompts with no chronological anchor — rendered at the end so a
    /// snapshot is never silently dropped.
    pub(crate) trailing: Vec<usize>,
}

/// Rendering disposition for a system-prompt entry when collapsing runs of
/// repeated identical hashes.
#[derive(Clone, Debug)]
pub(crate) enum CoalesceState {
    /// Render normally — either as the full system prompt or as a single
    /// "unchanged since previous call" stub. Default.
    Show,
    /// First repeat after the anchor: emit one banner row standing in for
    /// `run_count` suppressed siblings. `anchor_idx` points to the prompt
    /// that carries the full system text.
    BannerStart { run_count: usize, anchor_idx: usize },
    /// Suppress entirely — already represented by the banner.
    Suppress,
}

/// For each main-flow prompt, decide whether it renders normally, becomes a
/// banner, or is suppressed. Walks consecutive same-hash runs in
/// `main_indices` order and coalesces the tail when a run reaches
/// `min_run_len` (so `min_run_len = 3` means "≥ 2 stubs after the anchor
/// triggers coalescing").
pub(crate) fn compute_coalesce_states(
    prompts: &[LlmPromptSnapshot],
    main_indices: &[usize],
    min_run_len: usize,
) -> HashMap<usize, CoalesceState> {
    let mut out = HashMap::new();
    let mut i = 0;
    while i < main_indices.len() {
        let anchor = main_indices[i];
        let hash = &prompts[anchor].system_hash;
        let mut j = i + 1;
        while j < main_indices.len() && prompts[main_indices[j]].system_hash == *hash {
            j += 1;
        }
        let run_len = j - i;
        if run_len >= min_run_len {
            // i = anchor (rendered Show — full system text)
            // i+1 = banner start (replaces the first stub)
            // i+2..j = suppressed
            out.insert(anchor, CoalesceState::Show);
            out.insert(
                main_indices[i + 1],
                CoalesceState::BannerStart {
                    run_count: run_len - 1,
                    anchor_idx: anchor,
                },
            );
            for idx in main_indices.iter().take(j).skip(i + 2) {
                out.insert(*idx, CoalesceState::Suppress);
            }
        }
        // run_len < min_run_len → leave entries with implicit Show (default).
        i = j;
    }
    out
}

pub(crate) fn compute_prompt_insertions(
    chronological: &[lash_core::ChronologicalEntry],
    prompts: &[LlmPromptSnapshot],
) -> PromptInsertions {
    let mut before_index = vec![Vec::new(); chronological.len()];
    let mut consumed = vec![false; prompts.len()];

    let has_rlm_steps = chronological.iter().any(|e| {
        matches!(
            &e.payload,
            ChronologicalPayload::ProtocolEvent(event) if chronological_rlm_step(event).is_some()
        )
    });

    if has_rlm_steps {
        let mut by_protocol_iteration: HashMap<u64, VecDeque<usize>> = HashMap::new();
        for (idx, prompt) in prompts.iter().enumerate() {
            if let Some(protocol_iteration) = prompt.protocol_iteration {
                by_protocol_iteration
                    .entry(protocol_iteration)
                    .or_default()
                    .push_back(idx);
            }
        }

        // Mode iteration 0's prompt was sent to the model alongside the
        // initial user message, so anchor it to that user message
        // rather than to the mode iteration's first tool/rlm output —
        // otherwise the user message would appear *above* the prompt
        // that contextualised it.
        let mut initial_user_anchor_protocol_iteration = None;
        let first_user_idx = chronological.iter().position(|e| {
            matches!(&e.payload, ChronologicalPayload::Message(m)
                if matches!(m.role, MessageRole::User) && !m.is_transient())
        });
        if let (Some(idx), Some(queue)) = (first_user_idx, by_protocol_iteration.get_mut(&0))
            && let Some(pi) = queue.pop_front()
            && !consumed[pi]
        {
            before_index[idx].push(pi);
            consumed[pi] = true;
            initial_user_anchor_protocol_iteration = Some(0);
        }

        for (i, entry) in chronological.iter().enumerate() {
            match &entry.payload {
                ChronologicalPayload::ProtocolEvent(event) => {
                    let Some(step) = chronological_rlm_step(event) else {
                        continue;
                    };
                    let protocol_iteration = step.protocol_iteration as u64;
                    if initial_user_anchor_protocol_iteration == Some(protocol_iteration) {
                        initial_user_anchor_protocol_iteration = None;
                    } else if let Some(queue) = by_protocol_iteration.get_mut(&protocol_iteration) {
                        while let Some(pi) = queue.pop_front() {
                            if consumed[pi] {
                                continue;
                            }
                            before_index[i].push(pi);
                            consumed[pi] = true;
                            break;
                        }
                    }
                }
                ChronologicalPayload::Message(_) => {
                    // Non-RLM messages (the initial user prompt or a
                    // trailing assistant message) don't open a new
                    // iteration — they sit between or after RLM runs.
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

pub(crate) fn render_system_prompt_banner(
    out: &mut String,
    spine: &mut String,
    ctx: &mut RenderCtx,
    prompt: &LlmPromptSnapshot,
    run_count: usize,
    anchor: Option<&PromptAnchor>,
) -> String {
    let id = ctx.next_id();
    let hash_short = prompt.system_hash.chars().take(12).collect::<String>();
    let _ = writeln!(
        out,
        "    <article class=\"entry entry--system entry--system-coalesce\" id=\"{id}\" data-role=\"llm_call\" data-kind=\"system_prompt\" data-search=\"system prompt unchanged {short} ×{n}\">",
        short = escape_attr(&hash_short),
        n = run_count + 1,
    );
    out.push_str("      <div class=\"entry-rail\">\n");
    let _ = writeln!(
        out,
        "        <a class=\"entry-num\" href=\"#{id}\" title=\"permalink\">{id}</a>"
    );
    out.push_str("        <span class=\"entry-glyph\">⋯</span>\n");
    out.push_str("      </div>\n");
    out.push_str("      <div class=\"entry-body\">\n");
    out.push_str("        <div class=\"system-coalesce-banner\">\n");
    out.push_str("          <span class=\"entry-tag entry-tag--system\">system prompt</span>\n");
    let _ = writeln!(
        out,
        "          <span>unchanged across the next <span class=\"system-coalesce-banner-count\">{n}</span> calls</span>",
        n = run_count + 1,
    );
    if let Some(anchor) = anchor
        && !anchor.entry_id.is_empty()
    {
        let _ = writeln!(
            out,
            "          <a href=\"#{anchor_id}\" title=\"jump to full prompt body at {anchor_label}\">view full at {anchor_label} →</a>",
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
    out.push_str("        </div>\n");
    out.push_str("      </div>\n");
    out.push_str("    </article>\n");

    let _ = writeln!(
        spine,
        "    <a class=\"spine-tick\" href=\"#{id}\" data-spine=\"llm_call\" title=\"system prompt unchanged across {n} calls\"></a>",
        n = run_count + 1,
    );
    id
}

pub(crate) fn render_system_prompt(
    out: &mut String,
    spine: &mut String,
    ctx: &mut RenderCtx,
    prompt: &LlmPromptSnapshot,
    context_window_tokens: Option<u64>,
    prev_hash: Option<&str>,
    first_seen: Option<&PromptAnchor>,
) -> String {
    let id = ctx.next_id();
    let repeat_of = prev_hash.filter(|h| *h == prompt.system_hash.as_str());
    let iter_label = prompt
        .protocol_iteration
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
        "    <article class=\"entry entry--system{repeat_class}\" id=\"{id}\" data-role=\"llm_call\" data-kind=\"system_prompt\" data-search=\"{search}\">",
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

    // Reasoning-tokens-without-content: when the provider charged for
    // reasoning but didn't return reasoning content, surface the absence
    // explicitly instead of letting the user infer it from token math.
    if let Some(usage) = prompt.usage.as_ref()
        && usage.reasoning_tokens > 0
    {
        let _ = writeln!(
            out,
            "        <div class=\"reasoning-missing\" title=\"the provider returned a reasoning_tokens count but no reasoning content blocks\"><span class=\"reasoning-missing-tokens\">≈{} reasoning tokens</span> · content not returned by provider</div>",
            format_tokens(usage.reasoning_tokens),
        );
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
        "    <a class=\"spine-tick\" href=\"#{id}\" data-spine=\"llm_call\" title=\"{}\"></a>",
        escape_attr(&title)
    );

    id
}

// ─── helpers ────────────────────────────────────────────────────────────────

pub(crate) fn write_usage_chart_bar(
    out: &mut String,
    entry_id: &str,
    prompt: &LlmPromptSnapshot,
    context_window_tokens: Option<u64>,
) {
    let iter_label = prompt
        .protocol_iteration
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
