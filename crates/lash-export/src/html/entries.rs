use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use lash_core::session_model::{Message, MessageRole, Part, PartKind, PruneState};
use lash_core::{ChronologicalPayload, ToolCallRecord};
use lash_rlm_types::RlmTrajectoryEntry;

use crate::LoadedSession;
use crate::trace::LlmPromptSnapshot;

use super::chronological_rlm_step;
use super::escaping::{escape, escape_attr, escape_breaks, json_highlight};
use super::prompt::{
    CoalesceState, PromptAnchor, compute_coalesce_states, compute_prompt_insertions,
    render_system_prompt, render_system_prompt_banner, write_usage_chart_bar,
};
use super::view_model::{
    RenderCtx, format_count, format_duration, format_tokens, json_byte_size, message_matches_text,
    one_line_summary, pretty_json, strip_first_lashlang_fence, submit_value_text, summarize_args,
    truncate,
};

pub(crate) struct EntriesHtml {
    pub(crate) entries: String,
    pub(crate) spine: String,
    pub(crate) usage_chart: String,
}

pub(crate) fn render_entries(session: &LoadedSession, ctx: &mut RenderCtx<'_>) -> EntriesHtml {
    let mut entries = String::with_capacity(32 * 1024);
    let mut spine = String::with_capacity(2 * 1024);
    let mut usage_chart = String::with_capacity(2 * 1024);

    let insertions = compute_prompt_insertions(&session.chronological, &session.llm_prompts);
    let mut last_hash: Option<String> = None;
    let mut first_seen: HashMap<String, PromptAnchor> = HashMap::new();

    // Build fan-out index: any LLM call that carries an originating tool
    // call id is rendered as a child of that tool entry, NOT as a peer
    // in the chronological flow. This collapses tournament_rerank-style
    // batches under their parent tool.
    let mut fanout_index: HashMap<String, Vec<usize>> = HashMap::new();
    let mut fanout_consumed: HashSet<usize> = HashSet::new();
    for (idx, prompt) in session.llm_prompts.iter().enumerate() {
        if let Some(tcid) = &prompt.originating_tool_call_id {
            fanout_index.entry(tcid.clone()).or_default().push(idx);
            fanout_consumed.insert(idx);
        }
    }

    // Coalesce runs of consecutive identical-system-hash main-flow prompts
    // into one banner instead of N stub rows.
    let main_indices: Vec<usize> = (0..session.llm_prompts.len())
        .filter(|i| !fanout_consumed.contains(i))
        .collect();
    let coalesce: HashMap<usize, CoalesceState> = compute_coalesce_states(
        &session.llm_prompts,
        &main_indices,
        /* threshold (run length) = */ 3,
    );

    // Identify assistant messages that just echo the prior RlmStep's submit
    // value — they're a runtime side-effect of `submit`, not a separate
    // model utterance. Suppressing avoids showing the same text twice.
    let mut suppressed_message_ids: HashSet<String> = HashSet::new();
    let mut last_final_output: Option<String> = None;
    for entry in session.chronological.iter() {
        match &entry.payload {
            ChronologicalPayload::ProtocolEvent(event) => {
                last_final_output = chronological_rlm_step(event)
                    .and_then(|step| step.final_output.map(|value| submit_value_text(&value)));
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

    let tool_call_map = session
        .chronological
        .iter()
        .filter_map(|entry| match &entry.payload {
            ChronologicalPayload::ToolCall(record) => record
                .call_id
                .as_ref()
                .map(|call_id| (call_id.clone(), record)),
            ChronologicalPayload::Message(_) | ChronologicalPayload::ProtocolEvent(_) => None,
        })
        .collect::<HashMap<_, _>>();
    let rlm_owned_tool_call_ids = session
        .chronological
        .iter()
        .filter_map(|entry| match &entry.payload {
            ChronologicalPayload::ProtocolEvent(event) => chronological_rlm_step(event),
            ChronologicalPayload::Message(_) | ChronologicalPayload::ToolCall(_) => None,
        })
        .flat_map(|step| step.tool_call_ids)
        .collect::<HashSet<_>>();

    let emit_prompt = |entries: &mut String,
                       spine: &mut String,
                       ctx: &mut RenderCtx<'_>,
                       last_hash: &mut Option<String>,
                       first_seen: &mut HashMap<String, PromptAnchor>,
                       usage_chart: &mut String,
                       prompt_idx: usize| {
        if fanout_consumed.contains(&prompt_idx) {
            // Rendered under its parent tool call below, not in the main flow.
            return;
        }
        let prompt = &session.llm_prompts[prompt_idx];
        match coalesce.get(&prompt_idx) {
            Some(CoalesceState::BannerStart {
                run_count,
                anchor_idx,
            }) => {
                let anchor = first_seen.get(&prompt.system_hash).cloned().or_else(|| {
                    // fallback: derive from the prompt at anchor_idx
                    let anchor_prompt = &session.llm_prompts[*anchor_idx];
                    Some(PromptAnchor {
                        entry_id: String::new(),
                        iter_label: anchor_prompt
                            .protocol_iteration
                            .map(|i| format!("iter {i}"))
                            .unwrap_or_else(|| "first call".to_string()),
                    })
                });
                let id = render_system_prompt_banner(
                    entries,
                    spine,
                    ctx,
                    prompt,
                    *run_count,
                    anchor.as_ref(),
                );
                // Don't emit individual usage bars for suppressed siblings —
                // they would have nowhere to anchor. Keep the banner one
                // anchored to the banner row.
                write_usage_chart_bar(usage_chart, &id, prompt, session.context_window_tokens);
                *last_hash = Some(prompt.system_hash.clone());
            }
            Some(CoalesceState::Suppress) => {
                // Skip transcript entirely. Do not advance last_hash so the
                // post-run prompt still sees the same hash and can render
                // a stub if needed.
            }
            Some(CoalesceState::Show) | None => {
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
                            .protocol_iteration
                            .map(|i| format!("iter {i}"))
                            .unwrap_or_else(|| "first call".to_string()),
                    });
                *last_hash = Some(prompt.system_hash.clone());
            }
        }
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
                if record
                    .call_id
                    .as_ref()
                    .is_some_and(|call_id| rlm_owned_tool_call_ids.contains(call_id))
                {
                    continue;
                }
                render_tool_call_entry(&mut entries, &mut spine, ctx, record, None);
                if let Some(call_id) = record.call_id.as_deref()
                    && let Some(children) = fanout_index.get(call_id)
                {
                    render_tool_fanout(
                        &mut entries,
                        &mut spine,
                        ctx,
                        record,
                        &session.llm_prompts,
                        children,
                        session.context_window_tokens,
                    );
                }
            }
            ChronologicalPayload::ProtocolEvent(event) => {
                let Some(step) = chronological_rlm_step(event) else {
                    continue;
                };
                render_rlm_step_with_fanout(
                    &mut entries,
                    &mut spine,
                    ctx,
                    RlmStepFanoutInput {
                        step: &step,
                        tool_call_map: &tool_call_map,
                        fanout_index: &fanout_index,
                        prompts: &session.llm_prompts,
                        context_window_tokens: session.context_window_tokens,
                    },
                );
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

fn render_tool_fanout(
    out: &mut String,
    spine: &mut String,
    ctx: &mut RenderCtx<'_>,
    record: &ToolCallRecord,
    prompts: &[LlmPromptSnapshot],
    children: &[usize],
    context_window_tokens: Option<u64>,
) {
    if children.is_empty() {
        return;
    }
    let id = ctx.next_id();
    let n = children.len();
    let mut input = 0i64;
    let mut output = 0i64;
    let mut cached = 0i64;
    let mut reasoning = 0i64;
    let mut ok = 0usize;
    let mut errs = 0usize;
    for &idx in children {
        let p = &prompts[idx];
        if let Some(u) = &p.usage {
            input += u.input_tokens.max(0);
            output += u.output_tokens.max(0);
            cached += u.cached_input_tokens.max(0);
            reasoning += u.reasoning_tokens.max(0);
            ok += 1;
        } else {
            errs += 1;
        }
    }
    let parent_attr = record
        .call_id
        .as_deref()
        .map(|p| format!(" data-parent=\"{}\"", escape_attr(p)))
        .unwrap_or_default();
    let _ = writeln!(
        out,
        "    <article class=\"entry entry--child entry--fanout\" id=\"{id}\" data-role=\"llm_call\" data-kind=\"tool_fanout\" data-tool=\"{tool}\"{parent_attr}>",
        tool = escape_attr(&record.tool),
    );
    out.push_str("      <div class=\"entry-rail\"><span class=\"entry-glyph\">├─</span></div>\n");
    out.push_str("      <div class=\"entry-body\">\n");
    out.push_str("        <details class=\"tool-fanout\">\n");
    let _ = writeln!(
        out,
        "          <summary class=\"tool-fanout-summary\"><span class=\"entry-tag entry-tag--system\">fan-out</span><span><span class=\"tool-fanout-count\">{n}</span> direct llm calls · {in_k} in / {out_k} out · reasoning {r_k} · {ok}/{n} ok</span></summary>",
        in_k = format_tokens(input),
        out_k = format_tokens(output),
        r_k = format_tokens(reasoning),
    );
    let _ = ok; // consumed in summary above
    let _ = errs;
    let _ = cached;
    out.push_str("          <div class=\"tool-fanout-children\">\n");
    let mut inner_usage = String::new();
    let mut inner_spine = String::new();
    let mut inner_last_hash: Option<String> = None;
    let mut inner_first_seen: HashMap<String, PromptAnchor> = HashMap::new();
    for &idx in children {
        let prompt = &prompts[idx];
        let anchor = inner_first_seen.get(&prompt.system_hash).cloned();
        let child_id = render_system_prompt(
            out,
            &mut inner_spine,
            ctx,
            prompt,
            context_window_tokens,
            inner_last_hash.as_deref(),
            anchor.as_ref(),
        );
        write_usage_chart_bar(&mut inner_usage, &child_id, prompt, context_window_tokens);
        inner_first_seen
            .entry(prompt.system_hash.clone())
            .or_insert(PromptAnchor {
                entry_id: child_id,
                iter_label: prompt
                    .protocol_iteration
                    .map(|i| format!("iter {i}"))
                    .unwrap_or_else(|| "fan-out".to_string()),
            });
        inner_last_hash = Some(prompt.system_hash.clone());
    }
    out.push_str("          </div>\n");
    out.push_str("        </details>\n");
    out.push_str("      </div>\n");
    out.push_str("    </article>\n");
    // The fan-out summary tick replaces N peer ticks — one summary tick on
    // the spine per fan-out (instead of 25 noisy ones).
    let _ = writeln!(
        spine,
        "    <a class=\"spine-tick\" href=\"#{id}\" data-spine=\"llm_call\" title=\"{tool} fan-out · {n} direct calls\"></a>",
        tool = escape_attr(&record.tool),
    );
}

/// Wraps render_rlm_step so its inline tool calls also receive fan-out
/// children (RLM mode invokes tools from inside a lashlang block; if any
/// of those tools issue direct_completion calls, those should fold under
/// the inline tool entry, not appear as flat peers).
struct RlmStepFanoutInput<'a> {
    step: &'a RlmTrajectoryEntry,
    tool_call_map: &'a HashMap<String, &'a ToolCallRecord>,
    fanout_index: &'a HashMap<String, Vec<usize>>,
    prompts: &'a [LlmPromptSnapshot],
    context_window_tokens: Option<u64>,
}

fn render_rlm_step_with_fanout(
    out: &mut String,
    spine: &mut String,
    ctx: &mut RenderCtx<'_>,
    input: RlmStepFanoutInput<'_>,
) {
    let RlmStepFanoutInput {
        step,
        tool_call_map,
        fanout_index,
        prompts,
        context_window_tokens,
    } = input;
    render_rlm_step(out, spine, ctx, step, tool_call_map);
    // After the RLM step's body, render fan-outs for any of its inline
    // tool calls that have direct_completion children.
    for call_id in &step.tool_call_ids {
        if let Some(record) = tool_call_map.get(call_id)
            && let Some(children) = fanout_index.get(call_id)
        {
            render_tool_fanout(
                out,
                spine,
                ctx,
                record,
                prompts,
                children,
                context_window_tokens,
            );
        }
    }
}

pub(crate) fn render_message(
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
        MessageRole::Event => ("event", "event", "◆"),
    };

    let user_text = if matches!(message.role, MessageRole::User) {
        first_message_text(message)
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

pub(crate) fn first_message_text(message: &Message) -> Option<String> {
    message
        .parts
        .iter()
        .find(|part| matches!(part.kind, PartKind::Text | PartKind::Prose))
        .map(|part| part.content.trim().to_string())
        .filter(|text| !text.is_empty())
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

pub(crate) fn call_id_short(call_id: &str) -> String {
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

pub(crate) fn render_tool_call_entry(
    out: &mut String,
    spine: &mut String,
    ctx: &mut RenderCtx<'_>,
    record: &ToolCallRecord,
    parent: Option<&str>,
) {
    let id = ctx.next_id();
    let (status_key, status_label) = match record.output.status() {
        lash_core::ToolCallStatus::Success => ("ok", "ok"),
        lash_core::ToolCallStatus::Failure => ("err", "error"),
        lash_core::ToolCallStatus::Cancelled => ("cancelled", "cancelled"),
    };
    let glyph = match record.output.status() {
        lash_core::ToolCallStatus::Success => "•",
        lash_core::ToolCallStatus::Failure => "×",
        lash_core::ToolCallStatus::Cancelled => "◌",
    };
    let summary = summarize_args(&record.args);
    let result_size = json_byte_size(&record.output.value_for_projection());
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
    search.push_str(&pretty_json(&record.output.value_for_projection()).to_lowercase());

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
    let busy_session =
        ctx.stats.tool_calls_ok + ctx.stats.tool_calls_err + ctx.stats.tool_calls_cancelled > 30;
    let auto_open =
        !record.output.is_success() || (!busy_session && (args_size + result_size) <= 4096);
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
    let result_size = json_byte_size(&record.output.value_for_projection());
    let args_str = pretty_json(&record.args);
    let result_str = pretty_json(&record.output.value_for_projection());

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
    let result_class = if record.output.is_success() {
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

pub(crate) fn render_rlm_step(
    out: &mut String,
    spine: &mut String,
    ctx: &mut RenderCtx<'_>,
    step: &RlmTrajectoryEntry,
    tool_call_map: &HashMap<String, &ToolCallRecord>,
) {
    let id = ctx.next_id();
    let has_err = step.error.is_some();
    let status_key = if has_err { "err" } else { "ok" };
    let nested_calls = step
        .tool_call_ids
        .iter()
        .filter(|call_id| tool_call_map.contains_key(*call_id))
        .count();
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
        step.protocol_iteration
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
    render_inline_rlm_tool_calls(out, step, tool_call_map);

    out.push_str("        </div>\n");
    out.push_str("      </div>\n");
    out.push_str("    </article>\n");

    let _ = writeln!(
        spine,
        "    <a class=\"spine-tick spine-tick--rlm\" href=\"#{id}\" data-spine=\"rlm\" data-status=\"{status_key}\" title=\"RLM step {it}{nested}\"></a>",
        it = step.protocol_iteration,
        nested = if nested_calls > 0 {
            format!(" · {nested_calls} calls")
        } else {
            String::new()
        },
    );

    // Render nested tool calls as child entries.
    for call_id in &step.tool_call_ids {
        if let Some(record) = tool_call_map.get(call_id) {
            render_tool_call_entry(out, spine, ctx, record, Some(&id));
        }
    }
}

fn render_inline_rlm_tool_calls(
    out: &mut String,
    step: &RlmTrajectoryEntry,
    tool_call_map: &HashMap<String, &ToolCallRecord>,
) {
    let records = step
        .tool_call_ids
        .iter()
        .filter_map(|call_id| tool_call_map.get(call_id).copied())
        .collect::<Vec<_>>();
    if records.is_empty() {
        return;
    }
    let has_error = records.iter().any(|record| !record.output.is_success());
    let open_attr = if has_error { " open" } else { "" };
    let total_ms: u64 = records.iter().map(|record| record.duration_ms).sum();
    let _ = writeln!(
        out,
        "          <details class=\"rlm-tool-list\"{open_attr}>"
    );
    let _ = writeln!(
        out,
        "            <summary class=\"code-bar\"><span class=\"code-tag\">tool calls</span><span class=\"code-size\">{} · {}</span></summary>",
        records.len(),
        format_duration(total_ms),
    );
    out.push_str("            <div class=\"rlm-tool-stack\">\n");
    for (idx, record) in records.iter().enumerate() {
        render_inline_tool_call(out, idx + 1, record);
    }
    out.push_str("            </div>\n");
    out.push_str("          </details>\n");
}

fn render_inline_tool_call(out: &mut String, ordinal: usize, record: &ToolCallRecord) {
    let (status_key, status_label) = match record.output.status() {
        lash_core::ToolCallStatus::Success => ("ok", "ok"),
        lash_core::ToolCallStatus::Failure => ("err", "error"),
        lash_core::ToolCallStatus::Cancelled => ("cancelled", "cancelled"),
    };
    let summary = summarize_args(&record.args);
    let result_size = json_byte_size(&record.output.value_for_projection());
    let open_attr = match record.output.status() {
        lash_core::ToolCallStatus::Success => "",
        lash_core::ToolCallStatus::Failure | lash_core::ToolCallStatus::Cancelled => " open",
    };
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
