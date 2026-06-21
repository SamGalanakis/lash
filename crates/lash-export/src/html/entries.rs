use std::collections::HashMap;
use std::fmt::Write as _;

use lash_core::ChronologicalPayload;
use lash_core::session_model::{Message, MessageRole, Part, PartKind, PruneState};

use super::escaping::{escape, escape_attr, json_highlight};
use super::prompt::{
    CoalesceState, PromptAnchor, compute_coalesce_states, compute_prompt_insertions,
    render_system_prompt, render_system_prompt_banner, write_usage_chart_bar,
};
use super::view_model::{RenderCtx, json_byte_size, one_line_summary, pretty_json, truncate};
use crate::LoadedSession;
use crate::transcript::{
    LashlangTranscriptStep, TranscriptEntryKind, format_count, project_chronological_entries,
    suppressed_rlm_final_output_message_ids,
};

pub(crate) struct EntriesHtml {
    pub(crate) entries: String,
    pub(crate) spine: String,
    pub(crate) usage_chart: String,
}

pub(crate) fn render_entries(session: &LoadedSession, ctx: &mut RenderCtx) -> EntriesHtml {
    let mut entries = String::with_capacity(32 * 1024);
    let mut spine = String::with_capacity(2 * 1024);
    let mut usage_chart = String::with_capacity(2 * 1024);

    let insertions = compute_prompt_insertions(&session.chronological, &session.llm_prompts);
    let mut last_hash: Option<String> = None;
    let mut first_seen: HashMap<String, PromptAnchor> = HashMap::new();

    // Coalesce runs of consecutive identical-system-hash main-flow prompts
    // into one banner instead of N stub rows.
    let main_indices: Vec<usize> = (0..session.llm_prompts.len()).collect();
    let coalesce: HashMap<usize, CoalesceState> = compute_coalesce_states(
        &session.llm_prompts,
        &main_indices,
        /* threshold (run length) = */ 3,
    );

    let suppressed_message_ids = suppressed_rlm_final_output_message_ids(&session.chronological);

    let emit_prompt = |entries: &mut String,
                       spine: &mut String,
                       ctx: &mut RenderCtx,
                       last_hash: &mut Option<String>,
                       first_seen: &mut HashMap<String, PromptAnchor>,
                       usage_chart: &mut String,
                       prompt_idx: usize| {
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
        if let ChronologicalPayload::Message(message) = &entry.payload
            && (message.is_transient() || suppressed_message_ids.contains(&message.id))
        {
            continue;
        }
        for transcript_entry in project_chronological_entries(entry) {
            match transcript_entry.kind {
                TranscriptEntryKind::Message(message) => {
                    render_message(&mut entries, &mut spine, ctx, message);
                }
                TranscriptEntryKind::AssistantReasoning(text) => {
                    render_assistant_reasoning_entry(&mut entries, &mut spine, ctx, &text);
                }
                TranscriptEntryKind::AssistantText(text) => {
                    render_assistant_text_entry(&mut entries, &mut spine, ctx, &text);
                }
                TranscriptEntryKind::LashlangStep(step) => {
                    render_lashlang_step(&mut entries, &mut spine, ctx, &step);
                }
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

pub(crate) fn render_message(
    out: &mut String,
    spine: &mut String,
    ctx: &mut RenderCtx,
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

pub(crate) fn render_assistant_reasoning_entry(
    out: &mut String,
    spine: &mut String,
    ctx: &mut RenderCtx,
    text: &str,
) {
    render_assistant_content_entry(
        out,
        spine,
        ctx,
        "assistant_reasoning",
        "reasoning",
        text,
        true,
    );
}

pub(crate) fn render_assistant_text_entry(
    out: &mut String,
    spine: &mut String,
    ctx: &mut RenderCtx,
    text: &str,
) {
    render_assistant_content_entry(out, spine, ctx, "assistant_text", "assistant", text, false);
}

fn render_assistant_content_entry(
    out: &mut String,
    spine: &mut String,
    ctx: &mut RenderCtx,
    kind: &str,
    label: &str,
    text: &str,
    reasoning: bool,
) {
    if text.trim().is_empty() {
        return;
    }
    let id = ctx.next_id();
    let headline = one_line_summary(text, 200);
    let search = text.to_lowercase();
    let _ = writeln!(
        out,
        "    <article class=\"entry entry--assistant\" id=\"{id}\" data-role=\"assistant\" data-kind=\"{kind}\" data-search=\"{search}\">",
        search = escape_attr(&search)
    );
    out.push_str("      <div class=\"entry-rail\">\n");
    let _ = writeln!(
        out,
        "        <a class=\"entry-num\" href=\"#{id}\" title=\"permalink\">{id}</a>"
    );
    out.push_str("        <span class=\"entry-glyph\">■</span>\n");
    out.push_str("      </div>\n");
    out.push_str("      <div class=\"entry-body\">\n");
    out.push_str("        <header class=\"entry-head\">\n");
    let _ = writeln!(out, "          <span class=\"entry-tag\">{label}</span>");
    out.push_str("          <span class=\"entry-headline entry-headline--ghost\"></span>\n");
    let _ = writeln!(
        out,
        "          <span class=\"entry-meta\">{}</span>",
        format_count(text.chars().count() as u64)
    );
    out.push_str("        </header>\n");
    out.push_str("        <div class=\"entry-content\">\n");
    if reasoning {
        render_reasoning(out, text);
    } else {
        render_prose(out, text);
    }
    out.push_str("        </div>\n");
    out.push_str("      </div>\n");
    out.push_str("    </article>\n");
    let _ = writeln!(
        spine,
        "    <a class=\"spine-tick\" href=\"#{id}\" data-spine=\"assistant\" title=\"{label} · {h}\"></a>",
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

mod leaf;
pub(crate) use leaf::pick_display_title;
use leaf::{render_code, render_part, render_prose, render_reasoning};

// ─── lashlang step ──────────────────────────────────────────────────────────

pub(crate) fn render_lashlang_step(
    out: &mut String,
    spine: &mut String,
    ctx: &mut RenderCtx,
    step: &LashlangTranscriptStep,
) {
    let id = ctx.next_id();
    let has_err = step.error.is_some();
    let status_key = if has_err { "err" } else { "ok" };
    let output_preview = if let Some(out_item) = step.output.iter().find(|o| !o.trim().is_empty()) {
        one_line_summary(out_item, 200)
    } else {
        String::new()
    };
    let mut search = String::new();
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
        "    <article class=\"entry entry--lashlang entry--{status_key}\" id=\"{id}\" data-role=\"lashlang\" data-kind=\"lashlang_step\" data-status=\"{status_key}\" data-search=\"{search}\">",
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
        "          <span class=\"entry-tag entry-tag--lashlang\">lashlang step {}</span>",
        step.protocol_iteration
    );
    let _ = writeln!(
        out,
        "          <span class=\"entry-headline\">{}</span>",
        escape(&output_preview)
    );
    if has_err {
        out.push_str("          <span class=\"entry-meta entry-meta--err\">error</span>\n");
    }
    let _ = writeln!(
        out,
        "          <span class=\"entry-meta\">{}</span>",
        format_count(total_chars as u64)
    );
    out.push_str("        </header>\n");
    out.push_str("        <div class=\"entry-content\">\n");

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
    out.push_str("        </div>\n");
    out.push_str("      </div>\n");
    out.push_str("    </article>\n");

    let _ = writeln!(
        spine,
        "    <a class=\"spine-tick spine-tick--lashlang\" href=\"#{id}\" data-spine=\"lashlang\" data-status=\"{status_key}\" title=\"lashlang step {it}\"></a>",
        it = step.protocol_iteration,
    );
}

// ─── system prompt (per-LLM-call snapshot from trace) ───────────────────────
