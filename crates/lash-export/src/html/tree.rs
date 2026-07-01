use std::collections::HashMap;
use std::fmt::Write as _;

use lash_core::ChronologicalPayload;

use crate::LoadedSession;
use crate::transcript::{
    TranscriptEntryKind, project_chronological_entries, suppressed_rlm_final_output_message_ids,
};
use crate::tree::{LoadedSessionNode, LoadedSessionTree, NodeRelation};

use super::assets::{CSS, JS};
use super::entries::{
    render_assistant_reasoning_entry, render_assistant_text_entry, render_lashlang_step,
    render_message,
};
use super::escaping::{escape, escape_attr, js_escape};
use super::prompt::{
    PromptAnchor, compute_prompt_insertions, render_system_prompt, write_usage_chart_bar,
};
use super::session::write_controls;
use super::stats::{SessionStats, compute_stats};
use super::view_model::{RenderCtx, one_line_summary};

pub fn render_tree(tree: &LoadedSessionTree) -> String {
    let mut out = String::with_capacity(128 * 1024);

    let title = tree.root().meta.session_name.clone();
    out.push_str("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n");
    out.push_str("<meta charset=\"utf-8\">\n");
    out.push_str("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n");
    let _ = writeln!(out, "<title>{} · lash trace</title>", escape(&title));
    out.push_str("<style>\n");
    out.push_str(CSS);
    out.push_str("\n</style>\n</head>\n<body>\n");

    out.push_str("<div class=\"page\"><div class=\"frame\">\n");
    write_crumb_bar(&mut out);

    let view_heads: Vec<&LoadedSessionNode> = tree.nodes.iter().collect();

    for head in &view_heads {
        write_view(&mut out, tree, head);
    }

    out.push_str("</div></div>\n");

    out.push_str("<script>\n");
    out.push_str(&render_tree_data_script(tree, &view_heads));
    out.push_str("\n</script>\n");
    out.push_str("<script>\n");
    out.push_str(JS);
    out.push_str("\n</script>\n");
    out.push_str("</body>\n</html>\n");
    out
}

pub(crate) fn write_crumb_bar(out: &mut String) {
    out.push_str("<nav class=\"crumb-bar\" aria-label=\"session breadcrumb\">\n");
    out.push_str("  <div class=\"crumb-trail\" id=\"crumb-trail\">\n");
    out.push_str(
        "    <span class=\"crumb-eyebrow\">lash <span class=\"slash\">/</span> trace</span>\n",
    );
    out.push_str("  </div>\n");
    out.push_str("  <div class=\"crumb-actions\">\n");
    out.push_str("    <button class=\"back-link\" id=\"back-btn\" disabled>back</button>\n");
    out.push_str("  </div>\n");
    out.push_str("</nav>\n");
}

fn write_view(out: &mut String, tree: &LoadedSessionTree, head: &LoadedSessionNode) {
    let chain: Vec<&LoadedSessionNode> = vec![head];
    let chain_stats = compute_chain_stats(&chain);
    let mut ctx = RenderCtx::new();
    let view_id = view_id_of(head);
    let active_attr = if matches!(head.kind, NodeRelation::Root) {
        " is-active"
    } else {
        ""
    };
    let _ = writeln!(
        out,
        "<section class=\"view{active_attr}\" data-view=\"{}\" data-session=\"{}\">",
        escape_attr(&view_id),
        escape_attr(&head.meta.session_id)
    );

    write_view_hero(out, tree, head, &chain);
    write_lineage(out, tree, head);
    if matches!(head.kind, NodeRelation::Root) {
        write_controls(out, &chain_stats);
    }

    out.push_str("<div class=\"body\">\n");
    out.push_str("  <aside class=\"spine\" aria-label=\"trace minimap\">\n");
    let mut spine = String::new();
    let mut entries = String::new();

    for node in &chain {
        render_node_entries(&mut entries, &mut spine, &mut ctx, node);
    }

    out.push_str(&spine);
    out.push_str("  </aside>\n");
    out.push_str("  <main class=\"transcript\">\n");
    out.push_str(&entries);
    out.push_str("  </main>\n");
    out.push_str("</div>\n");

    out.push_str("</section>\n");
}

fn view_id_of(node: &LoadedSessionNode) -> String {
    match &node.kind {
        NodeRelation::Root => "root".to_string(),
        NodeRelation::Subagent { task, .. } => {
            let base = task
                .clone()
                .unwrap_or_else(|| short_session_id(&node.meta.session_id));
            slug(&base)
        }
    }
}

fn short_session_id(sid: &str) -> String {
    sid.chars().take(8).collect()
}

fn slug(input: &str) -> String {
    input
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect()
}

fn compute_chain_stats(chain: &[&LoadedSessionNode]) -> SessionStats {
    let mut s = SessionStats::default();
    let mut tool_counts: HashMap<String, usize> = HashMap::new();
    for node in chain {
        let session = node_as_session(node);
        let part = compute_stats(&session);
        s.user_messages += part.user_messages;
        s.assistant_messages += part.assistant_messages;
        s.system_messages += part.system_messages;
        s.tool_calls_ok += part.tool_calls_ok;
        s.tool_calls_err += part.tool_calls_err;
        s.tool_calls_cancelled += part.tool_calls_cancelled;
        s.tool_total_ms = s.tool_total_ms.saturating_add(part.tool_total_ms);
        s.rlm_iterations += part.rlm_iterations;
        s.rlm_errors += part.rlm_errors;
        s.pruned_parts += part.pruned_parts;
        s.cleared_parts += part.cleared_parts;
        s.total_chars = s.total_chars.saturating_add(part.total_chars);
        s.chronological += part.chronological;
        s.llm_calls_with_usage += part.llm_calls_with_usage;
        s.input_tokens = s.input_tokens.saturating_add(part.input_tokens);
        s.output_tokens = s.output_tokens.saturating_add(part.output_tokens);
        s.cache_read_input_tokens = s
            .cache_read_input_tokens
            .saturating_add(part.cache_read_input_tokens);
        s.cache_write_input_tokens = s
            .cache_write_input_tokens
            .saturating_add(part.cache_write_input_tokens);
        s.reasoning_output_tokens = s
            .reasoning_output_tokens
            .saturating_add(part.reasoning_output_tokens);
        for (name, count) in part.tool_freq {
            *tool_counts.entry(name).or_insert(0) += count;
        }
    }
    let mut freq: Vec<(String, usize)> = tool_counts.into_iter().collect();
    freq.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    s.tool_names_set = freq.iter().map(|(n, _)| n.clone()).collect();
    s.tool_freq = freq;
    s
}

fn node_as_session(node: &LoadedSessionNode) -> LoadedSession {
    LoadedSession {
        meta: Some(node.meta.clone()),
        chronological: node.chronological.clone(),
        trace_path: node.db_path.clone(),
        context_window_tokens: node.context_window_tokens,
        llm_prompts: node.llm_prompts.clone(),
    }
}

fn write_view_hero(
    out: &mut String,
    _tree: &LoadedSessionTree,
    head: &LoadedSessionNode,
    chain: &[&LoadedSessionNode],
) {
    let meta = &head.meta;
    let display_title =
        if meta.session_name.trim().is_empty() || meta.session_name == meta.session_id {
            match &head.kind {
                NodeRelation::Root => "lash session".to_string(),
                NodeRelation::Subagent { task, .. } => task
                    .as_deref()
                    .map(|task| one_line_summary(task, 80))
                    .unwrap_or_else(|| "subagent".to_string()),
            }
        } else {
            meta.session_name.clone()
        };
    let role_label = match &head.kind {
        NodeRelation::Root => "root session".to_string(),
        NodeRelation::Subagent {
            capability, task, ..
        } => {
            let cap = capability.as_deref().unwrap_or("");
            let name = task
                .as_deref()
                .map(|task| one_line_summary(task, 48))
                .unwrap_or_else(|| "subagent".to_string());
            if cap.is_empty() {
                format!("subagent · {name}")
            } else {
                format!("subagent · {name} · {cap}")
            }
        }
    };

    out.push_str("<header class=\"trace-hero\">\n");
    out.push_str("  <div>\n");
    let _ = writeln!(
        out,
        "    <span class=\"eyebrow\">{} <span class=\"slash\">·</span> {}</span>",
        escape(&role_label),
        escape(&short_session_id(&meta.session_id))
    );
    let _ = writeln!(
        out,
        "    <h1 class=\"hero-title\">{}</h1>",
        escape(&display_title)
    );
    out.push_str("  </div>\n");

    out.push_str("  <div class=\"hero-meta\">\n");
    let view_label = match &head.kind {
        NodeRelation::Root => "root".to_string(),
        NodeRelation::Subagent { .. } => "subagent".to_string(),
    };
    write_meta_row(
        out,
        "view",
        &format!("{} · {}", short_session_id(&meta.session_id), view_label),
    );
    if !meta.model.is_empty() {
        write_meta_row(out, "model", &meta.model);
    }
    if let Some(cwd) = &meta.cwd {
        write_meta_row(out, "cwd", cwd);
    }
    let subagent_count: usize = chain.iter().map(|n| n.subagent_children.len()).sum();
    if subagent_count > 0 {
        write_meta_row(out, "subagents", &format!("{subagent_count} spawned"));
    }
    write_meta_row(out, "id", &meta.session_id);
    out.push_str("  </div>\n");
    out.push_str("</header>\n");
}

fn write_meta_row(out: &mut String, key: &str, val: &str) {
    let _ = writeln!(
        out,
        "    <span class=\"meta-row\"><span class=\"meta-key\">{}</span><span class=\"meta-val\">{}</span></span>",
        escape(key),
        escape(val)
    );
}

fn write_lineage(out: &mut String, tree: &LoadedSessionTree, current: &LoadedSessionNode) {
    out.push_str("<nav class=\"lineage\" aria-label=\"call tree\">\n");
    out.push_str("  <span class=\"lineage-label\">call tree</span>\n");
    out.push_str("  <div class=\"lineage-track\">\n");

    let order = lineage_order(tree);
    let mut prev_kind: Option<&NodeRelation> = None;
    for node in &order {
        if prev_kind.is_some() {
            let (glyph, label, kind) = match &node.kind {
                NodeRelation::Subagent { .. } => ("▾", "spawn", "child"),
                NodeRelation::Root => ("/", "", ""),
            };
            if !label.is_empty() {
                let _ = writeln!(
                    out,
                    "    <span class=\"lineage-edge\" data-edge=\"{kind}\"><span class=\"lineage-edge-glyph\">{glyph}</span><span>{label}</span></span>"
                );
            }
        }
        let is_current = node.meta.session_id == current.meta.session_id;
        let cur_attr = if is_current {
            " aria-current=\"true\""
        } else {
            ""
        };
        let view_id = view_id_of(node);
        let role = match &node.kind {
            NodeRelation::Root => "root".to_string(),
            NodeRelation::Subagent { capability, .. } => {
                let cap = capability.as_deref().unwrap_or("");
                if cap.is_empty() {
                    "subagent".to_string()
                } else {
                    format!("subagent · {cap}")
                }
            }
        };
        let id_short = short_session_id(&node.meta.session_id);
        let label_text = match &node.kind {
            NodeRelation::Subagent { task, .. } => task
                .as_deref()
                .map(|task| one_line_summary(task, 56))
                .unwrap_or_else(|| short_session_id(&node.meta.session_id)),
            NodeRelation::Root => "root".to_string(),
        };
        let _ = writeln!(
            out,
            "    <button class=\"lineage-node\" data-go=\"{vid}\"{cur_attr}><span class=\"lineage-node-label\">{role}</span><span class=\"lineage-node-id\"><span class=\"accent\">{id}</span> — {label}</span></button>",
            vid = escape_attr(&view_id),
            role = escape(&role),
            id = escape(&id_short),
            label = escape(&label_text)
        );
        prev_kind = Some(&node.kind);
    }

    out.push_str("  </div>\n");
    out.push_str("</nav>\n");
}

fn lineage_order(tree: &LoadedSessionTree) -> Vec<&LoadedSessionNode> {
    let mut order = Vec::new();
    fn visit<'a>(
        tree: &'a LoadedSessionTree,
        node: &'a LoadedSessionNode,
        out: &mut Vec<&'a LoadedSessionNode>,
    ) {
        out.push(node);
        for edge in &node.subagent_children {
            if let Some(child) = tree.get(&edge.child_session_id) {
                visit(tree, child, out);
            }
        }
    }
    visit(tree, tree.root(), &mut order);
    order
}

fn render_node_entries(
    out: &mut String,
    spine: &mut String,
    ctx: &mut RenderCtx,
    node: &LoadedSessionNode,
) {
    let session = node_as_session(node);
    let insertions = compute_prompt_insertions(&session.chronological, &session.llm_prompts);
    let mut last_hash: Option<String> = None;
    let mut first_seen: HashMap<String, PromptAnchor> = HashMap::new();
    let mut usage_chart = String::new();

    let suppressed_message_ids = suppressed_rlm_final_output_message_ids(&session.chronological);

    for (i, entry) in session.chronological.iter().enumerate() {
        for &prompt_idx in &insertions.before_index[i] {
            emit_prompt_inline(
                out,
                spine,
                ctx,
                &session,
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
                    render_message(out, spine, ctx, message);
                }
                TranscriptEntryKind::AssistantReasoning(text) => {
                    render_assistant_reasoning_entry(out, spine, ctx, &text);
                }
                TranscriptEntryKind::AssistantText(text) => {
                    render_assistant_text_entry(out, spine, ctx, &text);
                }
                TranscriptEntryKind::LashlangStep(step) => {
                    render_lashlang_step(out, spine, ctx, &step);
                }
            }
        }
    }

    for &prompt_idx in &insertions.trailing {
        emit_prompt_inline(
            out,
            spine,
            ctx,
            &session,
            &mut last_hash,
            &mut first_seen,
            &mut usage_chart,
            prompt_idx,
        );
    }
    let _ = (last_hash, usage_chart);
}

#[allow(clippy::too_many_arguments)]
fn emit_prompt_inline(
    out: &mut String,
    spine: &mut String,
    ctx: &mut RenderCtx,
    session: &LoadedSession,
    last_hash: &mut Option<String>,
    first_seen: &mut HashMap<String, PromptAnchor>,
    usage_chart: &mut String,
    prompt_idx: usize,
) {
    let prompt = &session.llm_prompts[prompt_idx];
    let anchor = first_seen.get(&prompt.system_hash).cloned();
    let id = render_system_prompt(
        out,
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

pub fn render_tree_data_script(
    tree: &LoadedSessionTree,
    view_heads: &[&LoadedSessionNode],
) -> String {
    let mut entries: Vec<String> = Vec::new();
    for head in view_heads {
        let view_id = view_id_of(head);
        let label = match &head.kind {
            NodeRelation::Root => "root".to_string(),
            NodeRelation::Subagent { task, .. } => task
                .as_deref()
                .map(|task| one_line_summary(task, 56))
                .unwrap_or_else(|| short_session_id(&head.meta.session_id)),
        };
        let sid = short_session_id(&head.meta.session_id);
        let parent_view = match &head.kind {
            NodeRelation::Root => None,
            NodeRelation::Subagent {
                parent_session_id, ..
            } => tree.get(parent_session_id).map(view_id_of),
        };
        let parent_str = parent_view
            .map(|p| format!("\"{}\"", js_escape(&p)))
            .unwrap_or_else(|| "null".to_string());
        entries.push(format!(
            "{{\"id\":\"{vid}\",\"label\":\"{label}\",\"sid\":\"{sid}\",\"parent\":{parent}}}",
            vid = js_escape(&view_id),
            label = js_escape(&label),
            sid = js_escape(&sid),
            parent = parent_str
        ));
    }
    format!("window.__lashTraceTree = [{}];", entries.join(","))
}
