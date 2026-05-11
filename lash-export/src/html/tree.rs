use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use lash_core::session_model::MessageRole;
use lash_core::{ChronologicalPayload, ToolCallRecord};

use crate::LoadedSession;
use crate::tree::{LoadedSessionNode, LoadedSessionTree, NodeRelation, SubagentEdge};

use super::assets::{CSS, JS};
use super::chronological_rlm_step;
use super::entries::{first_message_text, render_message, render_rlm_step, render_tool_call_entry};
use super::escaping::{escape, escape_attr, js_escape};
use super::prompt::{
    PromptAnchor, compute_prompt_insertions, render_system_prompt, write_usage_chart_bar,
};
use super::session::write_controls;
use super::stats::{SessionStats, compute_stats};
use super::view_model::{
    RenderCtx, format_count, format_duration, message_matches_text, one_line_summary,
    submit_value_text,
};

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

    let view_heads: Vec<&LoadedSessionNode> = tree
        .nodes
        .iter()
        .filter(|n| !matches!(n.kind, NodeRelation::Handoff { .. }))
        .collect();

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
    let chain: Vec<&LoadedSessionNode> = view_chain(tree, head);
    let chain_stats = compute_chain_stats(&chain);
    let mut ctx = RenderCtx::new(&chain_stats);
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

    for (idx, node) in chain.iter().enumerate() {
        if idx > 0 {
            let parent = chain[idx - 1];
            let handoff_call = parent
                .chronological
                .iter()
                .rev()
                .find_map(|e| match &e.payload {
                    ChronologicalPayload::ToolCall(r) if r.tool == "continue_as" => Some(r),
                    _ => None,
                });
            render_handoff_divider(
                &mut entries,
                &mut spine,
                &mut ctx,
                parent,
                node,
                handoff_call,
            );
        }
        render_node_entries(&mut entries, &mut spine, &mut ctx, tree, node);
    }

    out.push_str(&spine);
    out.push_str("  </aside>\n");
    out.push_str("  <main class=\"transcript\">\n");
    out.push_str(&entries);
    out.push_str("  </main>\n");
    out.push_str("</div>\n");

    out.push_str("</section>\n");
}

fn view_chain<'a>(
    tree: &'a LoadedSessionTree,
    head: &'a LoadedSessionNode,
) -> Vec<&'a LoadedSessionNode> {
    let mut chain = vec![head];
    let mut cur = head;
    while let Some(succ_id) = &cur.handoff_successor {
        let Some(next) = tree.get(succ_id) else { break };
        chain.push(next);
        cur = next;
    }
    chain
}

fn view_id_of(node: &LoadedSessionNode) -> String {
    match &node.kind {
        NodeRelation::Root => "root".to_string(),
        NodeRelation::Subagent { agent_name, .. } => {
            let base = agent_name
                .clone()
                .unwrap_or_else(|| short_session_id(&node.meta.session_id));
            slug(&base)
        }
        NodeRelation::Handoff { .. } => short_session_id(&node.meta.session_id),
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
        s.cached_input_tokens = s
            .cached_input_tokens
            .saturating_add(part.cached_input_tokens);
        s.reasoning_tokens = s.reasoning_tokens.saturating_add(part.reasoning_tokens);
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
                NodeRelation::Subagent { agent_name, .. } => {
                    agent_name.clone().unwrap_or_else(|| "subagent".to_string())
                }
                NodeRelation::Handoff { .. } => "handoff".to_string(),
            }
        } else {
            meta.session_name.clone()
        };
    let role_label = match &head.kind {
        NodeRelation::Root => "root session".to_string(),
        NodeRelation::Subagent {
            capability,
            agent_name,
            ..
        } => {
            let cap = capability.as_deref().unwrap_or("");
            let name = agent_name.as_deref().unwrap_or("subagent");
            if cap.is_empty() {
                format!("subagent · {name}")
            } else {
                format!("subagent · {name} · {cap}")
            }
        }
        NodeRelation::Handoff { .. } => "handoff successor".to_string(),
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
        NodeRelation::Handoff { .. } => "handoff".to_string(),
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
    if chain.len() > 1 {
        write_meta_row(
            out,
            "chain",
            &format!("{} sessions joined by continue_as", chain.len()),
        );
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
                NodeRelation::Handoff { .. } => ("↪", "continue_as", "handoff"),
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
            NodeRelation::Handoff { .. } => "handoff".to_string(),
        };
        let id_short = short_session_id(&node.meta.session_id);
        let label_text = match &node.kind {
            NodeRelation::Subagent { agent_name, .. } => agent_name
                .clone()
                .unwrap_or_else(|| short_session_id(&node.meta.session_id)),
            NodeRelation::Root => "root".to_string(),
            NodeRelation::Handoff { .. } => "handoff".to_string(),
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
        if let Some(succ_id) = &node.handoff_successor
            && let Some(succ) = tree.get(succ_id)
        {
            let mut c = succ;
            loop {
                for edge in &c.subagent_children {
                    if let Some(child) = tree.get(&edge.child_session_id) {
                        visit(tree, child, out);
                    }
                }
                let Some(next_id) = &c.handoff_successor else {
                    break;
                };
                let Some(next) = tree.get(next_id) else {
                    break;
                };
                c = next;
            }
        }
    }
    visit(tree, tree.root(), &mut order);
    order
}

fn render_node_entries(
    out: &mut String,
    spine: &mut String,
    ctx: &mut RenderCtx<'_>,
    tree: &LoadedSessionTree,
    node: &LoadedSessionNode,
) {
    let session = node_as_session(node);
    let insertions = compute_prompt_insertions(&session.chronological, &session.llm_prompts);
    let mut last_hash: Option<String> = None;
    let mut first_seen: HashMap<String, PromptAnchor> = HashMap::new();
    let mut usage_chart = String::new();

    let mut suppressed_message_ids: HashSet<String> = HashSet::new();
    let mut last_final_output: Option<String> = None;
    for entry in session.chronological.iter() {
        match &entry.payload {
            ChronologicalPayload::ModeEvent(event) => {
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
        match &entry.payload {
            ChronologicalPayload::Message(message) => {
                if message.is_transient() || suppressed_message_ids.contains(&message.id) {
                    continue;
                }
                render_message(out, spine, ctx, message);
            }
            ChronologicalPayload::ToolCall(record) => match record.tool.as_str() {
                "continue_as" => {
                    // Suppressed — the seam is rendered by `write_view` between
                    // chain elements as a handoff divider.
                }
                "spawn_agent" => {
                    if let Some(edge) = node
                        .subagent_children
                        .iter()
                        .find(|e| e.call_id.as_deref() == record.call_id.as_deref())
                        && let Some(child) = tree.get(&edge.child_session_id)
                    {
                        render_drill_card(out, spine, ctx, edge, child);
                    } else {
                        render_tool_call_entry(out, spine, ctx, record, None);
                    }
                }
                _ => {
                    render_tool_call_entry(out, spine, ctx, record, None);
                }
            },
            ChronologicalPayload::ModeEvent(event) => {
                let Some(step) = chronological_rlm_step(event) else {
                    continue;
                };
                render_rlm_step(out, spine, ctx, &step);
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
    ctx: &mut RenderCtx<'_>,
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
                .mode_iteration
                .map(|i| format!("iter {i}"))
                .unwrap_or_else(|| "first call".to_string()),
        });
    *last_hash = Some(prompt.system_hash.clone());
}

fn render_drill_card(
    out: &mut String,
    spine: &mut String,
    ctx: &mut RenderCtx<'_>,
    edge: &SubagentEdge,
    child: &LoadedSessionNode,
) {
    let id = ctx.next_id();
    let view_id = view_id_of(child);
    let agent = edge.agent_name.as_deref().unwrap_or("subagent");
    let cap = edge.capability.as_deref().unwrap_or("");
    let task = child
        .chronological
        .iter()
        .find_map(|e| match &e.payload {
            ChronologicalPayload::Message(m)
                if matches!(m.role, MessageRole::User) && !m.is_transient() =>
            {
                first_message_text(m)
            }
            _ => None,
        })
        .unwrap_or_default();
    let turns = child
        .chronological
        .iter()
        .filter(|e| {
            matches!(&e.payload,
                ChronologicalPayload::Message(m)
                    if matches!(m.role, MessageRole::Assistant) && !m.is_transient()
            )
        })
        .count();
    let token_total: i64 = child
        .llm_prompts
        .iter()
        .filter_map(|p| p.usage.as_ref())
        .map(|u| u.input_tokens.max(0) + u.output_tokens.max(0))
        .sum();
    let status_class = if edge.success { "ok" } else { "err" };
    let status_label = if edge.success { "ok" } else { "error" };
    let dur = format_duration(edge.duration_ms);
    let short_id = short_session_id(&child.meta.session_id);

    let _ = writeln!(
        out,
        "    <button class=\"drill\" data-go=\"{vid}\" id=\"{id}\" data-role=\"subagent\">",
        vid = escape_attr(&view_id)
    );
    out.push_str("      <div class=\"drill-left\">\n");
    out.push_str("        <div class=\"drill-rail\">▾</div>\n");
    out.push_str("        <div class=\"drill-content\">\n");
    out.push_str("          <span class=\"drill-eyebrow\">▾ subagent · spawn_agent</span>\n");
    let _ = writeln!(
        out,
        "          <span class=\"drill-title\">{}{}</span>",
        escape(agent),
        if cap.is_empty() {
            String::new()
        } else {
            format!(" <span class=\"drill-cap\">· {}</span>", escape(cap))
        }
    );
    if !task.trim().is_empty() {
        let _ = writeln!(
            out,
            "          <p class=\"drill-task\">{}</p>",
            escape(&one_line_summary(&task, 240))
        );
    }
    out.push_str("          <div class=\"drill-pills\">\n");
    let _ = writeln!(
        out,
        "            <span class=\"drill-pill\" data-kind=\"status\" data-status=\"{status_class}\">{status_label}</span>"
    );
    let _ = writeln!(
        out,
        "            <span class=\"drill-pill\"><span class=\"drill-pill-key\">turns</span><span class=\"drill-pill-val\">{}</span></span>",
        turns
    );
    if token_total > 0 {
        let _ = writeln!(
            out,
            "            <span class=\"drill-pill\"><span class=\"drill-pill-key\">tokens</span><span class=\"drill-pill-val\">{}</span></span>",
            format_count(token_total as u64)
        );
    }
    out.push_str("          </div>\n");
    out.push_str("        </div>\n");
    out.push_str("      </div>\n");
    out.push_str("      <div class=\"drill-right\">\n");
    let _ = writeln!(
        out,
        "        <span class=\"drill-route\"><span class=\"drill-route-arrow\">→</span>{}</span>",
        escape(&short_id)
    );
    let _ = writeln!(
        out,
        "        <span class=\"drill-duration\">{}</span>",
        escape(&dur)
    );
    out.push_str("      </div>\n");
    out.push_str("    </button>\n");

    let _ = writeln!(
        spine,
        "    <a class=\"spine-tick\" href=\"#{id}\" data-spine=\"child\" data-status=\"{status_class}\" title=\"spawn_agent · {}\"></a>",
        escape_attr(agent)
    );
}

fn render_handoff_divider(
    out: &mut String,
    spine: &mut String,
    ctx: &mut RenderCtx<'_>,
    parent: &LoadedSessionNode,
    successor: &LoadedSessionNode,
    record: Option<&ToolCallRecord>,
) {
    let id = ctx.next_id();
    let task = record
        .and_then(|r| r.args.get("task"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let summary = if task.trim().is_empty() {
        "continued in successor session".to_string()
    } else {
        one_line_summary(&task, 200)
    };
    let parent_short = short_session_id(&parent.meta.session_id);
    let succ_short = short_session_id(&successor.meta.session_id);

    let _ = writeln!(
        out,
        "    <section class=\"handoff\" id=\"{id}\" aria-label=\"handoff from {p} to {s}\">",
        p = escape_attr(&parent_short),
        s = escape_attr(&succ_short)
    );
    out.push_str("      <div class=\"handoff-banner\">\n");
    out.push_str("        <div class=\"handoff-glyph\">↪</div>\n");
    out.push_str("        <div class=\"handoff-title\">\n");
    out.push_str("          <span class=\"handoff-eyebrow\">↪ continued as</span>\n");
    let _ = writeln!(
        out,
        "          <span class=\"handoff-summary\">{}</span>",
        escape(&summary)
    );
    out.push_str("        </div>\n");
    let _ = writeln!(
        out,
        "        <div class=\"handoff-route\"><span class=\"route-id\">{}</span><span class=\"route-arrow\">→</span><span class=\"route-id sodium\">{}</span></div>",
        escape(&parent_short),
        escape(&succ_short)
    );
    out.push_str("      </div>\n");

    out.push_str("      <div class=\"handoff-body\">\n");
    if !task.trim().is_empty() {
        let _ = writeln!(
            out,
            "        <p class=\"handoff-task\">{}</p>",
            escape(&task)
        );
    }

    if let Some(seed_value) = record.and_then(|r| r.args.get("seed"))
        && let Some(seed_obj) = seed_value.as_object()
        && !seed_obj.is_empty()
    {
        let _ = writeln!(
            out,
            "        <div><span class=\"handoff-seed-label\">seed · {} entries</span><div class=\"seed-list\">",
            seed_obj.len()
        );
        for (name, value) in seed_obj {
            let projected = lash_rlm_types::projection_inner(value).is_some();
            let kind_label = if projected { "projected" } else { "global" };
            let kind_attr = if projected { "projected" } else { "global" };
            let _ = writeln!(
                out,
                "          <span class=\"seed-pill\" data-kind=\"{kind}\"><span class=\"seed-name\">{name}</span><span class=\"seed-kind\">{label}</span></span>",
                kind = kind_attr,
                name = escape(name),
                label = kind_label
            );
        }
        out.push_str("        </div></div>\n");
    }

    let parent_turns = parent
        .chronological
        .iter()
        .filter(|e| {
            matches!(&e.payload,
                ChronologicalPayload::Message(m)
                    if matches!(m.role, MessageRole::User) && !m.is_transient()
            )
        })
        .count()
        + parent
            .chronological
            .iter()
            .filter(|e| {
                matches!(
                    &e.payload,
                    ChronologicalPayload::ModeEvent(event) if chronological_rlm_step(event).is_some()
                )
            })
            .count();
    let max_ctx_pct = parent
        .llm_prompts
        .iter()
        .filter_map(|p| {
            let usage = p.usage.as_ref()?;
            let window = parent.context_window_tokens.filter(|w| *w > 0)?;
            Some(usage.input_tokens.max(0) as f64 * 100.0 / window as f64)
        })
        .fold(None::<f64>, |acc, x| Some(acc.map_or(x, |m| m.max(x))));

    out.push_str("        <div class=\"handoff-stats\">\n");
    if let Some(pct) = max_ctx_pct {
        let _ = writeln!(
            out,
            "          <span class=\"handoff-stat\"><span class=\"handoff-stat-key\">parent ctx</span><span class=\"handoff-stat-val\">{:.0}%</span></span>",
            pct
        );
    }
    let _ = writeln!(
        out,
        "          <span class=\"handoff-stat\"><span class=\"handoff-stat-key\">turns</span><span class=\"handoff-stat-val\">{}</span></span>",
        parent_turns
    );
    let _ = writeln!(
        out,
        "          <span class=\"handoff-stat\"><span class=\"handoff-stat-key\">reason</span><span class=\"handoff-stat-val\">continue_as</span></span>"
    );
    out.push_str("        </div>\n");

    out.push_str("      </div>\n");
    out.push_str("    </section>\n");

    let _ = writeln!(
        spine,
        "    <a class=\"spine-tick\" href=\"#{id}\" data-spine=\"handoff\" title=\"handoff to {}\"></a>",
        escape_attr(&succ_short)
    );
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
            NodeRelation::Subagent { agent_name, .. } => agent_name
                .clone()
                .unwrap_or_else(|| short_session_id(&head.meta.session_id)),
            NodeRelation::Handoff { .. } => "handoff".to_string(),
        };
        let sid = short_session_id(&head.meta.session_id);
        let parent_view = match &head.kind {
            NodeRelation::Root => None,
            NodeRelation::Subagent {
                parent_session_id, ..
            }
            | NodeRelation::Handoff {
                parent_session_id, ..
            } => {
                let mut cur = tree.get(parent_session_id);
                while let Some(node) = cur {
                    match &node.kind {
                        NodeRelation::Handoff { .. } => {
                            cur = tree.parent_of(&node.meta.session_id);
                        }
                        _ => break,
                    }
                }
                cur.map(view_id_of)
            }
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
