//! Multi-session export — discover descendant sessions reachable from a
//! root `.db` and classify each cross-session edge as either a `Handoff`
//! (continue_as) or a `Subagent` (spawn_agent).
//!
//! The on-disk shape is one `.db` per session in `sessions_dir/`, each
//! carrying its own `parent_session_id` in `session_meta`. The relation
//! kind is *not* persisted — we recover it by inspecting the parent's
//! chronological projection: if the parent has a `continue_as` tool call
//! whose `result.session_id` matches the child, it's a handoff; otherwise
//! it's a subagent.
//!
//! One provider trace JSONL covers the whole `lash` invocation. Each
//! record carries `context.session_id`; we partition `LlmPromptSnapshot`s
//! by that field so each session in the tree gets its matching prompts.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use lash::{ChronologicalEntry, ChronologicalPayload, SessionGraph, SessionMeta};
use lash_sqlite_store::Store;

use crate::trace::{LlmPromptSnapshot, load_prompts_from_trace};

/// One session in the tree, ready to render.
pub struct LoadedSessionNode {
    pub meta: SessionMeta,
    pub chronological: Vec<ChronologicalEntry>,
    pub context_window_tokens: Option<u64>,
    pub llm_prompts: Vec<LlmPromptSnapshot>,
    pub db_path: PathBuf,
    pub kind: NodeRelation,
    /// Sessions that this session spawned (`spawn_agent`), in tool-call order.
    pub subagent_children: Vec<SubagentEdge>,
    /// If this session ended with `continue_as`, the successor's session_id.
    pub handoff_successor: Option<String>,
}

/// What kind of edge points *into* this session from its parent.
#[derive(Clone, Debug)]
pub enum NodeRelation {
    Root,
    /// Parent ended with `continue_as` and this session took over.
    Handoff {
        parent_session_id: String,
    },
    /// Parent called `spawn_agent` and waited for this session to submit.
    Subagent {
        parent_session_id: String,
        agent_name: Option<String>,
        capability: Option<String>,
        /// The parent's `spawn_agent` tool call_id, used to anchor the
        /// drill-in card to its position in the parent's transcript.
        parent_call_id: Option<String>,
    },
}

/// Records one `spawn_agent` edge from a parent to a child session.
#[derive(Clone, Debug)]
pub struct SubagentEdge {
    pub child_session_id: String,
    pub agent_name: Option<String>,
    pub capability: Option<String>,
    pub call_id: Option<String>,
    pub success: bool,
    pub duration_ms: u64,
}

/// The full discovered tree.
pub struct LoadedSessionTree {
    pub root_id: String,
    pub trace_path: PathBuf,
    pub nodes: Vec<LoadedSessionNode>,
}

impl LoadedSessionTree {
    pub fn root(&self) -> &LoadedSessionNode {
        self.nodes
            .iter()
            .find(|n| n.meta.session_id == self.root_id)
            .expect("root must exist in tree")
    }

    pub fn get(&self, session_id: &str) -> Option<&LoadedSessionNode> {
        self.nodes.iter().find(|n| n.meta.session_id == session_id)
    }

    pub fn parent_of(&self, session_id: &str) -> Option<&LoadedSessionNode> {
        let node = self.get(session_id)?;
        let parent_id = match &node.kind {
            NodeRelation::Root => return None,
            NodeRelation::Handoff { parent_session_id } => parent_session_id.as_str(),
            NodeRelation::Subagent {
                parent_session_id, ..
            } => parent_session_id.as_str(),
        };
        self.get(parent_id)
    }

    /// Ancestor chain from root → … → `session_id` inclusive. Empty if not
    /// found.
    pub fn ancestors(&self, session_id: &str) -> Vec<&LoadedSessionNode> {
        let mut chain = Vec::new();
        let mut cur = self.get(session_id);
        while let Some(node) = cur {
            chain.push(node);
            cur = self.parent_of(&node.meta.session_id);
        }
        chain.reverse();
        chain
    }
}

/// Discover descendant sessions starting from `root_db` and load the tree.
///
/// `trace_path` may cover any subset of sessions in the tree; prompts are
/// partitioned by `context.session_id`. Sessions for which no prompts are
/// found in the trace render fine — they just don't show LLM-call rows.
pub fn load_tree_from_paths(root_db: &Path, trace_path: &Path) -> Result<LoadedSessionTree> {
    let prompts_all = load_prompts_from_trace(trace_path)?;
    let mut prompts_by_session: HashMap<String, Vec<LlmPromptSnapshot>> = HashMap::new();
    let mut prompts_unkeyed: Vec<LlmPromptSnapshot> = Vec::new();
    for prompt in prompts_all {
        match prompt.session_id.clone() {
            Some(sid) => prompts_by_session.entry(sid).or_default().push(prompt),
            None => prompts_unkeyed.push(prompt),
        }
    }

    let root_dir = root_db
        .parent()
        .ok_or_else(|| anyhow!("root db path has no parent dir: {}", root_db.display()))?
        .to_path_buf();

    // First pass: load every .db in the directory. Sessions without a
    // session_meta row get skipped silently.
    struct CandidateLoad {
        db_path: PathBuf,
        meta: SessionMeta,
        chronological: Vec<ChronologicalEntry>,
        context_window_tokens: Option<u64>,
    }
    let mut candidates: Vec<CandidateLoad> = Vec::new();
    let entries = fs::read_dir(&root_dir)
        .with_context(|| format!("scanning sessions dir {}", root_dir.display()))?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("db") {
            continue;
        }
        let Ok(store) = Store::open_readonly(&path) else {
            continue;
        };
        let Some(meta) = store.load_session_meta() else {
            continue;
        };
        let head = store.load_session_head();
        let context_window_tokens = head
            .as_ref()
            .map(|h| h.config.context_window)
            .filter(|t| *t > 0);
        let graph = head
            .map(|h| h.graph)
            .unwrap_or_else(|| store.load_session_graph());
        let chronological = build_chronological(graph);
        candidates.push(CandidateLoad {
            db_path: path,
            meta,
            chronological,
            context_window_tokens,
        });
    }

    // Locate the root by matching the root_db path.
    let root_idx = candidates
        .iter()
        .position(|c| same_file(&c.db_path, root_db))
        .ok_or_else(|| anyhow!("root db {} not found in sessions dir", root_db.display()))?;
    let root_id = candidates[root_idx].meta.session_id.clone();

    // Walk the parent chain to keep only sessions reachable from root.
    let by_id: HashMap<String, usize> = candidates
        .iter()
        .enumerate()
        .map(|(i, c)| (c.meta.session_id.clone(), i))
        .collect();

    let mut keep: Vec<bool> = vec![false; candidates.len()];
    keep[root_idx] = true;
    // BFS-style: a candidate stays if any ancestor is the root.
    loop {
        let mut changed = false;
        for (i, c) in candidates.iter().enumerate() {
            if keep[i] {
                continue;
            }
            let Some(parent_id) = c.meta.parent_session_id.as_deref() else {
                continue;
            };
            let Some(&parent_idx) = by_id.get(parent_id) else {
                continue;
            };
            if keep[parent_idx] {
                keep[i] = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    // Build child-of-parent index by walking each parent's chronological
    // for spawn_agent / continue_as tool calls.
    let mut node_kinds: HashMap<String, NodeRelation> = HashMap::new();
    let mut subagent_edges: HashMap<String, Vec<SubagentEdge>> = HashMap::new();
    let mut handoff_targets: HashMap<String, String> = HashMap::new();

    for (i, c) in candidates.iter().enumerate() {
        if !keep[i] {
            continue;
        }
        let parent_sid = c.meta.session_id.clone();
        let mut edges = Vec::new();
        for entry in &c.chronological {
            let ChronologicalPayload::ToolCall(record) = &entry.payload else {
                continue;
            };
            match record.tool.as_str() {
                "spawn_agent" => {
                    if let Some(child_id) = extract_session_id(&record.result) {
                        edges.push(SubagentEdge {
                            child_session_id: child_id.clone(),
                            agent_name: extract_str(&record.args, "agent_name"),
                            capability: extract_str(&record.args, "capability"),
                            call_id: record.call_id.clone(),
                            success: record.success,
                            duration_ms: record.duration_ms,
                        });
                        node_kinds.insert(
                            child_id,
                            NodeRelation::Subagent {
                                parent_session_id: parent_sid.clone(),
                                agent_name: extract_str(&record.args, "agent_name"),
                                capability: extract_str(&record.args, "capability"),
                                parent_call_id: record.call_id.clone(),
                            },
                        );
                    }
                }
                "continue_as" => {
                    if let Some(child_id) = extract_session_id(&record.result) {
                        handoff_targets.insert(parent_sid.clone(), child_id.clone());
                        node_kinds.insert(
                            child_id,
                            NodeRelation::Handoff {
                                parent_session_id: parent_sid.clone(),
                            },
                        );
                    }
                }
                _ => {}
            }
        }
        subagent_edges.insert(parent_sid, edges);
    }

    // Assemble the final node list.
    let mut nodes = Vec::new();
    for (i, c) in candidates.into_iter().enumerate() {
        if !keep[i] {
            continue;
        }
        let sid = c.meta.session_id.clone();
        let kind = if sid == root_id {
            NodeRelation::Root
        } else {
            // Prefer the kind we derived from the parent's tool call. If
            // the parent didn't carry a session_id in its tool result,
            // fall back to a generic Subagent (this is the safe default
            // since handoff is rare and continue_as always emits the
            // session_id).
            node_kinds.remove(&sid).unwrap_or(NodeRelation::Subagent {
                parent_session_id: c
                    .meta
                    .parent_session_id
                    .clone()
                    .unwrap_or_else(|| root_id.clone()),
                agent_name: None,
                capability: None,
                parent_call_id: None,
            })
        };
        let llm_prompts = prompts_by_session.remove(&sid).unwrap_or_else(|| {
            if sid == root_id {
                std::mem::take(&mut prompts_unkeyed)
            } else {
                Vec::new()
            }
        });
        nodes.push(LoadedSessionNode {
            meta: c.meta,
            chronological: c.chronological,
            context_window_tokens: c.context_window_tokens,
            llm_prompts,
            db_path: c.db_path,
            kind,
            subagent_children: subagent_edges.remove(&sid).unwrap_or_default(),
            handoff_successor: handoff_targets.remove(&sid),
        });
    }

    Ok(LoadedSessionTree {
        root_id,
        trace_path: trace_path.to_path_buf(),
        nodes,
    })
}

fn build_chronological(graph: SessionGraph) -> Vec<ChronologicalEntry> {
    let state = lash::SessionStateEnvelope {
        session_graph: graph,
        ..lash::SessionStateEnvelope::default()
    };
    state.read_view().chronological_projection().into_entries()
}

fn extract_str(value: &serde_json::Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
}

fn extract_session_id(value: &serde_json::Value) -> Option<String> {
    extract_str(value, "session_id")
}

fn same_file(a: &Path, b: &Path) -> bool {
    fs::canonicalize(a)
        .ok()
        .zip(fs::canonicalize(b).ok())
        .map(|(a, b)| a == b)
        .unwrap_or(false)
}
