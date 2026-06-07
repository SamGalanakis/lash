//! Multi-session export — discover descendant sessions reachable from a
//! root `.db` and classify cross-session edges created by `spawn_agent`.
//!
//! The on-disk shape is one `.db` per session in `sessions_dir/`, each
//! carrying its full session relation in `session_meta`. For subagents,
//! `SessionRelation::Child.caused_by` anchors the child to
//! the parent `spawn_agent` call without relying on model-authored names.
//!
//! One provider trace JSONL covers the whole `lash` invocation. Each
//! record carries `context.session_id`; we partition `LlmPromptSnapshot`s
//! by that field so each session in the tree gets its matching prompts.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use lash_core::{ChronologicalEntry, SessionGraph, SessionMeta};
use lash_turso_store::Store;

use crate::trace::{LlmPromptSnapshot, load_prompts_from_trace};

/// One session in the tree, ready to render.
pub struct LoadedSessionNode {
    pub meta: SessionMeta,
    pub chronological: Vec<ChronologicalEntry>,
    pub context_window_tokens: Option<u64>,
    pub llm_prompts: Vec<LlmPromptSnapshot>,
    pub db_path: PathBuf,
    pub kind: NodeRelation,
    /// Sessions that this session spawned, in persisted relation order.
    pub subagent_children: Vec<SubagentEdge>,
}

/// What kind of edge points *into* this session from its parent.
#[derive(Clone, Debug)]
pub enum NodeRelation {
    Root,
    /// Parent called `spawn_agent` and waited for this session to submit.
    Subagent {
        parent_session_id: String,
        task: Option<String>,
        capability: Option<String>,
        /// The parent's causal tool call id, when available.
        parent_call_id: Option<String>,
    },
}

/// Records one `spawn_agent` edge from a parent to a child session.
#[derive(Clone, Debug)]
pub struct SubagentEdge {
    pub child_session_id: String,
    pub task: Option<String>,
    pub capability: Option<String>,
    pub call_id: Option<String>,
}

/// The full discovered tree.
pub struct LoadedSessionTree {
    pub root_id: String,
    pub trace_path: PathBuf,
    pub nodes: Vec<LoadedSessionNode>,
}

struct CandidateLoad {
    db_path: PathBuf,
    meta: SessionMeta,
    chronological: Vec<ChronologicalEntry>,
    context_window_tokens: Option<u64>,
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
pub async fn load_tree_from_paths(root_db: &Path, trace_path: &Path) -> Result<LoadedSessionTree> {
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
    let mut candidates: Vec<CandidateLoad> = Vec::new();
    let entries = fs::read_dir(&root_dir)
        .with_context(|| format!("scanning sessions dir {}", root_dir.display()))?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("db") {
            continue;
        }
        let Ok(store) = Store::open_readonly(&path).await else {
            continue;
        };
        let Some(meta) = store.load_session_meta().await else {
            continue;
        };
        let head = store.load_session_head().await;
        let context_window_tokens = head
            .as_ref()
            .map(|h| h.config.model.context_window_tokens() as u64);
        let graph = match head {
            Some(head) => head.graph,
            None => store.load_session_graph().await,
        };
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
            let Some(parent_id) = c.meta.parent_session_id() else {
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

    // Build child-of-parent index from persisted relations. The parent call
    // id survives as causal metadata, but detached invocation history is no
    // longer part of the session graph.
    let mut node_kinds: HashMap<String, NodeRelation> = HashMap::new();
    let mut subagent_edges: HashMap<String, Vec<SubagentEdge>> = HashMap::new();

    for (i, c) in candidates.iter().enumerate() {
        if !keep[i] || c.meta.session_id == root_id {
            continue;
        }
        match &c.meta.relation {
            lash_core::SessionRelation::Root => {}
            lash_core::SessionRelation::Child {
                parent_session_id,
                caused_by,
            } => {
                node_kinds.insert(
                    c.meta.session_id.clone(),
                    NodeRelation::Subagent {
                        parent_session_id: parent_session_id.clone(),
                        task: None,
                        capability: None,
                        parent_call_id: tool_call_id_from_cause(caused_by),
                    },
                );
            }
        }
    }

    for (i, c) in candidates.iter().enumerate() {
        if !keep[i] {
            continue;
        }
        let parent_sid = c.meta.session_id.clone();
        let mut edges: Vec<SubagentEdge> = Vec::new();
        for (child_idx, child) in candidates.iter().enumerate() {
            if !keep[child_idx] {
                continue;
            }
            let lash_core::SessionRelation::Child {
                parent_session_id,
                caused_by,
            } = &child.meta.relation
            else {
                continue;
            };
            if parent_session_id != &parent_sid {
                continue;
            }
            if edges
                .iter()
                .any(|edge| edge.child_session_id == child.meta.session_id)
            {
                continue;
            }
            edges.push(SubagentEdge {
                child_session_id: child.meta.session_id.clone(),
                task: None,
                capability: None,
                call_id: tool_call_id_from_cause(caused_by),
            });
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
            node_kinds.remove(&sid).unwrap_or(NodeRelation::Subagent {
                parent_session_id: c
                    .meta
                    .parent_session_id()
                    .map(ToOwned::to_owned)
                    .unwrap_or_else(|| root_id.clone()),
                task: None,
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
        });
    }

    Ok(LoadedSessionTree {
        root_id,
        trace_path: trace_path.to_path_buf(),
        nodes,
    })
}

fn build_chronological(graph: SessionGraph) -> Vec<ChronologicalEntry> {
    let state = lash_core::SessionSnapshot {
        session_graph: graph,
        ..lash_core::SessionSnapshot::default()
    };
    state.read_view().chronological_projection().into_entries()
}

fn tool_call_id_from_cause(caused_by: &Option<lash_core::CausalRef>) -> Option<String> {
    match caused_by {
        Some(lash_core::CausalRef::ToolCall { call_id, .. }) => Some(call_id.clone()),
        _ => None,
    }
}

fn same_file(a: &Path, b: &Path) -> bool {
    fs::canonicalize(a)
        .ok()
        .zip(fs::canonicalize(b).ok())
        .map(|(a, b)| a == b)
        .unwrap_or(false)
}
