//! Multi-session export — discover descendant sessions reachable from a
//! root `.db` and classify each cross-session edge as either a `Handoff`
//! (continue_as) or a `Subagent` (spawn_agent).
//!
//! The on-disk shape is one `.db` per session in `sessions_dir/`, each
//! carrying its full session relation in `session_meta`. For subagents,
//! `SessionRelation::Child.originating_tool_call_id` anchors the child to
//! the parent `spawn_agent` call without relying on model-authored names.
//!
//! One provider trace JSONL covers the whole `lash` invocation. Each
//! record carries `context.session_id`; we partition `LlmPromptSnapshot`s
//! by that field so each session in the tree gets its matching prompts.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use lash_core::{
    ChronologicalEntry, ChronologicalPayload, SessionGraph, SessionMeta, ToolCallStatus,
};
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
        task: Option<String>,
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
    pub task: Option<String>,
    pub capability: Option<String>,
    pub call_id: Option<String>,
    pub status: ToolCallStatus,
    pub duration_ms: u64,
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
            .map(|h| h.config.model.context_window_tokens() as u64);
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

    // Build child-of-parent index from persisted relations, then use the
    // parent's chronological projection only to recover display metadata and
    // order edges at the exact parent call.
    let mut node_kinds: HashMap<String, NodeRelation> = HashMap::new();
    let mut subagent_edges: HashMap<String, Vec<SubagentEdge>> = HashMap::new();
    let mut handoff_targets: HashMap<String, String> = HashMap::new();

    for (i, c) in candidates.iter().enumerate() {
        if !keep[i] || c.meta.session_id == root_id {
            continue;
        }
        match &c.meta.relation {
            lash_core::SessionRelation::Root => {}
            lash_core::SessionRelation::Child {
                parent_session_id,
                originating_tool_call_id,
            } => {
                node_kinds.insert(
                    c.meta.session_id.clone(),
                    NodeRelation::Subagent {
                        parent_session_id: parent_session_id.clone(),
                        task: None,
                        capability: None,
                        parent_call_id: originating_tool_call_id.clone(),
                    },
                );
            }
            lash_core::SessionRelation::Handoff {
                parent_session_id, ..
            } => {
                handoff_targets.insert(parent_session_id.clone(), c.meta.session_id.clone());
                node_kinds.insert(
                    c.meta.session_id.clone(),
                    NodeRelation::Handoff {
                        parent_session_id: parent_session_id.clone(),
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
        let mut edges = Vec::new();
        for entry in &c.chronological {
            let ChronologicalPayload::ToolCall(record) = &entry.payload else {
                continue;
            };
            match record.tool.as_str() {
                "spawn_agent" => {
                    if let Some(child_id) =
                        find_spawn_child_for_record(&candidates, &keep, &parent_sid, record)
                    {
                        edges.push(SubagentEdge {
                            child_session_id: child_id.clone(),
                            task: extract_str(&record.args, "task"),
                            capability: extract_str(&record.args, "capability"),
                            call_id: record.call_id.clone(),
                            status: record.output.status(),
                            duration_ms: record.duration_ms,
                        });
                        node_kinds.insert(
                            child_id,
                            NodeRelation::Subagent {
                                parent_session_id: parent_sid.clone(),
                                task: extract_str(&record.args, "task"),
                                capability: extract_str(&record.args, "capability"),
                                parent_call_id: record.call_id.clone(),
                            },
                        );
                    }
                }
                "continue_as" => {
                    if let Some(child_id) =
                        extract_session_id(&record.output.value_for_projection())
                    {
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
        for (child_idx, child) in candidates.iter().enumerate() {
            if !keep[child_idx] {
                continue;
            }
            let lash_core::SessionRelation::Child {
                parent_session_id,
                originating_tool_call_id,
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
                call_id: originating_tool_call_id.clone(),
                status: ToolCallStatus::Success,
                duration_ms: 0,
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
            // Prefer the kind we derived from the parent's tool call. If
            // the parent didn't carry a session_id in its tool result,
            // fall back to a generic Subagent (this is the safe default
            // since handoff is rare and continue_as always emits the
            // session_id).
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
    let state = lash_core::SessionStateEnvelope {
        session_graph: graph,
        ..lash_core::SessionStateEnvelope::default()
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

fn find_spawn_child_for_record(
    candidates: &[CandidateLoad],
    keep: &[bool],
    parent_session_id: &str,
    record: &lash_core::ToolCallRecord,
) -> Option<String> {
    if let Some(call_id) = &record.call_id {
        for (idx, candidate) in candidates.iter().enumerate() {
            if !keep[idx] {
                continue;
            }
            let lash_core::SessionRelation::Child {
                parent_session_id: child_parent,
                originating_tool_call_id,
            } = &candidate.meta.relation
            else {
                continue;
            };
            if child_parent == parent_session_id
                && originating_tool_call_id.as_ref() == Some(call_id)
            {
                return Some(candidate.meta.session_id.clone());
            }
        }
    }
    extract_session_id(&record.output.value_for_projection())
}

fn same_file(a: &Path, b: &Path) -> bool {
    fs::canonicalize(a)
        .ok()
        .zip(fs::canonicalize(b).ok())
        .map(|(a, b)| a == b)
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lash_core::{ToolCallOutput, ToolCallRecord};
    use serde_json::json;

    fn meta(session_id: &str, relation: lash_core::SessionRelation) -> SessionMeta {
        SessionMeta {
            session_id: session_id.to_string(),
            session_name: session_id.to_string(),
            created_at: "unix:0".to_string(),
            model: "test-model".to_string(),
            cwd: None,
            relation,
        }
    }

    fn candidate(session_id: &str, relation: lash_core::SessionRelation) -> CandidateLoad {
        CandidateLoad {
            db_path: PathBuf::from(format!("{session_id}.db")),
            meta: meta(session_id, relation),
            chronological: Vec::new(),
            context_window_tokens: None,
        }
    }

    #[test]
    fn spawn_child_lookup_uses_persisted_originating_call_id() {
        let candidates = vec![
            candidate("root", lash_core::SessionRelation::Root),
            candidate(
                "child-from-relation",
                lash_core::SessionRelation::Child {
                    parent_session_id: "root".to_string(),
                    originating_tool_call_id: Some("call-7".to_string()),
                },
            ),
        ];
        let keep = vec![true, true];
        let record = ToolCallRecord {
            call_id: Some("call-7".to_string()),
            tool: "spawn_agent".to_string(),
            args: json!({
                "task": "inspect relation anchoring",
                "capability": "explore"
            }),
            output: ToolCallOutput::success(json!({ "result": "done" })),
            duration_ms: 12,
        };

        assert_eq!(
            find_spawn_child_for_record(&candidates, &keep, "root", &record),
            Some("child-from-relation".to_string())
        );
    }
}
