//! Render persisted lash sessions into human-viewable formats.
//!
//! This crate is independent of `lash-cli`. It reads a session's Sqlite
//! store, projects the `SessionGraph` into messages + tool calls, and
//! writes a self-contained HTML (or JSON) document.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use lash_core::{ChronologicalEntry, SessionGraph, SessionMeta, SessionSnapshot};
use lash_sqlite_store::Store;

pub mod html;
pub mod json;
pub mod markdown;
pub mod trace;
pub mod tree;

pub use trace::LlmPromptSnapshot;
pub use tree::{
    LoadedSessionNode, LoadedSessionTree, NodeRelation, SubagentEdge, load_tree_from_paths,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExportFormat {
    Html,
    Json,
}

impl ExportFormat {
    pub fn parse(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "html" => Ok(Self::Html),
            "json" => Ok(Self::Json),
            other => Err(anyhow!(
                "unknown export format `{other}` (expected html|json)"
            )),
        }
    }
}

/// A loaded session ready to be rendered.
pub struct LoadedSession {
    pub meta: Option<SessionMeta>,
    pub chronological: Vec<ChronologicalEntry>,
    pub trace_path: PathBuf,
    pub context_window_tokens: Option<u64>,
    /// One snapshot per `llm_call_started` event found in the required
    /// provider trace, in trace order.
    pub llm_prompts: Vec<LlmPromptSnapshot>,
}

/// Load a session by its Sqlite store path and full provider trace path.
pub async fn load_session_from_paths(
    store_path: &Path,
    trace_path: &Path,
) -> Result<LoadedSession> {
    let store = Store::open_readonly(store_path)
        .await
        .with_context(|| format!("opening session store at {}", store_path.display()))?;
    let meta = store.load_session_meta().await;
    let head = store.load_session_head().await;
    let context_window_tokens = head
        .as_ref()
        .map(|head| head.config.model.context_window_tokens() as u64);
    let graph = match head {
        Some(head) => head.graph,
        None => load_graph(&store).await,
    };
    let state = SessionSnapshot {
        session_graph: graph,
        ..SessionSnapshot::default()
    };
    let chronological = state.read_view().chronological_projection().into_entries();
    let llm_prompts = trace::load_prompts_from_trace(trace_path)?;
    Ok(LoadedSession {
        meta,
        chronological,
        trace_path: trace_path.to_path_buf(),
        context_window_tokens,
        llm_prompts,
    })
}

/// Render a loaded session to a string in the requested format.
pub fn render(session: &LoadedSession, format: ExportFormat) -> String {
    match format {
        ExportFormat::Html => html::render(session),
        ExportFormat::Json => json::render(session),
    }
}

/// Render a multi-session tree. Currently html-only; json falls back to
/// rendering the root session alone.
pub fn render_tree(tree: &LoadedSessionTree, format: ExportFormat) -> String {
    match format {
        ExportFormat::Html => html::render_tree(tree),
        ExportFormat::Json => {
            let root_session = LoadedSession {
                meta: Some(tree.root().meta.clone()),
                chronological: tree.root().chronological.clone(),
                trace_path: tree.trace_path.clone(),
                context_window_tokens: tree.root().context_window_tokens,
                llm_prompts: tree.root().llm_prompts.clone(),
            };
            json::render(&root_session)
        }
    }
}

/// End-to-end: load a session DB plus full provider trace and write the
/// rendered output to disk. If `out` is `None`, returns the rendered string
/// instead of writing it.
///
/// Multi-session aware: walks the sessions directory next to `store_path`
/// for descendants reachable via `parent_session_id`. If any are found the
/// html exporter renders them as a tree of views with breadcrumb navigation;
/// otherwise it falls back to single-session rendering.
pub async fn export(
    store_path: &Path,
    trace_path: &Path,
    format: ExportFormat,
    out: Option<&Path>,
) -> Result<String> {
    let rendered = match format {
        ExportFormat::Html => {
            // Try multi-session first; fall back to single-session for
            // sessions whose .db isn't co-located in a sessions dir
            // (e.g. ad-hoc per-run artifacts under .benchmarks/...).
            match load_tree_from_paths(store_path, trace_path).await {
                Ok(tree) if tree.nodes.len() > 1 => render_tree(&tree, format),
                Ok(_) => {
                    let session = load_session_from_paths(store_path, trace_path).await?;
                    render(&session, format)
                }
                Err(_) => {
                    let session = load_session_from_paths(store_path, trace_path).await?;
                    render(&session, format)
                }
            }
        }
        ExportFormat::Json => {
            let session = load_session_from_paths(store_path, trace_path).await?;
            render(&session, format)
        }
    };
    if let Some(path) = out {
        fs::write(path, &rendered)
            .with_context(|| format!("writing export to {}", path.display()))?;
    }
    Ok(rendered)
}

async fn load_graph(store: &Store) -> SessionGraph {
    if let Some(head) = store.load_session_head().await {
        return head.graph;
    }
    // No committed head but graph nodes may still have been appended —
    // read the raw node stream directly.
    store.load_session_graph().await
}
