//! Render persisted lash sessions into human-viewable formats.
//!
//! This crate is independent of `lash-cli`. It reads a session's SQLite
//! store, projects the `SessionGraph` into messages + tool calls, and
//! writes a self-contained HTML (or JSON) document.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use lash::{ChronologicalEntry, SessionGraph, SessionMeta, SessionStateEnvelope};
use lash_sqlite_store::Store;

pub mod html;
pub mod json;
pub mod markdown;
pub mod trace;

pub use trace::LlmPromptSnapshot;

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

/// Load a session by its SQLite store path and full provider trace path.
pub fn load_session_from_paths(store_path: &Path, trace_path: &Path) -> Result<LoadedSession> {
    let store = Store::open_readonly(store_path)
        .with_context(|| format!("opening session store at {}", store_path.display()))?;
    let meta = store.load_session_meta();
    let head = store.load_session_head();
    let context_window_tokens = head
        .as_ref()
        .map(|head| head.config.context_window)
        .filter(|tokens| *tokens > 0);
    let graph = head
        .map(|head| head.graph)
        .unwrap_or_else(|| load_graph(&store));
    let state = SessionStateEnvelope {
        session_graph: graph,
        ..SessionStateEnvelope::default()
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

/// End-to-end: load a session DB plus full provider trace and write the
/// rendered output to disk. If `out` is `None`, returns the rendered string
/// instead of writing it.
pub fn export(
    store_path: &Path,
    trace_path: &Path,
    format: ExportFormat,
    out: Option<&Path>,
) -> Result<String> {
    let session = load_session_from_paths(store_path, trace_path)?;
    let rendered = render(&session, format);
    if let Some(path) = out {
        fs::write(path, &rendered)
            .with_context(|| format!("writing export to {}", path.display()))?;
    }
    Ok(rendered)
}

fn load_graph(store: &Store) -> SessionGraph {
    if let Some(head) = store.load_session_head() {
        return head.graph;
    }
    // No committed head but graph nodes may still have been appended —
    // read the raw node stream directly.
    store.load_session_graph()
}
