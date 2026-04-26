//! Render persisted lash sessions into human-viewable formats.
//!
//! This crate is independent of `lash-cli`. It reads a session's SQLite
//! store, projects the `SessionGraph` into messages + tool calls, and
//! writes a self-contained HTML (or JSON) document.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::{Context, Result, anyhow};
use lash::session_model::Message;
use lash::{SessionGraph, SessionMeta, Store, ToolCallRecord};

pub mod html;
pub mod json;

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
    pub messages: Vec<Message>,
    pub tool_calls: Vec<ToolCallRecord>,
}

/// Load a session by its SQLite store path (the `.db` file).
pub fn load_session_from_path(store_path: &Path) -> Result<LoadedSession> {
    let store = Store::open_readonly(store_path)
        .with_context(|| format!("opening session store at {}", store_path.display()))?;
    let meta = store.load_session_meta();
    let graph = load_graph(&store);
    let messages = graph.project_conversation_messages();
    let tool_calls = graph.project_tool_calls();
    Ok(LoadedSession {
        meta,
        messages,
        tool_calls,
    })
}

/// Load a session by id — searches `sessions_dir` for a `.db` file
/// whose metadata matches. Host decides where sessions live (lash-cli
/// passes `crate::paths::lash_home().join("sessions")`).
pub fn load_session_by_id(sessions_dir: &Path, session_id: &str) -> Result<LoadedSession> {
    let path = resolve_session_path_by_id(sessions_dir, session_id)?;
    load_session_from_path(&path)
}

/// Render a loaded session to a string in the requested format.
pub fn render(session: &LoadedSession, format: ExportFormat) -> String {
    match format {
        ExportFormat::Html => html::render(session),
        ExportFormat::Json => json::render(session),
    }
}

/// End-to-end: load a session (by id or path) and write the rendered
/// output to disk. If `out` is `None`, returns the rendered string
/// instead of writing it. `sessions_dir` is consulted only for
/// `SessionSelector::Id`; pass any `Path` when the selector is
/// `Path(...)`.
pub fn export(
    selector: SessionSelector<'_>,
    sessions_dir: &Path,
    format: ExportFormat,
    out: Option<&Path>,
) -> Result<String> {
    let session = match selector {
        SessionSelector::Path(path) => load_session_from_path(path)?,
        SessionSelector::Id(id) => load_session_by_id(sessions_dir, id)?,
    };
    let rendered = render(&session, format);
    if let Some(path) = out {
        fs::write(path, &rendered)
            .with_context(|| format!("writing export to {}", path.display()))?;
    }
    Ok(rendered)
}

pub enum SessionSelector<'a> {
    Id(&'a str),
    Path(&'a Path),
}

fn load_graph(store: &Store) -> SessionGraph {
    if let Some(head) = store.load_session_head() {
        return head.graph;
    }
    // No committed head but graph nodes may still have been appended —
    // read the raw node stream directly.
    store.load_session_graph()
}

fn resolve_session_path_by_id(sessions_dir: &Path, session_id: &str) -> Result<PathBuf> {
    let entries = fs::read_dir(sessions_dir)
        .with_context(|| format!("reading sessions dir {}", sessions_dir.display()))?;

    let mut candidates: Vec<(PathBuf, SystemTime)> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("db") {
            continue;
        }
        let modified = fs::metadata(&path)
            .and_then(|meta| meta.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        candidates.push((path, modified));
    }
    candidates.sort_by_key(|entry| std::cmp::Reverse(entry.1));

    for (path, _) in candidates {
        let Ok(store) = Store::open_readonly(&path) else {
            continue;
        };
        let Some(meta) = store.load_session_meta() else {
            continue;
        };
        if meta.session_id == session_id {
            return Ok(path);
        }
    }

    Err(anyhow!(
        "no session with id `{session_id}` found under {}",
        sessions_dir.display()
    ))
}
