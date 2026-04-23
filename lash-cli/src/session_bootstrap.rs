use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use lash::{PersistedSessionConfig, SessionGraph, SessionHead, Store};

use crate::session_log::{self, SessionLogger, SessionStart};

pub(crate) enum SessionBootstrapSource {
    Fresh,
    Resume(String),
    ForkChild { parent_session_id: String },
}

pub(crate) struct SessionBootstrap {
    source: SessionBootstrapSource,
    sessions_dir: PathBuf,
    filename: String,
    db_path: PathBuf,
    store: Arc<Store>,
    resume_start: Option<SessionStart>,
    resume_head: Option<SessionHead>,
    session_name: String,
}

impl SessionBootstrapSource {
    pub(crate) fn from_resume_arg(resume: Option<String>) -> Self {
        match resume {
            Some(identifier) => Self::Resume(
                session_log::filename_for_session_identifier(&identifier).unwrap_or(identifier),
            ),
            None => Self::Fresh,
        }
    }
}

impl SessionBootstrap {
    pub(crate) fn open(source: SessionBootstrapSource) -> Result<Self> {
        let sessions_dir = session_log::sessions_dir();
        std::fs::create_dir_all(&sessions_dir)?;
        let filename = match &source {
            SessionBootstrapSource::Fresh | SessionBootstrapSource::ForkChild { .. } => {
                session_log::new_session_filename()
            }
            SessionBootstrapSource::Resume(filename) => filename.clone(),
        };
        let db_path = sessions_dir.join(&filename);
        let store = Arc::new(Store::open(&db_path)?);
        let resume_start = if matches!(source, SessionBootstrapSource::Resume(_)) {
            store.load_session_meta().map(|meta| SessionStart {
                session_id: meta.session_id,
                session_name: meta.session_name,
            })
        } else {
            None
        };
        let session_name = resume_start
            .as_ref()
            .map(|start| start.session_name.clone())
            .unwrap_or_else(|| crate::generate_session_name(&sessions_dir));
        let resume_head = if matches!(source, SessionBootstrapSource::Resume(_)) {
            store.load_session_head()
        } else {
            None
        };
        Ok(Self {
            source,
            sessions_dir,
            filename,
            db_path,
            store,
            resume_start,
            resume_head,
            session_name,
        })
    }

    pub(crate) fn sessions_dir(&self) -> &Path {
        &self.sessions_dir
    }

    pub(crate) fn filename(&self) -> &str {
        &self.filename
    }

    pub(crate) fn db_path(&self) -> &Path {
        &self.db_path
    }

    pub(crate) fn store(&self) -> Arc<Store> {
        Arc::clone(&self.store)
    }

    pub(crate) fn run_session_id(&self) -> Option<String> {
        self.resume_start
            .as_ref()
            .map(|start| start.session_id.clone())
            .or_else(|| Some(uuid::Uuid::new_v4().to_string()))
    }

    pub(crate) fn persisted_config(&self) -> Option<PersistedSessionConfig> {
        self.resume_head.as_ref().map(|head| head.config.clone())
    }

    pub(crate) fn initial_graph(&self) -> SessionGraph {
        self.resume_head
            .as_ref()
            .map(|head| head.graph.clone())
            .unwrap_or_default()
    }

    pub(crate) fn session_name(&self) -> String {
        self.session_name.clone()
    }

    pub(crate) fn logger(&self, model: &str, session_id: Option<String>) -> Result<SessionLogger> {
        match &self.source {
            SessionBootstrapSource::Resume(_) => {
                SessionLogger::resume(Arc::clone(&self.store), &self.filename)
            }
            SessionBootstrapSource::Fresh => SessionLogger::new(
                Arc::clone(&self.store),
                self.filename.clone(),
                model,
                session_id,
                self.session_name(),
            ),
            SessionBootstrapSource::ForkChild { parent_session_id } => {
                let logger = SessionLogger::new(
                    Arc::clone(&self.store),
                    self.filename.clone(),
                    model,
                    session_id,
                    self.session_name(),
                )?;
                logger.mark_as_child_of(parent_session_id)?;
                Ok(logger)
            }
        }
    }

    pub(crate) fn fork_child(parent_session_id: &str, model: &str) -> Result<Self> {
        let bootstrap = Self::open(SessionBootstrapSource::ForkChild {
            parent_session_id: parent_session_id.to_string(),
        })?;
        let _logger = bootstrap.logger(model, Some(uuid::Uuid::new_v4().to_string()))?;
        Ok(bootstrap)
    }
}
