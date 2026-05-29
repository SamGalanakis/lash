use std::sync::Arc;

use arc_swap::ArcSwap;
use tokio::sync::Mutex;

use super::{LashRuntime, ProcessHandleGrantEntry, ProcessRegistry};

#[derive(Clone)]
pub struct RuntimeObservation {
    pub session_id: Arc<str>,
    pub policy: crate::SessionPolicy,
    pub read_view: crate::SessionReadView,
    pub persisted_state: super::RuntimeSessionState,
    pub usage_report: super::SessionUsageReport,
    pub tool_state: Option<crate::ToolState>,
    pub tool_catalog: Arc<Vec<serde_json::Value>>,
    pub tool_catalog_error: Option<String>,
    pub process_registry: Option<Arc<dyn ProcessRegistry>>,
    pub queue_store: Option<Arc<dyn crate::RuntimePersistence>>,
}

impl RuntimeObservation {
    fn from_runtime(runtime: &LashRuntime, previous: Option<&RuntimeObservation>) -> Self {
        let (tool_catalog, tool_catalog_error) = match runtime.active_tool_catalog_shared() {
            Ok(catalog) => (catalog, None),
            Err(err) => (Arc::new(Vec::new()), Some(err.to_string())),
        };
        let tool_state_generation = runtime
            .session
            .as_ref()
            .map(|session| session.plugins().tool_registry().generation());
        let tool_state = match (
            tool_state_generation,
            previous.and_then(|observation| observation.tool_state.as_ref()),
        ) {
            (Some(generation), Some(snapshot)) if snapshot.generation() == generation => {
                Some(snapshot.clone())
            }
            (Some(_), _) => runtime.tool_state().ok(),
            (None, _) => None,
        };
        Self {
            session_id: Arc::from(runtime.session_id()),
            policy: runtime.read_view().policy().clone(),
            read_view: runtime.read_view(),
            persisted_state: runtime.export_persisted_state(),
            usage_report: runtime.usage_report(),
            tool_state,
            tool_catalog,
            tool_catalog_error,
            process_registry: runtime.host.process_registry.clone(),
            queue_store: runtime
                .session
                .as_ref()
                .and_then(|session| session.history_store()),
        }
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn process_scope(&self) -> crate::ProcessScope {
        crate::ProcessScope::new(self.session_id.as_ref())
    }

    pub fn process_scope_id(&self) -> crate::ProcessScopeId {
        self.process_scope().id()
    }

    pub async fn list_process_handles(&self) -> Vec<ProcessHandleGrantEntry> {
        let Some(executor) = self.process_registry.as_ref() else {
            return Vec::new();
        };
        let root_scope = self.process_scope();
        let mut entries = executor
            .list_handle_grants(&root_scope)
            .await
            .unwrap_or_default();
        let agent_frame_id = self.persisted_state.current_agent_frame_id.as_str();
        if !agent_frame_id.is_empty() {
            let frame_scope =
                crate::ProcessScope::for_agent_frame(self.session_id.as_ref(), agent_frame_id);
            if frame_scope.id() != root_scope.id() {
                entries.extend(
                    executor
                        .list_handle_grants(&frame_scope)
                        .await
                        .unwrap_or_default(),
                );
                entries.sort_by(|(left, _), (right, _)| left.process_id.cmp(&right.process_id));
                entries.dedup_by(|(left, _), (right, _)| left.process_id == right.process_id);
            }
        }
        entries
    }
}

#[derive(Clone)]
pub struct RuntimeHandle {
    pub(in crate::runtime) runtime: Arc<Mutex<LashRuntime>>,
    observation: Arc<ArcSwap<RuntimeObservation>>,
}

impl RuntimeHandle {
    pub fn new(runtime: LashRuntime) -> Self {
        let observation = RuntimeObservation::from_runtime(&runtime, None);
        Self {
            runtime: Arc::new(Mutex::new(runtime)),
            observation: Arc::new(ArcSwap::from_pointee(observation)),
        }
    }

    pub fn writer(&self) -> Arc<Mutex<LashRuntime>> {
        Arc::clone(&self.runtime)
    }

    pub fn observe(&self) -> Arc<RuntimeObservation> {
        self.observation.load_full()
    }

    pub fn publish_from(&self, runtime: &LashRuntime) {
        let previous = self.observation.load_full();
        self.observation
            .store(Arc::new(RuntimeObservation::from_runtime(
                runtime,
                Some(previous.as_ref()),
            )));
    }

    pub fn try_into_runtime(self) -> Result<LashRuntime, Self> {
        match Arc::try_unwrap(self.runtime) {
            Ok(mutex) => Ok(mutex.into_inner()),
            Err(runtime) => Err(Self {
                runtime,
                observation: self.observation,
            }),
        }
    }
}
