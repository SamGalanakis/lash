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
    pub runtime_scope_id: Arc<str>,
    pub process_registry: Option<Arc<dyn ProcessRegistry>>,
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
            runtime_scope_id: Arc::clone(&runtime.runtime_scope_id),
            process_registry: runtime.host.process_registry.clone(),
        }
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn process_scope_key(&self) -> String {
        format!("{}:{}", self.runtime_scope_id, self.session_id)
    }

    pub async fn list_process_handles(&self) -> Vec<ProcessHandleGrantEntry> {
        let Some(executor) = self.process_registry.as_ref() else {
            return Vec::new();
        };
        executor
            .list_handle_grants(&self.process_scope_key())
            .await
            .unwrap_or_default()
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
