use std::sync::Arc;

use arc_swap::ArcSwap;
use tokio::sync::Mutex;

use super::{LashRuntime, ManagedTaskStatus, SessionTaskExecutor};

#[derive(Clone)]
pub struct RuntimeObservation {
    pub session_id: Arc<str>,
    pub policy: crate::SessionPolicy,
    pub read_view: crate::SessionReadView,
    pub persisted_state: super::PersistedSessionState,
    pub usage_report: super::SessionUsageReport,
    pub tool_state: Option<crate::ToolState>,
    pub tool_catalog: Arc<Vec<serde_json::Value>>,
    pub runtime_scope_id: Arc<str>,
    pub session_task_executor: Option<Arc<dyn SessionTaskExecutor>>,
}

impl RuntimeObservation {
    fn from_runtime(runtime: &LashRuntime) -> Self {
        Self {
            session_id: Arc::from(runtime.session_id()),
            policy: runtime.read_view().policy().clone(),
            read_view: runtime.read_view(),
            persisted_state: runtime.export_persisted_state(),
            usage_report: runtime.usage_report(),
            tool_state: runtime.tool_state().ok(),
            tool_catalog: runtime.active_tool_catalog_shared(),
            runtime_scope_id: Arc::clone(&runtime.runtime_scope_id),
            session_task_executor: runtime.host.session_task_executor.clone(),
        }
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn background_scope_key(&self) -> String {
        format!("{}:{}", self.runtime_scope_id, self.session_id)
    }

    pub async fn list_background_tasks(&self) -> Vec<ManagedTaskStatus> {
        let Some(executor) = self.session_task_executor.as_ref() else {
            return Vec::new();
        };
        executor.list_managed(&self.background_scope_key()).await
    }
}

#[derive(Clone)]
pub struct RuntimeHandle {
    pub(in crate::runtime) runtime: Arc<Mutex<LashRuntime>>,
    observation: Arc<ArcSwap<RuntimeObservation>>,
}

impl RuntimeHandle {
    pub fn new(runtime: LashRuntime) -> Self {
        let observation = RuntimeObservation::from_runtime(&runtime);
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
        self.observation
            .store(Arc::new(RuntimeObservation::from_runtime(runtime)));
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
