//! Shared process-level infrastructure for lash embedders.
//!
//! `RuntimeEnvironment` is the type an embedder constructs ONCE at
//! startup and reuses across every `LashRuntime` instance it spawns.
//! Fields are all `Arc`-wrapped or cheap-to-clone so building a runtime
//! from an environment never rebuilds expensive state (plugin host,
//! prompt layer, …).
//!
//! Three embedder patterns this enables:
//!
//! * **CLI interactive (single runtime, default):** `RuntimeEnvironment::default()`.
//!   Behaviour byte-identical to the pre-environment world.
//! * **Long autonomous agent:** set `residency` to `ActivePathOnly`,
//!   then have the host periodically call
//!   `runtime.orphaned_node_ids()` + `store.tombstone_nodes(...)` +
//!   `store.vacuum()` on its own schedule. lash owns RAM; the host owns
//!   disk lifecycle.
//! * **Webserver multi-tenant:** one `RuntimeEnvironment` per process,
//!   `residency: ActivePathOnly`, and `park()` / `resume()` per
//!   request. HTTP connection pooling is a provider concern —
//!   provider crates accept an optional shared HTTP client in
//!   their constructors, so the host can share one pool across every
//!   materialized provider.

use std::path::PathBuf;
use std::sync::Arc;

use lash_trace::{JsonlTraceSink, TraceContext, TraceLevel, TraceSink};

use super::TerminationPolicy;
use super::host::{RuntimeCoreConfig, SessionTaskExecutor};

/// Where session nodes live at runtime.
///
/// lash owns RAM; the host owns disk lifecycle. Under `ActivePathOnly`
/// the runtime trims orphans from memory on load, but disk-side
/// retention (tombstoning + vacuum) is the host's decision — call
/// `LashRuntime::orphaned_node_ids` when you want the current orphan
/// set and feed it into `store.tombstone_nodes` / `store.vacuum` on
/// whatever schedule fits your deployment.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Residency {
    /// Every node resident in RAM. Default. Best for interactive /
    /// branching UX where the user may rewind.
    #[default]
    KeepAll,
    /// Only nodes reachable from `leaf_node_id` are resident. Orphans
    /// live on disk and are loaded on demand via
    /// `LashRuntime::get_historic_node`. Best for webserver embedders
    /// with many concurrent idle sessions, and for autonomous agents
    /// (combined with host-scheduled `tombstone_nodes` + `vacuum`).
    ActivePathOnly,
}

/// Shared runtime infrastructure an embedder builds once and reuses
/// across every `LashRuntime` it constructs.
///
/// Cloning is cheap — every field is either `Arc`-wrapped or small.
/// Default values preserve legacy behaviour so existing embedders can
/// adopt incrementally.
#[derive(Clone)]
pub struct RuntimeEnvironment {
    // Shared plugin infrastructure. Created once; every session's
    // `PluginSession` is built from it via `PluginHost::build_session`.
    pub plugin_host: Option<Arc<crate::PluginHost>>,

    // RAM footprint policy for the session graph. Default `KeepAll`
    // matches legacy behaviour. Webserver and autonomous-agent
    // embedders set `ActivePathOnly`; disk lifecycle is then the
    // host's responsibility via `orphaned_node_ids` + `tombstone_nodes`
    // + `vacuum`.
    pub residency: Residency,

    // Background task executor for `PluginSessionTask` futures.
    pub session_task_executor: Option<Arc<dyn SessionTaskExecutor>>,

    // Store factory used by managed child sessions created from runtimes
    // built with this environment.
    pub session_store_factory: Option<Arc<dyn crate::SessionStoreFactory>>,

    // All fields below mirror `RuntimeCoreConfig` and carry the same
    // semantics. They live on `RuntimeEnvironment` directly so
    // embedders don't have to build a separate core config.
    pub attachment_store: Arc<dyn crate::AttachmentStore>,
    pub prompt: crate::PromptLayer,
    pub trace_sink: Option<Arc<dyn TraceSink>>,
    pub trace_level: TraceLevel,
    pub trace_context: TraceContext,
    pub termination: TerminationPolicy,
}

impl Default for RuntimeEnvironment {
    fn default() -> Self {
        Self {
            plugin_host: None,
            residency: Residency::default(),
            session_task_executor: None,
            session_store_factory: None,
            attachment_store: Arc::new(crate::InMemoryAttachmentStore::new()),
            prompt: crate::PromptLayer::new(),
            trace_sink: None,
            trace_level: TraceLevel::Standard,
            trace_context: TraceContext::default(),
            termination: TerminationPolicy::default(),
        }
    }
}

impl RuntimeEnvironment {
    pub fn builder() -> RuntimeEnvironmentBuilder {
        RuntimeEnvironmentBuilder::default()
    }

    /// Materialize the legacy `RuntimeCoreConfig` view of this
    /// environment. Used by the existing `from_persistent_*_state` code
    /// paths during the migration to `from_environment`.
    pub fn to_runtime_core_config(&self) -> RuntimeCoreConfig {
        RuntimeCoreConfig {
            attachment_store: Arc::clone(&self.attachment_store),
            prompt: self.prompt.clone(),
            trace_sink: self.trace_sink.clone(),
            trace_level: self.trace_level,
            trace_context: self.trace_context.clone(),
            termination: self.termination.clone(),
        }
    }
}

/// Lightweight handle returned by `LashRuntime::park`. Holds no graph
/// nodes, no plugin session, no HTTP client — just enough to
/// `LashRuntime::resume` later. Cheap to cache per-session on a
/// webserver; bounded memory cost regardless of session history size.
pub struct ParkedSession {
    pub(crate) session_id: String,
    pub(crate) store: Arc<dyn crate::store::RuntimePersistence>,
    pub(crate) policy: crate::SessionPolicy,
}

impl ParkedSession {
    pub fn session_id(&self) -> &str {
        &self.session_id
    }
}

/// Fluent builder for `RuntimeEnvironment`.
#[derive(Default)]
pub struct RuntimeEnvironmentBuilder {
    env: RuntimeEnvironment,
}

impl RuntimeEnvironmentBuilder {
    pub fn with_plugin_host(mut self, host: Arc<crate::PluginHost>) -> Self {
        self.env.plugin_host = Some(if self.env.session_task_executor.is_some() {
            Arc::new(host.as_ref().clone().with_background_tasks())
        } else {
            host
        });
        self
    }

    pub fn with_residency(mut self, residency: Residency) -> Self {
        self.env.residency = residency;
        self
    }

    pub fn with_session_task_executor(mut self, executor: Arc<dyn SessionTaskExecutor>) -> Self {
        self.env.session_task_executor = Some(executor);
        if let Some(host) = self.env.plugin_host.take() {
            self.env.plugin_host = Some(Arc::new(host.as_ref().clone().with_background_tasks()));
        }
        self
    }

    pub fn with_session_store_factory(
        mut self,
        factory: Arc<dyn crate::SessionStoreFactory>,
    ) -> Self {
        self.env.session_store_factory = Some(factory);
        self
    }

    pub fn with_attachment_store(mut self, store: Arc<dyn crate::AttachmentStore>) -> Self {
        self.env.attachment_store = store;
        self
    }

    pub fn with_prompt_template(mut self, template: crate::PromptTemplate) -> Self {
        self.env.prompt.template = Some(template);
        self
    }

    pub fn with_prompt_contribution(mut self, contribution: crate::PromptContribution) -> Self {
        self.env.prompt.add_contribution(contribution);
        self
    }

    pub fn with_replaced_prompt_slot(
        mut self,
        slot: crate::PromptSlot,
        contributions: impl IntoIterator<Item = crate::PromptContribution>,
    ) -> Self {
        self.env.prompt.replace_slot(slot, contributions);
        self
    }

    pub fn with_cleared_prompt_slot(mut self, slot: crate::PromptSlot) -> Self {
        self.env.prompt.clear_slot(slot);
        self
    }

    pub fn with_prompt_layer(mut self, prompt: crate::PromptLayer) -> Self {
        self.env.prompt = prompt;
        self
    }

    pub fn with_trace_jsonl_path(mut self, path: Option<PathBuf>) -> Self {
        self.env.trace_sink = path.map(|p| Arc::new(JsonlTraceSink::new(p)) as Arc<dyn TraceSink>);
        self
    }

    pub fn with_trace_sink(mut self, sink: Option<Arc<dyn TraceSink>>) -> Self {
        self.env.trace_sink = sink;
        self
    }

    pub fn with_trace_level(mut self, level: TraceLevel) -> Self {
        self.env.trace_level = level;
        self
    }

    pub fn with_trace_context(mut self, context: TraceContext) -> Self {
        self.env.trace_context = context;
        self
    }

    pub fn with_termination(mut self, termination: TerminationPolicy) -> Self {
        self.env.termination = termination;
        self
    }

    pub fn build(self) -> RuntimeEnvironment {
        self.env
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_default_matches_legacy_core_config() {
        let env = RuntimeEnvironment::default();
        let cfg = env.to_runtime_core_config();
        assert!(Arc::ptr_eq(&cfg.attachment_store, &env.attachment_store));
    }
}
