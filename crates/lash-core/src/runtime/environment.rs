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

use lash_trace::{TraceContext, TraceLevel, TraceSink};

use super::host::ProcessRegistry;
use super::{RuntimeCoreConfig, RuntimeEffectController, TerminationPolicy};

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
#[derive(Clone, Default)]
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

    // Host-owned process lifecycle and local execution support.
    pub process_registry: Option<Arc<dyn ProcessRegistry>>,

    // Store factory used by managed child sessions created from runtimes
    // built with this environment.
    pub session_store_factory: Option<Arc<dyn crate::SessionStoreFactory>>,

    pub core: RuntimeCoreConfig,
}

impl RuntimeEnvironment {
    pub fn builder() -> RuntimeEnvironmentBuilder {
        RuntimeEnvironmentBuilder::default()
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
        self.env.plugin_host = Some(Arc::new(host.as_ref().clone().with_processes()));
        self
    }

    pub fn with_residency(mut self, residency: Residency) -> Self {
        self.env.residency = residency;
        self
    }

    pub fn with_process_registry(mut self, process_registry: Arc<dyn ProcessRegistry>) -> Self {
        self.env.process_registry = Some(process_registry);
        if let Some(host) = self.env.plugin_host.take() {
            self.env.plugin_host = Some(Arc::new(host.as_ref().clone().with_processes()));
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

    pub fn with_runtime_core_config(mut self, core: RuntimeCoreConfig) -> Self {
        self.env.core = core;
        self
    }

    pub fn with_attachment_store(mut self, store: Arc<dyn crate::AttachmentStore>) -> Self {
        self.env.core = self.env.core.with_attachment_store(store);
        self
    }

    pub fn with_prompt_template(mut self, template: crate::PromptTemplate) -> Self {
        self.env.core = self.env.core.with_prompt_template(template);
        self
    }

    pub fn with_prompt_contribution(mut self, contribution: crate::PromptContribution) -> Self {
        self.env.core = self.env.core.with_prompt_contribution(contribution);
        self
    }

    pub fn with_replaced_prompt_slot(
        mut self,
        slot: crate::PromptSlot,
        contributions: impl IntoIterator<Item = crate::PromptContribution>,
    ) -> Self {
        self.env.core = self.env.core.with_replaced_prompt_slot(slot, contributions);
        self
    }

    pub fn with_cleared_prompt_slot(mut self, slot: crate::PromptSlot) -> Self {
        self.env.core = self.env.core.with_cleared_prompt_slot(slot);
        self
    }

    pub fn with_prompt_layer(mut self, prompt: crate::PromptLayer) -> Self {
        self.env.core = self.env.core.with_prompt_layer(prompt);
        self
    }

    pub fn with_trace_jsonl_path(mut self, path: Option<PathBuf>) -> Self {
        self.env.core = self.env.core.with_trace_jsonl_path(path);
        self
    }

    pub fn with_trace_sink(mut self, sink: Option<Arc<dyn TraceSink>>) -> Self {
        self.env.core = self.env.core.with_trace_sink(sink);
        self
    }

    pub fn with_trace_level(mut self, level: TraceLevel) -> Self {
        self.env.core = self.env.core.with_trace_level(level);
        self
    }

    pub fn with_trace_context(mut self, context: TraceContext) -> Self {
        self.env.core = self.env.core.with_trace_context(context);
        self
    }

    pub fn with_termination(mut self, termination: TerminationPolicy) -> Self {
        self.env.core = self.env.core.with_termination(termination);
        self
    }

    pub fn with_effect_controller(
        mut self,
        effect_controller: Arc<dyn RuntimeEffectController>,
    ) -> Self {
        self.env.core = self.env.core.with_effect_controller(effect_controller);
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
    fn builder_methods_configure_runtime_core() {
        let attachment_store: Arc<dyn crate::AttachmentStore> =
            Arc::new(crate::InMemoryAttachmentStore::new());
        let effect_controller: Arc<dyn RuntimeEffectController> =
            Arc::new(crate::runtime::InlineRuntimeEffectController::default());
        let trace_context = TraceContext::default().for_session("session-1");
        let termination = TerminationPolicy {
            treat_missing_done_as_failure: false,
        };

        let env = RuntimeEnvironment::builder()
            .with_attachment_store(Arc::clone(&attachment_store))
            .with_prompt_template(crate::default_prompt_template())
            .with_trace_jsonl_path(Some(
                std::env::temp_dir().join("lash-runtime-environment-builder-test.jsonl"),
            ))
            .with_trace_level(TraceLevel::Extended)
            .with_trace_context(trace_context.clone())
            .with_termination(termination.clone())
            .with_effect_controller(Arc::clone(&effect_controller))
            .build();

        assert!(Arc::ptr_eq(&env.core.attachment_store, &attachment_store));
        assert!(env.core.prompt.template.is_some());
        assert!(env.core.trace_sink.is_some());
        assert_eq!(env.core.trace_level, TraceLevel::Extended);
        assert_eq!(env.core.trace_context, trace_context);
        assert_eq!(
            env.core.termination.treat_missing_done_as_failure,
            termination.treat_missing_done_as_failure
        );
        assert!(Arc::ptr_eq(&env.core.effect_controller, &effect_controller));
    }

    #[test]
    fn runtime_core_config_replaces_core_config() {
        let core = RuntimeCoreConfig::default()
            .with_trace_level(TraceLevel::Extended)
            .with_termination(TerminationPolicy {
                treat_missing_done_as_failure: false,
            });

        let env = RuntimeEnvironment::builder()
            .with_trace_level(TraceLevel::Standard)
            .with_runtime_core_config(core)
            .build();

        assert_eq!(env.core.trace_level, TraceLevel::Extended);
        assert!(!env.core.termination.treat_missing_done_as_failure);
    }

    #[test]
    fn runtime_environment_does_not_mirror_runtime_core_config_fields() {
        let source = std::fs::read_to_string(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/runtime/environment.rs"),
        )
        .expect("read environment source");
        for field in [
            ["pub ", "attachment_store:"].concat(),
            ["pub ", "prompt:"].concat(),
            ["pub ", "trace_sink:"].concat(),
            ["pub ", "trace_level:"].concat(),
            ["pub ", "trace_context:"].concat(),
            ["pub ", "termination:"].concat(),
            ["pub ", "effect_controller:"].concat(),
            ["mirror ", "`RuntimeCoreConfig`"].concat(),
        ] {
            assert!(
                !source.contains(&field),
                "found mirrored field/comment: {field}"
            );
        }
    }
}
