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
//! * **CLI interactive (single runtime, default):**
//!   `RuntimeEnvironment::builder().build()`. The builder seeds an explicit
//!   in-memory core (`RuntimeHostConfig::in_memory`); override it with
//!   `with_runtime_host_config` for durable stores.
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

use std::sync::Arc;

use lash_trace::{TraceContext, TraceLevel, TraceSink};

#[cfg(test)]
use super::InlineEffectHost;
use super::process::ProcessRegistry;
use super::{EffectHost, RuntimeHostConfig, TerminationPolicy};

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
/// Default values build an embedded runtime without process lifecycle
/// support. Hosts that want long-running tools, async handles, subagents,
/// or process controls must provide a process registry explicitly.
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

    // Host-owned process lifecycle and local execution support.
    pub process_registry: Option<Arc<dyn ProcessRegistry>>,

    // Host-owned trigger subscription and host-event occurrence routing.
    pub host_event_store: Option<Arc<dyn crate::HostEventStore>>,

    // Store factory used by managed child sessions created from runtimes
    // built with this environment.
    pub session_store_factory: Option<Arc<dyn crate::SessionStoreFactory>>,

    // Wakes the host's `ProcessWorkRunner` so a successful process start is
    // consumed promptly. Threaded onto every `RuntimeHost` built from this
    // environment (see `LashRuntime::from_environment`); `None` when no work
    // runner is wired, in which case poking is a no-op.
    pub process_work_poke: Option<super::ProcessWorkPoke>,

    // Wakes the host's `QueuedWorkRunner` so queued turn work, including
    // process wakes, drains promptly through the host-selected queue runner.
    pub queued_work_poke: Option<super::QueuedWorkPoke>,

    pub core: RuntimeHostConfig,
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
pub struct RuntimeEnvironmentBuilder {
    env: RuntimeEnvironment,
}

impl Default for RuntimeEnvironmentBuilder {
    fn default() -> Self {
        // `RuntimeHostConfig` has no `Default`; the builder starts from an
        // explicitly named in-memory core so the choice is visible in source.
        // The `lash` facade always overrides this via `with_runtime_host_config`
        // and rejects builds that never named their stores.
        Self {
            env: RuntimeEnvironment {
                plugin_host: None,
                residency: Residency::default(),
                process_registry: None,
                host_event_store: Some(Arc::new(crate::InMemoryHostEventStore::default())),
                session_store_factory: None,
                process_work_poke: None,
                queued_work_poke: None,
                core: RuntimeHostConfig::in_memory(),
            },
        }
    }
}

impl RuntimeEnvironmentBuilder {
    pub fn with_plugin_host(mut self, host: Arc<crate::PluginHost>) -> Self {
        self.env.plugin_host = Some(host);
        self
    }

    pub fn with_residency(mut self, residency: Residency) -> Self {
        self.env.residency = residency;
        self
    }

    pub fn with_process_registry(mut self, process_registry: Arc<dyn ProcessRegistry>) -> Self {
        self.env.process_registry = Some(process_registry);
        if let Some(host) = self.env.plugin_host.take() {
            let abilities = super::builder::lashlang_abilities_for_process_registry(
                host.lashlang_abilities(),
                true,
            );
            self.env.plugin_host = Some(Arc::new(
                host.as_ref().clone().with_lashlang_abilities(abilities),
            ));
        }
        self
    }

    pub fn with_host_event_store(mut self, store: Arc<dyn crate::HostEventStore>) -> Self {
        self.env.host_event_store = Some(store);
        self
    }

    pub fn with_session_store_factory(
        mut self,
        factory: Arc<dyn crate::SessionStoreFactory>,
    ) -> Self {
        self.env.session_store_factory = Some(factory);
        self
    }

    /// Set the poke handle that wakes the host's `ProcessWorkRunner`. Every
    /// `RuntimeHost` built from this environment carries the poke, so the
    /// process control seam can make consumption prompt after a start.
    pub fn with_process_work_poke(mut self, poke: super::ProcessWorkPoke) -> Self {
        self.env.process_work_poke = Some(poke);
        self
    }

    pub fn with_queued_work_poke(mut self, poke: super::QueuedWorkPoke) -> Self {
        self.env.queued_work_poke = Some(poke);
        self
    }

    pub fn with_runtime_host_config(mut self, core: RuntimeHostConfig) -> Self {
        self.env.core = core;
        self
    }

    pub fn with_attachment_store(mut self, store: Arc<dyn crate::AttachmentStore>) -> Self {
        self.env.core.durability.attachment_store = store;
        self
    }

    pub fn with_prompt_template(mut self, template: crate::PromptTemplate) -> Self {
        self.env.core.prompt.prompt.template = Some(template);
        self
    }

    pub fn with_prompt_contribution(mut self, contribution: crate::PromptContribution) -> Self {
        self.env.core.prompt.prompt.add_contribution(contribution);
        self
    }

    pub fn with_replaced_prompt_slot(
        mut self,
        slot: crate::PromptSlot,
        contributions: impl IntoIterator<Item = crate::PromptContribution>,
    ) -> Self {
        self.env
            .core
            .prompt
            .prompt
            .replace_slot(slot, contributions);
        self
    }

    pub fn with_cleared_prompt_slot(mut self, slot: crate::PromptSlot) -> Self {
        self.env.core.prompt.prompt.clear_slot(slot);
        self
    }

    pub fn with_prompt_layer(mut self, prompt: crate::PromptLayer) -> Self {
        self.env.core.prompt.prompt = prompt;
        self
    }

    pub fn with_trace_sink(mut self, sink: Option<Arc<dyn TraceSink>>) -> Self {
        self.env.core.tracing.trace_sink = sink;
        self
    }

    pub fn with_lashlang_execution_sink(mut self, sink: Option<Arc<dyn TraceSink>>) -> Self {
        self.env.core.tracing.lashlang_execution_sink = sink;
        self
    }

    pub fn with_lashlang_execution_jsonl_path(mut self, path: Option<std::path::PathBuf>) -> Self {
        self.env.core.tracing.lashlang_execution_sink =
            path.map(|path| Arc::new(lash_trace::JsonlTraceSink::new(path)) as Arc<dyn TraceSink>);
        self
    }

    pub fn with_trace_level(mut self, level: TraceLevel) -> Self {
        self.env.core.tracing.trace_level = level;
        self
    }

    pub fn with_trace_context(mut self, context: TraceContext) -> Self {
        self.env.core.tracing.trace_context = context;
        self
    }

    pub fn with_termination(mut self, termination: TerminationPolicy) -> Self {
        self.env.core.control.termination = termination;
        self
    }

    pub fn with_effect_host(mut self, effect_host: Arc<dyn EffectHost>) -> Self {
        self.env.core.control.effect_host = effect_host;
        self
    }

    pub fn with_provider_resolver(
        mut self,
        provider_resolver: Arc<dyn crate::RuntimeProviderResolver>,
    ) -> Self {
        self.env.core.providers.provider_resolver = provider_resolver;
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
    fn builder_methods_configure_runtime_host() {
        let attachment_store: Arc<dyn crate::AttachmentStore> =
            Arc::new(crate::InMemoryAttachmentStore::new());
        let effect_host: Arc<dyn EffectHost> = Arc::new(InlineEffectHost::default());
        let trace_context = TraceContext::default().for_session("session-1");
        let termination = TerminationPolicy {
            treat_missing_done_as_failure: false,
        };

        let env = RuntimeEnvironment::builder()
            .with_attachment_store(Arc::clone(&attachment_store))
            .with_prompt_template(crate::default_prompt_template())
            .with_trace_sink(Some(Arc::new(lash_trace::JsonlTraceSink::new(
                std::env::temp_dir().join("lash-runtime-environment-builder-test.jsonl"),
            ))))
            .with_trace_level(TraceLevel::Extended)
            .with_trace_context(trace_context.clone())
            .with_termination(termination.clone())
            .with_effect_host(Arc::clone(&effect_host))
            .build();

        assert!(Arc::ptr_eq(
            &env.core.durability.attachment_store,
            &attachment_store
        ));
        assert!(env.core.prompt.prompt.template.is_some());
        assert!(env.core.tracing.trace_sink.is_some());
        assert_eq!(env.core.tracing.trace_level, TraceLevel::Extended);
        assert_eq!(env.core.tracing.trace_context, trace_context);
        assert_eq!(
            env.core.control.termination.treat_missing_done_as_failure,
            termination.treat_missing_done_as_failure
        );
        assert!(Arc::ptr_eq(&env.core.control.effect_host, &effect_host));
    }

    #[test]
    fn runtime_host_config_replaces_core_config() {
        let mut core = RuntimeHostConfig::in_memory();
        core.tracing.trace_level = TraceLevel::Extended;
        core.control.termination = TerminationPolicy {
            treat_missing_done_as_failure: false,
        };

        let env = RuntimeEnvironment::builder()
            .with_trace_level(TraceLevel::Standard)
            .with_runtime_host_config(core)
            .build();

        assert_eq!(env.core.tracing.trace_level, TraceLevel::Extended);
        assert!(!env.core.control.termination.treat_missing_done_as_failure);
    }

    #[test]
    fn runtime_environment_does_not_mirror_runtime_host_config_fields() {
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
            ["pub ", "effect_host:"].concat(),
            ["mirror ", "`RuntimeHostConfig`"].concat(),
        ] {
            assert!(
                !source.contains(&field),
                "found mirrored field/comment: {field}"
            );
        }
    }
}
