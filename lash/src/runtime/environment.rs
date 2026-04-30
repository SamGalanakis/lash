//! Shared process-level infrastructure for lash embedders.
//!
//! `RuntimeEnvironment` is the type an embedder constructs ONCE at
//! startup and reuses across every `LashRuntime` instance it spawns.
//! Fields are all `Arc`-wrapped or cheap-to-clone so building a runtime
//! from an environment never rebuilds expensive state (plugin host,
//! path resolver, prompt template, …).
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
//!   provider crates accept an optional `Arc<reqwest::Client>` in
//!   their constructors, so the host can share one pool across every
//!   materialized provider.

use std::path::PathBuf;
use std::sync::Arc;

use lash_trace::{JsonlTraceSink, TraceContext, TraceSink};

use super::host::{DefaultPathResolver, RuntimeCoreConfig, SessionTaskExecutor};
use super::{PathResolver, SanitizerPolicy, TerminationPolicy};

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

    // All fields below mirror `RuntimeCoreConfig` and carry the same
    // semantics. They live on `RuntimeEnvironment` directly so
    // embedders don't have to build a separate core config.
    pub base_dir: PathBuf,
    pub path_resolver: Arc<dyn PathResolver>,
    pub prompt_template: crate::PromptTemplate,
    pub trace_sink: Option<Arc<dyn TraceSink>>,
    pub trace_stream_events: bool,
    pub trace_context: TraceContext,
    pub sanitizer: SanitizerPolicy,
    pub termination: TerminationPolicy,

    // Retry policy for LLM calls that return `retryable: true` errors.
    // Default matches legacy behaviour (3 retries at 2s / 5s / 10s).
    // Use `RetryPolicy::disabled()` in tests or when the host has its
    // own retry layer.
    pub retry_policy: lash_sansio::RetryPolicy,

    // Host-owned destination for refreshed OAuth credentials. lash-cli
    // points this at `paths::config_file()`; library embedders can
    // leave it `None` and persist via their own channel.
    pub credential_store_path: Option<PathBuf>,
}

impl Default for RuntimeEnvironment {
    fn default() -> Self {
        Self {
            plugin_host: None,
            residency: Residency::default(),
            session_task_executor: None,
            base_dir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            path_resolver: Arc::new(DefaultPathResolver),
            prompt_template: crate::default_prompt_template(),
            trace_sink: None,
            trace_stream_events: false,
            trace_context: TraceContext::default(),
            sanitizer: SanitizerPolicy::default(),
            termination: TerminationPolicy::default(),
            retry_policy: lash_sansio::RetryPolicy::default(),
            credential_store_path: None,
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
            base_dir: self.base_dir.clone(),
            path_resolver: Arc::clone(&self.path_resolver),
            prompt_template: self.prompt_template.clone(),
            trace_sink: self.trace_sink.clone(),
            trace_stream_events: self.trace_stream_events,
            trace_context: self.trace_context.clone(),
            sanitizer: self.sanitizer.clone(),
            termination: self.termination.clone(),
            retry_policy: self.retry_policy.clone(),
            credential_store_path: self.credential_store_path.clone(),
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
        self.env.plugin_host = Some(host);
        self
    }

    pub fn with_residency(mut self, residency: Residency) -> Self {
        self.env.residency = residency;
        self
    }

    pub fn with_session_task_executor(mut self, executor: Arc<dyn SessionTaskExecutor>) -> Self {
        self.env.session_task_executor = Some(executor);
        self
    }

    pub fn with_base_dir(mut self, base_dir: impl Into<PathBuf>) -> Self {
        self.env.base_dir = base_dir.into();
        self
    }

    pub fn with_path_resolver(mut self, resolver: Arc<dyn PathResolver>) -> Self {
        self.env.path_resolver = resolver;
        self
    }

    pub fn with_prompt_template(mut self, template: crate::PromptTemplate) -> Self {
        self.env.prompt_template = template;
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

    pub fn with_trace_stream_events(mut self, enabled: bool) -> Self {
        self.env.trace_stream_events = enabled;
        self
    }

    pub fn with_trace_context(mut self, context: TraceContext) -> Self {
        self.env.trace_context = context;
        self
    }

    pub fn with_sanitizer(mut self, sanitizer: SanitizerPolicy) -> Self {
        self.env.sanitizer = sanitizer;
        self
    }

    pub fn with_termination(mut self, termination: TerminationPolicy) -> Self {
        self.env.termination = termination;
        self
    }

    pub fn with_retry_policy(mut self, policy: lash_sansio::RetryPolicy) -> Self {
        self.env.retry_policy = policy;
        self
    }

    pub fn with_credential_store_path(mut self, path: Option<PathBuf>) -> Self {
        self.env.credential_store_path = path;
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
        // Cheap sanity: base dir comes through.
        assert_eq!(cfg.base_dir, env.base_dir);
    }
}
