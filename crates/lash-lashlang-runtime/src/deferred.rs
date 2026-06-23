//! RLM-only deferred tool resolution.
//!
//! A host-provided [`DeferredToolResolver`] resolves a Lashlang call-path that
//! is absent from the link-time [`LashlangHostEnvironment`] into a [`ToolGrant`]
//! (which carries its Tool Execution Binding) or reports `NotAvailable`. The
//! resolver resolves on demand only — it does not enumerate, advertise, or rank
//! tools.
//!
//! Linking runs a `gather → resolve → link` pass around the synchronous
//! [`lashlang::LinkedModule::link`]: collect the call-paths the program
//! references but the host environment does not provide, resolve the unknowns,
//! fold `Resolved` grants into the host environment, then link. Each resolution
//! is recorded so a re-driven turn reuses it without calling the resolver
//! again, and the flat Tool Catalog is never mutated — resolution is
//! link-scoped only.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;

use crate::{LashlangHostEnvironment, required_tool_lashlang_executable};

/// A host-authorized tool capability resolved for a deferred call-path. It
/// carries the callable contract and Lashlang identity (via the tool
/// definition) plus the host-owned Tool Execution Binding that routes a call to
/// the backing account, service, secret, or remote executor.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ToolGrant {
    /// The callable contract and Lashlang identity for the resolved tool.
    pub definition: lash_core::ToolDefinition,
    /// Optional registry source route authorized by the host. Registry-backed
    /// grants require this route at execution time; direct host providers may
    /// ignore it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_id: Option<String>,
    /// Host-owned routing authority that connects the grant to the backing
    /// account/service/secret/executor. Opaque to the runtime; the host
    /// interprets it when fulfilling the call and when rebuilding for replay.
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub execution_binding: serde_json::Value,
}

impl ToolGrant {
    pub fn new(definition: lash_core::ToolDefinition) -> Self {
        Self {
            definition,
            source_id: None,
            execution_binding: serde_json::Value::Null,
        }
    }

    pub fn with_source_id(mut self, source_id: impl Into<String>) -> Self {
        self.source_id = Some(source_id.into());
        self
    }

    pub fn with_execution_binding(mut self, execution_binding: serde_json::Value) -> Self {
        self.execution_binding = execution_binding;
        self
    }
}

/// Outcome of resolving one deferred call-path.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Resolution {
    /// The call-path resolved to a host-authorized tool.
    Resolved(ToolGrant),
    /// No tool is available for the call-path; linking leaves the symbol
    /// unresolved so the model sees a clean link error.
    NotAvailable,
}

/// RLM-only, host-provided resolution of a Lashlang call-path absent from the
/// link-time host environment. The resolver resolves on demand only.
#[async_trait]
pub trait DeferredToolResolver: Send + Sync {
    /// Resolve a fully-qualified Lashlang call-path (e.g. `web.fetch`) into a
    /// [`ToolGrant`] or report `NotAvailable`.
    async fn resolve(&self, path: &str) -> Resolution;
}

/// A handle to the host's deferred resolver, optional because most hosts ship
/// no deferral.
pub type SharedDeferredToolResolver = Arc<dyn DeferredToolResolver>;

/// A per-link record of every deferred resolution, keyed by call-path within
/// the execution scope. Replay/recovery applies the record so the resolver is
/// never called twice for the same link. Captures both `Resolved` grants (with
/// their Tool Execution Binding) and negative `NotAvailable` results.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct DeferredResolutionRecord {
    pub resolutions: BTreeMap<String, Resolution>,
}

impl DeferredResolutionRecord {
    pub fn get(&self, path: &str) -> Option<&Resolution> {
        self.resolutions.get(path)
    }

    pub fn record(&mut self, path: impl Into<String>, resolution: Resolution) {
        self.resolutions.insert(path.into(), resolution);
    }

    pub fn is_empty(&self) -> bool {
        self.resolutions.is_empty()
    }
}

/// Fold a resolved [`ToolGrant`] into the host environment so the subsequent
/// link can bind its call-path. The flat catalog is untouched.
fn fold_grant(
    host_environment: &mut LashlangHostEnvironment,
    grant: &ToolGrant,
) -> Result<(), String> {
    let binding = required_tool_lashlang_executable(&grant.definition.manifest)?;
    host_environment.resources.add_module_operation(
        binding.module_path.iter().map(String::as_str),
        binding.authority_type.clone(),
        binding.operation.clone(),
        grant.definition.manifest.id.to_string(),
        lashlang::TypeExpr::Any,
        lashlang::TypeExpr::Any,
    );
    Ok(())
}

/// Whether the host environment already binds `call_path` (dotted
/// `module.operation`), so it does not need deferral.
fn already_provided(host_environment: &LashlangHostEnvironment, call_path: &str) -> bool {
    let Some((module_path, operation)) = call_path.rsplit_once('.') else {
        return false;
    };
    host_environment
        .resources
        .provides_module_operation(module_path, operation)
}

/// `gather → resolve`: collect every module call-path `program` references,
/// resolve the ones `host_environment` does not already provide (without
/// failing fast on the first), record the outcomes, and fold `Resolved` grants
/// into `host_environment`. Returns the augmented host environment; the caller
/// links (or compiles via a cache) against it.
///
/// Every resolution is written to `record` so a re-driven or recovered turn
/// replays the recorded outcomes without calling the resolver again. The flat
/// Tool Catalog is never mutated — resolution is link-scoped only.
pub async fn resolve_and_fold_deferred(
    program: &lashlang::Program,
    mut host_environment: LashlangHostEnvironment,
    resolver: Option<&SharedDeferredToolResolver>,
    record: &mut DeferredResolutionRecord,
) -> LashlangHostEnvironment {
    let referenced = lashlang::referenced_module_call_paths(program);
    let unresolved = referenced
        .into_iter()
        .filter(|path| !already_provided(&host_environment, path))
        .collect::<Vec<_>>();

    for path in unresolved {
        // Replay: a recorded outcome wins and is never re-resolved.
        let resolution = if let Some(recorded) = record.get(&path) {
            recorded.clone()
        } else if let Some(resolver) = resolver {
            let resolution = resolver.resolve(&path).await;
            record.record(path.clone(), resolution.clone());
            resolution
        } else {
            // No resolver: leave the symbol unresolved for a clean link error.
            continue;
        };
        if let Resolution::Resolved(grant) = resolution {
            // A corrupt grant is treated as a clean unresolved link.
            let _ = fold_grant(&mut host_environment, &grant);
        }
    }

    host_environment
}

/// `gather → resolve → link`: [`resolve_and_fold_deferred`] then link. Used by
/// callers that do not maintain their own compile cache. `NotAvailable` (and no
/// resolver) leaves the symbol unresolved, surfacing a clean model-visible link
/// error.
pub async fn link_with_deferred_resolution(
    program: lashlang::Program,
    host_environment: LashlangHostEnvironment,
    resolver: Option<&SharedDeferredToolResolver>,
    record: &mut DeferredResolutionRecord,
) -> Result<lashlang::LinkedModule, lashlang::LinkError> {
    let host_environment =
        resolve_and_fold_deferred(&program, host_environment, resolver, record).await;
    lashlang::LinkedModule::link(program, host_environment)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LashlangSurface, LashlangToolBinding, ToolDefinitionLashlangExt};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn grant(name: &str, module: &str, operation: &str) -> ToolGrant {
        let definition = lash_core::ToolDefinition::raw(
            format!("tool:{name}"),
            name,
            format!("Tool {name}"),
            lash_core::ToolDefinition::default_input_schema(),
            serde_json::json!({ "type": "string" }),
        )
        .with_lashlang_binding(LashlangToolBinding::new([module], operation));
        ToolGrant::new(definition).with_execution_binding(serde_json::json!({ "account": name }))
    }

    struct CountingResolver {
        grant: ToolGrant,
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl DeferredToolResolver for CountingResolver {
        async fn resolve(&self, path: &str) -> Resolution {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if path == "web.fetch" {
                Resolution::Resolved(self.grant.clone())
            } else {
                Resolution::NotAvailable
            }
        }
    }

    fn empty_host_environment() -> LashlangHostEnvironment {
        let catalog = lash_core::ToolCatalog::default();
        LashlangSurface::default()
            .host_environment(&catalog)
            .expect("empty host environment")
    }

    #[tokio::test]
    async fn resolves_deferred_call_path_and_records_grant() {
        let calls = Arc::new(AtomicUsize::new(0));
        let resolver: SharedDeferredToolResolver = Arc::new(CountingResolver {
            grant: grant("fetch_url", "web", "fetch"),
            calls: Arc::clone(&calls),
        });
        let program = lashlang::parse(r#"await web.fetch({ url: "x" })?"#).expect("parse");
        let mut record = DeferredResolutionRecord::default();

        link_with_deferred_resolution(
            program,
            empty_host_environment(),
            Some(&resolver),
            &mut record,
        )
        .await
        .expect("deferred resolution links");

        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(matches!(
            record.get("web.fetch"),
            Some(Resolution::Resolved(_))
        ));
    }

    #[tokio::test]
    async fn replay_reuses_record_without_calling_resolver() {
        let calls = Arc::new(AtomicUsize::new(0));
        let resolver: SharedDeferredToolResolver = Arc::new(CountingResolver {
            grant: grant("fetch_url", "web", "fetch"),
            calls: Arc::clone(&calls),
        });
        let program = lashlang::parse(r#"await web.fetch({ url: "x" })?"#).expect("parse");

        let mut record = DeferredResolutionRecord::default();
        link_with_deferred_resolution(
            program.clone(),
            empty_host_environment(),
            Some(&resolver),
            &mut record,
        )
        .await
        .expect("first link");
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        // Re-drive the same link with the recorded resolutions: the resolver is
        // never called again.
        link_with_deferred_resolution(
            program,
            empty_host_environment(),
            Some(&resolver),
            &mut record,
        )
        .await
        .expect("replayed link");
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "replay must not re-resolve"
        );
    }

    #[tokio::test]
    async fn not_available_surfaces_clean_link_error_and_is_recorded() {
        let calls = Arc::new(AtomicUsize::new(0));
        let resolver: SharedDeferredToolResolver = Arc::new(CountingResolver {
            grant: grant("fetch_url", "web", "fetch"),
            calls: Arc::clone(&calls),
        });
        let program = lashlang::parse(r#"await mystery.run({})?"#).expect("parse");
        let mut record = DeferredResolutionRecord::default();

        let err = link_with_deferred_resolution(
            program.clone(),
            empty_host_environment(),
            Some(&resolver),
            &mut record,
        )
        .await
        .expect_err("unavailable call-path must surface a link error");
        assert!(format!("{err:?}").len() > 0);
        assert!(matches!(
            record.get("mystery.run"),
            Some(Resolution::NotAvailable)
        ));

        // Replay reuses the recorded NotAvailable without re-resolving.
        let calls_before = calls.load(Ordering::SeqCst);
        link_with_deferred_resolution(
            program,
            empty_host_environment(),
            Some(&resolver),
            &mut record,
        )
        .await
        .expect_err("replayed unavailable call-path still errors");
        assert_eq!(calls.load(Ordering::SeqCst), calls_before);
    }
}
