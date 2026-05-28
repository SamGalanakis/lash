//! Pluggable capability model for subagents.
//!
//! A `Capability` describes how to translate an `agents.spawn({ capability: "name" })`
//! call into the model, plugin source, tool surface, and recursion authority
//! of the spawned session. Built-in `explore` / `peer` tiers are tiny
//! `TierCapability` instances; downstream code can register arbitrary
//! additional impls (different model lookup, dynamic inheritance,
//! config-driven, ...) without touching the spawn pipeline.

use std::collections::BTreeMap;
use std::sync::Arc;

use lash_core::{ModelSpec, SessionPluginSource, SessionPolicy, SessionSpec};

pub fn default_explore_plugin_source() -> TierPluginSource {
    TierPluginSource::CurrentHostFresh
}

/// State the registry exposes to a `Capability` while it resolves a spawn.
pub struct CapabilityContext<'a> {
    pub parent_policy: &'a SessionPolicy,
}

pub trait Capability: Send + Sync {
    fn name(&self) -> &str;
    fn resolve(&self, ctx: &CapabilityContext<'_>) -> CapabilityResolution;
}

#[derive(Clone, Debug)]
pub struct CapabilityResolution {
    pub spec: SessionSpec,
    pub plugin_source: SessionPluginSource,
}

impl CapabilityResolution {
    pub fn current_host_fresh(spec: SessionSpec) -> Self {
        Self {
            spec,
            plugin_source: SessionPluginSource::CurrentHostFresh,
        }
    }

    pub fn current_session_fork(spec: SessionSpec) -> Self {
        Self {
            spec,
            plugin_source: SessionPluginSource::CurrentSessionFork,
        }
    }
}

/// Fixed capability for callers that already know the exact child authority
/// they want and do not need provider-tier lookup.
pub struct StaticCapability {
    name: String,
    resolution: CapabilityResolution,
}

impl StaticCapability {
    pub fn new(name: impl Into<String>, spec: SessionSpec) -> Self {
        Self {
            name: name.into(),
            resolution: CapabilityResolution::current_host_fresh(spec),
        }
    }

    pub fn with_plugin_source(mut self, plugin_source: SessionPluginSource) -> Self {
        self.resolution.plugin_source = plugin_source;
        self
    }
}

impl Capability for StaticCapability {
    fn name(&self) -> &str {
        &self.name
    }

    fn resolve(&self, _ctx: &CapabilityContext<'_>) -> CapabilityResolution {
        self.resolution.clone()
    }
}

/// How a tier picks plugin instances relative to the parent session.
#[derive(Clone, Debug)]
pub enum TierPluginSource {
    CurrentHostFresh,
    CurrentSessionFork,
}

/// Built-in capability that maps a tier name to: an optional explicit
/// model, plugin-source policy, and the conventional `explore` / `peer`
/// authority split. Reproduces the historic tiered model behaviour when
/// registered through [`default_registry`].
pub struct TierCapability {
    name: String,
    model: Option<ModelSpec>,
    plugin_source: TierPluginSource,
}

impl TierCapability {
    pub fn new(
        name: impl Into<String>,
        model: Option<ModelSpec>,
        plugin_source: TierPluginSource,
    ) -> Self {
        Self {
            name: name.into(),
            model,
            plugin_source,
        }
    }
}

impl Capability for TierCapability {
    fn name(&self) -> &str {
        &self.name
    }

    fn resolve(&self, ctx: &CapabilityContext<'_>) -> CapabilityResolution {
        let model = pick_tier_model(self, ctx.parent_policy);
        let spec = SessionSpec::inherit().model(model);
        match self.plugin_source {
            TierPluginSource::CurrentHostFresh => CapabilityResolution::current_host_fresh(spec),
            TierPluginSource::CurrentSessionFork => {
                CapabilityResolution::current_session_fork(spec)
            }
        }
    }
}

fn pick_tier_model(tier: &TierCapability, policy: &SessionPolicy) -> ModelSpec {
    if let Some(model) = &tier.model {
        return model.clone();
    }
    policy.model.clone()
}

/// Registry of named capabilities. Order is preserved so that the JSON
/// schema enum and tool-list documentation list capabilities in the order
/// they were registered.
#[derive(Default)]
pub struct CapabilityRegistry {
    capabilities: Vec<Arc<dyn Capability>>,
}

impl CapabilityRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with(mut self, capability: Arc<dyn Capability>) -> Self {
        self.add(capability);
        self
    }

    pub fn add(&mut self, capability: Arc<dyn Capability>) {
        // Replace if a capability with the same name is already registered,
        // so a downstream caller can override a built-in tier without
        // duplicating it.
        let name = capability.name().to_string();
        if let Some(slot) = self
            .capabilities
            .iter_mut()
            .find(|existing| existing.name() == name)
        {
            *slot = capability;
        } else {
            self.capabilities.push(capability);
        }
    }

    pub fn get(&self, name: &str) -> Option<&Arc<dyn Capability>> {
        self.capabilities.iter().find(|c| c.name() == name)
    }

    pub fn names(&self) -> Vec<String> {
        self.capabilities
            .iter()
            .map(|c| c.name().to_string())
            .collect()
    }

    pub fn is_empty(&self) -> bool {
        self.capabilities.is_empty()
    }
}

/// Build the default `explore` / `peer` registry. `tier_models` supplies
/// optional explicit model overrides keyed by tier name; absent tiers fall
/// back to the provider's default agent model and then to the parent
/// session's model. The built-in `explore` tier uses
/// [`default_explore_plugin_source`] while `peer` forks the current session's
/// plugin instances.
///
/// The `explore` tier is read-only and cannot recurse: investigative
/// subagents that scan, summarise, or verify without mutating state. The
/// `peer` tier is a parallel-self with the parent's full affordances:
/// edits, recursion, anything the parent can do, in a fresh window.
/// Interactive-only tools (`ask`, `showcase`, `plan_exit`) are stripped from every subagent surface regardless of
/// capability — see `subagent_surface_contribution`.
pub fn default_registry(tier_models: &BTreeMap<String, ModelSpec>) -> CapabilityRegistry {
    let model_for = |name: &str| tier_models.get(name).cloned();
    let mut registry = CapabilityRegistry::new();
    registry.add(Arc::new(TierCapability::new(
        "explore",
        model_for("explore"),
        default_explore_plugin_source(),
    )));
    registry.add(Arc::new(TierCapability::new(
        "peer",
        model_for("peer"),
        TierPluginSource::CurrentSessionFork,
    )));
    registry
}
