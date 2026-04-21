//! Pluggable capability model for subagents.
//!
//! A `Capability` describes how to translate a `spawn_agent { capability: "name" }`
//! call into the model, execution mode, and tool denylist of the spawned
//! session. Built-in `low` / `medium` / `high` tiers are tiny `TierCapability`
//! instances; downstream code can register arbitrary additional impls
//! (different model lookup, dynamic inheritance, config-driven, …) without
//! touching the spawn pipeline.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use lash::{ExecutionMode, ProviderHandle, SessionPolicy};

/// State the registry exposes to a `Capability` while it resolves a spawn.
pub struct CapabilityContext<'a> {
    pub parent_policy: &'a SessionPolicy,
    pub parent_denied_tools: &'a HashSet<String>,
}

/// Resolved spawn configuration. Each field with `None` means "inherit from
/// the parent session"; concrete values override.
pub struct CapabilitySpec {
    pub model: Option<String>,
    pub model_variant: Option<String>,
    pub execution_mode: Option<ExecutionMode>,
    pub denied_tools: DenyList,
}

/// Tool denylist semantics. `Replace` discards whatever the parent denied;
/// `AddTo` extends the parent's list with extra entries.
pub enum DenyList {
    Replace(HashSet<String>),
    AddTo(HashSet<String>),
}

impl DenyList {
    pub fn apply(self, parent: &HashSet<String>) -> HashSet<String> {
        match self {
            DenyList::Replace(set) => set,
            DenyList::AddTo(extra) => parent.union(&extra).cloned().collect(),
        }
    }
}

pub trait Capability: Send + Sync {
    fn name(&self) -> &str;
    fn resolve(&self, ctx: &CapabilityContext<'_>) -> CapabilitySpec;
}

/// How a tier picks its execution mode relative to the parent session.
#[derive(Clone, Copy, Debug)]
pub enum TierExecutionMode {
    Inherit,
    Explicit(ExecutionMode),
}

/// Built-in capability that maps a tier name to: an optional explicit
/// model, a denylist, and an execution mode policy. Reproduces the historic
/// low/medium/high behaviour exactly when registered through
/// [`default_registry`].
pub struct TierCapability {
    name: String,
    model: Option<String>,
    denied_tools: HashSet<String>,
    execution_mode: TierExecutionMode,
}

impl TierCapability {
    pub fn new(
        name: impl Into<String>,
        model: Option<String>,
        denied_tools: impl IntoIterator<Item = String>,
        execution_mode: TierExecutionMode,
    ) -> Self {
        Self {
            name: name.into(),
            model,
            denied_tools: denied_tools.into_iter().collect(),
            execution_mode,
        }
    }
}

impl Capability for TierCapability {
    fn name(&self) -> &str {
        &self.name
    }

    fn resolve(&self, ctx: &CapabilityContext<'_>) -> CapabilitySpec {
        let (model, variant) = pick_tier_model(self, ctx.parent_policy);
        let execution_mode = match self.execution_mode {
            TierExecutionMode::Inherit => None,
            TierExecutionMode::Explicit(mode) => Some(mode),
        };
        CapabilitySpec {
            model: Some(model),
            model_variant: variant,
            execution_mode,
            denied_tools: DenyList::Replace(self.denied_tools.clone()),
        }
    }
}

fn pick_tier_model(tier: &TierCapability, policy: &SessionPolicy) -> (String, Option<String>) {
    if let Some(model) = &tier.model {
        let variant = preferred_variant(&policy.provider, model, tier.name());
        return (model.clone(), variant);
    }
    if let Some(selection) = policy.provider.default_agent_model(tier.name()) {
        return (selection.model, selection.variant);
    }
    let model = policy.model.clone();
    let variant = policy
        .provider
        .default_model_variant(&model)
        .map(str::to_string)
        .or_else(|| policy.model_variant.clone());
    (model, variant)
}

fn preferred_variant(provider: &ProviderHandle, model: &str, tier_name: &str) -> Option<String> {
    if provider.supported_variants(model).contains(&tier_name) {
        return Some(tier_name.to_string());
    }
    provider.default_model_variant(model).map(str::to_string)
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

/// Build the historic `low` / `medium` / `high` registry. `tier_models`
/// supplies optional explicit model overrides keyed by tier name; absent
/// tiers fall back to the provider's default agent model and then to the
/// parent session's model. `low_tier_execution_mode` pins the spawn mode
/// for the `low` tier (used to coerce light-weight subagents to Standard
/// mode while leaving Medium/High to inherit).
pub fn default_registry(
    tier_models: &BTreeMap<String, String>,
    low_tier_execution_mode: ExecutionMode,
) -> CapabilityRegistry {
    let model_for = |name: &str| tier_models.get(name).cloned();
    let mut registry = CapabilityRegistry::new();
    registry.add(Arc::new(TierCapability::new(
        "low",
        model_for("low"),
        [
            "apply_patch".to_string(),
            "ask".to_string(),
            "spawn_agent".to_string(),
        ],
        TierExecutionMode::Explicit(low_tier_execution_mode),
    )));
    registry.add(Arc::new(TierCapability::new(
        "medium",
        model_for("medium"),
        ["ask".to_string()],
        TierExecutionMode::Inherit,
    )));
    registry.add(Arc::new(TierCapability::new(
        "high",
        model_for("high"),
        ["ask".to_string()],
        TierExecutionMode::Inherit,
    )));
    registry
}
