//! Pluggable capability model for subagents.
//!
//! A `Capability` describes how to translate a `spawn_agent { capability: "name" }`
//! call into the model, execution mode, tool surface, and recursion authority
//! of the spawned session. Built-in `explore` / `peer` tiers are tiny
//! `TierCapability` instances; downstream code can register arbitrary
//! additional impls (different model lookup, dynamic inheritance,
//! config-driven, ...) without touching the spawn pipeline.

use std::collections::BTreeMap;
use std::sync::Arc;

use lash::{ExecutionMode, ProviderHandle, SessionPolicy, ToolDefinition};

/// State the registry exposes to a `Capability` while it resolves a spawn.
pub struct CapabilityContext<'a> {
    pub parent_policy: &'a SessionPolicy,
}

/// Required policy field resolved by a capability.
#[derive(Clone, Debug)]
pub enum CapabilityField<T> {
    Inherit,
    Set(T),
}

/// Optional policy field resolved by a capability.
#[derive(Clone, Debug)]
pub enum CapabilityOptionalField<T> {
    Inherit,
    Set(T),
    Clear,
}

/// Tool definitions a child session should receive.
#[derive(Clone, Debug)]
pub enum CapabilityToolSurface {
    InheritParent,
    BuiltinExplore,
    Explicit(Vec<ToolDefinition>),
}

/// Whether a spawned session may expose recursive subagent tools.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CapabilityRecursion {
    Inherit,
    Disabled,
}

/// Resolved spawn configuration.
#[derive(Clone, Debug)]
pub struct CapabilitySpec {
    pub model: CapabilityField<String>,
    pub model_variant: CapabilityOptionalField<String>,
    pub execution_mode: CapabilityField<ExecutionMode>,
    pub tool_surface: CapabilityToolSurface,
    pub recursion: CapabilityRecursion,
}

impl CapabilitySpec {
    pub fn inherit() -> Self {
        Self {
            model: CapabilityField::Inherit,
            model_variant: CapabilityOptionalField::Inherit,
            execution_mode: CapabilityField::Inherit,
            tool_surface: CapabilityToolSurface::InheritParent,
            recursion: CapabilityRecursion::Inherit,
        }
    }
}

pub trait Capability: Send + Sync {
    fn name(&self) -> &str;
    fn resolve(&self, ctx: &CapabilityContext<'_>) -> CapabilitySpec;
}

/// Fixed capability for callers that already know the exact child authority
/// they want and do not need provider-tier lookup.
pub struct StaticCapability {
    name: String,
    spec: CapabilitySpec,
}

impl StaticCapability {
    pub fn new(name: impl Into<String>, spec: CapabilitySpec) -> Self {
        Self {
            name: name.into(),
            spec,
        }
    }
}

impl Capability for StaticCapability {
    fn name(&self) -> &str {
        &self.name
    }

    fn resolve(&self, _ctx: &CapabilityContext<'_>) -> CapabilitySpec {
        self.spec.clone()
    }
}

/// How a tier picks its execution mode relative to the parent session.
#[derive(Clone, Debug)]
pub enum TierExecutionMode {
    Inherit,
    Explicit(ExecutionMode),
}

/// Built-in capability that maps a tier name to: an optional explicit
/// model, execution mode policy, and the conventional `explore` / `peer`
/// authority split. Reproduces the historic tiered model behaviour when
/// registered through [`default_registry`].
pub struct TierCapability {
    name: String,
    model: Option<String>,
    execution_mode: TierExecutionMode,
}

impl TierCapability {
    pub fn new(
        name: impl Into<String>,
        model: Option<String>,
        execution_mode: TierExecutionMode,
    ) -> Self {
        Self {
            name: name.into(),
            model,
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
        let execution_mode = match &self.execution_mode {
            TierExecutionMode::Inherit => CapabilityField::Inherit,
            TierExecutionMode::Explicit(mode) => CapabilityField::Set(mode.clone()),
        };
        let model_variant = match variant {
            Some(variant) => CapabilityOptionalField::Set(variant),
            None => CapabilityOptionalField::Inherit,
        };
        let is_explore = self.name == "explore";
        CapabilitySpec {
            model: CapabilityField::Set(model),
            model_variant,
            execution_mode,
            tool_surface: if is_explore {
                CapabilityToolSurface::BuiltinExplore
            } else {
                CapabilityToolSurface::InheritParent
            },
            recursion: if is_explore {
                CapabilityRecursion::Disabled
            } else {
                CapabilityRecursion::Inherit
            },
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

/// Build the default `explore` / `peer` registry. `tier_models` supplies
/// optional explicit model overrides keyed by tier name; absent tiers fall
/// back to the provider's default agent model and then to the parent
/// session's model. `explore_tier_execution_mode` pins the spawn mode for
/// the `explore` tier (used to coerce read-only subagents to Standard mode
/// while leaving `peer` to inherit).
///
/// The `explore` tier is read-only and cannot recurse: investigative
/// subagents that scan, summarise, or verify without mutating state. The
/// `peer` tier is a parallel-self with the parent's full affordances:
/// edits, recursion, anything the parent can do, in a fresh window.
/// Interactive-only tools (`ask`, `showcase`, `plan_exit`) are stripped from every subagent surface regardless of
/// capability — see `subagent_surface_contribution`.
pub fn default_registry(
    tier_models: &BTreeMap<String, String>,
    explore_tier_execution_mode: ExecutionMode,
) -> CapabilityRegistry {
    let model_for = |name: &str| tier_models.get(name).cloned();
    let mut registry = CapabilityRegistry::new();
    registry.add(Arc::new(TierCapability::new(
        "explore",
        model_for("explore"),
        TierExecutionMode::Explicit(explore_tier_execution_mode),
    )));
    registry.add(Arc::new(TierCapability::new(
        "peer",
        model_for("peer"),
        TierExecutionMode::Inherit,
    )));
    registry
}
