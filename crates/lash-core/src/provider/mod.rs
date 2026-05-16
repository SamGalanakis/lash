//! Provider components and registry for pluggable LLM backends.
//!
//! A provider is split into narrow capabilities: configured state,
//! request transport, failure classification, and model policy.
//! [`ProviderHandle`] owns those components and is the executable handle
//! stored by session policy and host config.
//!
//! Serialization: [`ProviderHandle`] is the owning handle that
//! [`SessionPolicy`] stores. It round-trips through [`ProviderSpec`] —
//! a `{ "type": kind, …config }` JSON object whose shape matches the
//! legacy `#[serde(tag = "type")]` enum exactly, so existing
//! `~/.lash/config.json` files load without migration.

mod handle;
mod model_policy;
mod options;
mod rate_limit;
mod registry;
mod spec;
mod support;
#[cfg(test)]
mod tests;
mod traits;

pub use handle::{ProviderComponents, ProviderHandle, UnconfiguredProvider};
pub use model_policy::StaticModelPolicy;
pub use options::{
    AgentModelSelection, CacheRetention, DEFAULT_CHUNK_TIMEOUT_MS, DEFAULT_REQUEST_TIMEOUT_MS,
    LlmTimeouts, ProviderOptions, ProviderRateLimitPolicy, ProviderReliability,
    ProviderReliabilityBuilder, ProviderRetryPolicy, ProviderThinkingPolicy, ProviderTimeoutPolicy,
    RequestTimeout, VariantRequestConfig,
};
pub use rate_limit::{ProviderRateLimitPermit, ProviderRateLimiter};
pub use registry::{
    ProviderFactory, ProviderRegistry, build_provider, provider_factory, register_provider_factory,
};
pub use spec::ProviderSpec;
pub use traits::{
    DefaultProviderFailureClassifier, ProviderFailureClassifier, ProviderModelPolicy,
    ProviderState, ProviderTransport, is_context_overflow_text,
};
