use std::num::NonZeroUsize;

use crate::provider::ModelCapability;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSpec {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub variant: Option<String>,
    pub limits: ModelLimits,
    /// Host-supplied capability metadata: what effort levels the model exposes
    /// and how they encode on the wire. Lash validates the requested variant
    /// against this and threads it onto every request the provider receives.
    #[serde(default, skip_serializing_if = "ModelCapability::is_empty")]
    pub capability: ModelCapability,
}

impl ModelSpec {
    pub fn new(id: impl Into<String>, context_window_tokens: NonZeroUsize) -> Self {
        Self {
            id: id.into(),
            variant: None,
            limits: ModelLimits {
                context_window_tokens,
                output_token_capacity: None,
            },
            capability: ModelCapability::default(),
        }
    }

    pub fn with_limits(
        id: impl Into<String>,
        variant: Option<String>,
        limits: ModelLimits,
    ) -> Self {
        Self {
            id: id.into(),
            variant,
            limits,
            capability: ModelCapability::default(),
        }
    }

    pub fn with_variant(mut self, variant: Option<String>) -> Self {
        self.variant = variant;
        self
    }

    pub fn with_capability(mut self, capability: ModelCapability) -> Self {
        self.capability = capability;
        self
    }

    /// Build a spec from the prompt budget (`context_window_tokens` — the
    /// maximum input the provider accepts for this model on this route) and the
    /// optional output cap. The budget is what history pruning and the UI
    /// measure against.
    pub fn from_token_limits(
        id: impl Into<String>,
        variant: Option<String>,
        context_window_tokens: usize,
        output_token_capacity: Option<usize>,
    ) -> Result<Self, String> {
        Ok(Self::with_limits(
            id,
            variant,
            ModelLimits::from_token_limits(context_window_tokens, output_token_capacity)?,
        ))
    }

    pub fn context_window_tokens(&self) -> usize {
        self.limits.context_window_tokens.get()
    }
}

impl Default for ModelSpec {
    fn default() -> Self {
        Self::new(
            String::new(),
            NonZeroUsize::new(1).expect("one is non-zero"),
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelLimits {
    /// The prompt budget: the maximum input tokens the provider accepts for
    /// this model on this route. History pruning and the UI measure against
    /// this — not the model's total context (input + output), which would
    /// over-budget by the whole response reservation.
    pub context_window_tokens: NonZeroUsize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_token_capacity: Option<NonZeroUsize>,
}

impl ModelLimits {
    pub fn from_token_limits(
        context_window_tokens: usize,
        output_token_capacity: Option<usize>,
    ) -> Result<Self, String> {
        Ok(Self {
            context_window_tokens: nonzero_token_limit(
                "context_window_tokens",
                context_window_tokens,
            )?,
            output_token_capacity: optional_nonzero_token_limit(
                "output_token_capacity",
                output_token_capacity,
            )?,
        })
    }
}

impl Default for ModelLimits {
    fn default() -> Self {
        Self {
            context_window_tokens: NonZeroUsize::new(1).expect("one is non-zero"),
            output_token_capacity: None,
        }
    }
}

fn nonzero_token_limit(name: &str, value: usize) -> Result<NonZeroUsize, String> {
    NonZeroUsize::new(value).ok_or_else(|| format!("{name} must be greater than zero"))
}

fn optional_nonzero_token_limit(
    name: &str,
    value: Option<usize>,
) -> Result<Option<NonZeroUsize>, String> {
    value
        .map(|value| nonzero_token_limit(name, value))
        .transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_spec_constructors_preserve_identity_variant_and_limits() {
        let limits = ModelLimits::from_token_limits(8_192, Some(1_024)).expect("valid limits");

        let spec =
            ModelSpec::with_limits("provider/model", Some("fast".to_string()), limits.clone());

        assert_eq!(spec.id, "provider/model");
        assert_eq!(spec.variant.as_deref(), Some("fast"));
        assert_eq!(spec.limits, limits);

        let changed = spec.clone().with_variant(Some("accurate".to_string()));
        assert_eq!(changed.id, "provider/model");
        assert_eq!(changed.variant.as_deref(), Some("accurate"));
        assert_eq!(changed.limits, spec.limits);

        let cleared = changed.with_variant(None);
        assert_eq!(cleared.id, "provider/model");
        assert_eq!(cleared.variant, None);
        assert_eq!(cleared.context_window_tokens(), 8_192);
    }

    #[test]
    fn model_token_limit_constructors_reject_zero_and_preserve_output_cap() {
        let spec = ModelSpec::from_token_limits(
            "provider/model",
            Some("variant-a".to_string()),
            200_000,
            Some(4_096),
        )
        .expect("valid token limits");

        assert_eq!(spec.id, "provider/model");
        assert_eq!(spec.variant.as_deref(), Some("variant-a"));
        assert_eq!(spec.context_window_tokens(), 200_000);
        assert_eq!(
            spec.limits.output_token_capacity.map(NonZeroUsize::get),
            Some(4_096)
        );

        let context_error = ModelSpec::from_token_limits("bad-context", None, 0, Some(1))
            .expect_err("zero context");
        assert!(
            context_error.contains("context_window_tokens"),
            "context error should name the invalid field: {context_error}"
        );

        let output_error = ModelLimits::from_token_limits(1, Some(0)).expect_err("zero output cap");
        assert!(
            output_error.contains("output_token_capacity"),
            "output error should name the invalid field: {output_error}"
        );
    }
}
