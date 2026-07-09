//! Host-supplied model capability metadata.
//!
//! Capability is data the host attaches to a model spec and threads onto every
//! [`LlmRequest`](crate::llm::types::LlmRequest). Lash validates a requested
//! effort against it and normalizes (alias-clamps) the value before a provider
//! sees it. Providers consume capability; they never produce it.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Capability metadata for a single model on a route, supplied by the host.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ModelCapability {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningCapability>,
}

/// What reasoning/effort the model exposes and how effort maps onto the wire.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ReasoningCapability {
    #[serde(default)]
    pub efforts: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_effort: Option<String>,
    /// requested-effort -> canonical-effort clamp map (e.g. "xhigh" -> "max", "minimal" -> "low")
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub aliases: BTreeMap<String, String>,
    #[serde(default)]
    pub encoding: ReasoningEncoding,
    #[serde(default)]
    pub mandatory: bool,
}

/// How a resolved effort level is encoded on the wire.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEncoding {
    /// Named effort level sent as-is on the wire (Anthropic adaptive thinking, OpenAI reasoning.effort, Gemini thinkingLevel)
    #[default]
    Effort,
    /// Effort name resolves to a token budget (Anthropic budget thinking, Gemini 2.5 thinkingBudget); map is effort -> tokens
    Budget(BTreeMap<String, u32>),
}

/// Deterministic taxonomy of effort-validation failures. The serde snake_case
/// codes are a stable contract: downstream consumers match on them.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelEffortValidationCategory {
    UnsupportedEffort,
    EffortNotConfigurable,
    EffortRequired,
}

impl ModelEffortValidationCategory {
    /// Stable snake_case code, matching the serde representation. Turn-driver
    /// validation surfaces this as the turn-issue code.
    pub fn code(&self) -> &'static str {
        match self {
            Self::UnsupportedEffort => "unsupported_effort",
            Self::EffortNotConfigurable => "effort_not_configurable",
            Self::EffortRequired => "effort_required",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelEffortValidationError {
    pub category: ModelEffortValidationCategory,
    pub message: String,
}

impl std::fmt::Display for ModelEffortValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ModelEffortValidationError {}

impl ModelCapability {
    pub fn is_empty(&self) -> bool {
        self.reasoning.is_none()
    }

    /// Resolve a requested effort to its canonical form: alias-map first
    /// (input lowercased/trimmed), then direct membership in `efforts`.
    /// Returns `None` when the request maps to nothing this model exposes.
    pub fn resolve_effort(&self, requested: &str) -> Option<String> {
        let reasoning = self.reasoning.as_ref()?;
        let key = requested.trim().to_lowercase();
        if let Some(canonical) = reasoning.aliases.get(&key) {
            return Some(canonical.clone());
        }
        if reasoning.efforts.iter().any(|effort| effort == &key) {
            return Some(key);
        }
        None
    }

    /// Validate a requested effort against this capability and return the
    /// canonical (alias-normalized) effort to send on the wire. `Ok(None)`
    /// means no effort is configured on the request.
    pub fn validate_effort(
        &self,
        model: &str,
        provider_kind: &str,
        requested: Option<&str>,
    ) -> Result<Option<String>, ModelEffortValidationError> {
        match (self.reasoning.as_ref(), requested) {
            (None, Some(effort)) => Err(ModelEffortValidationError {
                category: ModelEffortValidationCategory::EffortNotConfigurable,
                message: format!(
                    "Model `{model}` on {provider_kind} does not expose configurable effort (requested `{effort}`)."
                ),
            }),
            (None, None) => Ok(None),
            (Some(reasoning), None) => {
                if reasoning.mandatory {
                    Err(ModelEffortValidationError {
                        category: ModelEffortValidationCategory::EffortRequired,
                        message: format!(
                            "Model `{model}` on {provider_kind} requires an explicit effort. Available: {}",
                            reasoning.efforts.join(", ")
                        ),
                    })
                } else {
                    Ok(None)
                }
            }
            (Some(reasoning), Some(effort)) => {
                if reasoning.efforts.is_empty() {
                    return Err(ModelEffortValidationError {
                        category: ModelEffortValidationCategory::EffortNotConfigurable,
                        message: format!(
                            "Model `{model}` on {provider_kind} does not expose configurable effort (requested `{effort}`)."
                        ),
                    });
                }
                match self.resolve_effort(effort) {
                    Some(resolved) if reasoning.efforts.contains(&resolved) => Ok(Some(resolved)),
                    _ => Err(ModelEffortValidationError {
                        category: ModelEffortValidationCategory::UnsupportedEffort,
                        message: format!(
                            "Unsupported effort `{effort}` for `{model}` on {provider_kind}. Available: {}",
                            reasoning.efforts.join(", ")
                        ),
                    }),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn efforts() -> Vec<String> {
        ["low", "medium", "high", "max"]
            .into_iter()
            .map(String::from)
            .collect()
    }

    fn reasoning() -> ReasoningCapability {
        ReasoningCapability {
            efforts: efforts(),
            default_effort: Some("medium".to_string()),
            aliases: BTreeMap::new(),
            encoding: ReasoningEncoding::Effort,
            mandatory: false,
        }
    }

    fn capability(reasoning: Option<ReasoningCapability>) -> ModelCapability {
        ModelCapability { reasoning }
    }

    #[test]
    fn is_empty_tracks_reasoning_presence() {
        assert!(capability(None).is_empty());
        assert!(!capability(Some(reasoning())).is_empty());
    }

    #[test]
    fn resolve_effort_prefers_alias_then_membership() {
        let mut r = reasoning();
        r.aliases.insert("xhigh".to_string(), "max".to_string());
        r.aliases.insert("minimal".to_string(), "low".to_string());
        let cap = capability(Some(r));

        // alias, including case/whitespace normalization
        assert_eq!(cap.resolve_effort("xhigh").as_deref(), Some("max"));
        assert_eq!(cap.resolve_effort("  XHigh ").as_deref(), Some("max"));
        assert_eq!(cap.resolve_effort("MINIMAL").as_deref(), Some("low"));
        // direct membership
        assert_eq!(cap.resolve_effort("high").as_deref(), Some("high"));
        // neither
        assert_eq!(cap.resolve_effort("turbo"), None);
    }

    #[test]
    fn resolve_effort_none_when_no_reasoning() {
        assert_eq!(capability(None).resolve_effort("low"), None);
    }

    #[test]
    fn validate_effort_none_reasoning_with_request_is_not_configurable() {
        let err = capability(None)
            .validate_effort("m", "test", Some("low"))
            .expect_err("no reasoning + request must fail");
        assert_eq!(
            err.category,
            ModelEffortValidationCategory::EffortNotConfigurable
        );
        assert!(err.message.contains("does not expose configurable effort"));
        assert!(err.message.contains("`low`"));
    }

    #[test]
    fn validate_effort_none_reasoning_no_request_is_ok_none() {
        assert_eq!(
            capability(None).validate_effort("m", "test", None),
            Ok(None)
        );
    }

    #[test]
    fn validate_effort_optional_reasoning_no_request_is_ok_none() {
        assert_eq!(
            capability(Some(reasoning())).validate_effort("m", "test", None),
            Ok(None)
        );
    }

    #[test]
    fn validate_effort_mandatory_reasoning_no_request_is_required() {
        let mut r = reasoning();
        r.mandatory = true;
        let err = capability(Some(r))
            .validate_effort("m", "test", None)
            .expect_err("mandatory + no request must fail");
        assert_eq!(err.category, ModelEffortValidationCategory::EffortRequired);
        assert!(err.message.contains("requires an explicit effort"));
        assert!(err.message.contains("low, medium, high, max"));
    }

    #[test]
    fn validate_effort_empty_efforts_is_not_configurable() {
        let mut r = reasoning();
        r.efforts.clear();
        let err = capability(Some(r))
            .validate_effort("m", "test", Some("low"))
            .expect_err("empty efforts + request must fail");
        assert_eq!(
            err.category,
            ModelEffortValidationCategory::EffortNotConfigurable
        );
    }

    #[test]
    fn validate_effort_alias_resolves_to_canonical() {
        let mut r = reasoning();
        r.aliases.insert("xhigh".to_string(), "max".to_string());
        let resolved = capability(Some(r))
            .validate_effort("m", "test", Some("xhigh"))
            .expect("alias must resolve");
        assert_eq!(resolved.as_deref(), Some("max"));
    }

    #[test]
    fn validate_effort_direct_membership_ok() {
        let resolved = capability(Some(reasoning()))
            .validate_effort("m", "test", Some("High"))
            .expect("membership must resolve");
        assert_eq!(resolved.as_deref(), Some("high"));
    }

    #[test]
    fn validate_effort_unknown_is_unsupported() {
        let err = capability(Some(reasoning()))
            .validate_effort("m", "test", Some("turbo"))
            .expect_err("unknown effort must fail");
        assert_eq!(
            err.category,
            ModelEffortValidationCategory::UnsupportedEffort
        );
        assert!(err.message.contains("Unsupported effort `turbo`"));
        assert!(err.message.contains("low, medium, high, max"));
    }

    #[test]
    fn validate_effort_alias_to_missing_effort_is_unsupported() {
        let mut r = reasoning();
        r.aliases
            .insert("ludicrous".to_string(), "ludicrous".to_string());
        let err = capability(Some(r))
            .validate_effort("m", "test", Some("ludicrous"))
            .expect_err("alias to non-existent effort must fail");
        assert_eq!(
            err.category,
            ModelEffortValidationCategory::UnsupportedEffort
        );
    }

    #[test]
    fn category_codes_are_stable_snake_case() {
        assert_eq!(
            ModelEffortValidationCategory::UnsupportedEffort.code(),
            "unsupported_effort"
        );
        assert_eq!(
            ModelEffortValidationCategory::EffortNotConfigurable.code(),
            "effort_not_configurable"
        );
        assert_eq!(
            ModelEffortValidationCategory::EffortRequired.code(),
            "effort_required"
        );
    }

    #[test]
    fn model_capability_serde_roundtrips_and_skips_empties() {
        let cap = capability(None);
        let json = serde_json::to_value(&cap).expect("serialize");
        assert_eq!(json, serde_json::json!({}));
        let back: ModelCapability = serde_json::from_value(json).expect("deserialize");
        assert_eq!(back, cap);

        let mut r = reasoning();
        r.aliases.insert("xhigh".to_string(), "max".to_string());
        r.encoding = ReasoningEncoding::Budget(BTreeMap::from([
            ("low".to_string(), 1024u32),
            ("high".to_string(), 8192u32),
        ]));
        let cap = capability(Some(r));
        let json = serde_json::to_value(&cap).expect("serialize");
        let back: ModelCapability = serde_json::from_value(json).expect("deserialize");
        assert_eq!(back, cap);
    }
}
