//! Model selection: parsing, validation, and resolution of `<model>` /
//! variant inputs against the active provider and the models.dev catalog.

use std::sync::Arc;

use lash::provider::ProviderHandle;

use crate::model_catalog::{
    CachedModelCatalog, FileModelCatalogStore, ModelInfo, ModelsDevHttpSource, ResolvedModelSpec,
};

#[derive(Debug, Clone)]
pub(crate) struct ModelSelection {
    pub(crate) model: String,
}

pub(crate) fn models_dev_catalog() -> Result<Arc<CachedModelCatalog>, String> {
    CachedModelCatalog::models_dev(
        Arc::new(FileModelCatalogStore::new(
            crate::paths::model_catalog_cache_file(),
        )),
        Some(Arc::new(ModelsDevHttpSource::default_models_dev())),
    )
    .map(Arc::new)
}

pub(crate) fn parse_model_selection(input: &str) -> Result<ModelSelection, String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err("Model cannot be empty.".to_string());
    }

    let parts: Vec<&str> = trimmed.split_whitespace().collect();
    match parts.as_slice() {
        [model] => Ok(ModelSelection {
            model: (*model).to_string(),
        }),
        _ => Err("Model input must be a single `<model>` token.".to_string()),
    }
}

fn parse_variant_input(input: &str) -> Result<String, String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err("Variant cannot be empty.".to_string());
    }
    if trimmed.contains(char::is_whitespace) {
        return Err("Variant must be a single token.".to_string());
    }
    Ok(trimmed.to_ascii_lowercase())
}

pub(crate) fn validate_model_selection(
    provider: &ProviderHandle,
    selection: &ModelSelection,
) -> Result<(), String> {
    provider.validate_model_name(&selection.model)
}

/// Curated model limits that take precedence over models.dev — only for
/// `(provider, model)` pairs the catalog gets wrong or omits. Keep this minimal
/// and dated; everything else flows from models.dev so it stays current. The
/// per-provider input ceiling (below) handles route-wide caps; use this table
/// for one-off model corrections.
fn builtin_model_info(_provider_kind: &str, _model: &str) -> Option<ModelInfo> {
    None
}

/// The maximum input a provider's *route* accepts when it caps below the
/// model's nominal catalog limit. The Codex OAuth route serves OpenAI models
/// through a much smaller window than the API, and the catalog has no
/// codex-specific row for every model (e.g. `gpt-5.5` resolves to the generic
/// `openai/gpt-5.5` at ~922k input). gpt-5-codex catalogs at 256k input and the
/// OAuth route rejected ~270k, so clamp every codex request to 256k.
fn provider_input_ceiling(provider_kind: &str) -> Option<u64> {
    match provider_kind {
        "codex" => Some(256_000),
        _ => None,
    }
}

pub(crate) fn resolve_model_selection(
    provider: &ProviderHandle,
    selection: &ModelSelection,
    catalog: &CachedModelCatalog,
) -> Result<ResolvedModelSpec, String> {
    provider.validate_model_name(&selection.model)?;
    let configured_model = selection.model.trim();
    let catalog_model_id =
        crate::provider_metadata::provider_catalog_model_id(provider.kind(), configured_model);
    // Built-in overrides win over models.dev; fall back to the catalog.
    let Some(mut info) = builtin_model_info(provider.kind(), configured_model)
        .or_else(|| catalog.get(&catalog_model_id))
    else {
        let normalized =
            crate::provider_metadata::provider_wire_model_id(provider.kind(), configured_model);
        let mut message = format!(
            "model `{}` has no context-window entry in the models.dev catalog for {}. Choose a cataloged model.",
            configured_model,
            provider.kind(),
        );
        if normalized != configured_model {
            message.push_str(&format!("\nResolved provider model ID: `{normalized}`"));
        }
        return Err(message);
    };
    // Clamp the prompt budget to the route's real input ceiling (e.g. codex).
    if let Some(ceiling) = provider_input_ceiling(provider.kind()) {
        info.max_input_tokens = Some(info.prompt_budget_tokens().min(ceiling));
    }
    Ok(ResolvedModelSpec {
        configured_model: configured_model.to_string(),
        provider_model: crate::provider_metadata::provider_wire_model_id(
            provider.kind(),
            configured_model,
        ),
        catalog_model_id,
        info,
    })
}

/// The catalog's recommended default variant for `model` on this provider, or
/// `None` when the model exposes no configurable effort.
pub(crate) fn default_variant(provider: &ProviderHandle, model: &str) -> Option<String> {
    crate::capability_catalog::capability_for(provider.kind(), model)
        .reasoning
        .and_then(|reasoning| reasoning.default_effort)
}

/// The effort levels `model` exposes on this provider (empty when none).
pub(crate) fn supported_efforts(provider: &ProviderHandle, model: &str) -> Vec<String> {
    crate::capability_catalog::capability_for(provider.kind(), model)
        .reasoning
        .map(|reasoning| reasoning.efforts)
        .unwrap_or_default()
}

pub(crate) fn resolve_model_variant(
    provider: &ProviderHandle,
    model: &str,
    requested: Option<&str>,
) -> Result<Option<String>, String> {
    let capability = crate::capability_catalog::capability_for(provider.kind(), model);
    let Some(raw) = requested else {
        return Ok(capability
            .reasoning
            .and_then(|reasoning| reasoning.default_effort));
    };
    let variant = parse_variant_input(raw)?;
    if variant == "default" {
        return Ok(capability
            .reasoning
            .and_then(|reasoning| reasoning.default_effort));
    }
    // Validate through the same seam the runtime uses; the returned effort is
    // alias-normalized (e.g. `minimal` -> `low`).
    capability
        .validate_effort(model, provider.kind(), Some(&variant))
        .map_err(|err| err.to_string())
}

pub(crate) fn variant_lines(
    provider: &ProviderHandle,
    model: &str,
    current_variant: Option<&str>,
) -> Vec<String> {
    let supported = supported_efforts(provider, model);
    let mut lines = Vec::new();
    if supported.is_empty() {
        lines.push(format!(
            "`{}` on {} does not expose configurable variants.",
            model,
            provider_display_label(provider)
        ));
        return lines;
    }
    lines.push(format!(
        "Current variant: `{}`",
        current_variant.unwrap_or("(none)")
    ));
    if let Some(default_variant) = default_variant(provider, model) {
        lines.push(format!("Recommended default: `{}`", default_variant));
    }
    lines.push(format!("Available variants: {}", supported.join(", ")));
    lines.push("Usage: `/variant <name>` or `/variant default`".to_string());
    lines
}

pub(crate) fn provider_display_label(provider: &ProviderHandle) -> &'static str {
    crate::provider_metadata::provider_cli_label(provider.kind())
}

pub(crate) fn expose_provider_thinking(provider: &mut ProviderHandle) {
    let mut options = provider.options();
    options.expose_thinking = true;
    provider.set_options(options);
}

#[cfg(test)]
mod tests {
    use super::*;
    use lash_core::testing::TestProvider;

    fn provider(kind: &'static str) -> ProviderHandle {
        TestProvider::builder().kind(kind).build().into_handle()
    }

    fn resolved(configured: &str, wire: &str) -> ResolvedModelSpec {
        ResolvedModelSpec {
            configured_model: configured.to_string(),
            provider_model: wire.to_string(),
            catalog_model_id: format!("catalog/{configured}"),
            info: ModelInfo {
                context_window: 200_000,
                max_input_tokens: Some(200_000),
                max_output_tokens: None,
            },
        }
    }

    #[test]
    fn selecting_a_model_attaches_the_matching_capability_to_the_spec() {
        let spec = resolved("claude-opus-4-7", "claude-opus-4-7")
            .into_model_spec("anthropic", Some("xhigh".to_string()))
            .expect("spec");
        assert_eq!(spec.variant.as_deref(), Some("xhigh"));
        assert_eq!(
            spec.capability,
            crate::capability_catalog::capability_for("anthropic", "claude-opus-4-7")
        );
        let reasoning = spec.capability.reasoning.expect("reasoning attached");
        assert_eq!(reasoning.default_effort.as_deref(), Some("xhigh"));
        assert!(reasoning.efforts.contains(&"xhigh".to_string()));
    }

    #[test]
    fn unmatched_model_attaches_empty_capability() {
        let spec = resolved("mystery-model", "mystery-model")
            .into_model_spec("anthropic", None)
            .expect("spec");
        assert!(spec.capability.is_empty());
    }

    #[test]
    fn default_variant_matches_the_catalog_row() {
        assert_eq!(
            default_variant(&provider("anthropic"), "claude-opus-4-7").as_deref(),
            Some("xhigh")
        );
        assert_eq!(
            default_variant(&provider("codex"), "gpt-5.5").as_deref(),
            Some("xhigh")
        );
        assert_eq!(
            default_variant(&provider("openai"), "gpt-5.4").as_deref(),
            Some("medium")
        );
    }

    #[test]
    fn resolve_default_variant_falls_back_to_catalog_default() {
        assert_eq!(
            resolve_model_variant(&provider("openai"), "gpt-5.4", None).expect("ok"),
            Some("medium".to_string())
        );
        assert_eq!(
            resolve_model_variant(&provider("openai"), "gpt-5.4", Some("default")).expect("ok"),
            Some("medium".to_string())
        );
    }

    #[test]
    fn unsupported_variant_is_rejected_with_unsupported_effort_message() {
        let err = resolve_model_variant(&provider("openai"), "gpt-5.4", Some("turbo"))
            .expect_err("turbo is not an exposed effort");
        assert!(err.contains("Unsupported effort `turbo`"), "message: {err}");
        assert!(err.contains("gpt-5.4"), "names the model: {err}");
    }

    #[test]
    fn minimal_on_openai_gpt_5_4_resolves_to_low_via_alias() {
        assert_eq!(
            resolve_model_variant(&provider("openai"), "gpt-5.4", Some("minimal")).expect("ok"),
            Some("low".to_string())
        );
    }

    #[test]
    fn variant_on_a_model_without_efforts_is_not_configurable() {
        let err = resolve_model_variant(&provider("anthropic"), "mystery-model", Some("high"))
            .expect_err("no efforts exposed");
        assert!(
            err.contains("does not expose configurable effort"),
            "message: {err}"
        );
    }

    #[test]
    fn codex_route_clamps_prompt_budget_to_input_ceiling() {
        assert_eq!(provider_input_ceiling("codex"), Some(256_000));
        assert_eq!(provider_input_ceiling("openai"), None);

        // On codex, gpt-5.5 resolves to the generic openai entry (922k input);
        // the route clamp must bring the prompt budget down to the codex
        // ceiling so history pruning trims before the real limit.
        let mut info = ModelInfo {
            context_window: 1_050_000,
            max_input_tokens: Some(922_000),
            max_output_tokens: Some(128_000),
        };
        let ceiling = provider_input_ceiling("codex").expect("codex ceiling");
        info.max_input_tokens = Some(info.prompt_budget_tokens().min(ceiling));
        assert_eq!(info.prompt_budget_tokens(), 256_000);
    }
}
