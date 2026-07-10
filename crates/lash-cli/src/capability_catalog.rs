//! Host-side model-capability catalog.
//!
//! Capability is host-supplied data on [`ModelSpec`](lash_core::ModelSpec): the
//! effort levels a model exposes, the default effort, alias clamps, and how the
//! effort encodes on the wire. lash-core validates a requested variant against
//! this and the provider consumes it — providers no longer sniff model names.
//!
//! This module is the reference host's catalog: an ordered table of pattern
//! rules (first match wins), keyed by provider kind. It mirrors the model-limit
//! table in [`model_selection`](crate::model_selection) — data, not code, with a
//! single [`builtin_capability_override`] precedence hook (returns `None` today,
//! same as `builtin_model_info`).
//!
//! Adding a model means adding a row. No model-name capability logic lives
//! anywhere else in the CLI.

use lash_core::{
    ModelCapability, ReasoningCapability, ReasoningDisableEncoding, ReasoningEncoding,
};

/// How a rule matches a model id. `full` is the lowercased model id as
/// configured; `bare` is that id after stripping any leading `provider/`
/// segment (the part after the last `/`), used where the deleted provider
/// policies matched on the prefix-stripped id.
enum Match {
    /// `full` contains any of these substrings.
    Contains(&'static [&'static str]),
    /// `bare` starts with any of these prefixes.
    Prefix(&'static [&'static str]),
    /// `bare` equals this id exactly.
    ExactId(&'static str),
    /// `full` contains every string in `all` and at least one string in `any`.
    ContainsAllAndAny {
        all: &'static [&'static str],
        any: &'static [&'static str],
    },
}

impl Match {
    fn matches(&self, full: &str, bare: &str) -> bool {
        match self {
            Match::Contains(needles) => needles.iter().any(|needle| full.contains(needle)),
            Match::Prefix(prefixes) => prefixes.iter().any(|prefix| bare.starts_with(prefix)),
            Match::ExactId(id) => bare == *id,
            Match::ContainsAllAndAny { all, any } => {
                all.iter().all(|needle| full.contains(needle))
                    && any.iter().any(|needle| full.contains(needle))
            }
        }
    }
}

/// Wire encoding for a rule, in const-friendly form.
enum Encoding {
    /// Named effort level sent as-is (OpenAI reasoning.effort, Anthropic
    /// adaptive thinking, Gemini thinkingLevel).
    Effort,
    /// Effort name resolves to a token budget (Anthropic budget thinking,
    /// Gemini 2.5 thinkingBudget); pairs are `effort -> tokens`.
    Budget(&'static [(&'static str, u32)]),
}

enum DisableEncoding {
    Native,
    Omit,
    Effort(&'static str),
    Budget(u32),
}

/// One catalog row: a `(provider kind, match rule, capability)` tuple, stored as
/// fully const data so the whole table reads as a spec.
struct Row {
    kind: &'static str,
    rule: Match,
    efforts: &'static [&'static str],
    default_effort: &'static str,
    /// requested-effort -> canonical-effort clamps (e.g. `minimal -> low`).
    aliases: &'static [(&'static str, &'static str)],
    encoding: Encoding,
    disable: Option<DisableEncoding>,
}

impl Row {
    fn to_capability(&self) -> ModelCapability {
        let encoding = match self.encoding {
            Encoding::Effort => ReasoningEncoding::Effort,
            Encoding::Budget(pairs) => ReasoningEncoding::Budget(
                pairs
                    .iter()
                    .map(|(effort, tokens)| ((*effort).to_string(), *tokens))
                    .collect(),
            ),
        };
        ModelCapability {
            reasoning: Some(ReasoningCapability {
                efforts: self.efforts.iter().map(|e| (*e).to_string()).collect(),
                default_effort: Some(self.default_effort.to_string()),
                aliases: self
                    .aliases
                    .iter()
                    .map(|(from, to)| ((*from).to_string(), (*to).to_string()))
                    .collect(),
                encoding,
                disable: self.disable.as_ref().map(|encoding| match encoding {
                    DisableEncoding::Native => ReasoningDisableEncoding::Native,
                    DisableEncoding::Omit => ReasoningDisableEncoding::Omit,
                    DisableEncoding::Effort(effort) => {
                        ReasoningDisableEncoding::Effort((*effort).to_string())
                    }
                    DisableEncoding::Budget(budget) => ReasoningDisableEncoding::Budget(*budget),
                }),
                mandatory: false,
            }),
        }
    }
}

/// The ordered capability table. First matching row wins, so ordering is
/// load-bearing: adaptive/newer rows precede budget/older fallbacks, and
/// `gemini-3.1` precedes `gemini-3`. Behavior-neutral port of the deleted
/// provider-crate sniffs.
const ROWS: &[Row] = &[
    // ── anthropic (direct API) ──
    // Adaptive-thinking rows come before the budget fallback so e.g.
    // `claude-opus-4-7` matches opus-4-7 (adaptive), not claude-opus-4 (budget).
    Row {
        kind: "anthropic",
        rule: Match::Contains(&["opus-4-7", "opus-4.7"]),
        efforts: &["low", "medium", "high", "xhigh"],
        default_effort: "xhigh",
        aliases: &[],
        encoding: Encoding::Effort,
        disable: Some(DisableEncoding::Native),
    },
    Row {
        kind: "anthropic",
        rule: Match::Contains(&["opus-4-6", "opus-4.6"]),
        efforts: &["low", "medium", "high", "max"],
        default_effort: "max",
        aliases: &[],
        encoding: Encoding::Effort,
        disable: Some(DisableEncoding::Native),
    },
    Row {
        kind: "anthropic",
        rule: Match::Contains(&["sonnet-4-6", "sonnet-4.6"]),
        efforts: &["low", "medium", "high"],
        default_effort: "high",
        aliases: &[],
        encoding: Encoding::Effort,
        disable: Some(DisableEncoding::Native),
    },
    // Budget fallback disables by explicitly omitting the thinking block.
    Row {
        kind: "anthropic",
        rule: Match::Contains(&["haiku-4", "claude-opus-4", "claude-sonnet-4"]),
        efforts: &["low", "medium", "high"],
        default_effort: "high",
        aliases: &[],
        encoding: Encoding::Budget(&[("low", 1024), ("medium", 4096), ("high", 12288)]),
        disable: Some(DisableEncoding::Omit),
    },
    // ── google_oauth ──
    Row {
        kind: "google_oauth",
        rule: Match::Contains(&["gemini-2.5"]),
        efforts: &["high", "max"],
        default_effort: "high",
        aliases: &[],
        encoding: Encoding::Budget(&[("high", 16000), ("max", 24576)]),
        disable: Some(DisableEncoding::Budget(0)),
    },
    // gemini-3.1 must precede gemini-3 so `gemini-3.1-pro-preview` matches here.
    Row {
        kind: "google_oauth",
        rule: Match::Contains(&["gemini-3.1"]),
        efforts: &["low", "medium", "high"],
        default_effort: "high",
        aliases: &[],
        encoding: Encoding::Effort,
        disable: None,
    },
    Row {
        kind: "google_oauth",
        rule: Match::Contains(&["gemini-3"]),
        efforts: &["low", "high"],
        default_effort: "high",
        aliases: &[],
        encoding: Encoding::Effort,
        disable: None,
    },
    // ── openai (direct Responses API) ──
    // Prefixes match the prefix-stripped id. gpt-5.2/5.3/5.4 drop `minimal`
    // from efforts and clamp it to `low`.
    Row {
        kind: "openai",
        rule: Match::Prefix(&["gpt-5.2", "gpt-5.3", "gpt-5.4"]),
        efforts: &["low", "medium", "high", "xhigh"],
        default_effort: "medium",
        aliases: &[("minimal", "low")],
        encoding: Encoding::Effort,
        disable: Some(DisableEncoding::Effort("none")),
    },
    Row {
        kind: "openai",
        rule: Match::Prefix(&["gpt-5"]),
        efforts: &["minimal", "low", "medium", "high", "xhigh"],
        default_effort: "medium",
        aliases: &[],
        encoding: Encoding::Effort,
        disable: Some(DisableEncoding::Effort("none")),
    },
    Row {
        kind: "openai",
        rule: Match::Prefix(&["o"]),
        efforts: &["minimal", "low", "medium", "high", "xhigh"],
        default_effort: "medium",
        aliases: &[],
        encoding: Encoding::Effort,
        disable: Some(DisableEncoding::Effort("none")),
    },
    // ── openai-compatible (OpenRouter base URL) ──
    Row {
        kind: "openai-compatible",
        rule: Match::Contains(&["gpt-5.2", "gpt-5.3", "gpt-5.4"]),
        efforts: &["low", "medium", "high", "xhigh"],
        default_effort: "medium",
        aliases: &[("minimal", "low")],
        encoding: Encoding::Effort,
        disable: Some(DisableEncoding::Effort("none")),
    },
    Row {
        kind: "openai-compatible",
        rule: Match::Contains(&["gpt"]),
        efforts: &["minimal", "low", "medium", "high", "xhigh"],
        default_effort: "medium",
        aliases: &[],
        encoding: Encoding::Effort,
        disable: Some(DisableEncoding::Effort("none")),
    },
    Row {
        kind: "openai-compatible",
        rule: Match::Contains(&["claude"]),
        efforts: &["minimal", "low", "medium", "high", "xhigh"],
        default_effort: "high",
        aliases: &[],
        encoding: Encoding::Effort,
        disable: Some(DisableEncoding::Effort("none")),
    },
    Row {
        kind: "openai-compatible",
        rule: Match::Contains(&["gemini-3"]),
        efforts: &["minimal", "low", "medium", "high", "xhigh"],
        default_effort: "high",
        aliases: &[],
        encoding: Encoding::Effort,
        disable: Some(DisableEncoding::Effort("none")),
    },
    // ── codex (OAuth route; only gpt-5* models) ──
    // Exact gpt-5.5 first, then codex variants, then plain gpt-5 variants.
    // "xhigh suffix" = the id contains 5.2/5.3/5.4.
    Row {
        kind: "codex",
        rule: Match::ExactId("gpt-5.5"),
        efforts: &["low", "medium", "high", "xhigh"],
        default_effort: "xhigh",
        aliases: &[],
        encoding: Encoding::Effort,
        disable: Some(DisableEncoding::Effort("none")),
    },
    Row {
        kind: "codex",
        rule: Match::ContainsAllAndAny {
            all: &["codex"],
            any: &["5.2", "5.3", "5.4"],
        },
        efforts: &["low", "medium", "high", "xhigh"],
        default_effort: "xhigh",
        aliases: &[],
        encoding: Encoding::Effort,
        disable: Some(DisableEncoding::Effort("none")),
    },
    Row {
        kind: "codex",
        rule: Match::Contains(&["codex"]),
        efforts: &["low", "medium", "high"],
        default_effort: "high",
        aliases: &[],
        encoding: Encoding::Effort,
        disable: Some(DisableEncoding::Effort("none")),
    },
    // gpt-5.2/5.3/5.4 drop `minimal` and clamp it to `low`.
    Row {
        kind: "codex",
        rule: Match::ContainsAllAndAny {
            all: &["gpt-5"],
            any: &["5.2", "5.3", "5.4"],
        },
        efforts: &["low", "medium", "high", "xhigh"],
        default_effort: "xhigh",
        aliases: &[("minimal", "low")],
        encoding: Encoding::Effort,
        disable: Some(DisableEncoding::Effort("none")),
    },
    Row {
        kind: "codex",
        rule: Match::Contains(&["gpt-5"]),
        efforts: &["minimal", "low", "medium", "high"],
        default_effort: "high",
        aliases: &[],
        encoding: Encoding::Effort,
        disable: Some(DisableEncoding::Effort("none")),
    },
];

/// Curated capability overrides that take precedence over the pattern table —
/// only for `(kind, model)` pairs the table gets wrong. Mirrors
/// `builtin_model_info`: returns `None` today, kept as the precedence hook.
fn builtin_capability_override(_kind: &str, _model: &str) -> Option<ModelCapability> {
    None
}

/// Resolve the host-supplied capability for a model on a provider kind. Built-in
/// overrides win; then the first matching pattern row; no match yields
/// [`ModelCapability::default`] (no configurable effort).
pub(crate) fn capability_for(kind: &str, model: &str) -> ModelCapability {
    if let Some(override_capability) = builtin_capability_override(kind, model) {
        return override_capability;
    }
    let full = model.to_lowercase();
    let bare = full.rsplit('/').next().unwrap_or(&full);
    ROWS.iter()
        .find(|row| row.kind == kind && row.rule.matches(&full, bare))
        .map(Row::to_capability)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn efforts(cap: &ModelCapability) -> Vec<String> {
        cap.reasoning
            .as_ref()
            .expect("reasoning present")
            .efforts
            .clone()
    }

    fn default_effort(cap: &ModelCapability) -> Option<String> {
        cap.reasoning
            .as_ref()
            .expect("reasoning present")
            .default_effort
            .clone()
    }

    #[test]
    fn no_match_yields_default_empty_capability() {
        let cap = capability_for("anthropic", "some-unknown-model");
        assert!(cap.is_empty());
        let cap = capability_for("unknown-kind", "gpt-5.4");
        assert!(cap.is_empty());
    }

    #[test]
    fn anthropic_opus_4_7_is_adaptive_xhigh_not_budget_fallback() {
        // Ordering: opus-4-7 (adaptive) must win over the claude-opus-4 budget
        // fallback even though the id also contains claude-opus-4.
        let cap = capability_for("anthropic", "claude-opus-4-7");
        assert_eq!(default_effort(&cap).as_deref(), Some("xhigh"));
        assert_eq!(
            efforts(&cap),
            vec!["low", "medium", "high", "xhigh"],
            "adaptive row efforts"
        );
        assert_eq!(
            cap.reasoning.as_ref().unwrap().encoding,
            ReasoningEncoding::Effort
        );
    }

    #[test]
    fn anthropic_budget_fallback_uses_budget_encoding() {
        let cap = capability_for("anthropic", "claude-opus-4-1");
        assert_eq!(default_effort(&cap).as_deref(), Some("high"));
        assert_eq!(efforts(&cap), vec!["low", "medium", "high"]);
        assert_eq!(
            cap.reasoning.as_ref().unwrap().disable,
            Some(ReasoningDisableEncoding::Omit)
        );
        match &cap.reasoning.as_ref().unwrap().encoding {
            ReasoningEncoding::Budget(map) => {
                assert_eq!(map.get("low"), Some(&1024));
                assert_eq!(map.get("medium"), Some(&4096));
                assert_eq!(map.get("high"), Some(&12288));
                assert!(!map.contains_key("none"), "none absent from budget map");
            }
            other => panic!("expected budget encoding, got {other:?}"),
        }
    }

    #[test]
    fn google_gemini_3_1_precedes_gemini_3() {
        let cap31 = capability_for("google_oauth", "gemini-3.1-pro-preview");
        assert_eq!(efforts(&cap31), vec!["low", "medium", "high"]);
        // Plain gemini-3 falls through to the two-effort row.
        let cap3 = capability_for("google_oauth", "gemini-3-pro");
        assert_eq!(efforts(&cap3), vec!["low", "high"]);
    }

    #[test]
    fn google_gemini_2_5_uses_thinking_budget() {
        let cap = capability_for("google_oauth", "gemini-2.5-pro");
        assert_eq!(efforts(&cap), vec!["high", "max"]);
        match &cap.reasoning.as_ref().unwrap().encoding {
            ReasoningEncoding::Budget(map) => {
                assert_eq!(map.get("high"), Some(&16000));
                assert_eq!(map.get("max"), Some(&24576));
            }
            other => panic!("expected budget encoding, got {other:?}"),
        }
    }

    #[test]
    fn openai_gpt_5_4_drops_minimal_and_aliases_it_to_low() {
        let cap = capability_for("openai", "gpt-5.4");
        assert_eq!(default_effort(&cap).as_deref(), Some("medium"));
        assert_eq!(efforts(&cap), vec!["low", "medium", "high", "xhigh"]);
        assert!(!efforts(&cap).contains(&"minimal".to_string()));
        assert_eq!(cap.resolve_effort("minimal").as_deref(), Some("low"));
        assert_eq!(
            cap.reasoning.as_ref().unwrap().disable,
            Some(ReasoningDisableEncoding::Effort("none".to_string()))
        );
    }

    #[test]
    fn openai_pre_5_2_gpt_5_keeps_minimal() {
        let cap = capability_for("openai", "gpt-5.1");
        assert!(efforts(&cap).contains(&"minimal".to_string()));
        assert!(cap.reasoning.as_ref().unwrap().aliases.is_empty());
    }

    #[test]
    fn openai_o_series_prefix_matches() {
        let cap = capability_for("openai", "o4-mini");
        assert_eq!(default_effort(&cap).as_deref(), Some("medium"));
        assert!(efforts(&cap).contains(&"minimal".to_string()));
    }

    #[test]
    fn openai_compatible_claude_defaults_high() {
        let cap = capability_for("openai-compatible", "anthropic/claude-sonnet-4.6");
        assert_eq!(default_effort(&cap).as_deref(), Some("high"));
        // gpt substring wins earlier only when present; claude row here.
        let gpt = capability_for("openai-compatible", "openai/gpt-4o");
        assert_eq!(default_effort(&gpt).as_deref(), Some("medium"));
    }

    #[test]
    fn openai_compatible_gpt_5_4_aliases_minimal() {
        let cap = capability_for("openai-compatible", "openai/gpt-5.4");
        assert_eq!(cap.resolve_effort("minimal").as_deref(), Some("low"));
        assert!(!efforts(&cap).contains(&"minimal".to_string()));
    }

    #[test]
    fn codex_gpt_5_5_exact_id_is_xhigh() {
        let cap = capability_for("codex", "gpt-5.5");
        assert_eq!(default_effort(&cap).as_deref(), Some("xhigh"));
        assert_eq!(efforts(&cap), vec!["low", "medium", "high", "xhigh"]);
    }

    #[test]
    fn codex_plain_gpt_5_4_aliases_minimal_and_is_xhigh() {
        let cap = capability_for("codex", "gpt-5.4");
        assert_eq!(default_effort(&cap).as_deref(), Some("xhigh"));
        assert_eq!(cap.resolve_effort("minimal").as_deref(), Some("low"));
    }

    #[test]
    fn codex_pre_5_2_gpt_5_defaults_high_and_keeps_minimal() {
        let cap = capability_for("codex", "gpt-5.1");
        assert_eq!(default_effort(&cap).as_deref(), Some("high"));
        assert_eq!(efforts(&cap), vec!["minimal", "low", "medium", "high"]);
    }

    #[test]
    fn codex_codex_model_without_xhigh_suffix_defaults_high() {
        let cap = capability_for("codex", "gpt-5-codex");
        assert_eq!(default_effort(&cap).as_deref(), Some("high"));
        assert_eq!(efforts(&cap), vec!["low", "medium", "high"]);
    }
}
