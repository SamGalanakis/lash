//! Token usage accounting: ledger entries, usage totals, reports, and diff helpers.
//!
//! Extracted from `runtime/mod.rs` as part of the runtime split. All items
//! keep their original public paths via `pub use` in `mod.rs` — no API
//! changes.

use std::collections::{BTreeMap, HashMap};

use lash_sansio::PromptUsage;

use crate::provider::ProviderHandle;
use crate::session_model::TokenUsage;

/// A single row in the token cost ledger. One per unique
/// `(source, model)` pair — accumulated, not per-call.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct TokenLedgerEntry {
    /// Caller-supplied label: `"turn"`, `"subagent"`, `"compaction"`,
    /// `"observer"`, `"reflector"`, or any plugin-defined
    /// string. Core doesn't interpret the value; the UI uses it for
    /// grouping and display.
    pub source: String,
    /// Model identifier used for the LLM call (e.g.
    /// `"anthropic/claude-haiku-4-5"`).
    pub model: String,
    /// Accumulated token counts for this `(source, model)` pair.
    pub usage: TokenUsage,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct UsageTotals {
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cached_input_tokens: i64,
    #[serde(default)]
    pub reasoning_tokens: i64,
    pub total_tokens: i64,
    pub context_total_tokens: i64,
}

impl UsageTotals {
    fn from_usage(usage: &TokenUsage) -> Self {
        let total_tokens = usage.total();
        Self {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            cached_input_tokens: usage.cached_input_tokens,
            reasoning_tokens: usage.reasoning_tokens,
            total_tokens,
            context_total_tokens: total_tokens + usage.cached_input_tokens,
        }
    }
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct UsageReportRow {
    pub source: String,
    pub model: String,
    pub usage: UsageTotals,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct SessionUsageReport {
    pub entry_count: usize,
    pub usage: UsageTotals,
    pub by_source: BTreeMap<String, UsageTotals>,
    pub by_model: BTreeMap<String, UsageTotals>,
    pub by_source_model: Vec<UsageReportRow>,
}

impl SessionUsageReport {
    pub fn from_entries(entries: &[TokenLedgerEntry]) -> Self {
        let mut total = TokenUsage::default();
        let mut by_source_usage = BTreeMap::<String, TokenUsage>::new();
        let mut by_model_usage = BTreeMap::<String, TokenUsage>::new();
        let mut by_source_model = Vec::with_capacity(entries.len());

        for entry in entries {
            total.add(&entry.usage);
            by_source_usage
                .entry(entry.source.clone())
                .or_default()
                .add(&entry.usage);
            by_model_usage
                .entry(entry.model.clone())
                .or_default()
                .add(&entry.usage);
            by_source_model.push(UsageReportRow {
                source: entry.source.clone(),
                model: entry.model.clone(),
                usage: UsageTotals::from_usage(&entry.usage),
            });
        }

        Self {
            entry_count: entries.len(),
            usage: UsageTotals::from_usage(&total),
            by_source: by_source_usage
                .into_iter()
                .map(|(key, usage)| (key, UsageTotals::from_usage(&usage)))
                .collect(),
            by_model: by_model_usage
                .into_iter()
                .map(|(key, usage)| (key, UsageTotals::from_usage(&usage)))
                .collect(),
            by_source_model,
        }
    }
}

pub fn diff_token_ledger(
    before: &[TokenLedgerEntry],
    after: &[TokenLedgerEntry],
) -> Result<Vec<TokenLedgerEntry>, String> {
    let before_index = before
        .iter()
        .map(|entry| ((entry.source.as_str(), entry.model.as_str()), &entry.usage))
        .collect::<HashMap<_, _>>();
    let after_index = after
        .iter()
        .map(|entry| ((entry.source.as_str(), entry.model.as_str()), &entry.usage))
        .collect::<HashMap<_, _>>();

    let mut keys = before_index
        .keys()
        .copied()
        .chain(after_index.keys().copied())
        .collect::<Vec<_>>();
    keys.sort_unstable();
    keys.dedup();

    let mut out = Vec::new();
    for (source, model) in keys {
        let before_usage = before_index
            .get(&(source, model))
            .copied()
            .cloned()
            .unwrap_or_default();
        let after_usage = after_index
            .get(&(source, model))
            .copied()
            .cloned()
            .unwrap_or_default();
        let delta = TokenUsage {
            input_tokens: after_usage.input_tokens - before_usage.input_tokens,
            output_tokens: after_usage.output_tokens - before_usage.output_tokens,
            cached_input_tokens: after_usage.cached_input_tokens - before_usage.cached_input_tokens,
            reasoning_tokens: after_usage.reasoning_tokens - before_usage.reasoning_tokens,
        };
        if delta.input_tokens < 0
            || delta.output_tokens < 0
            || delta.cached_input_tokens < 0
            || delta.reasoning_tokens < 0
        {
            return Err(format!(
                "token ledger decreased for source/model ({source}, {model})"
            ));
        }
        if delta.total() == 0 && delta.cached_input_tokens == 0 {
            continue;
        }
        out.push(TokenLedgerEntry {
            source: source.to_string(),
            model: model.to_string(),
            usage: delta,
        });
    }
    Ok(out)
}

pub fn diff_usage_reports(
    before: &SessionUsageReport,
    after: &SessionUsageReport,
) -> Result<Vec<TokenLedgerEntry>, String> {
    let before_entries = before
        .by_source_model
        .iter()
        .map(|row| TokenLedgerEntry {
            source: row.source.clone(),
            model: row.model.clone(),
            usage: TokenUsage {
                input_tokens: row.usage.input_tokens,
                output_tokens: row.usage.output_tokens,
                cached_input_tokens: row.usage.cached_input_tokens,
                reasoning_tokens: row.usage.reasoning_tokens,
            },
        })
        .collect::<Vec<_>>();
    let after_entries = after
        .by_source_model
        .iter()
        .map(|row| TokenLedgerEntry {
            source: row.source.clone(),
            model: row.model.clone(),
            usage: TokenUsage {
                input_tokens: row.usage.input_tokens,
                output_tokens: row.usage.output_tokens,
                cached_input_tokens: row.usage.cached_input_tokens,
                reasoning_tokens: row.usage.reasoning_tokens,
            },
        })
        .collect::<Vec<_>>();
    diff_token_ledger(&before_entries, &after_entries)
}

pub(super) fn merge_ledger_entry(ledger: &mut Vec<TokenLedgerEntry>, entry: TokenLedgerEntry) {
    if entry.usage.total() == 0 && entry.usage.cached_input_tokens == 0 {
        return;
    }
    if let Some(existing) = ledger
        .iter_mut()
        .find(|e| e.source == entry.source && e.model == entry.model)
    {
        existing.usage.input_tokens += entry.usage.input_tokens;
        existing.usage.output_tokens += entry.usage.output_tokens;
        existing.usage.cached_input_tokens += entry.usage.cached_input_tokens;
        existing.usage.reasoning_tokens += entry.usage.reasoning_tokens;
    } else {
        ledger.push(entry);
    }
}

pub(super) fn merge_usage_delta_entries(entries: Vec<TokenLedgerEntry>) -> Vec<TokenLedgerEntry> {
    let mut merged = Vec::new();
    for entry in entries {
        merge_ledger_entry(&mut merged, entry);
    }
    merged
}

pub(super) fn normalize_prompt_usage(
    provider: &ProviderHandle,
    usage: &TokenUsage,
) -> Option<PromptUsage> {
    let input_tokens = usage.input_tokens.max(0) as usize;
    let output_tokens = usage.output_tokens.max(0) as usize;
    let cached_input_tokens = usage.cached_input_tokens.max(0) as usize;
    if input_tokens == 0 && cached_input_tokens == 0 && output_tokens == 0 {
        return None;
    }

    let prompt_context_tokens = if provider.input_usage_excludes_cached_tokens() {
        input_tokens.saturating_add(cached_input_tokens)
    } else {
        input_tokens
    };
    let adjusted_input_tokens = if provider.input_usage_excludes_cached_tokens() {
        input_tokens
    } else {
        input_tokens.saturating_sub(cached_input_tokens)
    };
    let context_budget_tokens = adjusted_input_tokens
        .saturating_add(output_tokens)
        .saturating_add(cached_input_tokens);

    Some(PromptUsage {
        prompt_context_tokens,
        input_tokens,
        cached_input_tokens,
        context_budget_tokens,
    })
}
