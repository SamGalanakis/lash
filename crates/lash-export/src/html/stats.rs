use std::collections::HashMap;

use lash_core::session_model::{MessageRole, PruneState};
use lash_core::{ChronologicalPayload, ToolCallRecord};

use crate::LoadedSession;

use super::chronological_rlm_step;

#[derive(Default, Debug)]
pub(crate) struct SessionStats {
    pub(crate) user_messages: usize,
    pub(crate) assistant_messages: usize,
    pub(crate) system_messages: usize,
    pub(crate) tool_calls_ok: usize,
    pub(crate) tool_calls_err: usize,
    pub(crate) tool_calls_cancelled: usize,
    pub(crate) tool_total_ms: u64,
    pub(crate) rlm_iterations: usize,
    pub(crate) rlm_errors: usize,
    pub(crate) pruned_parts: usize,
    pub(crate) cleared_parts: usize,
    pub(crate) total_chars: usize,
    pub(crate) chronological: usize,
    pub(crate) llm_calls_with_usage: usize,
    pub(crate) input_tokens: i64,
    pub(crate) output_tokens: i64,
    pub(crate) cached_input_tokens: i64,
    pub(crate) reasoning_tokens: i64,
    pub(crate) max_context_percent: Option<f64>,
    pub(crate) tool_freq: Vec<(String, usize)>,
    pub(crate) tool_names_set: Vec<String>,
    /// Best-effort dollar cost estimate from per-model pricing. Conservative
    /// when the model is unknown.
    pub(crate) est_cost_usd: f64,
    /// Number of LLM calls where reasoning_tokens > 0 but no reasoning text
    /// was returned by the provider — surfaced so the renderer can show
    /// "≈Nk reasoning tokens · content not returned".
    pub(crate) reasoning_only_calls: usize,
    pub(crate) reasoning_only_tokens: i64,
}

/// Coarse per-million-token pricing in USD. Falls back to a conservative
/// "frontier reasoning model" estimate when the model is unknown.
fn model_pricing_per_million(model: Option<&str>) -> (f64, f64, f64) {
    // (input, cached_input, output)
    let m = model.unwrap_or("").to_ascii_lowercase();
    if m.starts_with("gpt-5") {
        (1.25, 0.125, 10.0)
    } else if m.starts_with("gpt-4.1") || m.starts_with("gpt-4o") {
        (2.50, 0.50, 10.0)
    } else if m.contains("claude-opus") {
        (15.0, 1.50, 75.0)
    } else if m.contains("claude-sonnet") || m.contains("claude-3-7") {
        (3.0, 0.30, 15.0)
    } else if m.contains("claude-haiku") {
        (0.80, 0.08, 4.0)
    } else if m.contains("gemini-2") {
        (1.25, 0.30, 10.0)
    } else {
        // Conservative frontier-reasoning default
        (3.0, 0.30, 15.0)
    }
}

pub(crate) fn compute_stats(session: &LoadedSession) -> SessionStats {
    let mut s = SessionStats {
        chronological: session.chronological.len(),
        ..SessionStats::default()
    };
    let mut tool_counts: HashMap<String, usize> = HashMap::new();

    let record_tool_call = |s: &mut SessionStats,
                            tool_counts: &mut HashMap<String, usize>,
                            record: &ToolCallRecord| {
        match record.output.status() {
            lash_core::ToolCallStatus::Success => s.tool_calls_ok += 1,
            lash_core::ToolCallStatus::Failure => s.tool_calls_err += 1,
            lash_core::ToolCallStatus::Cancelled => s.tool_calls_cancelled += 1,
        }
        s.tool_total_ms = s.tool_total_ms.saturating_add(record.duration_ms);
        *tool_counts.entry(record.tool.clone()).or_insert(0) += 1;
    };

    for entry in &session.chronological {
        match &entry.payload {
            ChronologicalPayload::Message(message) => {
                if message.is_transient() {
                    continue;
                }
                match message.role {
                    MessageRole::User => s.user_messages += 1,
                    MessageRole::Assistant => s.assistant_messages += 1,
                    MessageRole::System => s.system_messages += 1,
                }
                for part in message.parts.iter() {
                    s.total_chars = s.total_chars.saturating_add(part.content.chars().count());
                    match &part.prune_state {
                        PruneState::Cleared => s.cleared_parts += 1,
                        PruneState::Deleted { .. } | PruneState::Summarized { .. } => {
                            s.pruned_parts += 1
                        }
                        PruneState::Intact => {}
                    }
                }
            }
            ChronologicalPayload::ToolCall(record) => {
                record_tool_call(&mut s, &mut tool_counts, record);
            }
            ChronologicalPayload::ProtocolEvent(event) => {
                let Some(step) = chronological_rlm_step(event) else {
                    continue;
                };
                s.rlm_iterations += 1;
                if step.error.is_some() {
                    s.rlm_errors += 1;
                }
                s.total_chars = s.total_chars.saturating_add(step.output_chars());
                s.total_chars = s.total_chars.saturating_add(step.code.chars().count());
            }
        }
    }

    let mut freq: Vec<(String, usize)> = tool_counts.into_iter().collect();
    freq.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    s.tool_names_set = freq.iter().map(|(name, _)| name.clone()).collect();
    s.tool_freq = freq;

    for prompt in &session.llm_prompts {
        let Some(usage) = &prompt.usage else {
            continue;
        };
        s.llm_calls_with_usage += 1;
        s.input_tokens = s.input_tokens.saturating_add(usage.input_tokens);
        s.output_tokens = s.output_tokens.saturating_add(usage.output_tokens);
        s.cached_input_tokens = s
            .cached_input_tokens
            .saturating_add(usage.cached_input_tokens);
        s.reasoning_tokens = s.reasoning_tokens.saturating_add(usage.reasoning_tokens);
        if usage.reasoning_tokens > 0 {
            s.reasoning_only_calls += 1;
            s.reasoning_only_tokens = s
                .reasoning_only_tokens
                .saturating_add(usage.reasoning_tokens);
        }
        let (in_per_m, cached_per_m, out_per_m) =
            model_pricing_per_million(prompt.model.as_deref());
        let billed_input = (usage.input_tokens.saturating_sub(usage.cached_input_tokens)).max(0);
        s.est_cost_usd += billed_input as f64 * in_per_m / 1_000_000.0;
        s.est_cost_usd += usage.cached_input_tokens.max(0) as f64 * cached_per_m / 1_000_000.0;
        s.est_cost_usd += usage.output_tokens.max(0) as f64 * out_per_m / 1_000_000.0;
        if let Some(context_window) = session.context_window_tokens
            && context_window > 0
        {
            let pct = usage.input_tokens.max(0) as f64 * 100.0 / context_window as f64;
            s.max_context_percent = Some(s.max_context_percent.map_or(pct, |max| max.max(pct)));
        }
    }
    s
}

// ─── render context ─────────────────────────────────────────────────────────
