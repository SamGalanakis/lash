use lash_core::ChronologicalPayload;
use lash_core::session_model::{MessageRole, PruneState};

use crate::LoadedSession;
use crate::transcript::lashlang_transcript_step_from_event;

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
    pub(crate) cache_read_input_tokens: i64,
    pub(crate) cache_write_input_tokens: i64,
    pub(crate) reasoning_output_tokens: i64,
    pub(crate) max_context_percent: Option<f64>,
    pub(crate) tool_freq: Vec<(String, usize)>,
    pub(crate) tool_names_set: Vec<String>,
    /// Best-effort dollar cost estimate from per-model pricing. Conservative
    /// when the model is unknown.
    pub(crate) est_cost_usd: f64,
    /// Number of LLM calls where reasoning_output_tokens > 0 but no reasoning text
    /// was returned by the provider — surfaced so the renderer can show
    /// "≈Nk reasoning tokens · content not returned".
    pub(crate) reasoning_only_calls: usize,
    pub(crate) reasoning_only_tokens: i64,
}

/// Coarse per-million-token pricing in USD. Falls back to a conservative
/// "frontier reasoning model" estimate when the model is unknown.
fn model_pricing_per_million(model: Option<&str>) -> (f64, f64, f64, f64) {
    // (input, cache_read, cache_write, output)
    let m = model.unwrap_or("").to_ascii_lowercase();
    if m.starts_with("gpt-5") {
        (1.25, 0.125, 1.25, 10.0)
    } else if m.starts_with("gpt-4.1") || m.starts_with("gpt-4o") {
        (2.50, 0.50, 2.50, 10.0)
    } else if m.contains("claude-opus") {
        (15.0, 1.50, 18.75, 75.0)
    } else if m.contains("claude-sonnet") || m.contains("claude-3-7") {
        (3.0, 0.30, 3.75, 15.0)
    } else if m.contains("claude-haiku") {
        (0.80, 0.08, 1.0, 4.0)
    } else if m.contains("gemini-2") {
        (1.25, 0.30, 1.25, 10.0)
    } else {
        // Conservative frontier-reasoning default
        (3.0, 0.30, 3.0, 15.0)
    }
}

pub(crate) fn compute_stats(session: &LoadedSession) -> SessionStats {
    let mut s = SessionStats {
        chronological: session.chronological.len(),
        ..SessionStats::default()
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
                    MessageRole::Event => s.system_messages += 1,
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
            ChronologicalPayload::ProtocolEvent(event) => {
                let Some(step) = lashlang_transcript_step_from_event(event) else {
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

    for prompt in &session.llm_prompts {
        let Some(usage) = &prompt.usage else {
            continue;
        };
        s.llm_calls_with_usage += 1;
        s.input_tokens = s.input_tokens.saturating_add(usage.input_tokens);
        s.output_tokens = s.output_tokens.saturating_add(usage.output_tokens);
        s.cache_read_input_tokens = s
            .cache_read_input_tokens
            .saturating_add(usage.cache_read_input_tokens);
        s.cache_write_input_tokens = s
            .cache_write_input_tokens
            .saturating_add(usage.cache_write_input_tokens);
        s.reasoning_output_tokens = s
            .reasoning_output_tokens
            .saturating_add(usage.reasoning_output_tokens);
        if usage.reasoning_output_tokens > 0 {
            s.reasoning_only_calls += 1;
            s.reasoning_only_tokens = s
                .reasoning_only_tokens
                .saturating_add(usage.reasoning_output_tokens);
        }
        let (in_per_m, cache_read_per_m, cache_write_per_m, out_per_m) =
            model_pricing_per_million(prompt.model.as_deref());
        s.est_cost_usd += usage.input_tokens.max(0) as f64 * in_per_m / 1_000_000.0;
        s.est_cost_usd +=
            usage.cache_read_input_tokens.max(0) as f64 * cache_read_per_m / 1_000_000.0;
        s.est_cost_usd +=
            usage.cache_write_input_tokens.max(0) as f64 * cache_write_per_m / 1_000_000.0;
        s.est_cost_usd += usage.output_tokens.max(0) as f64 * out_per_m / 1_000_000.0;
        if let Some(context_window) = session.context_window_tokens
            && context_window > 0
        {
            let prompt_tokens = usage
                .input_tokens
                .saturating_add(usage.cache_read_input_tokens)
                .saturating_add(usage.cache_write_input_tokens)
                .max(0);
            let pct = prompt_tokens as f64 * 100.0 / context_window as f64;
            s.max_context_percent = Some(s.max_context_percent.map_or(pct, |max| max.max(pct)));
        }
    }
    s
}

// ─── render context ─────────────────────────────────────────────────────────
