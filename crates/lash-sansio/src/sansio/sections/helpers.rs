fn token_usage_from_llm_usage(usage: &crate::llm::types::LlmUsage) -> TokenUsage {
    TokenUsage {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        cache_read_input_tokens: usage.cache_read_input_tokens,
        cache_write_input_tokens: usage.cache_write_input_tokens,
        reasoning_output_tokens: usage.reasoning_output_tokens,
    }
}

/// Reclassify a zero-output `OutputLimit` terminal reason as `ContextOverflow`
/// when the prompt nearly filled the model's context window.
///
/// Pure policy: the kernel owns the terminal-reason interpretation, so the
/// provider's raw reason is refined here (before it drives the finish decision
/// in `handle_terminal_llm_response`) rather than in the host I/O layer. A
/// `None` window disables the refinement.
fn refine_terminal_reason_for_context_window(
    response: &mut LlmResponse,
    max_context_tokens: Option<usize>,
) {
    if response.terminal_reason != LlmTerminalReason::OutputLimit {
        return;
    }
    if response.usage.output_tokens != 0 {
        return;
    }
    let Some(max_context_tokens) = max_context_tokens.filter(|value| *value > 0) else {
        return;
    };
    let prompt_tokens = response.usage.input_total().max(0) as usize;
    if prompt_tokens >= max_context_tokens.saturating_mul(95) / 100 {
        response.terminal_reason = LlmTerminalReason::ContextOverflow;
        response.terminal_diagnostic = Some(
            "Model produced no output because the prompt reached the configured context window."
                .to_string(),
        );
    }
}
