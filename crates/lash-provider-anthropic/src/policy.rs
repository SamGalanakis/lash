//! Anthropic wire constants and the internal thinking-config shape.
//!
//! Model capability (which efforts a model exposes and how they map onto the
//! wire) is host-supplied data threaded onto every [`LlmRequest`] and validated
//! by lash-core before a provider sees it. This crate never sniffs model names
//! to infer capability; it only translates the already-resolved variant plus
//! the request's `model_capability` into the Anthropic `thinking`/`output_config`
//! wire shape (see `request.rs`).

pub(crate) const ANTHROPIC_VERSION: &str = "2023-06-01";
pub(crate) const FINE_GRAINED_BETA: &str = "fine-grained-tool-streaming-2025-05-14";
pub(crate) const INTERLEAVED_THINKING_BETA: &str = "interleaved-thinking-2025-05-14";
pub(crate) const DEFAULT_MAX_OUTPUT_TOKENS: u64 = 32_768;

/// Resolved thinking configuration for a single request, in the two shapes
/// Anthropic accepts: adaptive (named effort via `output_config`) and budget
/// (token budget via `thinking`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum AnthropicThinkingConfig {
    Adaptive { effort: String },
    Budget { budget_tokens: i32 },
}
