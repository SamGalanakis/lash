# Stream termination is explicit dialect policy

A clean transport EOF does not prove that an LLM response completed. Chat Completions can
lose its final `finish_reason`; Responses can lose `response.completed`; Anthropic Messages
can lose `message_stop`. Lash previously assembled whatever parts had arrived and inferred a
successful stop or tool use, which made a truncated response indistinguishable from a complete
one and could make an incomplete tool call executable. Usage observed before the failure also
fell out of the retry attempt ledger.

We decided that **stream completion requires dialect-specific terminal evidence unless the host
explicitly selects EOF tolerance**. The host-supplied policy is
`StreamTermination::RequireTerminalEvidence | EofTolerated`. It follows the same ADR 0026/0034
configuration path as cache capability data: `ModelCapability.stream_termination` is a
route/model override; OpenAI-compatible endpoint defaults live in
`OpenAiCompat.stream_termination`; direct Anthropic and Google providers expose the same policy
on their constructors and serialized config. There is no URL, model-name, payload, or `[DONE]`
inference.

The defaults are semantic dialect facts. OpenAI Chat Completions requires a nonempty
`finish_reason`. OpenAI and Codex Responses require a terminal response event
(`response.completed`, `response.incomplete`, `response.failed`, or `response.done`). Anthropic
requires `message_stop` after `message_start`. Google tolerates EOF because its streaming
dialect legitimately uses that boundary in deployments Lash supports. OpenAI-compatible hosts
whose endpoints also require EOF tolerance must set it explicitly; `OpenAiCompat::openrouter()`
selects strict terminal evidence.

Missing required evidence is a retryable `ProviderFailureKind::Stream` failure, never a partial
success. The transport error carries a partial `LlmResponse` containing accumulated text,
reasoning, tool-call fragments, normalized usage, raw `provider_usage`, and execution evidence.
That partial crosses the runtime effect boundary for diagnosis and accounting but is never
passed to the protocol as a completed response. `ProviderHandle` records its observed usage and
evidence on the ADR 0032 `Interrupted` attempt. Before a whole-call retry it emits an attempt
reset, so runtime accumulators discard provisional tool parts and usage from the failed try.
Explicit user cancellation remains `Cancelled`, non-retryable, and distinct from truncation.

This is intentionally breaking for nonconforming OpenAI-compatible embedders. Streams that
formerly appeared successful at bare EOF now fail. Hosts may opt a known EOF-terminated route
into `EofTolerated`; doing so is an explicit compatibility contract, not a heuristic fallback.
