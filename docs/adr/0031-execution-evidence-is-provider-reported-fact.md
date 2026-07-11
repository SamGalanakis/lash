# Execution evidence is provider-reported fact, never echoed intent

Providers report what actually ran — the served model id, a response identity, a reasoning
output token breakdown, a finish reason — and lash parsed those fields and dropped them,
leaving hosts to label pre-run resolution as "actual" in their audit records: an
unfalsifiable echo that cannot detect routing substitution or a dropped reasoning setting.
We decided evidence is a first-class, typed, optional contract: **`ExecutionEvidence`
carries only what the provider explicitly reported, is never populated from requested or
resolved intent, and absence always means unreported — an explicit zero is information**
(`Some(0) != None`; usage normalization that collapses missing breakdowns to zero is
accounting, not evidence).

Each LLM response carries its own evidence. The turn result additionally carries a
per-attempt ledger: every attempt — successful, failed, retried, or interrupted — becomes an
entry with its outcome, request identity, protocol position, and whatever evidence was
actually observed before the attempt ended. A successful response's evidence stays pure;
failed-attempt facts live in the ledger, because served-model drift, billed failures, and
abort states matter most precisely when the attempt did not complete. Ledger entries carry
an open label describing why the call ran; the label vocabulary is host- and
plugin-supplied, not a core enumeration — internal machinery that makes model calls labels
its own work through the same interface.

"What produced the final output" is a tagged provenance value, never an index inference:
final output can come from a model call, from a terminal value or tool result without any
assistant text (an RLM Final Value ends a turn this way), from host synthesis, or from
nothing (incomplete). Turn activity and trace records project the same evidence types
rather than defining parallel shapes, and the remote mirrors carry them behind a protocol
version bump. Providers implement extraction independently; a provider that cannot report a
field leaves it absent rather than guessing.
