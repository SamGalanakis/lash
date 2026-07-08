# Bounded journals are an effect-controller obligation

A Runtime Process may run arbitrarily many effects (authored loops, large batch drivers) and
for arbitrary duration. In the durable tier that currently means one Restate invocation per
process whose journal grows with every effect — the classic unbounded-history cliff other
durable-execution systems cap outright (Temporal kills at ~51k history events; Inngest caps at
1000 steps) and solve with an author-visible reset (continue-as-new, new function runs). We
decided the reset is not the author's or the host's problem: **the `RuntimeEffectController`
seam guarantees that executing a process never requires an unbounded single-invocation
journal**, and each controller meets the guarantee with its backend's native mechanism. The
inline controller already satisfies it (effects re-drive against lash persistence; there is no
journal replay). `lash-restate` will segment a long process across chained invocations keyed
by (process id, segment), carrying an effect-result checkpoint across the boundary. A future
engine with native continue-as-new maps to that directly.

Above the seam nothing changes: process identity, leases, durable waits, provenance, replay
keys, and observation are segment-invariant, and no authoring construct or host projection
ever sees a segment. We rejected author-visible chaining (a `continue`-as-successor terminal
that hosts stitch into lineages) because it exports a backend limitation into every authoring
surface, host projection, and UI forever — and rejected a substrate-level incarnation
primitive as duplicating what each engine already does natively. Consequence: the lash-restate
segmentation is real, phased work — hosts must not ship unboundedly-looping processes on the
Restate tier before it lands — and segment handover (including handover while parked on a
Durable Wait) needs Deterministic Simulation and fault-matrix coverage.
