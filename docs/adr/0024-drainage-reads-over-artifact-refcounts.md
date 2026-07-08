# Drainage reads over artifact refcounts

A host that retires old definition artifacts or env blobs must know when nothing in-flight
still needs them — recovery of a process whose artifact is gone is a hard
`process_module_artifact_missing` failure, so retirement without evidence is data loss. We
added a registry aggregate: `live_reference_summary()` groups non-terminal processes by
(identity definition, env ref) with counts, computed on demand from process rows. A definition
or env ref absent from the summary is drained and safe to retire; the counts double as an
"in-flight per version" drainage signal for host UIs.

We rejected maintaining live reference counts inside the artifact/env stores: it couples two
deliberately separate store families, adds a write to every process lifecycle transition, and a
drift bug silently corrupts retirement decisions — whereas the aggregate is recomputed from
truth on every call. Client-side counting via paged list reads was rejected as a substrate
gap: every retiring host would re-implement the same scan, paying full row payloads to compute
a count.
