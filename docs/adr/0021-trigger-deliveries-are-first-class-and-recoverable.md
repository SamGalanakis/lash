# Trigger deliveries are first-class and recoverable

A Trigger Delivery — the reserved (occurrence, subscription) pair that starts one process — was
persisted durably but invisible: `CausalRef::TriggerOccurrence` recorded only the occurrence
half of the delivery identity, `TriggerEmitReport` flattened deliveries to bare started process
ids, the `TriggerStore` trait exposed no reads over occurrences or deliveries, and a crash
between reserving a delivery and starting its process lost the delivery forever (replayed emits
skip already-reserved pairs). We decided the delivery is a first-class, observable, recoverable
substrate fact: process provenance carries `subscription_id` alongside `occurrence_id`, the
emit report returns per-delivery outcomes (started / already reserved / failed with reason),
the trigger store exposes occurrence and delivery reads, and recovery sweeps deliveries that
have no registered process and starts them idempotently (safe because delivery process ids are
deterministic and registration is idempotent by hash).

The alternative — hosts joining process → delivery at read time and running their own repair
sweeps — was rejected: every host would pay a per-process lookup to learn provenance lash
already knows, and a host repairing the substrate's own emit crash window is a layering
inversion. Host-agnostic by construction: occurrence, subscription, delivery, and process are
all lash-native concepts; product mappings built on top of `subscription_id` (e.g. a host's
release or run attribution) stay in the host.
