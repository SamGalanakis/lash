# Hosts register immutable deployments

Lash's durability story on a journaling host rests on an obligation the host
must meet and that nothing, until now, wrote down: **a deployment's code is
immutable for as long as any invocation may replay against it.**

Three independent properties assume it, and each degrades silently if it does
not hold.

## What assumes it

**Whole-envelope hashing.** A journaled effect records the hash of its full
envelope, and replay rejects a mismatch (`validate_recorded_effect_hash`,
`crates/lash-restate/src/lib.rs`). This detects lash's own nondeterminism
bugs, which is why the hash covers content rather than an identity tuple. It
cannot distinguish "the runtime computed a different envelope" from "the code
that computed it changed underneath the invocation". Mutating a deployment
converts a correctness detector into a false alarm, and worse, trains readers
to dismiss it.

**The rejection of version markers.** Temporal-style `patched()` gates are
deliberately absent. They are unnecessary here because an in-flight invocation
completes against the code it started on, and they are unusable here anyway:
journal correlation is positional, so a marker read inserted into an existing
program shifts every later entry. That argument only holds while the deployment
is immutable.

**Content-addressed module pinning.** A lashlang process resolves its pinned
`module_ref` for its whole life (`crates/lash-lashlang-runtime/src/process.rs`).
This gives lash Temporal's build-id behaviour for free — old runs finish on old
artifacts. Pinning the module while mutating the deployment around it pins the
wrong half.

## The obligation

A host must register a new deployment rather than replace the code behind an
existing one. Evolution is pin-and-drain: publish the new deployment, let
in-flight invocations finish against the old one, retire it when drained.

Restate's own model already works this way — deployments are immutable by
design and `patch_deployment_id` exists precisely for the repair case. The
obligation is therefore not a lash-specific burden so much as a property hosts
must not defeat, for example by redeploying to a fixed identifier in a
container platform that treats the image tag as mutable.

## What is not covered

Restate object state is not part of a replayed journal and does survive a
deployment upgrade. The versioned-metadata miss is the compatibility boundary
there, and it is handled explicitly
(`load_durable_wait_index_metadata`, `crates/lash-restate/src/lib.rs`). That
mechanism is unaffected by this decision; state migration and code immutability
are separate concerns.

Segment handover crosses invocations, so a successor segment is routed to the
latest deployment. Immutability of any single deployment does not make handover
artifacts self-describing, and it is not a substitute for versioning them.

## Consequences

Hosts that mutate deployments will see replay hash mismatches that look like
lash nondeterminism bugs but are not, and will lose the guarantee that a
started invocation completes against consistent code. Neither failure names its
cause today.

Lash does not detect the violation and should not pretend to: a runtime cannot
distinguish its own nondeterminism from code changing beneath it. The remedy is
documentation and host-side deployment discipline, not a runtime check.

Written after three independent reviews of separate subsystems each discovered
this assumption and none found it stated.
