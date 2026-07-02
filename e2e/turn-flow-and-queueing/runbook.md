# E2E Scenario: Turn Flow — Normal Turn, Early Injection vs Next Full Turn

> **Read [../RULES.md](../RULES.md) first** — operator surface, poll-don't-sleep, stop
> triggers, and reporting/RCA conventions. This runbook only adds the scenario-specific
> parts.

**Purpose.** Prove the primary operator loop end to end: a normal turn round-trips (draft
→ submit → runtime → settled assistant message), and mid-turn submission splits into the
two documented ingresses with the **exact** queue-preview labels as objective gates —
Enter is **Early Injection** (`◆ Will send in this turn`), Tab is **Next Full Turn**
(`◇ Queued for next turn`) — and a queued next-turn draft actually **dispatches** as its
own committed turn once the active turn commits.

**Why this matters.** The queue previews are the user's only signal of *when* their words
reach the model. If Enter and Tab drew the same label, or a label contradicted the ingress
the runtime recorded, the operator would be lying about turn boundaries. CONTEXT.md → "Queue
previews" and the glossary (Early Injection / Next Full Turn / Pending Turn Input) are the
contract.

## Scenario-specific golden rules

1. **The label is the gate, not the text.** `◆ Will send in this turn` must be drawn by
   lash's queue projector *because* you pressed Enter during an active turn — not because
   the words are in the draft. Confirm the label section, then the item under it.
2. **Enter ≠ Tab.** Early Injection (Enter) lands in `◆ Will send in this turn`; Next Full
   Turn (Tab) lands in `◇ Queued for next turn`. A draft that appears under the wrong
   header is a contract violation → Abort/RCA.
3. **Hold the turn to read the label.** Use `standard-gated-escape` so the active turn
   blocks indefinitely and both queue sections are observable at once; don't race a fast
   turn.

## Working material

- Normal turn: scenario `standard-echo`, prompt `hello from pty` → the provider echoes
  `test-provider echo: hello from pty` verbatim (a distinctive round-trip, not a coincidence).
- Label coexistence: scenario `standard-gated-escape`, holding prompt `gated initial prompt`.
- Delivery: scenario `standard-slow-echo`, holding prompt `slow initial prompt` (~2s window).

## Phase 0 — Pre-flight

Per [../RULES.md](../RULES.md). Launch `scripts/lash-operator.py --provider test`
(`standard-echo`), confirm the deterministic provider loaded, and gate the idle prompt:

```
expect 20 Message · / for commands
```

Optionally add `--trace trace.json` so Phase 2 can cross-check the ingress ops.

## Phase 1 — Normal turn round-trip (`standard-echo`)

```
type hello from pty
expect 5 hello from pty
key enter
expect 25 test-provider echo: hello from pty
```

Gates: the draft echoes as a user row `● hello from pty`; the footer walks
`Working → Thinking → Responding → Idle`; the settled assistant row is
`■ test-provider echo: hello from pty`. A missing or altered echo → Abort/RCA (turn submit
/ runtime / render). Then `lash-exit 10`.

## Phase 2 — Early Injection vs Next Full Turn labels (`standard-gated-escape`)

Relaunch with `--scenario standard-gated-escape`. Start the holding turn and confirm it is
active (blocked in `Thinking`):

```
expect 20 Message · / for commands
type gated initial prompt
key enter
expect 20 gated initial prompt
expect 15 Thinking
```

**Early Injection (Enter, mid-turn):**

```
type inject this now
key enter
expect 10 ◆ Will send in this turn
expect 5 inject this now
```

**Next Full Turn (Tab, mid-turn):**

```
type hold for next turn
key tab
expect 10 ◇ Queued for next turn
expect 5 hold for next turn
```

Objective gate: both sections render above the input, `◆ Will send in this turn` carrying
`↳ inject this now` and `◇ Queued for next turn` carrying `↳ hold for next turn`. If Enter's
text lands under `◇` (or Tab's under `◆`), or a label never renders, that is the contract
violation → Abort/RCA. With `--trace`, confirm the trace recorded one
`queue_current_turn_input` for `inject this now` and one `queue_turn` for `hold for next
turn` (Early Injection must not also appear as `queue_turn`). Then `kill` (the gate holds
the turn indefinitely by design).

## Phase 3 — Next-Full-Turn delivery (`standard-slow-echo`)

Relaunch with `--scenario standard-slow-echo`. Queue a next-turn draft during the bounded
active window, then let the first turn commit and watch the queued draft dispatch on its own:

```
expect 20 Message · / for commands
type slow initial prompt
key enter
expect 10 slow initial prompt
type hold for next turn
key tab
expect 8 ◇ Queued for next turn
expect 20 test-provider echo: slow initial prompt
expect 20 hold for next turn
```

Gate: after the first turn settles (`■ test-provider echo: slow initial prompt`), the
queued draft becomes its own committed user turn — a `● hold for next turn` row below the
turn separator, followed by a fresh turn that runs to `Idle`. (Under rolling history the
second turn's request still contains `slow initial prompt`, so the deterministic echo is
again `test-provider echo: slow initial prompt`; the delivery gate is the new `● hold for
next turn` turn, not the echo wording.) The queued draft never firing → Abort/RCA
(queued-turn dispatch). Then `lash-exit 10`.

## Phase 4 — Score

| Item | Objective gate | Verdict | Notes |
|------|----------------|---------|-------|
| Normal turn round-trips | `■ test-provider echo: hello from pty` rendered |  |  |
| Early Injection label | `◆ Will send in this turn` + `↳ inject this now` |  |  |
| Next Full Turn label | `◇ Queued for next turn` + `↳ hold for next turn` |  |  |
| Ingress recorded correctly | trace: 1× `queue_current_turn_input`, 1× `queue_turn` |  |  |
| Next-turn draft delivers | new `● hold for next turn` committed turn after commit |  |  |

**Aggregate:** does a turn round-trip, do Enter and Tab render the two **distinct** queue
labels matching their ingress, and does a queued next-turn draft dispatch on its own.

---

_Stop triggers and the Abort/RCA + reporting protocol are in [../RULES.md](../RULES.md)._
