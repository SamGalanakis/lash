# E2E Scenario: Cancel Semantics — The Ctrl+C Ladder

> **Read [../RULES.md](../RULES.md) first** — operator surface, poll-don't-sleep, stop
> triggers, and reporting/RCA conventions. This runbook only adds the scenario-specific
> parts.

**Purpose.** Prove `Ctrl+C` walks its documented ladder, one rung per press, in order:
**close a suggestion/overlay → cancel the active turn → clear a non-empty draft → quit only
from an idle empty prompt.** Each rung is gated objectively, and a single escalating
sequence of four presses peels all four layers so the *ordering* — not just each behavior in
isolation — is what's under test.

**Why this matters.** `Ctrl+C` is overloaded on purpose (CONTEXT.md → "Operator UI":
"reserved for cancel/dismiss/quit semantics"). If the rungs fired out of order — quitting
while a turn ran, or cancelling a turn when the user only meant to close a popup — the
operator would destroy work on a keystroke people press reflexively. Copy is `Ctrl+Shift+C`
precisely so `Ctrl+C` can own this ladder.

## Scenario-specific golden rules

1. **One rung per press, in order.** With a suggestion popup open, an active turn, and a
   non-empty draft all present, the four presses must fire suggestion-dismiss → turn-cancel
   → draft-clear → quit — never skip a rung, never quit early.
2. **Cancel must reach a terminal state.** Rung 2 is not "the turn looked interrupted" — it
   is the turn committing to `Manually interrupted.` and the footer returning to `Idle`.
3. **Quit only from idle + empty.** The final press exits the process (clean status 0) only
   because the prompt is idle and empty; any earlier press that exits is a violation.

## Working material

- Scenario `standard-gated-escape`, holding prompt `gated initial prompt` (an unbounded
  active turn, so rung 2 can't win a race by accident). Typing `/` opens the slash-command
  suggestion popup for rung 1; that `/` draft is the non-empty draft rung 3 clears.

## Phase 0 — Pre-flight

Per [../RULES.md](../RULES.md). Launch `scripts/lash-operator.py --provider test --scenario
standard-gated-escape`, confirm the deterministic provider, gate the idle prompt
(`expect 20 Message · / for commands`).

## Phase 1 — Arm all four layers

Start the holding turn, confirm it is active, then open the suggestion popup on top of it:

```
type gated initial prompt
key enter
expect 20 gated initial prompt
expect 15 Thinking
type /
expect 8 Reset conversation
```

State now: active turn (`Thinking`), a non-empty draft (`/`), and the suggestion popup open
(the bordered list showing `/clear  Reset conversation`, `/controls`, …). All four ladder
layers are armed.

## Phase 2 — Walk the ladder, one `Ctrl+C` at a time

**Rung 1 — close the suggestion overlay.** The popup closes; the `/` draft and the active
turn survive.

```
key ctrl-c
wait 1
screen 20
```

Gate: the suggestion border is gone, the input still shows `❯ /`, and the footer still reads
`Thinking`. (Absence of the popup — inspect the `screen`; the border box and `Reset
conversation` must be gone.)

**Rung 2 — cancel the active turn.** The turn commits to its interrupted terminal state.

```
key ctrl-c
expect 20 Manually interrupted
expect 10 Idle
```

Gate: `Manually interrupted.` renders and the footer returns to `Idle`. A turn that keeps
running, or errors instead of interrupting, → Abort/RCA (turn cancel). The `/` draft still
survives.

**Rung 3 — clear the non-empty draft.** The `/` draft is discarded; the idle placeholder
returns.

```
clear
key ctrl-c
wait 1
expect 5 Message · / for commands
```

Gate: the input returns to the `❯ Message · / for commands` placeholder (empty draft). The
process is **still running** — confirm with `status` (`None`).

**Rung 4 — quit from idle + empty.** Only now does `Ctrl+C` exit.

```
key ctrl-c
wait 1
status
```

Gate: the child has exited — `status` reports `STATUS 0`. (The `wait` only lets the process
reap so `status` doesn't poll a still-shutting-down child; the exit — not the sleep — is the
gate. A *clean* exit here is the expected outcome of this rung, not an Abort trigger.)

## Phase 3 — Score

| Item | Objective gate | Verdict | Notes |
|------|----------------|---------|-------|
| Rung 1: suggestion closes | popup + `Reset conversation` gone; `/` draft + `Thinking` survive |  |  |
| Rung 2: active turn cancels | `Manually interrupted.` + footer `Idle` |  |  |
| Rung 3: draft clears | `❯ Message · / for commands` returns; process still alive |  |  |
| Rung 4: quit from idle empty | child exits `0` |  |  |
| Order held | exactly four presses, one rung each, none out of order |  |  |

**Aggregate:** did four presses peel four layers in the documented order, with the turn
reaching a real interrupted terminal state and the quit firing only from idle + empty.

---

_Stop triggers and the Abort/RCA + reporting protocol are in [../RULES.md](../RULES.md)._
