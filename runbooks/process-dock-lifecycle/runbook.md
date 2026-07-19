# E2E Scenario: Process Dock Lifecycle — Visibility, Cancel, and the Deletion Invariant

> **Read [../RULES.md](../RULES.md) first** — operator surface, poll-don't-sleep, stop
> triggers, and reporting/RCA conventions. This runbook only adds the scenario-specific
> parts.

**Purpose.** Prove the CLI process dock's contract: a **Runtime Process** shows in the dock
(header `Background`), the focused process can be cancelled (`Delete`), and — the invariant
— **ending or deleting a session never ends a process by itself**. Only runtime processes
appear in the dock; a subagent spawn does not.

**Why this matters.** CONTEXT.md → "Runtime Process": "Its lifecycle is independent of any
session: ending or deleting a session never ends a process by itself … Only runtime
processes appear in the CLI process dock." That independence is the whole point of a durable
process; if session teardown silently killed it, the durability guarantee would be a lie.

## Pre-flight harness gap (read before scoring)

**A Runtime Process visible in the dock is not reachable under `--provider test`.** Verified
by dry-run against every deterministic scenario:

- There is **no `/process` slash command** — the builtin set is `/clear /compact /controls
  /fork /tree /version /info /model /variant /mode /provider /logout /retry /resume /skills
  /help /exit` (`crates/lash-cli/src/command.rs`). The dock is a render surface fed by
  `session.processes().list()`, not a command.
- The RLM `rlm-subagent-smoke` scenario **does** spawn a subagent, but it renders as an
  **inline tool activity** — `◆ spawn subagent · …` with footer `Running tool ·
  spawn_agent` — and the `Background` dock header never appears. A subagent is not a dock
  Runtime Process (consistent with the contract: "Only runtime processes appear").
- The deterministic test provider only returns canned responses, so no scenario leaves a
  durable Runtime Process running; the dock stays empty (`app.processes.is_empty()` →
  `render/sections/docks.rs` draws nothing). A **Process Engine** that produces dock
  processes is contributed by an installed plugin, not wired by the test provider.

Per [../RULES.md](../RULES.md)'s "missing capability → note as harness gap": this runbook
runs the **drivable negative-space gates** below under `--provider test`, and specifies the
**full positive procedure** for `--provider real` with a process-engine plugin. Do not
fabricate a dock process that the test provider cannot produce.

## Scenario-specific golden rules

1. **Subagent ≠ dock process.** A spawned subagent is an inline tool activity; it must
   **not** raise the `Background` dock. Confirming that separation is a real gate, not a
   consolation prize.
2. **The deletion invariant is the crown jewel.** In the positive (real-provider) run, the
   process must **survive** its originating session being ended/deleted. A process that dies
   with its session is a hard fail.
3. **Don't invent a process.** If the dock is empty, gate the emptiness honestly; don't
   read a tool activity as a dock entry.

## Phase 0 — Pre-flight

Per [../RULES.md](../RULES.md). For the negative-space gates, launch
`scripts/lash-operator.py --provider test --scenario rlm-subagent-smoke -- -em rlm` and
confirm the deterministic provider and idle prompt.

## Phase 1 — Drivable gates under `--provider test` (negative space)

**Subagent renders as a tool activity, not a dock process.**

```
type Does your subagent tool work
key enter
expect 15 Running tool
expect 45 subagent-ok
screen 40
```

Gate: the spawn shows as `◆ spawn subagent · …` and the footer passes through `Running tool
· spawn_agent`; the settled value is `■ subagent-ok`. The `Background` dock header is
**absent** the entire time (inspect the `screen`). A `Background` dock appearing here would
itself be a contract surprise → investigate/report. Then `lash-exit 10`.

**Empty dock has nothing to focus.** Relaunch `--scenario standard-echo`. With an idle empty
prompt and no processes, the dock-focus keys have no target:

```
expect 20 Message · / for commands
key tab
wait 1
screen 18
```

Gate: no `Background` dock, and `Tab` opens no process overview (the dock-focus binding —
CONTEXT/`docs/cli.html`: "With an empty prompt, cycle focus through the dock of background
processes" — falls through when the dock is empty; note that `Shift+Tab` here falls through
to the plan-mode toggle, further evidence there is no dock to cycle). The process **cancel**
(`Delete`) and **overview** (`Enter`) rungs cannot be exercised — no process exists to
focus. Record this as the harness gap. Then `lash-exit 10`.

## Phase 2 — Full positive procedure (`--provider real`, needs a process-engine plugin)

Not runnable under the deterministic provider; specified so a real-provider run (RLM mode
plus a plugin whose tool starts a **durable Runtime Process**, e.g. a long-running
background job that yields a process handle) can execute it verbatim. Launch
`scripts/lash-operator.py --provider real -- --model <provider/model> -em rlm` (spends
tokens — deliberate).

1. **Start a process.** Drive a turn whose tool starts a durable background process. Gate:
   the `Background` dock renders a row `◆ running · <producer> · <label> · <elapsed>`
   (`render/sections/docks.rs`). Objective cross-check: `LASH_HOME/sessions/*.processes.db`
   records the process; a `list_process_handles` activity, if driven, reports it `running`.
2. **Focus and inspect.** With an empty prompt, `Tab` to focus the dock row (it gains the
   `SELECTED` badge / `▶` glyph); `Enter` opens its overview. Gate: the focused row and the
   overview name the same process.
3. **Cancel path.** `Delete` on the focused process. Gate: a `Process \`<label>\`
   cancellation requested: …` message, and the dock row transitions `running → cancelled`
   (or drops after its transient window). Objective: the process store shows a terminal
   `cancelled` state.
4. **Deletion invariant (crown jewel).** Start a **fresh** process, then delete/end its
   **session** (`/clear` opens a new session, retiring the current one; or delete the
   session db out of band). Gate: the process **remains** — still `running` in the process
   store and still listed by `list_process_handles` for a session it is granted to. A
   process that ends because its session ended → **hard fail** (contract violation) →
   Abort/RCA.

## Phase 3 — Score

| Item | Objective gate | Verdict | Notes |
|------|----------------|---------|-------|
| Subagent is a tool activity, not a dock process | `◆ spawn subagent` + `Running tool`, no `Background` dock |  |  |
| Empty dock has no focus target | `Tab` opens no overview; no `Background` header |  |  |
| Dock process visible *(real only)* | `Background` row `◆ running · … `; process store row |  |  |
| Cancel path *(real only)* | `running → cancelled`; terminal state in store |  |  |
| Deletion invariant *(real only)* | process survives its session being ended |  |  |
| Harness gap recorded | dock-process rungs un-drivable under `--provider test` |  |  |

**Aggregate (under `--provider test`):** does the subagent stay a tool activity while the
`Background` dock stays absent, and is the harness gap for dock-process lifecycle recorded
honestly rather than faked. **Full aggregate (real):** does a Runtime Process show, cancel,
and survive session deletion.

---

_Stop triggers and the Abort/RCA + reporting protocol are in [../RULES.md](../RULES.md)._
