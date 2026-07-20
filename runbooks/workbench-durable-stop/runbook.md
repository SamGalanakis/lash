# E2E Scenario: Workbench Durable Stop — Exact Turn Cancellation Across Restart

> **Read [../RULES.md](../RULES.md) first** — especially "The browser surface (example
> apps)": browser tooling, objective gate order, screenshots, real-token designation,
> and teardown ownership. This runbook only adds the scenario-specific parts.

**Purpose.** Prove the Agent Workbench Stop control uses Lash's exact-turn,
keyed-promise cancellation primitive end to end. Stop one live turn normally, then start
another, restart only the workbench web process while Restate owns the turn, and Stop it
from the reconstructed UI. In both cases the rendered state and HTTP receipt must agree
on an authoritative `Cancelled` terminal and its cancellation evidence.

**Why this matters.** A web-process-local token or tracked Restate invocation id cannot
survive this scenario. The workbench persists only the routing address, Restate owns the
running turn, and `TurnWorkDriver` resolves the reserved gate and terminal promises. A
successful restart case therefore proves the Stop path is not secretly process-local or
using the Restate Admin API.

**Real tokens.** Turns use OpenRouter. Their prose and duration are nondeterministic;
gate on the running affordance, cancel receipt, terminal state, and evidence—not text
quality. This runbook is authored for a deliberate token-spending browser run.

## Scenario-specific golden rules

1. **Stop only after the running gate.** The Stop button is visible and `/api/state`
   reports the exact active turn before it is pressed. A fast completion that wins first
   is a retry of that phase, not cancellation evidence.
2. **HTTP terminal is authoritative.** `POST /api/turn/cancel` must return a cancellation
   receipt whose terminal is committed as stopped/cancelled and whose `cancellation`
   contains a non-empty `request_id`, `origin: "user"`, and the workbench `reason`.
3. **UI and receipt agree.** The UI renders `turn stopped · request <id>` using the same
   request id returned in `terminal.cancellation`; the transcript/API converges on the
   interrupted terminal. Any disagreement is a contract violation → Abort/RCA.
4. **Restart only the web process.** Use `just agent-workbench-restart <port>`, which
   preserves the data directory and Restate container. Tearing down Restate invalidates
   the durability proof.
5. **Break-glass is not success.** Never use Restate Admin cancel/kill to pass a gate. If
   cleanup requires it after an Abort, record that separately; it must not be reported as
   a Lash `Cancelled` terminal.

## Working material

- Boot with a fresh durable directory:
  `AGENT_WORKBENCH_DATA_DIR=<fresh-tmp> AGENT_WORKBENCH_OPEN=0 just agent-workbench <port>`.
  Gate `GET /healthz` → 200. The entire Restate stack is port-isolated by default: the
  helper derives its endpoint, ingress, admin port, node port, and container name from
  `<port>`, so concurrent runs on distinct workbench ports do not need manual Restate
  overrides. Teardown:
  `just agent-workbench-down <port>`.
- Browser affordances: chat composer, **stop turn** button, running/idle pill, transcript.
- Backend truth: `GET /api/state`; `POST /api/turn`; `POST /api/turn/cancel`.
  `/api/state.active_turns` exposes routing addresses so reload can restore the Stop
  affordance. The cancel response contains `accepted` and `cancellations[]`, each with
  `address`, gate `outcome`, and authoritative `terminal`.
- Disk evidence: `<data-dir>/session-id` and `<data-dir>/active-turns.json` retain routing
  state across the web-process restart; `trace.jsonl` records
  `agent_workbench.turn.cancel_requested` with the same evidence.

## Phase 0 — Boot and pre-flight

Require `OPENROUTER_API_KEY`; a missing key is a harness gap → Abort. Boot the workbench,
gate `/healthz`, open the browser, and confirm `/api/state.settings.session_id` matches the
rendered session id. Screenshot `00-ready.png`.

## Phase 1 — Stop a live turn without restart

Submit a task likely to expose a long-running window (for example, ask for a researched
comparison that uses web search). Gate on the running pill and visible **stop turn**
button, then poll `/api/state` until `active_turns` contains exactly one address for the
rendered session.

Press **stop turn** while capturing the `POST /api/turn/cancel` response. Gates:

1. response `accepted` is true and has one cancellation for the active address;
2. its gate outcome is `requested` or `already_requested`;
3. its terminal is committed, encodes `TurnStop::Cancelled`, and carries evidence per
   golden rule 2;
4. the UI renders `turn stopped · request <id>` with the same id, returns to idle, and
   hides Stop;
5. `GET /api/state` no longer lists that address and its messages agree with the rendered
   interrupted terminal.

Screenshot `01-cancelled.png`; save the cancel response as `01-cancel-receipt.json`.

## Phase 2 — Restart the web process mid-turn, then Stop

Submit another long-running turn. Gate on Stop plus one `/api/state.active_turns` entry
and record its session/turn ids. Run `just agent-workbench-restart <port>` without
touching Restate. Poll `/healthz` until the replacement process is ready, reload the page,
and gate all of the following before pressing Stop:

- the rendered session id is unchanged;
- `/api/state.active_turns` contains the exact pre-restart address;
- the running pill reads restored/running and **stop turn** is visible;
- `<data-dir>/active-turns.json` contains the same address.

Screenshot `02-restored-running.png`. Press Stop and repeat every receipt/UI/API gate from
Phase 1. Additionally require the terminal evidence request id to be new and the trace to
show the recovered request against the pre-restart turn id. Screenshot
`03-restored-cancelled.png`; save `03-cancel-receipt.json`.

## Phase 3 — Teardown and score

Run `just agent-workbench-down <port>` and confirm both the workbench process and its
Restate container are gone.

| Item | Objective gate | Verdict | Evidence |
|------|----------------|---------|----------|
| Boot | `/healthz` 200; rendered/API session ids agree | | `00-ready.png` |
| Normal Stop | committed Cancelled terminal + evidence | | `01-cancelled.png`, `01-cancel-receipt.json` |
| Routing persistence | same session/turn address before and after restart | | `02-restored-running.png`, state files/API |
| Restored Stop affordance | running pill + Stop restored from `/api/state.active_turns` | | `02-restored-running.png` |
| Post-restart Stop | committed Cancelled terminal + evidence for original address | | `03-restored-cancelled.png`, `03-cancel-receipt.json` |
| UI/API agreement | rendered request ids equal terminal evidence ids; active addresses clear | | screenshots + receipts + `/api/state` |
| No break-glass substitution | no Admin cancel/kill used as a passing action | | command log |

**Aggregate:** did exact cooperative cancellation produce authoritative evidence both
normally and after reconstructing the entire web process, with UI, API, disk routing
state, and trace in agreement?

---

_Stop triggers and the Abort/RCA + reporting protocol are in [../RULES.md](../RULES.md)._
