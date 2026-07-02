# E2E Runbook Rules

Read this before running any scenario in `e2e/`. Each runbook links here and does
**not** repeat these rules — it only adds its scenario-specific purpose, golden rules,
phases, and scorecard.

`e2e/` has **two layers**. Scripted deterministic harnesses
(`e2e/restate-postgres-workers/`, driven by `just restate-postgres-workers-e2e` and the
`scripts/*-e2e.sh` runners) are gate **evidence**: they boot real infrastructure and
assert exact outcomes, and they stay scripts. Runbooks are the **agent-judged semantic
layer** on top: you (the agent) drive the real `lash` binary through the PTY operator and
judge the result with your own reasoning, gating on what the CLI actually renders. Keep
the layers separate — a runbook never re-implements a scripted harness, and a scripted
harness never asks for judgement.

These are **agent-driven runbooks**, not scripts. Use judgement freely — but never skip a
scenario's verification gates or the Abort rule below.

## What you're testing

You are testing **lash's operator surface**, not the model and not your own operator
script. The scenario is only valid if the **CLI** — its input handling, runtime, and
render — produces the observed result, not the literal text you typed. When a runbook
asserts a rendered label (`◆ Will send in this turn`), the gate is that lash's queue
projector drew it in response to the ingress you drove, not that the string appears
because you typed it into the draft. When a scenario calls for isolation (a fresh session
that must not see another session's transcript), a leak **voids the run**.

The contract source for every operator behavior a runbook asserts is **CONTEXT.md →
"Operator UI"** (plus the Interaction Glossary: Early Injection, Next Full Turn, Runtime
Process, Document Overlay, Slash Command). Cross-check a claim against CONTEXT.md before
gating on it. **Do not edit CONTEXT.md** — a divergence between it and the CLI is a
finding to report, never to reconcile by editing the doc.

## The operator surface

Everything is driven through `scripts/lash-operator.py`, which launches the real `lash`
binary in a child PTY and reads line commands from stdin or `--script FILE`.

**Launch lines**

- `scripts/lash-operator.py --provider test` — build + launch with an **isolated
  deterministic** provider: a temp `LASH_HOME`, `config.json` pinned to
  `{"active_provider":"test"}`, model `test/cli-e2e-model`, scenario `standard-echo`.
- `--scenario <name>` selects the deterministic test-provider scenario. Available:
  `standard-echo` (echoes `test-provider echo: <prompt>`; a bare prompt echoes
  `interactive prompt`, the literal `hello from pty` echoes itself), `standard-slow-echo`
  (as echo, but the prompt `slow initial prompt` sleeps ~2s before replying — a bounded
  active-turn window), `standard-gated-escape` (the prompt `gated initial prompt` blocks
  the turn **indefinitely** until a `gated-initial-release` marker file appears in
  `LASH_HOME` — an unbounded active-turn window), and the RLM scenarios
  `rlm-subagent-smoke` / `rlm-workspace-smoke` / `rlm-nonzero-exit-smoke` (need `-- -em rlm`).
- `--no-build` launches the existing `target/debug/lash` instead of rebuilding.
- `--lash-home DIR` fixes `LASH_HOME` so sessions and artifacts **persist across runs**
  (sessions live in `DIR/sessions/*.db`). Omit it for a throwaway temp home.
- `--trace PATH` passes `--debug-ui-trace PATH` to lash: a replayable UI-trace JSON plus a
  final snapshot — the same on-disk artifact the PTY smoke test asserts on.
- `-- <lash args>` — everything after `--` is forwarded to `lash` (e.g. `-- -em rlm`).
- `scripts/lash-operator.py --provider real -- --model <provider/model>` drives the user's
  configured provider/credentials. **Deliberate only** — it spends real tokens.

**Commands** (one per line; `#` and blank lines are ignored)

- `type TEXT` — type literal text into the draft (no newline).
- `send ESCAPED` — write raw bytes; `\xNN`/`\t`/`\r` escapes are decoded (e.g. `send \x03`).
- `key NAME [COUNT]` — send a named key `COUNT` times. Known: `enter`/`return`, `tab`,
  `backtab`, `esc`/`escape`, `ctrl-c`, `ctrl-d`, `backspace`, `delete`, `up`, `down`,
  `left`, `right`, `alt-up`, `alt-down`, `pageup`, `pagedown`, `home`, `end`. An unknown
  key name is an operator error → Abort.
- `expect [SECS] TEXT` — poll the rendered screen until `TEXT` appears (substring, matched
  against raw or ANSI-stripped output). Default 10s. **This is a gate.** A timeout, or
  lash exiting before the text appears, raises and stops the run.
- `expect-re [SECS] REGEX` — as `expect`, but a multiline regex over the stripped screen.
- `wait SECS` — sleep. **Never a gate** (see below).
- `screen [LINES]` / `raw [LINES]` — print the last N stripped / raw lines for inspection.
- `clear` — drop the capture buffer (so the next `screen`/`expect` only sees new frames).
- `status` — print the child exit code (`None` while running).
- `lash-exit [SECS]` — send `/exit`, then wait up to SECS for a clean process exit.
- `kill` — SIGTERM/SIGKILL the child process group. `quit` — stop the driver.

**Pre-flight** (every runbook's Phase 0): launch, and confirm the launched binary carries
the deterministic provider. A startup failure `provider 'test' is not supported by this
CLI build` means the binary was compiled **without** `--features test-provider` — rebuild
(`cargo build -p lash-cli --features test-provider`) and make sure the launched
`target/debug/lash` is that fresh artifact, then Abort/RCA the run and note it as a
harness gap. Confirm the idle prompt renders (`expect 20 Message · / for commands`) before
driving anything. If a scenario needs a key, scenario, or surface the operator cannot
reach, that missing capability → Abort/RCA, noted as a harness gap (do not pretend around
it).

## Poll, don't sleep

Turns, tool calls, subagent spawns, and queued-turn dispatch are all async and render over
several frames. Gate on the **rendered outcome** with `expect <secs> <text>` — never `wait
N`. Submit a turn, then `expect <secs> <assistant-marker-text>`; the footer status walks
`Idle → Working → Thinking → Responding → Idle` (and `Running tool · <name>` for tool
calls), so gate on the terminal render, not a fixed sleep. `wait` is only for letting the
render settle before a `screen`, or spacing session-file writes so timestamp filenames
don't collide — **never** to decide that async work finished. An `expect` timeout at a gate
is a hard failure → Abort/RCA. `expect` also fails fast if lash exits first (`lash exited
with N before '<text>' appeared`) — that is itself an Abort trigger.

## Gate objectively before you judge

Prefer an objective signal over eyeballing. In order of authority:

1. **Rendered screen** — the `expect`/`screen` gate: footer status labels, the user `●` /
   assistant `■` markers, queue previews (`◆ Will send in this turn` / `◇ Queued for next
   turn`), overlay boxes (`Resume Session (n/m)`, the suggestion popup, `/help` / `/info`
   document overlays). This is what the operator surface actually shows a user.
2. **UI trace / snapshot** (`--trace PATH`) — the durable record of what was *submitted*,
   independent of what rendered: ops like `user_turn`, `queue_current_turn_input` (Early
   Injection), `queue_turn` (Next Full Turn). Use it to prove intent-vs-render (e.g. Enter
   mid-turn recorded exactly one `queue_current_turn_input` and zero `queue_turn`).
3. **On-disk state** — `LASH_HOME/sessions/*.db` (durable session evidence; message counts
   drive the resume picker), and `LASH_HOME/test-provider-requests.jsonl` (the visible user
   texts each provider call actually saw — objective proof of what reached the model, and
   how many times).

Run the structural gate **before** judging behavior. If the objective signal is missing
(the label never rendered, the trace op is absent), the failure is upstream of anything you
would judge — Abort/RCA, don't score the vibe.

## When to STOP (Abort triggers)

Stop immediately on **any** of:

- an operator command error (an `ERROR ...` line / non-zero driver exit);
- an `expect` / `expect-re` timeout at a gate;
- lash exiting unexpectedly before a gate (`lash exited with N before ...`);
- a **contract violation** — an isolation or invariant break: a supposedly-hidden empty
  session surfacing in the resume picker, a rendered queue label that contradicts the
  ingress you drove, a Runtime Process ending merely because its session was deleted, or a
  document-overlay that won't close on `Esc`/`Ctrl+C`;
- an assertion that contradicts the scenario's answer key.

Do not push through, do not paper over, do not attempt a fix as part of the run.

## How to REPORT

**On abort — RCA, then stop:**
1. **Stop.** Do not continue the scenario.
2. **Capture evidence** — the failing operator command and its `ERROR` tail; the last
   `screen`; the child `status`; the `LASH_HOME` path and any `--trace` artifact / session
   db involved; and the exact gate string that failed.
3. **RCA** — symptom → the CLI stage it broke at (input handling / suggestion+modal
   dismissal / turn submit / runtime execution / render / session persistence) → root
   cause → the evidence that proves it. Never stop at "the expect timed out."
4. **Report and stop.** This is a diagnosis, not a repair. A divergence between an observed
   behavior and CONTEXT.md or the docs is reported as a finding — **do not** edit the doc
   or the code to make the run pass.

**On success — score, don't vibe:** for each scored item, name the **specific rendered
string** (or trace op / on-disk fact) the gate matched — no credit for vibes. Mark the
objective gate (screen / trace / disk) separately from any judged behavior. Fill the
scenario's scorecard verbatim.
