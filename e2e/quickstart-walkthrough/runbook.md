# E2E Scenario: Quickstart Walkthrough — Docs-Accuracy Gate

> **Read [../RULES.md](../RULES.md) first** — operator surface, poll-don't-sleep, stop
> triggers, and reporting/RCA conventions. This runbook only adds the scenario-specific
> parts.

**Purpose.** Turn every claim `docs/quickstart.html` makes about **visible behavior** into a
gate against the real artifact, so the page can't drift away from what the code does. A
mismatch is a **docs bug — report it, do not fix** (do not edit the page or the code to make
the walkthrough pass).

## Pre-flight scope note (read before scoring)

`docs/quickstart.html` is the **embedding** quickstart — its cover source is `crates/lash` ·
`crates/lash-provider-openai`, and its "steps" are Rust API calls (add crates, build a
`LashCore`, open a session, run one turn, print the settled prose), **not** `lash`-binary
keystrokes. There is no CLI walkthrough on the page to type into the operator. So this
runbook gates the page in the two forms its claims actually take:

- **Structural / API claims** (crate names, version pins, the code snippet, the API
  surface) → machine-verified evidence (`scripts/lint_docs.py` + a Cargo.toml cross-check).
- **The one runtime visible-behavior claim** ("run one turn; read the settled assistant
  message", `println!("{prose}")`) → reproduced against the real `lash` binary through the
  operator: one turn in → one settled assistant message out.

**Report the framing itself as a finding:** a literal keystroke-by-keystroke CLI walkthrough
of `quickstart.html` is not possible because the page targets the library surface, not the
operator surface. That is a scope clarification to record, not a page defect.

## Scenario-specific golden rules

1. **Every visible claim is a gate.** No claim gets a pass on trust — each maps to
   `lint_docs`, a Cargo.toml value, or a rendered operator outcome.
2. **A mismatch is reported, never fixed.** If a version pin, a crate name, the snippet, or
   the one-turn behavior diverges from the page, RCA it and stop — do not reconcile by
   editing docs or code.
3. **Don't over-claim the analogue.** The operator round-trip stands in for the compiled
   program's "one turn → one settled message" behavior; it is not a line-by-line execution
   of the Rust snippet. Say so.

## Phase 0 — Pre-flight

Per [../RULES.md](../RULES.md). Read `docs/quickstart.html` and enumerate its checkable
claims. Confirm the checkers exist: `scripts/lint_docs.py`, `Cargo.toml`, and the operator.

## Phase 1 — Structural / API claims (machine-verified, objective)

| # | Page claim | Gate | Expected |
|---|-----------|------|----------|
| 1 | Crate is published as `lash-runtime`; the Rust import stays `use lash::...` | `Cargo.toml` alias | `lash = { package = "lash-runtime", … }` |
| 2 | `lash-runtime = "=0.1.0-alpha.81"` and `lash-provider-openai = "=0.1.0-alpha.81"` | `Cargo.toml` workspace `version` | matches (`0.1.0-alpha.81`) |
| 3 | The "Minimal Example" code (`standard_builder → provider → model(from_token_limits) → effect_host(InlineEffectHost) → attachment_store(InMemoryAttachmentStore) → build`; `core.session(..).open()`; `session.turn(TurnInput::text(..)).run()`; `result.assistant_message()`) | `python3 scripts/lint_docs.py` (the page's `data-snippet="quickstart#hello-lash"` is diffed against `examples/docs-snippets/src/quickstart.rs`) | `docs lint: ok` |

Run `python3 scripts/lint_docs.py` — a clean `docs lint: ok` is the objective gate that the
snippet and page structure haven't drifted from the compiled snippet source. Cross-check
claims 1–2 against `Cargo.toml` directly. **Any divergence here is a docs bug → report/RCA,
don't fix.** (At time of authoring: all three pass — the version pins match the workspace,
so no drift bug to report.)

## Phase 2 — The one runtime visible-behavior claim (real binary, drivable)

The page's only user-visible **runtime** claim is: run **one** turn, then read the **settled
assistant message** (`let prose = result.assistant_message()…; println!("{prose}")`). Drive
its analogue against the real `lash` binary through the operator — one turn in, one settled
assistant message out:

```
scripts/lash-operator.py --provider test
```
```
expect 20 Message · / for commands
type hello from pty
key enter
expect 25 test-provider echo: hello from pty
lash-exit 10
```

Gate: a single submitted turn produces exactly one settled assistant message
(`■ test-provider echo: hello from pty`), matching the page's "run one turn → read the
settled prose" contract. (The deterministic provider stands in for the page's OpenRouter
model; the observable shape — one turn, one settled message — is the claim under test, not
the specific words.) If a single turn produced no settled message, or more than one, the
page's central visible-behavior claim is wrong → report/RCA (do not fix).

## Phase 3 — Score

| Item | Objective gate | Verdict | Notes |
|------|----------------|---------|-------|
| Crate-name / import claim | `Cargo.toml` `package = "lash-runtime"` |  |  |
| Version pins match workspace | `=0.1.0-alpha.81` == workspace `version` |  |  |
| Snippet / API surface un-drifted | `scripts/lint_docs.py` → `docs lint: ok` |  |  |
| One-turn settled-message behavior | one turn → one `■` settled message |  |  |
| Scope framing recorded | page is a library quickstart, not a CLI walkthrough |  |  |

**Aggregate:** do the page's structural claims match the code (crate name, version pins,
snippet), does the real binary honor its one runtime claim (one turn → one settled message),
and is the library-vs-CLI scope of the page reported rather than papered over. Any concrete
divergence is surfaced as a docs bug, never edited away.

---

_Stop triggers and the Abort/RCA + reporting protocol are in [../RULES.md](../RULES.md)._
