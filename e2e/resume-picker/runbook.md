# E2E Scenario: Resume Picker — Hiding Rules and the Direct-Target Escape Hatch

> **Read [../RULES.md](../RULES.md) first** — operator surface, poll-don't-sleep, stop
> triggers, and reporting/RCA conventions. This runbook only adds the scenario-specific
> parts.

**Purpose.** Prove `/resume`'s picker follows its documented visibility rules and that the
direct-target form escapes them. With a mix of empty and non-empty sessions on disk, the
picker **hides zero-turn sessions** (a `1/1` list showing only the session that has a
message); when **only** empty sessions exist it shows them all as `No messages yet`; and
`/resume <target>` reaches a specific session even one the picker would hide.

**Why this matters.** The picker's job is to surface the sessions worth resuming and stay
out of the way of throwaway empties — but never at the cost of making a real session
unreachable. CONTEXT.md → "The `/resume` picker …" is the contract: hide zero-turn sessions
when any non-empty session exists; show `No messages yet` when only empties exist; direct
`/resume <id-or-name>` may still target any session.

## Scenario-specific golden rules

1. **Hiding is conditional, not absolute.** Empty sessions are hidden **only when a
   non-empty session exists**. In an only-empty store they must all appear (`No messages
   yet`). Both halves are gates.
2. **The current session is never in the picker.** The picker lists other sessions; the one
   you are in is excluded.
3. **The escape hatch must reach a hidden session.** `/resume <target>` targeting a session
   the picker suppressed must switch to it — otherwise the hidden session is unreachable, a
   fail.

## Working material

- One `--lash-home DIR` fixed across the run so sessions persist in `DIR/sessions/*.db`.
  Seed with `/clear`, which persists the current session and opens a fresh one, printing
  `Started new session: <name>`. Space each `/clear` with `wait 2` — session filenames are
  `YYYYMMDD_HHMMSS.db`, so rolls within one second collide.
- Non-empty seed: submit `hello from pty` (its user row makes the session's `message_count
  > 0` and its picker preview `hello from pty`).

## Phase 0 — Pre-flight

Per [../RULES.md](../RULES.md). Launch `scripts/lash-operator.py --provider test --lash-home
$LH` against a **clean** `$LH`, confirm the deterministic provider, gate the idle prompt.

## Phase 1 — Seed a mixed store (one non-empty, one hidden empty)

```
type hello from pty
key enter
expect 25 test-provider echo: hello from pty
wait 2
type /clear
key enter
expect 10 Started new session
```

You are now in an **empty** session (call it B) with the non-empty session A persisted. Read
B's on-disk identity before rolling past it — open `/info` and record the `session db`
path's basename (`<B>.db`) from the **Paths** section:

```
type /info
key enter
expect 10 session db
screen 40
key esc
```

Then roll once more so B becomes a non-current, hidden empty:

```
wait 2
type /clear
key enter
expect 10 Started new session
```

State: A non-empty (non-current), B empty (non-current, hideable), C empty (current).

## Phase 2 — Hiding gate (mixed store)

```
clear
type /resume
key enter
expect 10 Resume Session
screen 22
key esc
expect 6 Message · / for commands
```

Gate: the picker header is `Resume Session (1/1)` and the single row is A's preview —
`just now  1 hello from pty …`. The empty session B is **absent**, and `No messages yet`
does **not** appear. A picker that lists B (or shows `No messages yet` here) is a hiding-rule
violation → Abort/RCA.

## Phase 3 — Direct-target escape hatch

Target the hidden empty session B directly by its `.db` filename (from Phase 1):

```
clear
type /resume <B>.db
key enter
expect 10 Resumed: <B>.db
```

Gate: `Resumed: <B>.db` — the picker suppressed B, but the direct form reached it.

**Rigor (contract check on the identifier surface).** CONTEXT.md's glossary and
`docs/cli.html` present `/resume <id-or-name>` as accepting the session **id or
human-readable name** too, not only the `.db` filename. As a rigor case, also drive
`/resume <B-session-name>` (the `name` shown by `/info`). If it returns `Could not resolve
session ...`, that is a divergence between the documented identifier surface and the CLI —
**report it as a finding** (RCA it to the resume-identifier resolution path), and do **not**
edit the doc or code to hide it. Filename resolution passing is the escape-hatch gate;
name/id resolution is the rigor case.

## Phase 4 — Only-empty store

Relaunch against a **fresh** clean `--lash-home`, roll two empty sessions, and open the
picker:

```
expect 20 Message · / for commands
type /clear
key enter
expect 10 Started new session
wait 2
type /clear
key enter
expect 10 Started new session
wait 1
clear
type /resume
key enter
expect 10 Resume Session
screen 22
```

Gate: the picker lists the empty sessions with `No messages yet` on every row (title `Resume
Session (n/n)`). Nothing is hidden here — with no non-empty session, the empties are the only
thing to resume. Then `kill`.

## Phase 5 — Score

| Item | Objective gate | Verdict | Notes |
|------|----------------|---------|-------|
| Hides empty when non-empty exists | `Resume Session (1/1)`, only `hello from pty`, no `No messages yet` |  |  |
| Current session excluded | current session absent from the list |  |  |
| Direct target reaches hidden session | `Resumed: <B>.db` |  |  |
| Only-empty shows all | every row `No messages yet` |  |  |
| Identifier surface (rigor) | name/id resolves, or reported as a doc/CLI divergence |  |  |

**Aggregate:** does the picker hide zero-turn sessions only when a non-empty exists, show
`No messages yet` when only empties exist, exclude the current session, and does the direct
form reach a session the picker would hide.

---

_Stop triggers and the Abort/RCA + reporting protocol are in [../RULES.md](../RULES.md)._
