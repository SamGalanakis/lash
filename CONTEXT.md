# Lash Context

## Interaction Glossary

- **Queued Work**: Durable, session-scoped runtime ingress. It may carry turn input, process wakes, host events, timers, or runtime maintenance command payloads.
- **Queued Turn**: Queued Work whose payload is user-visible `TurnInput`.
- **Early Injection**: User input submitted with Enter while a turn is active. The runtime accepts it at the earliest safe boundary in the active turn.
- **Next Full Turn**: User input submitted with Tab while a turn is active. The runtime delivers it only after the current turn commits.
- **Slash Command**: CLI host command. Slash commands are not queued as model work.
- **Document Overlay**: Non-history UI surface used by `/help`, `/controls`, and `/info`. It starts at the top, scrolls with PgUp/PgDn, and closes with Esc or Ctrl+C.
- **Runtime Process**: Durable Lash process handle owned by the runtime. Only runtime processes appear in the CLI process dock.

The CLI owns presentation state: editor contents, overlays, scroll position, process focus, and disposable draft metadata. The runtime and store own durable work and operational state.

## Operator UI

- `Ctrl+C` is reserved for cancel/dismiss/quit semantics: close suggestions or overlays, cancel an active turn, clear a non-empty draft, then quit only from an idle empty prompt.
- Copy uses `Ctrl+Shift+C` by default. `Ctrl+U` deletes draft text to the start of the line, `Ctrl+K` deletes to the end, and history/document scrolling uses PgUp/PgDn, mouse wheel, and scroll gestures.
- The status bar exposes execution mode beside model and variant, for example `lash · gpt-5.5 · standard · medium`; context usage is labeled as `ctx`.
- Queue previews sit directly above the input. Early-injected work is labeled `Will send in this turn`; next-turn work is labeled `Queued for next turn`.
- The `/resume` picker hides zero-turn sessions when any non-empty session exists. If only empty sessions exist it shows them with `No messages yet`; direct `/resume <id-or-name>` may still target any session.

## Autonomous CLI Testing

- Run `cargo test -p lash-cli --features test-provider --test cli_e2e` to exercise the real `lash` binary without live provider credentials.
- The PTY smoke test launches interactive mode with a deterministic `test` provider, types a prompt, waits for rendered output, exits with `/exit`, and validates the generated UI trace/snapshot.
- Run `scripts/lash-operator.py --provider test` for an agent-operated PTY session. It builds/launches the real `lash` binary with an isolated deterministic provider, then accepts commands such as `expect 15 Idle`, `type hello`, `key enter`, `screen`, and `lash-exit`.
- Run `scripts/lash-operator.py --provider real -- --model <provider/model>` to drive Lash against the user's configured provider/API credentials. Use this deliberately because it may spend real model tokens.
