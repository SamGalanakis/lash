# Lash Context

## Interaction Glossary

- **Trigger Source**: Host-owned source instance of Trigger Occurrences, such as a UI control, webhook, schedule, or host-authorized tool catalog.
- **Trigger Source Key**: Stable routing identity for one configured Trigger Source.
- **Trigger Event Type**: Declared event kind that may occur on a Trigger Source, with a named payload shape. Examples include a button press, schedule tick, or webhook delivery.
- **Trigger Occurrence**: Session-agnostic world occurrence recorded for a Trigger Source and routed to interested runtime subscribers. It is not queued session work, and it is distinct from a Process Signal addressed to one process.
- **Queued Work**: Durable, session-scoped runtime ingress for work that must enter one session at a turn boundary. It may carry turn input, process wakes, or runtime maintenance command payloads; it does not carry Trigger Occurrences or timers.
- **Queued Turn**: Queued Work whose payload is user-visible `TurnInput`.
- **Agent Frame**: Durable context boundary inside a session. Opening a new Agent Frame continues the same session from new initial context while prior frame content remains durable and inspectable.
- **Agent Frame Reason**: Open label describing why an Agent Frame was opened. Core defines common labels but the label set is not exhaustive.
- **Compaction**: Deliberate transition to a new Agent Frame seeded from an assistant summary message of earlier context. It continues the same assignment with reduced future model context, without deleting or rewriting prior frame content.
- **Prompt View**: Ephemeral projection of session content prepared for one model turn. Prompt View shaping does not mutate durable session history.
- **Session Observation**: Host-facing view of a session at a point in time, paired with a Session Cursor. Live Replay may extend an observation with recent in-flight activity.
- **Session Observation Event**: Observer-visible session activity that advances a Session Cursor. It may include preview activity before it is durable and committed activity that settles the session view; it is not scoped to a single turn.
- **Turn Activity**: App-facing live stream item for one active turn.
- **Turn Event**: Semantic payload carried by Turn Activity.
- **Session Event**: Runtime-internal turn/protocol machinery tolerated inside implementation boundaries. Avoid using Session Event as app-facing vocabulary.
- **Session Revision**: Durable point in the committed session graph as observed through a session read view.
- **Session Cursor**: Opaque position in a session observation stream. It includes a Session Revision and any live replay position needed for reconnecting observers without making live activity permanent history.
- **Live Replay**: Bounded replay of recent in-flight session activity for short reconnects. It is recovery semantics, not durable session history; when it is unavailable, observers recover from the latest durable read view.
- **Live Replay Gap**: Observer-visible signal that a requested Session Cursor can no longer be replayed from recent live activity. The observer reconciles from a fresh Session Observation and continues from its new cursor.
- **Early Injection**: User input submitted with Enter while a turn is active. The runtime accepts it at the earliest safe boundary in the active turn.
- **Next Full Turn**: User input submitted with Tab while a turn is active. The runtime delivers it only after the current turn commits.
- **Slash Command**: CLI host command. Slash commands are not queued as model work.
- **Document Overlay**: Non-history UI surface used by `/help`, `/controls`, and `/info`. It starts at the top, scrolls with PgUp/PgDn, and closes with Esc or Ctrl+C.
- **Runtime Process**: Globally addressable, durable unit of work owned by the runtime. Its lifecycle is independent of any session: ending or deleting a session never ends a process by itself; what happens to related processes is host policy. Only runtime processes appear in the CLI process dock.
- **Process Event**: Durable fact recorded in a Runtime Process's own event log, such as a wake, yield, wait, resume, or terminal event. Distinct from a Trigger Occurrence, which is broadcast through subscriptions, and from a Process Signal, which is an addressed input to one process.
- **Process Originator**: Provenance recording where a process came from — a session or the host. Pure metadata; carries no behavior. Children started by a process inherit its originator and wake target (the chain, not the execution machinery), and a process may always await or cancel handles it created itself.
- **Execution Environment**: The captured description of tools, plugins, and policy that work executes against. Sessions run in one; a process captures the required subset of its creator's at creation time as immutable references, and is self-contained thereafter. It is a description, never live state; it cannot drift after capture, and process arguments are the only state handover.
- **Tool Catalog**: Effective set of host-provided tools available to model/runtime work, including their model-facing names, schemas, availability, and prompt-visible documentation. Avoid: Tool Surface.
- **Lashlang Tool Binding**: Authored mapping that makes a host tool callable as a Lashlang module operation in RLM execution. Avoid: Tool Lashlang Tool Binding, Lashlang Tool Binding.
- **Lashlang Host Environment**: Link-time description of host capabilities visible to Lashlang, including module operations, trigger sources, abilities, and language features. Avoid: Lashlang Surface, Host Surface.
- **Host Resource**: Family of host-provided Lashlang capabilities that expose typed operations to authored Lashlang code.
- **Resource Operation**: Effectful Lashlang call against a Host Resource.
- **Host Descriptor**: Pure, typed Lashlang value describing host-owned configuration or identity for later host interpretation. Avoid: Host Value, Opaque Host Descriptor.
- **Trigger Source Descriptor**: Host Descriptor for a Trigger Source policy or configuration.
- **Host Requirements**: Content-addressed description of the Lashlang Host Environment capabilities a linked Lashlang artifact requires. Avoid: Surface Requirements, Required Surface.
- **Wake Target**: Optional session that receives a process's wakes as Queued Work. A process without one still records wake events; they are observable but deliver nowhere.
- **Process Handle Grant**: Per-session visibility of a process. Grants are additive and revocable; they never affect whether the process runs.
- **Process Signal**: Named, typed message delivered to one specific Runtime Process. A process declares its signals and senders are validated against the declaration. Distinct from a Trigger Occurrence, which is a broadcast world occurrence routed by subscriptions.
- **Runtime Control**: Control-flow or durable effect machinery that changes execution trajectory or routes replayable effects. Avoid using Control for ordinary facade handles or tool-context clients.
- **Replay Key**: Stable identity for one logical runtime operation when execution is re-driven. Re-executing the same operation with the same Replay Key is replay, not a new operation.
- **Idempotency Key**: Stable identity for a submitted request or occurrence at an ingress boundary. Re-submitting matching content with the same Idempotency Key collapses to the prior accepted request or occurrence.

The CLI owns presentation state: editor contents, overlays, scroll position, process focus, and disposable draft metadata. The runtime and store own durable work and operational state.

The `lash` facade owns app-facing vocabulary. Lower-level crates may expose integration plumbing for sibling crates, but facade modules should publish canonical Lash terms rather than leaking implementation or legacy names.

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
