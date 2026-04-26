# Handoff: `pass_baton` for rlmpure execution mode

**Status:** Designed, approved, not implemented.
**Date:** 2026-04-26
**Author:** prior session (full design context preserved here).

This handoff is self-contained — a fresh agent should be able to read this once and execute.

---

## 1. Why we're doing this

DSPy's RLM (the published Recursive Language Model paper) has a "mandatory extract" fallback: when iteration cap is hit without `submit`, the system runs an `extract` Predict over the full trajectory to coerce a final answer. That's a *terminal* fallback applied after the context has already blown up — last-ditch salvage from a messy state.

Our predecessor session (search session id `20260424_224905`, look for "baton") arrived at a strictly better mechanic: instead of reactively extracting at the end, **the model itself proactively initiates a fresh-context handoff** mid-execution, with explicit curation of what the successor sees. The model knows what's worth keeping; the runtime doesn't. The metaphor:

> You are one runner in a relay. Your job is not to remember forever; your job is to leave a perfect baton.

The framing the previous session arrived at:

> The root agent should not be the mind. The root should be the process tree. The active mind should be whichever agent currently has the baton.

Lash is positioned as a *"context-budgeted operating system for model cognition"* rather than "one agent with helpers." Baton-passing is the execution semantics that delivers on that.

This handoff is for v1 of that mechanic, in rlmpure mode only.

## 2. Final design (settled after extensive iteration — do not relitigate)

### 2.1 Tool surface (lashlang-callable)

```
call pass_baton {
  task = "<instructions for the successor>",   // required, str
  seed = { x = candidates,                     // optional, record
           query = original_query }
}
```

That's it. **Single shot. No `task_name`, `capability`, `output`, or `fork_turns`.**

Why each was rejected:
- `task_name` — auto-generate from the chain index (`baton_<n>`) or short uuid. Don't ask the model.
- `capability` — successor inherits the parent's capability/model/mode. Switching mid-chain creates surprises about the user-facing contract.
- `output` — the *session's* termination/output schema is set once at session creation and is the user-facing contract. Letting any agent in the chain redefine it would mean a user who asked for `{x: int, y: str}` could get something else from a later baton. Schema is a session-level invariant, inherited verbatim.
- `fork_turns` — baton has fixed inheritance semantics by design (fresh trajectory + seed only). Exposing the knob would tempt callers to ask for `"all"`, defeating the entire point.

### 2.2 Inheritance shape — what the successor sees

| | Carried | Source |
|---|---|---|
| Conversation events | ❌ | empty |
| Trajectory entries | ❌ | empty |
| Prior `RlmGlobalsPatch` events | ❌ | empty |
| Initial globals | ✅ | one synthesized `RlmGlobalsPatch` from `seed` |
| First user message (`user_input_1`) | ✅ | the `task` string injected as `first_turn_input` |
| Output schema (`RlmTermination`) | ✅ | cloned from parent's policy/extras |
| Model / provider / `max_context_tokens` | ✅ | cloned from parent's policy |
| Agent tree edge | ✅ | `parent_session_id = parent` |

Everything in this table was deliberated. The dominant principle is: **successor is genuinely a fresh root-style rlmpure agent** that happens to be parented in the agent tree. There is no special projector rendering, no special "baton from previous agent" timeline section. The successor cannot tell whether it was created by a human typing into the CLI or by `pass_baton` from its predecessor.

The clean property this delivers: zero new infrastructure for the successor — it's a normal rlmpure session with one initial conversation event and one initial globals patch. Both mechanisms already exist (root agents are seeded the same way via `--rlm-vars-file` + a user prompt).

### 2.3 Parent session lifecycle

- Parent transitions to `Completed` status (no submit value).
- Parent's full event history (trajectory, conversation events, etc.) **stays on disk** for debugging via the agent tree.
- The user-facing session chain's "active leaf" rotates from parent to successor.
- No new `RlmTermination` variant is added. The parent looks like a session that completed without a final answer; the successor is the one that will produce the final answer for the chain.

### 2.4 Per-turn budget visibility

A new prompt contribution rendered every turn:

```
## Context budget
Used: 47,213 / 200,000 tokens (24%)
Hand off via `pass_baton` when this becomes inefficient. The next agent starts fresh.
```

Sourced from `state.last_prompt_usage.context_budget_tokens` (already tracked, see `lash/src/runtime/usage.rs:230-262` where `normalize_prompt_usage` is computed) and `policy.max_context_tokens` (already required, validated at startup in `lash/src/runtime/lifecycle.rs:22-25`).

If `last_prompt_usage` is `None` (first turn of a session, no LLM call yet), either omit the section or render `Used: estimating / Y tokens`.

### 2.5 Soft warning at scalar token threshold

**Threshold is a scalar token count, not a percentage.** Configurable per-session via `RlmpureModePluginConfig`:

```rust
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct RlmpureModePluginConfig {
    pub observe_projection: ToolResultProjectionPluginConfig,
    #[serde(default = "default_max_output_chars")]
    pub max_output_chars: usize,
    #[serde(default)]
    pub baton_soft_warn_tokens: Option<usize>, // None = disabled
}
```

When `last_prompt_usage.context_budget_tokens` first crosses `baton_soft_warn_tokens`, runtime injects an out-of-band reminder before the next turn:

> Context budget at 72,341 tokens (threshold 70,000). If progress so far is captured in named lashlang variables, prefer `pass_baton(task=..., seed={...})` over continuing in this trajectory.

Fires **once per session per crossing** (track via plugin state). Uses the existing-but-unused `TurnInputInjectionBridge` at `lash/src/session.rs:86-128`.

### 2.6 Out of scope for v1

Explicitly NOT in v1, do not implement:
- Forced-baton synthesis at hard cap (the "what if model ignores warnings" path). This is v2.
- Multi-threshold warnings (e.g., 70%+90%). Only the single configurable scalar threshold.
- Soft warnings or pass_baton in standard / rlm modes. Rlmpure-only.
- Special projector rendering of "baton handoff" — the successor's prompt looks like any other rlmpure prompt.
- A `lash baton inspect` CLI command or any new tooling beyond what the existing agent tree already gives us.

## 3. Implementation steps

### Step 1 — Add `pass_baton` tool definition

**File:** `lash-subagents/src/policy.rs`

Add `pass_baton_definition(examples: Vec<String>) -> ToolDefinition` modeled on `spawn_agent_definition` (lines 119-143) but stripped down:

```rust
fn pass_baton_definition(examples: Vec<String>) -> ToolDefinition {
    let description = "Hand off to a fresh successor session and end the current session immediately. Use when most of your trajectory has become irrelevant or when context budget is approaching the limit. The successor inherits ONLY: (1) the values you put in `seed` (bound as lashlang globals), (2) the `task` string (becomes its `user_input_1`). It does NOT inherit prior conversation, trajectory, or globals. The successor's output schema is the same as the current session's. The successor's `submit` becomes the user-facing answer.".into();
    ToolDefinition {
        name: "pass_baton".into(),
        description,
        params: vec![
            ToolParam::typed("task", "str"),
            ToolParam::optional("seed", "dict"),
        ],
        returns: "dict".into(),
        examples,
        availability: lash::ToolAvailabilityConfig::documented(),
        activation: lash::ToolActivation::Always,
        availability_override: None,
        input_schema_override: None, // or build one analogous to spawn_agent_input_schema
        output_schema_override: None,
        execution_mode: ToolExecutionMode::Parallel,
    }
}
```

Include it in `rlm_subagent_tool_definitions` (lines 57-89). Examples should use `call pass_baton { ... }` syntax (lashlang form), e.g.:

```
call pass_baton {
  task = "filter the candidates by relevance to the user's query and submit the top 3",
  seed = { candidates = candidates, query = user_input_1 }
}
```

Do **not** add it to `standard_subagent_tool_definitions`. Standard mode is out of scope.

### Step 2 — Dispatch in `SubagentToolsProvider`

**File:** `lash-subagents/src/lib.rs`

In `execute_with_context` (around lines 323-342, search for `"spawn_agent" =>`):

```rust
"pass_baton" => self.pass_baton(args, context).await,
```

Implement `pass_baton(args, context)`:

```rust
async fn pass_baton(
    &self,
    args: serde_json::Value,
    context: ToolExecutionContext,
) -> ToolResult {
    // 1. Parse args
    let task: String = /* args["task"] */;
    let seed: serde_json::Map<String, serde_json::Value> = /* args["seed"] or empty */;

    // 2. Resolve parent session policy
    let parent_policy = context.host.session_policy(&context.session_id).await?;

    // 3. Build successor SessionCreateRequest (see step 3 below for exact shape)

    // 4. Insert successor into agent tree, parented to current
    //    (reuse logic from local.rs:495-656 spawn_agent handler)

    // 5. Apply initial RlmGlobalsPatch event for `seed`
    //    (append a SessionEventRecord::Mode(ModeEvent::rlm(RlmModeEvent::RlmGlobalsPatch(...)))
    //     to the new session's session_graph BEFORE its first turn runs)

    // 6. Queue successor's first turn (see local.rs:576-579 for pattern)

    // 7. Mark current session as terminated; rotate chain leaf

    // 8. Return a ToolResult that signals to the rlm driver that the
    //    parent's turn loop should end (see Step 4 below for the sentinel)
}
```

### Step 3 — Build the successor `SessionCreateRequest`

Use `build_spawn_create_request` (lash-subagents/src/lib.rs:89-145) as the template, but with these specific values:

| Field | Value |
|---|---|
| `session_id` | fresh UUID |
| `parent_session_id` | `Some(parent_session_id)` |
| `start` | `SessionStartPoint::Empty` |
| `policy` | clone of parent's policy (do NOT route through capability resolver — there's no capability) |
| `mode_extras` | `ModeExtras::typed(ExecutionMode::new("rlmpure"), RlmpureCreateExtras { termination: parent_termination.clone() })?` |
| `plugin_mode` | `SessionPluginMode::Fresh` |
| `first_turn_input` | a `TurnInput` whose conversation has one user-prose part containing `task` (the existing rlmpure user-input binding logic at `lash-mode-rlmpure/src/plugin.rs:228-281` will turn this into `user_input_1` automatically — see also Step 6 about origin filter) |
| `initial_nodes` | one `SessionAppendNode::Event { event: SessionEventRecord::Mode(ModeEvent::rlm(RlmModeEvent::RlmGlobalsPatch(RlmGlobalsPatchPluginBody { set: seed, unset: vec![] }))) }` so the successor's globals are pre-populated before any turn runs |

Important: the rlmpure session's `restore_session` (lash-mode-rlmpure/src/plugin.rs:157-187) walks active events and applies `RlmGlobalsPatch` events. So including the seed as an initial event Just Works — the successor restores it on init like any other patch.

### Step 4 — Detect baton in the rlm driver and end the parent's turn

**File:** `lash-mode-rlm/src/driver.rs` (`handle_exec_result`, lines 413-540)

The `pass_baton` tool returns a sentinel value the driver can recognize. Two implementation choices:

**Option A (recommended) — sentinel in tool result.** Have `pass_baton` return `{"_baton": <successor_session_id>, "ok": true}`. In `handle_exec_result`, after the lashlang exec completes and produces `result.terminal_finish` or a tool-call list, check whether any tool-call result contains `"_baton"`. If so, treat it as a successful submit-without-schema-validation: emit `Effect::Done` with no final answer, and signal session-level termination.

**Option B — new effect/action variant.** Add `DriverAction::HandOff { successor_session_id: String }` and have the rlm driver emit it when the tool result indicates baton. Slightly more invasive.

A is simpler and reuses the existing `submit` finalization path. Lean on A unless implementation reveals a problem.

### Step 5 — Parent halt + chain leaf rotation

This is the trickiest piece because no existing primitive does "parent voluntarily completes; child takes over." Two questions to resolve in code:

**Q5a: Where does "this rlmpure session has handed off, no more turns" live?**

Today rlmpure loops within a session until `submit` or `prose-without-fence`. Baton must terminate the entire session, not just the turn. Reuse the existing session-cancellation path used by `tasks_stop` if possible — check `lash-subagents/src/local.rs` around 888-957 for the cancellation pattern. The mechanism is probably a session state flag or queue drain that prevents new turns from starting.

If reuse isn't clean, add a `SessionCompletionReason::HandedOff { successor_session_id }` field on the session manager's completion record. Don't add a new top-level `RlmTermination` variant — the parent simply completes, full stop.

**Q5b: Active leaf rotation in the user-facing session chain.**

The `SessionGraph.leaf_node_id` field at `lash/src/session_graph.rs:18` tracks the active leaf within a single session graph. The chain-level concept ("which session is currently the user-facing leaf of this multi-session chain") may not exist explicitly today — verify by tracing how `lash-cli` decides which session to forward user input to.

If chain-level leaf tracking is implicit (e.g., the CLI always operates on the most recent session for a given chain id), then the rotation is automatic once the successor is created and the parent is marked completed. Verify by walking the CLI's session-resolution path in `lash-cli/src/resume.rs` and `lash-cli/src/interactive/mod.rs`.

If chain-level leaf tracking is explicit, add a method to update it. Probably one of:
- A field on the persisted session graph (chain id → leaf session id).
- A query at runtime that finds the leaf by walking parent pointers.

### Step 6 — Plugin-origin filter for `user_input_N`

**File:** `lash-mode-rlmpure/src/plugin.rs` (lines 228-281)

Inspect `user_input_patch_from_nodes` and `user_input_patch_from_events`. Currently they filter on `record.role == MessageRole::User` only. Add a check that skips `record.origin == Some(MessageOrigin::Plugin { ... })` for non-user-typed messages, so:
- Soft-warning injections (system-origin user messages) don't get bound as `user_input_N`.
- The successor's first user message (the `task`) DOES get bound as `user_input_1` — so make sure it's emitted with `MessageOrigin::User` or with `origin: None` (whichever is the convention for genuine user inputs). Verify by inspecting how a normal user message from the CLI gets origin set.

Add a test that confirms a plugin-origin user-role message does not increment the `user_input_N` counter.

### Step 7 — Budget-visibility prompt contribution

**File:** `lash-mode-rlmpure/src/plugin.rs`

In `RlmpureModePlugin::register` (lines 79-103), add a third prompt-contributor hook alongside `bound_vars_hook` and `print_output_hook`:

```rust
let budget_hook: lash::plugin::PromptContributor = Arc::new(move |ctx| {
    Box::pin(async move { Ok(budget_prompt_contributions(&ctx)) })
});
reg.prompt().contribute(budget_hook);
```

Implement `budget_prompt_contributions(ctx: &PromptHookContext) -> Vec<PromptContribution>` in `lash-mode-rlmpure/src/rlm_support.rs` (alongside `bound_variables_prompt_contributions`):

```rust
pub fn budget_prompt_contributions(ctx: &PromptHookContext) -> Vec<PromptContribution> {
    let max = match ctx.state.policy().max_context_tokens {
        Some(n) => n,
        None => return Vec::new(),
    };
    let used = ctx.state.last_prompt_usage()
        .map(|u| u.context_budget_tokens)
        .unwrap_or(0);
    let pct = if max > 0 { (used * 100) / max } else { 0 };
    let body = format!(
        "Used: {used} / {max} tokens ({pct}%)\nHand off via `pass_baton` when this becomes inefficient. The next agent starts fresh."
    );
    vec![PromptContribution::execution("Context Budget", body)]
}
```

If `PromptHookContext::state::last_prompt_usage()` getter doesn't exist (verify), add it — same pattern as `projected_rlm_globals` at `lash/src/plugin/history.rs:287`. The data is already in `RuntimeState::last_prompt_usage` at `lash/src/runtime/state.rs:33`.

### Step 8 — Soft-warning injection

**File:** `lash-mode-rlmpure/src/plugin.rs`

Add config field (already shown in Section 2.5). Track per-session warned-state on `RlmpureModeSession`:

```rust
struct RlmpureModeSession {
    config: RlmpureModePluginConfig,
    user_input_count: Mutex<usize>,
    warned_at_threshold: Mutex<bool>,  // new
}
```

Hook into a runtime event that fires after each LLM completion. Inspect `lash::plugin::PluginRuntimeEvent` to find the right event. Most likely:
- After every turn completes — check usage, decide whether to inject.

In the handler:
1. Read `state.last_prompt_usage().context_budget_tokens`.
2. Read `self.config.baton_soft_warn_tokens` — if `None`, skip.
3. If `usage > threshold` and `!warned_at_threshold`:
   - Build a system-origin user message (`MessageOrigin::Plugin { plugin_id: "mode_rlmpure", transient: false }` — verify the right origin shape so the user_input_N filter from Step 6 catches it).
   - Enqueue via `services.turn_input_injection_bridge().enqueue(vec![InjectedTurnInput { ... }])`.
   - Set `warned_at_threshold = true`.

The warning message:
```
Context budget at <used> tokens (soft threshold <threshold>). If your progress so far is captured in named lashlang variables, prefer `pass_baton(task=..., seed={...})` over continuing in this trajectory. Otherwise, continue but stay aware of the budget.
```

## 4. Tests (write these — they're the contract)

**File:** new `lash-subagents/tests/pass_baton.rs` (or in `lash-subagents/src/lib.rs` cfg-test module):

| Test | Asserts |
|---|---|
| `pass_baton_creates_empty_successor` | Successor session has zero conversation events from parent (only the synthesized task), has seed values bound in globals, has `task` as `user_input_1`, has `parent_session_id` = caller. |
| `pass_baton_inherits_termination_schema` | Parent has `RlmTermination::Finish { schema: Some(...) }`; successor's `RlmpureCreateExtras::termination` equals parent's. |
| `pass_baton_terminates_parent` | After call, parent session is marked Completed and refuses new turns. |
| `pass_baton_rotates_active_leaf` | The CLI/runtime's notion of "current leaf" for the chain is the successor, not the parent. (Test the actual mechanism whatever Step 5 lands on.) |
| `seed_values_are_evaluated_in_parent_repl` | A seed value referencing a parent variable carries the actual JSON value forward — i.e., lashlang record literal with `seed = { x = some_var }` resolves `some_var` from the parent's globals. |
| `pass_baton_with_no_seed` | Optional `seed` parameter omitted → successor starts with empty globals, `task` as `user_input_1`. |

**File:** `lash-mode-rlmpure/src/plugin.rs` cfg-test module additions:

| Test | Asserts |
|---|---|
| `budget_prompt_contribution_renders_used_over_max` | With non-empty `last_prompt_usage` and a known max, the rendered contribution contains both numbers and a percentage. |
| `budget_prompt_contribution_handles_first_turn` | With `last_prompt_usage = None`, no panic; either empty contribution or `Used: 0`. |
| `soft_warn_injection_fires_once_at_threshold` | Cross threshold once → message enqueued; cross again on next turn → not enqueued again. |
| `soft_warn_disabled_when_threshold_none` | With `baton_soft_warn_tokens: None`, no warning ever fires. |
| `plugin_origin_user_message_does_not_pollute_user_input_count` | A plugin-origin user-role message does not increment `user_input_N`. |

## 5. Critical files to modify

| File | What changes | Reason |
|---|---|---|
| `lash-subagents/src/policy.rs` (lines 57-89) | Add `pass_baton_definition()`; include in rlm/rlmpure tool list | Tool definition |
| `lash-subagents/src/lib.rs` (lines 323-342) | Dispatch `"pass_baton"`; implement handler | Tool dispatch |
| `lash-subagents/src/local.rs` (lines 495-656 as reference) | Reuse spawn_agent's session-creation path; possibly factor a shared helper | Successor session creation, agent tree wiring |
| `lash-mode-rlm/src/driver.rs` (lines 413-540, `handle_exec_result`) | Detect baton sentinel and emit `Effect::Done` for parent's turn | Parent session termination |
| `lash-mode-rlmpure/src/plugin.rs` | Budget contribution; soft-warning injection; `baton_soft_warn_tokens` config; plugin-origin filter | Mode-specific UX |
| `lash-mode-rlmpure/src/rlm_support.rs` | Add `budget_prompt_contributions` next to existing `bound_variables_prompt_contributions` | Prompt section logic |
| `lash/src/plugin/history.rs` (line 287, `projected_rlm_globals` pattern) | Possibly: expose `last_prompt_usage` on the projected state surface | Make budget data accessible to prompt hooks |
| `lash/src/session_graph.rs` | Possibly: chain-level leaf rotation helper | Active leaf rotation |
| `lash-subagents/tests/pass_baton.rs` (new) | Tests above | Contract |

## 6. Existing infrastructure to reuse — do not reinvent

- **`lash-subagents/src/lib.rs:89-145`** — `build_spawn_create_request` is the closest template for successor session construction. Most of its logic transfers; only `start` becomes hard-coded `Empty`, `fork_turns` is removed, and `first_turn_input` is synthesized from the `task` string.
- **`lash-subagents/src/local.rs:495-656`** — agent tree insertion, parent edge wiring, background task registration. Reuse directly.
- **`lash/src/session.rs:86-128`** — `TurnInputInjectionBridge`. Currently exists but is unused by anything in the codebase. Soft-warning injection is its first consumer.
- **`lash/src/runtime/usage.rs:230-262`** — `normalize_prompt_usage` already produces `PromptUsage.context_budget_tokens` after each LLM call; stored at `RuntimeState.last_prompt_usage`.
- **`lash/src/session_model/mod.rs:174`** — `SessionPolicy.max_context_tokens` already required, validated at runtime startup at `lash/src/runtime/lifecycle.rs:22-25`.
- **`lash-mode-rlmpure/src/plugin.rs` (lines 91-99)** — existing prompt-contribution hook pattern (`bound_vars_hook`, `print_output_hook`) — copy the shape for `budget_hook`.
- **`lash-mode-rlmpure/src/plugin.rs` (lines 228-281)** — `user_input_patch_from_nodes` / `user_input_patch_from_events` — extend with origin filter.
- **`lash-mode-rlmpure/src/plugin.rs` (lines 157-187, `restore_session`)** — already walks active events and applies `RlmGlobalsPatch` events. The seed-as-initial-event approach falls out for free.
- **`RlmGlobalsPatchPluginBody`** at `lash-rlm-types/src/lib.rs:30-41` — the shape for the seed patch. `set: serde_json::Map<String, Value>`, `unset: Vec<String>`.
- **Existing `--rlm-vars-file` flow at `lash-cli/src/bootstrap.rs:129-167`** — proves the pattern of "create a session with an initial RlmGlobalsPatch event applied before any turn runs." Mechanically what `pass_baton`'s seed does, just sourced from a tool call rather than a CLI flag.

## 7. Open implementation questions (resolve in code, not ahead of time)

These are the spots where you'll need to read code carefully and make judgment calls:

1. **Parent halt mechanism (Step 5, Q5a).** Trace `tasks_stop` to find the existing session-cancellation pathway. Is it reusable for "voluntarily complete with status Completed and prevent further turns"? If yes, reuse. If no, add a `SessionCompletionReason::HandedOff` minimum sufficient extension.
2. **Active leaf rotation (Step 5, Q5b).** Walk `lash-cli/src/resume.rs` and `lash-cli/src/interactive/mod.rs` to understand how the CLI decides which session is "current" for a given chain. If it's implicit, baton may Just Work. If it's explicit, you'll need to update wherever the chain leaf is stored.
3. **`PromptHookContext` access to `last_prompt_usage`.** Verify whether the existing surface (the same one `bound_variables_prompt_contributions` uses to read `projected_rlm_globals`) exposes `last_prompt_usage`. If not, add a getter — same pattern as the projected globals.
4. **Plugin-origin user message filter.** Verify exactly how `MessageOrigin` is set on real user inputs vs plugin-injected ones. Looking at `lash/src/session_model/mod.rs:plugin_message_to_message` (around line 31-84) shows that plugin messages get `Some(MessageOrigin::Plugin { plugin_id: "plugin", transient: false })`. Make sure the `task` synthesized for the successor's first turn doesn't accidentally pick up a Plugin origin (it should be a real user input, since the `task` IS `user_input_1` from the successor's perspective).
5. **Sentinel return shape for `pass_baton` tool.** Decide between `{"_baton": "<session_id>"}` or a richer structured return. Make it consistent with how `submit` is detected — read `lash-mode-rlm/src/driver.rs:413-540` (`handle_exec_result`) carefully for the exact pattern.

## 8. Verification

End-to-end test after implementation:

1. **Compile:** `cargo check --workspace`
2. **Unit + integration tests:** `cargo test -p lash-subagents -p lash-mode-rlmpure -p lash-mode-rlm`
3. **Manual integration via lash-cli:**
   - Start an rlmpure session with small `max_context_tokens` (say `8000`) and `baton_soft_warn_tokens: 6000`.
   - Issue a multi-step task. Verify each turn's prompt has the budget section.
   - Force enough trajectory to cross 6000 tokens. Verify the next turn's prompt includes the soft-warning injection.
   - Either let the model voluntarily call `pass_baton`, or include "use pass_baton when convenient" in the original task.
   - When `pass_baton` fires:
     - Parent session log (`~/.lash/sessions/<parent>.llm.jsonl`) shows it terminated as Completed with no submit value.
     - Successor session exists with `parent_session_id` = parent, no inherited conversation events, only seed values in globals, `task` as `user_input_1`.
     - CLI tree view (`lash sessions tree <chain-root>` or interactive equivalent) shows parent → successor lineage.
   - Successor's `submit` produces the user-facing answer for the chain.
4. **Restart edge case:** Quit lash mid-chain, reopen the chain. Verify the successor's seed is still bound (it's an `RlmGlobalsPatch` event in the active path, which `restore_session` re-applies).

## 9. Decision log

For the future you (or a fresh agent) — these are the "why we said no to X" moments worth preserving:

- **"Why not allow `output` on `pass_baton`?"** Output schema is the user-facing contract. Letting any agent in the chain redefine it means the user who asked for `{x: int}` could get something else. Schema is set once at session creation.
- **"Why not allow `capability`?"** Switching capability mid-chain creates surprises. v1 keeps the chain homogeneous. v2 can revisit with a clear story for "I planned, now you implement."
- **"Why no `fork_turns`?"** Baton has fixed inheritance by design. Exposing it tempts callers to ask for `"all"`, defeating the purpose.
- **"Why all REPL globals carry forward (originally proposed) vs explicit `seed` (final)?"** Earlier draft had implicit "carry all model-assigned globals; model can `unset` to clean up." Final design has explicit `seed` because it forces the model to *explicitly curate* what survives. That's the whole point of baton — explicit context curation. Implicit-everything reverts to "regular session that's just had its trajectory zapped."
- **"Why not a separate `briefing` field on top of `task`?"** The model embeds whatever briefing context belongs into the task description. `spawn_agent` proves a single `task` works for self-contained handoffs.
- **"Why no implicit conversation-event inheritance?"** The successor must be genuinely fresh. If we inherited the parent's user messages, the successor would have a partial view of "what the user asked" and a corrupted notion of `user_input_N`. Cleanest semantics: successor is a fresh session that received `task` as its first user message, period.
- **"Why not a new `RlmTermination::Baton` variant?"** Adds invasive surface (lash-rlm-types, driver state machine, projector branches) for a shape that's expressible as "session completed without submit." Parent just looks like it ended; runtime handles chain-leaf rotation orthogonally.
- **"Why scalar token threshold and not a percentage?"** User-specified design choice. Scalar = predictable across model context sizes; percentage = relative. For configuring "warn me at 70k tokens regardless of which model," scalar wins.

## 10. State of working tree at handoff time

- All three foundation tasks (sansio mode-agnostic refactor, rlmpure history unification, ContextApproach scoping) are complete and verified. See `docs/handoffs/sansio-mode-agnostic.md`.
- No partial pass_baton work exists in the tree. Start fresh.
- Branch: `staging`. Recent commits unrelated to this work — see `git log --oneline -10`.

## 11. After v1 ships, the obvious v2 work

Not for this handoff — but useful context for sequencing:

1. **Forced-baton synthesis at hard cap.** When the model ignores soft warnings and crosses the hard cap, runtime synthesizes `pass_baton { task: <user_input_1>, seed: <current globals minus user_input_*> }` and force-passes. Replaces DSPy's mandatory extract entirely.
2. **Automatic "focus" trigger.** Beyond budget — even with budget headroom, the model could call `pass_baton` to drop a dead exploration branch. v1 supports this implicitly (it's just a voluntary call). v2 might add prompt guidance / examples that explicitly suggest it for the "I went down the wrong path" case.
3. **Multi-threshold warnings.** If single-threshold turns out to be insufficient (model doesn't react), add a second threshold closer to the hard cap.
4. **`pass_baton` for rlm and standard modes.** Trickier because chat-mode "globals" don't exist in the same way. Probably needs a different inheritance shape (typed continuation capsule with structured fields like `goal / discoveries / decisions / open questions`) — that's the original baton-mode discussion's "typed capsule" idea.
