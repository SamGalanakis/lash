use lash::{PromptContribution, ToolDefinition, ToolExecutionMode};

use crate::CapabilityRegistry;

pub(super) fn subagent_prompt_contributions() -> Vec<PromptContribution> {
    vec![PromptContribution::guidance(
        "Subagents",
        "Use `spawn_agent` for bounded subagent work that can run in parallel with your current step. Pick `capability` to match the task: `low` for read-heavy exploration, `medium` for contained implementation or validation, `high` for larger independent ownership. Keep each spawned task concrete and scoped, and avoid overlapping file edits across concurrently running subagents. In user-facing prose, call them subagents, not delegates or child agents.\n\nLifecycle: `send_message` (concise out-of-band notes; may target `/root`), `followup_task` (hand an existing subagent more work; cannot target `/root`; use `delivery: \"interrupt\"` to cancel its current turn), `wait_agent` (defaults to task completion; read `completed` and `pending`), `list_agents` (inspect the live tree), and `tasks_stop` with the `task_id` returned by `spawn_agent` to stop a subtree.\n\n`fork_turns` controls how much of the current session context a new subagent receives: `none`, `all`, or a positive integer string for only the most recent turns.",
    )]
}

pub(super) fn subagent_tool_definitions(
    execution_mode: lash::ExecutionMode,
    registry: &CapabilityRegistry,
) -> Vec<ToolDefinition> {
    let names = registry.names();
    if execution_mode == lash::ExecutionMode::standard() {
        standard_subagent_tool_definitions(&names)
    } else if execution_mode == lash::ExecutionMode::new("rlm") {
        rlm_subagent_tool_definitions(&names)
    } else {
        standard_subagent_tool_definitions(&names)
    }
}

fn standard_subagent_tool_definitions(capability_names: &[String]) -> Vec<ToolDefinition> {
    let example_capability = example_capability_name(capability_names);
    vec![
        spawn_agent_definition(
            capability_names,
            vec![
                format!(
                    r#"worker = spawn_agent(task_name="inspect_auth", task="Summarize the auth flow", capability="{example_capability}")"#
                ),
                format!(
                    r#"typed = spawn_agent(task_name="extract_line", task="Find the longest line in src/main.rs", capability="{example_capability}", output={{"line":"str","length":"int"}})"#
                ),
                r#"update = wait_agent(targets=[worker.target], timeout_ms=30000)"#.into(),
            ],
        ),
        send_message_definition(vec![
            r#"send_message(target="/root", message="Found the config file and traced the parse path.")"#.into(),
        ]),
        followup_task_definition(vec![
            r#"followup_task(target="/root/inspect_auth", task="Turn the summary into a concise bug report")"#.into(),
        ]),
        wait_agent_definition(vec![
            r#"update = wait_agent(timeout_ms=30000)"#.into(),
            r#"update = wait_agent(targets=["/root/inspect_auth"], timeout_ms=30000)"#.into(),
        ]),
        list_agents_definition(vec![r#"agents = list_agents(path_prefix="/root")"#.into()]),
    ]
}

fn rlm_subagent_tool_definitions(capability_names: &[String]) -> Vec<ToolDefinition> {
    let example_capability = example_capability_name(capability_names);
    vec![
        spawn_agent_definition(
            capability_names,
            vec![
                format!(
                    r#"worker = call spawn_agent {{ task_name: "inspect_auth", task: "Summarize the auth flow", capability: "{example_capability}" }}"#
                ),
                format!(
                    r#"typed = call spawn_agent {{ task_name: "extract_line", task: "Find the longest line in src/main.rs", capability: "{example_capability}", output: {{ line: "str", length: "int" }} }}"#
                ),
                r#"Shape = Type { name: str, tags: list[str], status: enum["ok", "err"] }"#.into(),
                format!(
                    r#"signed = call spawn_agent {{ task_name: "catalog", task: "Parse the book listing", capability: "{example_capability}", output: Shape }}"#
                ),
                r#"wait = start call wait_agent { targets: [worker.target], timeout_ms: 30000 }"#.into(),
                r#"update = (await wait)?"#.into(),
                r#"if !empty(update.pending) { update = (call wait_agent { targets: update.pending, timeout_ms: 30000 })? }"#.into(),
            ],
        ),
        continue_as_definition(vec![
            r#"call continue_as { task: "filter the candidates by relevance to the user's query and submit the top 3", seed: { candidates: candidates, query: user_input_1 } }"#.into(),
        ]),
        send_message_definition(vec![
            r#"call send_message { target: "/root", message: "Found the config file and traced the parse path." }"#.into(),
        ]),
        followup_task_definition(vec![
            r#"call followup_task { target: "/root/inspect_auth", task: "Turn the summary into a concise bug report" }"#.into(),
        ]),
        wait_agent_definition(vec![
            r#"call wait_agent { timeout_ms: 30000 }"#.into(),
            r#"call wait_agent { targets: ["/root/inspect_auth"], timeout_ms: 30000 }"#.into(),
        ]),
        list_agents_definition(vec![r#"call list_agents { path_prefix: "/root" }"#.into()]),
    ]
}

fn example_capability_name(capability_names: &[String]) -> String {
    capability_names
        .iter()
        .find(|name| name.as_str() == "low")
        .or_else(|| capability_names.first())
        .cloned()
        .unwrap_or_else(|| "low".to_string())
}

fn capability_list_for_description(capability_names: &[String]) -> String {
    if capability_names.is_empty() {
        return "(no capabilities registered)".to_string();
    }
    let quoted: Vec<String> = capability_names
        .iter()
        .map(|name| format!("`{name}`"))
        .collect();
    match quoted.len() {
        1 => quoted.into_iter().next().expect("len 1"),
        2 => format!("{} or {}", quoted[0], quoted[1]),
        _ => {
            let last = quoted.last().expect("non-empty").clone();
            let head = quoted[..quoted.len() - 1].join(", ");
            format!("{head}, or {last}")
        }
    }
}

fn continue_as_definition(examples: Vec<String>) -> ToolDefinition {
    tool_definition(
        "continue_as",
        "Tail-call: end this session and continue the work as a fresh successor with the same tools and a clean window. The successor inherits tool access and output schema but does not inherit prior conversation, trajectory, or globals — it sees only `task` and `seed`. Use when most of your trajectory has gone stale (failed attempts, large observations you've already extracted from) or context budget is tight. Pack `task` and `seed` with what the successor needs to keep going: concrete goals, constraints, discovered facts, IDs, tokens, file paths, partial results, and next steps. Be selective; leave dead ends behind. If useful, include a short slice of prior lashlang/repl history in `seed` so the successor can resume without rediscovering it.",
        continue_as_input_schema(),
        examples,
        ToolExecutionMode::Parallel,
    )
}

fn spawn_agent_definition(capability_names: &[String], examples: Vec<String>) -> ToolDefinition {
    let cap_list = capability_list_for_description(capability_names);
    let description = format!(
        "Spawn a subagent under the current agent target and start it in the background. Pick `capability` from {cap_list}. The response returns `target` for agent operations (`wait_agent`, `followup_task`, `send_message`) and `task_id` for task control (`tasks_stop`). Set `fork_turns` to `none`, `all`, or a positive integer string to control inherited context. If `output` is present, the subagent must return a value matching that shape — pass either a record of scalar type descriptors (`{{ line: \"str\", length: \"int\" }}`) or a `Type {{ ... }}` literal (supports nested objects, enums, `list[T]`, and `?` optional fields). `task_name` is auto-normalized (lowercased; spaces, hyphens, and other non-alphanumeric characters collapse to `_`); the response includes a `task_name_note` when normalization changed what you sent."
    );
    tool_definition(
        "spawn_agent",
        description,
        spawn_agent_input_schema(capability_names),
        examples,
        ToolExecutionMode::Parallel,
    )
}

fn send_message_definition(examples: Vec<String>) -> ToolDefinition {
    tool_definition(
        "send_message",
        "Deliver a concise out-of-band message to another agent without assigning a task. `delivery` defaults to `next_possible`: inject into the current turn if the target is running, or wake an idle non-root target with this message. Use `next_turn` to queue it without waking the target. Use `interrupt` to cancel the current target turn and deliver the message next.",
        send_message_input_schema(),
        examples,
        ToolExecutionMode::Parallel,
    )
}

fn followup_task_definition(examples: Vec<String>) -> ToolDefinition {
    tool_definition(
        "followup_task",
        "Give an existing non-root agent another task. `delivery` defaults to `next_possible`: start immediately if idle or queue after the current turn if busy. Use `interrupt` to cancel the target's current turn and run this task next. Use `next_turn` to queue without waking an idle target. Optional `output` supplies a per-follow-up output schema (same shape as `spawn_agent.output`) that retypes the subagent for this single follow-up — it overrides any schema baked in at spawn. Omit `output` to run the follow-up as free-form (no schema validation for that turn).",
        followup_task_input_schema(),
        examples,
        ToolExecutionMode::Serial,
    )
}

fn wait_agent_definition(examples: Vec<String>) -> ToolDefinition {
    ToolDefinition::new(
        "wait_agent",
        "Wait for **subagent** lifecycle events. By default `until` is `task_completed`, so messages and `task_started` events do not complete the wait. Responses are plural and task-oriented: `completed` contains finished task results and `pending` contains targets that still have task work in flight. A timeout is a successful tool result with `timed_out: true`, `completed: []`, and `pending` populated when targets are still running; wait again with `targets: response.pending` if you need all results. Do not read `events[0].result` unless you explicitly requested `any_event`. Use `until=\"message\"`, `\"terminal\"`, `\"any_result\"`, or `\"any_event\"` only when that exact behavior is needed. With no `targets`, waits on the current agent and descendants. This tool only receives subagent events — it does **not** collect `monitor` tool output, shell process events, or any other background stream. Monitor events are delivered automatically as new turn input, so never call `wait_agent` hoping to drain a monitor.",
        wait_agent_input_schema(),
        wait_agent_output_schema(),
    )
    .with_examples(examples)
    .with_execution_mode(ToolExecutionMode::Parallel)
}

fn list_agents_definition(examples: Vec<String>) -> ToolDefinition {
    tool_definition(
        "list_agents",
        "List the live agent tree under the current agent path or an optional `path_prefix` subtree.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "path_prefix": { "type": "string" }
            },
            "additionalProperties": false
        }),
        examples,
        ToolExecutionMode::Parallel,
    )
}

fn tool_definition(
    name: &str,
    description: impl Into<String>,
    input_schema: serde_json::Value,
    examples: Vec<String>,
    execution_mode: ToolExecutionMode,
) -> ToolDefinition {
    ToolDefinition::new(
        name,
        description,
        input_schema,
        serde_json::json!({ "type": "object", "additionalProperties": true }),
    )
    .with_examples(examples)
    .with_execution_mode(execution_mode)
}

fn continue_as_input_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "task": { "type": "string" },
            "seed": { "type": "object", "additionalProperties": true }
        },
        "required": ["task"],
        "additionalProperties": false
    })
}

fn spawn_agent_input_schema(capability_names: &[String]) -> serde_json::Value {
    let enum_values: Vec<serde_json::Value> = capability_names
        .iter()
        .map(|name| serde_json::Value::String(name.clone()))
        .collect();
    serde_json::json!({
        "type": "object",
        "properties": {
            "task_name": { "type": "string" },
            "task": { "type": "string" },
            "capability": { "type": "string", "enum": enum_values },
            "fork_turns": {
                "oneOf": [
                    { "type": "string" },
                    { "type": "integer", "minimum": 1 }
                ]
            },
            "output": { "type": "object", "additionalProperties": true }
        },
        "required": ["task_name", "task", "capability"],
        "additionalProperties": false
    })
}

fn wait_agent_input_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "targets": { "type": "array", "items": { "type": "string" } },
            "until": {
                "type": "string",
                "enum": ["task_completed", "terminal", "message", "any_result", "any_event"],
                "default": "task_completed"
            },
            "timeout_ms": { "type": "integer", "minimum": 0 }
        },
        "additionalProperties": false
    })
}

fn wait_agent_output_schema() -> serde_json::Value {
    let session = serde_json::json!({
        "type": "object",
        "properties": {
            "id": { "type": "string" },
            "parent_session_id": { "type": ["string", "null"] },
            "task": { "type": "string" },
            "iterations": { "type": "integer", "minimum": 0 },
            "tool_calls": { "type": "integer", "minimum": 0 },
            "model": { "type": "string" },
            "model_variant": { "type": ["string", "null"] },
            "token_usage": { "type": "object", "additionalProperties": true }
        },
        "required": ["id", "task", "iterations", "tool_calls", "model", "token_usage"],
        "additionalProperties": false
    });
    serde_json::json!({
        "type": "object",
        "properties": {
            "timed_out": { "type": "boolean" },
            "completed": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "target": { "type": "string" },
                        "task": { "type": "string" },
                        "status": { "type": "string" },
                        "result": {
                            "description": "Subagent result value; typed by spawn_agent.output when provided."
                        },
                        "error": { "type": ["string", "null"] },
                        "session": session
                    },
                    "required": ["target", "task", "status", "result", "session"],
                    "additionalProperties": false
                }
            },
            "pending": { "type": "array", "items": { "type": "string" } },
            "messages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "from": { "type": "string" },
                        "to": { "type": "string" },
                        "message": { "type": "string" }
                    },
                    "required": ["from", "to", "message"],
                    "additionalProperties": false
                }
            },
            "closed": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": { "target": { "type": "string" } },
                    "required": ["target"],
                    "additionalProperties": false
                }
            },
            "events": { "type": "array", "items": { "type": "object", "additionalProperties": true } }
        },
        "required": ["timed_out", "completed", "pending", "messages", "closed", "events"],
        "additionalProperties": false
    })
}

fn send_message_input_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "target": { "type": "string" },
            "message": { "type": "string" },
            "delivery": delivery_schema()
        },
        "required": ["target", "message"],
        "additionalProperties": false
    })
}

fn followup_task_input_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "target": { "type": "string" },
            "task": { "type": "string" },
            "delivery": delivery_schema(),
            "output": { "type": "object", "additionalProperties": true }
        },
        "required": ["target", "task"],
        "additionalProperties": false
    })
}

fn delivery_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "string",
        "enum": ["next_possible", "interrupt", "next_turn"],
        "default": "next_possible"
    })
}
