use std::collections::HashSet;

use lash::provider::AgentModels;
use lash::{PromptContribution, SessionPolicy, ToolDefinition, ToolParam};

use crate::Capability;

pub(super) fn pick_model_and_variant(
    config: &SessionPolicy,
    models: &Option<AgentModels>,
    capability: Capability,
) -> (String, Option<String>) {
    if let Some(models) = models {
        let selected = match capability {
            Capability::Low => models.low.as_ref(),
            Capability::Medium => models.medium.as_ref(),
            Capability::High => models.high.as_ref(),
        };
        if let Some(model) = selected {
            let variant = preferred_override_variant(&config.provider, model, capability);
            return (model.clone(), variant);
        }
    }

    if let Some((model, variant)) = config.provider.default_agent_model(capability.as_str()) {
        return (model.to_string(), variant.map(str::to_string));
    }

    let model = config.model.clone();
    let variant = config
        .provider
        .default_model_variant(&model)
        .map(str::to_string)
        .or_else(|| config.model_variant.clone());
    (model, variant)
}

fn preferred_override_variant(
    provider: &lash::Provider,
    model: &str,
    capability: Capability,
) -> Option<String> {
    let variant = capability.as_str();
    if provider.supported_variants(model).contains(&variant) {
        return Some(variant.to_string());
    }
    provider.default_model_variant(model).map(str::to_string)
}

pub(super) fn denied_tools(capability: Capability) -> HashSet<&'static str> {
    match capability {
        Capability::Low => ["apply_patch", "ask", "spawn_agent"].into_iter().collect(),
        Capability::Medium | Capability::High => ["ask"].into_iter().collect(),
    }
}

pub(super) fn subagent_prompt_contributions() -> Vec<PromptContribution> {
    vec![
        PromptContribution::guidance(
            "Subagents",
            "Use `spawn_agent` for bounded subagent work that can run in parallel with your current step. Use `capability` to match the task: `low` for read-heavy exploration, `medium` for contained implementation or validation, and `high` for larger independent ownership. Keep each spawned task concrete and scoped. In user-facing prose, call them subagents, not delegates or child agents. Avoid overlapping file edits across concurrently running subagents.",
        ),
        PromptContribution::guidance(
            "Agent Lifecycle",
            "Use `send_message` for concise out-of-band notes, `followup_task` to give an existing subagent more work, `wait_agent` to consume subagent updates or completions, and `list_agents` to inspect the live tree. Stop a subagent subtree with the shared `tasks_stop` tool (target id is `subagent:{path}`). `send_message` may target `/root`; `followup_task` may not. `fork_turns` controls how much of the current session context a new subagent receives: `none`, `all`, or a positive integer string for only the most recent turns.",
        ),
    ]
}

pub(super) fn subagent_tool_definitions(
    execution_mode: lash::ExecutionMode,
) -> Vec<ToolDefinition> {
    match execution_mode {
        lash::ExecutionMode::Standard => standard_subagent_tool_definitions(),
        lash::ExecutionMode::Rlm => rlm_subagent_tool_definitions(),
    }
}

fn standard_subagent_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        spawn_agent_definition(vec![
            r#"worker = spawn_agent(task_name="inspect_auth", task="Summarize the auth flow", capability="low")"#.into(),
            r#"typed = spawn_agent(task_name="extract_line", task="Find the longest line in src/main.rs", capability="low", output={"line":"str","length":"int"})"#.into(),
            r#"update = wait_agent(targets=["/root/inspect_auth"], timeout_ms=30000)"#.into(),
        ]),
        send_message_definition(vec![
            r#"send_message(target="/root", message="Found the config file and traced the parse path.")"#.into(),
        ]),
        followup_task_definition(vec![
            r#"followup_task(target="/root/inspect_auth", task="Turn the summary into a concise bug report", interrupt=false)"#.into(),
        ]),
        wait_agent_definition(vec![
            r#"update = wait_agent(timeout_ms=30000)"#.into(),
            r#"update = wait_agent(targets=["/root/inspect_auth"], timeout_ms=30000)"#.into(),
        ]),
        list_agents_definition(vec![r#"agents = list_agents(path_prefix="/root")"#.into()]),
    ]
}

fn rlm_subagent_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        spawn_agent_definition(vec![
            r#"worker = call spawn_agent { task_name: "inspect_auth", task: "Summarize the auth flow", capability: "low" }"#.into(),
            r#"typed = call spawn_agent { task_name: "extract_line", task: "Find the longest line in src/main.rs", capability: "low", output: { line: "str", length: "int" } }"#.into(),
            r#"Shape = Type { name: str, tags: list[str], status: enum["ok", "err"] }"#.into(),
            r#"signed = call spawn_agent { task_name: "catalog", task: "Parse the book listing", capability: "low", output: Shape }"#.into(),
            r#"wait = start call wait_agent { targets: [worker.path], timeout_ms: 30000 }"#.into(),
            r#"update = await wait"#.into(),
        ]),
        send_message_definition(vec![
            r#"call send_message { target: "/root", message: "Found the config file and traced the parse path." }"#.into(),
        ]),
        followup_task_definition(vec![
            r#"call followup_task { target: "/root/inspect_auth", task: "Turn the summary into a concise bug report", interrupt: false }"#.into(),
        ]),
        wait_agent_definition(vec![
            r#"call wait_agent { timeout_ms: 30000 }"#.into(),
            r#"call wait_agent { targets: ["/root/inspect_auth"], timeout_ms: 30000 }"#.into(),
        ]),
        list_agents_definition(vec![r#"call list_agents { path_prefix: "/root" }"#.into()]),
    ]
}

fn spawn_agent_definition(examples: Vec<String>) -> ToolDefinition {
    ToolDefinition {
        name: "spawn_agent".into(),
        description: "Spawn a subagent under the current agent path and start it in the background. Pick `capability` from `low`, `medium`, or `high`. Set `fork_turns` to `none`, `all`, or a positive integer string to control inherited context. If `output` is present, the subagent must return a value matching that shape — pass either a record of scalar type descriptors (`{ line: \"str\", length: \"int\" }`) or a `Type { ... }` literal (supports nested objects, enums, `list[T]`, and `?` optional fields). `task_name` is auto-normalized (lowercased; spaces, hyphens, and other non-alphanumeric characters collapse to `_`); the response includes a `task_name_note` when normalization changed what you sent.".into(),
        params: vec![
            ToolParam::typed("task_name", "str"),
            ToolParam::typed("task", "str"),
            ToolParam::typed("capability", "str"),
            ToolParam::optional("fork_turns", "str"),
            ToolParam::optional("output", "dict"),
        ],
        returns: "dict".into(),
        examples,
        enabled: true,
        injected: true,
        input_schema_override: Some(spawn_agent_input_schema()),
        output_schema_override: None,
    }
}

fn send_message_definition(examples: Vec<String>) -> ToolDefinition {
    ToolDefinition {
        name: "send_message".into(),
        description: "Queue a concise out-of-band message for another agent. Use this for notes, findings, or status updates without assigning a new task.".into(),
        params: vec![ToolParam::typed("target", "str"), ToolParam::typed("message", "str")],
        returns: "dict".into(),
        examples,
        enabled: true,
        injected: true,
        input_schema_override: None,
        output_schema_override: None,
    }
}

fn followup_task_definition(examples: Vec<String>) -> ToolDefinition {
    ToolDefinition {
        name: "followup_task".into(),
        description: "Give an existing non-root agent another task. If `interrupt` is true and the target is busy, its current turn is cancelled and the new task runs next.".into(),
        params: vec![
            ToolParam::typed("target", "str"),
            ToolParam::typed("task", "str"),
            ToolParam::optional("interrupt", "bool"),
        ],
        returns: "dict".into(),
        examples,
        enabled: true,
        injected: true,
        input_schema_override: None,
        output_schema_override: None,
    }
}

fn wait_agent_definition(examples: Vec<String>) -> ToolDefinition {
    ToolDefinition {
        name: "wait_agent".into(),
        description: "Wait for **subagent** lifecycle events: task starts, completions, messages, or closes. With no `targets`, waits on the current agent's mailbox and descendant agents. This tool only receives subagent events — it does **not** collect `monitor` tool output, shell process events, or any other background stream. Monitor events are delivered automatically as new turn input, so never call `wait_agent` hoping to drain a monitor.".into(),
        params: vec![
            ToolParam::optional("targets", "list"),
            ToolParam::optional("timeout_ms", "int"),
        ],
        returns: "dict".into(),
        examples,
        enabled: true,
        injected: true,
        input_schema_override: Some(wait_agent_input_schema()),
        output_schema_override: None,
    }
}

fn list_agents_definition(examples: Vec<String>) -> ToolDefinition {
    ToolDefinition {
        name: "list_agents".into(),
        description: "List the live agent tree under the current agent path or an optional `path_prefix` subtree.".into(),
        params: vec![ToolParam::optional("path_prefix", "str")],
        returns: "dict".into(),
        examples,
        enabled: true,
        injected: true,
        input_schema_override: None,
        output_schema_override: None,
    }
}

fn spawn_agent_input_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "task_name": { "type": "string" },
            "task": { "type": "string" },
            "capability": { "type": "string", "enum": ["low", "medium", "high"] },
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
            "timeout_ms": { "type": "integer", "minimum": 0 }
        },
        "additionalProperties": false
    })
}
