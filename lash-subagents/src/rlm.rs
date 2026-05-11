//! RLM-mode subagent spawning surface.
//!
//! Examples are written in lashlang `call <tool> { ... }` syntax. Prompt prose
//! is tuned for schema-first results and binding subagent output.

use std::sync::Arc;

use async_trait::async_trait;
use lash_core::{ToolCall, ToolContext, ToolDefinition, ToolExecutionMode, ToolProvider, ToolResult};
use serde_json::Value;

use crate::SubagentSessionConfigurator;
use crate::capability::CapabilityRegistry;
use crate::host::SubagentHost;
use crate::shared::{
    self, build_spawn_create_request, capability_list_for_description, example_capability_name,
    finalise_tool_result, render_task_prompt, required_string, spawn_agent_input_schema,
    tool_definition, turn_input_for_task, unknown_capability_message,
};
use crate::types::{CloseAgentRequest, SpawnAgentRequest, WaitAgentRequest, WaitUntil};

pub(crate) struct RlmSubagentToolsProvider {
    pub(crate) registry: Arc<CapabilityRegistry>,
    pub(crate) host: Arc<dyn SubagentHost>,
    pub(crate) configurator: Arc<dyn SubagentSessionConfigurator>,
    pub(crate) include_submit_error: bool,
}

impl RlmSubagentToolsProvider {
    async fn spawn_agent(&self, args: &Value, context: &ToolContext) -> Result<Value, String> {
        let agent_name = required_string(args, "agent_name")?;
        let task = required_string(args, "task")?;
        let capability_name = required_string(args, "capability")?;
        if self.registry.get(&capability_name).is_none() {
            return Err(unknown_capability_message(&capability_name, &self.registry));
        }
        let output_schema = lash_llm_tools::parse_output_schema(args.get("output"))?;
        let seed = lash_rlm_types::classify_seed(args)?;
        let create_request = build_spawn_create_request(
            &self.registry,
            context,
            &capability_name,
            output_schema.clone(),
            seed,
            self.configurator.as_ref(),
        )
        .await?;
        let turn_input = turn_input_for_task(render_task_prompt(&task, output_schema.as_ref()));
        let response = self
            .host
            .spawn_agent(
                context,
                SpawnAgentRequest {
                    agent_name,
                    task,
                    capability: capability_name,
                    hidden_tools: create_request.tool_access.hidden_tools.clone(),
                    create_request,
                    turn_input,
                },
            )
            .await?;
        let wait = self
            .host
            .wait_agent(
                context,
                WaitAgentRequest {
                    agents: vec![response.agent_name.clone()],
                    until: WaitUntil::TaskCompleted,
                    timeout_ms: None,
                    all: true,
                },
            )
            .await?;
        if context
            .cancellation_token()
            .is_some_and(|token| token.is_cancelled())
        {
            let _ = self
                .host
                .close_agent(
                    context,
                    CloseAgentRequest {
                        agent_name: response.agent_name.clone(),
                    },
                )
                .await;
            return Err("spawn_agent was cancelled".to_string());
        }
        let completion = wait
            .completed
            .get(&response.agent_name)
            .ok_or_else(|| "spawn_agent completed without a result".to_string())?;
        if completion.status == "failed" {
            return Err(completion
                .error
                .clone()
                .unwrap_or_else(|| "subagent failed".to_string()));
        }
        Ok(completion.result.clone())
    }
}

#[async_trait]
impl ToolProvider for RlmSubagentToolsProvider {
    fn definitions(&self) -> Vec<ToolDefinition> {
        let mut definitions = rlm_subagent_tool_definitions(&self.registry.names());
        if self.include_submit_error {
            definitions.push(shared::submit_error_tool_definition());
        }
        definitions
    }

    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        let result = match call.name {
            "spawn_agent" => self.spawn_agent(call.args, call.context).await,
            "submit_error" => return shared::submit_error_tool_result(call.args),
            other => Err(format!("Unknown tool: {other}")),
        };
        finalise_tool_result(result)
    }
}

pub(crate) fn rlm_subagent_tool_definitions(capability_names: &[String]) -> Vec<ToolDefinition> {
    vec![spawn_agent_tool_definition(capability_names)]
}

pub fn spawn_agent_tool_definition(capability_names: &[String]) -> ToolDefinition {
    let example_capability = example_capability_name(capability_names);
    spawn_agent_definition(
        capability_names,
        vec![
            // Schema-first: the highest-leverage shape — bind a typed result.
            format!(
                r#"typed = (call spawn_agent {{ agent_name: "extract_line", task: "Find the longest line in src/main.rs", capability: "{example_capability}", output: {{ line: "str", length: "int" }} }})?"#
            ),
            // Reusable Type literal for richer shapes.
            r#"Shape = Type { name: str, tags: list[str], status: enum["ok", "err"] }"#.into(),
            format!(
                r#"signed = (call spawn_agent {{ agent_name: "catalog", task: "Parse the book listing in data/books.json", capability: "{example_capability}", output: Shape }})?"#
            ),
            // Canonical fan-out: start N and await generic handles.
            format!(
                r#"a = start call spawn_agent {{ agent_name: "auth_files", task: "List files under src/auth/ that handle session tokens", capability: "{example_capability}", output: {{ files: "list[str]" }} }}"#
            ),
            format!(
                r#"b = start call spawn_agent {{ agent_name: "db_migrations", task: "Summarise migrations/ schema changes since v3", capability: "{example_capability}", output: {{ summary: "str" }} }}"#
            ),
            r#"results = parallel { auth: (await a)?, db: (await b)? }"#.into(),
            // seed: pass projected source through to the child as a projected
            // binding; pass plain values as RLM globals on the child.
            format!(
                r#"answer = (call spawn_agent {{ agent_name: "extract", task: "Solve sub-problem 3 using the bound problem text and the running findings.", capability: "{example_capability}", seed: {{ problem: input.prompt, findings: findings }}, output: {{ value: "int" }} }})?"#
            ),
            // Untyped is fine for free-form prose results.
            format!(
                r#"prose = call spawn_agent {{ agent_name: "audit_endpoints", task: "Skim the routes in api/ and flag any missing auth checks", capability: "{example_capability}" }}"#
            ),
        ],
    )
}

fn spawn_agent_definition(capability_names: &[String], examples: Vec<String>) -> ToolDefinition {
    let cap_list = capability_list_for_description(capability_names);
    let capability_detail = capability_detail_for_tool_description(capability_names);
    let description = format!(
        "Run a subagent and return its final result. Plain `call spawn_agent {{ ... }}` blocks until the child finishes. Use `start call spawn_agent {{ ... }}` for fan-out; it returns a generic lashlang async handle immediately. Pick `capability` from {cap_list}. {capability_detail} `output` defines the typed return shape. \
        \n\nThe child starts with **no** inherited state — globals, projected bindings, message history are all blank. Hand it specific data via `seed: {{ name: value, ... }}`. Each entry's kind is preserved automatically: if `value`'s lashlang source root is a host-projected binding (e.g. `seed: {{ problem: input.prompt }}`) the child receives `problem` as a read-only projected binding, identical to how it appeared on the parent. Otherwise it lands as a regular RLM global. Computed expressions default to global. Projected seed entries require an RLM child; passing one to a non-RLM capability is an error.\
        \n\nA child can fail terminally with `call submit_error {{ reason: \"...\" }}`; this tool returns an error with that reason. `agent_name` is auto-normalized."
    );
    tool_definition(
        "spawn_agent",
        description,
        spawn_agent_input_schema(capability_names),
        examples,
        ToolExecutionMode::Serial,
    )
    .with_output_from_input_schema("output", None)
}

fn capability_detail_for_tool_description(capability_names: &[String]) -> String {
    if capability_names.len() == 1 {
        return "Only the listed capability is available in this session.".to_string();
    }
    "Use only the listed capabilities; unavailable capability names are rejected.".to_string()
}
