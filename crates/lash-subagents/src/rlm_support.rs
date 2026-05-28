//! Private helpers for the RLM subagent tool surface.

use lash_core::plugin::PluginError;
use lash_core::{
    AssembledTurn, CausalRef, InputItem, PluginOptions, SessionCreateRequest, SessionPolicy,
    SessionSnapshot, SessionSpec, SessionStartPoint, SessionToolAccess, SubagentSessionContext,
    ToolDefinition, ToolResult, ToolScheduling, ToolSurfaceContribution, TurnFinish, TurnInput,
    TurnOutcome, TurnStop,
};
use lash_rlm_types::RlmTermination;
use serde_json::{Value, json};

use crate::capability::{CapabilityContext, CapabilityRegistry, CapabilityResolution};
use crate::{SubagentSessionConfigurator, SubagentSpawnContext};

#[cfg(test)]
pub(crate) fn build_session_policy(
    registry: &CapabilityRegistry,
    current_policy: &SessionPolicy,
    capability_name: &str,
) -> Result<SessionPolicy, String> {
    let resolution = resolve_capability_spec(registry, current_policy, capability_name)?;
    Ok(resolution.spec.resolve_against(current_policy))
}

pub(crate) fn resolve_capability_spec(
    registry: &CapabilityRegistry,
    current_policy: &SessionPolicy,
    capability_name: &str,
) -> Result<CapabilityResolution, String> {
    let capability = registry
        .get(capability_name)
        .ok_or_else(|| unknown_capability_message(capability_name, registry))?;
    Ok(capability.resolve(&CapabilityContext {
        parent_policy: current_policy,
    }))
}

pub(crate) struct SpawnCreateRequestInput<'a> {
    pub(crate) registry: &'a CapabilityRegistry,
    pub(crate) parent_session_id: &'a str,
    pub(crate) current_snapshot: SessionSnapshot,
    pub(crate) session_spec: &'a SessionSpec,
    pub(crate) capability_name: &'a str,
    pub(crate) output_schema: Option<Value>,
    pub(crate) seed: lash_protocol_rlm::RlmSeed,
    pub(crate) parent_subagent: Option<&'a SubagentSessionContext>,
    pub(crate) caused_by: Option<CausalRef>,
    pub(crate) configurator: &'a dyn SubagentSessionConfigurator,
}

pub(crate) async fn build_spawn_create_request(
    input: SpawnCreateRequestInput<'_>,
) -> Result<SessionCreateRequest, String> {
    let SpawnCreateRequestInput {
        registry,
        parent_session_id,
        current_snapshot,
        session_spec,
        capability_name,
        output_schema,
        seed,
        parent_subagent,
        caused_by,
        configurator,
    } = input;
    let mut policy = session_spec.resolve_against(&current_snapshot.policy);
    let capability_resolution = resolve_capability_spec(registry, &policy, capability_name)?;
    policy = capability_resolution.spec.resolve_against(&policy);
    let termination = match output_schema.clone() {
        Some(schema) => RlmTermination::SubmitRequired {
            schema: Some(schema),
        },
        None => RlmTermination::default(),
    };
    let plugin_options = PluginOptions::typed(
        lash_protocol_rlm::RLM_PROTOCOL_PLUGIN_ID,
        lash_rlm_types::RlmCreateExtras { termination },
    )
    .map_err(|err| format!("failed to encode rlm plugin options: {err}"))?;

    let initial_nodes = lash_protocol_rlm::rlm_seed_initial_nodes(seed);
    let mut request = SessionCreateRequest::child(
        parent_session_id,
        SessionStartPoint::Empty,
        policy,
        plugin_options,
        "subagent",
    )
    .with_plugin_source(capability_resolution.plugin_source)
    .with_initial_nodes(initial_nodes);
    let child_policy = request.policy.as_ref().expect("child policy set").clone();
    configurator.configure(
        &SubagentSpawnContext {
            parent_session_id,
            capability: capability_name,
            parent_policy: &current_snapshot.policy,
            child_policy: &child_policy,
        },
        &mut request,
    )?;
    finalize_subagent_create_request(
        request,
        parent_session_id,
        capability_name,
        parent_subagent,
        caused_by,
    )
}

fn finalize_subagent_create_request(
    mut request: SessionCreateRequest,
    parent_session_id: &str,
    capability_name: &str,
    parent_subagent: Option<&SubagentSessionContext>,
    caused_by: Option<CausalRef>,
) -> Result<SessionCreateRequest, String> {
    if let Some(caused_by) = caused_by {
        request = request.with_caused_by(caused_by);
    }
    let child_depth = parent_subagent
        .map(|parent| parent.depth.saturating_add(1))
        .unwrap_or(1);
    if child_depth > MAX_SUBAGENT_DEPTH {
        return Err(format!(
            "subagent recursion depth exceeded: max depth is {MAX_SUBAGENT_DEPTH}"
        ));
    }
    let mut hidden_tools = request.tool_access.hidden_tools.clone();
    if child_depth >= MAX_SUBAGENT_DEPTH {
        hidden_tools.extend(SUBAGENT_SUITE_DENY.iter().map(|name| name.to_string()));
    }
    let tools = request.tool_access.tools.clone();
    Ok(request
        .with_tool_access(SessionToolAccess {
            tools,
            hidden_tools,
        })
        .with_subagent_context(SubagentSessionContext {
            parent_session_id: parent_session_id.to_string(),
            capability: capability_name.to_string(),
            depth: child_depth,
            max_depth: MAX_SUBAGENT_DEPTH,
        }))
}

pub(crate) fn unknown_capability_message(name: &str, registry: &CapabilityRegistry) -> String {
    let known = registry.names();
    if known.is_empty() {
        format!("unknown capability `{name}`: no capabilities registered")
    } else {
        format!(
            "unknown capability `{name}`: expected one of {}",
            known
                .iter()
                .map(|n| format!("`{n}`"))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

pub(crate) fn render_task_prompt(task: &str, output_schema: Option<&Value>) -> String {
    let mut sections = vec![task.to_string()];
    if let Some(schema) = output_schema {
        let schema_pretty =
            serde_json::to_string_pretty(schema).unwrap_or_else(|_| schema.to_string());
        sections.push(format!(
            "## Required output\n\nWhen done, end the task with `submit <expr>`. The value MUST match this JSON Schema exactly:\n\n```json\n{schema_pretty}\n```"
        ));
    }
    sections.join("\n\n")
}

pub(crate) fn required_string(args: &Value, key: &str) -> Result<String, String> {
    args.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| format!("missing required parameter: {key}"))
}

pub(crate) fn turn_input_for_task(text: String) -> TurnInput {
    TurnInput {
        items: vec![InputItem::Text { text }],
        image_blobs: std::collections::HashMap::new(),
        protocol_turn_options: None,
        trace_turn_id: None,
        protocol_extension: None,
        turn_context: lash_core::TurnContext::default(),
    }
}

pub(crate) fn capability_list_for_description(capability_names: &[String]) -> String {
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

pub(crate) fn example_capability_name(capability_names: &[String]) -> String {
    capability_names
        .iter()
        .find(|name| name.as_str() == "explore")
        .or_else(|| capability_names.first())
        .cloned()
        .unwrap_or_else(|| "explore".to_string())
}

pub(crate) fn tool_definition(
    name: &str,
    description: impl Into<String>,
    input_schema: Value,
    examples: Vec<String>,
    execution_mode: ToolScheduling,
) -> ToolDefinition {
    ToolDefinition::raw(
        format!("tool:{name}"),
        name,
        description,
        input_schema,
        json!({ "type": "object", "additionalProperties": true }),
    )
    .with_examples(examples)
    .with_scheduling(execution_mode)
}

pub(crate) fn spawn_agent_input_schema(capability_names: &[String]) -> Value {
    let enum_values: Vec<Value> = capability_names
        .iter()
        .map(|name| Value::String(name.clone()))
        .collect();
    let mut required = vec!["task"];
    if capability_names.len() != 1 {
        required.push("capability");
    }
    json!({
        "type": "object",
        "properties": {
            "task": { "type": "string" },
            "capability": { "type": "string", "enum": enum_values },
            "output": { "type": "object", "additionalProperties": true },
            "seed": {
                "type": "object",
                "additionalProperties": true,
                "description": "Optional record of state to seed into the child. Each entry's kind is preserved automatically: if its lashlang source root is a host-projected binding (e.g. `seed: { problem: input.prompt }`), the child receives it as a read-only projected binding; otherwise it lands as a regular RLM global. Computed values default to global. Children inherit nothing else from the parent — pass everything they need explicitly."
            }
        },
        "required": required,
        "additionalProperties": false
    })
}

/// Tools that are inert in any subagent session because there is no
/// human attached to the session: prompts return nothing, UI surfaces
/// have no thread to render against. Hidden universally on top of
/// the capability's explicit surface. `update_plan` is
/// already gated to root-only by `UpdatePlanPluginFactory`, so it is
/// not listed here.
pub(crate) const SUBAGENT_SUITE_DENY: &[&str] = &["spawn_agent"];

pub(crate) const MAX_SUBAGENT_DEPTH: u8 = 5;

pub(crate) fn submit_error_tool_definition() -> ToolDefinition {
    ToolDefinition::raw(
        "tool:submit_error",
        "submit_error",
        "End the current subagent task as a terminal failure with a concise reason. Use this when the child cannot produce a valid result.",
        json!({
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Failure reason returned to the parent agents.spawn call."
                }
            },
            "required": ["reason"],
            "additionalProperties": false
        }),
        json!({ "type": "object", "additionalProperties": true }),
    )
    .with_agent_surface(lash_core::ToolAgentSurface::new(["tools"], "submit_error"))
    .with_scheduling(ToolScheduling::Serial)
}

pub(crate) fn submit_error_tool_result(args: &Value) -> ToolResult {
    ToolResult::ok(args.clone()).with_control(lash_core::ToolControl::Fail {
        failure: lash_core::ToolFailure::tool(
            lash_core::ToolFailureClass::Execution,
            "subagent_submit_error",
            args.to_string(),
        ),
    })
}

/// Apply the spawned subagent's tool authority as a tool surface override.
pub(crate) fn subagent_surface_contribution(
    ctx: lash_core::plugin::ToolSurfaceContext,
) -> Result<ToolSurfaceContribution, PluginError> {
    let Some(authority) = ctx.subagent else {
        return Ok(ToolSurfaceContribution::default());
    };
    Ok(ToolSurfaceContribution {
        tool_list_notes: vec![format!(
            "Subagent capability: {}. Depth: {}/{}.",
            authority.capability, authority.depth, authority.max_depth
        )],
        ..Default::default()
    })
}

pub(crate) fn task_result_value(turn: &AssembledTurn) -> Value {
    match &turn.outcome {
        TurnOutcome::Finished(TurnFinish::SubmittedValue { value }) => return value.clone(),
        TurnOutcome::Finished(TurnFinish::ToolValue { value, .. }) => return value.clone(),
        TurnOutcome::Finished(TurnFinish::AssistantMessage { text }) => {
            if !text.trim().is_empty() {
                return json!(text.trim().to_string());
            }
        }
        TurnOutcome::Stopped(TurnStop::SubmittedError { value }) => return value.clone(),
        TurnOutcome::Stopped(TurnStop::ToolError { value, .. }) => return value.clone(),
        TurnOutcome::Handoff { session_id } => return json!({ "session_id": session_id }),
        TurnOutcome::Stopped(_) => {}
    }
    if !turn.assistant_output.safe_text.trim().is_empty() {
        return json!(turn.assistant_output.safe_text.trim().to_string());
    }
    json!(turn.assistant_output.raw_text.trim().to_string())
}

/// Wrap an `Ok`/`Err` result as a `ToolResult`. Used by both providers'
/// `execute` so error encoding stays identical.
pub(crate) fn finalise_tool_result(result: Result<Value, String>) -> ToolResult {
    match result {
        Ok(value) => ToolResult::ok(value),
        Err(err) => ToolResult::err(json!(err)),
    }
}
