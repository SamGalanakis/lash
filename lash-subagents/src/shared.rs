//! Shared helpers used by both `StandardSubagentToolsProvider` and
//! `RlmSubagentToolsProvider`.
//!
//! Anything that is genuinely mode-agnostic — argument parsing, schema
//! validation, child-session request shaping, capability lookup,
//! tool-surface authority resolution, output-schema rendering — lives here.
//! Per-mode prose, tool descriptions, and dispatch live in `standard.rs`
//! and `rlm.rs`.

use lash::plugin::PluginError;
use lash::{
    InputItem, ModeExtras, SessionCreateRequest, SessionPluginMode, SessionPolicy,
    SessionStartPoint, SessionToolAccess, ToolContext, ToolDefinition, ToolExecutionMode,
    ToolResult, ToolSurfaceContribution, TurnInput,
};
use lash_rlm_types::{ClassifiedSeed, RlmTermination, unwrap_projected_arg};
use serde_json::{Value, json};

use crate::capability::{
    CapabilityContext, CapabilityField, CapabilityOptionalField, CapabilityRegistry, CapabilitySpec,
};
use crate::{SubagentSessionConfigurator, SubagentSpawnContext};

pub(crate) fn fresh_child_request(
    parent_session_id: String,
    start: SessionStartPoint,
    policy: SessionPolicy,
    mode_extras: ModeExtras,
    usage_source: impl Into<String>,
) -> SessionCreateRequest {
    SessionCreateRequest {
        session_id: Some(uuid::Uuid::new_v4().to_string()),
        relation: lash::SessionRelation::Child { parent_session_id },
        start,
        policy: Some(policy),
        plugin_mode: SessionPluginMode::Fresh,
        initial_nodes: Vec::new(),
        first_turn_input: None,
        tool_access: SessionToolAccess::default(),
        subagent: None,
        context_surface: lash::SessionContextSurface::default(),
        mode_extras,
        usage_source: Some(usage_source.into()),
    }
}

#[cfg(test)]
pub(crate) fn build_session_policy(
    registry: &CapabilityRegistry,
    current_policy: &SessionPolicy,
    capability_name: &str,
) -> Result<SessionPolicy, String> {
    let spec = resolve_capability_spec(registry, current_policy, capability_name)?;
    Ok(session_policy_from_spec(&spec, current_policy))
}

pub(crate) fn resolve_capability_spec(
    registry: &CapabilityRegistry,
    current_policy: &SessionPolicy,
    capability_name: &str,
) -> Result<CapabilitySpec, String> {
    let capability = registry
        .get(capability_name)
        .ok_or_else(|| unknown_capability_message(capability_name, registry))?;
    Ok(capability.resolve(&CapabilityContext {
        parent_policy: current_policy,
    }))
}

fn session_policy_from_spec(
    spec: &CapabilitySpec,
    current_policy: &SessionPolicy,
) -> SessionPolicy {
    SessionPolicy {
        model: resolve_field(&spec.model, &current_policy.model),
        model_variant: resolve_optional_field(&spec.model_variant, &current_policy.model_variant),
        provider: current_policy.provider.clone(),
        max_context_tokens: current_policy.max_context_tokens,
        max_turns: None,
        execution_mode: resolve_field(&spec.execution_mode, &current_policy.execution_mode),
        ..Default::default()
    }
}

fn resolve_field<T: Clone>(field: &CapabilityField<T>, inherited: &T) -> T {
    match field {
        CapabilityField::Inherit => inherited.clone(),
        CapabilityField::Set(value) => value.clone(),
    }
}

fn resolve_optional_field<T: Clone>(
    field: &CapabilityOptionalField<T>,
    inherited: &Option<T>,
) -> Option<T> {
    match field {
        CapabilityOptionalField::Inherit => inherited.clone(),
        CapabilityOptionalField::Set(value) => Some(value.clone()),
        CapabilityOptionalField::Clear => None,
    }
}

pub(crate) fn normalize_context_policy(policy: &mut SessionPolicy) {
    if policy.execution_mode != lash::ExecutionMode::standard() {
        policy.standard_context_approach = None;
    }
}

pub(crate) async fn build_spawn_create_request(
    registry: &CapabilityRegistry,
    context: &ToolContext,
    capability_name: &str,
    output_schema: Option<Value>,
    seed: ClassifiedSeed,
    configurator: &dyn SubagentSessionConfigurator,
) -> Result<SessionCreateRequest, String> {
    let current_snapshot = context
        .host()
        .snapshot_session(context.session_id())
        .await
        .map_err(|err| format!("failed to snapshot current session: {err}"))?;
    let spec = resolve_capability_spec(registry, &current_snapshot.policy, capability_name)?;
    let mut policy = session_policy_from_spec(&spec, &current_snapshot.policy);
    if output_schema.is_some() && policy.execution_mode != lash::ExecutionMode::new("rlm") {
        return Err(format!(
            "structured output is RLM-only; child capability `{capability_name}` runs in `{}` mode",
            policy.execution_mode.plugin_id()
        ));
    }
    if !seed.projected.is_empty() && policy.execution_mode != lash::ExecutionMode::new("rlm") {
        return Err(format!(
            "projected seed is RLM-only; child capability `{capability_name}` runs in `{}` mode",
            policy.execution_mode.plugin_id()
        ));
    }
    let mut mode_extras = ModeExtras::default();
    if policy.execution_mode == lash::ExecutionMode::new("rlm") {
        let termination = match output_schema.clone() {
            Some(schema) => RlmTermination::SubmitRequired {
                schema: Some(schema),
            },
            None => RlmTermination::default(),
        };
        let projected_seed = if seed.projected.is_empty() {
            None
        } else {
            Some(seed.projected.clone())
        };
        mode_extras = ModeExtras::typed(
            lash::ExecutionMode::new("rlm"),
            lash_rlm_types::RlmCreateExtras {
                termination,
                projected_seed,
            },
        )
        .map_err(|err| format!("failed to encode rlm mode extras: {err}"))?;
    }
    normalize_context_policy(&mut policy);

    let mut request = fresh_child_request(
        context.session_id().to_string(),
        SessionStartPoint::Empty,
        policy,
        mode_extras,
        "subagent",
    );
    request.tool_access = SessionToolAccess::default();
    if !seed.globals.is_empty() {
        request.initial_nodes = lash_mode_rlm::rlm_seed_initial_nodes(seed.globals);
    }
    let child_policy = request.policy.as_ref().expect("child policy set").clone();
    configurator.configure(
        &SubagentSpawnContext {
            parent_session_id: context.session_id(),
            capability: capability_name,
            parent_policy: &current_snapshot.policy,
            child_policy: &child_policy,
        },
        &mut request,
    )?;
    Ok(request)
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
        .map(unwrap_projected_arg)
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
        mode: None,
        mode_turn_options: None,
        trace_turn_id: None,
        mode_extension: None,
        turn_context: lash::TurnContext::default(),
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
    execution_mode: ToolExecutionMode,
) -> ToolDefinition {
    ToolDefinition::raw(
        name,
        description,
        input_schema,
        json!({ "type": "object", "additionalProperties": true }),
    )
    .with_examples(examples)
    .with_execution_mode(execution_mode)
}

pub(crate) fn spawn_agent_input_schema(capability_names: &[String]) -> Value {
    let enum_values: Vec<Value> = capability_names
        .iter()
        .map(|name| Value::String(name.clone()))
        .collect();
    json!({
        "type": "object",
        "properties": {
            "agent_name": { "type": "string" },
            "task": { "type": "string" },
            "capability": { "type": "string", "enum": enum_values },
            "output": { "type": "object", "additionalProperties": true },
            "seed": {
                "type": "object",
                "additionalProperties": true,
                "description": "Optional record of state to seed into the child. Each entry's kind is preserved automatically: if its lashlang source root is a host-projected binding (e.g. `seed: { problem: input.prompt }`), the child receives it as a read-only projected binding; otherwise it lands as a regular RLM global. Computed values default to global. Children inherit nothing else from the parent — pass everything they need explicitly."
            }
        },
        "required": ["agent_name", "task", "capability"],
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
        "submit_error",
        "End the current subagent task as a terminal failure with a concise reason. Use `call submit_error { reason: \"...\" }` when the child cannot produce a valid result.",
        json!({
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Failure reason returned to the parent spawn_agent call."
                }
            },
            "required": ["reason"],
            "additionalProperties": false
        }),
        json!({ "type": "object", "additionalProperties": true }),
    )
    .with_execution_mode(ToolExecutionMode::Serial)
}

pub(crate) fn submit_error_tool_result(args: &Value) -> ToolResult {
    ToolResult::ok(args.clone()).with_control(lash::ToolControl::Fail {
        value: args.clone(),
    })
}

/// Apply the spawned subagent's tool authority as a tool surface override.
pub(crate) fn subagent_surface_contribution(
    ctx: lash::plugin::ToolSurfaceContext,
) -> Result<ToolSurfaceContribution, PluginError> {
    let Some(authority) = ctx.subagent else {
        return Ok(ToolSurfaceContribution::default());
    };
    Ok(ToolSurfaceContribution {
        tool_list_notes: vec![format!(
            "Subagent agent_name: {}. Capability: {}.",
            authority.agent_name, authority.capability
        )],
        ..Default::default()
    })
}

/// Wrap an `Ok`/`Err` result as a `ToolResult`. Used by both providers'
/// `execute` so error encoding stays identical.
pub(crate) fn finalise_tool_result(result: Result<Value, String>) -> ToolResult {
    match result {
        Ok(value) => ToolResult::ok(value),
        Err(err) => ToolResult::err(json!(err)),
    }
}
