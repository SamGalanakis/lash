//! Shared helpers used by both `StandardSubagentToolsProvider` and
//! `RlmSubagentToolsProvider`.
//!
//! Anything that is genuinely mode-agnostic — argument parsing, schema
//! validation, child-session request shaping, capability lookup,
//! tool-surface deny resolution, output-schema rendering — lives here.
//! Per-mode prose, tool descriptions, and dispatch live in `standard.rs`
//! and `rlm.rs`.

use std::collections::HashSet;

use lash::plugin::PluginError;
use lash::session_model::{ModeEvent, SessionEventRecord};
use lash::{
    InputItem, ModeExtras, SessionAppendNode, SessionCreateRequest, SessionPluginMode,
    SessionPolicy, SessionStartPoint, SessionToolAccess, ToolDefinition, ToolExecutionContext,
    ToolExecutionMode, ToolProvider, ToolResult, ToolSurfaceContribution, TurnInput,
};
use lash_rlm_types::RlmTermination;
use serde_json::{Value, json};

use crate::capability::{CapabilityContext, CapabilityRegistry};

pub(crate) fn fresh_child_request(
    parent_session_id: String,
    start: SessionStartPoint,
    policy: SessionPolicy,
    mode_extras: ModeExtras,
    usage_source: impl Into<String>,
) -> SessionCreateRequest {
    SessionCreateRequest {
        session_id: Some(uuid::Uuid::new_v4().to_string()),
        parent_session_id: Some(parent_session_id),
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

pub(crate) fn build_session_policy(
    registry: &CapabilityRegistry,
    current_policy: &SessionPolicy,
    capability_name: &str,
) -> Result<SessionPolicy, String> {
    let capability = registry
        .get(capability_name)
        .ok_or_else(|| unknown_capability_message(capability_name, registry))?;
    // Subagent denylist semantics today never read the parent's denylist
    // (TierCapability returns DenyList::Replace), so passing an empty
    // parent set is correct. When a future Capability uses
    // DenyList::AddTo, this is the right hook to thread the parent's
    // actual denylist through.
    let parent_denied: HashSet<String> = HashSet::new();
    let spec = capability.resolve(&CapabilityContext {
        parent_policy: current_policy,
        parent_denied_tools: &parent_denied,
    });
    Ok(SessionPolicy {
        model: spec
            .model
            .clone()
            .unwrap_or_else(|| current_policy.model.clone()),
        model_variant: spec
            .model_variant
            .clone()
            .or_else(|| current_policy.model_variant.clone()),
        provider: current_policy.provider.clone(),
        max_context_tokens: current_policy.max_context_tokens,
        max_turns: None,
        execution_mode: spec
            .execution_mode
            .unwrap_or_else(|| current_policy.execution_mode.clone()),
        ..Default::default()
    })
}

pub(crate) fn normalize_context_policy(policy: &mut SessionPolicy) {
    if policy.execution_mode != lash::ExecutionMode::standard() {
        policy.standard_context_approach = None;
    }
}

pub(crate) async fn build_spawn_create_request(
    registry: &CapabilityRegistry,
    context: &ToolExecutionContext,
    capability_name: &str,
    output_schema: Option<Value>,
) -> Result<SessionCreateRequest, String> {
    let current_snapshot = context
        .host
        .snapshot_session(&context.session_id)
        .await
        .map_err(|err| format!("failed to snapshot current session: {err}"))?;
    let mut policy = build_session_policy(registry, &current_snapshot.policy, capability_name)?;
    if capability_name != "explore" && output_schema.is_some() {
        policy.execution_mode = lash::ExecutionMode::new("rlm");
    }
    let mut mode_extras = ModeExtras::default();
    if policy.execution_mode == lash::ExecutionMode::new("rlm") {
        let termination = match output_schema.clone() {
            Some(schema) => RlmTermination::Finish {
                schema: Some(schema),
                include_submit_prompt: true,
            },
            None => RlmTermination::default(),
        };
        mode_extras = ModeExtras::typed(
            lash::ExecutionMode::new("rlm"),
            lash_rlm_types::RlmCreateExtras { termination },
        )
        .map_err(|err| format!("failed to encode rlm mode extras: {err}"))?;
    }
    normalize_context_policy(&mut policy);

    let hidden_tools = resolve_hidden_tools(registry, &policy, capability_name, 0)?;
    let mut request = fresh_child_request(
        context.session_id.clone(),
        SessionStartPoint::Empty,
        policy,
        mode_extras,
        "subagent",
    );
    request.tool_access = SessionToolAccess {
        tools: if capability_name == "explore" {
            explore_tools(output_schema)
        } else {
            Vec::new()
        },
        hidden_tools: hidden_tools.into_iter().collect(),
    };
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

pub(crate) fn parse_output_schema(value: Option<&Value>) -> Result<Option<Value>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let output = value.as_object().ok_or_else(|| {
        "invalid `output`: expected a record describing the typed shape".to_string()
    })?;
    if output.is_empty() {
        return Err("at least one output field is required".to_string());
    }

    // Fast path: lashlang `Type { ... }` literals are delivered wrapped with a
    // single `$lash_type` key whose value is already a JSON Schema. Pass it
    // through directly so richer types (enums, nested records, optional fields,
    // `list[<composite>]`) are preserved end-to-end.
    if output.len() == 1
        && let Some(schema) = output.get(lashlang::LASH_TYPE_KEY)
    {
        validate_schema(schema)?;
        return Ok(Some(schema.clone()));
    }

    let mut properties = serde_json::Map::new();
    let mut required = Vec::new();
    for (name, descriptor) in output {
        let type_str = descriptor
            .as_str()
            .ok_or_else(|| format!("field `{name}`: type descriptor must be a string"))?;
        properties.insert(name.clone(), type_descriptor_to_json_schema(type_str)?);
        required.push(Value::String(name.clone()));
    }
    Ok(Some(json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false,
    })))
}

/// Lightweight sanity check for a JSON Schema handed to us by the runtime.
/// We only reject shapes that would produce a useless prompt; the provider's
/// structured-output path does the full enforcement.
fn validate_schema(schema: &Value) -> Result<(), String> {
    let object = schema
        .as_object()
        .ok_or_else(|| "Type schema must be a JSON object".to_string())?;
    let kind = object
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| "Type schema missing `type` field".to_string())?;
    match kind {
        "object" | "array" | "string" | "integer" | "number" | "boolean" => Ok(()),
        other => Err(format!("unsupported Type schema kind `{other}`")),
    }
}

fn type_descriptor_to_json_schema(descriptor: &str) -> Result<Value, String> {
    let scalar = |ty: &str| -> Result<Value, String> {
        match ty {
            "str" | "string" => Ok(json!({"type": "string"})),
            "int" | "integer" => Ok(json!({"type": "integer"})),
            "float" | "number" => Ok(json!({"type": "number"})),
            "bool" | "boolean" => Ok(json!({"type": "boolean"})),
            "record" | "dict" | "object" => {
                Ok(json!({"type": "object", "additionalProperties": true}))
            }
            other => Err(format!("unknown scalar type `{other}`")),
        }
    };
    let trimmed = descriptor.trim();
    if let Some(inner) = trimmed
        .strip_prefix("list[")
        .and_then(|rest| rest.strip_suffix(']'))
    {
        return Ok(json!({
            "type": "array",
            "items": scalar(inner.trim())?,
        }));
    }
    scalar(trimmed)
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
        user_input: None,
        mode: None,
        mode_turn_options: None,
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
    ToolDefinition::new(
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
            "output": { "type": "object", "additionalProperties": true }
        },
        "required": ["agent_name", "task", "capability"],
        "additionalProperties": false
    })
}

pub(crate) fn llm_query_input_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "task": { "type": "string" },
            "inputs": {},
            "output": { "type": "object", "additionalProperties": true }
        },
        "required": ["task"],
        "additionalProperties": false
    })
}

/// Tools that are inert in any subagent session because there is no
/// human attached to the session: prompts return nothing, UI surfaces
/// have no thread to render against. Hidden universally on top of
/// whatever the capability's denylist resolves to. `update_plan` is
/// already gated to root-only by `UpdatePlanPluginFactory`, so it is
/// not listed here.
pub(crate) const SUBAGENT_INTERACTIVE_DENY: &[&str] =
    &["ask", "show_snippet_to_user", "showcase", "plan_exit"];

pub(crate) const SUBAGENT_SUITE_DENY: &[&str] = &["spawn_agent"];

pub(crate) const MAX_SUBAGENT_DEPTH: u8 = 5;

pub(crate) fn explore_tools(output_schema: Option<Value>) -> Vec<ToolDefinition> {
    let mut tools = Vec::new();
    tools.extend(lash::tools::ReadFile::new().definitions());
    tools.extend(lash::tools::Grep::new().definitions());
    tools.extend(lash::tools::WebSearch::new("").definitions());
    tools.extend(lash::tools::FetchUrl::new("").definitions());
    tools.extend(lash::tools::StandardShell::new().definitions());
    tools.retain(|tool| {
        matches!(
            tool.name.as_str(),
            "read_file"
                | "grep"
                | "search_web"
                | "fetch_url"
                | "exec_command"
                | "start_command"
                | "write_stdin"
        )
    });
    let _ = output_schema;
    tools.push(llm_query_tool_definition());
    tools.push(continue_as_tool_definition());
    tools
}

pub(crate) fn llm_query_tool_definition() -> ToolDefinition {
    tool_definition(
        "llm_query",
        "Run one lightweight LLM call and return its result. Use this for semantic extraction, summarization, classification, judging, or transforming data already available in variables. It does not create a child session, cannot use tools, and does not run a REPL loop. `inputs` can be any structured value. `output` is optional and defaults to a string; when present, it accepts the same record descriptors and `Type { ... }` literals as `spawn_agent`.",
        llm_query_input_schema(),
        Vec::new(),
        ToolExecutionMode::Parallel,
    )
    .with_output_from_input_schema("output", Some(json!({ "type": "string" })))
}

pub(crate) fn submit_error_tool_definition() -> ToolDefinition {
    ToolDefinition::new(
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
    ToolResult::ok(args.clone())
}

pub(crate) fn continue_as_tool_definition() -> ToolDefinition {
    tool_definition(
        "continue_as",
        "Tail-call: end this session and continue the work as a fresh successor with the same tools and a clean window. Use when the current trajectory is stale or context budget is tight; pack only what the successor needs into `task` and `seed`.",
        continue_as_input_schema(),
        vec![
            r#"call continue_as { task: "continue the audit from the summarized findings", seed: { findings: findings } }"#.into(),
        ],
        ToolExecutionMode::Parallel,
    )
}

pub(crate) fn continue_as_input_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Task for the successor session."
            },
            "seed": {
                "type": "object",
                "additionalProperties": true,
                "description": "Optional record/dict of concrete state for the successor."
            }
        },
        "required": ["task"],
        "additionalProperties": false
    })
}

pub(crate) fn resolve_hidden_tools(
    registry: &CapabilityRegistry,
    current_policy: &SessionPolicy,
    capability_name: &str,
    child_depth: u8,
) -> Result<HashSet<String>, String> {
    let capability = registry
        .get(capability_name)
        .ok_or_else(|| unknown_capability_message(capability_name, registry))?;
    let parent_denied: HashSet<String> = HashSet::new();
    let spec = capability.resolve(&CapabilityContext {
        parent_policy: current_policy,
        parent_denied_tools: &parent_denied,
    });
    let mut denied = spec.denied_tools.apply(&parent_denied);
    denied.extend(
        SUBAGENT_INTERACTIVE_DENY
            .iter()
            .map(|name| name.to_string()),
    );
    if capability_name == "explore" || child_depth >= MAX_SUBAGENT_DEPTH {
        denied.extend(SUBAGENT_SUITE_DENY.iter().map(|name| name.to_string()));
    }
    Ok(denied)
}

/// Apply the spawned subagent's denied-tool list as a tool surface
/// override. Shared between the standard and rlm provider builds.
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

/// Build a fresh `RlmGlobalsPatch` initial-node sequence for `continue_as`
/// successors that need to seed RLM globals. Returned as a `Vec` so the
/// caller can pass it straight into the create request.
pub(crate) fn rlm_seed_initial_nodes(
    seed: serde_json::Map<String, Value>,
) -> Vec<SessionAppendNode> {
    if seed.is_empty() {
        return Vec::new();
    }
    vec![SessionAppendNode::event(SessionEventRecord::Mode(
        ModeEvent::rlm(lash_rlm_types::RlmModeEvent::RlmGlobalsPatch(
            lash_rlm_types::RlmGlobalsPatchPluginBody {
                set: seed,
                unset: Vec::new(),
            },
        )),
    ))]
}

/// Wrap an `Ok`/`Err` result as a `ToolResult`. Used by both providers'
/// `execute_with_context` so error encoding stays identical.
pub(crate) fn finalise_tool_result(result: Result<Value, String>) -> ToolResult {
    match result {
        Ok(value) => ToolResult::ok(value),
        Err(err) => ToolResult::err(json!(err)),
    }
}
