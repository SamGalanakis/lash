mod capability;
mod host;
mod local;
mod policy;
mod queue;
mod routing;
mod types;

pub use capability::{
    Capability, CapabilityContext, CapabilityRegistry, CapabilitySpec, DenyList, TierCapability,
    TierExecutionMode, default_registry,
};

use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use lash::plugin::{PluginError, PluginFactory, PluginSessionContext, ToolSurfaceOverride};
use lash::session_model::{ModeEvent, SessionEventRecord};
use lash::{
    InputItem, MessageRole, ModeExtras, PluginMessage, PluginSpec, PluginSpecFactory,
    ProgressSender, SessionAppendNode, SessionCreateRequest, SessionPluginMode, SessionPolicy,
    SessionStartPoint, ToolDefinition, ToolExecutionContext, ToolProvider, ToolResult,
    ToolSurfaceContribution, TurnInput,
};
use lash_rlm_types::{RlmCreateExtras, RlmGlobalsPatchPluginBody, RlmModeEvent, RlmTermination};
use serde_json::{Value, json};

pub use host::{
    AgentSummary, CloseAgentRequest, CloseAgentResponse, DeliveryMode, FollowupTaskRequest,
    FollowupTaskResponse, ListAgentsRequest, ListAgentsResponse, LocalSubagentHost,
    SendMessageRequest, SendMessageResponse, SessionAgentInfo, SpawnAgentRequest,
    SpawnAgentResponse, SubagentHost, WaitAgentClosed, WaitAgentCompletion, WaitAgentEvent,
    WaitAgentMessage, WaitAgentRequest, WaitAgentResponse, WaitAgentSessionSummary, WaitUntil,
    truncate_snapshot_to_recent_turns,
};
use policy::{subagent_prompt_contributions, subagent_tool_definitions};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ForkTurns {
    None,
    All,
    Recent(usize),
}

struct SubagentToolsProvider {
    execution_mode: lash::ExecutionMode,
    registry: Arc<CapabilityRegistry>,
    host: Arc<dyn SubagentHost>,
}

impl SubagentToolsProvider {
    fn fresh_child_request(
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
            context_surface: lash::SessionContextSurface::default(),
            mode_extras,
            usage_source: Some(usage_source.into()),
        }
    }

    fn finish_mode_extras(
        mode: &lash::ExecutionMode,
        termination: RlmTermination,
    ) -> Result<ModeExtras, String> {
        match mode.plugin_id() {
            "rlm" => ModeExtras::typed(
                lash::ExecutionMode::new("rlm"),
                RlmCreateExtras { termination },
            )
            .map_err(|err| format!("failed to encode rlm mode extras: {err}")),
            other => Err(format!("unsupported RLM finish mode extras for {other}")),
        }
    }

    fn build_session_policy(
        &self,
        current_policy: &SessionPolicy,
        capability_name: &str,
    ) -> Result<SessionPolicy, String> {
        let capability = self
            .registry
            .get(capability_name)
            .ok_or_else(|| unknown_capability_message(capability_name, &self.registry))?;
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

    async fn build_spawn_create_request(
        &self,
        context: &ToolExecutionContext,
        capability_name: &str,
        fork_turns: ForkTurns,
        output_schema: Option<Value>,
    ) -> Result<SessionCreateRequest, String> {
        let current_snapshot = context
            .host
            .snapshot_session(&context.session_id)
            .await
            .map_err(|err| format!("failed to snapshot current session: {err}"))?;
        let mut policy = self.build_session_policy(&current_snapshot.policy, capability_name)?;
        let mut mode_extras = ModeExtras::default();
        if let Some(schema) = output_schema.clone() {
            let termination = RlmTermination::Finish {
                schema: Some(schema),
            };
            policy.execution_mode = lash::ExecutionMode::new("rlm");
            mode_extras = Self::finish_mode_extras(&policy.execution_mode, termination)?;
        }
        let start = match fork_turns {
            ForkTurns::None => SessionStartPoint::Empty,
            ForkTurns::All => SessionStartPoint::ExistingSession {
                session_id: context.session_id.clone(),
            },
            ForkTurns::Recent(turns) => SessionStartPoint::Snapshot {
                snapshot: Box::new(truncate_snapshot_to_recent_turns(current_snapshot, turns)),
            },
        };

        Ok(Self::fresh_child_request(
            context.session_id.clone(),
            start,
            policy,
            mode_extras,
            "subagent",
        ))
    }

    async fn spawn_agent(
        &self,
        args: &Value,
        context: &ToolExecutionContext,
    ) -> Result<Value, String> {
        let task_name = required_string(args, "task_name")?;
        let task = required_string(args, "task")?;
        let capability_name = required_string(args, "capability")?;
        if self.registry.get(&capability_name).is_none() {
            return Err(unknown_capability_message(&capability_name, &self.registry));
        }
        let fork_turns = parse_fork_turns(args.get("fork_turns"))?;
        let output_schema = parse_output_schema(args.get("output"))?;
        let create_request = self
            .build_spawn_create_request(
                context,
                &capability_name,
                fork_turns,
                output_schema.clone(),
            )
            .await?;
        let turn_input = TurnInput {
            items: vec![InputItem::Text {
                text: render_task_prompt(&task, output_schema.as_ref()),
            }],
            image_blobs: std::collections::HashMap::new(),
            user_input: None,
            mode: None,
            mode_turn_options: None,
        };
        let response = self
            .host
            .spawn_agent(
                context,
                SpawnAgentRequest {
                    task_name,
                    task,
                    capability: capability_name,
                    create_request,
                    turn_input,
                },
            )
            .await?;
        serde_json::to_value(response).map_err(|err| err.to_string())
    }

    async fn pass_baton(
        &self,
        args: &Value,
        context: &ToolExecutionContext,
    ) -> Result<Value, String> {
        if self.execution_mode != lash::ExecutionMode::new("rlm") {
            return Err("pass_baton is only available in rlm mode".to_string());
        }

        let task = required_string(args, "task")?;
        let seed = match args.get("seed") {
            None | Some(Value::Null) => serde_json::Map::new(),
            Some(Value::Object(map)) => map.clone(),
            Some(_) => return Err("pass_baton `seed` must be a record/dict".to_string()),
        };

        let current_snapshot = context
            .host
            .snapshot_session(&context.session_id)
            .await
            .map_err(|err| format!("failed to snapshot current session: {err}"))?;
        let termination = current_snapshot.mode_turn_options.rlm_termination();
        let mut policy = current_snapshot.policy.clone();
        policy.execution_mode = lash::ExecutionMode::new("rlm");

        let mut initial_nodes = Vec::new();
        if !seed.is_empty() {
            initial_nodes.push(SessionAppendNode::event(SessionEventRecord::Mode(
                ModeEvent::rlm(RlmModeEvent::RlmGlobalsPatch(RlmGlobalsPatchPluginBody {
                    set: seed,
                    unset: Vec::new(),
                })),
            )));
        }

        let mode_extras = Self::finish_mode_extras(&policy.execution_mode, termination)?;
        let mut request = Self::fresh_child_request(
            context.session_id.clone(),
            SessionStartPoint::Empty,
            policy,
            mode_extras,
            "baton",
        );
        let successor_session_id = request
            .session_id
            .clone()
            .expect("fresh child request sets session id");
        request.initial_nodes = initial_nodes;
        request.first_turn_input = Some(PluginMessage::text(MessageRole::User, task.clone()));
        context
            .host
            .create_session(request)
            .await
            .map_err(|err| format!("failed to create baton successor: {err}"))?;

        Ok(json!({
            "ok": true,
            "_baton": successor_session_id,
            "task": task,
        }))
    }

    async fn send_message(
        &self,
        args: &Value,
        context: &ToolExecutionContext,
    ) -> Result<Value, String> {
        let response = self
            .host
            .send_message(
                context,
                SendMessageRequest {
                    target: required_string(args, "target")?,
                    message: required_string(args, "message")?,
                    delivery: DeliveryMode::parse(args.get("delivery").and_then(Value::as_str))?,
                },
            )
            .await?;
        serde_json::to_value(response).map_err(|err| err.to_string())
    }

    async fn followup_task(
        &self,
        args: &Value,
        context: &ToolExecutionContext,
    ) -> Result<Value, String> {
        let task = required_string(args, "task")?;
        let output_schema = parse_output_schema(args.get("output"))?;
        // Always set an override for follow-ups so the per-turn
        // termination contract matches what the prompt tells the
        // subagent. When `output` is omitted, the follow-up explicitly
        // drops any inherited schema by running with
        // `ProseWithoutFence`.
        let mode_turn_options = Some(lash::ModeTurnOptions::rlm(match output_schema.clone() {
            Some(schema) => RlmTermination::Finish {
                schema: Some(schema),
            },
            None => RlmTermination::ProseWithoutFence,
        }));
        let response = self
            .host
            .followup_task(
                context,
                FollowupTaskRequest {
                    target: required_string(args, "target")?,
                    turn_input: TurnInput {
                        items: vec![InputItem::Text {
                            text: render_task_prompt_with_kind(
                                &task,
                                output_schema.as_ref(),
                                TaskPromptKind::Followup,
                            ),
                        }],
                        image_blobs: std::collections::HashMap::new(),
                        user_input: None,
                        mode: None,
                        mode_turn_options,
                    },
                    task,
                    delivery: DeliveryMode::parse(args.get("delivery").and_then(Value::as_str))?,
                },
            )
            .await?;
        serde_json::to_value(response).map_err(|err| err.to_string())
    }

    async fn wait_agent(
        &self,
        args: &Value,
        context: &ToolExecutionContext,
    ) -> Result<Value, String> {
        let response = self
            .host
            .wait_agent(
                context,
                WaitAgentRequest {
                    targets: optional_string_list(args, "targets")?,
                    until: WaitUntil::parse(args.get("until").and_then(|value| value.as_str()))?,
                    timeout_ms: optional_u64(args, "timeout_ms")?,
                },
            )
            .await?;
        serde_json::to_value(response).map_err(|err| err.to_string())
    }

    async fn close_agent(
        &self,
        args: &Value,
        context: &ToolExecutionContext,
    ) -> Result<Value, String> {
        let response = self
            .host
            .close_agent(
                context,
                CloseAgentRequest {
                    target: required_string(args, "target")?,
                },
            )
            .await?;
        serde_json::to_value(response).map_err(|err| err.to_string())
    }

    async fn list_agents(
        &self,
        args: &Value,
        context: &ToolExecutionContext,
    ) -> Result<Value, String> {
        let response = self
            .host
            .list_agents(
                context,
                ListAgentsRequest {
                    path_prefix: optional_string(args, "path_prefix"),
                },
            )
            .await?;
        serde_json::to_value(response).map_err(|err| err.to_string())
    }
}

#[async_trait]
impl ToolProvider for SubagentToolsProvider {
    fn definitions(&self) -> Vec<ToolDefinition> {
        subagent_tool_definitions(self.execution_mode.clone(), &self.registry)
    }

    async fn execute(&self, name: &str, _args: &Value) -> ToolResult {
        ToolResult::err_fmt(format_args!(
            "`{name}` requires session context and cannot run without it"
        ))
    }

    async fn execute_with_context(
        &self,
        name: &str,
        args: &Value,
        context: &ToolExecutionContext,
    ) -> ToolResult {
        let result = match name {
            "spawn_agent" => self.spawn_agent(args, context).await,
            "pass_baton" => self.pass_baton(args, context).await,
            "send_message" => self.send_message(args, context).await,
            "followup_task" => self.followup_task(args, context).await,
            "wait_agent" => self.wait_agent(args, context).await,
            "close_agent" => self.close_agent(args, context).await,
            "list_agents" => self.list_agents(args, context).await,
            _ => Err(format!("Unknown tool: {name}")),
        };
        match result {
            Ok(value) => ToolResult::ok(value),
            Err(err) => ToolResult::err(json!(err)),
        }
    }

    async fn execute_streaming_with_context(
        &self,
        name: &str,
        args: &Value,
        context: &ToolExecutionContext,
        _progress: Option<&ProgressSender>,
    ) -> ToolResult {
        self.execute_with_context(name, args, context).await
    }
}

pub struct SubagentsPluginFactory {
    policy: SessionPolicy,
    registry: Arc<CapabilityRegistry>,
    host: Arc<dyn SubagentHost>,
}

impl SubagentsPluginFactory {
    pub fn new(
        policy: SessionPolicy,
        registry: Arc<CapabilityRegistry>,
        host: Arc<dyn SubagentHost>,
    ) -> Self {
        Self {
            policy,
            registry,
            host,
        }
    }
}

impl PluginFactory for SubagentsPluginFactory {
    fn id(&self) -> &'static str {
        "subagents"
    }

    fn build(
        &self,
        ctx: &PluginSessionContext,
    ) -> Result<Arc<dyn lash::SessionPlugin>, PluginError> {
        let mut policy = self.policy.clone();
        policy.execution_mode = ctx.execution_mode.clone();
        let provider = Arc::new(SubagentToolsProvider {
            execution_mode: ctx.execution_mode.clone(),
            registry: Arc::clone(&self.registry),
            host: Arc::clone(&self.host),
        });
        let prompt_contributions = subagent_prompt_contributions();
        let host = Arc::clone(&self.host);
        let registry = Arc::clone(&self.registry);
        PluginSpecFactory::new(
            "subagents",
            Arc::new(move |_ctx| {
                let provider = Arc::clone(&provider);
                let host = Arc::clone(&host);
                let registry = Arc::clone(&registry);
                let contributions = prompt_contributions.clone();
                Ok(PluginSpec::new()
                    .with_tool_provider(provider as Arc<dyn ToolProvider>)
                    .with_prompt_contributor(Arc::new(move |_ctx| {
                        let contributions = contributions.clone();
                        Box::pin(async move { Ok(contributions) })
                    }))
                    .with_tool_surface_contributor(Arc::new(move |ctx| {
                        subagent_surface_contribution(&host, &registry, ctx)
                    })))
            }),
        )
        .build(ctx)
    }
}

fn subagent_surface_contribution(
    host: &Arc<dyn SubagentHost>,
    registry: &Arc<CapabilityRegistry>,
    ctx: lash::plugin::ToolSurfaceContext,
) -> Result<ToolSurfaceContribution, PluginError> {
    let Some(info) = host.session_info(&ctx.session_id) else {
        return Ok(ToolSurfaceContribution::default());
    };
    let Some(capability_name) = info.capability else {
        return Ok(ToolSurfaceContribution::default());
    };
    let Some(capability) = registry.get(&capability_name) else {
        return Ok(ToolSurfaceContribution::default());
    };
    // Resolve only to read denied_tools; we don't have a parent policy
    // here (we're shaping this subagent's own surface) so use a minimal
    // dummy policy. If a future Capability needs the parent for surface
    // shaping, this is the place to thread it through.
    let dummy_policy = SessionPolicy::default();
    let parent_denied: HashSet<String> = HashSet::new();
    let spec = capability.resolve(&CapabilityContext {
        parent_policy: &dummy_policy,
        parent_denied_tools: &parent_denied,
    });
    let denied = spec.denied_tools.apply(&parent_denied);
    let overrides = ctx
        .tools
        .iter()
        .filter(|tool| denied.contains(tool.name.as_str()))
        .map(|tool| ToolSurfaceOverride {
            tool_name: tool.name.clone(),
            availability: Some(lash::ToolAvailability::Hidden),
        })
        .collect::<Vec<_>>();
    let tool_list_notes = vec![format!(
        "Subagent target: {}. Capability: {}.",
        info.path, capability_name
    )];
    Ok(ToolSurfaceContribution {
        overrides,
        tool_list_notes,
    })
}

fn unknown_capability_message(name: &str, registry: &CapabilityRegistry) -> String {
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

fn parse_fork_turns(value: Option<&Value>) -> Result<ForkTurns, String> {
    let Some(value) = value else {
        return Ok(ForkTurns::None);
    };
    match value {
        Value::String(text) if text == "none" => Ok(ForkTurns::None),
        Value::String(text) if text == "all" => Ok(ForkTurns::All),
        Value::String(text) => text
            .parse::<usize>()
            .ok()
            .filter(|count| *count > 0)
            .map(ForkTurns::Recent)
            .ok_or_else(|| {
                "invalid fork_turns: use `none`, `all`, or a positive integer string".to_string()
            }),
        Value::Number(number) => number
            .as_u64()
            .filter(|count| *count > 0)
            .map(|count| ForkTurns::Recent(count as usize))
            .ok_or_else(|| {
                "invalid fork_turns: use `none`, `all`, or a positive integer".to_string()
            }),
        _ => Err("invalid fork_turns: use `none`, `all`, or a positive integer string".to_string()),
    }
}

fn parse_output_schema(value: Option<&Value>) -> Result<Option<Value>, String> {
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

fn render_task_prompt(task: &str, output_schema: Option<&Value>) -> String {
    render_task_prompt_with_kind(task, output_schema, TaskPromptKind::Initial)
}

#[derive(Clone, Copy)]
enum TaskPromptKind {
    Initial,
    Followup,
}

fn render_task_prompt_with_kind(
    task: &str,
    output_schema: Option<&Value>,
    kind: TaskPromptKind,
) -> String {
    let mut sections = vec![task.to_string()];
    if let Some(schema) = output_schema {
        let schema_pretty =
            serde_json::to_string_pretty(schema).unwrap_or_else(|_| schema.to_string());
        let replaces_note = match kind {
            TaskPromptKind::Initial => "",
            TaskPromptKind::Followup => {
                "\n\nThis replaces any output schema from earlier tasks in this session."
            }
        };
        sections.push(format!(
            "## Required output\n\nWhen done, end the task by calling `submit <expr>` from inside a fenced ```lashlang block. The value MUST match this JSON Schema exactly:\n\n```json\n{schema_pretty}\n```{replaces_note}"
        ));
    } else if matches!(kind, TaskPromptKind::Followup) {
        sections.push(
            "## Required output\n\nNo structured output schema is required for this follow-up task. `submit <expr>` may return any value (a string, a record, etc.); it will not be schema-validated. This overrides any output schema from earlier tasks in this session.".to_string(),
        );
    }
    sections.join("\n\n")
}

fn required_string(args: &Value, key: &str) -> Result<String, String> {
    args.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| format!("missing required parameter: {key}"))
}

fn optional_string(args: &Value, key: &str) -> Option<String> {
    args.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn optional_string_list(args: &Value, key: &str) -> Result<Vec<String>, String> {
    let Some(value) = args.get(key) else {
        return Ok(Vec::new());
    };
    let items = value
        .as_array()
        .ok_or_else(|| format!("invalid `{key}`: expected a list of strings"))?;
    items
        .iter()
        .map(|item| {
            item.as_str()
                .map(str::trim)
                .filter(|text| !text.is_empty())
                .map(ToOwned::to_owned)
                .ok_or_else(|| format!("invalid `{key}`: expected a list of strings"))
        })
        .collect()
}

fn optional_u64(args: &Value, key: &str) -> Result<Option<u64>, String> {
    let Some(value) = args.get(key) else {
        return Ok(None);
    };
    value
        .as_u64()
        .map(Some)
        .ok_or_else(|| format!("invalid `{key}`: expected a positive integer"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use lash::PersistedSessionState;
    use lash::plugin::{PluginError, SessionHandle, SessionManager, SessionTurnHandle};
    // Provider/ProviderOptions imports were used by inline Codex/Google
    // enum constructions, which the refactor replaced with test-only
    // stub providers defined inside each test.

    #[test]
    fn output_schema_supports_scalars_and_lists() {
        let schema = parse_output_schema(Some(&json!({
            "answer": "str",
            "count": "int",
            "items": "list[str]"
        })))
        .expect("schema")
        .expect("present");
        assert_eq!(schema["properties"]["answer"]["type"], json!("string"));
        assert_eq!(schema["properties"]["count"]["type"], json!("integer"));
        assert_eq!(schema["properties"]["items"]["type"], json!("array"));
    }

    #[test]
    fn output_schema_passes_through_lash_type_wrapper() {
        let inner_schema = json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "tags": { "type": "array", "items": { "type": "string" } },
                "status": { "type": "string", "enum": ["ok", "err"] }
            },
            "required": ["name", "tags", "status"],
            "additionalProperties": false
        });
        let wrapped = json!({ lashlang::LASH_TYPE_KEY: inner_schema.clone() });
        let schema = parse_output_schema(Some(&wrapped))
            .expect("schema")
            .expect("present");
        assert_eq!(schema, inner_schema);
    }

    #[test]
    fn output_schema_rejects_lash_type_without_type_field() {
        let wrapped = json!({ lashlang::LASH_TYPE_KEY: {"properties": {}} });
        let err = parse_output_schema(Some(&wrapped)).expect_err("missing type");
        assert!(err.contains("type"), "error: {err}");
    }

    #[test]
    fn output_schema_accepts_array_top_level_type() {
        let wrapped = json!({
            lashlang::LASH_TYPE_KEY: {
                "type": "array",
                "items": {"type": "string"}
            }
        });
        let schema = parse_output_schema(Some(&wrapped))
            .expect("schema")
            .expect("present");
        assert_eq!(schema["type"], json!("array"));
    }

    #[test]
    fn fork_turns_defaults_to_none() {
        assert!(matches!(
            parse_fork_turns(None).expect("fork"),
            ForkTurns::None
        ));
        assert!(matches!(
            parse_fork_turns(Some(&json!("all"))).expect("fork"),
            ForkTurns::All
        ));
        assert!(matches!(
            parse_fork_turns(Some(&json!(3))).expect("fork"),
            ForkTurns::Recent(3)
        ));
    }

    #[test]
    fn tool_definitions_are_mode_specific_but_mode_neutral_in_description() {
        let registry = default_registry(
            &std::collections::BTreeMap::new(),
            lash::ExecutionMode::standard(),
        );
        let standard = subagent_tool_definitions(lash::ExecutionMode::standard(), &registry);
        let rlm = subagent_tool_definitions(lash::ExecutionMode::new("rlm"), &registry);

        let standard_spawn = standard
            .iter()
            .find(|tool| tool.name == "spawn_agent")
            .expect("standard spawn_agent");
        let rlm_spawn = rlm
            .iter()
            .find(|tool| tool.name == "spawn_agent")
            .expect("rlm spawn_agent");

        assert_eq!(standard_spawn.description, rlm_spawn.description);
        assert!(!standard_spawn.description.contains("RLM"));
        assert!(!standard_spawn.description.contains("standard mode"));

        assert!(
            standard_spawn
                .examples
                .iter()
                .any(|example| example.contains("spawn_agent("))
        );
        assert!(
            rlm_spawn
                .examples
                .iter()
                .any(|example| example.contains("call spawn_agent"))
        );

        let standard_send = standard
            .iter()
            .find(|tool| tool.name == "send_message")
            .expect("standard send_message");
        let rlm_send = rlm
            .iter()
            .find(|tool| tool.name == "send_message")
            .expect("rlm send_message");
        assert!(
            standard_send
                .examples
                .iter()
                .all(|example| !example.contains("call "))
        );
        assert!(
            rlm_send
                .examples
                .iter()
                .all(|example| example.contains("call "))
        );
    }

    #[tokio::test]
    async fn spawn_uses_live_parent_provider_when_selecting_subagent_model() {
        struct SnapshotManager {
            snapshot: PersistedSessionState,
        }

        #[async_trait]
        impl SessionManager for SnapshotManager {
            async fn snapshot_current(&self) -> Result<PersistedSessionState, PluginError> {
                Ok(self.snapshot.clone())
            }

            async fn snapshot_session(
                &self,
                _session_id: &str,
            ) -> Result<PersistedSessionState, PluginError> {
                Ok(self.snapshot.clone())
            }

            async fn tool_catalog(
                &self,
                _session_id: &str,
            ) -> Result<Vec<serde_json::Value>, PluginError> {
                Ok(Vec::new())
            }

            async fn create_session(
                &self,
                _request: SessionCreateRequest,
            ) -> Result<SessionHandle, PluginError> {
                Err(PluginError::Session("not used in test".to_string()))
            }

            async fn close_session(&self, _session_id: &str) -> Result<(), PluginError> {
                Ok(())
            }

            async fn start_turn_stream(
                &self,
                _session_id: &str,
                _input: TurnInput,
            ) -> Result<SessionTurnHandle, PluginError> {
                Err(PluginError::Session("not used in test".to_string()))
            }

            async fn await_turn(&self, _turn_id: &str) -> Result<lash::AssembledTurn, PluginError> {
                Err(PluginError::Session("not used in test".to_string()))
            }

            async fn cancel_turn(&self, _turn_id: &str) -> Result<(), PluginError> {
                Ok(())
            }
        }

        // Two distinct stub providers so we can verify that spawn
        // resolves against the *live* policy, not the factory's stale
        // one. Each stub returns a different per-tier model from
        // `default_agent_model` so the final child policy's model shows
        // which provider the capability lookup resolved against.
        fn tiered_provider(tag: &'static str) -> lash::testing::TestProvider {
            let (kind, default_model, low_model) = match tag {
                "stale" => ("stale-stub", "stale-model", "stale-low"),
                "live" => ("live-stub", "live-model", "live-low"),
                _ => ("stub", "mock-model", "mock-low"),
            };
            lash::testing::TestProvider::builder()
                .kind(kind)
                .default_model(default_model)
                .default_agent_model(move |tier| {
                    if tier == "low" {
                        Some(lash::AgentModelSelection {
                            model: low_model.to_string(),
                            variant: None,
                        })
                    } else {
                        None
                    }
                })
                .complete_error("stub")
                .build()
        }
        let stale_policy = SessionPolicy {
            provider: tiered_provider("stale").into_handle(),
            execution_mode: lash::ExecutionMode::standard(),
            ..SessionPolicy::default()
        };
        let live_policy = SessionPolicy {
            provider: tiered_provider("live").into_handle(),
            execution_mode: lash::ExecutionMode::standard(),
            max_context_tokens: Some(1234),
            ..SessionPolicy::default()
        };
        let registry = Arc::new(default_registry(
            &std::collections::BTreeMap::new(),
            lash::ExecutionMode::standard(),
        ));
        let provider = SubagentToolsProvider {
            execution_mode: lash::ExecutionMode::standard(),
            registry: Arc::clone(&registry),
            host: Arc::new(LocalSubagentHost::default()),
        };
        let context = ToolExecutionContext {
            session_id: "root".to_string(),
            host: Arc::new(SnapshotManager {
                snapshot: PersistedSessionState {
                    policy: live_policy.clone(),
                    ..PersistedSessionState::default()
                },
            }),
            cancellation_token: None,
            async_task_id: None,
        };

        let request = provider
            .build_spawn_create_request(&context, "low", ForkTurns::None, None)
            .await
            .expect("spawn request");
        let child_policy = request.policy.expect("child policy");

        // The capability looked up the live policy's provider (Google), not
        // the stale Codex policy. This pins the behaviour where the spawn
        // pipeline always resolves models against the *current* session
        // policy snapshot, even when the factory was built earlier.
        let stale_choice = provider
            .build_session_policy(&stale_policy, "low")
            .expect("stale policy")
            .model;
        assert_eq!(child_policy.provider, live_policy.provider);
        assert_eq!(
            child_policy.max_context_tokens,
            live_policy.max_context_tokens
        );
        assert_ne!(child_policy.model, stale_choice);
        assert_eq!(child_policy.model, "live-low");
    }

    #[tokio::test]
    async fn pass_baton_creates_empty_rlm_successor_with_seed_and_task() {
        #[derive(Default)]
        struct BatonManager {
            snapshot: PersistedSessionState,
            created: Mutex<Vec<SessionCreateRequest>>,
        }

        #[async_trait]
        impl SessionManager for BatonManager {
            async fn snapshot_current(&self) -> Result<PersistedSessionState, PluginError> {
                Ok(self.snapshot.clone())
            }

            async fn snapshot_session(
                &self,
                _session_id: &str,
            ) -> Result<PersistedSessionState, PluginError> {
                Ok(self.snapshot.clone())
            }

            async fn tool_catalog(
                &self,
                _session_id: &str,
            ) -> Result<Vec<serde_json::Value>, PluginError> {
                Ok(Vec::new())
            }

            async fn create_session(
                &self,
                request: SessionCreateRequest,
            ) -> Result<SessionHandle, PluginError> {
                self.created.lock().expect("created").push(request.clone());
                Ok(SessionHandle {
                    session_id: request.session_id.unwrap_or_else(|| "child".to_string()),
                    parent_session_id: request.parent_session_id,
                    policy: request.policy.unwrap_or_default(),
                })
            }

            async fn close_session(&self, _session_id: &str) -> Result<(), PluginError> {
                Ok(())
            }

            async fn start_turn_stream(
                &self,
                _session_id: &str,
                _input: TurnInput,
            ) -> Result<SessionTurnHandle, PluginError> {
                Err(PluginError::Session("not used".to_string()))
            }

            async fn await_turn(&self, _turn_id: &str) -> Result<lash::AssembledTurn, PluginError> {
                Err(PluginError::Session("not used".to_string()))
            }

            async fn cancel_turn(&self, _turn_id: &str) -> Result<(), PluginError> {
                Ok(())
            }
        }

        let manager = Arc::new(BatonManager {
            snapshot: PersistedSessionState {
                policy: SessionPolicy {
                    execution_mode: lash::ExecutionMode::new("rlm"),
                    model: "model".to_string(),
                    max_context_tokens: Some(200_000),
                    ..SessionPolicy::default()
                },
                mode_turn_options: lash::ModeTurnOptions::rlm(RlmTermination::Finish {
                    schema: Some(json!({
                        "type": "object",
                        "properties": { "answer": { "type": "string" } },
                        "required": ["answer"]
                    })),
                }),
                ..PersistedSessionState::default()
            },
            created: Mutex::new(Vec::new()),
        });
        let registry = Arc::new(default_registry(
            &std::collections::BTreeMap::new(),
            lash::ExecutionMode::new("rlm"),
        ));
        let provider = SubagentToolsProvider {
            execution_mode: lash::ExecutionMode::new("rlm"),
            registry,
            host: Arc::new(LocalSubagentHost::default()),
        };
        let context = ToolExecutionContext {
            session_id: "parent".to_string(),
            host: manager.clone(),
            cancellation_token: None,
            async_task_id: None,
        };

        let result = provider
            .execute_with_context(
                "pass_baton",
                &json!({
                    "task": "finish from here",
                    "seed": { "x": 1, "query": "original" }
                }),
                &context,
            )
            .await;

        assert!(result.success, "{:?}", result.result);
        assert!(
            result
                .result
                .get("_baton")
                .and_then(Value::as_str)
                .is_some()
        );
        let created = manager.created.lock().expect("created");
        assert_eq!(created.len(), 1);
        let request = &created[0];
        assert!(matches!(request.start, SessionStartPoint::Empty));
        assert_eq!(request.parent_session_id.as_deref(), Some("parent"));
        assert_eq!(
            request
                .first_turn_input
                .as_ref()
                .map(|message| message.content.as_str()),
            Some("finish from here")
        );
        assert_eq!(request.initial_nodes.len(), 1);
        let SessionAppendNode::Event {
            event: SessionEventRecord::Mode(mode_event),
        } = &request.initial_nodes[0]
        else {
            panic!("expected seed globals event");
        };
        let Some(RlmModeEvent::RlmGlobalsPatch(seed)) = mode_event.rlm_event() else {
            panic!("expected RlmGlobalsPatch");
        };
        assert_eq!(seed.set["x"], json!(1));
        assert_eq!(seed.set["query"], json!("original"));
        let extras = request
            .mode_extras
            .decode::<RlmCreateExtras>(&lash::ExecutionMode::new("rlm"))
            .expect("decode extras")
            .expect("rlm extras");
        assert!(matches!(
            extras.termination,
            RlmTermination::Finish { schema: Some(_) }
        ));
    }
}
