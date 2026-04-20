mod host;
mod policy;

use std::sync::Arc;

use async_trait::async_trait;
use lash::plugin::{PluginError, PluginFactory, PluginSessionContext, ToolSurfaceOverride};
use lash::provider::AgentModels;
use lash::{
    InputItem, ModeExtras, PluginSpec, PluginSpecFactory, ProgressSender, RlmCreateExtras,
    RlmTermination, SessionCreateRequest, SessionPluginMode, SessionPolicy, SessionStartPoint,
    ToolDefinition, ToolExecutionContext, ToolProvider, ToolResult, ToolSurfaceContribution,
    TurnInput,
};
use serde_json::{Value, json};

pub use host::{
    AgentSummary, Capability, CloseAgentRequest, CloseAgentResponse, FollowupTaskRequest,
    FollowupTaskResponse, ListAgentsRequest, ListAgentsResponse, LocalSubagentHost,
    SendMessageRequest, SendMessageResponse, SessionAgentInfo, SpawnAgentRequest,
    SpawnAgentResponse, SubagentHost, TaskToolCallSummary, WaitAgentEvent, WaitAgentRequest,
    WaitAgentResponse, WaitAgentSessionSummary, truncate_snapshot_to_recent_turns,
};
use policy::{
    denied_tools, pick_model_and_variant, subagent_prompt_contributions, subagent_tool_definitions,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubagentToolConfig {
    pub low_tier_execution_mode: lash::ExecutionMode,
}

impl Default for SubagentToolConfig {
    fn default() -> Self {
        Self {
            low_tier_execution_mode: lash::ExecutionMode::Standard,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ForkTurns {
    None,
    All,
    Recent(usize),
}

struct SubagentToolsProvider {
    execution_mode: lash::ExecutionMode,
    tool_config: SubagentToolConfig,
    agent_models: Option<AgentModels>,
    host: Arc<dyn SubagentHost>,
}

impl SubagentToolsProvider {
    fn build_session_policy(
        &self,
        current_policy: &SessionPolicy,
        capability: Capability,
    ) -> SessionPolicy {
        let (model, model_variant) =
            pick_model_and_variant(current_policy, &self.agent_models, capability);
        SessionPolicy {
            model,
            model_variant,
            provider: current_policy.provider.clone(),
            max_context_tokens: current_policy.max_context_tokens,
            max_turns: None,
            execution_mode: match capability {
                Capability::Low => self.tool_config.low_tier_execution_mode,
                Capability::Medium | Capability::High => current_policy.execution_mode,
            },
            ..Default::default()
        }
    }

    async fn build_spawn_create_request(
        &self,
        context: &ToolExecutionContext,
        capability: Capability,
        fork_turns: ForkTurns,
        output_schema: Option<Value>,
    ) -> Result<SessionCreateRequest, String> {
        let current_snapshot = context
            .host
            .snapshot_session(&context.session_id)
            .await
            .map_err(|err| format!("failed to snapshot current session: {err}"))?;
        let mut policy = self.build_session_policy(&current_snapshot.policy, capability);
        let mut mode_extras = ModeExtras::default();
        if let Some(schema) = output_schema.clone() {
            policy.execution_mode = lash::ExecutionMode::Rlm;
            mode_extras = ModeExtras::Rlm(RlmCreateExtras {
                termination: RlmTermination::Finish {
                    schema: Some(schema),
                },
            });
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

        Ok(SessionCreateRequest {
            session_id: Some(uuid::Uuid::new_v4().to_string()),
            parent_session_id: Some(context.session_id.clone()),
            start,
            policy: Some(policy),
            plugin_mode: SessionPluginMode::Fresh,
            initial_nodes: Vec::new(),
            context_surface: lash::SessionContextSurface::default(),
            mode_extras,
            usage_source: Some("subagent".to_string()),
        })
    }

    async fn spawn_agent(
        &self,
        args: &Value,
        context: &ToolExecutionContext,
    ) -> Result<Value, String> {
        let task_name = required_string(args, "task_name")?;
        let task = required_string(args, "task")?;
        let capability = required_string(args, "capability")?
            .parse::<Capability>()
            .map_err(|_| "invalid capability: expected `low`, `medium`, or `high`".to_string())?;
        let fork_turns = parse_fork_turns(args.get("fork_turns"))?;
        let output_schema = parse_output_schema(args.get("output"))?;
        let create_request = self
            .build_spawn_create_request(context, capability, fork_turns, output_schema.clone())
            .await?;
        let turn_input = TurnInput {
            items: vec![InputItem::Text {
                text: render_task_prompt(&task, output_schema.as_ref()),
            }],
            image_blobs: std::collections::HashMap::new(),
            user_input: None,
            mode: None,
            rlm_termination_override: None,
        };
        let response = self
            .host
            .spawn_agent(
                context,
                SpawnAgentRequest {
                    task_name,
                    task,
                    capability,
                    create_request,
                    turn_input,
                },
            )
            .await?;
        serde_json::to_value(response).map_err(|err| err.to_string())
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
        let rlm_termination_override = Some(match output_schema.clone() {
            Some(schema) => RlmTermination::Finish {
                schema: Some(schema),
            },
            None => RlmTermination::ProseWithoutFence,
        });
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
                        rlm_termination_override,
                    },
                    task,
                    interrupt: optional_bool(args, "interrupt").unwrap_or(false),
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
        subagent_tool_definitions(self.execution_mode)
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
    tool_config: SubagentToolConfig,
    agent_models: Option<AgentModels>,
    host: Arc<dyn SubagentHost>,
}

impl SubagentsPluginFactory {
    pub fn new(
        policy: SessionPolicy,
        tool_config: SubagentToolConfig,
        agent_models: Option<AgentModels>,
        host: Arc<dyn SubagentHost>,
    ) -> Self {
        Self {
            policy,
            tool_config,
            agent_models,
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
        policy.execution_mode = ctx.execution_mode;
        let provider = Arc::new(SubagentToolsProvider {
            execution_mode: ctx.execution_mode,
            tool_config: self.tool_config,
            agent_models: self.agent_models.clone(),
            host: Arc::clone(&self.host),
        });
        let prompt_contributions = subagent_prompt_contributions();
        let host = Arc::clone(&self.host);
        PluginSpecFactory::new(
            "subagents",
            Arc::new(move |_ctx| {
                let provider = Arc::clone(&provider);
                let host = Arc::clone(&host);
                let contributions = prompt_contributions.clone();
                Ok(PluginSpec::new()
                    .with_tool_provider(provider as Arc<dyn ToolProvider>)
                    .with_prompt_contributor(Arc::new(move |_ctx| {
                        let contributions = contributions.clone();
                        Box::pin(async move { Ok(contributions) })
                    }))
                    .with_tool_surface_contributor(Arc::new(move |ctx| {
                        subagent_surface_contribution(&host, ctx)
                    })))
            }),
        )
        .build(ctx)
    }
}

fn subagent_surface_contribution(
    host: &Arc<dyn SubagentHost>,
    ctx: lash::plugin::ToolSurfaceContext,
) -> Result<ToolSurfaceContribution, PluginError> {
    let Some(info) = host.session_info(&ctx.session_id) else {
        return Ok(ToolSurfaceContribution::default());
    };
    let Some(capability) = info.capability else {
        return Ok(ToolSurfaceContribution::default());
    };
    let denied = denied_tools(capability);
    let overrides = ctx
        .tools
        .iter()
        .filter(|tool| denied.contains(tool.name.as_str()))
        .map(|tool| ToolSurfaceOverride {
            tool_name: tool.name.clone(),
            enabled: Some(false),
            injected: Some(false),
        })
        .collect::<Vec<_>>();
    let tool_list_notes = vec![format!(
        "Subagent path: {}. Capability: {}.",
        info.path,
        capability.as_str()
    )];
    Ok(ToolSurfaceContribution {
        overrides,
        tool_list_notes,
    })
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

fn optional_bool(args: &Value, key: &str) -> Option<bool> {
    args.get(key).and_then(Value::as_bool)
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
    use std::sync::Arc;

    use async_trait::async_trait;
    use lash::PersistedSessionState;
    use lash::plugin::{PluginError, SessionHandle, SessionManager, SessionTurnHandle};
    use lash::provider::{Provider, ProviderOptions};

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
        let standard = subagent_tool_definitions(lash::ExecutionMode::Standard);
        let rlm = subagent_tool_definitions(lash::ExecutionMode::Rlm);

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

        let stale_policy = SessionPolicy {
            provider: Provider::Codex {
                access_token: "token".to_string(),
                refresh_token: "refresh".to_string(),
                expires_at: 0,
                account_id: None,
                options: ProviderOptions::default(),
            },
            execution_mode: lash::ExecutionMode::Standard,
            ..SessionPolicy::default()
        };
        let live_policy = SessionPolicy {
            provider: Provider::GoogleOAuth {
                access_token: "token".to_string(),
                refresh_token: "refresh".to_string(),
                expires_at: 0,
                project_id: None,
                options: ProviderOptions::default(),
            },
            execution_mode: lash::ExecutionMode::Standard,
            max_context_tokens: Some(1234),
            ..SessionPolicy::default()
        };
        let provider = SubagentToolsProvider {
            execution_mode: lash::ExecutionMode::Standard,
            tool_config: SubagentToolConfig::default(),
            agent_models: None,
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
            .build_spawn_create_request(&context, Capability::Low, ForkTurns::None, None)
            .await
            .expect("spawn request");
        let child_policy = request.policy.expect("child policy");

        let stale_choice = pick_model_and_variant(&stale_policy, &None, Capability::Low).0;
        assert_eq!(child_policy.provider, live_policy.provider);
        assert_eq!(
            child_policy.max_context_tokens,
            live_policy.max_context_tokens
        );
        assert_ne!(child_policy.model, stale_choice);
        assert_eq!(child_policy.model, "gemini-3-flash-preview");
    }
}
