//! RLM-mode subagent spawning surface.
//!
//! Examples are written in lashlang `call <tool> { ... }` syntax. Prompt prose
//! is tuned for schema-first results and binding subagent output.

use std::sync::Arc;

use async_trait::async_trait;
use lash_core::{
    BackgroundTaskKind, BackgroundTaskRegistration, BackgroundTaskState, SessionSpec,
    SubagentSessionContext, ToolArgumentProjectionPolicy, ToolCall, ToolContext, ToolContract,
    ToolDefinition, ToolExecutionMode, ToolManifest, ToolProvider, ToolResult,
};
use serde_json::Value;

use crate::SubagentSessionConfigurator;
use crate::capability::CapabilityRegistry;
use crate::rlm_support::{
    self, SpawnCreateRequestInput, build_spawn_create_request, capability_list_for_description,
    example_capability_name, finalise_tool_result, render_task_prompt, required_string,
    spawn_agent_input_schema, task_result_value, task_terminal_state, terminal_error_message,
    tool_definition, turn_input_for_task, unknown_capability_message,
};

pub(crate) struct RlmSubagentToolsProvider {
    pub(crate) registry: Arc<CapabilityRegistry>,
    pub(crate) session_spec: SessionSpec,
    pub(crate) configurator: Arc<dyn SubagentSessionConfigurator>,
    pub(crate) parent_subagent: Option<SubagentSessionContext>,
    pub(crate) include_submit_error: bool,
}

impl RlmSubagentToolsProvider {
    async fn spawn_agent(&self, args: &Value, context: &ToolContext) -> Result<Value, String> {
        let task = required_string(args, "task")?;
        let capability_name = capability_name_from_args(args, &self.registry)?;
        if self.registry.get(&capability_name).is_none() {
            return Err(unknown_capability_message(&capability_name, &self.registry));
        }
        let output_schema = lash_llm_tools::parse_output_schema(args.get("output"))?;
        let seed = lash_mode_rlm::RlmSeed::from_tool_args(args)?;
        let create_request = build_spawn_create_request(SpawnCreateRequestInput {
            registry: &self.registry,
            context,
            session_spec: &self.session_spec,
            capability_name: &capability_name,
            output_schema: output_schema.clone(),
            seed,
            parent_subagent: self.parent_subagent.as_ref(),
            originating_tool_call_id: context.tool_call_id(),
            configurator: self.configurator.as_ref(),
        })
        .await?;
        let turn_input = turn_input_for_task(render_task_prompt(&task, output_schema.as_ref()));
        run_child_session(context, create_request, turn_input).await
    }
}

async fn run_child_session(
    context: &ToolContext,
    create_request: lash_core::SessionCreateRequest,
    turn_input: lash_core::TurnInput,
) -> Result<Value, String> {
    if context.tool_catalog().await.ok().is_some_and(|catalog| {
        catalog
            .iter()
            .all(|tool| tool.get("name").and_then(serde_json::Value::as_str) != Some("spawn_agent"))
    }) {
        return Err("subagent spawning is unavailable in this session".to_string());
    }

    let requested_child_session_id = create_request
        .session_id
        .clone()
        .ok_or_else(|| "child session id is required".to_string())?;
    let sessions = context.sessions();
    let child_session = tokio::spawn(async move { sessions.create_session(create_request).await })
        .await
        .map_err(|err| format!("child session creation task failed: {err}"))?
        .map_err(|err| format!("failed to create child session: {err}"))?;
    let child_session_id = child_session.session_id;
    let task_id = format!("subagent:{child_session_id}");
    let mut task_create_request = lash_core::SessionCreateRequest::child(
        context.session_id().to_string(),
        lash_core::SessionStartPoint::Empty,
        child_session.policy.clone(),
        lash_core::ModeExtras::default(),
        "subagent",
    );
    task_create_request.session_id = Some(child_session_id.clone());
    let registration_input = lash_core::BackgroundTaskInput::SessionTurn {
        create_request: Box::new(task_create_request),
        turn_input: Box::new(turn_input.clone()),
        output_contract: lash_core::ToolOutputContract::from_input_schema("output", None),
    };

    let task_context = context.clone();
    let executor_child_session_id = child_session_id.clone();
    let executor_turn_input = turn_input.clone();
    let executor = lash_core::BackgroundTaskExecutor::new(move |cancellation| async move {
        let sessions = task_context.sessions();
        let turn = match sessions
            .start_turn_stream(&executor_child_session_id, executor_turn_input)
            .await
        {
            Ok(turn) => turn,
            Err(err) => {
                close_child_session(&task_context, &executor_child_session_id, None).await;
                return lash_core::BackgroundTaskCompletion {
                    state: BackgroundTaskState::Failed,
                    summary: Some(format!("failed to start child turn: {err}")),
                    output: Some(lash_core::ToolCallOutput::failure(
                        lash_core::ToolFailure::tool(
                            lash_core::ToolFailureClass::Execution,
                            "subagent_start_failed",
                            err.to_string(),
                        ),
                    )),
                };
            }
        };
        let turn_id = turn.turn_id.clone();
        drop(turn.events);

        let sessions = task_context.sessions();
        let outcome = tokio::select! {
            biased;
            _ = cancellation.cancelled() => {
                let _ = sessions.cancel_turn(&turn_id).await;
                sessions.await_turn(&turn_id).await
            }
            outcome = sessions.await_turn(&turn_id) => outcome,
        };
        let terminal_state = if cancellation.is_cancelled()
            && outcome
                .as_ref()
                .err()
                .is_some_and(|err| err.to_string().contains("unknown turn"))
        {
            BackgroundTaskState::Cancelled
        } else {
            task_terminal_state(&outcome)
        };
        close_child_session(&task_context, &executor_child_session_id, None).await;
        lash_core::BackgroundTaskCompletion {
            state: terminal_state,
            summary: completion_summary(&outcome, terminal_state),
            output: Some(output_from_child_outcome(&outcome, terminal_state)),
        }
    });

    if let Err(err) = context
        .tasks()
        .start_background_task(
            BackgroundTaskRegistration::new(
                task_id.clone(),
                BackgroundTaskKind::SessionTurn,
                "subagent",
                lash_core::BackgroundTaskScope {
                    session_id: context.session_id().to_string(),
                },
                registration_input,
            )
            .with_child_session_id(child_session_id.clone())
            .with_close_policy(lash_core::BackgroundClosePolicy::Cancel)
            .with_optional_parent_task_id(context.async_task_id().map(str::to_string)),
            executor,
        )
        .await
    {
        let _ = context.sessions().close_session(&child_session_id).await;
        return Err(format!(
            "failed to register subagent background task for `{requested_child_session_id}`: {err}"
        ));
    }

    let completion = context
        .tasks()
        .await_background_task(&task_id)
        .await
        .map_err(|err| format!("subagent failed while executing its task: {err}"))?;
    if completion.state == BackgroundTaskState::Cancelled {
        return Err("spawn_agent was cancelled".to_string());
    }
    if completion.state == BackgroundTaskState::Failed {
        return Err(completion
            .summary
            .unwrap_or_else(|| "subagent failed".to_string()));
    }
    completion_value(completion)
}

async fn close_child_session(context: &ToolContext, child_session_id: &str, turn_id: Option<&str>) {
    if let Some(turn_id) = turn_id {
        let _ = context.sessions().cancel_turn(turn_id).await;
        let _ = context.sessions().await_turn(turn_id).await;
    }
    let _ = context.sessions().close_session(child_session_id).await;
}

fn completion_summary(
    outcome: &Result<lash_core::AssembledTurn, lash_core::PluginError>,
    terminal_state: BackgroundTaskState,
) -> Option<String> {
    match outcome {
        Ok(turn) if terminal_state == BackgroundTaskState::Failed => {
            terminal_error_message(&turn.outcome).or_else(|| Some("subagent failed".to_string()))
        }
        Ok(_) => None,
        Err(err) => Some(err.to_string()),
    }
}

fn output_from_child_outcome(
    outcome: &Result<lash_core::AssembledTurn, lash_core::PluginError>,
    terminal_state: BackgroundTaskState,
) -> lash_core::ToolCallOutput {
    match outcome {
        Ok(turn) if terminal_state == BackgroundTaskState::Completed => {
            lash_core::ToolCallOutput::success(task_result_value(turn))
        }
        Ok(turn) if terminal_state == BackgroundTaskState::Cancelled => {
            lash_core::ToolCallOutput::cancelled(lash_core::ToolCancellation::runtime(
                terminal_error_message(&turn.outcome)
                    .unwrap_or_else(|| "spawn_agent was cancelled".to_string()),
            ))
        }
        Ok(turn) => lash_core::ToolCallOutput::failure(lash_core::ToolFailure::tool(
            lash_core::ToolFailureClass::Execution,
            "subagent_failed",
            terminal_error_message(&turn.outcome).unwrap_or_else(|| "subagent failed".to_string()),
        )),
        Err(err) if terminal_state == BackgroundTaskState::Cancelled => {
            lash_core::ToolCallOutput::cancelled(lash_core::ToolCancellation::runtime(
                err.to_string(),
            ))
        }
        Err(err) => lash_core::ToolCallOutput::failure(lash_core::ToolFailure::tool(
            lash_core::ToolFailureClass::Execution,
            "subagent_failed",
            err.to_string(),
        )),
    }
}

fn completion_value(completion: lash_core::BackgroundTaskCompletion) -> Result<Value, String> {
    let output = completion
        .output
        .ok_or_else(|| "subagent completed without output".to_string())?;
    match output.outcome {
        lash_core::ToolCallOutcome::Success(value) => Ok(value.to_json_value()),
        lash_core::ToolCallOutcome::Failure(failure) => Err(failure.message),
        lash_core::ToolCallOutcome::Cancelled(cancellation) => Err(cancellation.message),
    }
}

#[async_trait]
impl ToolProvider for RlmSubagentToolsProvider {
    fn tool_manifests(&self) -> Vec<ToolManifest> {
        self.tool_definitions()
            .into_iter()
            .map(|tool| tool.manifest())
            .collect()
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<ToolContract>> {
        self.tool_definitions()
            .into_iter()
            .find(|tool| tool.name == name)
            .map(|tool| Arc::new(tool.contract()))
    }

    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        let result = match call.name {
            "spawn_agent" => self.spawn_agent(call.args, call.context).await,
            "submit_error" => return rlm_support::submit_error_tool_result(call.args),
            other => Err(format!("Unknown tool: {other}")),
        };
        finalise_tool_result(result)
    }
}

impl RlmSubagentToolsProvider {
    fn tool_definitions(&self) -> Vec<ToolDefinition> {
        let mut definitions = rlm_subagent_tool_definitions(&self.registry.names());
        if self.include_submit_error {
            definitions.push(rlm_support::submit_error_tool_definition());
        }
        definitions
    }
}

pub(crate) fn rlm_subagent_tool_definitions(capability_names: &[String]) -> Vec<ToolDefinition> {
    vec![spawn_agent_tool_definition(capability_names)]
}

pub fn spawn_agent_tool_definition(capability_names: &[String]) -> ToolDefinition {
    let example_capability = example_capability_name(capability_names);
    let capability_arg = capability_example_arg(capability_names, &example_capability);
    spawn_agent_definition(
        capability_names,
        vec![
            // Schema-first: the highest-leverage shape — bind a typed result.
            format!(
                r#"typed = (call spawn_agent {{ task: "Find the longest line in src/main.rs"{capability_arg}, output: {{ line: "str", length: "int" }} }})?"#
            ),
            // Reusable Type literal for richer shapes.
            r#"Shape = Type { name: str, tags: list[str], status: enum["ok", "err"] }"#.into(),
            format!(
                r#"signed = (call spawn_agent {{ task: "Parse the book listing in data/books.json"{capability_arg}, output: Shape }})?"#
            ),
            // Canonical fan-out: start N and await generic handles.
            format!(
                r#"a = start call spawn_agent {{ task: "List files under src/auth/ that handle session tokens"{capability_arg}, output: {{ files: "list[str]" }} }}"#
            ),
            format!(
                r#"b = start call spawn_agent {{ task: "Summarise migrations/ schema changes since v3"{capability_arg}, output: {{ summary: "str" }} }}"#
            ),
            r#"results = parallel { auth: (await a)?, db: (await b)? }"#.into(),
            // seed: pass projected source through to the child as a projected
            // binding; pass plain values as RLM globals on the child.
            format!(
                r#"answer = (call spawn_agent {{ task: "Solve sub-problem 3 using the bound problem text and the running findings."{capability_arg}, seed: {{ problem: input.prompt, findings: findings }}, output: {{ value: "int" }} }})?"#
            ),
            // Untyped is fine for free-form prose results.
            format!(
                r#"prose = call spawn_agent {{ task: "Skim the routes in api/ and flag any missing auth checks"{capability_arg} }}"#
            ),
        ],
    )
}

fn spawn_agent_definition(capability_names: &[String], examples: Vec<String>) -> ToolDefinition {
    let cap_list = capability_list_for_description(capability_names);
    let capability_detail = capability_detail_for_tool_description(capability_names);
    let description = format!(
        "Run a subagent and return its final result. Plain `call spawn_agent {{ ... }}` blocks until the child finishes. Use `start call spawn_agent {{ ... }}` for fan-out; it returns a generic lashlang async handle immediately. {capability_detail} `output` defines the typed return shape. Available capabilities: {cap_list}. \
        \n\nThe child starts with **no** inherited state — globals, projected bindings, message history are all blank. Hand it specific data via `seed: {{ name: value, ... }}`. Each entry's kind is preserved automatically: if `value`'s lashlang source root is a host-projected binding (e.g. `seed: {{ problem: input.prompt }}`) the child receives `problem` as a read-only projected binding, identical to how it appeared on the parent. Otherwise it lands as a regular RLM global. Computed expressions default to global. Projected seed entries require an RLM child; passing one to a non-RLM capability is an error.\
        \n\nA child can fail terminally with `call submit_error {{ reason: \"...\" }}`; this tool returns an error with that reason."
    );
    tool_definition(
        "spawn_agent",
        description,
        spawn_agent_input_schema(capability_names),
        examples,
        ToolExecutionMode::Serial,
    )
    .with_argument_projection(
        ToolArgumentProjectionPolicy::preserve_projected_refs_in_field("seed"),
    )
    .with_output_from_input_schema("output", None)
}

fn capability_detail_for_tool_description(capability_names: &[String]) -> String {
    if capability_names.len() == 1 {
        return "Only one capability is available in this session, so omit `capability` unless you need to be explicit.".to_string();
    }
    "Pick `capability` from the available list; unavailable capability names are rejected."
        .to_string()
}

fn capability_example_arg(capability_names: &[String], example_capability: &str) -> String {
    if capability_names.len() == 1 {
        String::new()
    } else {
        format!(r#", capability: "{example_capability}""#)
    }
}

fn capability_name_from_args(
    args: &Value,
    registry: &CapabilityRegistry,
) -> Result<String, String> {
    match args.get("capability") {
        Some(Value::String(capability)) => Ok(capability.clone()),
        Some(_) => Err("field `capability` must be a string".to_string()),
        None => {
            let names = registry.names();
            match names.as_slice() {
                [only] => Ok(only.clone()),
                [] => Err(
                    "field `capability` is required: no default capability is registered"
                        .to_string(),
                ),
                _ => Err(format!(
                    "field `capability` is required when multiple capabilities are available: {}",
                    capability_list_for_description(&names)
                )),
            }
        }
    }
}
