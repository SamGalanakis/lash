//! RLM-mode subagent spawning surface.
//!
//! Examples are written in lashlang `call <tool> { ... }` syntax. Prompt prose
//! is tuned for schema-first results and binding subagent output.

use std::sync::{Arc, Mutex};

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
    let child_session = context
        .sessions()
        .create_session(create_request)
        .await
        .map_err(|err| format!("failed to create child session: {err}"))?;
    let child_session_id = child_session.session_id;
    let task_id = format!("subagent:{child_session_id}");
    let active_turn_id: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

    let cancel = child_cancel_callback(
        context,
        child_session_id.clone(),
        task_id.clone(),
        Arc::clone(&active_turn_id),
    );
    if let Err(err) = context
        .tasks()
        .register_background_task(
            BackgroundTaskRegistration {
                id: task_id.clone(),
                kind: BackgroundTaskKind::Subagent,
                producer: "subagent",
                child_session_id: Some(child_session_id.clone()),
                parent_task_id: context.async_task_id().map(str::to_string),
            },
            Some(cancel),
        )
        .await
    {
        let _ = context.sessions().close_session(&child_session_id).await;
        return Err(format!(
            "failed to register subagent background task for `{requested_child_session_id}`: {err}"
        ));
    }

    let turn = match context
        .sessions()
        .start_turn_stream(&child_session_id, turn_input)
        .await
    {
        Ok(turn) => turn,
        Err(err) => {
            cleanup_child_session(
                context,
                &child_session_id,
                &task_id,
                None,
                BackgroundTaskState::Failed,
            )
            .await;
            return Err(format!("failed to start child turn: {err}"));
        }
    };
    let turn_id = turn.turn_id.clone();
    drop(turn.events);
    *active_turn_id.lock().expect("active turn lock") = Some(turn_id.clone());

    let sessions = context.sessions();
    let cancellation = context.cancellation_token().cloned();
    let outcome = if let Some(token) = cancellation {
        tokio::select! {
            biased;
            _ = token.cancelled() => {
                let _ = sessions.cancel_turn(&turn_id).await;
                sessions.await_turn(&turn_id).await
            }
            outcome = sessions.await_turn(&turn_id) => outcome,
        }
    } else {
        sessions.await_turn(&turn_id).await
    };
    *active_turn_id.lock().expect("active turn lock") = None;

    let terminal_state = task_terminal_state(&outcome);
    cleanup_child_session(context, &child_session_id, &task_id, None, terminal_state).await;

    if let Err(err) = &outcome
        && err.to_string().contains("unknown turn")
    {
        return Err("spawn_agent was cancelled".to_string());
    }
    let turn = outcome.map_err(|err| format!("subagent failed while executing its task: {err}"))?;
    if let Some(error) = terminal_error_message(&turn.outcome) {
        return Err(error);
    }
    if terminal_state == BackgroundTaskState::Cancelled {
        return Err("spawn_agent was cancelled".to_string());
    }
    if terminal_state == BackgroundTaskState::Failed {
        return Err("subagent failed".to_string());
    }
    Ok(task_result_value(&turn))
}

fn child_cancel_callback(
    context: &ToolContext,
    child_session_id: String,
    task_id: String,
    active_turn_id: Arc<Mutex<Option<String>>>,
) -> lash_core::LocalBackgroundTaskCancel {
    let sessions = context.sessions();
    let tasks = context.tasks();
    Arc::new(move || {
        let sessions = sessions.clone();
        let tasks = tasks.clone();
        let child_session_id = child_session_id.clone();
        let task_id = task_id.clone();
        let active_turn_id = Arc::clone(&active_turn_id);
        Box::pin(async move {
            let turn_id = active_turn_id.lock().expect("active turn lock").clone();
            if let Some(turn_id) = turn_id {
                let _ = sessions.cancel_turn(&turn_id).await;
                let _ = sessions.await_turn(&turn_id).await;
            }
            let _ = tasks
                .cancel_all_background_tasks_for_session(&child_session_id)
                .await;
            let _ = sessions.close_session(&child_session_id).await;
            tasks
                .complete_background_task(&task_id, BackgroundTaskState::Cancelled)
                .await;
        })
    })
}

async fn cleanup_child_session(
    context: &ToolContext,
    child_session_id: &str,
    task_id: &str,
    turn_id: Option<&str>,
    terminal_state: BackgroundTaskState,
) {
    if let Some(turn_id) = turn_id {
        let _ = context.sessions().cancel_turn(turn_id).await;
        let _ = context.sessions().await_turn(turn_id).await;
    }
    let _ = context
        .tasks()
        .cancel_all_background_tasks_for_session(child_session_id)
        .await;
    let _ = context.sessions().close_session(child_session_id).await;
    context
        .tasks()
        .complete_background_task(task_id, terminal_state)
        .await;
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
