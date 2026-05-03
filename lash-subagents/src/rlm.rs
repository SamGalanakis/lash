//! RLM-mode subagent tool surface.
//!
//! Exposes everything the standard provider does plus `continue_as`
//! (clean-window successor for tail-calling). Examples are written in
//! lashlang `call <tool> { ... }` syntax. Prompt prose is tuned for
//! schema-first results and binding subagent output.

use std::sync::Arc;

use async_trait::async_trait;
use lash::{
    DirectJsonSchema, DirectMessage, DirectOutputSpec, DirectPart, DirectRequest, DirectRole,
    MessageRole, PluginMessage, ProgressSender, PromptContribution, ToolDefinition,
    ToolExecutionContext, ToolExecutionMode, ToolProvider, ToolResult,
};
use serde_json::{Value, json};

use crate::capability::CapabilityRegistry;
use crate::host::SubagentHost;
use crate::shared::{
    self, build_spawn_create_request, capability_list_for_description, example_capability_name,
    finalise_tool_result, fresh_child_request, normalize_context_policy, parse_output_schema,
    render_task_prompt, required_string, rlm_seed_initial_nodes, spawn_agent_input_schema,
    tool_definition, turn_input_for_task, unknown_capability_message,
};
use crate::types::{CloseAgentRequest, SpawnAgentRequest, WaitAgentRequest, WaitUntil};

pub(crate) struct RlmSubagentToolsProvider {
    pub(crate) registry: Arc<CapabilityRegistry>,
    pub(crate) host: Arc<dyn SubagentHost>,
}

impl RlmSubagentToolsProvider {
    async fn llm_query(
        &self,
        args: &Value,
        context: &ToolExecutionContext,
    ) -> Result<Value, String> {
        let task = required_string(args, "task")?;
        let inputs = args.get("inputs").cloned().unwrap_or(Value::Null);
        let output_schema = parse_output_schema(args.get("output"))?;
        let current_snapshot = context
            .host
            .snapshot_session(&context.session_id)
            .await
            .map_err(|err| format!("failed to snapshot current session: {err}"))?;
        let policy =
            shared::build_session_policy(&self.registry, &current_snapshot.policy, "explore")?;
        let response_schema = llm_query_response_schema(output_schema.as_ref());
        let prompt = llm_query_prompt(&task, &inputs, output_schema.as_ref());

        let output = DirectOutputSpec::JsonSchema(DirectJsonSchema {
            name: "llm_query_result".to_string(),
            schema: response_schema.clone(),
            strict: true,
        });

        let completion = context
            .host
            .direct_completion(
                DirectRequest {
                    model: policy.model,
                    model_variant: policy.model_variant,
                    messages: vec![
                        DirectMessage {
                            role: DirectRole::System,
                            parts: vec![DirectPart::Text(
                                "You answer a focused sub-question for another agent. Use only the task and inputs supplied. Return only JSON matching the requested result wrapper. Use kind=\"error\" with a concise error only when the task cannot be answered from the supplied inputs."
                                    .to_string(),
                            )],
                        },
                        DirectMessage {
                            role: DirectRole::User,
                            parts: vec![DirectPart::Text(prompt)],
                        },
                    ],
                    attachments: Vec::new(),
                    output,
                    stream_events: None,
                    session_id: Some(format!("{}-llm-query", context.session_id)),
                },
                "llm_query",
            )
            .await
            .map_err(|err| format!("llm_query failed: {err}"))?;

        parse_llm_query_result(&completion.text, &response_schema)
    }

    async fn spawn_agent(
        &self,
        args: &Value,
        context: &ToolExecutionContext,
    ) -> Result<Value, String> {
        let agent_name = required_string(args, "agent_name")?;
        let task = required_string(args, "task")?;
        let capability_name = required_string(args, "capability")?;
        if self.registry.get(&capability_name).is_none() {
            return Err(unknown_capability_message(&capability_name, &self.registry));
        }
        let output_schema = parse_output_schema(args.get("output"))?;
        let create_request = build_spawn_create_request(
            &self.registry,
            context,
            &capability_name,
            output_schema.clone(),
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
            .cancellation_token
            .as_ref()
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

    async fn continue_as(
        &self,
        args: &Value,
        context: &ToolExecutionContext,
    ) -> Result<Value, String> {
        let task = required_string(args, "task")?;
        let seed = match args.get("seed") {
            None | Some(Value::Null) => serde_json::Map::new(),
            Some(Value::Object(map)) => map.clone(),
            Some(_) => return Err("continue_as `seed` must be a record/dict".to_string()),
        };

        let current_snapshot = context
            .host
            .snapshot_session(&context.session_id)
            .await
            .map_err(|err| format!("failed to snapshot current session: {err}"))?;
        let termination = current_snapshot.mode_turn_options.rlm_termination();
        let mut policy = current_snapshot.policy.clone();
        policy.execution_mode = lash::ExecutionMode::new("rlm");
        normalize_context_policy(&mut policy);

        let initial_nodes = rlm_seed_initial_nodes(seed);

        let mode_extras = lash::ModeExtras::typed(
            lash::ExecutionMode::new("rlm"),
            lash_rlm_types::RlmCreateExtras { termination },
        )
        .map_err(|err| format!("failed to encode rlm mode extras: {err}"))?;
        let mut request = fresh_child_request(
            context.session_id.clone(),
            lash::SessionStartPoint::Empty,
            policy,
            mode_extras,
            "continue_as",
        );
        request.plugin_mode = lash::SessionPluginMode::InheritCurrent;
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
            .map_err(|err| format!("failed to create continue_as successor: {err}"))?;

        Ok(json!({
            "ok": true,
            "_continue_as": successor_session_id,
            "task": task,
        }))
    }
}

#[async_trait]
impl ToolProvider for RlmSubagentToolsProvider {
    fn definitions(&self) -> Vec<ToolDefinition> {
        let mut definitions = rlm_subagent_tool_definitions(&self.registry.names());
        definitions.push(shared::submit_error_tool_definition());
        definitions.push(list_async_handles_definition());
        definitions
    }

    async fn execute(&self, name: &str, _args: &Value) -> ToolResult {
        if name == "submit_error" {
            return shared::submit_error_tool_result(_args);
        }
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
            "llm_query" => self.llm_query(args, context).await,
            "spawn_agent" => self.spawn_agent(args, context).await,
            "continue_as" => self.continue_as(args, context).await,
            "submit_error" => return shared::submit_error_tool_result(args),
            _ => Err(format!("Unknown tool: {name}")),
        };
        finalise_tool_result(result)
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

pub(crate) fn rlm_subagent_prompt_contributions() -> Vec<PromptContribution> {
    vec![PromptContribution::guidance(
        "Subagents and lightweight LLM calls",
        "`llm_query` is the cheap decomposition primitive: one focused LLM call, no child session, no tools, no REPL loop. Use it for semantic extraction, summarization, classification, judging, or transforming data you already have in variables. Shape it as `call llm_query { task: \"...\", inputs: { text: chunk }, output: { answer: \"str\" } }`. Omit `output` for a plain string. `output` accepts the same record descriptors and `Type { ... }` literals as `spawn_agent`.\n\nUse `spawn_agent` when the subproblem needs tool use, file/repo inspection, shell commands, edits, multi-step exploration, its own context window, cancellation, or recursive subagents. Plain `call spawn_agent { ... }` blocks until the child finishes and returns the child result. For fan-out, use generic lashlang async handles: `h = start call spawn_agent { agent_name: \"auth\", task: \"Summarise auth\", capability: \"explore\", output: { summary: \"str\" } }`, then `result = (await h)?`. Use `parallel` or record-shaped awaits to collect independent handles, and `cancel h` to stop a live child subtree.\n\n`list_async_handles()` returns only live handles, grouped as `subagent.<agent_name>` for normalized subagent names and `tool.<handle_id>` for other async tool calls. Cancel stale subagent handles through those values when the work is no longer needed.\n\nTwo subagent capabilities. `explore` is read-only and cannot recurse. `peer` has full edit + recurse powers. Default to `explore` for parallel investigation; use `peer` only when the subagent must mutate or spawn its own subagents.\n\n`output` defines the typed return shape. Pass either a record of scalar type descriptors (`{ line: \"str\", length: \"int\" }`) or a `Type { ... }` literal. With `output` set the subagent ends with `submit <expr>` and the value flows straight into your bound variable. A child can fail terminally with `call submit_error { reason: \"...\" }`; parent `spawn_agent` returns an error so `?` short-circuits naturally.\n\nCanonical fan-out:\n\n```lashlang\na = start call spawn_agent { agent_name: \"auth\", task: \"Summarise auth flow\", capability: \"explore\", output: { summary: \"str\" } }\nb = start call spawn_agent { agent_name: \"db\", task: \"Summarise migrations\", capability: \"explore\", output: { summary: \"str\" } }\nhandles = (call list_async_handles {})?\nresults = parallel { auth: (await handles.subagent.auth)?, db: (await handles.subagent.db)? }\nsubmit results\n```\n\nIn user-facing prose, call them subagents, not delegates or child agents.",
    )]
}

pub(crate) fn rlm_continue_as_prompt_contributions() -> Vec<PromptContribution> {
    vec![PromptContribution::guidance(
        "Exploration",
        "Keep investigation bounded: start with agent_nameed `grep`, `read_file`, and small shell probes. Avoid repo-wide expensive analyzers, full builds, full test suites, or unbounded scripts unless the task explicitly asks for them. Prefer extracting the needed facts over accumulating large raw observations.\n\nUse `llm_query` for semantic extraction, summarization, classification, judging, or transforming data already available in variables. It is one lightweight LLM call: no child session, no tools, no REPL loop. Shape it as `call llm_query { task: \"...\", inputs: { text: chunk }, output: { answer: \"str\" } }`. Omit `output` for a plain string.\n\nWhen a Required output schema is present, finish with `submit <expr>`.\n\nUse `continue_as` when context is tight or the current trajectory has gone stale. Pack `task` and `seed` with the concrete goal, constraints, paths, facts already learned, partial results, and next steps; leave failed attempts and bulky raw output behind.",
    )]
}

pub(crate) fn rlm_subagent_tool_definitions(capability_names: &[String]) -> Vec<ToolDefinition> {
    let example_capability = example_capability_name(capability_names);
    vec![
        llm_query_definition(vec![
            r#"summary = (call llm_query { task: "Summarize this log in one sentence.", inputs: { log: log_tail } })?"#.into(),
            r#"facts = (call llm_query { task: "Extract the root cause and confidence.", inputs: { log: log_tail, command: cmd }, output: { root_cause: "str", confidence: "float" } })?"#.into(),
            r#"Shape = Type { category: enum["config", "code", "network", "unknown"], evidence: list[str] }"#.into(),
            r#"typed = (call llm_query { task: "Classify the failure.", inputs: { log: log_tail }, output: Shape })?"#.into(),
        ]),
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
                r#"handles = (call list_async_handles {})?"#.into(),
                r#"results = parallel { auth: (await handles.subagent.auth_files)?, db: (await handles.subagent.db_migrations)? }"#.into(),
                // Untyped is fine for free-form prose results.
                format!(
                    r#"prose = call spawn_agent {{ agent_name: "audit_endpoints", task: "Skim the routes in api/ and flag any missing auth checks", capability: "{example_capability}" }}"#
                ),
            ],
        ),
        continue_as_definition(vec![
            r#"call continue_as { task: "filter the candidates by relevance to the user's query and submit the top 3", seed: { candidates: candidates, query: user_input_1 } }"#.into(),
        ]),
    ]
}

fn llm_query_definition(examples: Vec<String>) -> ToolDefinition {
    let mut definition = shared::llm_query_tool_definition();
    definition.description = "Run one lightweight LLM call and return its result. Use this for semantic extraction, summarization, classification, judging, or transforming data already available in variables. It does not create a child session, cannot use tools, and does not run a REPL loop. Use `spawn_agent` instead when the subproblem needs tool use, repo/file inspection, shell commands, edits, multi-step work, its own context window, or recursive subagents. `inputs` can be any structured value and is rendered for the model as data. `output` is optional and defaults to a string; when present, it accepts the same record descriptors and `Type { ... }` literals as `spawn_agent`.".to_string();
    definition.examples = examples;
    definition
}

fn spawn_agent_definition(capability_names: &[String], examples: Vec<String>) -> ToolDefinition {
    let cap_list = capability_list_for_description(capability_names);
    let description = format!(
        "Run a subagent and return its final result. Plain `call spawn_agent {{ ... }}` blocks until the child finishes. Use `start call spawn_agent {{ ... }}` for fan-out; it returns a generic lashlang async handle immediately, visible through `list_async_handles`. Pick `capability` from {cap_list}. `explore` is read-only and cannot recurse; `peer` has full edit + recurse powers. `output` defines the typed return shape. A child can fail terminally with `call submit_error {{ reason: \"...\" }}`; this tool returns an error with that reason. `agent_name` is auto-normalized."
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

fn llm_query_prompt(task: &str, inputs: &Value, output_schema: Option<&Value>) -> String {
    let mut sections = vec![format!("Task\n{task}")];
    if !inputs.is_null() {
        let inputs_pretty =
            serde_json::to_string_pretty(inputs).unwrap_or_else(|_| inputs.to_string());
        sections.push(format!("Inputs\n```json\n{inputs_pretty}\n```"));
    }
    if let Some(schema) = output_schema {
        let schema_pretty =
            serde_json::to_string_pretty(schema).unwrap_or_else(|_| schema.to_string());
        sections.push(format!(
            "Required output\nReturn `{{\"kind\":\"value\",\"value\":...,\"error\":null}}` where `value` matches this JSON Schema exactly:\n```json\n{schema_pretty}\n```\nIf the task cannot be answered from the supplied inputs, return `{{\"kind\":\"error\",\"value\":null,\"error\":\"...\"}}`."
        ));
    } else {
        sections.push(
            "Required output\nReturn `{\"kind\":\"value\",\"value\":\"...\",\"error\":null}` with the answer string. If the task cannot be answered from the supplied inputs, return `{\"kind\":\"error\",\"value\":null,\"error\":\"...\"}`."
                .to_string(),
        );
    }
    sections.join("\n\n")
}

fn llm_query_response_schema(output_schema: Option<&Value>) -> Value {
    let value_schema = output_schema
        .cloned()
        .unwrap_or_else(|| json!({ "type": "string" }));
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["kind", "value", "error"],
        "properties": {
            "kind": { "type": "string", "enum": ["value", "error"] },
            "value": {
                "anyOf": [
                    value_schema,
                    { "type": "null" }
                ]
            },
            "error": {
                "anyOf": [
                    { "type": "string" },
                    { "type": "null" }
                ]
            }
        }
    })
}

fn parse_llm_query_result(text: &str, schema: &Value) -> Result<Value, String> {
    let trimmed = text.trim();
    let value = serde_json::from_str::<Value>(trimmed).or_else(|err| {
        let Some(start) = trimmed.find(['{', '[', '"']) else {
            return Err(format!("llm_query returned non-JSON output: {err}"));
        };
        let end = trimmed
            .rfind(['}', ']', '"'])
            .ok_or_else(|| format!("llm_query returned malformed JSON output: {err}"))?;
        if end < start {
            return Err(format!("llm_query returned malformed JSON output: {err}"));
        }
        serde_json::from_str::<Value>(&trimmed[start..=end])
            .map_err(|parse_err| format!("llm_query returned malformed JSON output: {parse_err}"))
    })?;
    let compiled = jsonschema::JSONSchema::compile(schema)
        .map_err(|err| format!("llm_query output schema is invalid: {err}"))?;
    if let Err(errors) = compiled.validate(&value) {
        let message = errors
            .map(|err| err.to_string())
            .collect::<Vec<_>>()
            .join("; ");
        return Err(format!("llm_query output did not match schema: {message}"));
    }
    match value.get("kind").and_then(Value::as_str) {
        Some("value") => value
            .get("value")
            .cloned()
            .filter(|value| !value.is_null())
            .ok_or_else(|| "llm_query returned value result without value".to_string()),
        Some("error") => Err(value
            .get("error")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|message| !message.is_empty())
            .unwrap_or("llm_query returned an error")
            .to_string()),
        Some(other) => Err(format!("llm_query returned unknown result kind `{other}`")),
        None => Err("llm_query returned result without kind field".to_string()),
    }
}

fn continue_as_definition(examples: Vec<String>) -> ToolDefinition {
    let mut definition = shared::continue_as_tool_definition();
    definition.examples = examples;
    definition.description = "Tail-call: end this session and continue the work as a fresh successor with the same tools and a clean window. The successor inherits tool access and output schema but does not inherit prior conversation, trajectory, or globals — it sees only `task` and `seed`. Use when most of your trajectory has gone stale (failed attempts, large observations you've already extracted from) or context budget is tight. Pack `task` and `seed` with what the successor needs to keep going: concrete goals, constraints, discovered facts, IDs, tokens, file paths, partial results, and next steps. Be selective; leave dead ends behind. If useful, include a short slice of prior lashlang/repl history in `seed` so the successor can resume without rediscovering it.".to_string();
    definition
}

fn list_async_handles_definition() -> ToolDefinition {
    ToolDefinition::new(
        "list_async_handles",
        "List live lashlang async handles only. Returns `{ subagent: { name: handle }, tool: { id: handle } }`; terminal, awaited, or cancelled handles are omitted.",
        ToolDefinition::default_input_schema(),
        json!({
            "type": "object",
            "properties": {
                "subagent": { "type": "object" },
                "tool": { "type": "object" }
            },
            "required": ["subagent", "tool"]
        }),
    )
    .with_examples(vec![r#"handles = (call list_async_handles {})?"#.into()])
    .with_execution_mode(ToolExecutionMode::Parallel)
}
