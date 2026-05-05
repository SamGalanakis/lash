use std::sync::Arc;

use async_trait::async_trait;
use lash::plugin::{PluginError, PluginFactory, PluginSessionContext};
use lash::{
    DirectJsonSchema, DirectMessage, DirectOutputSpec, DirectPart, DirectRequest, DirectRole,
    PluginSpec, PluginSpecFactory, ProgressSender, PromptContribution, SessionToolAccess,
    ToolDefinition, ToolExecutionContext, ToolExecutionMode, ToolProvider, ToolResult,
};
use serde_json::{Value, json};

pub struct LlmToolsPluginFactory;

impl PluginFactory for LlmToolsPluginFactory {
    fn id(&self) -> &'static str {
        "llm_tools"
    }

    fn build(
        &self,
        ctx: &PluginSessionContext,
    ) -> Result<Arc<dyn lash::SessionPlugin>, PluginError> {
        let is_rlm = ctx.execution_mode == lash::ExecutionMode::new("rlm");
        let provider: Option<Arc<dyn ToolProvider>> =
            is_rlm.then(|| Arc::new(LlmToolsProvider) as Arc<dyn ToolProvider>);
        let prompt_contributions =
            if is_rlm && tool_callable_from_authority(&ctx.tool_access, "llm_query") {
                llm_prompt_contributions()
            } else {
                Vec::new()
            };

        PluginSpecFactory::new(
            "llm_tools",
            Arc::new(move |_ctx| {
                let contributions = prompt_contributions.clone();
                let mut spec = PluginSpec::new().with_prompt_contributor(Arc::new(move |_ctx| {
                    let contributions = contributions.clone();
                    Box::pin(async move { Ok(contributions) })
                }));
                if let Some(provider) = provider.as_ref() {
                    spec = spec.with_tool_provider(Arc::clone(provider));
                }
                Ok(spec)
            }),
        )
        .build(ctx)
    }
}

struct LlmToolsProvider;

impl LlmToolsProvider {
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
        let policy = current_snapshot.policy;
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
}

#[async_trait]
impl ToolProvider for LlmToolsProvider {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![llm_query_tool_definition()]
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
            "llm_query" => self.llm_query(args, context).await,
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

pub fn llm_query_tool_definition() -> ToolDefinition {
    tool_definition(
        "llm_query",
        "Run a one-shot LLM prompt and return its result. The `task` plus everything in `inputs` is rendered into that single prompt; the call cannot use tools, inspect files, or gather more context. Use this for extracting information from supplied data, classification, summarization, judging, or transformation. `inputs` can be any structured value. `output` is optional and defaults to a string; when present, it requests structured output using record descriptors or `Type { ... }` literals.",
        llm_query_input_schema(),
        Vec::new(),
        ToolExecutionMode::Parallel,
    )
    .with_output_from_input_schema("output", Some(json!({ "type": "string" })))
}

pub fn parse_output_schema(value: Option<&Value>) -> Result<Option<Value>, String> {
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

fn llm_query_input_schema() -> Value {
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

fn llm_prompt_contributions() -> Vec<PromptContribution> {
    vec![PromptContribution::guidance(
        "LLM tools",
        "Use `llm_query` when you already have the relevant data in variables and need one semantic pass over it: extract information, classify, summarize, judge, or transform. Its `task` and `inputs` become a single prompt; it has no tools or follow-up loop. Shape it as `call llm_query { task: \"...\", inputs: { text: chunk }, output: { answer: \"str\" } }`. Omit `output` for plain text.",
    )]
}

fn tool_callable_from_authority(access: &SessionToolAccess, name: &str) -> bool {
    if access.hides(name) {
        return false;
    }
    access.tools.is_empty() || access.tools.iter().any(|tool| tool.name == name)
}

fn llm_query_prompt(task: &str, inputs: &Value, output_schema: Option<&Value>) -> String {
    let mut sections = Vec::new();
    sections.push(format!("Task:\n{task}"));
    sections.push(format!(
        "Inputs:\n```json\n{}\n```",
        serde_json::to_string_pretty(inputs).unwrap_or_else(|_| inputs.to_string())
    ));
    if let Some(schema) = output_schema {
        sections.push(format!(
            "Return `kind=\"value\"` with `value` matching this JSON Schema, or `kind=\"error\"` with a concise error if the task cannot be answered from the supplied inputs:\n```json\n{}\n```",
            serde_json::to_string_pretty(schema).unwrap_or_else(|_| schema.to_string())
        ));
    } else {
        sections.push("Return `kind=\"value\"` with a concise string `value`, or `kind=\"error\"` with a concise error if the task cannot be answered from the supplied inputs.".to_string());
    }
    sections.join("\n\n")
}

fn llm_query_response_schema(output_schema: Option<&Value>) -> Value {
    let value_schema = output_schema
        .cloned()
        .unwrap_or_else(|| json!({"type": "string"}));
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
    if trimmed.is_empty() {
        return Err("llm_query returned empty output".to_string());
    }
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

fn tool_definition(
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

fn required_string(args: &Value, key: &str) -> Result<String, String> {
    args.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| format!("missing required parameter: {key}"))
}

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

fn finalise_tool_result(result: Result<Value, String>) -> ToolResult {
    match result {
        Ok(value) => ToolResult::ok(value),
        Err(err) => ToolResult::err(json!(err)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    use async_trait::async_trait;
    use lash::plugin::{
        DirectCompletionHost, DynamicToolHost, MonitorHost, PluginError, PromptHost,
        SessionGraphHost, SessionHandle, SessionLifecycleHost, SessionSnapshotHost,
        SessionTurnHandle, TaskHost, ToolCatalogHost, TraceHost, TurnHost,
    };
    use lash::{PersistedSessionState, SessionCreateRequest, ToolExecutionContext, TurnInput};

    #[derive(Default)]
    struct DirectCompletionManager {
        snapshot: PersistedSessionState,
        requests: Mutex<Vec<(lash::DirectRequest, String)>>,
        response_text: String,
    }

    #[async_trait]
    impl SessionSnapshotHost for DirectCompletionManager {
        async fn snapshot_current(&self) -> Result<PersistedSessionState, PluginError> {
            Ok(self.snapshot.clone())
        }

        async fn snapshot_session(
            &self,
            _session_id: &str,
        ) -> Result<PersistedSessionState, PluginError> {
            Ok(self.snapshot.clone())
        }
    }

    #[async_trait]
    impl ToolCatalogHost for DirectCompletionManager {
        async fn tool_catalog(
            &self,
            _session_id: &str,
        ) -> Result<Vec<serde_json::Value>, PluginError> {
            Ok(Vec::new())
        }
    }

    impl DynamicToolHost for DirectCompletionManager {}

    #[async_trait]
    impl SessionLifecycleHost for DirectCompletionManager {
        async fn create_session(
            &self,
            _request: SessionCreateRequest,
        ) -> Result<SessionHandle, PluginError> {
            Err(PluginError::Session("not used".to_string()))
        }

        async fn close_session(&self, _session_id: &str) -> Result<(), PluginError> {
            Ok(())
        }
    }

    #[async_trait]
    impl TurnHost for DirectCompletionManager {
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

    #[async_trait]
    impl DirectCompletionHost for DirectCompletionManager {
        async fn direct_completion(
            &self,
            request: lash::DirectRequest,
            usage_source: &str,
        ) -> Result<lash::DirectCompletion, PluginError> {
            self.requests
                .lock()
                .expect("requests")
                .push((request, usage_source.to_string()));
            Ok(lash::DirectCompletion {
                text: self.response_text.clone(),
                usage: lash::TokenUsage::default(),
            })
        }
    }

    impl TaskHost for DirectCompletionManager {}
    impl MonitorHost for DirectCompletionManager {}
    impl SessionGraphHost for DirectCompletionManager {}
    impl PromptHost for DirectCompletionManager {}
    impl TraceHost for DirectCompletionManager {}

    #[test]
    fn llm_definitions_include_llm_query_only() {
        let provider = LlmToolsProvider;
        let names = provider
            .definitions()
            .into_iter()
            .map(|tool| tool.name)
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["llm_query"]);
    }

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

    #[tokio::test]
    async fn llm_query_uses_current_policy_and_direct_completion() {
        let manager = Arc::new(DirectCompletionManager {
            snapshot: PersistedSessionState {
                policy: lash::SessionPolicy {
                    model: "root-model".to_string(),
                    model_variant: Some("fast".to_string()),
                    execution_mode: lash::ExecutionMode::new("rlm"),
                    ..lash::SessionPolicy::default()
                },
                ..PersistedSessionState::default()
            },
            requests: Mutex::new(Vec::new()),
            response_text:
                r#"{"kind":"value","value":{"root_cause":"missing config","confidence":0.8},"error":null}"#
                    .to_string(),
        });
        let provider = LlmToolsProvider;
        let context = ToolExecutionContext {
            session_id: "parent".to_string(),
            host: manager.clone(),
            cancellation_token: None,
            async_task_id: None,
        };

        let result = provider
            .execute_with_context(
                "llm_query",
                &json!({
                    "task": "extract root cause",
                    "inputs": { "log": "failed" },
                    "output": { "root_cause": "str", "confidence": "float" }
                }),
                &context,
            )
            .await;

        assert!(result.success, "{:?}", result.result);
        assert_eq!(result.result["root_cause"], json!("missing config"));
        assert_eq!(result.result["confidence"], json!(0.8));

        let requests = manager.requests.lock().expect("requests");
        assert_eq!(requests.len(), 1);
        let (request, usage_source) = &requests[0];
        assert_eq!(usage_source, "llm_query");
        assert_eq!(request.model, "root-model");
        assert_eq!(request.model_variant.as_deref(), Some("fast"));
        assert!(matches!(
            request.output,
            lash::DirectOutputSpec::JsonSchema(_)
        ));
        let prompt = request
            .messages
            .iter()
            .flat_map(|message| message.parts.iter())
            .filter_map(|part| match part {
                lash::DirectPart::Text(text) => Some(text.as_str()),
                lash::DirectPart::Image(_) => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(prompt.contains("extract root cause"));
        assert!(prompt.contains("\"log\": \"failed\""));
    }

    #[tokio::test]
    async fn llm_query_error_result_fails_tool_call() {
        let manager = Arc::new(DirectCompletionManager {
            snapshot: PersistedSessionState {
                policy: lash::SessionPolicy {
                    execution_mode: lash::ExecutionMode::new("rlm"),
                    ..lash::SessionPolicy::default()
                },
                ..PersistedSessionState::default()
            },
            requests: Mutex::new(Vec::new()),
            response_text: r#"{"kind":"error","value":null,"error":"missing required evidence"}"#
                .to_string(),
        });
        let provider = LlmToolsProvider;
        let context = ToolExecutionContext {
            session_id: "parent".to_string(),
            host: manager,
            cancellation_token: None,
            async_task_id: None,
        };

        let result = provider
            .execute_with_context(
                "llm_query",
                &json!({ "task": "answer from missing evidence" }),
                &context,
            )
            .await;

        assert!(!result.success);
        assert_eq!(result.result, json!("missing required evidence"));
    }
}
