use std::collections::HashMap;

use serde_json::{Value, json};

use lash::{
    InputItem, ModeExtras, ProgressSender, RlmCreateExtras, RlmTermination, SandboxMessage,
    SessionAppendNode, SessionEvent, SessionPluginMode, SessionStartPoint, ToolExecutionContext,
    ToolResult, TurnInput,
};

use super::{DelegateTools, policy::Tier};

impl DelegateTools {
    pub(super) async fn delegate(
        &self,
        args: &serde_json::Value,
        context: &ToolExecutionContext,
        progress: Option<&ProgressSender>,
    ) -> ToolResult {
        let task = args
            .get("task")
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        if task.is_empty() {
            return ToolResult::err(json!("Missing required parameter: task"));
        }

        let intelligence = args
            .get("intelligence")
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        let tier = match Tier::from_str(intelligence) {
            Some(tier) => tier,
            None => {
                return ToolResult::err(json!(
                    "Missing or invalid 'intelligence' parameter: must be \"low\", \"medium\", or \"high\""
                ));
            }
        };

        let vars = args.get("vars").cloned().unwrap_or_else(|| json!({}));
        if !vars.is_object() {
            return ToolResult::err(json!("`vars` must be a record (got non-object)"));
        }

        let output = match args.get("output") {
            Some(value) if value.is_null() => None,
            Some(value) if value.is_object() => Some(value.clone()),
            Some(_) => {
                return ToolResult::err(json!(
                    "Invalid `output`: expected a record describing the typed shape"
                ));
            }
            None => None,
        };

        let output_schema = match output.as_ref() {
            Some(output) => match build_output_schema(output) {
                Ok(schema) => Some(schema),
                Err(err) => {
                    return ToolResult::err(json!(format!("Invalid `output` schema: {err}")));
                }
            },
            None => None,
        };

        let session_id = uuid::Uuid::new_v4().to_string();
        let mut create_request =
            self.build_create_request(session_id.clone(), context.session_id.clone(), &tier);
        create_request.start = SessionStartPoint::Empty;
        create_request.plugin_mode = SessionPluginMode::InheritCurrent;
        create_request.initial_nodes = vars
            .as_object()
            .filter(|obj| !obj.is_empty())
            .map(|obj| {
                vec![SessionAppendNode::plugin(
                    lash::INTERNAL_RLM_GLOBALS_PATCH_PLUGIN_TYPE,
                    serde_json::to_value(lash::RlmGlobalsPatchPluginBody {
                        set: obj.clone(),
                        unset: Vec::new(),
                    })
                    .unwrap_or(serde_json::Value::Null),
                )]
            })
            .unwrap_or_default();
        if let Some(schema) = output_schema.clone() {
            let mut policy = create_request.policy.take().unwrap_or_default();
            policy.execution_mode = lash::ExecutionMode::Rlm;
            create_request.policy = Some(policy);
            create_request.mode_extras = ModeExtras::Rlm(RlmCreateExtras {
                termination: RlmTermination::Finish {
                    schema: Some(schema),
                },
            });
        }

        let session = match context.host.create_session(create_request).await {
            Ok(session) => session,
            Err(err) => {
                return ToolResult::err_fmt(format_args!("Failed to create child session: {err}"));
            }
        };

        let user_content = render_delegate_initial_message(task, &vars, output_schema.as_ref());
        let mut turn = match context
            .host
            .start_turn_stream(
                &session.session_id,
                TurnInput {
                    items: vec![InputItem::Text { text: user_content }],
                    image_blobs: HashMap::new(),
                    user_input: None,
                    mode: None,
                },
            )
            .await
        {
            Ok(turn) => turn,
            Err(err) => {
                let _ = context.host.close_session(&session.session_id).await;
                return ToolResult::err_fmt(format_args!("Failed to start child session: {err}"));
            }
        };

        if let Some(progress) = progress {
            let _ = progress.send(SandboxMessage {
                text: json!({
                    "name": "delegate",
                    "task": task,
                    "model": session.policy.model,
                    "model_variant": session.policy.model_variant,
                    "id": session.session_id,
                    "parent_session_id": session.parent_session_id,
                })
                .to_string(),
                kind: "delegate_start".into(),
            });
        }

        let mut final_message: Option<String> = None;
        let mut current_prose = String::new();
        let cancellation = context.cancellation_token.clone();

        loop {
            tokio::select! {
                biased;
                _ = async {
                    if let Some(token) = &cancellation {
                        token.cancelled().await;
                    } else {
                        std::future::pending::<()>().await;
                    }
                } => {
                    let _ = context.host.cancel_turn(&turn.turn_id).await;
                    break;
                }
                maybe_event = turn.events.recv() => {
                    let Some(event) = maybe_event else {
                        break;
                    };
                    match event {
                        SessionEvent::TextDelta { content } => {
                            current_prose.push_str(&content);
                            if let Some(progress) = progress {
                                let _ = progress.send(SandboxMessage {
                                    text: content,
                                    kind: "tool_output".into(),
                                });
                            }
                        }
                        SessionEvent::Message { text, kind } => {
                            if kind == "final" {
                                final_message = Some(text);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        match context.host.await_turn(&turn.turn_id).await {
            Ok(turn) => {
                let result_value = if let Some(value) = turn.typed_finish.clone() {
                    value
                } else if let Some(message) = final_message {
                    json!(message)
                } else if !turn.assistant_output.safe_text.trim().is_empty() {
                    json!(turn.assistant_output.safe_text.trim().to_string())
                } else if !current_prose.trim().is_empty() {
                    json!(current_prose.trim().to_string())
                } else {
                    json!(turn.assistant_output.raw_text.trim().to_string())
                };
                let status = match turn.status {
                    lash::TurnStatus::Completed => "completed",
                    lash::TurnStatus::Interrupted => "interrupted",
                    lash::TurnStatus::Failed => "failed",
                };
                let tool_call_summaries = turn
                    .state
                    .projected_tool_calls()
                    .iter()
                    .map(|tool_call| {
                        json!({
                            "tool": tool_call.tool,
                            "success": tool_call.success,
                            "duration_ms": tool_call.duration_ms,
                        })
                    })
                    .collect::<Vec<_>>();

                ToolResult::ok(json!({
                    "result": result_value,
                    "status": status,
                    "session": {
                        "id": session.session_id,
                        "parent_session_id": session.parent_session_id,
                        "task": task,
                        "iterations": turn.state.iteration,
                        "tool_calls": turn.state.projected_tool_calls().len(),
                        "tool_call_details": tool_call_summaries,
                        "model": turn.state.policy.model,
                        "model_variant": turn.state.policy.model_variant,
                        "token_usage": {
                            "input_tokens": turn.token_usage.input_tokens,
                            "output_tokens": turn.token_usage.output_tokens,
                            "cached_input_tokens": turn.token_usage.cached_input_tokens,
                            "reasoning_tokens": turn.token_usage.reasoning_tokens,
                            "total_tokens": turn.token_usage.total(),
                        },
                    },
                }))
            }
            Err(err) => ToolResult::ok(json!({
                "result": "",
                "status": "failed",
                "error": err.to_string(),
            })),
        }
    }
}

fn build_output_schema(output: &Value) -> Result<Value, String> {
    let obj = output
        .as_object()
        .ok_or_else(|| "expected a record".to_string())?;
    if obj.is_empty() {
        return Err("at least one output field is required".to_string());
    }
    let mut properties = serde_json::Map::new();
    let mut required = Vec::with_capacity(obj.len());
    for (name, type_descriptor) in obj {
        let type_str = type_descriptor
            .as_str()
            .ok_or_else(|| format!("field `{name}`: type descriptor must be a string"))?;
        let property_schema = type_descriptor_to_json_schema(type_str)
            .map_err(|err| format!("field `{name}`: {err}"))?;
        properties.insert(name.clone(), property_schema);
        required.push(Value::String(name.clone()));
    }
    Ok(json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false,
    }))
}

fn type_descriptor_to_json_schema(descriptor: &str) -> Result<Value, String> {
    let trimmed = descriptor.trim();
    let scalar = |ty: &str| -> Result<Value, String> {
        match ty {
            "str" | "string" => Ok(json!({"type": "string"})),
            "int" | "integer" => Ok(json!({"type": "integer"})),
            "float" | "number" => Ok(json!({"type": "number"})),
            "bool" | "boolean" => Ok(json!({"type": "boolean"})),
            "record" | "dict" | "object" => Ok(json!({"type": "object"})),
            other => Err(format!("unknown scalar type `{other}`")),
        }
    };
    if let Some(inner) = trimmed
        .strip_prefix("list[")
        .and_then(|rest| rest.strip_suffix(']'))
    {
        let item_schema = scalar(inner.trim()).map_err(|err| format!("inside list[]: {err}"))?;
        return Ok(json!({"type": "array", "items": item_schema}));
    }
    scalar(trimmed)
}

fn render_delegate_initial_message(task: &str, vars: &Value, schema: Option<&Value>) -> String {
    let mut sections = vec![task.to_string()];
    if let Some(obj) = vars.as_object()
        && !obj.is_empty()
    {
        let vars_pretty = serde_json::to_string_pretty(vars).unwrap_or_else(|_| vars.to_string());
        sections.push(format!("## Inputs\n\n```json\n{vars_pretty}\n```"));
    }
    if let Some(schema) = schema {
        let schema_pretty =
            serde_json::to_string_pretty(schema).unwrap_or_else(|_| schema.to_string());
        sections.push(format!(
            "## Required output\n\nWhen done, end the task by calling `finish <expr>` from inside a fenced ```lashlang block. The value MUST match this JSON Schema exactly:\n\n```json\n{schema_pretty}\n```\n\nIf your `finish` value fails validation, you'll see the error on the next iteration and can retry."
        ));
    }
    sections.join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_output_schema_handles_scalars() {
        let output = json!({
            "answer": "str",
            "confidence": "float",
            "count": "int",
            "ok": "bool",
        });
        let schema = build_output_schema(&output).expect("schema");
        assert_eq!(schema["type"], json!("object"));
        assert_eq!(schema["properties"]["answer"]["type"], json!("string"));
        assert_eq!(schema["properties"]["confidence"]["type"], json!("number"));
        assert_eq!(schema["properties"]["count"]["type"], json!("integer"));
        assert_eq!(schema["properties"]["ok"]["type"], json!("boolean"));
        assert_eq!(schema["additionalProperties"], json!(false));
    }

    #[test]
    fn build_output_schema_handles_list_of_scalar() {
        let output = json!({"items": "list[str]"});
        let schema = build_output_schema(&output).expect("schema");
        assert_eq!(schema["properties"]["items"]["type"], json!("array"));
        assert_eq!(
            schema["properties"]["items"]["items"]["type"],
            json!("string")
        );
    }

    #[test]
    fn build_output_schema_rejects_unknown_type() {
        let output = json!({"x": "matrix[float]"});
        let err = build_output_schema(&output).expect_err("should fail");
        assert!(err.contains("unknown scalar type"), "{err}");
    }

    #[test]
    fn render_delegate_initial_message_includes_vars_and_schema() {
        let task = "Extract the longest line";
        let vars = json!({"path": "src/main.rs", "limit": 200});
        let schema = json!({"type": "object", "properties": {"line": {"type": "string"}}});
        let rendered = render_delegate_initial_message(task, &vars, Some(&schema));
        assert!(rendered.contains("Extract the longest line"));
        assert!(rendered.contains("## Inputs"));
        assert!(rendered.contains("\"path\": \"src/main.rs\""));
        assert!(rendered.contains("## Required output"));
        assert!(rendered.contains("`finish <expr>`"));
    }
}
