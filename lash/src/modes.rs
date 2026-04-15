use lash_sansio::llm::types::{LlmOutputPart, LlmResponse};
use lash_sansio::sansio::{
    CheckpointResumeAction, CompletedToolCall, DriverAction, DriverContextView, PendingToolCall,
    ProtocolDriverHandle, RlmTermination, WaitingExecState, WaitingLlmState, driver_state,
};
use lash_sansio::session_model::message::{PartAttachment, data_url_for_bytes};
use lash_sansio::session_model::{
    Message, MessageRole, Part, PartKind, PruneState, SessionEvent, format_tool_result_content,
    fresh_message_id, make_error_event, reassign_part_ids,
};
use lash_sansio::{CheckpointKind, ToolCallRecord, ToolImage};
use serde_json::Value;

pub type ProtocolDriverRef = std::sync::Arc<dyn ProtocolDriverHandle>;

pub struct StandardDriver;
pub struct RlmDriver;

#[derive(Default)]
struct RlmDriverState {
    tool_calls: Vec<ToolCallRecord>,
    images: Vec<ToolImage>,
    combined_output: String,
    exec_error: Option<String>,
    executed_code: Option<String>,
    terminal_finish: Option<serde_json::Value>,
}

struct FenceExtraction {
    code: String,
    had_extra_fences: bool,
}

impl ProtocolDriverHandle for StandardDriver {
    fn prepare_iteration(&self, ctx: DriverContextView<'_>) -> Vec<DriverAction> {
        vec![DriverAction::StartLlm {
            request: ctx.build_llm_request(true),
            driver_state: None,
        }]
    }

    fn handle_llm_success(
        &self,
        ctx: DriverContextView<'_>,
        _waiting: WaitingLlmState,
        llm_response: LlmResponse,
        text_streamed: bool,
    ) -> Vec<DriverAction> {
        let response_parts = normalized_response_parts(&llm_response);
        let mut assistant_text = String::new();
        let mut tool_calls: Vec<(String, String, String)> = Vec::new();
        let mut actions = Vec::new();

        for part in response_parts {
            match part {
                LlmOutputPart::Text { text } => {
                    if !text.is_empty() {
                        let previous_len = assistant_text.len();
                        append_assistant_text_part(&mut assistant_text, &text);
                        if !text_streamed {
                            actions.push(DriverAction::Emit(SessionEvent::TextDelta {
                                content: assistant_text[previous_len..].to_string(),
                            }));
                        }
                    }
                }
                LlmOutputPart::ToolCall {
                    call_id,
                    tool_name,
                    input_json,
                } => {
                    tool_calls.push((call_id, tool_name, input_json));
                }
            }
        }

        actions.push(DriverAction::Emit(SessionEvent::LlmResponse {
            iteration: ctx.iteration(),
            content: assistant_text.clone(),
            duration_ms: 0,
        }));

        if tool_calls.is_empty() {
            if assistant_text.trim().is_empty() {
                actions.push(DriverAction::Emit(make_error_event(
                    "llm_provider",
                    Some("empty_response"),
                    "Model returned no assistant text or tool calls.",
                    None,
                )));
                actions.push(DriverAction::Finish);
                return actions;
            }

            actions.push(DriverAction::AppendMessages(vec![assistant_prose_message(
                assistant_text,
            )]));
            actions.push(DriverAction::StartCheckpoint {
                checkpoint: CheckpointKind::BeforeCompletion,
                on_empty: CheckpointResumeAction::Finish,
            });
            return actions;
        }

        let asst_id = fresh_message_id();
        let mut assistant_parts = Vec::new();
        if !assistant_text.trim().is_empty() {
            assistant_parts.push(Part {
                id: format!("{}.p{}", asst_id, assistant_parts.len()),
                kind: PartKind::Prose,
                content: assistant_text,
                attachment: None,
                tool_call_id: None,
                tool_name: None,
                prune_state: PruneState::Intact,
            });
        }

        let mut calls = Vec::new();
        for (call_id, tool_name, input_json) in tool_calls {
            assistant_parts.push(Part {
                id: format!("{}.p{}", asst_id, assistant_parts.len()),
                kind: PartKind::ToolCall,
                content: input_json.clone(),
                attachment: None,
                tool_call_id: Some(call_id.clone()),
                tool_name: Some(tool_name.clone()),
                prune_state: PruneState::Intact,
            });

            let args = serde_json::from_str::<Value>(&input_json)
                .unwrap_or_else(|_| serde_json::json!({}));
            calls.push(PendingToolCall {
                call_id,
                tool_name,
                args,
            });
        }

        if !assistant_parts.is_empty() {
            actions.push(DriverAction::AppendMessages(vec![Message {
                id: asst_id,
                role: MessageRole::Assistant,
                parts: assistant_parts,
                user_input: None,
                origin: None,
            }]));
        }

        actions.push(DriverAction::StartTools { calls });
        actions
    }

    fn handle_tool_results(
        &self,
        ctx: DriverContextView<'_>,
        completed: Vec<CompletedToolCall>,
    ) -> Vec<DriverAction> {
        let mut actions = Vec::new();
        let mut result_parts = Vec::new();

        for outcome in completed {
            result_parts.push(Part {
                id: String::new(),
                kind: PartKind::ToolResult,
                content: format_tool_result_content(
                    outcome.model_result.success,
                    &outcome.model_result.result,
                ),
                attachment: None,
                tool_call_id: Some(outcome.call_id.clone()),
                tool_name: Some(outcome.tool_name.clone()),
                prune_state: PruneState::Intact,
            });

            for (image_offset, image) in outcome.model_result.images.into_iter().enumerate() {
                result_parts.push(Part {
                    id: String::new(),
                    kind: PartKind::Text,
                    content: format!("[Tool image: {}]", image.label),
                    attachment: None,
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: PruneState::Intact,
                });
                result_parts.push(Part {
                    id: String::new(),
                    kind: PartKind::Image,
                    content: String::new(),
                    attachment: Some(PartAttachment {
                        mime: image.mime.clone(),
                        url: data_url_for_bytes(&image.mime, &image.data),
                        filename: Some(format!("tool-image-{image_offset}")),
                    }),
                    tool_call_id: None,
                    tool_name: None,
                    prune_state: PruneState::Intact,
                });
            }
        }

        if !result_parts.is_empty() {
            let user_id = fresh_message_id();
            reassign_part_ids(&user_id, &mut result_parts);
            actions.push(DriverAction::AppendMessages(vec![Message {
                id: user_id,
                role: MessageRole::User,
                parts: result_parts,
                user_input: None,
                origin: None,
            }]));
        }

        actions.push(DriverAction::AdvanceIteration);
        let next_iteration = ctx.iteration() + 1;
        if let Some(max_turns) = ctx.max_turns()
            && next_iteration >= ctx.run_offset() + max_turns
        {
            actions.push(DriverAction::AppendMessages(vec![
                turn_limit_exhausted_message(max_turns),
            ]));
            actions.push(DriverAction::Finish);
            return actions;
        }

        actions.push(DriverAction::StartCheckpoint {
            checkpoint: CheckpointKind::AfterWork,
            on_empty: CheckpointResumeAction::PrepareIteration,
        });
        actions
    }

    fn handle_exec_result(
        &self,
        _ctx: DriverContextView<'_>,
        _waiting: WaitingExecState,
        _result: Result<lash_sansio::ExecResponse, String>,
    ) -> Vec<DriverAction> {
        Vec::new()
    }
}

impl ProtocolDriverHandle for RlmDriver {
    fn prepare_iteration(&self, ctx: DriverContextView<'_>) -> Vec<DriverAction> {
        vec![DriverAction::StartLlm {
            request: ctx.build_llm_request(false),
            driver_state: Some(driver_state(RlmDriverState::default())),
        }]
    }

    fn handle_llm_success(
        &self,
        ctx: DriverContextView<'_>,
        mut waiting: WaitingLlmState,
        llm_response: LlmResponse,
        _text_streamed: bool,
    ) -> Vec<DriverAction> {
        let mut actions = vec![DriverAction::Emit(SessionEvent::LlmResponse {
            iteration: ctx.iteration(),
            content: llm_response.full_text.clone(),
            duration_ms: 0,
        })];

        let mut assistant_text = String::new();
        for part in normalized_response_parts(&llm_response) {
            if let LlmOutputPart::Text { text } = part {
                append_assistant_text_part(&mut assistant_text, &text);
            }
        }

        if assistant_text.trim().is_empty() {
            actions.push(DriverAction::Emit(make_error_event(
                "llm_provider",
                Some("empty_response"),
                "Model returned no assistant text.",
                None,
            )));
            actions.push(DriverAction::Finish);
            return actions;
        }

        let extraction = extract_first_lashlang_fence(&assistant_text);
        let Some(fence) = extraction else {
            match ctx.rlm_termination() {
                RlmTermination::ProseWithoutFence => {
                    actions.push(DriverAction::AppendMessages(vec![assistant_prose_message(
                        assistant_text,
                    )]));
                    actions.push(DriverAction::StartCheckpoint {
                        checkpoint: CheckpointKind::BeforeCompletion,
                        on_empty: CheckpointResumeAction::Finish,
                    });
                }
                RlmTermination::Finish { .. } => {
                    actions.push(DriverAction::AppendMessages(vec![
                        assistant_prose_message(assistant_text),
                        typed_rlm_finish_reminder_message(),
                    ]));
                    actions.push(DriverAction::AdvanceIteration);
                    actions.push(DriverAction::StartCheckpoint {
                        checkpoint: CheckpointKind::AfterWork,
                        on_empty: CheckpointResumeAction::PrepareIteration,
                    });
                }
            }
            return actions;
        };

        let _ = fence.had_extra_fences;

        let mut state = waiting
            .take_driver_state::<RlmDriverState>()
            .unwrap_or_default();
        state.executed_code = Some(fence.code.clone());

        actions.push(DriverAction::AppendMessages(vec![assistant_prose_message(
            assistant_text,
        )]));
        actions.push(DriverAction::StartExec {
            code: fence.code,
            driver_state: driver_state(state),
        });
        actions
    }

    fn handle_tool_results(
        &self,
        _ctx: DriverContextView<'_>,
        _completed: Vec<CompletedToolCall>,
    ) -> Vec<DriverAction> {
        Vec::new()
    }

    fn handle_exec_result(
        &self,
        ctx: DriverContextView<'_>,
        waiting: WaitingExecState,
        result: Result<lash_sansio::ExecResponse, String>,
    ) -> Vec<DriverAction> {
        let mut state = waiting
            .into_driver_state::<RlmDriverState>()
            .unwrap_or_default();
        let mut actions = Vec::new();

        match result {
            Ok(response) => {
                for tool_call in &response.tool_calls {
                    actions.push(DriverAction::Emit(SessionEvent::ToolCall {
                        call_id: None,
                        name: tool_call.tool.clone(),
                        args: tool_call.args.clone(),
                        result: tool_call.result.clone(),
                        success: tool_call.success,
                        duration_ms: tool_call.duration_ms,
                    }));
                }
                state.tool_calls.extend(response.tool_calls);
                state.images.extend(response.images);
                if !response.output.is_empty() {
                    state.combined_output.push_str(&response.output);
                }
                for observation in response.observations {
                    if !observation.is_empty() {
                        if !state.combined_output.is_empty()
                            && !state.combined_output.ends_with('\n')
                        {
                            state.combined_output.push('\n');
                        }
                        state.combined_output.push_str(&observation);
                        if !state.combined_output.ends_with('\n') {
                            state.combined_output.push('\n');
                        }
                    }
                }
                if let Some(raw_error) = response.error {
                    state.exec_error = Some(raw_error);
                }
                if let Some(finish_value) = response.terminal_finish {
                    state.terminal_finish = Some(finish_value);
                }
            }
            Err(error) => {
                state.exec_error = Some(error);
            }
        }

        if let Some(finish_value) = &state.terminal_finish
            && let RlmTermination::Finish { schema } = ctx.rlm_termination()
        {
            if let Some(schema) = schema
                && let Err(error_text) = validate_finish_value(finish_value, schema)
            {
                actions.push(DriverAction::AppendMessages(vec![
                    typed_rlm_schema_error_message(&error_text),
                ]));
                actions.push(DriverAction::AdvanceIteration);
                actions.push(DriverAction::StartCheckpoint {
                    checkpoint: CheckpointKind::AfterWork,
                    on_empty: CheckpointResumeAction::PrepareIteration,
                });
                return actions;
            }

            let rendered = match finish_value {
                serde_json::Value::String(text) => text.clone(),
                other => serde_json::to_string_pretty(other).unwrap_or_else(|_| other.to_string()),
            };
            actions.push(DriverAction::AppendMessages(vec![assistant_prose_message(
                rendered,
            )]));
            actions.push(DriverAction::Emit(SessionEvent::TypedFinish {
                value: finish_value.clone(),
            }));
            actions.push(DriverAction::StartCheckpoint {
                checkpoint: CheckpointKind::BeforeCompletion,
                on_empty: CheckpointResumeAction::Finish,
            });
            return actions;
        }

        let mut result_payload = serde_json::json!({
            "observations": state.combined_output,
            "tool_calls": state.tool_calls,
            "error": state.exec_error,
        });
        if !state.images.is_empty() {
            let images = state
                .images
                .iter()
                .map(|img| {
                    serde_json::json!({
                        "label": img.label,
                        "mime": img.mime,
                    })
                })
                .collect::<Vec<_>>();
            result_payload["images"] = serde_json::Value::Array(images);
        }

        let success = result_payload
            .get("error")
            .is_none_or(|value| value.is_null());
        let result_call_id = format!("rlm_exec_{}", ctx.iteration());
        let execute_args = state
            .executed_code
            .as_ref()
            .map(|code| serde_json::json!({ "code": code }))
            .unwrap_or_else(|| serde_json::json!({}));
        actions.push(DriverAction::Emit(SessionEvent::ToolCall {
            call_id: Some(result_call_id),
            name: "execute_lashlang".to_string(),
            args: execute_args,
            result: result_payload.clone(),
            success,
            duration_ms: 0,
        }));
        actions.push(DriverAction::AppendMessages(vec![rlm_result_message(
            success,
            &result_payload,
            &state.images,
        )]));
        actions.push(DriverAction::AdvanceIteration);
        if ctx.should_force_exit_after_grace_turn() {
            actions.push(DriverAction::Finish);
            return actions;
        }
        actions.push(DriverAction::ScheduleTurnLimitFinal);
        actions.push(DriverAction::StartCheckpoint {
            checkpoint: CheckpointKind::AfterWork,
            on_empty: CheckpointResumeAction::PrepareIteration,
        });
        actions
    }
}

fn normalized_response_parts(llm_response: &LlmResponse) -> Vec<LlmOutputPart> {
    if llm_response.parts.is_empty() && !llm_response.full_text.is_empty() {
        vec![LlmOutputPart::Text {
            text: llm_response.full_text.clone(),
        }]
    } else {
        llm_response.parts.clone()
    }
}

fn assistant_prose_message(content: String) -> Message {
    let id = fresh_message_id();
    Message {
        id: id.clone(),
        role: MessageRole::Assistant,
        parts: vec![Part {
            id: format!("{id}.p0"),
            kind: PartKind::Prose,
            content,
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            prune_state: PruneState::Intact,
        }],
        user_input: None,
        origin: None,
    }
}

fn typed_rlm_finish_reminder_message() -> Message {
    let id = fresh_message_id();
    Message {
        id: id.clone(),
        role: MessageRole::User,
        parts: vec![Part {
            id: format!("{id}.p0"),
            kind: PartKind::Text,
            content: "[runtime] You're in a typed RLM session. End by emitting a fenced ```lashlang block that calls `finish <expr>` with a value matching the required output schema. Prose-only replies are not accepted as the final answer here.".to_string(),
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            prune_state: PruneState::Intact,
        }],
        user_input: None,
        origin: None,
    }
}

fn typed_rlm_schema_error_message(error_text: &str) -> Message {
    let id = fresh_message_id();
    Message {
        id: id.clone(),
        role: MessageRole::User,
        parts: vec![Part {
            id: format!("{id}.p0"),
            kind: PartKind::Text,
            content: format!(
                "[runtime] Your `finish` value didn't match the required output schema:\n{error_text}\n\nFix the value and call `finish <corrected>` from another fenced ```lashlang block."
            ),
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            prune_state: PruneState::Intact,
        }],
        user_input: None,
        origin: None,
    }
}

fn turn_limit_exhausted_message(max_turns: usize) -> Message {
    let id = fresh_message_id();
    Message {
        id: id.clone(),
        role: MessageRole::System,
        parts: vec![Part {
            id: format!("{id}.p0"),
            kind: PartKind::Error,
            content: format!("Turn limit reached ({max_turns}) before a final assistant response."),
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            prune_state: PruneState::Intact,
        }],
        user_input: None,
        origin: None,
    }
}

fn rlm_result_message(success: bool, result_payload: &Value, images: &[ToolImage]) -> Message {
    let id = fresh_message_id();
    let mut parts = vec![Part {
        id: format!("{id}.p0"),
        kind: PartKind::Text,
        content: format_repl_result_text(success, result_payload),
        attachment: None,
        tool_call_id: None,
        tool_name: None,
        prune_state: PruneState::Intact,
    }];
    for (image_offset, image) in images.iter().enumerate() {
        parts.push(Part {
            id: format!("{id}.p{}", parts.len()),
            kind: PartKind::Text,
            content: format!("[Tool image: {}]", image.label),
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            prune_state: PruneState::Intact,
        });
        parts.push(Part {
            id: format!("{id}.p{}", parts.len()),
            kind: PartKind::Image,
            content: String::new(),
            attachment: Some(PartAttachment {
                mime: image.mime.clone(),
                url: data_url_for_bytes(&image.mime, &image.data),
                filename: Some(format!("tool-image-{image_offset}")),
            }),
            tool_call_id: None,
            tool_name: None,
            prune_state: PruneState::Intact,
        });
    }
    Message {
        id,
        role: MessageRole::User,
        parts,
        user_input: None,
        origin: None,
    }
}

fn extract_first_lashlang_fence(text: &str) -> Option<FenceExtraction> {
    fn find_fence(text: &str) -> Option<(usize, usize, usize)> {
        let mut search_from = 0usize;
        while let Some(rel) = text[search_from..].find("```") {
            let open = search_from + rel;
            let preceded_by_newline =
                open == 0 || text.as_bytes().get(open - 1).copied() == Some(b'\n');
            if !preceded_by_newline {
                search_from = open + 3;
                continue;
            }
            let after_open = open + 3;
            let rest = &text[after_open..];
            let lang_end = rest.find('\n').unwrap_or(rest.len());
            let lang = rest[..lang_end].trim();
            if !matches!(lang, "lashlang" | "rlm") {
                search_from = after_open;
                continue;
            }
            let body_start = after_open + lang_end + 1;
            if body_start > text.len() {
                return None;
            }

            let mut cursor = body_start;
            loop {
                let Some(rel) = text[cursor..].find("```") else {
                    return Some((open, body_start, text.len()));
                };
                let close = cursor + rel;
                let preceded_by_newline =
                    close == 0 || text.as_bytes().get(close - 1).copied() == Some(b'\n');
                if preceded_by_newline {
                    return Some((open, body_start, close));
                }
                cursor = close + 3;
            }
        }
        None
    }

    let (_open, body_start, body_end) = find_fence(text)?;
    let code = text[body_start..body_end]
        .trim_end_matches('\n')
        .to_string();
    let after_close = (body_end + 3).min(text.len());
    let had_extra_fences = find_fence(&text[after_close..]).is_some();

    Some(FenceExtraction {
        code,
        had_extra_fences,
    })
}

fn validate_finish_value(value: &Value, schema: &Value) -> Result<(), String> {
    fn matches_type(value: &Value, expected: &str) -> bool {
        match expected {
            "string" => value.is_string(),
            "number" => value.is_number(),
            "integer" => value
                .as_f64()
                .is_some_and(|n| n.is_finite() && n.fract() == 0.0),
            "boolean" => value.is_boolean(),
            "array" => value.is_array(),
            "object" => value.is_object(),
            "null" => value.is_null(),
            _ => true,
        }
    }

    fn type_name(value: &Value) -> &'static str {
        match value {
            Value::Null => "null",
            Value::Bool(_) => "boolean",
            Value::Number(_) => "number",
            Value::String(_) => "string",
            Value::Array(_) => "array",
            Value::Object(_) => "object",
        }
    }

    fn check(value: &Value, schema: &Value, path: &str) -> Result<(), String> {
        let Some(schema_obj) = schema.as_object() else {
            return Ok(());
        };

        if let Some(ty) = schema_obj.get("type").and_then(Value::as_str)
            && !matches_type(value, ty)
        {
            return Err(format!("{path}: expected {ty}, got {}", type_name(value)));
        }

        if let Some(properties) = schema_obj.get("properties").and_then(Value::as_object)
            && let Some(obj) = value.as_object()
        {
            if let Some(required) = schema_obj.get("required").and_then(Value::as_array) {
                for required_field in required {
                    if let Some(name) = required_field.as_str()
                        && !obj.contains_key(name)
                    {
                        return Err(format!("{path}: missing required field `{name}`"));
                    }
                }
            }
            for (name, sub_schema) in properties {
                if let Some(sub_value) = obj.get(name) {
                    let sub_path = if path.is_empty() {
                        name.clone()
                    } else {
                        format!("{path}.{name}")
                    };
                    check(sub_value, sub_schema, &sub_path)?;
                }
            }
        }

        if let Some(items_schema) = schema_obj.get("items")
            && let Some(arr) = value.as_array()
        {
            for (idx, item) in arr.iter().enumerate() {
                let sub_path = format!("{path}[{idx}]");
                check(item, items_schema, &sub_path)?;
            }
        }

        Ok(())
    }

    check(value, schema, "")
}

fn format_repl_result_text(success: bool, result_payload: &Value) -> String {
    let mut sections = vec!["[Lashlang execution result]".to_string()];
    if !success {
        sections.push("status: error".to_string());
    }
    if let Some(observations) = result_payload
        .get("observations")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
    {
        sections.push(format!("observations:\n{observations}"));
    }
    if let Some(arr) = result_payload
        .get("tool_calls")
        .and_then(Value::as_array)
        .filter(|arr| !arr.is_empty())
    {
        sections.push(format!(
            "tool_calls: {}",
            serde_json::to_string(arr).unwrap_or_else(|_| "[]".into())
        ));
    }
    if let Some(error) = result_payload
        .get("error")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
    {
        sections.push(format!("error:\n{error}"));
    }
    if let Some(images) = result_payload
        .get("images")
        .and_then(Value::as_array)
        .filter(|arr| !arr.is_empty())
    {
        sections.push(format!(
            "images: {}",
            serde_json::to_string(images).unwrap_or_else(|_| "[]".into())
        ));
    }
    sections.join("\n\n")
}

fn append_assistant_text_part(out: &mut String, next: &str) {
    if out.is_empty() {
        out.push_str(next);
        return;
    }

    let prev_trailing_newlines = out.chars().rev().take_while(|ch| *ch == '\n').count();
    let next_leading_newlines = next.chars().take_while(|ch| *ch == '\n').count();
    let total_boundary_newlines = prev_trailing_newlines + next_leading_newlines;
    if total_boundary_newlines < 2 {
        out.push_str(&"\n".repeat(2 - total_boundary_newlines));
    }

    out.push_str(next);
}
