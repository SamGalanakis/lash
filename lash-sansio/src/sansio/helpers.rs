use serde_json::Value;

use crate::llm::types::LlmUsage;
use crate::session_model::TokenUsage;

/// Outcome of pulling the first ` ```lashlang ` (or ` ```rlm `) fenced
/// block out of an assistant prose response.
pub(super) struct FenceExtraction {
    pub(super) code: String,
    /// True when the prose contained more than one matching fenced block.
    /// The dispatch loop only runs the first; this signals "tell the
    /// model to consolidate next time."
    pub(super) had_extra_fences: bool,
}

/// Find the first ` ```lashlang ` or ` ```rlm ` fenced block in `text`
/// and return its code body. Both language tags are accepted as aliases.
/// `None` ⇒ no recognized fence ⇒ caller treats the prose as a "finish
/// in prose" terminal response.
pub(super) fn extract_first_lashlang_fence(text: &str) -> Option<FenceExtraction> {
    fn find_fence(text: &str) -> Option<(usize, usize, usize)> {
        // Returns (open_byte, body_start, body_end_open_fence_byte). The
        // open fence is exactly three backticks followed by a recognized
        // language tag and a newline. The closing fence is three
        // backticks at the start of a line. Trailing characters on the
        // closing line are tolerated.
        let mut search_from = 0usize;
        while let Some(rel) = text[search_from..].find("```") {
            let open = search_from + rel;
            // Require either start-of-string or a newline immediately
            // before the opening fence so we don't match inline triple
            // backticks inside prose.
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
            // Find a closing fence at the start of a line.
            let mut cursor = body_start;
            loop {
                let Some(rel) = text[cursor..].find("```") else {
                    // No closing fence ⇒ treat the rest of the buffer as
                    // the code body. Forgiving for partial responses.
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

    // Look for any *additional* fenced block past the closing fence so
    // we can tell the model only the first ran.
    let after_close = (body_end + 3).min(text.len());
    let had_extra_fences = find_fence(&text[after_close..]).is_some();

    Some(FenceExtraction {
        code,
        had_extra_fences,
    })
}

/// Validate a `finish` value against the JSON Schema embedded in the
/// session's `RlmTermination::Finish { schema }`. Returns `Ok(())` on
/// success or a human-readable error string on mismatch. Supports the
/// subset of JSON Schema that `predict` generates: `type` (string,
/// number, integer, boolean, array, object, null), nested object
/// `properties` with `required`, and `items` for arrays. Unsupported
/// keywords are ignored (permissive).
pub(super) fn validate_finish_value(value: &Value, schema: &Value) -> Result<(), String> {
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
            _ => true, // unknown ⇒ permissive
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
            // Required fields
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

/// Render the lashlang exec result as a plain user-message text body.
/// Mirrors the JSON shape `format_tool_result_content` produced for the
/// legacy `execute_lashlang` tool result, but without the
/// tool-result-message envelope.
pub(super) fn format_repl_result_text(success: bool, result_payload: &Value) -> String {
    let mut sections: Vec<String> = Vec::new();
    sections.push("[Lashlang execution result]".to_string());
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

pub(super) fn token_usage_from_llm_usage(usage: &LlmUsage) -> TokenUsage {
    TokenUsage {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        cached_input_tokens: usage.cached_input_tokens,
        reasoning_tokens: usage.reasoning_tokens,
    }
}

pub(super) fn llm_usage_is_empty(usage: &LlmUsage) -> bool {
    usage.input_tokens == 0
        && usage.output_tokens == 0
        && usage.cached_input_tokens == 0
        && usage.reasoning_tokens == 0
}

pub(super) fn append_assistant_text_part(out: &mut String, next: &str) {
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
