use serde_json::Value;

use crate::ToolDefinition;

pub(crate) fn validate_tool_input(tool: &ToolDefinition, args: &Value) -> Result<(), String> {
    let _ = jsonschema::JSONSchema::compile(&tool.input_schema);
    validate_schema("", &tool.input_schema, args)
}

fn validate_schema(path: &str, schema: &Value, value: &Value) -> Result<(), String> {
    if let Some(any_of) = schema
        .get("anyOf")
        .or_else(|| schema.get("oneOf"))
        .and_then(Value::as_array)
    {
        if any_of
            .iter()
            .any(|subschema| validate_schema(path, subschema, value).is_ok())
        {
            return Ok(());
        }
        return Err(format!(
            "{}: expected {}, got {}",
            display_path(path),
            expected_description(schema),
            display_value(value)
        ));
    }

    if let Some(enum_values) = schema.get("enum").and_then(Value::as_array)
        && !enum_values.iter().any(|candidate| candidate == value)
    {
        return Err(format!(
            "{}: expected one of {}, got {}",
            display_path(path),
            enum_values
                .iter()
                .map(display_value)
                .collect::<Vec<_>>()
                .join(", "),
            display_value(value)
        ));
    }

    if let Some(type_value) = schema.get("type")
        && !matches_type(type_value, value)
    {
        return Err(format!(
            "{}: expected {}, got {}",
            display_path(path),
            expected_description(schema),
            display_value(value)
        ));
    }

    if let Some(maximum) = schema.get("maximum").and_then(Value::as_f64)
        && value.as_f64().is_some_and(|actual| actual > maximum)
    {
        return Err(format!(
            "{}: expected {}, got {}",
            display_path(path),
            expected_description(schema),
            display_value(value)
        ));
    }
    if let Some(minimum) = schema.get("minimum").and_then(Value::as_f64)
        && value.as_f64().is_some_and(|actual| actual < minimum)
    {
        return Err(format!(
            "{}: expected {}, got {}",
            display_path(path),
            expected_description(schema),
            display_value(value)
        ));
    }

    if let Some(max_length) = schema.get("maxLength").and_then(Value::as_u64)
        && value
            .as_str()
            .is_some_and(|actual| actual.chars().count() as u64 > max_length)
    {
        return Err(format!(
            "{}: expected {}, got {}",
            display_path(path),
            expected_description(schema),
            display_value(value)
        ));
    }
    if let Some(min_length) = schema.get("minLength").and_then(Value::as_u64)
        && value
            .as_str()
            .is_some_and(|actual| (actual.chars().count() as u64) < min_length)
    {
        return Err(format!(
            "{}: expected {}, got {}",
            display_path(path),
            expected_description(schema),
            display_value(value)
        ));
    }

    if let Some(items) = value.as_array() {
        if let Some(max_items) = schema.get("maxItems").and_then(Value::as_u64)
            && items.len() as u64 > max_items
        {
            return Err(format!(
                "{}: expected {}, got {} items",
                display_path(path),
                expected_description(schema),
                items.len()
            ));
        }
        if let Some(min_items) = schema.get("minItems").and_then(Value::as_u64)
            && (items.len() as u64) < min_items
        {
            return Err(format!(
                "{}: expected {}, got {} items",
                display_path(path),
                expected_description(schema),
                items.len()
            ));
        }
        if let Some(item_schema) = schema.get("items") {
            for (idx, item) in items.iter().enumerate() {
                validate_schema(&format!("{path}[{idx}]"), item_schema, item)?;
            }
        }
    }

    if let Some(object) = value.as_object() {
        let required = schema
            .get("required")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str);
        for property in required {
            if !object.contains_key(property) {
                return Err(format!(
                    "{}: required property missing",
                    join_path(path, property)
                ));
            }
        }

        if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
            for (name, property_schema) in properties {
                if let Some(property_value) = object.get(name) {
                    validate_schema(&join_path(path, name), property_schema, property_value)?;
                }
            }
            if schema.get("additionalProperties") == Some(&Value::Bool(false)) {
                for name in object.keys() {
                    if name.starts_with("__") {
                        continue;
                    }
                    if !properties.contains_key(name) {
                        return Err(format!("{}: unexpected property", join_path(path, name)));
                    }
                }
            }
        }
    }

    Ok(())
}

fn matches_type(type_value: &Value, value: &Value) -> bool {
    match type_value {
        Value::String(ty) => matches_single_type(ty, value),
        Value::Array(types) => types
            .iter()
            .filter_map(Value::as_str)
            .any(|ty| matches_single_type(ty, value)),
        _ => true,
    }
}

fn matches_single_type(ty: &str, value: &Value) -> bool {
    match ty {
        "null" => value.is_null(),
        "boolean" => value.is_boolean(),
        "string" => value.is_string(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "number" => value.is_number(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        _ => true,
    }
}

fn expected_description(schema: &Value) -> String {
    let mut parts = Vec::new();
    if let Some(type_value) = schema.get("type") {
        parts.push(type_description(type_value, schema));
    } else if schema.get("anyOf").is_some() || schema.get("oneOf").is_some() {
        parts.push("matching schema".to_string());
    }
    if let Some(minimum) = schema.get("minimum") {
        parts.push(format!(">= {}", display_value(minimum)));
    }
    if let Some(maximum) = schema.get("maximum") {
        parts.push(format!("<= {}", display_value(maximum)));
    }
    if let Some(min_length) = schema.get("minLength") {
        parts.push(format!("length >= {}", display_value(min_length)));
    }
    if let Some(max_length) = schema.get("maxLength") {
        parts.push(format!("length <= {}", display_value(max_length)));
    }
    if let Some(min_items) = schema.get("minItems") {
        parts.push(format!("items >= {}", display_value(min_items)));
    }
    if let Some(max_items) = schema.get("maxItems") {
        parts.push(format!("items <= {}", display_value(max_items)));
    }
    if parts.is_empty() {
        "valid value".to_string()
    } else {
        parts.join(" ")
    }
}

fn type_description(type_value: &Value, schema: &Value) -> String {
    match type_value {
        Value::String(ty) => type_name(ty, schema).to_string(),
        Value::Array(types) => types
            .iter()
            .filter_map(Value::as_str)
            .map(|ty| type_name(ty, schema).to_string())
            .collect::<Vec<_>>()
            .join(" or "),
        _ => "valid value".to_string(),
    }
}

fn type_name<'a>(ty: &'a str, schema: &'a Value) -> &'a str {
    if ty == "array" && schema.get("items").is_some() {
        "array"
    } else {
        ty
    }
}

fn display_path(path: &str) -> String {
    if path.is_empty() {
        "arguments".to_string()
    } else {
        path.to_string()
    }
}

fn join_path(base: &str, key: &str) -> String {
    if base.is_empty() {
        key.to_string()
    } else {
        format!("{base}.{key}")
    }
}

fn display_value(value: &Value) -> String {
    match value {
        Value::String(text) => format!("{text:?}"),
        _ => value.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_reports_missing_required_property_by_path() {
        let tool = ToolDefinition::new(
            "spotify",
            "",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "access_token": { "type": "string" }
                },
                "required": ["access_token"],
                "additionalProperties": false
            }),
            serde_json::json!({}),
        );

        let error = validate_tool_input(&tool, &serde_json::json!({})).unwrap_err();
        assert_eq!(error, "access_token: required property missing");
    }

    #[test]
    fn validation_reports_numeric_limits_by_path() {
        let tool = ToolDefinition::new(
            "spotify",
            "",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "page_limit": { "type": "integer", "maximum": 20 }
                },
                "required": ["page_limit"],
                "additionalProperties": false
            }),
            serde_json::json!({}),
        );

        let error =
            validate_tool_input(&tool, &serde_json::json!({ "page_limit": 100 })).unwrap_err();
        assert_eq!(error, "page_limit: expected integer <= 20, got 100");
    }
}
