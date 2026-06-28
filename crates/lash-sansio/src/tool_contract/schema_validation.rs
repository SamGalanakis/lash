use serde_json::Value;

use crate::tool_contract::ToolContract;

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LashSchema {
    pub schema: Value,
}

impl LashSchema {
    pub fn new(schema: Value) -> Self {
        Self { schema }
    }

    pub fn any() -> Self {
        Self::new(serde_json::json!({}))
    }

    pub fn object(properties: serde_json::Map<String, Value>, required: Vec<String>) -> Self {
        let mut schema = serde_json::Map::new();
        schema.insert("type".to_string(), Value::String("object".to_string()));
        schema.insert("properties".to_string(), Value::Object(properties));
        if !required.is_empty() {
            schema.insert(
                "required".to_string(),
                Value::Array(required.into_iter().map(Value::String).collect()),
            );
        }
        schema.insert("additionalProperties".to_string(), Value::Bool(true));
        Self::new(Value::Object(schema))
    }

    pub fn validate(&self, value: &Value) -> Result<(), String> {
        validate_schema("", &self.schema, value)
    }
}

pub fn validate_tool_input(contract: &ToolContract, args: &Value) -> Result<(), String> {
    LashSchema::new(contract.input_schema.canonical().clone()).validate(args)
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
            match schema.get("additionalProperties") {
                Some(Value::Bool(true)) => {}
                Some(Value::Object(additional_schema)) => {
                    let additional_schema = Value::Object(additional_schema.clone());
                    for (name, property_value) in object {
                        if is_internal_argument(name) || properties.contains_key(name) {
                            continue;
                        }
                        validate_schema(
                            &join_path(path, name),
                            &additional_schema,
                            property_value,
                        )?;
                    }
                }
                _ => {
                    for name in object.keys() {
                        if is_internal_argument(name) {
                            continue;
                        }
                        if !properties.contains_key(name) {
                            return Err(format!("{}: unexpected property", join_path(path, name)));
                        }
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

fn is_internal_argument(name: &str) -> bool {
    name == "__session_id__"
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
    use crate::ToolDefinition;

    #[test]
    fn validation_reports_missing_required_property_by_path() {
        let tool = ToolDefinition::raw(
            "tool:spotify",
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

        let error = validate_tool_input(&tool.contract(), &serde_json::json!({})).unwrap_err();
        assert_eq!(error, "access_token: required property missing");
    }

    #[test]
    fn validation_reports_numeric_limits_by_path() {
        let tool = ToolDefinition::raw(
            "tool:spotify",
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
            validate_tool_input(&tool.contract(), &serde_json::json!({ "page_limit": 100 }))
                .unwrap_err();
        assert_eq!(error, "page_limit: expected integer <= 20, got 100");
    }

    #[test]
    fn validation_rejects_unknown_property_when_additional_properties_is_omitted() {
        let tool = ToolDefinition::raw(
            "tool:mcp__appworld__venmo_show_transactions",
            "mcp__appworld__venmo_show_transactions",
            "",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "min_created_at": { "type": "string" },
                    "max_created_at": { "type": "string" },
                    "limit": { "type": "integer", "maximum": 100 }
                },
                "required": ["limit"]
            }),
            serde_json::json!({}),
        );

        let error = validate_tool_input(
            &tool.contract(),
            &serde_json::json!({
                "min_datetime": "2024-01-01T00:00:00Z",
                "limit": 20
            }),
        )
        .unwrap_err();
        assert_eq!(error, "min_datetime: unexpected property");
    }

    #[test]
    fn validation_allows_unknown_property_when_additional_properties_is_true() {
        let tool = ToolDefinition::raw(
            "tool:open",
            "open",
            "",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" }
                },
                "additionalProperties": true
            }),
            serde_json::json!({}),
        );

        validate_tool_input(
            &tool.contract(),
            &serde_json::json!({
                "path": "README.md",
                "unknown": "preserved"
            }),
        )
        .unwrap();
    }

    #[test]
    fn validation_preserves_internal_session_id_argument() {
        let tool = ToolDefinition::raw(
            "tool:session_scoped_tool",
            "session_scoped_tool",
            "",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" }
                }
            }),
            serde_json::json!({}),
        );

        validate_tool_input(
            &tool.contract(),
            &serde_json::json!({
                "query": "hello",
                "__session_id__": "session"
            }),
        )
        .unwrap();
    }
}
