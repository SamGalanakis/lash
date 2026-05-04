use serde_json::Value;

#[cfg(feature = "semantic-tool-search")]
use crate::catalog::CatalogTool;

pub(crate) fn schema_index_text(schema: Option<&Value>) -> String {
    let mut parts = Vec::new();
    if let Some(schema) = schema {
        collect_schema_index_text("", schema, &mut parts);
    }
    parts.join("\n")
}

#[cfg(feature = "semantic-tool-search")]
pub(crate) fn semantic_index_text(tool: &CatalogTool) -> String {
    let definition = tool.compact_definition();
    let contract = definition.compact_contract();
    let mut parts = vec![
        contract.name.clone(),
        contract.render_signature(),
        contract.render_returns(),
        contract.description.clone(),
    ];
    parts.extend(contract.examples.clone());
    parts
        .into_iter()
        .map(|part| part.trim().to_string())
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn collect_schema_index_text(path: &str, schema: &Value, parts: &mut Vec<String>) {
    if let Some(any_of) = schema
        .get("anyOf")
        .or_else(|| schema.get("oneOf"))
        .or_else(|| schema.get("allOf"))
        .and_then(Value::as_array)
    {
        for subschema in any_of {
            collect_schema_index_text(path, subschema, parts);
        }
    }

    if !path.is_empty() {
        parts.push(path.to_string());
    }
    if let Some(description) = schema
        .get("description")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|description| !description.is_empty())
    {
        parts.push(description.to_string());
    }
    if let Some(values) = schema.get("enum").and_then(Value::as_array) {
        parts.extend(values.iter().filter_map(enum_index_value));
    }

    if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
        for (name, property_schema) in properties {
            let field_path = join_schema_path(path, name);
            collect_schema_index_text(&field_path, property_schema, parts);
        }
    }

    if let Some(items) = schema.get("items") {
        let item_path = if path.is_empty() {
            "[]".to_string()
        } else {
            format!("{path}[]")
        };
        collect_schema_index_text(&item_path, items, parts);
    }
}

fn join_schema_path(parent: &str, child: &str) -> String {
    if parent.is_empty() {
        child.to_string()
    } else {
        format!("{parent}.{child}")
    }
}

fn enum_index_value(value: &Value) -> Option<String> {
    match value {
        Value::String(value) if !value.trim().is_empty() => Some(value.trim().to_string()),
        Value::Number(value) => Some(value.to_string()),
        Value::Bool(value) => Some(value.to_string()),
        _ => None,
    }
}
