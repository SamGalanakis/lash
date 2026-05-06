use lash_sansio::llm::types::{LlmProviderTraceEvent, LlmProviderTraceSender};
use serde_json::{Map, Value, json};

/// Strip a leading `vendor/` prefix from a model id (e.g. `openai/gpt-4o` →
/// `gpt-4o`). Used by the OpenAI-flavored providers when matching model
/// families.
pub fn model_id(model: &str) -> &str {
    model
        .rsplit_once('/')
        .map(|(_, tail)| tail)
        .unwrap_or(model)
}

/// Forward a raw provider event to the trace sink, deriving an event name from
/// the JSON `type` (or `event`) field when present.
pub fn emit_provider_trace(
    tx: Option<&LlmProviderTraceSender>,
    provider: &'static str,
    raw: &str,
) {
    let Some(tx) = tx else {
        return;
    };
    let event_name = serde_json::from_str::<Value>(raw)
        .ok()
        .and_then(|value| {
            value
                .get("type")
                .or_else(|| value.get("event"))
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or_else(|| "provider_event".to_string());
    tx.send(LlmProviderTraceEvent {
        provider,
        event_name,
        raw: raw.to_string(),
    });
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiSchemaProfile {
    ToolParameters,
    StrictToolParameters,
    StructuredOutput,
}

impl OpenAiSchemaProfile {
    pub const fn projection_id(self) -> &'static str {
        match self {
            Self::ToolParameters => "openai_tool_parameters",
            Self::StrictToolParameters => "openai_strict_tool_parameters",
            Self::StructuredOutput => "openai_structured_output",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SchemaProjection {
    pub schema: Value,
    pub diagnostics: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SchemaProjectionError {
    pub profile: OpenAiSchemaProfile,
    pub diagnostics: Vec<String>,
    first_diagnostic: String,
}

impl std::fmt::Display for SchemaProjectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "OpenAI schema projection for {:?} failed: {}",
            self.profile, self.first_diagnostic
        )
    }
}

impl std::error::Error for SchemaProjectionError {}

impl SchemaProjectionError {
    fn new(profile: OpenAiSchemaProfile, diagnostics: Vec<String>) -> Self {
        let first_diagnostic = diagnostics
            .first()
            .cloned()
            .unwrap_or_else(|| "schema projection failed".to_string());
        Self {
            profile,
            diagnostics,
            first_diagnostic,
        }
    }

    pub fn first_diagnostic(&self) -> &str {
        &self.first_diagnostic
    }
}

pub fn project_schema(
    schema: &Value,
    profile: OpenAiSchemaProfile,
) -> Result<SchemaProjection, SchemaProjectionError> {
    Projector::new(profile).project(schema)
}

pub fn project_tool_parameters(schema: &Value) -> Result<SchemaProjection, SchemaProjectionError> {
    project_schema(schema, OpenAiSchemaProfile::ToolParameters)
}

pub fn project_strict_tool_parameters(
    schema: &Value,
) -> Result<SchemaProjection, SchemaProjectionError> {
    project_schema(schema, OpenAiSchemaProfile::StrictToolParameters)
}

pub fn project_structured_output(
    schema: &Value,
) -> Result<SchemaProjection, SchemaProjectionError> {
    project_schema(schema, OpenAiSchemaProfile::StructuredOutput)
}

struct Projector {
    profile: OpenAiSchemaProfile,
    diagnostics: Vec<String>,
    errors: Vec<String>,
}

impl Projector {
    fn new(profile: OpenAiSchemaProfile) -> Self {
        Self {
            profile,
            diagnostics: Vec::new(),
            errors: Vec::new(),
        }
    }

    fn project(mut self, schema: &Value) -> Result<SchemaProjection, SchemaProjectionError> {
        let mut projected = schema.clone();
        self.project_value(&mut projected, Path::root(), true);
        self.ensure_object_root(&mut projected);

        if self.errors.is_empty() {
            Ok(SchemaProjection {
                schema: projected,
                diagnostics: self.diagnostics,
            })
        } else {
            Err(SchemaProjectionError::new(self.profile, self.errors))
        }
    }

    fn project_value(&mut self, value: &mut Value, path: Path, is_root: bool) {
        let Some(obj) = value.as_object_mut() else {
            self.errors
                .push(format!("{path}: schema must be a JSON object"));
            return;
        };

        self.convert_const(obj, &path);
        self.infer_type(obj, &path, is_root);

        if is_root
            && obj.contains_key("anyOf")
            && self.profile == OpenAiSchemaProfile::StructuredOutput
        {
            self.errors.push(format!(
                "{path}: OpenAI structured outputs do not allow root anyOf"
            ));
        }

        self.reject_unsupported_keywords(obj, &path);

        if schema_type_contains(obj, "object") {
            self.project_object(obj, &path);
        }

        if let Some(items) = obj.get_mut("items") {
            self.project_value(items, path.child("items"), false);
        }

        for key in ["$defs", "definitions"] {
            if let Some(defs) = obj.get_mut(key).and_then(Value::as_object_mut) {
                for (name, schema) in defs {
                    self.project_value(schema, path.child(key).child(name), false);
                }
            }
        }

        if let Some(any_of) = obj.get_mut("anyOf").and_then(Value::as_array_mut) {
            for (idx, schema) in any_of.iter_mut().enumerate() {
                self.project_value(schema, path.child("anyOf").index(idx), false);
            }
        }
    }

    fn ensure_object_root(&mut self, value: &mut Value) {
        let Some(obj) = value.as_object_mut() else {
            return;
        };
        if !schema_type_contains(obj, "object") {
            self.errors
                .push("$: root schema must be an object schema".to_string());
            return;
        }
        obj.entry("properties")
            .or_insert_with(|| Value::Object(Map::new()));
        if !obj.get("properties").is_some_and(Value::is_object) {
            self.errors
                .push("$: object schema properties must be an object".to_string());
        }
    }

    fn project_object(&mut self, obj: &mut Map<String, Value>, path: &Path) {
        let originally_required = required_set(obj);
        match obj.get_mut("properties") {
            Some(Value::Object(properties)) => {
                let property_names = properties.keys().cloned().collect::<Vec<_>>();
                for (name, schema) in properties.iter_mut() {
                    let optional = !originally_required.iter().any(|required| required == name);
                    self.project_value(schema, path.child("properties").child(name), false);
                    if optional && self.requires_strict_objects() {
                        make_nullable(schema);
                    }
                }
                if self.requires_strict_objects() {
                    obj.insert(
                        "required".to_string(),
                        Value::Array(property_names.into_iter().map(Value::String).collect()),
                    );
                }
            }
            Some(_) => self.errors.push(format!(
                "{path}: object schema properties must be an object"
            )),
            None => {
                obj.insert("properties".to_string(), Value::Object(Map::new()));
                self.diagnostics
                    .push(format!("{path}: inserted missing object properties"));
                if self.requires_strict_objects() {
                    obj.insert("required".to_string(), Value::Array(Vec::new()));
                }
            }
        }

        if self.requires_strict_objects() {
            obj.insert("additionalProperties".to_string(), Value::Bool(false));
        }
    }

    fn requires_strict_objects(&self) -> bool {
        matches!(
            self.profile,
            OpenAiSchemaProfile::StrictToolParameters | OpenAiSchemaProfile::StructuredOutput
        )
    }

    fn convert_const(&mut self, obj: &mut Map<String, Value>, path: &Path) {
        if let Some(value) = obj.remove("const") {
            obj.entry("enum".to_string())
                .or_insert_with(|| Value::Array(vec![value]));
            self.diagnostics
                .push(format!("{path}: converted const to single-value enum"));
        }
    }

    fn infer_type(&mut self, obj: &mut Map<String, Value>, path: &Path, is_root: bool) {
        if obj.contains_key("type") {
            return;
        }
        let inferred = if is_root
            || obj.contains_key("properties")
            || obj.contains_key("required")
            || obj.contains_key("additionalProperties")
        {
            Some("object")
        } else if obj.contains_key("items") {
            Some("array")
        } else if let Some(enum_values) = obj.get("enum").and_then(Value::as_array) {
            infer_enum_type(enum_values)
        } else {
            None
        };
        if let Some(inferred) = inferred {
            obj.insert("type".to_string(), Value::String(inferred.to_string()));
            self.diagnostics
                .push(format!("{path}: inferred missing type as {inferred}"));
        }
    }

    fn reject_unsupported_keywords(&mut self, obj: &Map<String, Value>, path: &Path) {
        let unsupported = [
            "allOf",
            "oneOf",
            "not",
            "dependentRequired",
            "dependentSchemas",
            "if",
            "then",
            "else",
            "patternProperties",
        ];
        for key in unsupported {
            if obj.contains_key(key) {
                self.errors
                    .push(format!("{path}: unsupported JSON Schema keyword `{key}`"));
            }
        }
    }
}

#[derive(Clone, Debug)]
struct Path(String);

impl Path {
    fn root() -> Self {
        Self("$".to_string())
    }

    fn child(&self, segment: &str) -> Self {
        Self(format!("{}.{}", self.0, segment))
    }

    fn index(&self, index: usize) -> Self {
        Self(format!("{}[{index}]", self.0))
    }
}

impl std::fmt::Display for Path {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

fn schema_type_contains(obj: &Map<String, Value>, expected: &str) -> bool {
    match obj.get("type") {
        Some(Value::String(value)) => value == expected,
        Some(Value::Array(values)) => values.iter().any(|value| value.as_str() == Some(expected)),
        _ => false,
    }
}

fn required_set(obj: &Map<String, Value>) -> Vec<String> {
    obj.get("required")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn infer_enum_type(values: &[Value]) -> Option<&'static str> {
    let mut inferred = None;
    for value in values {
        let value_type = match value {
            Value::String(_) => "string",
            Value::Bool(_) => "boolean",
            Value::Number(number) if number.is_i64() || number.is_u64() => "integer",
            Value::Number(_) => "number",
            Value::Null => continue,
            _ => return None,
        };
        match inferred {
            Some(existing) if existing != value_type => return None,
            Some(_) => {}
            None => inferred = Some(value_type),
        }
    }
    inferred
}

fn make_nullable(schema: &mut Value) {
    let Some(obj) = schema.as_object_mut() else {
        return;
    };
    if let Some(any_of) = obj.get_mut("anyOf").and_then(Value::as_array_mut) {
        if !any_of.iter().any(is_null_schema) {
            any_of.push(json!({ "type": "null" }));
        }
        return;
    }
    match obj.get_mut("type") {
        Some(Value::String(value)) if value != "null" => {
            let original = value.clone();
            obj.insert(
                "type".to_string(),
                Value::Array(vec![
                    Value::String(original),
                    Value::String("null".to_string()),
                ]),
            );
        }
        Some(Value::Array(values))
            if !values.iter().any(|value| value.as_str() == Some("null")) =>
        {
            values.push(Value::String("null".to_string()));
        }
        None => {
            let original = Value::Object(obj.clone());
            obj.clear();
            obj.insert(
                "anyOf".to_string(),
                Value::Array(vec![original, json!({ "type": "null" })]),
            );
        }
        _ => {}
    }
}

fn is_null_schema(value: &Value) -> bool {
    value
        .as_object()
        .is_some_and(|obj| schema_type_contains(obj, "null"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn required_names(schema: &Value) -> Vec<String> {
        let mut names = schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap().to_string())
            .collect::<Vec<_>>();
        names.sort();
        names
    }

    #[test]
    fn projection_does_not_mutate_canonical_schema() {
        let schema = json!({"type": "object"});
        let projected = project_tool_parameters(&schema).unwrap();
        assert_eq!(schema, json!({"type": "object"}));
        assert_eq!(projected.schema["properties"], json!({}));
    }

    #[test]
    fn tool_parameters_repairs_empty_root_object() {
        let projected = project_tool_parameters(&json!({})).unwrap();
        assert_eq!(projected.schema["type"], "object");
        assert_eq!(projected.schema["properties"], json!({}));
        assert!(projected.diagnostics.iter().any(|d| d.contains("inferred")));
    }

    #[test]
    fn tool_parameters_repairs_missing_properties_missing_type_and_const() {
        let schema = json!({
            "properties": {
                "mode": { "const": "fast" }
            }
        });
        let projected = project_tool_parameters(&schema).unwrap();
        assert_eq!(projected.schema["type"], "object");
        assert_eq!(
            projected.schema["properties"]["mode"]["enum"],
            json!(["fast"])
        );
        assert!(
            projected.schema["properties"]["mode"]
                .get("const")
                .is_none()
        );
    }

    #[test]
    fn tool_parameters_infers_array_and_enum_types() {
        let schema = json!({
            "type": "object",
            "properties": {
                "tags": { "items": { "type": "string" } },
                "level": { "enum": [1, 2, 3] }
            }
        });
        let projected = project_tool_parameters(&schema).unwrap();
        assert_eq!(projected.schema["properties"]["tags"]["type"], "array");
        assert_eq!(projected.schema["properties"]["level"]["type"], "integer");
    }

    #[test]
    fn strict_projection_requires_optional_nullable_properties() {
        let schema = json!({
            "type": "object",
            "properties": {
                "required_name": { "type": "string" },
                "optional_count": { "type": "integer" }
            },
            "required": ["required_name"]
        });
        let projected = project_strict_tool_parameters(&schema).unwrap();
        assert_eq!(
            required_names(&projected.schema),
            vec!["optional_count", "required_name"]
        );
        assert_eq!(
            projected.schema["properties"]["optional_count"]["type"],
            json!(["integer", "null"])
        );
        assert_eq!(projected.schema["additionalProperties"], false);
    }

    #[test]
    fn strict_projection_preserves_required_nonnullable_fields() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "age": { "type": "integer" }
            },
            "required": ["name", "age"]
        });
        let projected = project_strict_tool_parameters(&schema).unwrap();
        assert_eq!(projected.schema["properties"]["name"]["type"], "string");
        assert_eq!(projected.schema["properties"]["age"]["type"], "integer");
    }

    #[test]
    fn strict_projection_does_not_duplicate_existing_nullable_type() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": { "type": ["string", "null"] }
            }
        });
        let projected = project_strict_tool_parameters(&schema).unwrap();
        assert_eq!(
            projected.schema["properties"]["name"]["type"],
            json!(["string", "null"])
        );
    }

    #[test]
    fn strict_projection_adds_null_branch_to_optional_any_of() {
        let schema = json!({
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        { "type": "string" },
                        { "type": "integer" }
                    ]
                }
            }
        });
        let projected = project_strict_tool_parameters(&schema).unwrap();
        assert_eq!(
            projected.schema["properties"]["value"]["anyOf"][2],
            json!({ "type": "null" })
        );
    }

    #[test]
    fn structured_output_enforces_strict_objects() {
        let schema = json!({
            "type": "object",
            "properties": {
                "value": { "type": "string" }
            }
        });
        let projected = project_structured_output(&schema).unwrap();
        assert_eq!(projected.schema["required"], json!(["value"]));
        assert_eq!(projected.schema["additionalProperties"], false);
    }

    #[test]
    fn structured_output_recurses_into_nested_objects_and_arrays() {
        let schema = json!({
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": { "type": "string" },
                            "score": { "type": "number" }
                        },
                        "required": ["title"]
                    }
                }
            }
        });
        let projected = project_structured_output(&schema).unwrap();
        let nested = &projected.schema["properties"]["items"]["items"];
        assert_eq!(nested["additionalProperties"], false);
        assert_eq!(required_names(nested), vec!["score", "title"]);
        assert_eq!(
            nested["properties"]["score"]["type"],
            json!(["number", "null"])
        );
    }

    #[test]
    fn structured_output_recurses_into_defs_and_definitions() {
        let schema = json!({
            "type": "object",
            "properties": {
                "item": { "$ref": "#/$defs/Item" },
                "legacy": { "$ref": "#/definitions/Legacy" }
            },
            "$defs": {
                "Item": {
                    "type": "object",
                    "properties": { "id": { "type": "string" } }
                }
            },
            "definitions": {
                "Legacy": {
                    "type": "object",
                    "properties": { "flag": { "type": "boolean" } }
                }
            }
        });
        let projected = project_structured_output(&schema).unwrap();
        assert_eq!(
            projected.schema["$defs"]["Item"]["additionalProperties"],
            false
        );
        assert_eq!(projected.schema["$defs"]["Item"]["required"], json!(["id"]));
        assert_eq!(
            projected.schema["definitions"]["Legacy"]["additionalProperties"],
            false
        );
        assert_eq!(
            projected.schema["definitions"]["Legacy"]["required"],
            json!(["flag"])
        );
    }

    #[test]
    fn structured_output_allows_nested_any_of_and_projects_branches() {
        let schema = json!({
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        { "type": "string" },
                        {
                            "type": "object",
                            "properties": { "count": { "type": "integer" } }
                        }
                    ]
                }
            }
        });
        let projected = project_structured_output(&schema).unwrap();
        let object_branch = &projected.schema["properties"]["value"]["anyOf"][1];
        assert_eq!(object_branch["required"], json!(["count"]));
        assert_eq!(object_branch["additionalProperties"], false);
        assert_eq!(
            projected.schema["properties"]["value"]["anyOf"][2],
            json!({ "type": "null" })
        );
    }

    #[test]
    fn structured_output_rejects_root_scalar() {
        let err = project_structured_output(&json!({ "type": "string" })).unwrap_err();
        assert!(
            err.diagnostics
                .iter()
                .any(|diagnostic| diagnostic.contains("root schema must be an object"))
        );
    }

    #[test]
    fn structured_output_rejects_root_any_of() {
        let err = project_structured_output(&json!({
            "anyOf": [
                { "type": "object", "properties": {} },
                { "type": "object", "properties": {} }
            ]
        }))
        .unwrap_err();
        assert!(
            err.diagnostics
                .iter()
                .any(|diagnostic| diagnostic.contains("root anyOf"))
        );
    }

    #[test]
    fn projection_rejects_unsupported_lossy_keywords() {
        let err = project_structured_output(&json!({
            "type": "object",
            "properties": {},
            "allOf": [],
            "patternProperties": {}
        }))
        .unwrap_err();
        assert!(
            err.diagnostics
                .iter()
                .any(|diagnostic| diagnostic.contains("allOf"))
        );
        assert!(
            err.diagnostics
                .iter()
                .any(|diagnostic| diagnostic.contains("patternProperties"))
        );
    }

    #[test]
    fn projection_rejects_non_object_properties() {
        let err = project_tool_parameters(&json!({
            "type": "object",
            "properties": []
        }))
        .unwrap_err();
        assert!(
            err.diagnostics
                .iter()
                .any(|diagnostic| diagnostic.contains("properties must be an object"))
        );
    }

    #[test]
    fn profile_projection_ids_are_stable() {
        assert_eq!(
            OpenAiSchemaProfile::ToolParameters.projection_id(),
            "openai_tool_parameters"
        );
        assert_eq!(
            OpenAiSchemaProfile::StrictToolParameters.projection_id(),
            "openai_strict_tool_parameters"
        );
        assert_eq!(
            OpenAiSchemaProfile::StructuredOutput.projection_id(),
            "openai_structured_output"
        );
    }
}
