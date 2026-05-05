use lash::ToolDefinition;
use serde_json::{Value, json};

use crate::common::{output_contract_field, round_score, string_field, string_vec};

#[derive(Clone, Debug)]
pub(crate) struct CatalogTool {
    pub(crate) raw: Value,
    pub(crate) name: String,
    pub(crate) namespace: Option<String>,
    pub(crate) aliases: Vec<String>,
    pub(crate) callable: bool,
    pub(crate) documented: bool,
    pub(crate) discoverable: bool,
    pub(crate) loadable: bool,
}

impl CatalogTool {
    pub(crate) fn from_value(raw: Value) -> Option<Self> {
        let obj = raw.as_object()?;
        let name = obj.get("name")?.as_str()?.to_string();
        let namespace = obj
            .get("namespace")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string);
        let aliases = string_vec(obj.get("aliases"));
        let callable = obj
            .get("callable")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let documented = obj
            .get("documented")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let discoverable = obj
            .get("discoverable")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        let loadable = obj
            .get("loadable")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        Some(Self {
            raw,
            name,
            namespace,
            aliases,
            callable,
            documented,
            discoverable,
            loadable,
        })
    }

    pub(crate) fn project(&self, score: f64, debug: bool) -> Value {
        let definition = self.compact_definition();
        let contract = definition.compact_contract();
        let mut out = serde_json::Map::new();
        out.insert("name".to_string(), json!(contract.name));
        out.insert("signature".to_string(), json!(contract.render_signature()));
        if !contract.description.is_empty() {
            out.insert("description".to_string(), json!(contract.description));
        }
        if !contract.examples.is_empty() {
            out.insert("examples".to_string(), json!(contract.examples));
        }
        if debug {
            out.insert("score".to_string(), json!(round_score(score)));
        }
        Value::Object(out)
    }

    pub(crate) fn compact_definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            self.name.clone(),
            string_field(&self.raw, "description"),
            self.raw
                .get("input_schema")
                .cloned()
                .unwrap_or_else(ToolDefinition::default_input_schema),
            self.raw
                .get("output_schema")
                .cloned()
                .unwrap_or_else(|| json!({})),
        )
        .with_output_contract(output_contract_field(self.raw.get("output_contract")))
        .with_examples(string_vec(self.raw.get("examples")))
    }
}
