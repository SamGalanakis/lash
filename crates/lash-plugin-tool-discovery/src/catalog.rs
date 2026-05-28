use lash_core::CompactToolContract;
use serde_json::{Value, json};

use crate::common::{round_score, string_vec};

#[derive(Clone, Debug)]
pub(crate) struct CatalogTool {
    pub(crate) name: String,
    pub(crate) module_path: Vec<String>,
    pub(crate) operation: String,
    pub(crate) call: String,
    pub(crate) aliases: Vec<String>,
    pub(crate) searchable: bool,
    pub(crate) contract: CompactToolContract,
}

impl CatalogTool {
    pub(crate) fn from_value(raw: Value) -> Option<Self> {
        let obj = raw.as_object()?;
        let _id = obj.get("id")?.as_str()?;
        let name = obj.get("name")?.as_str()?.to_string();
        let module_path = string_vec(obj.get("module_path"));
        let operation = obj
            .get("operation")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| name.clone());
        let call = obj
            .get("call")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| {
                if module_path.is_empty() {
                    operation.clone()
                } else {
                    format!("{}.{}", module_path.join("."), operation)
                }
            });
        let aliases = string_vec(obj.get("aliases"));
        let searchable = obj
            .get("searchable")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        let contract = obj
            .get("contract")
            .cloned()
            .and_then(|value| serde_json::from_value(value).ok())?;
        Some(Self {
            name,
            module_path,
            operation,
            call,
            aliases,
            searchable,
            contract,
        })
    }

    pub(crate) fn project(&self, score: f64, debug: bool) -> Value {
        let mut out = serde_json::Map::new();
        out.insert("name".to_string(), json!(self.call.clone()));
        out.insert("module_path".to_string(), json!(self.module_path.clone()));
        out.insert("operation".to_string(), json!(self.operation.clone()));
        out.insert("call".to_string(), json!(self.call.clone()));
        out.insert(
            "signature".to_string(),
            json!(self.contract.render_signature()),
        );
        if !self.contract.description.is_empty() {
            out.insert(
                "description".to_string(),
                json!(self.contract.description.clone()),
            );
        }
        if !self.contract.examples.is_empty() {
            out.insert(
                "examples".to_string(),
                json!(self.contract.examples.clone()),
            );
        }
        if debug {
            out.insert("score".to_string(), json!(round_score(score)));
        }
        Value::Object(out)
    }
}
