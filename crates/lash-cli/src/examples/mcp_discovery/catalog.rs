use lash_core::CompactToolContract;
use serde_json::{Value, json};

use super::common::round_score;

#[derive(Clone, Debug)]
pub(crate) struct CatalogTool {
    pub(crate) id: String,
    pub(crate) name: String,
    #[cfg(feature = "lashlang")]
    pub(crate) module_path: Vec<String>,
    #[cfg(feature = "lashlang")]
    pub(crate) operation: String,
    #[cfg(feature = "lashlang")]
    pub(crate) call: String,
    #[cfg(feature = "lashlang")]
    pub(crate) aliases: Vec<String>,
    pub(crate) searchable: bool,
    pub(crate) contract: CompactToolContract,
    rendered_signature: String,
}

impl CatalogTool {
    pub(crate) fn from_value(raw: &Value) -> Option<Self> {
        let obj = raw.as_object()?;
        let id = obj.get("id")?.as_str()?.to_string();
        let name = obj.get("name")?.as_str()?.to_string();
        // Under the flat catalog, the projected record no longer carries
        // tiered catalog state or pre-resolved Lashlang call-paths. Derive the
        // call path from the tool's `lashlang.tool` binding, which is the only
        // discovery fact this example needs.
        #[cfg(feature = "lashlang")]
        let (module_path, operation, call, aliases) = {
            let binding: lash_lashlang_runtime::LashlangToolBinding = obj
                .get("bindings")
                .and_then(|bindings| bindings.get(lash_lashlang_runtime::LASHLANG_TOOL_BINDING_KEY))
                .cloned()
                .and_then(|value| serde_json::from_value(value).ok())?;
            let executable = binding.executable_for(&name).ok()?;
            (
                executable.module_path.clone(),
                executable.operation.clone(),
                executable.call_path(),
                executable.aliases.clone(),
            )
        };
        // Every catalog member is indexable under the flat model.
        let searchable = true;
        let contract: CompactToolContract = obj
            .get("contract")
            .cloned()
            .and_then(|value| serde_json::from_value(value).ok())?;
        let rendered_signature = contract.render_signature();
        Some(Self {
            id,
            name,
            #[cfg(feature = "lashlang")]
            module_path,
            #[cfg(feature = "lashlang")]
            operation,
            #[cfg(feature = "lashlang")]
            call,
            #[cfg(feature = "lashlang")]
            aliases,
            searchable,
            contract,
            rendered_signature,
        })
    }

    pub(crate) fn project(&self, score: f64, debug: bool) -> Value {
        let mut out = serde_json::Map::new();
        out.insert("id".to_string(), json!(self.id.clone()));
        out.insert("name".to_string(), json!(self.name.clone()));
        #[cfg(feature = "lashlang")]
        {
            out.insert("module_path".to_string(), json!(self.module_path.clone()));
            out.insert("operation".to_string(), json!(self.operation.clone()));
            out.insert("call".to_string(), json!(self.call.clone()));
        }
        out.insert(
            "signature".to_string(),
            json!(self.rendered_signature.clone()),
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
