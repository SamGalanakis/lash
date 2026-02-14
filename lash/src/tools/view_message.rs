use std::sync::Arc;

use crate::store::Store;
use crate::{ToolDefinition, ToolParam, ToolProvider, ToolResult};

/// Tool that retrieves original content of pruned/summarized message parts by archive hash.
pub struct ViewMessage {
    store: Arc<Store>,
}

impl ViewMessage {
    pub fn new(store: Arc<Store>) -> Self {
        Self { store }
    }
}

#[async_trait::async_trait]
impl ToolProvider for ViewMessage {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "view_message".into(),
            description: "Retrieve the original content of a pruned or summarized message part by its archive hash.".into(),
            params: vec![ToolParam::typed("hash", "str")],
            returns: "str".into(),
            hidden: false,
        }]
    }

    async fn execute(&self, _name: &str, args: &serde_json::Value) -> ToolResult {
        let hash = args["hash"].as_str().unwrap_or("");
        if hash.is_empty() {
            return ToolResult::err(serde_json::json!("Missing 'hash' argument"));
        }
        match self.store.get_archive(hash) {
            Some(content) => ToolResult::ok(serde_json::json!(content)),
            None => ToolResult::err(serde_json::json!("No archived content for hash")),
        }
    }
}
