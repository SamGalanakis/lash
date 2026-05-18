use std::sync::{Arc, RwLock};

use lash_core::{ToolCall, ToolContext, ToolContract, ToolManifest, ToolProvider, ToolResult};
use serde_json::{Value, json};

use crate::common::{LLM_CANDIDATE_LIMIT, args_with_limit, catalog_key, limit_from_args};
use crate::definitions::search_tools_definition;
use crate::ranking::ToolDiscoveryIndex;
use crate::rerank::{llm_rerank_request, merge_llm_selection, parse_llm_tool_names};

#[derive(Clone, Default)]
struct IndexCache {
    index: Option<Arc<ToolDiscoveryIndex>>,
}

#[derive(Clone)]
pub struct ToolDiscoveryToolsProvider {
    cache: Arc<RwLock<IndexCache>>,
}

impl Default for ToolDiscoveryToolsProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolDiscoveryToolsProvider {
    pub fn new() -> Self {
        Self {
            cache: Arc::default(),
        }
    }

    pub fn search_only() -> Self {
        Self::new()
    }

    fn index_for_catalog(&self, catalog: Vec<Value>) -> Arc<ToolDiscoveryIndex> {
        let key = catalog_key(&catalog);
        if let Some(index) = self
            .cache
            .read()
            .expect("tool discovery cache lock poisoned")
            .index
            .as_ref()
            .filter(|index| index.key == key)
            .cloned()
        {
            return index;
        }

        let index = Arc::new(ToolDiscoveryIndex::build(key, catalog));
        self.cache
            .write()
            .expect("tool discovery cache lock poisoned")
            .index = Some(Arc::clone(&index));
        index
    }

    async fn search_tools(
        &self,
        args: &Value,
        catalog: Vec<Value>,
        context: &ToolContext<'_>,
    ) -> ToolResult {
        let index = self.index_for_catalog(catalog);
        let limit = limit_from_args(args);
        let candidate_args = args_with_limit(args, LLM_CANDIDATE_LIMIT);
        let candidates = index.search(&candidate_args);
        if candidates.is_empty() {
            return ToolResult::ok(json!([]));
        }

        let request = llm_rerank_request(args, &candidates, limit);
        let completion = match context.direct_completion(request, "search_tools").await {
            Ok(completion) => completion,
            Err(err) => return ToolResult::err_fmt(format_args!("search_tools failed: {err}")),
        };

        let selected_names = match parse_llm_tool_names(&completion.text) {
            Ok(names) => names,
            Err(err) => {
                return ToolResult::err_fmt(format_args!(
                    "search_tools returned invalid JSON: {err}"
                ));
            }
        };

        ToolResult::ok(json!(merge_llm_selection(
            candidates,
            selected_names,
            limit
        )))
    }
}

#[async_trait::async_trait]
impl ToolProvider for ToolDiscoveryToolsProvider {
    fn tool_manifests(&self) -> Vec<ToolManifest> {
        vec![search_tools_definition().manifest()]
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<ToolContract>> {
        (name == "search_tools").then(|| Arc::new(search_tools_definition().contract()))
    }

    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        match call.name {
            "search_tools" => match call.context.tool_catalog().await {
                Ok(catalog) => self.search_tools(call.args, catalog, call.context).await,
                Err(err) => ToolResult::err_fmt(err.to_string()),
            },
            _ => ToolResult::err_fmt(format_args!("Unknown tool: {}", call.name)),
        }
    }
}
