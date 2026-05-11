use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

use lash_core::{
    ToolAvailability, ToolCall, ToolContext, ToolDefinition, ToolProvider, ToolResult,
};
use serde_json::{Value, json};

use crate::catalog::CatalogTool;
use crate::common::{LLM_CANDIDATE_LIMIT, args_with_limit, catalog_key, limit_from_args};
use crate::definitions::{load_tools_definition, search_tools_definition};
use crate::ranking::ToolDiscoveryIndex;
use crate::rerank::{llm_rerank_request, merge_llm_selection, parse_llm_tool_names};

#[derive(Clone, Default)]
struct IndexCache {
    index: Option<Arc<ToolDiscoveryIndex>>,
}

#[derive(Clone, Default)]
pub struct ToolDiscoveryToolsProvider {
    cache: Arc<RwLock<IndexCache>>,
}

impl ToolDiscoveryToolsProvider {
    pub fn new() -> Self {
        Self::default()
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
        context: &ToolContext,
    ) -> ToolResult {
        let index = self.index_for_catalog(catalog);
        let limit = limit_from_args(args);
        let candidate_args = args_with_limit(args, LLM_CANDIDATE_LIMIT);
        let candidates = index.search(&candidate_args);
        if candidates.is_empty() {
            return ToolResult::ok(json!([]));
        }

        let request = llm_rerank_request(args, &candidates, limit);
        let completion = match context
            .host()
            .direct_completion(request, "search_tools")
            .await
        {
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

    fn requested_names(args: &Value) -> Result<Vec<String>, ToolResult> {
        let Some(raw) = args.get("names") else {
            return Ok(Vec::new());
        };
        match raw {
            Value::Null => Ok(Vec::new()),
            Value::String(name) => {
                let trimmed = name.trim();
                if trimmed.is_empty() {
                    Ok(Vec::new())
                } else {
                    Ok(vec![trimmed.to_string()])
                }
            }
            Value::Array(values) => values
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .map(str::to_string)
                        .ok_or_else(|| {
                            ToolResult::err_fmt("load_tools.names must contain non-empty strings")
                        })
                })
                .collect(),
            _ => Err(ToolResult::err_fmt(
                "load_tools.names must be a string or list of strings",
            )),
        }
    }

    async fn load_tools(&self, args: &Value, context: &ToolContext) -> ToolResult {
        let catalog = match context.host().tool_catalog(context.session_id()).await {
            Ok(catalog) => catalog,
            Err(err) => return ToolResult::err_fmt(err.to_string()),
        };
        let requested_names = match Self::requested_names(args) {
            Ok(names) => names,
            Err(err) => return err,
        };
        if requested_names.is_empty() {
            return ToolResult::err_fmt("load_tools requires non-empty `names`");
        }

        let by_name = catalog
            .into_iter()
            .filter_map(CatalogTool::from_value)
            .map(|tool| (tool.name.clone(), tool))
            .collect::<BTreeMap<_, _>>();

        let mut loaded = Vec::new();
        let mut already_callable = Vec::new();
        let mut already_showcased = Vec::new();
        let mut not_loadable = Vec::new();
        let mut unknown = Vec::new();
        let mut to_promote = Vec::new();

        for name in requested_names {
            let Some(tool) = by_name.get(&name) else {
                unknown.push(name);
                continue;
            };
            if tool.callable {
                already_callable.push(name);
            } else if tool.showcased {
                already_showcased.push(name);
            } else if tool.loadable {
                to_promote.push(name.clone());
                loaded.push(name);
            } else {
                not_loadable.push(name);
            }
        }

        if !to_promote.is_empty()
            && let Err(err) = context
                .host()
                .set_tools_availability(
                    context.session_id(),
                    &to_promote,
                    Some(ToolAvailability::Showcased),
                )
                .await
        {
            return ToolResult::err_fmt(format_args!("failed to load tools: {err}"));
        }

        ToolResult::ok(json!({
            "loaded": loaded,
            "already_callable": already_callable,
            "already_showcased": already_showcased,
            "not_loadable": not_loadable,
            "unknown": unknown,
        }))
    }
}

#[async_trait::async_trait]
impl ToolProvider for ToolDiscoveryToolsProvider {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![search_tools_definition(), load_tools_definition()]
    }

    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        match call.name {
            "search_tools" => match call
                .context
                .host()
                .tool_catalog(call.context.session_id())
                .await
            {
                Ok(catalog) => self.search_tools(call.args, catalog, call.context).await,
                Err(err) => ToolResult::err_fmt(err.to_string()),
            },
            "load_tools" => self.load_tools(call.args, call.context).await,
            _ => ToolResult::err_fmt(format_args!("Unknown tool: {}", call.name)),
        }
    }
}
