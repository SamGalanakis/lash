use std::sync::{Arc, RwLock};

use lash_core::{ToolCall, ToolContext, ToolResult};
use lash_tool_support::{StaticToolExecute, StaticToolProvider};
use serde_json::{Value, json};

use crate::common::{LLM_CANDIDATE_LIMIT, args_with_limit, catalog_key, limit_from_args};
use crate::definitions::search_tools_definition;
use crate::ranking::ToolDiscoveryIndex;
use crate::rerank::{
    llm_rerank_request, merge_llm_selection, parse_llm_tool_names, rerank_payment_action_intent,
};

#[derive(Clone, Default)]
struct IndexCache {
    index: Option<CachedIndex>,
}

#[derive(Clone)]
struct CachedIndex {
    catalog_addr: usize,
    catalog_len: usize,
    index: Arc<ToolDiscoveryIndex>,
}

impl CachedIndex {
    fn matches_catalog_arc(&self, catalog: &Arc<Vec<Value>>) -> bool {
        self.catalog_addr == Arc::as_ptr(catalog) as usize && self.catalog_len == catalog.len()
    }
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

    fn index_for_catalog(&self, catalog: Arc<Vec<Value>>) -> Arc<ToolDiscoveryIndex> {
        if let Some(index) = self
            .cache
            .read()
            .expect("tool discovery cache lock poisoned")
            .index
            .as_ref()
            .filter(|cached| cached.matches_catalog_arc(&catalog))
            .map(|cached| Arc::clone(&cached.index))
        {
            return index;
        }

        let key = catalog_key(catalog.as_ref());
        if let Some(index) = self
            .cache
            .read()
            .expect("tool discovery cache lock poisoned")
            .index
            .as_ref()
            .filter(|cached| cached.index.key == key)
            .map(|cached| Arc::clone(&cached.index))
        {
            return index;
        }

        let index = Arc::new(ToolDiscoveryIndex::build(key, catalog.as_ref()));
        let cached = CachedIndex {
            catalog_addr: Arc::as_ptr(&catalog) as usize,
            catalog_len: catalog.len(),
            index: Arc::clone(&index),
        };
        self.cache
            .write()
            .expect("tool discovery cache lock poisoned")
            .index = Some(cached);
        index
    }

    async fn search_tools(
        &self,
        args: &Value,
        catalog: Arc<Vec<Value>>,
        context: &ToolContext<'_>,
    ) -> ToolResult {
        let index = self.index_for_catalog(catalog);
        let limit = limit_from_args(args);
        let candidate_args = args_with_limit(args, LLM_CANDIDATE_LIMIT);
        let candidates = index.search(&candidate_args);
        if candidates.is_empty() {
            return ToolResult::ok(json!([]));
        }
        let query = args
            .get("query")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let candidates = rerank_payment_action_intent(query, candidates);

        let model = match context.sessions().model().await {
            Ok(model) => model,
            Err(err) => {
                return ToolResult::err_fmt(format_args!(
                    "search_tools could not resolve parent model: {err}"
                ));
            }
        };
        let request =
            llm_rerank_request(args, &candidates, limit, model.model, model.model_variant);
        let completion = match context
            .direct_completions()
            .complete(request, "search_tools")
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
}

/// Build the `search_tools` provider backed by a fresh discovery cache.
pub fn tool_discovery_provider() -> StaticToolProvider<ToolDiscoveryToolsProvider> {
    StaticToolProvider::new(
        vec![search_tools_definition()],
        ToolDiscoveryToolsProvider::new(),
    )
}

#[async_trait::async_trait]
impl StaticToolExecute for ToolDiscoveryToolsProvider {
    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        match call.name {
            "search_tools" => match call.context.sessions().shared_tool_catalog().await {
                Ok(catalog) => self.search_tools(call.args, catalog, call.context).await,
                Err(err) => ToolResult::err_fmt(err.to_string()),
            },
            _ => ToolResult::err_fmt(format_args!("Unknown tool: {}", call.name)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lash_core::plugin::runtime_host::{
        SessionGraphService, SessionLifecycleService, SessionStateService,
    };
    use lash_core::plugin::{PluginError, SessionHandle, SessionSnapshot};
    use lash_core::{
        DirectCompletion, TokenUsage, ToolCall, ToolContract, ToolDefinition, ToolProvider,
    };
    use lash_tool_support::{LashlangToolBinding, ToolDefinitionLashlangExt};
    use serde_json::json;
    use std::sync::Mutex;

    #[derive(Default)]
    struct FakeSessionManager {
        snapshot: SessionSnapshot,
        catalog: Vec<Value>,
        direct_response: Mutex<Option<String>>,
        direct_requests: Mutex<Vec<lash_core::DirectRequest>>,
    }

    #[async_trait::async_trait]
    impl SessionStateService for FakeSessionManager {
        async fn snapshot_current(&self) -> Result<SessionSnapshot, PluginError> {
            Ok(self.snapshot.clone())
        }

        async fn snapshot_session(
            &self,
            _session_id: &str,
        ) -> Result<SessionSnapshot, PluginError> {
            Ok(self.snapshot.clone())
        }

        async fn tool_catalog(&self, _session_id: &str) -> Result<Vec<Value>, PluginError> {
            Ok(self.catalog.clone())
        }

        async fn set_tools_availability(
            &self,
            _session_id: &str,
            _tool_names: &[String],
            _availability: Option<lash_core::ToolAvailability>,
        ) -> Result<u64, PluginError> {
            Ok(0)
        }
    }

    #[async_trait::async_trait]
    impl SessionLifecycleService for FakeSessionManager {
        async fn create_session(
            &self,
            _request: lash_core::plugin::SessionCreateRequest,
        ) -> Result<SessionHandle, PluginError> {
            Err(PluginError::Session("unused".to_string()))
        }

        async fn close_session(&self, _session_id: &str) -> Result<(), PluginError> {
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl SessionGraphService for FakeSessionManager {}

    fn snapshot_with_model(model: &str, variant: Option<&str>) -> SessionSnapshot {
        let mut snapshot = SessionSnapshot::default();
        snapshot.policy.model.id = model.to_string();
        snapshot.policy.model.variant = variant.map(str::to_string);
        snapshot
    }

    fn discovery_context(host: Arc<FakeSessionManager>) -> lash_core::ToolContext<'static> {
        let direct_host = Arc::clone(&host);
        lash_core::testing::mock_tool_context_with_host_and_direct_completions(
            host,
            lash_core::DirectCompletionClient::from_fn(move |request, _usage_source| {
                direct_host
                    .direct_requests
                    .lock()
                    .expect("direct requests lock poisoned")
                    .push(request);
                let text = direct_host
                    .direct_response
                    .lock()
                    .expect("direct response lock poisoned")
                    .clone()
                    .unwrap_or_else(|| "{\"tool_names\":[]}".to_string());
                Ok(DirectCompletion {
                    text,
                    usage: TokenUsage::default(),
                })
            }),
        )
    }

    fn catalog_tool_with_metadata(
        name: &str,
        description: &str,
        module: Option<&str>,
        aliases: Vec<&str>,
    ) -> Value {
        let tool = ToolDefinition::raw(
            format!("tool:test/{name}"),
            name,
            description,
            ToolContract::default_input_schema(),
            json!({}),
        )
        .with_lashlang_binding(
            LashlangToolBinding::new(
                [module.unwrap_or(match name {
                    "read_file" => "files",
                    "search_web" => "web",
                    _ => "tools",
                })],
                match name {
                    "read_file" => "read",
                    "search_web" => "search",
                    _ => name,
                },
            )
            .with_aliases(aliases),
        );
        let manifest = tool.manifest();
        let projected = json!({
            "id": manifest.id,
            "name": manifest.name,
            "description": manifest.description,
            "availability": "searchable",
            "callable": false,
            "showcased": false,
            "searchable": true,
            "activation": manifest.activation,
            "contract": manifest.compact_contract.clone().expect("compact contract"),
        });
        #[cfg(feature = "lashlang")]
        {
            let mut projected = projected;
            let lashlang_binding =
                lash_lashlang_runtime::required_tool_lashlang_executable(&manifest)
                    .expect("catalog test tool has explicit Lashlang binding");
            let call = lashlang_binding.call_path();
            let projected_object = projected.as_object_mut().expect("catalog object");
            projected_object.insert(
                "module_path".to_string(),
                json!(lashlang_binding.module_path.clone()),
            );
            projected_object.insert(
                "operation".to_string(),
                json!(lashlang_binding.operation.clone()),
            );
            projected_object.insert("call".to_string(), json!(call));
            projected_object.insert(
                "aliases".to_string(),
                json!(lashlang_binding.aliases.clone()),
            );
            projected
        }
        #[cfg(not(feature = "lashlang"))]
        {
            projected
        }
    }

    fn ranked_names(results: &[Value]) -> Vec<String> {
        results
            .iter()
            .map(|result| {
                result
                    .get("name")
                    .and_then(Value::as_str)
                    .expect("ranked result name")
                    .to_string()
            })
            .collect()
    }

    #[test]
    fn provider_exposes_search_tools_only() {
        let manifests = tool_discovery_provider().tool_manifests();
        let names = manifests
            .iter()
            .map(|definition| definition.name.as_str())
            .collect::<Vec<_>>();

        assert_eq!(names, vec!["search_tools"]);
        #[cfg(not(feature = "lashlang"))]
        assert!(manifests[0].bindings.is_empty());
        #[cfg(feature = "lashlang")]
        assert!(
            manifests[0]
                .bindings
                .contains_key(lash_lashlang_runtime::LASHLANG_TOOL_BINDING_KEY)
        );
    }

    #[test]
    fn index_cache_records_and_reuses_shared_catalog_identity() {
        let provider = ToolDiscoveryToolsProvider::new();
        let catalog = Arc::new(vec![catalog_tool_with_metadata(
            "read_file",
            "Read file contents",
            Some("filesystem"),
            vec!["cat"],
        )]);

        let first = provider.index_for_catalog(Arc::clone(&catalog));
        let second = provider.index_for_catalog(Arc::clone(&catalog));
        assert!(Arc::ptr_eq(&first, &second));

        let cache = provider.cache.read().expect("cache lock");
        let cached = cache.index.as_ref().expect("cached index");
        assert_eq!(cached.catalog_addr, Arc::as_ptr(&catalog) as usize);
        assert_eq!(cached.catalog_len, catalog.len());
        drop(cache);

        let same_content = Arc::new(catalog.as_ref().clone());
        let third = provider.index_for_catalog(same_content);
        assert!(Arc::ptr_eq(&first, &third));
    }

    #[tokio::test]
    async fn search_tools_uses_host_catalog_and_projects_compact_contract() {
        let host = Arc::new(FakeSessionManager {
            catalog: vec![
                catalog_tool_with_metadata(
                    "read_file",
                    "Read file contents",
                    Some("filesystem"),
                    vec!["cat"],
                ),
                catalog_tool_with_metadata(
                    "search_web",
                    "Search the web",
                    Some("web"),
                    vec!["web_search"],
                ),
            ],
            ..Default::default()
        });
        let provider = tool_discovery_provider();
        let context = discovery_context(host);

        #[cfg(feature = "lashlang")]
        let args = json!({
            "query": "cat",
            "module": "filesystem",
            "limit": 1,
        });
        #[cfg(not(feature = "lashlang"))]
        let args = json!({
            "query": "read file",
            "limit": 1,
        });
        let result = provider
            .execute(ToolCall {
                name: "search_tools",
                args: &args,
                context: &context,
                progress: None,
            })
            .await;

        assert!(result.is_success(), "{result:?}");
        let value = result.value_for_projection();
        let results = value.as_array().expect("search result list");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["id"], json!("tool:test/read_file"));
        assert_eq!(results[0]["name"], json!("read_file"));
        #[cfg(feature = "lashlang")]
        assert_eq!(results[0]["call"], json!("filesystem.read"));
        #[cfg(not(feature = "lashlang"))]
        assert!(results[0].get("call").is_none());
        assert!(
            results[0]["signature"]
                .as_str()
                .expect("signature")
                .starts_with("read_file({})")
        );
        assert_eq!(results[0]["description"], json!("Read file contents"));
        #[cfg(feature = "lashlang")]
        assert_eq!(results[0]["module_path"], json!(["filesystem"]));
        #[cfg(not(feature = "lashlang"))]
        assert!(results[0].get("module_path").is_none());
        assert!(results[0].get("score").is_none());
    }

    #[tokio::test]
    async fn search_tools_reranks_candidates_with_direct_completion() {
        let host = Arc::new(FakeSessionManager {
            snapshot: snapshot_with_model("gpt-5.5", Some("medium")),
            catalog: vec![
                catalog_tool_with_metadata("read_file", "Read file contents", None, vec!["cat"]),
                catalog_tool_with_metadata("search_web", "Search the web", None, vec!["web"]),
            ],
            direct_response: Mutex::new(Some(
                "{\"tool_names\":[\"search_web\",\"search_web\",\"unknown\"]}".to_string(),
            )),
            ..Default::default()
        });
        let provider = tool_discovery_provider();
        let context = discovery_context(host.clone());

        let args = json!({
            "query": "",
            "exclude": ["read_file"],
            "limit": 2,
        });
        let result = provider
            .execute(ToolCall {
                name: "search_tools",
                args: &args,
                context: &context,
                progress: None,
            })
            .await;

        assert!(result.is_success(), "{result:?}");
        let value = result.value_for_projection();
        let results = value.as_array().expect("search result list");
        assert_eq!(ranked_names(results), vec!["search_web"]);
        let requests = host
            .direct_requests
            .lock()
            .expect("direct requests lock poisoned");
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].model, "gpt-5.5");
        assert_eq!(requests[0].model_variant.as_deref(), Some("medium"));
    }
}
