#[cfg(feature = "sqlite-store")]
use std::collections::HashMap;
use std::collections::{BTreeMap, BTreeSet};

use serde_json::json;

#[cfg(feature = "sqlite-store")]
use crate::search::{SearchDoc, SearchMode, limit_from_args, rank_docs};
use crate::tools::run_blocking;
use crate::{
    ToolActivation, ToolAvailability, ToolAvailabilityConfig, ToolDefinition, ToolExecutionContext,
    ToolExecutionMode, ToolParam, ToolProvider, ToolResult,
};

#[derive(Clone, Default)]
pub struct DiscoveryToolsProvider;

impl DiscoveryToolsProvider {
    pub fn new() -> Self {
        Self
    }

    fn catalog_matches(
        tool: &serde_json::Value,
        namespace: Option<&str>,
        callable_only: bool,
        documented_only: bool,
        loadable_only: bool,
    ) -> bool {
        let discoverable = tool
            .get("discoverable")
            .and_then(|value| value.as_bool())
            .unwrap_or(true);
        if !discoverable {
            return false;
        }
        if let Some(namespace) = namespace {
            let tool_namespace = tool
                .get("namespace")
                .and_then(|value| value.as_str())
                .unwrap_or_default();
            if tool_namespace != namespace {
                return false;
            }
        }
        if callable_only
            && !tool
                .get("callable")
                .and_then(|value| value.as_bool())
                .unwrap_or(false)
        {
            return false;
        }
        if documented_only
            && !tool
                .get("documented")
                .and_then(|value| value.as_bool())
                .unwrap_or(false)
        {
            return false;
        }
        if loadable_only
            && !tool
                .get("loadable")
                .and_then(|value| value.as_bool())
                .unwrap_or(false)
        {
            return false;
        }
        true
    }

    #[cfg(feature = "sqlite-store")]
    fn discover_tools(
        &self,
        args: &serde_json::Value,
        catalog: Vec<serde_json::Value>,
    ) -> ToolResult {
        let query = args
            .get("query")
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .to_string();
        let browse_all = query.trim().is_empty();
        let mode = SearchMode::parse(args.get("mode").and_then(|value| value.as_str()));
        let regex = args.get("regex").and_then(|value| value.as_str());
        let limit = if browse_all && args.get("limit").is_none() {
            usize::MAX
        } else {
            limit_from_args(args)
        };
        let namespace = args.get("namespace").and_then(|value| value.as_str());
        let callable_only = args
            .get("callable_only")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        let documented_only = args
            .get("documented_only")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        let loadable_only = args
            .get("loadable_only")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);

        let mut filtered = catalog
            .into_iter()
            .filter(|tool| {
                Self::catalog_matches(
                    tool,
                    namespace,
                    callable_only,
                    documented_only,
                    loadable_only,
                )
            })
            .collect::<Vec<_>>();

        if browse_all {
            filtered.sort_by(|left, right| {
                let left_name = left
                    .get("name")
                    .and_then(|value| value.as_str())
                    .unwrap_or_default();
                let right_name = right
                    .get("name")
                    .and_then(|value| value.as_str())
                    .unwrap_or_default();
                left_name.cmp(right_name)
            });
            return ToolResult::ok(json!(filtered.into_iter().take(limit).collect::<Vec<_>>()));
        }

        let docs: Vec<SearchDoc> = filtered
            .iter()
            .map(|tool| {
                let examples = tool
                    .get("examples")
                    .and_then(|value| value.as_array())
                    .map(|items| {
                        items
                            .iter()
                            .filter_map(|value| value.as_str())
                            .collect::<Vec<_>>()
                            .join("\n")
                    })
                    .unwrap_or_default();
                let mut fields = HashMap::new();
                fields.insert(
                    "name",
                    tool.get("name")
                        .and_then(|value| value.as_str())
                        .unwrap_or_default()
                        .to_string(),
                );
                fields.insert(
                    "namespace",
                    tool.get("namespace")
                        .and_then(|value| value.as_str())
                        .unwrap_or_default()
                        .to_string(),
                );
                fields.insert(
                    "description",
                    tool.get("description")
                        .and_then(|value| value.as_str())
                        .unwrap_or_default()
                        .to_string(),
                );
                fields.insert(
                    "activation_hint",
                    tool.get("activation_hint")
                        .and_then(|value| value.as_str())
                        .unwrap_or_default()
                        .to_string(),
                );
                fields.insert("examples", examples);
                SearchDoc { fields }
            })
            .collect();

        let ranked = rank_docs(
            &docs,
            &query,
            mode,
            regex,
            &[
                ("name", 4.0),
                ("namespace", 2.0),
                ("description", 2.0),
                ("activation_hint", 1.5),
                ("examples", 1.0),
            ],
        );
        let items: Vec<serde_json::Value> = ranked
            .into_iter()
            .take(limit)
            .map(|(idx, score, _)| {
                let mut tool = filtered[idx].clone();
                if let Some(obj) = tool.as_object_mut() {
                    obj.insert("score".to_string(), json!(score));
                }
                tool
            })
            .collect();
        ToolResult::ok(json!(items))
    }

    #[cfg(not(feature = "sqlite-store"))]
    fn discover_tools(
        &self,
        args: &serde_json::Value,
        catalog: Vec<serde_json::Value>,
    ) -> ToolResult {
        let query = args
            .get("query")
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();
        let namespace = args.get("namespace").and_then(|value| value.as_str());
        let callable_only = args
            .get("callable_only")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        let documented_only = args
            .get("documented_only")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        let loadable_only = args
            .get("loadable_only")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        let limit = args
            .get("limit")
            .and_then(|value| value.as_u64())
            .map(|value| value as usize);

        let mut filtered = catalog
            .into_iter()
            .filter(|tool| {
                Self::catalog_matches(
                    tool,
                    namespace,
                    callable_only,
                    documented_only,
                    loadable_only,
                )
            })
            .filter(|tool| {
                if query.is_empty() {
                    return true;
                }
                let name = tool
                    .get("name")
                    .and_then(|value| value.as_str())
                    .unwrap_or_default()
                    .to_ascii_lowercase();
                let tool_namespace = tool
                    .get("namespace")
                    .and_then(|value| value.as_str())
                    .unwrap_or_default()
                    .to_ascii_lowercase();
                let description = tool
                    .get("description")
                    .and_then(|value| value.as_str())
                    .unwrap_or_default()
                    .to_ascii_lowercase();
                name.contains(&query)
                    || tool_namespace.contains(&query)
                    || description.contains(&query)
            })
            .collect::<Vec<_>>();

        filtered.sort_by(|left, right| {
            let left_name = left
                .get("name")
                .and_then(|value| value.as_str())
                .unwrap_or_default();
            let right_name = right
                .get("name")
                .and_then(|value| value.as_str())
                .unwrap_or_default();
            left_name.cmp(right_name)
        });

        if let Some(limit) = limit {
            filtered.truncate(limit);
        }
        ToolResult::ok(json!(filtered))
    }

    fn requested_names(args: &serde_json::Value) -> Result<Vec<String>, ToolResult> {
        let Some(raw) = args.get("names") else {
            return Ok(Vec::new());
        };
        match raw {
            serde_json::Value::Null => Ok(Vec::new()),
            serde_json::Value::String(name) => {
                let trimmed = name.trim();
                if trimmed.is_empty() {
                    Ok(Vec::new())
                } else {
                    Ok(vec![trimmed.to_string()])
                }
            }
            serde_json::Value::Array(values) => values
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

    async fn load_tools(
        &self,
        args: &serde_json::Value,
        context: &ToolExecutionContext,
    ) -> ToolResult {
        let catalog = match context.host.tool_catalog(&context.session_id).await {
            Ok(catalog) => catalog,
            Err(err) => return ToolResult::err_fmt(err.to_string()),
        };
        let namespace = args
            .get("namespace")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let requested_names = match Self::requested_names(args) {
            Ok(names) => names,
            Err(err) => return err,
        };

        if requested_names.is_empty() && namespace.is_none() {
            return ToolResult::err_fmt(
                "load_tools requires `names` and/or `namespace` to select tools",
            );
        }

        let mut selected = BTreeSet::new();
        selected.extend(requested_names.iter().cloned());
        if let Some(namespace) = namespace {
            for tool in &catalog {
                let matches_namespace = tool
                    .get("namespace")
                    .and_then(|value| value.as_str())
                    .map(|value| value == namespace)
                    .unwrap_or(false);
                if matches_namespace
                    && let Some(name) = tool.get("name").and_then(|value| value.as_str())
                {
                    selected.insert(name.to_string());
                }
            }
        }

        if selected.is_empty() {
            return ToolResult::err_fmt("load_tools did not match any discoverable tools");
        }

        let catalog_by_name: BTreeMap<String, &serde_json::Value> = catalog
            .iter()
            .filter_map(|tool| {
                tool.get("name")
                    .and_then(|value| value.as_str())
                    .map(|name| (name.to_string(), tool))
            })
            .collect();

        let mut loaded = Vec::new();
        let mut already_available = Vec::new();
        let mut not_loadable = Vec::new();
        let mut unknown = Vec::new();
        let mut to_promote = Vec::new();

        for name in selected {
            let Some(tool) = catalog_by_name.get(&name) else {
                unknown.push(name);
                continue;
            };

            let documented = tool
                .get("documented")
                .and_then(|value| value.as_bool())
                .unwrap_or(false);
            let callable = tool
                .get("callable")
                .and_then(|value| value.as_bool())
                .unwrap_or(false);
            let loadable = tool
                .get("loadable")
                .and_then(|value| value.as_bool())
                .unwrap_or(false);

            if documented || callable {
                already_available.push(name);
            } else if loadable {
                to_promote.push(name.clone());
                loaded.push(name);
            } else {
                not_loadable.push(name);
            }
        }

        if !to_promote.is_empty()
            && let Err(err) = context
                .host
                .set_tools_availability(
                    &context.session_id,
                    &to_promote,
                    Some(ToolAvailability::Documented),
                )
                .await
        {
            return ToolResult::err_fmt(format_args!("failed to load tools: {err}"));
        }

        ToolResult::ok(json!({
            "loaded": loaded,
            "already_available": already_available,
            "not_loadable": not_loadable,
            "unknown": unknown,
        }))
    }
}

#[async_trait::async_trait]
impl ToolProvider for DiscoveryToolsProvider {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![
            ToolDefinition {
                name: "discover_tools".into(),
                description: "Browse or search the discoverable tool catalog for this session. Results include the current availability, whether the tool is callable now, whether it is loadable, and any activation hint.".into(),
                params: vec![
                    ToolParam::optional("query", "str"),
                    ToolParam::optional("namespace", "str"),
                    ToolParam::optional("limit", "int"),
                    ToolParam::optional("callable_only", "bool"),
                    ToolParam::optional("documented_only", "bool"),
                    ToolParam::optional("loadable_only", "bool"),
                    ToolParam::optional("mode", "str"),
                    ToolParam::optional("regex", "str"),
                ],
                returns: "list".into(),
                examples: vec![
                    "discover_tools()".into(),
                    "discover_tools(query=\"web\", loadable_only=true)".into(),
                ],
                availability: ToolAvailabilityConfig::documented(),
                activation: ToolActivation::Always,
                availability_override: None,
                input_schema_override: None,
                output_schema_override: None,
                execution_mode: ToolExecutionMode::Parallel,
            },
            ToolDefinition {
                name: "load_tools".into(),
                description: "Promote discoverable loadable tools into the current session's callable/documented surface. Use this for tools that discover_tools reports as loadable but not currently callable.".into(),
                params: vec![
                    ToolParam::optional("names", "list"),
                    ToolParam::optional("namespace", "str"),
                ],
                returns: "dict".into(),
                examples: vec![
                    "load_tools(names=[\"search_web\", \"fetch_url\"])".into(),
                    "load_tools(namespace=\"web\")".into(),
                ],
                availability: ToolAvailabilityConfig::documented(),
                activation: ToolActivation::Always,
                availability_override: None,
                input_schema_override: None,
                output_schema_override: None,
                execution_mode: ToolExecutionMode::Serial,
            },
        ]
    }

    async fn execute(&self, name: &str, _args: &serde_json::Value) -> ToolResult {
        ToolResult::err_fmt(format_args!("Unknown tool: {name}"))
    }

    async fn execute_with_context(
        &self,
        name: &str,
        args: &serde_json::Value,
        context: &ToolExecutionContext,
    ) -> ToolResult {
        match name {
            "discover_tools" => match context.host.tool_catalog(&context.session_id).await {
                Ok(catalog) => {
                    let this = self.clone();
                    let args = args.clone();
                    run_blocking(move || this.discover_tools(&args, catalog)).await
                }
                Err(err) => ToolResult::err_fmt(err.to_string()),
            },
            "load_tools" => self.load_tools(args, context).await,
            _ => self.execute(name, args).await,
        }
    }

    async fn execute_streaming_with_context(
        &self,
        name: &str,
        args: &serde_json::Value,
        context: &ToolExecutionContext,
        _progress: Option<&crate::ProgressSender>,
    ) -> ToolResult {
        self.execute_with_context(name, args, context).await
    }
}
