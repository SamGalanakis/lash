#[cfg(feature = "sqlite-store")]
use std::collections::HashMap;
use std::sync::Arc;

use serde_json::json;

use crate::plugin::{
    PluginError, PluginFactory, PluginRegistrar, PluginSessionContext, PromptHookContext,
    SessionPlugin, ToolSurfaceContribution, ToolSurfaceOverride,
};
#[cfg(feature = "sqlite-store")]
use crate::search::{SearchDoc, SearchMode, limit_from_args, rank_docs};
use crate::{
    PromptContext, PromptContribution, ToolDefinition, ToolExecutionContext, ToolParam,
    ToolProvider, ToolResult,
};

use super::run_blocking;

#[cfg(test)]
use crate::ExecutionMode;

pub struct StateToolsPluginFactory {
    provider: Arc<StateStore>,
}

struct RlmStateToolsPlugin {
    provider: Arc<StateStore>,
}

#[derive(Clone)]
pub struct StateStore;

impl StateStore {
    pub fn new() -> Self {
        Self
    }

    #[cfg(feature = "sqlite-store")]
    fn search_tools(
        &self,
        args: &serde_json::Value,
        catalog: Vec<serde_json::Value>,
    ) -> ToolResult {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let browse_all = query.trim().is_empty();
        let mode = SearchMode::parse(args.get("mode").and_then(|v| v.as_str()));
        let regex = args.get("regex").and_then(|v| v.as_str());
        let limit = if browse_all && args.get("limit").is_none() {
            usize::MAX
        } else {
            limit_from_args(args)
        };
        let injected_only = args.get("injected_only").and_then(|v| v.as_bool());

        let mut filtered = Vec::new();
        for tool in &catalog {
            if !tool
                .get("enabled")
                .and_then(|v| v.as_bool())
                .unwrap_or(true)
            {
                continue;
            }
            if let Some(injected) = injected_only {
                let inject = tool
                    .get("injected")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                if injected != inject {
                    continue;
                }
            }
            filtered.push(tool.clone());
        }

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
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str())
                            .collect::<Vec<_>>()
                            .join("\n")
                    })
                    .unwrap_or_default();
                let mut fields = HashMap::new();
                fields.insert(
                    "name",
                    tool.get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string(),
                );
                fields.insert(
                    "description",
                    tool.get("description")
                        .and_then(|v| v.as_str())
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
            &[("name", 4.0), ("description", 2.0), ("examples", 1.0)],
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
    fn search_tools(
        &self,
        args: &serde_json::Value,
        catalog: Vec<serde_json::Value>,
    ) -> ToolResult {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();

        let mut filtered = catalog
            .into_iter()
            .filter(|tool| {
                tool.get("enabled")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true)
            })
            .filter(|tool| {
                if query.is_empty() {
                    return true;
                }
                let name = tool
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_ascii_lowercase();
                let description = tool
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_ascii_lowercase();
                name.contains(&query) || description.contains(&query)
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

        ToolResult::ok(json!(filtered))
    }
}

impl Default for StateStore {
    fn default() -> Self {
        Self::new()
    }
}

impl StateToolsPluginFactory {
    pub fn new() -> Self {
        Self {
            provider: Arc::new(StateStore::new()),
        }
    }
}

impl Default for StateToolsPluginFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginFactory for StateToolsPluginFactory {
    fn id(&self) -> &'static str {
        "state"
    }

    fn build(&self, _ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(RlmStateToolsPlugin {
            provider: Arc::clone(&self.provider),
        }))
    }
}

pub(crate) fn rlm_state_prompt_contributions(context: &PromptContext) -> Vec<PromptContribution> {
    if context.omitted_tool_count == 0 {
        return Vec::new();
    }

    vec![PromptContribution::guidance(
        "tool_discovery",
        "Tool Discovery",
        "Use `search_tools` to inspect the additional available tools that are omitted from Available Tools for brevity. With no query, it browses the full active tool catalog; use focused queries when you know the kind of tool you need.",
    )]
}

fn rlm_bound_variables_prompt_contributions(ctx: &PromptHookContext) -> Vec<PromptContribution> {
    let globals = ctx.state.projected_rlm_globals();
    if globals.is_empty() {
        return Vec::new();
    }

    let mut lines = vec![
        "These variables are already bound in lashlang. Access them directly in fenced `lashlang` code; do not recreate them manually.".to_string(),
    ];
    let mut entries = globals.iter().collect::<Vec<_>>();
    entries.sort_by(|left, right| left.0.cmp(right.0));
    for (name, value) in entries {
        let ty = json_value_type_label(value);
        let preview = preview_value(value, 200);
        lines.push(format!("- `{name}`: {ty} = {preview}"));
    }

    vec![PromptContribution::guidance(
        "bound_variables",
        "Bound Variables",
        lines.join("\n"),
    )]
}

fn json_value_type_label(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(n) => {
            if n.is_f64() && n.as_f64().is_some_and(|v| v.fract() != 0.0) {
                "float"
            } else {
                "int"
            }
        }
        serde_json::Value::String(_) => "str",
        serde_json::Value::Array(_) => "list",
        serde_json::Value::Object(_) => "record",
    }
}

fn preview_value(value: &serde_json::Value, max_chars: usize) -> String {
    let serialized = match value {
        serde_json::Value::String(s) => format!("{s:?}"),
        other => other.to_string(),
    };
    let mut chars = serialized.chars();
    let truncated: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{truncated}…")
    } else {
        serialized
    }
}

impl SessionPlugin for RlmStateToolsPlugin {
    fn id(&self) -> &'static str {
        "state"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        reg.tools()
            .provider(Arc::clone(&self.provider) as Arc<dyn ToolProvider>)?;
        reg.prompt().contribute(Arc::new(move |ctx| {
            Box::pin(async move {
                let mut contributions = rlm_state_prompt_contributions(&ctx.prompt);
                contributions.extend(rlm_bound_variables_prompt_contributions(&ctx));
                Ok(contributions)
            })
        }));
        reg.surface().contribute(Arc::new(|ctx| {
            let omitted_tool_count = ctx
                .tools
                .iter()
                .filter(|tool| tool.enabled)
                .filter(|tool| tool.name != "search_tools")
                .filter(|tool| !tool.injected)
                .count();
            let has_search_tools = ctx.tools.iter().any(|tool| tool.name == "search_tools");
            let mut overrides = Vec::new();
            if has_search_tools {
                overrides.push(ToolSurfaceOverride {
                    tool_name: "search_tools".to_string(),
                    enabled: Some(omitted_tool_count > 0),
                    injected: Some(omitted_tool_count > 0),
                });
            }
            let tool_list_notes = if omitted_tool_count > 0 {
                vec![format!(
                    "- **Note:** {omitted_tool_count} additional tool(s) are available but omitted from this prompt for brevity."
                )]
            } else {
                Vec::new()
            };
            Ok(ToolSurfaceContribution {
                overrides,
                tool_list_notes,
            })
        }));
        Ok(())
    }
}

#[async_trait::async_trait]
impl ToolProvider for StateStore {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "search_tools".into(),
            description: "Discover available tools. With a focused `query`, returns ranked matches using hybrid/literal/regex search. With no `query`, returns the full active tool catalog in stable name order.".into(),
            params: vec![
                ToolParam::optional("query", "str"),
                ToolParam::optional("mode", "str"),
                ToolParam::optional("regex", "str"),
                ToolParam::optional("limit", "int"),
                ToolParam::optional("injected_only", "bool"),
            ],
            returns: "list".into(),
            examples: vec![
                "call search_tools { query: \"task planning\" }".into(),
                "call search_tools {}".into(),
            ],
            enabled: true,
            injected: false,
            input_schema_override: None,
            output_schema_override: None,
        }]
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
            "search_tools" => match context.host.tool_catalog(&context.session_id).await {
                Ok(catalog) => {
                    let this = self.clone();
                    let args = args.clone();
                    run_blocking(move || this.search_tools(&args, catalog)).await
                }
                Err(err) => ToolResult::err_fmt(err.to_string()),
            },
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

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_support::MockSessionManager;

    #[tokio::test]
    async fn search_tools_lists_all_without_query() {
        let store = StateStore::new();
        let context = ToolExecutionContext {
            session_id: "root".to_string(),
            host: Arc::new(MockSessionManager::default().with_tool_catalog(vec![
                json!({"name":"read_file","description":"Read files","enabled":true,"injected":true,"examples":[]}),
                json!({"name":"apply_patch","description":"Apply patches","enabled":true,"injected":true,"examples":[]}),
            ])),
        };

        let result = store
            .execute_with_context("search_tools", &json!({}), &context)
            .await;

        assert!(result.success);
        let items = result.result.as_array().expect("array");
        assert_eq!(items.len(), 2);
        assert_eq!(
            items[0].get("name").and_then(|v| v.as_str()),
            Some("apply_patch")
        );
        assert_eq!(
            items[1].get("name").and_then(|v| v.as_str()),
            Some("read_file")
        );
    }

    #[tokio::test]
    async fn search_tools_streaming_execution_preserves_session_context() {
        let store = StateStore::new();
        let context = ToolExecutionContext {
            session_id: "root".to_string(),
            host: Arc::new(MockSessionManager::default().with_tool_catalog(vec![
                json!({"name":"ask","description":"Pause and ask the user a targeted question.","enabled":true,"injected":true,"examples":[]}),
            ])),
        };

        let result = store
            .execute_streaming_with_context("search_tools", &json!({"query":"ask"}), &context, None)
            .await;

        assert!(result.success);
        let items = result.result.as_array().expect("array");
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].get("name").and_then(|v| v.as_str()), Some("ask"));
    }

    #[test]
    fn prompt_contributions_only_describe_tool_discovery_when_needed() {
        let mut prompt = PromptContext {
            omitted_tool_count: 1,
            ..PromptContext::default()
        };
        let with_omissions = rlm_state_prompt_contributions(&prompt);
        assert_eq!(with_omissions.len(), 1);
        assert!(with_omissions[0].content.contains("search_tools"));

        prompt.omitted_tool_count = 0;
        assert!(rlm_state_prompt_contributions(&prompt).is_empty());
    }

    #[test]
    fn prompt_contributions_include_bound_rlm_variables_from_graph_nodes() {
        let mut snapshot = crate::SessionStateEnvelope::default();
        snapshot.policy.execution_mode = crate::ExecutionMode::Rlm;
        snapshot.session_graph.append_plugin(
            crate::INTERNAL_RLM_GLOBALS_PATCH_PLUGIN_TYPE,
            serde_json::to_value(crate::RlmGlobalsPatchPluginBody {
                set: serde_json::Map::from_iter([
                    ("input".to_string(), json!({"path":"src/main.rs"})),
                    ("limit".to_string(), json!(200)),
                ]),
                unset: Vec::new(),
            })
            .expect("patch json"),
        );
        let ctx = crate::PromptHookContext {
            session_id: "root".to_string(),
            host: Arc::new(MockSessionManager::default()),
            prompt: PromptContext::default(),
            state: crate::SessionReadView::new(snapshot),
        };

        let contributions = rlm_bound_variables_prompt_contributions(&ctx);
        assert_eq!(contributions.len(), 1);
        assert_eq!(contributions[0].title.as_deref(), Some("Bound Variables"));
        assert!(contributions[0].content.contains("`input`: record"));
        assert!(contributions[0].content.contains("`limit`: int"));
    }

    #[test]
    fn rlm_tool_surface_hides_search_tools_when_nothing_is_omitted() {
        let host = crate::PluginHost::new(vec![Arc::new(StateToolsPluginFactory::new())]);
        let session = host.build_standard_session("root", None).expect("session");

        let surface = session
            .resolve_tool_surface(crate::plugin::ToolSurfaceContext {
                session_id: "root".to_string(),
                mode: ExecutionMode::Rlm,
                tools: vec![
                    ToolDefinition {
                        name: "search_tools".to_string(),
                        description: "Discover tools".to_string(),
                        params: vec![],
                        returns: "list".to_string(),
                        examples: vec![],
                        enabled: true,
                        injected: false,
                        input_schema_override: None,
                        output_schema_override: None,
                    },
                    ToolDefinition {
                        name: "read_file".to_string(),
                        description: "Read files".to_string(),
                        params: vec![],
                        returns: "str".to_string(),
                        examples: vec![],
                        enabled: true,
                        injected: true,
                        input_schema_override: None,
                        output_schema_override: None,
                    },
                ],
            })
            .expect("tool surface");

        assert!(
            surface
                .tools
                .iter()
                .any(|tool| tool.name == "search_tools" && !tool.enabled && !tool.injected)
        );
    }
}
