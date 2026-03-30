use std::collections::HashMap;
use std::sync::Arc;

use serde_json::json;

use crate::plugin::{
    PluginError, PluginFactory, PluginRegistrar, PluginSessionContext, SessionPlugin,
    ToolSurfaceContribution, ToolSurfaceOverride,
};
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

struct ReplStateToolsPlugin {
    provider: Arc<StateStore>,
}

#[derive(Clone)]
pub struct StateStore;

impl StateStore {
    pub fn new() -> Self {
        Self
    }

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
        Ok(Arc::new(ReplStateToolsPlugin {
            provider: Arc::clone(&self.provider),
        }))
    }
}

pub(crate) fn repl_state_prompt_contributions(context: &PromptContext) -> Vec<PromptContribution> {
    if context.omitted_tool_count == 0 {
        return Vec::new();
    }

    vec![PromptContribution::guidance(
        "### Tool Discovery\nUse `search_tools` to inspect the additional available tools that are omitted from Available Tools for brevity. With no query, it browses the full active tool catalog; use focused queries when you know the kind of tool you need.",
    )]
}

impl SessionPlugin for ReplStateToolsPlugin {
    fn id(&self) -> &'static str {
        "state"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        reg.tools()
            .provider(Arc::clone(&self.provider) as Arc<dyn ToolProvider>)?;
        reg.prompt().contribute(Arc::new(move |ctx| {
            Box::pin(async move { Ok(repl_state_prompt_contributions(&ctx.prompt)) })
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
                    injected: Some(false),
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
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockSessionManager {
        catalog: Vec<serde_json::Value>,
    }

    #[async_trait::async_trait]
    impl crate::SessionManager for MockSessionManager {
        async fn snapshot_current(&self) -> Result<crate::SessionSnapshot, crate::PluginError> {
            Ok(crate::AgentStateEnvelope::default())
        }

        async fn snapshot_session(
            &self,
            _session_id: &str,
        ) -> Result<crate::SessionSnapshot, crate::PluginError> {
            Ok(crate::AgentStateEnvelope::default())
        }

        async fn tool_catalog(
            &self,
            _session_id: &str,
        ) -> Result<Vec<serde_json::Value>, crate::PluginError> {
            Ok(self.catalog.clone())
        }

        async fn create_session(
            &self,
            request: crate::SessionCreateRequest,
        ) -> Result<crate::SessionHandle, crate::PluginError> {
            Ok(crate::SessionHandle {
                session_id: request.agent_id.unwrap_or_else(|| "child".to_string()),
                parent_session_id: request.parent_session_id,
                policy: crate::SessionPolicy {
                    provider: crate::Provider::OpenAiGeneric {
                        api_key: String::new(),
                        base_url: "https://example.invalid/v1".to_string(),
                        options: crate::ProviderOptions::default(),
                    },
                    model: "mock-model".to_string(),
                    execution_mode: ExecutionMode::Standard,
                    context_strategy: crate::default_context_strategy(),
                    ..Default::default()
                },
            })
        }

        async fn close_session(&self, _session_id: &str) -> Result<(), crate::PluginError> {
            Ok(())
        }

        async fn start_turn_stream(
            &self,
            session_id: &str,
            _input: crate::TurnInput,
        ) -> Result<crate::plugin::SessionTurnHandle, crate::PluginError> {
            let (_tx, rx) = tokio::sync::mpsc::channel(1);
            Ok(crate::plugin::SessionTurnHandle {
                turn_id: format!("{session_id}-turn"),
                session_id: session_id.to_string(),
                policy: crate::SessionPolicy {
                    provider: crate::Provider::OpenAiGeneric {
                        api_key: String::new(),
                        base_url: "https://example.invalid/v1".to_string(),
                        options: crate::ProviderOptions::default(),
                    },
                    model: "mock-model".to_string(),
                    execution_mode: ExecutionMode::Standard,
                    context_strategy: crate::default_context_strategy(),
                    ..Default::default()
                },
                events: rx,
            })
        }

        async fn await_turn(
            &self,
            _turn_id: &str,
        ) -> Result<crate::AssembledTurn, crate::PluginError> {
            Ok(crate::AssembledTurn {
                state: crate::AgentStateEnvelope::default(),
                status: crate::TurnStatus::Completed,
                assistant_output: crate::AssistantOutput {
                    safe_text: String::new(),
                    raw_text: String::new(),
                    state: crate::OutputState::Usable,
                },
                has_plugin_visible_output: false,
                done_reason: crate::DoneReason::ModelStop,
                execution: crate::ExecutionSummary {
                    mode: ExecutionMode::Standard,
                    had_tool_calls: false,
                    had_code_execution: false,
                },
                token_usage: crate::TokenUsage::default(),
                tool_calls: Vec::new(),
                code_outputs: Vec::new(),
                errors: Vec::new(),
            })
        }

        async fn cancel_turn(&self, _turn_id: &str) -> Result<(), crate::PluginError> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn search_tools_lists_all_without_query() {
        let store = StateStore::new();
        let context = ToolExecutionContext {
            session_id: "root".to_string(),
            host: Arc::new(MockSessionManager {
                catalog: vec![
                    json!({"name":"read_file","description":"Read files","enabled":true,"injected":true,"examples":[]}),
                    json!({"name":"apply_patch","description":"Apply patches","enabled":true,"injected":true,"examples":[]}),
                ],
            }),
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

    #[test]
    fn prompt_contributions_only_describe_tool_discovery_when_needed() {
        let mut prompt = PromptContext {
            omitted_tool_count: 1,
            ..PromptContext::default()
        };
        let with_omissions = repl_state_prompt_contributions(&prompt);
        assert_eq!(with_omissions.len(), 1);
        assert!(with_omissions[0].content.contains("search_tools"));

        prompt.omitted_tool_count = 0;
        assert!(repl_state_prompt_contributions(&prompt).is_empty());
    }

    #[test]
    fn repl_tool_surface_hides_search_tools_when_nothing_is_omitted() {
        let host = crate::PluginHost::new(vec![Arc::new(StateToolsPluginFactory::new())]);
        let session = host.build_standard_session("root", None).expect("session");

        let surface = session
            .resolve_tool_surface(crate::plugin::ToolSurfaceContext {
                session_id: "root".to_string(),
                mode: ExecutionMode::Repl,
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
