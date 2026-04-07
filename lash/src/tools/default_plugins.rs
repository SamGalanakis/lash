use std::collections::BTreeMap;
use std::sync::Arc;

use crate::instructions::InstructionSource;
use crate::plugin::{PluginFactory, PluginSpec, StaticPluginFactory};
use crate::{
    ExecutionMode, ProgressSender, ToolDefinition, ToolExecutionContext, ToolProvider, ToolResult,
};

#[derive(Default)]
pub struct DefaultToolPluginDeps {
    pub store: Option<Arc<dyn crate::store::RuntimeStore>>,
    pub tavily_api_key: Option<String>,
    pub enable_user_prompts: bool,
    pub instruction_source: Option<Arc<dyn InstructionSource>>,
}

fn shell_provider() -> Arc<dyn ToolProvider> {
    Arc::new(super::StandardShell::new())
}

pub fn default_tool_plugin_factories(
    mode: ExecutionMode,
    deps: DefaultToolPluginDeps,
) -> Vec<Arc<dyn PluginFactory>> {
    #[cfg(not(feature = "sqlite-store"))]
    let _ = mode;

    let mut factories: Vec<Arc<dyn PluginFactory>> = vec![
        Arc::new(crate::BuiltinToolResultProjectionPluginFactory::default()),
        Arc::new(StaticPluginFactory::new(
            "shell",
            PluginSpec::new()
                .with_tool_provider(shell_provider())
                .with_prompt_contributor(Arc::new(move |_ctx| {
                    Box::pin(async move { Ok(super::shell::shell_prompt_contributions()) })
                })),
        )),
        Arc::new(StaticPluginFactory::new(
            "apply_patch",
            PluginSpec::new()
                .with_tool_provider(Arc::new(super::ApplyPatchTool) as Arc<dyn ToolProvider>),
        )),
        Arc::new(super::ReadFilePluginFactory::new(
            deps.instruction_source.clone(),
        )),
        Arc::new(StaticPluginFactory::new(
            "glob",
            PluginSpec::new().with_tool_provider(Arc::new(super::Glob) as Arc<dyn ToolProvider>),
        )),
        Arc::new(StaticPluginFactory::new(
            "grep",
            PluginSpec::new().with_tool_provider(Arc::new(super::Grep) as Arc<dyn ToolProvider>),
        )),
        Arc::new(StaticPluginFactory::new(
            "ls",
            PluginSpec::new().with_tool_provider(Arc::new(super::Ls) as Arc<dyn ToolProvider>),
        )),
    ];

    if deps.enable_user_prompts {
        factories.push(Arc::new(StaticPluginFactory::new(
            "ask",
            PluginSpec::new()
                .with_tool_provider(Arc::new(super::AskTool::new()) as Arc<dyn ToolProvider>),
        )));
        factories.push(Arc::new(StaticPluginFactory::new(
            "showcase_snippet",
            PluginSpec::new().with_tool_provider(
                Arc::new(super::ShowcaseSnippet::new()) as Arc<dyn ToolProvider>
            ),
        )));
    }

    #[cfg(feature = "sqlite-store")]
    if deps.store.is_some() && matches!(mode, ExecutionMode::Repl) {
        factories.push(Arc::new(super::StateToolsPluginFactory::new()));
    }

    if let Some(key) = deps.tavily_api_key {
        let search_key = key.clone();
        factories.push(Arc::new(StaticPluginFactory::new(
            "search_web",
            PluginSpec::new().with_tool_provider(
                Arc::new(super::WebSearch::new(search_key)) as Arc<dyn ToolProvider>
            ),
        )));
        factories.push(Arc::new(StaticPluginFactory::new(
            "fetch_url",
            PluginSpec::new()
                .with_tool_provider(Arc::new(super::FetchUrl::new(key)) as Arc<dyn ToolProvider>),
        )));
    }
    factories
}

pub(crate) struct CompositeToolProvider {
    tools: BTreeMap<String, (ToolDefinition, usize)>,
    providers: Vec<(Arc<dyn ToolProvider>, Vec<String>)>,
}

impl CompositeToolProvider {
    pub(crate) fn from_providers(providers: Vec<Arc<dyn ToolProvider>>) -> Self {
        let mut tools = BTreeMap::new();
        let mut entries = Vec::new();
        for provider in providers {
            let tool_names = provider
                .definitions()
                .into_iter()
                .map(|def| {
                    let name = def.name.clone();
                    tools.insert(name.clone(), (def, entries.len()));
                    name
                })
                .collect::<Vec<_>>();
            entries.push((provider, tool_names));
        }
        Self {
            tools,
            providers: entries,
        }
    }
}

#[async_trait::async_trait]
impl ToolProvider for CompositeToolProvider {
    fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.values().map(|(def, _)| def.clone()).collect()
    }

    fn dynamic_generation(&self) -> Option<u64> {
        self.providers
            .iter()
            .filter_map(|(provider, _)| provider.dynamic_generation())
            .max()
    }

    async fn execute(&self, name: &str, args: &serde_json::Value) -> ToolResult {
        match self.tools.get(name) {
            Some((_, provider_idx)) => self.providers[*provider_idx].0.execute(name, args).await,
            None => ToolResult::err_fmt(format_args!("Unknown tool: {name}")),
        }
    }

    async fn execute_with_context(
        &self,
        name: &str,
        args: &serde_json::Value,
        context: &ToolExecutionContext,
    ) -> ToolResult {
        match self.tools.get(name) {
            Some((_, provider_idx)) => {
                self.providers[*provider_idx]
                    .0
                    .execute_with_context(name, args, context)
                    .await
            }
            None => ToolResult::err_fmt(format_args!("Unknown tool: {name}")),
        }
    }

    async fn execute_streaming(
        &self,
        name: &str,
        args: &serde_json::Value,
        progress: Option<&ProgressSender>,
    ) -> ToolResult {
        match self.tools.get(name) {
            Some((_, provider_idx)) => {
                self.providers[*provider_idx]
                    .0
                    .execute_streaming(name, args, progress)
                    .await
            }
            None => ToolResult::err_fmt(format_args!("Unknown tool: {name}")),
        }
    }

    async fn execute_streaming_with_context(
        &self,
        name: &str,
        args: &serde_json::Value,
        context: &ToolExecutionContext,
        progress: Option<&ProgressSender>,
    ) -> ToolResult {
        match self.tools.get(name) {
            Some((_, provider_idx)) => {
                self.providers[*provider_idx]
                    .0
                    .execute_streaming_with_context(name, args, context, progress)
                    .await
            }
            None => ToolResult::err_fmt(format_args!("Unknown tool: {name}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interactive_default_tools_include_showcase_snippet() {
        let factories = default_tool_plugin_factories(
            crate::ExecutionMode::Standard,
            DefaultToolPluginDeps {
                enable_user_prompts: true,
                ..Default::default()
            },
        );

        assert!(
            factories
                .iter()
                .any(|factory| factory.id() == "showcase_snippet")
        );
    }
}
