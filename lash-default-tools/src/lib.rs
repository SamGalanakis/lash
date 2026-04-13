use std::sync::Arc;

use lash::instructions::InstructionSource;
use lash::plugin::{PluginFactory, PluginSpec, StaticPluginFactory};
use lash::tools::{
    ApplyPatchTool, AskTool, FetchUrl, Glob, Grep, Ls, ReadFilePluginFactory, ShowSnippetToUser,
    StandardShell, StateToolsPluginFactory, WaitTool, WebSearch, shell_prompt_contributions,
};
use lash::{
    BuiltinObservationalMemoryPluginFactory, BuiltinRollingHistoryPluginFactory,
    BuiltinToolResultProjectionPluginFactory, ExecutionMode, FsInstructionSource, PluginHost,
    ToolProvider,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DefaultToolBundle {
    CoreRuntime,
    ObservationalMemory,
    Shell,
    Editing,
    Files,
    Search,
    UserPrompts,
    Web,
    RlmState,
}

impl DefaultToolBundle {
    pub fn default_surface(execution_mode: ExecutionMode, interactive: bool) -> Vec<Self> {
        let mut bundles = vec![
            Self::CoreRuntime,
            Self::Shell,
            Self::Editing,
            Self::Files,
            Self::Search,
        ];
        if interactive {
            bundles.push(Self::UserPrompts);
        }
        if matches!(execution_mode, ExecutionMode::Rlm) {
            bundles.push(Self::RlmState);
        }
        bundles
    }

    pub fn background_surface(execution_mode: ExecutionMode, interactive: bool) -> Vec<Self> {
        let mut bundles = Self::default_surface(execution_mode, interactive);
        bundles.push(Self::ObservationalMemory);
        bundles
    }
}

#[derive(Default)]
pub struct DefaultToolPluginOptions {
    pub execution_mode: ExecutionMode,
    pub bundles: Vec<DefaultToolBundle>,
    pub tavily_api_key: Option<String>,
    pub instruction_source: Option<Arc<dyn InstructionSource>>,
}

fn shell_provider() -> Arc<dyn ToolProvider> {
    Arc::new(StandardShell::new())
}

pub fn tool_plugin_factories(mut options: DefaultToolPluginOptions) -> Vec<Arc<dyn PluginFactory>> {
    if options.bundles.is_empty() {
        options.bundles = DefaultToolBundle::default_surface(options.execution_mode, true);
    }
    let mut factories: Vec<Arc<dyn PluginFactory>> = Vec::new();
    for bundle in options.bundles {
        match bundle {
            DefaultToolBundle::CoreRuntime => {
                factories.push(Arc::new(BuiltinToolResultProjectionPluginFactory::default()));
                factories.push(Arc::new(BuiltinRollingHistoryPluginFactory::default()));
            }
            DefaultToolBundle::ObservationalMemory => {
                factories.push(Arc::new(BuiltinObservationalMemoryPluginFactory));
            }
            DefaultToolBundle::Shell => {
                factories.push(Arc::new(StaticPluginFactory::new(
                    "shell",
                    PluginSpec::new()
                        .with_tool_provider(shell_provider())
                        .with_prompt_contributor(Arc::new(move |_ctx| {
                            Box::pin(async move { Ok(shell_prompt_contributions()) })
                        })),
                )));
            }
            DefaultToolBundle::Editing => {
                factories.push(Arc::new(StaticPluginFactory::new(
                    "apply_patch",
                    PluginSpec::new()
                        .with_tool_provider(Arc::new(ApplyPatchTool) as Arc<dyn ToolProvider>),
                )));
            }
            DefaultToolBundle::Files => {
                let instruction_source = options
                    .instruction_source
                    .clone()
                    .unwrap_or_else(|| Arc::new(FsInstructionSource::new()));
                factories.push(Arc::new(ReadFilePluginFactory::new(Some(
                    instruction_source,
                ))));
                factories.push(Arc::new(StaticPluginFactory::new(
                    "glob",
                    PluginSpec::new().with_tool_provider(Arc::new(Glob) as Arc<dyn ToolProvider>),
                )));
                factories.push(Arc::new(StaticPluginFactory::new(
                    "ls",
                    PluginSpec::new().with_tool_provider(Arc::new(Ls) as Arc<dyn ToolProvider>),
                )));
            }
            DefaultToolBundle::Search => {
                factories.push(Arc::new(StaticPluginFactory::new(
                    "grep",
                    PluginSpec::new()
                        .with_tool_provider(Arc::new(Grep::new()) as Arc<dyn ToolProvider>),
                )));
            }
            DefaultToolBundle::UserPrompts => {
                factories.push(Arc::new(StaticPluginFactory::new(
                    "ask",
                    PluginSpec::new()
                        .with_tool_provider(Arc::new(AskTool::new()) as Arc<dyn ToolProvider>),
                )));
                factories.push(Arc::new(StaticPluginFactory::new(
                    "wait",
                    PluginSpec::new()
                        .with_tool_provider(Arc::new(WaitTool::new()) as Arc<dyn ToolProvider>),
                )));
                factories.push(Arc::new(StaticPluginFactory::new(
                    "show_snippet_to_user",
                    PluginSpec::new().with_tool_provider(
                        Arc::new(ShowSnippetToUser::new()) as Arc<dyn ToolProvider>
                    ),
                )));
            }
            DefaultToolBundle::Web => {
                if let Some(key) = options.tavily_api_key.clone() {
                    let search_key = key.clone();
                    factories.push(Arc::new(StaticPluginFactory::new(
                        "search_web",
                        PluginSpec::new().with_tool_provider(
                            Arc::new(WebSearch::new(search_key)) as Arc<dyn ToolProvider>
                        ),
                    )));
                    factories.push(Arc::new(StaticPluginFactory::new(
                        "fetch_url",
                        PluginSpec::new().with_tool_provider(
                            Arc::new(FetchUrl::new(key)) as Arc<dyn ToolProvider>
                        ),
                    )));
                }
            }
            DefaultToolBundle::RlmState => {
                factories.push(Arc::new(StateToolsPluginFactory::new()));
            }
        }
    }
    factories
}

pub fn plugin_host_with_bundles(options: DefaultToolPluginOptions) -> PluginHost {
    PluginHost::new(tool_plugin_factories(options))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_surface_for_interactive_rlm_includes_prompt_and_state_tools() {
        let bundles = DefaultToolBundle::default_surface(ExecutionMode::Rlm, true);
        assert!(bundles.contains(&DefaultToolBundle::UserPrompts));
        assert!(bundles.contains(&DefaultToolBundle::RlmState));
        assert!(!bundles.contains(&DefaultToolBundle::ObservationalMemory));
    }

    #[test]
    fn background_surface_adds_observational_memory_bundle() {
        let bundles = DefaultToolBundle::background_surface(ExecutionMode::Standard, false);
        assert!(bundles.contains(&DefaultToolBundle::ObservationalMemory));
    }

    #[test]
    fn observational_memory_bundle_advertises_om_support() {
        let host = PluginHost::new(vec![
            Arc::new(BuiltinRollingHistoryPluginFactory::default()) as Arc<dyn lash::PluginFactory>,
            Arc::new(BuiltinObservationalMemoryPluginFactory) as Arc<dyn lash::PluginFactory>,
        ]);
        assert!(
            host.supports_context_approach(&lash::ContextApproach::ObservationalMemory(
                lash::ObservationalMemoryConfig::default(),
            ))
        );
    }
}
