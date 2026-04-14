use std::sync::Arc;

use lash::instructions::InstructionSource;
use lash::plugin::{PluginFactory, PluginSpec, StaticPluginFactory};
use lash::tools::{
    ApplyPatchTool, AskTool, FetchUrl, Glob, Grep, Ls, ReadFilePluginFactory, ShowSnippetToUser,
    StandardShell, StateToolsPluginFactory, WaitTool, WebSearch, shell_prompt_contributions,
};
use lash::{
    BuiltinObservationalMemoryPluginFactory, BuiltinRollingHistoryPluginFactory,
    BuiltinToolResultProjectionPluginFactory, ContextApproach, ExecutionMode, FsInstructionSource,
    PluginHost, ToolProvider,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DefaultToolBundle {
    CoreRuntime,
    RollingHistory,
    ObservationalMemory,
    Shell,
    Editing,
    Files,
    Search,
    UserPrompts,
    Web,
    RlmState,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DefaultToolSurfaceProfile {
    pub bundles: Vec<DefaultToolBundle>,
    pub interactive_extras: bool,
}

impl DefaultToolSurfaceProfile {
    pub fn for_runtime(
        execution_mode: ExecutionMode,
        context_approach: &ContextApproach,
        interactive: bool,
        web_enabled: bool,
    ) -> Self {
        let mut bundles = vec![
            DefaultToolBundle::CoreRuntime,
            match context_approach.kind() {
                lash::ContextApproachKind::RollingHistory => DefaultToolBundle::RollingHistory,
                lash::ContextApproachKind::ObservationalMemory => {
                    DefaultToolBundle::ObservationalMemory
                }
            },
            DefaultToolBundle::Shell,
            DefaultToolBundle::Editing,
            DefaultToolBundle::Files,
            DefaultToolBundle::Search,
        ];
        if interactive {
            bundles.push(DefaultToolBundle::UserPrompts);
        }
        if matches!(execution_mode, ExecutionMode::Rlm) {
            bundles.push(DefaultToolBundle::RlmState);
        }
        if web_enabled {
            bundles.push(DefaultToolBundle::Web);
        }
        Self {
            bundles,
            interactive_extras: interactive,
        }
    }
}

#[derive(Default)]
pub struct DefaultToolPluginOptions {
    pub execution_mode: ExecutionMode,
    pub context_approach: ContextApproach,
    pub bundles: Vec<DefaultToolBundle>,
    pub tavily_api_key: Option<String>,
    pub instruction_source: Option<Arc<dyn InstructionSource>>,
}

fn shell_provider() -> Arc<dyn ToolProvider> {
    Arc::new(StandardShell::new())
}

pub fn tool_plugin_factories(mut options: DefaultToolPluginOptions) -> Vec<Arc<dyn PluginFactory>> {
    if options.bundles.is_empty() {
        options.bundles = DefaultToolSurfaceProfile::for_runtime(
            options.execution_mode,
            &options.context_approach,
            true,
            options.tavily_api_key.is_some(),
        )
        .bundles;
    }
    let mut factories: Vec<Arc<dyn PluginFactory>> = Vec::new();
    for bundle in options.bundles {
        match bundle {
            DefaultToolBundle::CoreRuntime => {
                factories.push(Arc::new(BuiltinToolResultProjectionPluginFactory::default()));
            }
            DefaultToolBundle::RollingHistory => {
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

pub trait EmbeddedRuntimeBuilderExt {
    fn with_default_tool_bundles(self, options: DefaultToolPluginOptions) -> Self;
    fn with_default_tool_surface_profile(
        self,
        interactive: bool,
        web_enabled: bool,
        tavily_api_key: Option<String>,
        instruction_source: Option<Arc<dyn InstructionSource>>,
    ) -> Self;
}

impl EmbeddedRuntimeBuilderExt for lash::EmbeddedRuntimeBuilder {
    fn with_default_tool_bundles(self, mut options: DefaultToolPluginOptions) -> Self {
        if options.bundles.is_empty()
            && let Some(policy) = self.policy()
        {
            options.execution_mode = policy.execution_mode;
            options.context_approach = policy.context_approach.clone();
        }
        self.with_plugin_host(plugin_host_with_bundles(options).with_dynamic_tools())
    }

    fn with_default_tool_surface_profile(
        self,
        interactive: bool,
        web_enabled: bool,
        tavily_api_key: Option<String>,
        instruction_source: Option<Arc<dyn InstructionSource>>,
    ) -> Self {
        let policy = self.policy().cloned().unwrap_or_default();
        let profile = DefaultToolSurfaceProfile::for_runtime(
            policy.execution_mode,
            &policy.context_approach,
            interactive,
            web_enabled,
        );
        self.with_default_tool_bundles(DefaultToolPluginOptions {
            execution_mode: policy.execution_mode,
            context_approach: policy.context_approach,
            bundles: profile.bundles,
            tavily_api_key,
            instruction_source,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rolling_history_profile_for_interactive_rlm_includes_prompt_and_state_tools() {
        let profile = DefaultToolSurfaceProfile::for_runtime(
            ExecutionMode::Rlm,
            &ContextApproach::RollingHistory(Default::default()),
            true,
            false,
        );
        assert!(profile.bundles.contains(&DefaultToolBundle::UserPrompts));
        assert!(profile.bundles.contains(&DefaultToolBundle::RlmState));
        assert!(profile.bundles.contains(&DefaultToolBundle::RollingHistory));
        assert!(
            !profile
                .bundles
                .contains(&DefaultToolBundle::ObservationalMemory)
        );
    }

    #[test]
    fn observational_memory_profile_selects_om_bundle() {
        let profile = DefaultToolSurfaceProfile::for_runtime(
            ExecutionMode::Standard,
            &ContextApproach::ObservationalMemory(Default::default()),
            false,
            false,
        );
        assert!(
            profile
                .bundles
                .contains(&DefaultToolBundle::ObservationalMemory)
        );
        assert!(!profile.bundles.contains(&DefaultToolBundle::RollingHistory));
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
