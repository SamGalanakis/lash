use std::sync::Arc;

use lash_core::instructions::InstructionSource;
use lash_core::plugin::{PluginFactory, PluginSpec, StaticPluginFactory};
use lash_core::{
    ExecutionMode, FsInstructionSource, PluginHost, StandardContextApproach,
    ToolOutputBudgetPluginFactory, ToolProvider,
};
use lash_plugin_observational_memory::ObservationalMemoryPluginFactory;
use lash_plugin_rolling_history::RollingHistoryPluginFactory;
use lash_plugin_tool_discovery::ToolDiscoveryPluginFactory;
use lash_tool_apply_patch::ApplyPatchTool;
use lash_tool_files::{Glob, Ls, ReadFilePluginFactory};
use lash_tool_search::Grep;
use lash_tool_shell::StandardShellPluginFactory;
use lash_tool_web::{FetchUrl, WebSearch};

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
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DefaultToolSurfaceProfile {
    pub bundles: Vec<DefaultToolBundle>,
    pub interactive_extras: bool,
}

impl DefaultToolSurfaceProfile {
    pub fn for_runtime(
        standard_context_approach: Option<&StandardContextApproach>,
        interactive: bool,
        web_enabled: bool,
    ) -> Self {
        let mut bundles = vec![DefaultToolBundle::CoreRuntime];
        if let Some(standard_context_approach) = standard_context_approach {
            bundles.push(match standard_context_approach.kind() {
                lash_core::StandardContextApproachKind::RollingHistory => {
                    DefaultToolBundle::RollingHistory
                }
                lash_core::StandardContextApproachKind::ObservationalMemory => {
                    DefaultToolBundle::ObservationalMemory
                }
            });
        }
        bundles.extend([
            DefaultToolBundle::Shell,
            DefaultToolBundle::Editing,
            DefaultToolBundle::Files,
            DefaultToolBundle::Search,
        ]);
        if interactive {
            bundles.push(DefaultToolBundle::UserPrompts);
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
    pub standard_context_approach: Option<StandardContextApproach>,
    pub bundles: Vec<DefaultToolBundle>,
    pub tavily_api_key: Option<String>,
    pub instruction_source: Option<Arc<dyn InstructionSource>>,
}

pub fn tool_plugin_factories(mut options: DefaultToolPluginOptions) -> Vec<Arc<dyn PluginFactory>> {
    if options.bundles.is_empty() {
        options.bundles = DefaultToolSurfaceProfile::for_runtime(
            options.standard_context_approach.as_ref(),
            true,
            options.tavily_api_key.is_some(),
        )
        .bundles;
    }
    let mut factories: Vec<Arc<dyn PluginFactory>> = Vec::new();
    for bundle in options.bundles {
        match bundle {
            DefaultToolBundle::CoreRuntime => {
                factories.push(Arc::new(ToolDiscoveryPluginFactory::new()));
                factories.push(Arc::new(ToolOutputBudgetPluginFactory::default()));
            }
            DefaultToolBundle::RollingHistory => {
                factories.push(Arc::new(RollingHistoryPluginFactory::default()));
            }
            DefaultToolBundle::ObservationalMemory => {
                factories.push(Arc::new(ObservationalMemoryPluginFactory));
            }
            DefaultToolBundle::Shell => {
                factories.push(Arc::new(StandardShellPluginFactory::new()));
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
            DefaultToolBundle::UserPrompts => {}
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

impl EmbeddedRuntimeBuilderExt for lash_core::EmbeddedRuntimeBuilder {
    fn with_default_tool_bundles(self, mut options: DefaultToolPluginOptions) -> Self {
        if options.bundles.is_empty()
            && let Some(policy) = self.policy()
        {
            options.execution_mode = policy.execution_mode.clone();
            options.standard_context_approach = policy.standard_context_approach.clone();
        }
        self.with_plugin_host(plugin_host_with_bundles(options))
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
            policy.standard_context_approach.as_ref(),
            interactive,
            web_enabled,
        );
        self.with_default_tool_bundles(DefaultToolPluginOptions {
            execution_mode: policy.execution_mode,
            standard_context_approach: policy.standard_context_approach,
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
    fn rolling_history_profile_for_interactive_standard_includes_prompt_tools() {
        let profile = DefaultToolSurfaceProfile::for_runtime(
            Some(&StandardContextApproach::RollingHistory(Default::default())),
            true,
            false,
        );
        assert!(profile.bundles.contains(&DefaultToolBundle::UserPrompts));
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
            Some(&StandardContextApproach::ObservationalMemory(
                Default::default(),
            )),
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
            Arc::new(RollingHistoryPluginFactory::default()) as Arc<dyn lash_core::PluginFactory>,
            Arc::new(ObservationalMemoryPluginFactory) as Arc<dyn lash_core::PluginFactory>,
        ]);
        assert!(host.supports_standard_context_approach(
            &lash_core::StandardContextApproach::ObservationalMemory(
                lash_core::ObservationalMemoryConfig::default(),
            )
        ));
    }
}
