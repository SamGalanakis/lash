use std::sync::Arc;

use lash_core::plugin::{PluginSpec, StaticPluginFactory};
use lash_core::{PluginStack, StandardContextApproach, ToolProvider};
use lash_plugin_observational_memory::ObservationalMemoryPluginFactory;
use lash_plugin_process_controls::ProcessControlsPluginFactory;
use lash_plugin_rolling_history::RollingHistoryPluginFactory;
use lash_plugin_tool_discovery::ToolDiscoveryPluginFactory;
use lash_plugin_tool_output_budget::{ToolOutputBudgetPluginFactory, tool_output_budget_stack};
use lash_tool_apply_patch::ApplyPatchTool;
use lash_tool_files::{Glob, Ls, ReadFilePluginFactory};
use lash_tool_search::Grep;
use lash_tool_shell::StandardShellPluginFactory;
use lash_tool_web::{FetchUrl, WebSearch};

#[derive(Clone, Debug, Default)]
pub struct StandardToolStackOptions {
    pub standard_context_approach: Option<StandardContextApproach>,
    pub tavily_api_key: Option<String>,
}

pub fn standard_tool_stack(options: StandardToolStackOptions) -> PluginStack {
    let mut stack = PluginStack::new();
    push_core_runtime_tools(&mut stack);
    push_standard_context_tools(&mut stack, options.standard_context_approach.as_ref());
    push_local_runtime_tools(&mut stack);
    if let Some(key) = options.tavily_api_key {
        push_web_tools(&mut stack, key);
    }
    stack
}

pub fn locked_down_rlm_plugin_stack() -> PluginStack {
    tool_output_budget_stack()
}

fn push_core_runtime_tools(stack: &mut PluginStack) {
    stack.push(Arc::new(ToolDiscoveryPluginFactory::new()));
    stack.push(Arc::new(ToolOutputBudgetPluginFactory::default()));
}

fn push_standard_context_tools(
    stack: &mut PluginStack,
    standard_context_approach: Option<&StandardContextApproach>,
) {
    match standard_context_approach.map(StandardContextApproach::kind) {
        Some(lash_core::StandardContextApproachKind::RollingHistory) => {
            stack.push(Arc::new(RollingHistoryPluginFactory::default()));
        }
        Some(lash_core::StandardContextApproachKind::ObservationalMemory) => {
            stack.push(Arc::new(ObservationalMemoryPluginFactory));
        }
        None => {}
    }
}

fn push_local_runtime_tools(stack: &mut PluginStack) {
    stack.push(Arc::new(ProcessControlsPluginFactory::new()));
    stack.push(Arc::new(StandardShellPluginFactory::new()));
    stack.push(Arc::new(StaticPluginFactory::new(
        "apply_patch",
        PluginSpec::new().with_tool_provider(Arc::new(ApplyPatchTool) as Arc<dyn ToolProvider>),
    )));
    stack.push(Arc::new(ReadFilePluginFactory::new()));
    stack.push(Arc::new(StaticPluginFactory::new(
        "glob",
        PluginSpec::new().with_tool_provider(Arc::new(Glob) as Arc<dyn ToolProvider>),
    )));
    stack.push(Arc::new(StaticPluginFactory::new(
        "ls",
        PluginSpec::new().with_tool_provider(Arc::new(Ls) as Arc<dyn ToolProvider>),
    )));
    stack.push(Arc::new(StaticPluginFactory::new(
        "grep",
        PluginSpec::new().with_tool_provider(Arc::new(Grep::new()) as Arc<dyn ToolProvider>),
    )));
}

fn push_web_tools(stack: &mut PluginStack, tavily_api_key: String) {
    let search_key = tavily_api_key.clone();
    stack.push(Arc::new(StaticPluginFactory::new(
        "search_web",
        PluginSpec::new()
            .with_tool_provider(Arc::new(WebSearch::new(search_key)) as Arc<dyn ToolProvider>),
    )));
    stack.push(Arc::new(StaticPluginFactory::new(
        "fetch_url",
        PluginSpec::new()
            .with_tool_provider(Arc::new(FetchUrl::new(tavily_api_key)) as Arc<dyn ToolProvider>),
    )));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stack_ids(stack: &PluginStack) -> Vec<&'static str> {
        stack
            .factories()
            .iter()
            .map(|factory| factory.id())
            .collect()
    }

    #[test]
    fn rolling_history_context_installs_rolling_history_only() {
        let stack = standard_tool_stack(StandardToolStackOptions {
            standard_context_approach: Some(StandardContextApproach::RollingHistory(
                Default::default(),
            )),
            tavily_api_key: None,
        });
        let ids = stack_ids(&stack);

        assert!(ids.contains(&"rolling_history"));
        assert!(!ids.contains(&"observational_memory"));
    }

    #[test]
    fn observational_memory_context_installs_om_support() {
        let stack = standard_tool_stack(StandardToolStackOptions {
            standard_context_approach: Some(StandardContextApproach::ObservationalMemory(
                Default::default(),
            )),
            tavily_api_key: None,
        });
        let host = lash_core::PluginHost::new(stack.into_factories());

        assert!(host.supports_standard_context_approach(
            &lash_core::StandardContextApproach::ObservationalMemory(
                lash_core::ObservationalMemoryConfig::default(),
            )
        ));
    }

    #[test]
    fn web_tools_are_explicitly_keyed() {
        let without_web = stack_ids(&standard_tool_stack(StandardToolStackOptions::default()));
        let with_web = stack_ids(&standard_tool_stack(StandardToolStackOptions {
            tavily_api_key: Some("key".to_string()),
            ..Default::default()
        }));

        assert!(!without_web.contains(&"search_web"));
        assert!(with_web.contains(&"search_web"));
        assert!(with_web.contains(&"fetch_url"));
    }
}
