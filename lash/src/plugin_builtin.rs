#[path = "plugin_builtin/history.rs"]
mod history;
#[path = "plugin_builtin/plan_mode.rs"]
mod plan_mode;
#[path = "plugin_builtin/plan_tracker.rs"]
mod plan_tracker;
#[path = "plugin_builtin/prompt_context.rs"]
mod prompt_context;

pub use history::HistoryPluginFactory as BuiltinHistoryPluginFactory;
pub use plan_mode::PlanModePluginFactory as BuiltinPlanModePluginFactory;
pub use plan_tracker::PlanTrackerPluginFactory as BuiltinPlanTrackerPluginFactory;
pub use prompt_context::PromptContextPluginConfig;
pub use prompt_context::PromptContextPluginFactory as BuiltinPromptContextPluginFactory;

#[cfg(test)]
#[path = "plugin_builtin/tests.rs"]
mod tests;
