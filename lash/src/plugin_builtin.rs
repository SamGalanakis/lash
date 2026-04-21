#[path = "plugin_builtin/plan_mode.rs"]
mod plan_mode;
#[path = "plugin_builtin/prompt_context.rs"]
mod prompt_context;
#[path = "plugin_builtin/ui_activity.rs"]
mod ui_activity;
#[path = "plugin_builtin/update_plan.rs"]
mod update_plan;

pub use plan_mode::PlanModePluginFactory as BuiltinPlanModePluginFactory;
pub use prompt_context::PromptContextPluginConfig;
pub use prompt_context::PromptContextPluginFactory as BuiltinPromptContextPluginFactory;
pub use ui_activity::UiActivityPluginFactory as BuiltinUiActivityPluginFactory;
pub use update_plan::{
    PlanItem as UpdatePlanItem, PlanSnapshot as UpdatePlanSnapshot,
    UpdatePlanPluginFactory as BuiltinUpdatePlanPluginFactory,
};

#[cfg(test)]
#[path = "plugin_builtin/tests.rs"]
mod tests;
