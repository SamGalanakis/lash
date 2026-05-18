//! `plan_mode` and `update_plan` plugins.
//!
//! These ship as an optional first-party plugin crate rather than being
//! bundled into `lash` core. Embedders register them explicitly via
//! `plugin_factories.push(Arc::new(PlanModePluginFactory::new(...)))` etc.

mod plan_mode;
mod update_plan;

pub use plan_mode::{
    PlanModeDisableOp, PlanModeEnableOp, PlanModeExternalArgs, PlanModeExternalStatus,
    PlanModePluginConfig, PlanModePluginFactory, PlanModePrompt, PlanModePromptRequest,
    PlanModePromptResponse, PlanModePromptReview, PlanModeToggleOp,
};
pub use update_plan::{PlanItem, PlanSnapshot, UpdatePlanPluginFactory};
