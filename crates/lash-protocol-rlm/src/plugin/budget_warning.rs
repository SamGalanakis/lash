use lash_core::PreparedContext;
use lash_core::plugin::{HistoryError, TurnContextTransform, TurnTransformContext};

use crate::driver::SharedPromptUsage;

pub(super) const BUDGET_WARNING_STATUS: &str = "rlm_context_budget_warning";

pub(super) struct BudgetUsageObserver {
    pub(super) cell: SharedPromptUsage,
}

#[async_trait::async_trait]
impl TurnContextTransform for BudgetUsageObserver {
    fn id(&self) -> &'static str {
        "rlm.budget_usage_observer"
    }

    async fn transform(
        &self,
        ctx: &TurnTransformContext,
        input: PreparedContext,
    ) -> Result<PreparedContext, HistoryError> {
        if let Ok(mut guard) = self.cell.write() {
            *guard = ctx.prompt_usage.clone();
        }
        Ok(input)
    }
}
