use std::sync::Arc;

use crate::plugin::{
    PluginError, PluginFactory, PluginRegistrar, PluginSessionContext, PluginSnapshotMeta,
    PromptContribution, SnapshotReader, SnapshotWriter,
};
use crate::tools::UpdatePlanTool;
use crate::{SessionPlugin, ToolProvider};

fn plan_tracker_prompt_contributions() -> Vec<PromptContribution> {
    vec![PromptContribution::guidance(
        "### `update_plan`\nUse `update_plan` to keep an up-to-date, step-by-step plan for substantial multi-step work. Plans help show that you understand the task, make your approach visible to the user, and keep complex or ambiguous work organized.\n\nDo not use plans for simple, single-step requests you can just do immediately. Do not pad plans with filler, obvious steps, or actions you cannot actually perform.\n\nUse a plan when:\n- the task is non-trivial and will take multiple actions\n- sequencing or dependencies matter\n- the work has ambiguity that benefits from checkpoints\n- the user asked for a plan or TODOs\n- you discover additional steps you intend to complete before yielding\n\nPlan quality rules:\n- use short, meaningful, logically ordered, easy-to-verify steps\n- keep each step concise; prefer roughly 5-7 words when possible\n- avoid vague steps like \"work on feature\" or \"make it better\"\n- only include work you genuinely expect to do\n\nStatus rules:\n- keep exactly one step `in_progress` at a time\n- mark steps complete as soon as they are done\n- update the plan before continuing when scope changes materially\n- do not let the plan go stale while you work\n- finish with all steps marked `completed`\n\nAfter calling `update_plan`, do not repeat the full plan in prose. Briefly summarize what changed and what comes next.",
    )]
}

pub struct PlanTrackerPluginFactory;

impl PluginFactory for PlanTrackerPluginFactory {
    fn id(&self) -> &'static str {
        "plan_tracker"
    }

    fn build(&self, _ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(PlanTrackerPlugin {
            tools: Arc::new(UpdatePlanTool::default()),
        }))
    }
}

struct PlanTrackerPlugin {
    tools: Arc<UpdatePlanTool>,
}

impl SessionPlugin for PlanTrackerPlugin {
    fn id(&self) -> &'static str {
        "plan_tracker"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        reg.tools()
            .provider(Arc::clone(&self.tools) as Arc<dyn ToolProvider>)?;
        reg.prompt().contribute(Arc::new(|_ctx| {
            Box::pin(async move { Ok(plan_tracker_prompt_contributions()) })
        }));
        Ok(())
    }

    fn snapshot(
        &self,
        _writer: &mut dyn SnapshotWriter,
    ) -> Result<PluginSnapshotMeta, PluginError> {
        let snapshot = self
            .tools
            .snapshot()
            .map_err(|err| PluginError::Snapshot(err.to_string()))?;
        Ok(PluginSnapshotMeta {
            plugin_id: self.id().to_string(),
            plugin_version: self.version().to_string(),
            state: Some(
                serde_json::to_value(snapshot)
                    .map_err(|err| PluginError::Snapshot(err.to_string()))?,
            ),
        })
    }

    fn restore(
        &self,
        meta: &PluginSnapshotMeta,
        _reader: &dyn SnapshotReader,
    ) -> Result<(), PluginError> {
        let snapshot = meta
            .state
            .clone()
            .map(serde_json::from_value)
            .transpose()
            .map_err(|err| PluginError::Snapshot(err.to_string()))?
            .unwrap_or_default();
        self.tools
            .restore(snapshot)
            .map_err(PluginError::Snapshot)?;
        Ok(())
    }
}
