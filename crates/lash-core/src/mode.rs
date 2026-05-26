use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct ModeBuildInput {
    pub mode: crate::ExecutionMode,
    pub tool_surface: Arc<crate::ToolSurface>,
    pub lashlang_surface: lashlang::LashlangSurface,
    pub extra_prompt_contributions: Vec<crate::PromptContribution>,
}
