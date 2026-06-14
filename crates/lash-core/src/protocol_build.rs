use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct ProtocolBuildInput {
    pub tool_catalog: Arc<crate::ToolCatalog>,
    pub lashlang_host_environment: lashlang::LashlangHostEnvironment,
    pub extra_prompt_contributions: Vec<crate::PromptContribution>,
}
