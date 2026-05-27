use crate::plugin::{DirectCompletion, PluginError};

#[derive(Clone)]
pub struct ToolDirectCompletionControl<'run> {
    pub(super) session_id: String,
    pub(super) tool_call_id: Option<String>,
    pub(super) direct_completions: crate::DirectCompletionClient<'run>,
}

impl ToolDirectCompletionControl<'_> {
    pub async fn complete(
        &self,
        mut request: crate::DirectRequest,
        usage_source: &str,
    ) -> Result<DirectCompletion, PluginError> {
        if request.session_id.is_none() {
            request.session_id = Some(self.session_id.clone());
        }
        if request.originating_tool_call_id.is_none() {
            request.originating_tool_call_id = self.tool_call_id.clone();
        }
        self.direct_completions
            .direct_completion(request, usage_source)
            .await
    }
}
