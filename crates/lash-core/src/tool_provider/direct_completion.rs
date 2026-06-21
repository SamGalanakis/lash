use crate::plugin::{DirectCompletion, PluginError};

#[derive(Clone)]
pub struct ToolDirectCompletionClient<'run> {
    pub(super) session_id: String,
    pub(super) tool_call_id: Option<String>,
    pub(super) direct_completions: crate::DirectCompletionClient<'run>,
    pub(super) parent_invocation: Option<crate::RuntimeInvocation>,
    pub(super) parent_tool_attempt_is_durable: bool,
}

impl ToolDirectCompletionClient<'_> {
    pub async fn complete(
        &self,
        mut request: crate::DirectRequest,
        usage_source: &str,
    ) -> Result<DirectCompletion, PluginError> {
        if self.parent_invocation.as_ref().is_some_and(|invocation| {
            invocation.effect_kind() == Some(crate::RuntimeEffectKind::ToolAttempt)
        }) && self.parent_tool_attempt_is_durable
        {
            return Err(PluginError::Session(
                "direct completions are not available inside a journaled tool attempt".to_string(),
            ));
        }
        if request.session_id.is_none() {
            request.session_id = Some(self.session_id.clone());
        }
        if request.caused_by.is_none()
            && let Some(call_id) = self.tool_call_id.clone()
        {
            request.caused_by = Some(crate::CausalRef::ToolCall {
                session_id: self.session_id.clone(),
                call_id,
            });
        }
        self.direct_completions
            .direct_completion(request, usage_source)
            .await
    }
}
