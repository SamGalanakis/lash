use crate::{ToolInvocation, ToolInvocationReply, ToolManifest};

use super::ToolContext;

#[derive(Clone)]
pub struct ToolDispatchClient<'run> {
    pub(super) context: ToolContext<'run>,
}

impl<'run> ToolDispatchClient<'run> {
    pub fn callable_tool_manifest(&self, name: &str) -> Option<ToolManifest> {
        let dispatch = self.context.runtime_dispatch.as_ref()?;
        crate::tool_dispatch::resolve_callable_manifest(dispatch, name)
    }

    pub async fn batch(&self, calls: Vec<ToolInvocation>) -> Vec<ToolInvocationReply> {
        let Some(runtime) = self.context.runtime_execution_context.clone() else {
            return calls
                .into_iter()
                .map(|_| {
                    ToolInvocationReply::error(serde_json::json!(
                        "tool batch dispatch is unavailable outside runtime execution"
                    ))
                })
                .collect();
        };
        runtime.call_tool_batch(calls).await
    }
}
