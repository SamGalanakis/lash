use std::sync::Arc;

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
        let Some(dispatch) = self.context.runtime_dispatch.clone() else {
            return calls
                .into_iter()
                .map(|_| {
                    ToolInvocationReply::error(serde_json::json!(
                        "tool batch dispatch is unavailable outside runtime tool execution"
                    ))
                })
                .collect();
        };
        let indexed_calls = calls.into_iter().enumerate().collect::<Vec<_>>();
        crate::tool_dispatch::schedule_tool_batch(
            indexed_calls,
            |(index, _)| *index,
            {
                let dispatch = Arc::clone(&dispatch);
                move |(_, call)| {
                    crate::tool_dispatch::resolve_tool_scheduling_by_id(&dispatch, &call.tool_id)
                }
            },
            |(_, call)| {
                let dispatch = Arc::clone(&dispatch);
                let mut child_context = self.context.clone();
                async move {
                    child_context.tool_call_id = Some(call.id.clone());
                    child_context.prepared_payload = serde_json::Value::Null;
                    child_context.child_execution_trace_hook = call.child_execution_trace_hook;
                    let Some(manifest) = crate::tool_dispatch::resolve_callable_manifest_by_id(
                        dispatch.as_ref(),
                        &call.tool_id,
                    ) else {
                        return ToolInvocationReply::error(serde_json::json!(format!(
                            "Tool id `{}` is unavailable in this session",
                            call.tool_id
                        )));
                    };
                    let outcome = crate::tool_dispatch::dispatch_tool_call_with_execution_context(
                        dispatch.as_ref(),
                        manifest.name,
                        call.args,
                        None,
                        child_context,
                    )
                    .await;
                    let mut record = outcome.record;
                    record.call_id = Some(call.id);
                    ToolInvocationReply::from_output(record.output.clone()).with_record(record)
                }
            },
        )
        .await
    }
}
