use std::sync::Arc;

use crate::{ToolInvocation, ToolInvocationReply};

use super::ToolContext;

#[derive(Clone)]
pub struct ToolDispatchClient<'run> {
    pub(super) context: ToolContext<'run>,
}

impl<'run> ToolDispatchClient<'run> {
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
                    crate::tool_dispatch::resolve_tool_scheduling(&dispatch, &call.name)
                }
            },
            |(_, call)| {
                let dispatch = Arc::clone(&dispatch);
                let mut child_context = self.context.clone();
                async move {
                    child_context.tool_call_id = Some(call.id.clone());
                    child_context.prepared_payload = serde_json::Value::Null;
                    let outcome = crate::tool_dispatch::dispatch_tool_call_with_execution_context(
                        dispatch.as_ref(),
                        call.name,
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
