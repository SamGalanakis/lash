use std::sync::Arc;

use super::execution_context::ModeExecutionContext;
use crate::tool_dispatch::{dispatch_tool_call_with_execution_context, schedule_tool_batch};
use crate::{
    ModelToolReturn, SessionEvent, ToolCallOutput, ToolCallRecord, ToolCancellation, ToolContext,
    ToolFailure, ToolFailureClass, TurnActivityId, TurnEvent,
};

#[derive(Clone, Debug)]
pub struct ModeToolBatchItem {
    pub id: String,
    pub name: String,
    pub args: serde_json::Value,
}

#[derive(Clone, Debug)]
pub struct ModeToolReply {
    pub output: ToolCallOutput,
    pub record: Option<ToolCallRecord>,
}

impl ModeToolReply {
    pub fn success(value: serde_json::Value) -> Self {
        Self {
            output: ToolCallOutput::success(value),
            record: None,
        }
    }

    pub fn error(value: serde_json::Value) -> Self {
        let message = value
            .as_str()
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| value.to_string());
        let mut failure = ToolFailure::tool(ToolFailureClass::Execution, "tool_error", message);
        failure.raw =
            Some(serde_json::from_value(value).unwrap_or_else(|_| {
                crate::ToolValue::String("unserializable tool error".to_string())
            }));
        Self {
            output: ToolCallOutput::failure(failure),
            record: None,
        }
    }

    pub fn from_output(output: ToolCallOutput) -> Self {
        Self {
            output,
            record: None,
        }
    }

    pub fn cancelled(message: impl Into<String>) -> Self {
        Self::from_output(ToolCallOutput::cancelled(ToolCancellation::runtime(
            message,
        )))
    }

    pub(crate) fn with_record(mut self, record: ToolCallRecord) -> Self {
        self.record = Some(record);
        self
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CompletedModeToolCall {
    pub index: usize,
    pub completed: crate::sansio::CompletedToolCall,
    pub record: ToolCallRecord,
}

impl ModeExecutionContext {
    pub(crate) async fn execute_tool_call(
        &self,
        call_id: String,
        name: String,
        args: serde_json::Value,
        index: usize,
        replay: Option<crate::llm::types::ProviderReplayMeta>,
    ) -> CompletedModeToolCall {
        let _ = self
            .dispatch
            .event_tx
            .send(SessionEvent::ToolCallStart {
                call_id: Some(call_id.clone()),
                name: name.clone(),
                args: args.clone(),
            })
            .await;
        let tool_correlation_id = TurnActivityId::new(format!("tool:{call_id}"));
        self.emit_turn_activity(
            tool_correlation_id.clone(),
            TurnEvent::ToolCallStarted {
                call_id: Some(call_id.clone()),
                name: name.clone(),
                args: args.clone(),
            },
        )
        .await;

        let (progress_tx, mut progress_rx) =
            tokio::sync::mpsc::unbounded_channel::<crate::SandboxMessage>();
        let event_tx = self.dispatch.event_tx.clone();
        let progress_handle = tokio::spawn(async move {
            while let Some(sandbox_msg) = progress_rx.recv().await {
                if sandbox_msg.kind != "lashlang_code" {
                    let _ = event_tx
                        .send(SessionEvent::Message {
                            text: sandbox_msg.text,
                            kind: sandbox_msg.kind,
                        })
                        .await;
                }
            }
        });

        let mut tool_context = ToolContext::new(
            self.dispatch.session_id.clone(),
            Arc::clone(&self.dispatch.host),
            self.dispatch.turn_context.clone(),
            Arc::clone(&self.dispatch.attachment_store),
            Some(call_id.clone()),
        );
        tool_context.cancellation_token = self.cancellation_token.clone();
        let mut outcome = dispatch_tool_call_with_execution_context(
            &self.dispatch,
            name,
            args,
            Some(&progress_tx),
            tool_context,
        )
        .await;
        outcome.record.call_id = Some(call_id.clone());
        drop(progress_tx);
        let _ = progress_handle.await;

        let output = outcome.record.output.clone();
        let model_return = match self
            .dispatch
            .plugins
            .project_tool_result(crate::plugin::ToolResultProjectionContext {
                session_id: self.dispatch.session_id.clone(),
                tool_name: outcome.record.tool.clone(),
                args: outcome.record.args.clone(),
                output: output.clone(),
                duration_ms: outcome.record.duration_ms,
                call_id: call_id.clone(),
            })
            .await
        {
            Ok(projected) => projected,
            Err(err) => ModelToolReturn::text(
                call_id.clone(),
                outcome.record.tool.clone(),
                err.to_string(),
            ),
        };

        self.emit_turn_activity(
            tool_correlation_id,
            TurnEvent::ToolCallCompleted {
                call_id: Some(call_id.clone()),
                name: outcome.record.tool.clone(),
                args: outcome.record.args.clone(),
                output: output.clone(),
                duration_ms: outcome.record.duration_ms,
            },
        )
        .await;

        let record = ToolCallRecord {
            call_id: Some(call_id.clone()),
            tool: outcome.record.tool.clone(),
            args: outcome.record.args.clone(),
            output: output.clone(),
            duration_ms: outcome.record.duration_ms,
        };
        CompletedModeToolCall {
            index,
            completed: crate::sansio::CompletedToolCall {
                call_id,
                tool_name: outcome.record.tool,
                args: outcome.record.args,
                output,
                model_return,
                duration_ms: outcome.record.duration_ms,
                replay,
            },
            record,
        }
    }

    pub async fn call_tool(
        &self,
        call_id: String,
        name: String,
        args: serde_json::Value,
        index: usize,
    ) -> ModeToolReply {
        if name == "list_async_handles" {
            let live_monitor_tasks = self.live_monitor_tasks().await;
            return self.list_async_handles(live_monitor_tasks);
        }
        if name == "monitor" {
            return self.start_monitor_handle_call(call_id, args, index).await;
        }
        let executed = self
            .execute_tool_call(call_id, name, args, index, None)
            .await;
        let reply = ModeToolReply::from_output(executed.completed.output);
        reply.with_record(executed.record)
    }

    pub async fn call_tool_batch(&self, calls: Vec<ModeToolBatchItem>) -> Vec<ModeToolReply> {
        let indexed_calls = calls.into_iter().enumerate().collect::<Vec<_>>();
        schedule_tool_batch(
            indexed_calls,
            |(index, _)| *index,
            |(_, call)| self.tool_execution_mode(&call.name),
            |(index, call)| {
                let ctx = self.clone();
                async move { ctx.call_tool(call.id, call.name, call.args, index).await }
            },
        )
        .await
    }

    pub async fn start_tool_call(
        &self,
        call_id: String,
        name: String,
        args: serde_json::Value,
    ) -> ModeToolReply {
        if name == "monitor" {
            return self.start_monitor_handle_call(call_id, args, 0).await;
        }
        self.start_async_tool_call(call_id, name, args).await
    }

    pub async fn await_tool_handle(
        &self,
        _call_id: String,
        handle: serde_json::Value,
    ) -> ModeToolReply {
        self.await_async_tool_handle(handle).await
    }

    pub async fn cancel_tool_handle(
        &self,
        _call_id: String,
        handle: serde_json::Value,
    ) -> ModeToolReply {
        self.cancel_async_tool_handle(handle).await
    }
}
