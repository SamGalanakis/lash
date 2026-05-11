use std::sync::Arc;

use futures_util::stream::{FuturesUnordered, StreamExt};

use super::execution_context::ModeExecutionContext;
use crate::tool_dispatch::dispatch_tool_call_with_execution_context;
use crate::{SessionEvent, ToolCallRecord, ToolContext, ToolImage, TurnActivityId, TurnEvent};

#[derive(Clone, Debug)]
pub struct ModeToolBatchItem {
    pub id: String,
    pub name: String,
    pub args: serde_json::Value,
}

#[derive(Clone, Debug)]
pub struct ModeToolReply {
    pub success: bool,
    pub value: serde_json::Value,
    pub images: Vec<ToolImage>,
    pub record: Option<ToolCallRecord>,
}

impl ModeToolReply {
    pub fn success(value: serde_json::Value) -> Self {
        Self {
            success: true,
            value,
            images: Vec::new(),
            record: None,
        }
    }

    pub fn success_with_images(value: serde_json::Value, images: Vec<ToolImage>) -> Self {
        Self {
            success: true,
            value,
            images,
            record: None,
        }
    }

    pub fn error(value: serde_json::Value) -> Self {
        Self {
            success: false,
            value,
            images: Vec::new(),
            record: None,
        }
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
                if sandbox_msg.kind != "final" && sandbox_msg.kind != "lashlang_code" {
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

        let result = crate::ToolResult {
            success: outcome.record.success,
            result: outcome.record.result.clone(),
            images: outcome.images.clone(),
            control: outcome.record.control.clone(),
        };
        let model_result = match self
            .dispatch
            .plugins
            .project_tool_result(crate::plugin::ToolResultProjectionContext {
                session_id: self.dispatch.session_id.clone(),
                tool_name: outcome.record.tool.clone(),
                args: outcome.record.args.clone(),
                result: result.clone(),
                duration_ms: outcome.record.duration_ms,
            })
            .await
        {
            Ok(projected) => projected,
            Err(err) => crate::ToolResult::err_fmt(err.to_string()),
        };

        self.emit_turn_activity(
            tool_correlation_id,
            TurnEvent::ToolCallCompleted {
                call_id: Some(call_id.clone()),
                name: outcome.record.tool.clone(),
                args: outcome.record.args.clone(),
                result: result.result.clone(),
                success: result.success,
                duration_ms: outcome.record.duration_ms,
            },
        )
        .await;

        let record = ToolCallRecord {
            call_id: Some(call_id.clone()),
            tool: outcome.record.tool.clone(),
            args: outcome.record.args.clone(),
            result: result.result.clone(),
            success: result.success,
            duration_ms: outcome.record.duration_ms,
            control: result.control.clone(),
        };
        CompletedModeToolCall {
            index,
            completed: crate::sansio::CompletedToolCall {
                call_id,
                tool_name: outcome.record.tool,
                args: outcome.record.args,
                result,
                model_result,
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
        let reply = if executed.completed.result.success {
            ModeToolReply::success_with_images(
                executed.completed.result.result.clone(),
                executed.completed.result.images.clone(),
            )
        } else {
            ModeToolReply::error(executed.completed.result.result.clone())
        };
        reply.with_record(executed.record)
    }

    pub async fn call_tool_batch(&self, calls: Vec<ModeToolBatchItem>) -> Vec<ModeToolReply> {
        let mut pending = FuturesUnordered::new();
        for (offset, call) in calls.into_iter().enumerate() {
            let ctx = self.clone();
            pending.push(async move {
                let reply = ctx.call_tool(call.id, call.name, call.args, offset).await;
                (offset, reply)
            });
        }
        let mut replies = Vec::new();
        while let Some(reply) = pending.next().await {
            replies.push(reply);
        }
        replies.sort_by_key(|(offset, _)| *offset);
        replies.into_iter().map(|(_, reply)| reply).collect()
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
