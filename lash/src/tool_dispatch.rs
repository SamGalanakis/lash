use std::sync::Arc;
use std::time::Instant;

use tokio::sync::mpsc;

use crate::agent::resolve_tool_surface;
use crate::plugin::{
    PluginDirective, PluginSession, SessionManager, ToolCallHookContext, ToolResultHookContext,
    emit_plugin_surface_events,
};
use crate::tools::preflight_tool_args;
use crate::{
    AgentEvent, ExecutionMode, ProgressSender, ToolCallRecord, ToolImage, ToolProvider, ToolResult,
};

pub(crate) struct ToolDispatchContext {
    pub tool_provider: Arc<dyn ToolProvider>,
    pub plugins: Arc<PluginSession>,
    pub host: Arc<dyn SessionManager>,
    pub session_id: String,
    pub execution_mode: ExecutionMode,
    pub event_tx: mpsc::Sender<AgentEvent>,
}

pub(crate) struct ToolDispatchOutcome {
    pub record: ToolCallRecord,
    pub images: Vec<ToolImage>,
}

pub(crate) async fn dispatch_tool_call(
    context: &ToolDispatchContext,
    tool_name: String,
    args: serde_json::Value,
    progress: Option<&ProgressSender>,
) -> ToolDispatchOutcome {
    let surface = resolve_tool_surface(
        context.plugins.as_ref(),
        &context.session_id,
        context.execution_mode,
        context.tool_provider.definitions(),
    );
    let mut args = preflight_tool_args(&tool_name, args, surface.tools, context.execution_mode);

    let directives = match context
        .plugins
        .before_tool_call(ToolCallHookContext {
            session_id: context.session_id.clone(),
            tool_name: tool_name.clone(),
            args: args.clone(),
            host: Arc::clone(&context.host),
        })
        .await
    {
        Ok(directives) => directives,
        Err(err) => {
            return outcome(tool_name, args, ToolResult::err_fmt(err.to_string()), 0);
        }
    };

    let mut short_circuit: Option<ToolResult> = None;
    for emitted in directives {
        let plugin_id = emitted.plugin_id;
        let directive = emitted.value;
        match directive {
            PluginDirective::CreateSession { request } => {
                if let Err(err) = context.host.create_session(*request).await {
                    short_circuit = Some(ToolResult::err_fmt(err.to_string()));
                    break;
                }
            }
            PluginDirective::ReplaceToolArgs { args: replacement } => {
                args = replacement;
            }
            PluginDirective::ShortCircuitTool { result, success } => {
                short_circuit = Some(ToolResult {
                    success,
                    result,
                    images: Vec::new(),
                });
            }
            PluginDirective::AbortTurn { message, .. } => {
                short_circuit = Some(ToolResult::err_fmt(message));
            }
            PluginDirective::EmitEvents { events } => {
                emit_plugin_surface_events(&context.event_tx, &plugin_id, events).await;
            }
            PluginDirective::EnqueueMessages { .. } => {
                short_circuit = Some(ToolResult::err_fmt(
                    "before_tool_call does not support message injection",
                ));
            }
        }
    }
    if let Some(result) = short_circuit {
        return outcome(tool_name, args, result, 0);
    }

    let tool_start = Instant::now();
    let result = context
        .tool_provider
        .execute_streaming(&tool_name, &args, progress)
        .await;
    let duration_ms = tool_start.elapsed().as_millis() as u64;

    let result = match context
        .plugins
        .after_tool_call(ToolResultHookContext {
            session_id: context.session_id.clone(),
            tool_name: tool_name.clone(),
            args: args.clone(),
            result: result.clone(),
            duration_ms,
            host: Arc::clone(&context.host),
        })
        .await
    {
        Ok(directives) => {
            let mut final_result = result;
            for emitted in directives {
                let plugin_id = emitted.plugin_id;
                let directive = emitted.value;
                match directive {
                    PluginDirective::CreateSession { request } => {
                        if let Err(err) = context.host.create_session(*request).await {
                            final_result = ToolResult::err_fmt(err.to_string());
                            break;
                        }
                    }
                    PluginDirective::ShortCircuitTool { result, success } => {
                        final_result = ToolResult {
                            success,
                            result,
                            images: Vec::new(),
                        };
                    }
                    PluginDirective::AbortTurn { message, .. } => {
                        final_result = ToolResult::err_fmt(message);
                    }
                    PluginDirective::EmitEvents { events } => {
                        emit_plugin_surface_events(&context.event_tx, &plugin_id, events).await;
                    }
                    PluginDirective::ReplaceToolArgs { .. }
                    | PluginDirective::EnqueueMessages { .. } => {
                        final_result = ToolResult::err_fmt(
                            "after_tool_call only supports abort, short-circuit, and session creation",
                        );
                    }
                }
            }
            final_result
        }
        Err(err) => ToolResult::err_fmt(err.to_string()),
    };

    outcome(tool_name, args, result, duration_ms)
}

fn outcome(
    tool_name: String,
    args: serde_json::Value,
    mut result: ToolResult,
    duration_ms: u64,
) -> ToolDispatchOutcome {
    let images = std::mem::take(&mut result.images);
    let record = ToolCallRecord {
        tool: tool_name,
        args,
        result: result.result,
        success: result.success,
        duration_ms,
    };
    ToolDispatchOutcome { record, images }
}
