use std::sync::Arc;
use std::time::Instant;

use tokio::sync::mpsc;

use crate::plugin::{
    ExecutionSurface, PluginDirective, PluginSession, SessionManager, ToolCallHookContext,
    ToolResultHookContext, emit_plugin_surface_events,
};
use crate::{
    ProgressSender, SessionEvent, ToolCallRecord, ToolExecutionContext, ToolImage, ToolProvider,
    ToolResult, TurnInjectionBridge,
};

#[derive(Clone)]
pub(crate) struct ToolDispatchContext {
    pub plugins: Arc<PluginSession>,
    pub tools: Arc<dyn ToolProvider>,
    pub surface: ExecutionSurface,
    pub host: Arc<dyn SessionManager>,
    pub session_id: String,
    pub event_tx: mpsc::Sender<SessionEvent>,
    pub turn_injection_bridge: TurnInjectionBridge,
}

#[derive(Clone)]
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
    let tool_context = ToolExecutionContext {
        session_id: context.session_id.clone(),
        host: Arc::clone(&context.host),
        cancellation_token: None,
        async_task_id: None,
    };
    dispatch_tool_call_with_execution_context(context, tool_name, args, progress, tool_context)
        .await
}

pub(crate) async fn dispatch_tool_call_with_execution_context(
    context: &ToolDispatchContext,
    tool_name: String,
    args: serde_json::Value,
    progress: Option<&ProgressSender>,
    tool_context: ToolExecutionContext,
) -> ToolDispatchOutcome {
    let enabled_tools = context.surface.enabled_tools();
    if !enabled_tools.iter().any(|tool| tool.name == tool_name) {
        return outcome(
            tool_name,
            args,
            ToolResult::err_fmt("Tool is not enabled for this session"),
            0,
        );
    }
    let mut args = args;

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
    let result = if let Some(result) = context
        .plugins
        .mode_native_tools()
        .execute(context, &tool_name, &args, progress)
        .await
    {
        result
    } else {
        context
            .tools
            .execute_streaming_with_context(&tool_name, &args, &tool_context, progress)
            .await
    };
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
                    PluginDirective::EnqueueMessages { messages } => {
                        if let Err(err) = context.turn_injection_bridge.enqueue(messages) {
                            final_result = ToolResult::err_fmt(err);
                            break;
                        }
                    }
                    PluginDirective::ReplaceToolArgs { .. } => {
                        final_result = ToolResult::err_fmt(
                            "after_tool_call only supports abort, short-circuit, session creation, events, and message injection",
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
        call_id: None,
        tool: tool_name,
        args,
        result: result.result,
        success: result.success,
        duration_ms,
    };
    ToolDispatchOutcome { record, images }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::{PluginHost, StaticPluginFactory};
    use crate::{ExecutionMode, ToolProvider};
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::Barrier;
    use tokio::time::{Duration, timeout};

    struct MockTools;

    #[async_trait::async_trait]
    impl ToolProvider for MockTools {
        fn definitions(&self) -> Vec<crate::ToolDefinition> {
            vec![
                crate::ToolDefinition {
                    name: "alpha".into(),
                    description: String::new(),
                    params: vec![],
                    returns: "str".into(),
                    examples: vec![],
                    enabled: true,
                    injected: true,
                    input_schema_override: None,
                    output_schema_override: None,
                },
                crate::ToolDefinition {
                    name: "beta".into(),
                    description: String::new(),
                    params: vec![crate::ToolParam::typed("value", "str")],
                    returns: "str".into(),
                    examples: vec![],
                    enabled: true,
                    injected: true,
                    input_schema_override: None,
                    output_schema_override: None,
                },
            ]
        }

        async fn execute(&self, name: &str, args: &serde_json::Value) -> ToolResult {
            match name {
                "alpha" => ToolResult::ok(json!("alpha")),
                "beta" => {
                    if args.get("value").and_then(|value| value.as_str()) == Some("fail") {
                        ToolResult::err_fmt("beta failed")
                    } else {
                        ToolResult::ok(json!(args.get("value").cloned().unwrap_or(json!(null))))
                    }
                }
                _ => ToolResult::err_fmt(format!("Unknown tool: {name}")),
            }
        }
    }

    struct ParallelProbeTools {
        barrier: Arc<Barrier>,
        started: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl ToolProvider for ParallelProbeTools {
        fn definitions(&self) -> Vec<crate::ToolDefinition> {
            vec![
                crate::ToolDefinition {
                    name: "probe_a".into(),
                    description: String::new(),
                    params: vec![],
                    returns: "str".into(),
                    examples: vec![],
                    enabled: true,
                    injected: true,
                    input_schema_override: None,
                    output_schema_override: None,
                },
                crate::ToolDefinition {
                    name: "probe_b".into(),
                    description: String::new(),
                    params: vec![],
                    returns: "str".into(),
                    examples: vec![],
                    enabled: true,
                    injected: true,
                    input_schema_override: None,
                    output_schema_override: None,
                },
            ]
        }

        async fn execute(&self, name: &str, _args: &serde_json::Value) -> ToolResult {
            self.started.fetch_add(1, Ordering::SeqCst);
            let waited = timeout(Duration::from_millis(100), self.barrier.wait()).await;
            match waited {
                Ok(_) => ToolResult::ok(json!(name)),
                Err(_) => ToolResult::err_fmt(format!("{name} did not overlap with peer")),
            }
        }
    }

    fn test_plugins(provider: Arc<dyn ToolProvider>) -> Arc<PluginSession> {
        PluginHost::new(vec![Arc::new(StaticPluginFactory::new(
            "test_tools",
            crate::PluginSpec::new().with_tool_provider(Arc::clone(&provider)),
        ))])
        .build_standard_session("root", None)
        .expect("plugin session")
    }

    use crate::test_support::MockSessionManager;

    fn dispatch_context() -> ToolDispatchContext {
        let (event_tx, _event_rx) = mpsc::channel(8);
        let plugins = test_plugins(Arc::new(MockTools));
        let tools = plugins.tools();
        let surface = plugins.execution_surface("session", ExecutionMode::Standard);
        ToolDispatchContext {
            plugins,
            tools,
            surface,
            host: Arc::new(MockSessionManager::default()),
            session_id: "session".to_string(),
            event_tx,
            turn_injection_bridge: crate::TurnInjectionBridge::new(),
        }
    }

    fn parallel_dispatch_context(
        barrier: Arc<Barrier>,
        started: Arc<AtomicUsize>,
    ) -> ToolDispatchContext {
        let (event_tx, _event_rx) = mpsc::channel(8);
        let plugins = test_plugins(Arc::new(ParallelProbeTools { barrier, started }));
        let tools = plugins.tools();
        let surface = plugins.execution_surface("session", ExecutionMode::Standard);
        ToolDispatchContext {
            plugins,
            tools,
            surface,
            host: Arc::new(MockSessionManager::default()),
            session_id: "session".to_string(),
            event_tx,
            turn_injection_bridge: crate::TurnInjectionBridge::new(),
        }
    }

    #[tokio::test]
    async fn batch_executes_nested_calls_and_preserves_partial_failures() {
        let outcome = dispatch_tool_call(
            &dispatch_context(),
            "batch".to_string(),
            json!({
                "tool_calls": [
                    {"tool": "alpha", "parameters": {}},
                    {"tool": "beta", "parameters": {"value": "ok"}},
                    {"tool": "beta", "parameters": {"value": "fail"}}
                ]
            }),
            None,
        )
        .await;

        assert!(outcome.record.success);
        assert_eq!(outcome.record.tool, "batch");
        assert_eq!(outcome.record.result.get("successful"), Some(&json!(2)));
        assert_eq!(outcome.record.result.get("failed"), Some(&json!(1)));
        let results = outcome
            .record
            .result
            .get("results")
            .and_then(|value| value.as_array())
            .expect("results");
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].get("tool"), Some(&json!("alpha")));
        assert_eq!(results[2].get("error"), Some(&json!("beta failed")));
    }

    #[tokio::test]
    async fn batch_rejects_nested_batch_as_partial_failure() {
        let outcome = dispatch_tool_call(
            &dispatch_context(),
            "batch".to_string(),
            json!({
                "tool_calls": [
                    {"tool": "batch", "parameters": {"tool_calls": []}}
                ]
            }),
            None,
        )
        .await;

        assert!(outcome.record.success);
        assert_eq!(outcome.record.result.get("failed"), Some(&json!(1)));
        let first = outcome
            .record
            .result
            .get("results")
            .and_then(|value| value.as_array())
            .and_then(|items| items.first())
            .expect("first result");
        assert_eq!(
            first.get("error"),
            Some(&json!("Tool 'batch' is not allowed inside batch"))
        );
    }

    #[tokio::test]
    async fn batch_marks_overflow_calls_as_failures() {
        let tool_calls = (0..26)
            .map(|_| json!({"tool": "alpha", "parameters": {}}))
            .collect::<Vec<_>>();

        let outcome = dispatch_tool_call(
            &dispatch_context(),
            "batch".to_string(),
            json!({ "tool_calls": tool_calls }),
            None,
        )
        .await;

        assert!(outcome.record.success);
        assert_eq!(outcome.record.result.get("successful"), Some(&json!(25)));
        assert_eq!(outcome.record.result.get("failed"), Some(&json!(1)));
        let results = outcome
            .record
            .result
            .get("results")
            .and_then(|value| value.as_array())
            .expect("results");
        assert_eq!(results.len(), 26);
        assert_eq!(
            results[25].get("error"),
            Some(&json!("Maximum of 25 tool calls allowed in batch"))
        );
    }

    #[tokio::test]
    async fn batch_calls_make_progress_concurrently() {
        let barrier = Arc::new(Barrier::new(2));
        let started = Arc::new(AtomicUsize::new(0));
        let outcome = dispatch_tool_call(
            &parallel_dispatch_context(Arc::clone(&barrier), Arc::clone(&started)),
            "batch".to_string(),
            json!({
                "tool_calls": [
                    {"tool": "probe_a", "parameters": {}},
                    {"tool": "probe_b", "parameters": {}}
                ]
            }),
            None,
        )
        .await;

        assert!(outcome.record.success);
        assert_eq!(started.load(Ordering::SeqCst), 2);
        assert_eq!(outcome.record.result.get("successful"), Some(&json!(2)));
        assert_eq!(outcome.record.result.get("failed"), Some(&json!(0)));
    }
}
