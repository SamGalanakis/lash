use std::sync::Arc;
use std::time::Instant;

use futures_util::stream::{FuturesUnordered, StreamExt};
use tokio::sync::mpsc;

use crate::plugin::{
    PluginDirective, PluginSession, SessionManager, ToolCallHookContext, ToolResultHookContext,
    emit_plugin_surface_events,
};
use crate::tools::{NativeTool, find_native_tool};
use crate::{
    AgentEvent, ExecutionMode, ProgressSender, ToolCallRecord, ToolExecutionContext, ToolImage,
    ToolResult, TurnInjectionBridge,
};

#[derive(Clone)]
pub(crate) struct ToolDispatchContext {
    pub plugins: Arc<PluginSession>,
    pub host: Arc<dyn SessionManager>,
    pub session_id: String,
    pub execution_mode: ExecutionMode,
    pub event_tx: mpsc::Sender<AgentEvent>,
    pub turn_injection_bridge: TurnInjectionBridge,
}

pub(crate) struct ToolDispatchOutcome {
    pub record: ToolCallRecord,
    pub images: Vec<ToolImage>,
}

const BATCH_MAX_TOOL_CALLS: usize = 25;

pub(crate) async fn dispatch_tool_call(
    context: &ToolDispatchContext,
    tool_name: String,
    args: serde_json::Value,
    progress: Option<&ProgressSender>,
) -> ToolDispatchOutcome {
    let surface = context
        .plugins
        .execution_surface(&context.session_id, context.execution_mode);
    let enabled_tools = surface.enabled_tools();
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
    let tool_context = ToolExecutionContext {
        session_id: context.session_id.clone(),
        host: Arc::clone(&context.host),
    };
    let result = match find_native_tool(context.execution_mode, &tool_name) {
        Some(NativeTool::Batch) => execute_batch_tool_call(context, &args, progress).await,
        None => {
            context
                .plugins
                .tools()
                .execute_streaming_with_context(&tool_name, &args, &tool_context, progress)
                .await
        }
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

#[derive(Debug)]
struct BatchCallSpec {
    index: usize,
    tool: String,
    parameters: serde_json::Value,
}

struct BatchCallOutcome {
    index: usize,
    tool: String,
    success: bool,
    duration_ms: u64,
    result: serde_json::Value,
    images: Vec<ToolImage>,
}

async fn execute_batch_tool_call(
    context: &ToolDispatchContext,
    args: &serde_json::Value,
    progress: Option<&ProgressSender>,
) -> ToolResult {
    let specs = match parse_batch_specs(args) {
        Ok(specs) => specs,
        Err(err) => return err,
    };

    let progress = progress.cloned();
    let mut immediate_outcomes = Vec::new();
    let mut pending = FuturesUnordered::new();

    for spec in specs.into_iter().take(BATCH_MAX_TOOL_CALLS) {
        if spec.tool == "batch" {
            immediate_outcomes.push(BatchCallOutcome {
                index: spec.index,
                tool: spec.tool,
                success: false,
                duration_ms: 0,
                result: serde_json::json!("Tool 'batch' is not allowed inside batch"),
                images: Vec::new(),
            });
            continue;
        }

        let dispatch = context.clone();
        let progress = progress.clone();
        pending.push(async move {
            let outcome = dispatch_tool_call(
                &dispatch,
                spec.tool.clone(),
                spec.parameters,
                progress.as_ref(),
            )
            .await;
            BatchCallOutcome {
                index: spec.index,
                tool: outcome.record.tool,
                success: outcome.record.success,
                duration_ms: outcome.record.duration_ms,
                result: outcome.record.result,
                images: outcome.images,
            }
        });
    }

    while let Some(outcome) = pending.next().await {
        immediate_outcomes.push(outcome);
    }

    for overflow_index in BATCH_MAX_TOOL_CALLS
        ..args
            .get("tool_calls")
            .and_then(|value| value.as_array())
            .map(|value| value.len())
            .unwrap_or_default()
    {
        immediate_outcomes.push(BatchCallOutcome {
            index: overflow_index,
            tool: args
                .get("tool_calls")
                .and_then(|value| value.as_array())
                .and_then(|items| items.get(overflow_index))
                .and_then(|item| item.get("tool"))
                .and_then(|value| value.as_str())
                .unwrap_or("unknown")
                .to_string(),
            success: false,
            duration_ms: 0,
            result: serde_json::json!("Maximum of 25 tool calls allowed in batch"),
            images: Vec::new(),
        });
    }

    immediate_outcomes.sort_by_key(|outcome| outcome.index);
    let successful = immediate_outcomes
        .iter()
        .filter(|outcome| outcome.success)
        .count();
    let failed = immediate_outcomes.len().saturating_sub(successful);
    let summary = if failed == 0 {
        format!("All {successful} tools executed successfully.")
    } else {
        format!(
            "Executed {successful}/{} tools successfully. {failed} failed.",
            immediate_outcomes.len()
        )
    };

    let mut images = Vec::new();
    let results = immediate_outcomes
        .into_iter()
        .map(|mut outcome| {
            images.append(&mut outcome.images);
            let mut record = serde_json::Map::new();
            record.insert("tool".to_string(), serde_json::json!(outcome.tool));
            record.insert("success".to_string(), serde_json::json!(outcome.success));
            record.insert(
                "duration_ms".to_string(),
                serde_json::json!(outcome.duration_ms),
            );
            record.insert(
                if outcome.success {
                    "result".to_string()
                } else {
                    "error".to_string()
                },
                outcome.result,
            );
            serde_json::Value::Object(record)
        })
        .collect::<Vec<_>>();

    ToolResult::with_images(
        true,
        serde_json::json!({
            "summary": summary,
            "total": results.len(),
            "successful": successful,
            "failed": failed,
            "results": results,
        }),
        images,
    )
}

fn parse_batch_specs(args: &serde_json::Value) -> Result<Vec<BatchCallSpec>, ToolResult> {
    let Some(items) = args.get("tool_calls").and_then(|value| value.as_array()) else {
        return Err(ToolResult::err_fmt(
            "Missing required parameter: tool_calls",
        ));
    };
    if items.is_empty() {
        return Err(ToolResult::err_fmt(
            "Invalid tool_calls: provide at least one tool call",
        ));
    }

    items
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let Some(object) = item.as_object() else {
                return Err(ToolResult::err_fmt(format_args!(
                    "Invalid tool_calls[{index}]: expected object with `tool` and `parameters`"
                )));
            };
            let tool = object
                .get("tool")
                .and_then(|value| value.as_str())
                .filter(|value| !value.is_empty())
                .ok_or_else(|| {
                    ToolResult::err_fmt(format_args!(
                        "Invalid tool_calls[{index}].tool: expected non-empty string"
                    ))
                })?;
            let parameters = object.get("parameters").cloned().ok_or_else(|| {
                ToolResult::err_fmt(format_args!(
                    "Invalid tool_calls[{index}].parameters: expected object"
                ))
            })?;
            if !parameters.is_object() {
                return Err(ToolResult::err_fmt(format_args!(
                    "Invalid tool_calls[{index}].parameters: expected object"
                )));
            }
            Ok(BatchCallSpec {
                index,
                tool: tool.to_string(),
                parameters,
            })
        })
        .collect()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::{
        PluginError, PluginHost, SessionHandle, SessionSnapshot, StaticPluginFactory,
    };
    use crate::{AgentStateEnvelope, ContextFoldingConfig, ExecutionMode, ToolProvider, TurnInput};
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
                },
                crate::ToolDefinition {
                    name: "beta".into(),
                    description: String::new(),
                    params: vec![crate::ToolParam::typed("value", "str")],
                    returns: "str".into(),
                    examples: vec![],
                    enabled: true,
                    injected: true,
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
                },
                crate::ToolDefinition {
                    name: "probe_b".into(),
                    description: String::new(),
                    params: vec![],
                    returns: "str".into(),
                    examples: vec![],
                    enabled: true,
                    injected: true,
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

    struct MockSessionManager;

    #[async_trait::async_trait]
    impl SessionManager for MockSessionManager {
        async fn snapshot_current(&self) -> Result<SessionSnapshot, PluginError> {
            Ok(dummy_snapshot())
        }

        async fn snapshot_session(
            &self,
            _session_id: &str,
        ) -> Result<SessionSnapshot, PluginError> {
            Ok(dummy_snapshot())
        }

        async fn tool_catalog(
            &self,
            _session_id: &str,
        ) -> Result<Vec<serde_json::Value>, PluginError> {
            Ok(Vec::new())
        }

        async fn create_session(
            &self,
            _request: crate::SessionCreateRequest,
        ) -> Result<SessionHandle, PluginError> {
            Ok(SessionHandle {
                session_id: "s".to_string(),
                config: crate::SessionConfigSnapshot {
                    provider_kind: crate::provider::ProviderKind::OpenAiGeneric,
                    model: "mock-model".to_string(),
                    model_variant: None,
                    execution_mode: ExecutionMode::Standard,
                    context_folding: ContextFoldingConfig::default(),
                    context_window: None,
                    max_turns: None,
                    include_soul: false,
                    sub_agent: false,
                },
            })
        }

        async fn close_session(&self, _session_id: &str) -> Result<(), PluginError> {
            Ok(())
        }

        async fn start_turn_stream(
            &self,
            session_id: &str,
            _input: TurnInput,
        ) -> Result<crate::plugin::SessionTurnHandle, PluginError> {
            let (_tx, rx) = mpsc::channel(1);
            Ok(crate::plugin::SessionTurnHandle {
                turn_id: "turn".to_string(),
                session_id: session_id.to_string(),
                config: crate::SessionConfigSnapshot {
                    provider_kind: crate::provider::ProviderKind::OpenAiGeneric,
                    model: "mock-model".to_string(),
                    model_variant: None,
                    execution_mode: ExecutionMode::Standard,
                    context_folding: ContextFoldingConfig::default(),
                    context_window: None,
                    max_turns: None,
                    include_soul: false,
                    sub_agent: false,
                },
                events: rx,
            })
        }

        async fn await_turn(
            &self,
            _turn_id: &str,
        ) -> Result<crate::runtime::AssembledTurn, PluginError> {
            Err(PluginError::Invoke("unused".to_string()))
        }

        async fn cancel_turn(&self, _turn_id: &str) -> Result<(), PluginError> {
            Ok(())
        }
    }

    fn dummy_snapshot() -> SessionSnapshot {
        AgentStateEnvelope {
            agent_id: "root".to_string(),
            execution_mode: ExecutionMode::Standard,
            context_folding: ContextFoldingConfig::default(),
            messages: Vec::new(),
            iteration: 0,
            token_usage: crate::TokenUsage::default(),
            last_prompt_usage: None,
            task_state: None,
            replay_manifest: None,
            plugin_snapshot: None,
            repl_snapshot: None,
        }
    }

    fn dispatch_context() -> ToolDispatchContext {
        let (event_tx, _event_rx) = mpsc::channel(8);
        ToolDispatchContext {
            plugins: test_plugins(Arc::new(MockTools)),
            host: Arc::new(MockSessionManager),
            session_id: "session".to_string(),
            execution_mode: ExecutionMode::Standard,
            event_tx,
            turn_injection_bridge: crate::TurnInjectionBridge::new(),
        }
    }

    fn parallel_dispatch_context(
        barrier: Arc<Barrier>,
        started: Arc<AtomicUsize>,
    ) -> ToolDispatchContext {
        let (event_tx, _event_rx) = mpsc::channel(8);
        ToolDispatchContext {
            plugins: test_plugins(Arc::new(ParallelProbeTools { barrier, started })),
            host: Arc::new(MockSessionManager),
            session_id: "session".to_string(),
            execution_mode: ExecutionMode::Standard,
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
