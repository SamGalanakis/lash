use std::sync::Arc;
use std::time::Instant;

use futures_util::stream::{FuturesUnordered, StreamExt};
use tokio::sync::mpsc;

use crate::plugin::{
    PluginDirective, PluginSession, SessionManager, ToolCallHookContext, ToolResultHookContext,
    emit_plugin_surface_events,
};
use crate::{
    ProgressSender, SessionEvent, ToolCallRecord, ToolExecutionContext, ToolExecutionMode,
    ToolImage, ToolProvider, ToolResult, ToolSurface, TurnInjectionBridge,
};

#[derive(Clone)]
pub struct ToolDispatchContext {
    pub plugins: Arc<PluginSession>,
    pub tools: Arc<dyn ToolProvider>,
    pub surface: ToolSurface,
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

#[derive(Clone)]
pub struct ParallelToolCallSpec {
    pub index: usize,
    pub tool_name: String,
    pub args: serde_json::Value,
}

#[derive(Clone)]
pub struct ParallelToolCallOutcome {
    pub index: usize,
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
    if !context.surface.has_enabled_tool(&tool_name) {
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

pub(crate) async fn dispatch_parallel_tool_call(
    context: Arc<ToolDispatchContext>,
    spec: ParallelToolCallSpec,
    progress: Option<ProgressSender>,
) -> ParallelToolCallOutcome {
    let outcome = dispatch_tool_call(&context, spec.tool_name, spec.args, progress.as_ref()).await;
    ParallelToolCallOutcome {
        index: spec.index,
        record: outcome.record,
        images: outcome.images,
    }
}

/// Resolve the [`ToolExecutionMode`] declared on a tool's definition. Unknown
/// tool names default to [`ToolExecutionMode::Parallel`] — the dispatcher
/// will still surface an "unknown tool" error via the normal path.
pub(crate) fn resolve_tool_execution_mode(
    context: &ToolDispatchContext,
    tool_name: &str,
) -> ToolExecutionMode {
    context
        .surface
        .tools
        .iter()
        .find(|def| def.name == tool_name)
        .map(|def| def.execution_mode)
        .unwrap_or_default()
}

/// Dispatch a batch of tool calls produced by one model response.
///
/// Strategy (Option A from the design doc): tools with
/// [`ToolExecutionMode::Parallel`] run concurrently first; tools with
/// [`ToolExecutionMode::Serial`] then run one-at-a-time after all parallel
/// work has finished. Results are re-sorted by the caller-provided `index`
/// so downstream consumers see them in the original model-emitted order.
///
/// We chose Option A because it preserves parallelism for read-only tools
/// (`read_file`, `grep`, `glob`, ...) even when a single mutating tool
/// (`apply_patch`, `exec_command`, ...) is mixed into the same batch. The
/// alternative (Option B — serialise the whole batch when any call is
/// serial) would throw away that parallelism unnecessarily.
pub async fn dispatch_parallel_tool_calls(
    context: Arc<ToolDispatchContext>,
    specs: Vec<ParallelToolCallSpec>,
    progress: Option<&ProgressSender>,
) -> Vec<ParallelToolCallOutcome> {
    let progress = progress.cloned();

    let mut parallel_specs = Vec::new();
    let mut serial_specs = Vec::new();
    for spec in specs {
        match resolve_tool_execution_mode(&context, &spec.tool_name) {
            ToolExecutionMode::Parallel => parallel_specs.push(spec),
            ToolExecutionMode::Serial => serial_specs.push(spec),
        }
    }

    let mut outcomes = Vec::new();

    // 1. Run every parallel-safe call concurrently.
    let mut pending = FuturesUnordered::new();
    for spec in parallel_specs {
        pending.push(dispatch_parallel_tool_call(
            Arc::clone(&context),
            spec,
            progress.clone(),
        ));
    }
    while let Some(outcome) = pending.next().await {
        outcomes.push(outcome);
    }

    // 2. Run serial calls sequentially, in submission order. We preserve the
    //    original emission order by sorting the serial bucket by its
    //    caller-provided index before running — this way tools like
    //    `apply_patch` that were emitted in sequence by the model also run
    //    in that same sequence.
    serial_specs.sort_by_key(|spec| spec.index);
    for spec in serial_specs {
        let outcome =
            dispatch_parallel_tool_call(Arc::clone(&context), spec, progress.clone()).await;
        outcomes.push(outcome);
    }

    outcomes.sort_by_key(|outcome| outcome.index);
    outcomes
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
                    execution_mode: ToolExecutionMode::Parallel,
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
                    execution_mode: ToolExecutionMode::Parallel,
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
                    execution_mode: ToolExecutionMode::Parallel,
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
                    execution_mode: ToolExecutionMode::Parallel,
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

    use crate::testing::MockSessionManager;

    fn dispatch_context() -> ToolDispatchContext {
        let (event_tx, _event_rx) = mpsc::channel(8);
        let plugins = test_plugins(Arc::new(MockTools));
        let tools = plugins.tools();
        let surface = plugins.tool_surface("session", ExecutionMode::Standard);
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
        let surface = plugins.tool_surface("session", ExecutionMode::Standard);
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
        let results = outcome
            .record
            .result
            .get("results")
            .and_then(|value| value.as_array())
            .expect("results");
        assert_eq!(results.len(), 3);
        assert_eq!(
            results
                .iter()
                .filter(|item| item.get("success").and_then(|value| value.as_bool()) == Some(true))
                .count(),
            2
        );
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
        let results = outcome
            .record
            .result
            .get("results")
            .and_then(|value| value.as_array())
            .expect("results");
        assert_eq!(results.len(), 26);
        assert_eq!(
            results
                .iter()
                .filter(|item| item.get("success").and_then(|value| value.as_bool()) == Some(true))
                .count(),
            25
        );
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
        let results = outcome
            .record
            .result
            .get("results")
            .and_then(|value| value.as_array())
            .expect("results");
        assert_eq!(results.len(), 2);
        assert!(
            results
                .iter()
                .all(|item| item.get("success").and_then(|value| value.as_bool()) == Some(true))
        );
    }

    /// A tool provider whose tools are marked [`ToolExecutionMode::Serial`]
    /// and log (start, end) instants around a sleep into a shared `Mutex`.
    struct SerialProbeTools {
        /// (tool_name, start_instant, end_instant)
        log: Arc<std::sync::Mutex<Vec<(String, Instant, Instant)>>>,
    }

    #[async_trait::async_trait]
    impl ToolProvider for SerialProbeTools {
        fn definitions(&self) -> Vec<crate::ToolDefinition> {
            vec![
                crate::ToolDefinition {
                    name: "serial_a".into(),
                    description: String::new(),
                    params: vec![],
                    returns: "str".into(),
                    examples: vec![],
                    enabled: true,
                    injected: true,
                    input_schema_override: None,
                    output_schema_override: None,
                    execution_mode: ToolExecutionMode::Serial,
                },
                crate::ToolDefinition {
                    name: "serial_b".into(),
                    description: String::new(),
                    params: vec![],
                    returns: "str".into(),
                    examples: vec![],
                    enabled: true,
                    injected: true,
                    input_schema_override: None,
                    output_schema_override: None,
                    execution_mode: ToolExecutionMode::Serial,
                },
            ]
        }

        async fn execute(&self, name: &str, _args: &serde_json::Value) -> ToolResult {
            let start = Instant::now();
            // Sleep long enough that if the two tools *were* dispatched
            // concurrently, their windows would overlap by a detectable
            // margin.
            tokio::time::sleep(Duration::from_millis(40)).await;
            let end = Instant::now();
            self.log
                .lock()
                .expect("serial probe log")
                .push((name.to_string(), start, end));
            ToolResult::ok(json!(name))
        }
    }

    fn serial_dispatch_context(
        log: Arc<std::sync::Mutex<Vec<(String, Instant, Instant)>>>,
    ) -> ToolDispatchContext {
        let (event_tx, _event_rx) = mpsc::channel(8);
        let plugins = test_plugins(Arc::new(SerialProbeTools { log }));
        let tools = plugins.tools();
        let surface = plugins.tool_surface("session", ExecutionMode::Standard);
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

    /// Two Serial tools in the same batch must not interleave: the second
    /// call's start instant must be at or after the first call's end
    /// instant.
    #[tokio::test]
    async fn serial_tools_do_not_interleave() {
        let log: Arc<std::sync::Mutex<Vec<(String, Instant, Instant)>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let context = Arc::new(serial_dispatch_context(Arc::clone(&log)));

        let specs = vec![
            ParallelToolCallSpec {
                index: 0,
                tool_name: "serial_a".to_string(),
                args: json!({}),
            },
            ParallelToolCallSpec {
                index: 1,
                tool_name: "serial_b".to_string(),
                args: json!({}),
            },
        ];

        let outcomes = dispatch_parallel_tool_calls(context, specs, None).await;

        assert_eq!(outcomes.len(), 2);
        assert!(outcomes.iter().all(|outcome| outcome.record.success));
        // Outcomes are sorted by original index.
        assert_eq!(outcomes[0].index, 0);
        assert_eq!(outcomes[1].index, 1);
        assert_eq!(outcomes[0].record.tool, "serial_a");
        assert_eq!(outcomes[1].record.tool, "serial_b");

        let entries = log.lock().expect("log").clone();
        assert_eq!(entries.len(), 2, "both serial tools must have executed");
        // Sort entries by start time so we compare the first-to-run vs
        // second-to-run regardless of which tool happened to go first.
        let mut sorted = entries;
        sorted.sort_by_key(|(_, start, _)| *start);
        let (first_name, _first_start, first_end) = &sorted[0];
        let (second_name, second_start, _second_end) = &sorted[1];
        assert_ne!(first_name, second_name, "both tools should have run");
        assert!(
            second_start >= first_end,
            "serial tool ranges must not overlap: first ended at {:?}, second started at {:?}",
            first_end,
            second_start,
        );
    }

    /// When a batch contains a mix of parallel and serial tools, the
    /// parallel-safe ones should still run concurrently with each other
    /// (verified via a Barrier), and the serial one should run separately
    /// without interleaving with any parallel peer's window.
    #[tokio::test]
    async fn mixed_batch_runs_parallel_tools_concurrently_and_serial_alone() {
        struct MixedTools {
            barrier: Arc<Barrier>,
            serial_window: Arc<std::sync::Mutex<Option<(Instant, Instant)>>>,
            parallel_windows: Arc<std::sync::Mutex<Vec<(String, Instant, Instant)>>>,
        }

        #[async_trait::async_trait]
        impl ToolProvider for MixedTools {
            fn definitions(&self) -> Vec<crate::ToolDefinition> {
                vec![
                    crate::ToolDefinition {
                        name: "par_a".into(),
                        description: String::new(),
                        params: vec![],
                        returns: "str".into(),
                        examples: vec![],
                        enabled: true,
                        injected: true,
                        input_schema_override: None,
                        output_schema_override: None,
                        execution_mode: ToolExecutionMode::Parallel,
                    },
                    crate::ToolDefinition {
                        name: "par_b".into(),
                        description: String::new(),
                        params: vec![],
                        returns: "str".into(),
                        examples: vec![],
                        enabled: true,
                        injected: true,
                        input_schema_override: None,
                        output_schema_override: None,
                        execution_mode: ToolExecutionMode::Parallel,
                    },
                    crate::ToolDefinition {
                        name: "ser".into(),
                        description: String::new(),
                        params: vec![],
                        returns: "str".into(),
                        examples: vec![],
                        enabled: true,
                        injected: true,
                        input_schema_override: None,
                        output_schema_override: None,
                        execution_mode: ToolExecutionMode::Serial,
                    },
                ]
            }

            async fn execute(&self, name: &str, _args: &serde_json::Value) -> ToolResult {
                if name == "ser" {
                    let start = Instant::now();
                    tokio::time::sleep(Duration::from_millis(30)).await;
                    let end = Instant::now();
                    *self.serial_window.lock().expect("serial window") = Some((start, end));
                    ToolResult::ok(json!(name))
                } else {
                    let start = Instant::now();
                    // Block until both parallel tools have reached this
                    // barrier — proves they're running concurrently.
                    let waited = timeout(Duration::from_millis(200), self.barrier.wait()).await;
                    let end = Instant::now();
                    self.parallel_windows
                        .lock()
                        .expect("parallel windows")
                        .push((name.to_string(), start, end));
                    match waited {
                        Ok(_) => ToolResult::ok(json!(name)),
                        Err(_) => ToolResult::err_fmt(format!("{name} did not overlap with peer")),
                    }
                }
            }
        }

        let barrier = Arc::new(Barrier::new(2));
        let serial_window = Arc::new(std::sync::Mutex::new(None));
        let parallel_windows = Arc::new(std::sync::Mutex::new(Vec::new()));
        let (event_tx, _event_rx) = mpsc::channel(8);
        let provider = Arc::new(MixedTools {
            barrier: Arc::clone(&barrier),
            serial_window: Arc::clone(&serial_window),
            parallel_windows: Arc::clone(&parallel_windows),
        });
        let plugins = test_plugins(provider);
        let tools = plugins.tools();
        let surface = plugins.tool_surface("session", ExecutionMode::Standard);
        let context = Arc::new(ToolDispatchContext {
            plugins,
            tools,
            surface,
            host: Arc::new(MockSessionManager::default()),
            session_id: "session".to_string(),
            event_tx,
            turn_injection_bridge: crate::TurnInjectionBridge::new(),
        });

        let specs = vec![
            ParallelToolCallSpec {
                index: 0,
                tool_name: "par_a".to_string(),
                args: json!({}),
            },
            ParallelToolCallSpec {
                index: 1,
                tool_name: "ser".to_string(),
                args: json!({}),
            },
            ParallelToolCallSpec {
                index: 2,
                tool_name: "par_b".to_string(),
                args: json!({}),
            },
        ];

        let outcomes = dispatch_parallel_tool_calls(context, specs, None).await;

        assert_eq!(outcomes.len(), 3);
        assert!(
            outcomes.iter().all(|outcome| outcome.record.success),
            "all tools should succeed: {:?}",
            outcomes
                .iter()
                .map(|outcome| (&outcome.record.tool, outcome.record.success))
                .collect::<Vec<_>>()
        );

        let pw = parallel_windows.lock().expect("parallel windows");
        assert_eq!(pw.len(), 2);
        let sw = serial_window
            .lock()
            .expect("serial window")
            .expect("serial window recorded");

        // The serial tool's window must not overlap either parallel
        // tool's window (Option A: serial runs after parallel).
        for (name, p_start, p_end) in pw.iter() {
            assert!(
                sw.0 >= *p_end || sw.1 <= *p_start,
                "serial window {:?} overlaps parallel window {} {:?}..{:?}",
                sw,
                name,
                p_start,
                p_end,
            );
        }
    }
}
