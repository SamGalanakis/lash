use std::sync::Arc;
use std::sync::RwLock;
use std::time::Instant;

use crate::{
    DynamicStateSnapshot, ExecutionMode, ProgressSender, ResolvedProjection, ToolDefinition,
    ToolParam, ToolProvider, ToolResult,
};

const MAX_BATCH_CALLS: usize = 25;

#[derive(Clone)]
pub struct BatchingTools {
    inner: Arc<dyn ToolProvider>,
    execution_mode: Arc<RwLock<ExecutionMode>>,
}

#[derive(Clone, Debug)]
struct BatchCall {
    tool: String,
    parameters: serde_json::Value,
}

#[derive(Clone, Debug)]
struct BatchOutcome {
    tool: String,
    parameters: serde_json::Value,
    result: ToolResult,
    duration_ms: u64,
}

enum PendingBatchOutcome {
    Ready(BatchOutcome),
    Running(tokio::task::JoinHandle<BatchOutcome>),
}

impl BatchingTools {
    pub fn new(inner: Arc<dyn ToolProvider>) -> Self {
        Self::with_mode(inner, ExecutionMode::Repl)
    }

    pub fn with_mode(inner: Arc<dyn ToolProvider>, execution_mode: ExecutionMode) -> Self {
        Self {
            inner,
            execution_mode: Arc::new(RwLock::new(execution_mode)),
        }
    }

    fn current_mode(&self) -> ExecutionMode {
        *self
            .execution_mode
            .read()
            .expect("batching tools mode lock poisoned")
    }

    fn batch_definition() -> ToolDefinition {
        ToolDefinition {
            name: "batch".into(),
            description: vec![crate::ToolText::new(
                format!(
                    "Execute 1-{} independent visible tool calls concurrently to reduce latency. Use for parallel reads, searches, inspections, and unrelated diagnostics. Do not batch dependent steps or nest `batch` inside `batch`.",
                    MAX_BATCH_CALLS
                ),
                [
                    crate::ExecutionMode::Repl,
                    crate::ExecutionMode::NativeTools,
                ],
            )],
            params: vec![ToolParam {
                name: "tool_calls".into(),
                r#type: "list".into(),
                description:
                    "Array of objects like {\"tool\": \"read_file\", \"parameters\": {...}}".into(),
                required: true,
            }],
            returns: "dict".into(),
            examples: vec![crate::ToolText::new(
                "batch(tool_calls=[{\"tool\":\"grep\",\"parameters\":{\"pattern\":\"TODO\",\"include\":\"src/**/*.rs\"}},{\"tool\":\"read_file\",\"parameters\":{\"path\":\"src/lib.rs\",\"limit\":200}}])",
                [
                    crate::ExecutionMode::Repl,
                    crate::ExecutionMode::NativeTools,
                ],
            )],
            hidden: false,
            inject_into_prompt: true,
        }
    }

    fn visible_inner_definitions(&self) -> Vec<ToolDefinition> {
        self.inner
            .definitions()
            .into_iter()
            .filter(|def| def.name != "batch")
            .collect()
    }

    fn merged_definitions(&self) -> Vec<ToolDefinition> {
        let mut defs = self.visible_inner_definitions();
        if matches!(self.current_mode(), ExecutionMode::NativeTools)
            && !defs.iter().any(|def| def.name == "batch")
        {
            defs.push(Self::batch_definition());
        }
        defs
    }

    async fn execute_batch(
        &self,
        args: &serde_json::Value,
        progress: Option<&ProgressSender>,
    ) -> ToolResult {
        let calls = match parse_batch_calls(args) {
            Ok(calls) => calls,
            Err(err) => return err,
        };

        let available_defs = self.visible_inner_definitions();
        let available_tools: std::collections::BTreeSet<String> =
            available_defs.iter().map(|def| def.name.clone()).collect();
        let available_tools_list = available_tools
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join(", ");

        let mut pending = Vec::new();
        for call in calls.into_iter().take(MAX_BATCH_CALLS) {
            if call.tool == "batch" {
                pending.push(PendingBatchOutcome::Ready(BatchOutcome {
                    tool: call.tool,
                    parameters: call.parameters,
                    result: ToolResult::err_fmt("Tool `batch` cannot be nested inside `batch`."),
                    duration_ms: 0,
                }));
                continue;
            }

            if !available_tools.contains(&call.tool) {
                pending.push(PendingBatchOutcome::Ready(BatchOutcome {
                    tool: call.tool,
                    parameters: call.parameters,
                    result: ToolResult::err_fmt(format!(
                        "Tool is not available in this session. Available tools: {available_tools_list}"
                    )),
                    duration_ms: 0,
                }));
                continue;
            }

            let provider = Arc::clone(&self.inner);
            let tool = call.tool.clone();
            let parameters = call.parameters.clone();
            let progress = progress.cloned();
            pending.push(PendingBatchOutcome::Running(tokio::spawn(async move {
                let started = Instant::now();
                let result = provider
                    .execute_streaming(&tool, &parameters, progress.as_ref())
                    .await;
                BatchOutcome {
                    tool,
                    parameters,
                    result,
                    duration_ms: started.elapsed().as_millis() as u64,
                }
            })));
        }

        let discarded_count = args
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .map(|calls| calls.len().saturating_sub(MAX_BATCH_CALLS))
            .unwrap_or(0);
        for _ in 0..discarded_count {
            pending.push(PendingBatchOutcome::Ready(BatchOutcome {
                tool: "unknown".into(),
                parameters: serde_json::json!({}),
                result: ToolResult::err_fmt(format!(
                    "Maximum of {MAX_BATCH_CALLS} tool calls allowed in batch."
                )),
                duration_ms: 0,
            }));
        }

        let mut outcomes = Vec::with_capacity(pending.len());
        for item in pending {
            match item {
                PendingBatchOutcome::Ready(outcome) => outcomes.push(outcome),
                PendingBatchOutcome::Running(handle) => match handle.await {
                    Ok(outcome) => outcomes.push(outcome),
                    Err(err) => outcomes.push(BatchOutcome {
                        tool: "unknown".into(),
                        parameters: serde_json::json!({}),
                        result: ToolResult::err_fmt(format!("batch task panicked: {err}")),
                        duration_ms: 0,
                    }),
                },
            }
        }

        let successful = outcomes
            .iter()
            .filter(|outcome| outcome.result.success)
            .count();
        let failed = outcomes.len().saturating_sub(successful);
        let images = outcomes
            .iter()
            .filter(|outcome| outcome.result.success)
            .flat_map(|outcome| outcome.result.images.clone())
            .collect::<Vec<_>>();
        let message = if failed > 0 {
            format!(
                "Executed {successful}/{} tool calls successfully. {failed} failed.",
                outcomes.len()
            )
        } else {
            format!(
                "All {} tool calls executed successfully. Prefer `batch` again for independent work.",
                outcomes.len()
            )
        };

        ToolResult::with_images(
            true,
            serde_json::json!({
                "__type__": "batch_results",
                "message": message,
                "summary": {
                    "total_calls": outcomes.len(),
                    "successful": successful,
                    "failed": failed,
                    "max_calls": MAX_BATCH_CALLS,
                },
                "results": outcomes.iter().map(|outcome| serde_json::json!({
                    "tool": outcome.tool,
                    "parameters": outcome.parameters,
                    "success": outcome.result.success,
                    "result": outcome.result.result,
                    "duration_ms": outcome.duration_ms,
                })).collect::<Vec<_>>(),
            }),
            images,
        )
    }
}

fn parse_batch_calls(args: &serde_json::Value) -> Result<Vec<BatchCall>, ToolResult> {
    let Some(raw_calls) = args.get("tool_calls").and_then(|value| value.as_array()) else {
        return Err(ToolResult::err_fmt(
            "Missing required parameter: tool_calls",
        ));
    };
    if raw_calls.is_empty() {
        return Err(ToolResult::err_fmt(
            "Invalid parameters for tool `batch`: provide at least one tool call.",
        ));
    }

    let mut out = Vec::with_capacity(raw_calls.len());
    for (idx, raw_call) in raw_calls.iter().enumerate() {
        let Some(obj) = raw_call.as_object() else {
            return Err(ToolResult::err_fmt(format!(
                "Invalid tool_calls[{idx}]: expected object with `tool` and `parameters`."
            )));
        };
        let Some(tool) = obj.get("tool").and_then(|value| value.as_str()) else {
            return Err(ToolResult::err_fmt(format!(
                "Invalid tool_calls[{idx}].tool: expected non-empty string."
            )));
        };
        if tool.trim().is_empty() {
            return Err(ToolResult::err_fmt(format!(
                "Invalid tool_calls[{idx}].tool: expected non-empty string."
            )));
        }
        let parameters = match obj.get("parameters") {
            None => serde_json::json!({}),
            Some(value) if value.is_null() => serde_json::json!({}),
            Some(value) if value.is_object() => value.clone(),
            Some(_) => {
                return Err(ToolResult::err_fmt(format!(
                    "Invalid tool_calls[{idx}].parameters: expected object."
                )));
            }
        };
        out.push(BatchCall {
            tool: tool.to_string(),
            parameters,
        });
    }
    Ok(out)
}

#[async_trait::async_trait]
impl ToolProvider for BatchingTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        self.merged_definitions()
    }

    fn dynamic_projection(&self) -> Option<ResolvedProjection> {
        self.inner.dynamic_projection()
    }

    fn dynamic_snapshot(&self) -> Option<DynamicStateSnapshot> {
        self.inner.dynamic_snapshot()
    }

    fn fork_dynamic_with_snapshot(
        &self,
        snapshot: DynamicStateSnapshot,
    ) -> Option<Arc<dyn ToolProvider>> {
        self.inner.fork_dynamic_with_snapshot(snapshot).map(|fork| {
            Arc::new(Self::with_mode(fork, self.current_mode())) as Arc<dyn ToolProvider>
        })
    }

    fn dynamic_capabilities_payload_json(&self) -> Option<String> {
        self.inner.dynamic_capabilities_payload_json()
    }

    fn dynamic_generation(&self) -> Option<u64> {
        self.inner.dynamic_generation()
    }

    async fn execute(&self, name: &str, args: &serde_json::Value) -> ToolResult {
        self.execute_streaming(name, args, None).await
    }

    async fn execute_streaming(
        &self,
        name: &str,
        args: &serde_json::Value,
        progress: Option<&ProgressSender>,
    ) -> ToolResult {
        if name == "batch" {
            if !matches!(self.current_mode(), ExecutionMode::NativeTools) {
                return ToolResult::err_fmt("Unknown tool: batch");
            }
            return self.execute_batch(args, progress).await;
        }
        self.inner.execute_streaming(name, args, progress).await
    }

    fn set_execution_mode(&self, execution_mode: ExecutionMode) {
        *self
            .execution_mode
            .write()
            .expect("batching tools mode lock poisoned") = execution_mode;
        self.inner.set_execution_mode(execution_mode);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    struct MockTools {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl ToolProvider for MockTools {
        fn definitions(&self) -> Vec<ToolDefinition> {
            vec![
                ToolDefinition {
                    name: "alpha".into(),
                    description: vec![crate::ToolText::new(
                        "alpha",
                        [
                            crate::ExecutionMode::Repl,
                            crate::ExecutionMode::NativeTools,
                        ],
                    )],
                    params: vec![],
                    returns: "str".into(),
                    examples: vec![],
                    hidden: false,
                    inject_into_prompt: true,
                },
                ToolDefinition {
                    name: "beta".into(),
                    description: vec![crate::ToolText::new(
                        "beta",
                        [
                            crate::ExecutionMode::Repl,
                            crate::ExecutionMode::NativeTools,
                        ],
                    )],
                    params: vec![],
                    returns: "str".into(),
                    examples: vec![],
                    hidden: false,
                    inject_into_prompt: true,
                },
            ]
        }

        async fn execute(&self, name: &str, _args: &serde_json::Value) -> ToolResult {
            self.calls.fetch_add(1, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(10)).await;
            ToolResult::ok(serde_json::json!(format!("{name}_ok")))
        }
    }

    #[tokio::test]
    async fn definitions_include_batch_once() {
        let provider: Arc<dyn ToolProvider> = Arc::new(MockTools {
            calls: Arc::new(AtomicUsize::new(0)),
        });
        let wrapped = BatchingTools::with_mode(provider, crate::ExecutionMode::NativeTools);
        let defs = wrapped.definitions();
        assert_eq!(defs.iter().filter(|def| def.name == "batch").count(), 1);
    }

    #[tokio::test]
    async fn batch_executes_multiple_tools_and_preserves_order() {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider: Arc<dyn ToolProvider> = Arc::new(MockTools {
            calls: Arc::clone(&calls),
        });
        let wrapped = BatchingTools::with_mode(provider, crate::ExecutionMode::NativeTools);

        let result = wrapped
            .execute(
                "batch",
                &serde_json::json!({
                    "tool_calls": [
                        { "tool": "alpha", "parameters": {} },
                        { "tool": "beta", "parameters": {} }
                    ]
                }),
            )
            .await;

        assert!(result.success);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        let results = result
            .result
            .get("results")
            .and_then(|value| value.as_array())
            .expect("results array");
        assert_eq!(results[0]["tool"], "alpha");
        assert_eq!(results[1]["tool"], "beta");
    }

    #[tokio::test]
    async fn batch_rejects_nested_batch() {
        let provider: Arc<dyn ToolProvider> = Arc::new(MockTools {
            calls: Arc::new(AtomicUsize::new(0)),
        });
        let wrapped = BatchingTools::with_mode(provider, crate::ExecutionMode::NativeTools);

        let result = wrapped
            .execute(
                "batch",
                &serde_json::json!({
                    "tool_calls": [
                        { "tool": "batch", "parameters": {} }
                    ]
                }),
            )
            .await;

        assert!(result.success);
        let first = result
            .result
            .get("results")
            .and_then(|value| value.as_array())
            .and_then(|results| results.first())
            .expect("first batch result");
        assert_eq!(first["success"], false);
    }

    #[tokio::test]
    async fn repl_mode_hides_batch_and_rejects_execution() {
        let provider: Arc<dyn ToolProvider> = Arc::new(MockTools {
            calls: Arc::new(AtomicUsize::new(0)),
        });
        let wrapped = BatchingTools::new(provider);

        let defs = wrapped.definitions();
        assert!(!defs.iter().any(|def| def.name == "batch"));

        let result = wrapped
            .execute("batch", &serde_json::json!({"tool_calls": []}))
            .await;
        assert!(!result.success);
    }
}
