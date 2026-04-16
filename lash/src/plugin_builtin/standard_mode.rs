use std::sync::Arc;

use crate::plugin::{
    ModeNativeToolsPlugin, ModeSessionPlugin, PluginError, PluginFactory, PluginRegistrar,
    PluginSessionContext, SessionPlugin,
};
use crate::tool_dispatch::{
    ParallelToolCallSpec, ToolDispatchContext, dispatch_parallel_tool_calls,
};
use crate::tools::batch::batch_tool_definition;
use crate::{ExecutionMode, ProgressSender, SessionError, ToolResult};

pub(crate) struct StandardModePluginFactory;

impl PluginFactory for StandardModePluginFactory {
    fn id(&self) -> &'static str {
        "mode_standard"
    }

    fn build(&self, ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(StandardModePlugin {
            active: matches!(ctx.execution_mode, ExecutionMode::Standard),
        }))
    }
}

struct StandardModePlugin {
    active: bool,
}

impl SessionPlugin for StandardModePlugin {
    fn id(&self) -> &'static str {
        "mode_standard"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        if !self.active {
            return Ok(());
        }
        reg.mode().session(Arc::new(StandardModeSession))?;
        reg.mode().native_tools(Arc::new(StandardModeNativeTools))?;
        Ok(())
    }
}

struct StandardModeSession;

#[async_trait::async_trait]
impl ModeSessionPlugin for StandardModeSession {
    async fn initialize_session(
        &self,
        _session: &mut crate::Session,
        _session_id: &str,
    ) -> Result<(), SessionError> {
        Ok(())
    }
}

struct StandardModeNativeTools;

#[async_trait::async_trait]
impl ModeNativeToolsPlugin for StandardModeNativeTools {
    fn definitions(&self) -> Vec<crate::ToolDefinition> {
        vec![batch_tool_definition()]
    }

    async fn execute(
        &self,
        context: &ToolDispatchContext,
        name: &str,
        args: &serde_json::Value,
        progress: Option<&ProgressSender>,
    ) -> Option<ToolResult> {
        match name {
            "batch" => Some(execute_batch_tool_call(context, args, progress).await),
            _ => None,
        }
    }
}

#[derive(Debug)]
struct BatchCallSpec {
    index: usize,
    tool: String,
    parameters: serde_json::Value,
}

const BATCH_MAX_TOOL_CALLS: usize = 25;

async fn execute_batch_tool_call(
    context: &ToolDispatchContext,
    args: &serde_json::Value,
    progress: Option<&ProgressSender>,
) -> ToolResult {
    let specs = match parse_batch_specs(args) {
        Ok(specs) => specs,
        Err(err) => return err,
    };

    let mut immediate_outcomes = Vec::new();
    let mut parallel_specs = Vec::new();

    for spec in specs.into_iter().take(BATCH_MAX_TOOL_CALLS) {
        if spec.tool == "batch" {
            immediate_outcomes.push(serde_json::json!({
                "index": spec.index,
                "tool": spec.tool,
                "success": false,
                "duration_ms": 0,
                "error": "Tool 'batch' is not allowed inside batch",
            }));
            continue;
        }
        parallel_specs.push(ParallelToolCallSpec {
            index: spec.index,
            tool_name: spec.tool,
            args: spec.parameters,
        });
    }

    let mut images = Vec::new();
    let mut parallel_outcomes =
        dispatch_parallel_tool_calls(Arc::new(context.clone()), parallel_specs, progress).await;
    for outcome in parallel_outcomes.drain(..) {
        images.extend(outcome.images);
        let mut record = serde_json::Map::new();
        record.insert("index".to_string(), serde_json::json!(outcome.index));
        record.insert("tool".to_string(), serde_json::json!(outcome.record.tool));
        record.insert(
            "success".to_string(),
            serde_json::json!(outcome.record.success),
        );
        record.insert(
            "duration_ms".to_string(),
            serde_json::json!(outcome.record.duration_ms),
        );
        record.insert(
            if outcome.record.success {
                "result".to_string()
            } else {
                "error".to_string()
            },
            outcome.record.result,
        );
        immediate_outcomes.push(serde_json::Value::Object(record));
    }

    for overflow_index in BATCH_MAX_TOOL_CALLS
        ..args
            .get("tool_calls")
            .and_then(|value| value.as_array())
            .map(|value| value.len())
            .unwrap_or_default()
    {
        immediate_outcomes.push(serde_json::json!({
            "index": overflow_index,
            "tool": args
                .get("tool_calls")
                .and_then(|value| value.as_array())
                .and_then(|items| items.get(overflow_index))
                .and_then(|item| item.get("tool"))
                .and_then(|value| value.as_str())
                .unwrap_or("unknown"),
            "success": false,
            "duration_ms": 0,
            "error": "Maximum of 25 tool calls allowed in batch",
        }));
    }

    immediate_outcomes.sort_by_key(|outcome| {
        outcome
            .get("index")
            .and_then(|value| value.as_u64())
            .unwrap_or(u64::MAX)
    });
    ToolResult::with_images(
        true,
        serde_json::json!({
            "results": immediate_outcomes,
        }),
        images,
    )
}

fn parse_batch_specs(args: &serde_json::Value) -> Result<Vec<BatchCallSpec>, ToolResult> {
    let Some(raw_calls) = args.get("tool_calls").and_then(|value| value.as_array()) else {
        return Err(ToolResult::err_fmt(
            "Missing required parameter: tool_calls",
        ));
    };
    if raw_calls.is_empty() {
        return Err(ToolResult::err_fmt(
            "Invalid tool_calls: expected at least one call",
        ));
    }

    let mut specs = Vec::with_capacity(raw_calls.len());
    for (index, item) in raw_calls.iter().enumerate() {
        let Some(object) = item.as_object() else {
            return Err(ToolResult::err_fmt(format_args!(
                "Invalid tool_calls[{index}]: expected object with tool and parameters"
            )));
        };
        let Some(tool) = object
            .get("tool")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|tool| !tool.is_empty())
        else {
            return Err(ToolResult::err_fmt(format_args!(
                "Invalid tool_calls[{index}].tool: expected non-empty string"
            )));
        };
        let parameters = object
            .get("parameters")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));
        specs.push(BatchCallSpec {
            index,
            tool: tool.to_string(),
            parameters,
        });
    }

    Ok(specs)
}
