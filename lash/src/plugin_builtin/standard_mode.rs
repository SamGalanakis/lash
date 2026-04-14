use std::sync::Arc;

use futures_util::stream::{FuturesUnordered, StreamExt};

use crate::plugin::{
    ModeExecutionPlugin, ModeExecutionPreamble, ModeNativeToolsPlugin, ModeSessionPlugin,
    ModeTurnConfig, PluginError, PluginFactory, PluginRegistrar, PluginSessionContext,
    SessionPlugin,
};
use crate::tool_dispatch::{ToolDispatchContext, dispatch_tool_call};
use crate::tools::batch::batch_tool_definition;
use crate::{ExecutionMode, ProgressSender, SessionError, ToolImage, ToolResult};

const STANDARD_EXECUTION_SECTION: &str = r#"Use direct tool calls when execution is needed.

- Work in small, concrete steps and verify each meaningful step.
- Use `batch` for two or more independent tool calls. Serialize calls when later arguments depend on earlier results.
- Avoid filler prose between tool calls.
- If you are unsure, resolve the uncertainty with the smallest relevant check.
- Before concluding, verify the concrete end-state whenever possible.
- For direct conversational requests that need no tools, respond in prose only."#;

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
        reg.mode().execution(Arc::new(StandardModeExecution))?;
        reg.mode().session(Arc::new(StandardModeSession))?;
        reg.mode().native_tools(Arc::new(StandardModeNativeTools))?;
        Ok(())
    }
}

struct StandardModeExecution;

impl ModeExecutionPlugin for StandardModeExecution {
    fn build_execution_preamble(
        &self,
        surface: &crate::plugin::ExecutionSurface,
    ) -> ModeExecutionPreamble {
        let enabled_tools = surface.enabled_tools();
        ModeExecutionPreamble {
            tool_specs: Arc::new(lash_sansio::session_model::model_tool_specs(&enabled_tools)),
            tool_names: enabled_tools.iter().map(|tool| tool.name.clone()).collect(),
            omitted_tool_count: 0,
            execution_prompt: STANDARD_EXECUTION_SECTION.to_string(),
            prompt_contributions: Vec::new(),
        }
    }

    fn turn_config(&self) -> ModeTurnConfig {
        ModeTurnConfig {
            protocol: crate::sansio::TurnProtocol::Standard,
            sync_execution_surface: false,
        }
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

struct BatchCallOutcome {
    index: usize,
    tool: String,
    success: bool,
    duration_ms: u64,
    result: serde_json::Value,
    images: Vec<ToolImage>,
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
    let images = immediate_outcomes
        .iter()
        .flat_map(|outcome| outcome.images.clone())
        .collect::<Vec<_>>();
    let results = immediate_outcomes
        .into_iter()
        .map(|outcome| {
            let mut record = serde_json::Map::new();
            record.insert("index".to_string(), serde_json::json!(outcome.index));
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
