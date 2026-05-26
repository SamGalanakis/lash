//! Mode-agnostic runtime-control tools (`list_process_handles`,
//! `cancel_process`).
//!
//! Dedicated plugins register these tools into the native-tools surface,
//! so mode crates do not own or duplicate runtime control behavior. RLM
//! hides process-control tools when a mode exposes the same control through
//! native process handles.

use std::sync::Arc;

use serde_json::Value;

use crate::plugin::{
    ModeNativeToolsPlugin, PluginError, PluginFactory, PluginRegistrar, PluginSessionContext,
    SessionPlugin,
};
use crate::tool_dispatch::ToolDispatchContext;
use crate::{
    ProgressSender, ToolContract, ToolDefinition, ToolExecutionMode, ToolManifest, ToolResult,
};

/// Plugin factory for mode-agnostic process-control tools.
#[derive(Default)]
pub struct BuiltinProcessControlsPluginFactory;

impl BuiltinProcessControlsPluginFactory {
    pub fn new() -> Self {
        Self
    }
}

impl PluginFactory for BuiltinProcessControlsPluginFactory {
    fn id(&self) -> &'static str {
        "process_controls"
    }

    fn build(&self, ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(ProcessControlsPlugin {
            enabled: ctx.execution_mode != crate::ExecutionMode::new("rlm"),
        }))
    }
}

struct ProcessControlsPlugin {
    enabled: bool,
}

impl SessionPlugin for ProcessControlsPlugin {
    fn id(&self) -> &'static str {
        "process_controls"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        if self.enabled {
            reg.mode()
                .native_tools(Arc::new(ProcessControlsNativeTools))?;
        }
        Ok(())
    }
}

struct ProcessControlsNativeTools;

#[async_trait::async_trait]
impl ModeNativeToolsPlugin for ProcessControlsNativeTools {
    fn tool_manifests(&self) -> Vec<ToolManifest> {
        process_control_tool_definitions()
            .into_iter()
            .map(|tool| tool.manifest())
            .collect()
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<ToolContract>> {
        process_control_tool_definitions()
            .into_iter()
            .find(|tool| tool.name == name)
            .map(|tool| Arc::new(tool.contract()))
    }

    async fn execute(
        &self,
        context: &ToolDispatchContext<'_>,
        name: &str,
        args: &Value,
        _progress: Option<&ProgressSender>,
    ) -> Option<ToolResult> {
        match name {
            "list_process_handles" => Some(execute_process_list_tool_call(context).await),
            "cancel_process" => Some(execute_process_cancel_tool_call(context, args).await),
            _ => None,
        }
    }
}

pub fn process_list_tool_definition() -> ToolDefinition {
    ToolDefinition::raw(
        "tool:list_process_handles",
        "list_process_handles",
        "List every process handle granted to this session with its process id, descriptor, and terminal status.",
        ToolDefinition::default_input_schema(),
        serde_json::json!({ "type": "object", "additionalProperties": true }),
    )
    .with_examples(vec!["list_process_handles()".into()])
    .with_execution_mode(ToolExecutionMode::Parallel)
}

fn process_control_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        process_list_tool_definition(),
        process_cancel_tool_definition(),
    ]
}

pub fn process_cancel_tool_definition() -> ToolDefinition {
    ToolDefinition::raw(
        "tool:cancel_process",
        "cancel_process",
        "Request cancellation for a durable process by `process_id`.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "process_id": {
                    "type": "string",
                    "description": "Process id returned by a process handle or `list_process_handles`."
                }
            },
            "required": ["process_id"],
            "additionalProperties": false
        }),
        serde_json::json!({ "type": "object", "additionalProperties": true }),
    )
    .with_examples(vec![
        r#"cancel_process(process_id="monitor:app-errors")"#.into(),
        r#"cancel_process(process_id="subagent:session-01JZK7G4QP9Q4J7W3Q2E1H6M9C")"#.into(),
    ])
    .with_execution_mode(ToolExecutionMode::Parallel)
}

pub async fn execute_process_list_tool_call(context: &ToolDispatchContext<'_>) -> ToolResult {
    match context
        .host
        .list_process_handles(
            crate::ProcessListRequest::new(&context.session_id).with_scope(
                crate::ProcessRequestScope::new()
                    .with_effect_metadata(context.tool_effect_metadata.clone())
                    .with_effect_controller(context.effect_controller.as_controller()),
            ),
        )
        .await
    {
        Ok(entries) => {
            let entries: Vec<Value> = entries
                .into_iter()
                .map(|(grant, process)| {
                    serde_json::json!({
                        "process_id": process.id,
                        "descriptor": grant.descriptor,
                        "terminal": terminal_label(process.terminal.as_ref()),
                    })
                })
                .collect();
            ToolResult::ok(serde_json::json!({ "processes": entries }))
        }
        Err(err) => ToolResult::err_fmt(err.to_string()),
    }
}

pub async fn execute_process_cancel_tool_call(
    context: &ToolDispatchContext<'_>,
    args: &Value,
) -> ToolResult {
    let Some(id) = args
        .get("process_id")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return ToolResult::err_fmt("cancel_process requires `process_id`");
    };
    if let Err(err) = context
        .host
        .validate_process_handles_visible(&context.session_id, &[id.to_string()])
        .await
    {
        return ToolResult::err_fmt(err.to_string());
    }
    match context
        .host
        .cancel_process(
            crate::ProcessCancelRequest::new(&context.session_id, id).with_scope(
                crate::ProcessRequestScope::new()
                    .with_effect_metadata(context.tool_effect_metadata.clone())
                    .with_effect_controller(context.effect_controller.as_controller()),
            ),
        )
        .await
    {
        Ok(status) => ToolResult::ok(serde_json::json!({
            "process_id": status.id,
            "terminal": terminal_label(status.terminal.as_ref()),
        })),
        Err(err) => ToolResult::err_fmt(err.to_string()),
    }
}

fn terminal_label(terminal: Option<&crate::ProcessTerminalSemantics>) -> &'static str {
    match terminal.map(|terminal| terminal.state) {
        None => "running",
        Some(crate::ProcessTerminalState::Completed) => "completed",
        Some(crate::ProcessTerminalState::Failed) => "failed",
        Some(crate::ProcessTerminalState::Cancelled) => "cancelled",
    }
}
