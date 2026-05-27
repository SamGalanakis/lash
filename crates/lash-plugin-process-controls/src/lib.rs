//! Protocol-stack runtime-control tools (`list_process_handles`,
//! `cancel_process`).
//!
//! Dedicated plugins register these tools into the normal tool-provider
//! surface, so protocol crates do not own or duplicate runtime control behavior.

use std::sync::Arc;

use serde_json::Value;

use lash_core::plugin::{
    PluginError, PluginFactory, PluginRegistrar, PluginSessionContext, SessionPlugin,
};
use lash_core::{
    ToolAvailabilityConfig, ToolCall, ToolContract, ToolDefinition, ToolManifest, ToolProvider,
    ToolResult, ToolScheduling,
};

/// Plugin factory for process-control tools.
#[derive(Clone, Copy, Debug)]
pub struct ProcessControlsPluginFactory {
    include_cancel_process: bool,
}

impl ProcessControlsPluginFactory {
    pub fn new() -> Self {
        Self {
            include_cancel_process: true,
        }
    }

    pub fn without_cancel_process() -> Self {
        Self {
            include_cancel_process: false,
        }
    }
}

impl Default for ProcessControlsPluginFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginFactory for ProcessControlsPluginFactory {
    fn id(&self) -> &'static str {
        "process_controls"
    }

    fn build(&self, _ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(ProcessControlsPlugin {
            include_cancel_process: self.include_cancel_process,
        }))
    }
}

struct ProcessControlsPlugin {
    include_cancel_process: bool,
}

impl SessionPlugin for ProcessControlsPlugin {
    fn id(&self) -> &'static str {
        "process_controls"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        reg.tools().provider(Arc::new(ProcessControlsTools {
            include_cancel_process: self.include_cancel_process,
        }))?;
        Ok(())
    }
}

struct ProcessControlsTools {
    include_cancel_process: bool,
}

#[async_trait::async_trait]
impl ToolProvider for ProcessControlsTools {
    fn tool_manifests(&self) -> Vec<ToolManifest> {
        process_control_tool_definitions(self.include_cancel_process)
            .into_iter()
            .map(|tool| tool.manifest())
            .collect()
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<ToolContract>> {
        process_control_tool_definition(name, self.include_cancel_process)
            .map(|tool| Arc::new(tool.contract()))
    }

    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        match call.name {
            "list_process_handles" => execute_process_list_tool_call(call.context).await,
            "cancel_process" if self.include_cancel_process => {
                execute_process_cancel_tool_call(call.context, call.args).await
            }
            _ => ToolResult::err_fmt(format_args!("Unknown tool: {}", call.name)),
        }
    }
}

pub fn process_list_tool_definition() -> ToolDefinition {
    ToolDefinition::raw(
        "tool:list_process_handles",
        "list_process_handles",
        "List every process handle granted to this session with its process id, descriptor, and terminal status.",
        ToolDefinition::default_input_schema(),
        serde_json::json!({
            "type": "object",
            "properties": {
                "processes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "process_id": { "type": "string" },
                            "descriptor": { "type": "object", "additionalProperties": true },
                            "terminal": { "type": "string" }
                        },
                        "required": ["process_id", "descriptor", "terminal"],
                        "additionalProperties": false
                    }
                }
            },
            "required": ["processes"],
            "additionalProperties": false
        }),
    )
    .with_examples(vec!["list_process_handles()".into()])
    .with_availability(ToolAvailabilityConfig::callable())
    .with_scheduling(ToolScheduling::Parallel)
}

fn process_control_tool_definitions(include_cancel_process: bool) -> Vec<ToolDefinition> {
    let mut definitions = vec![process_list_tool_definition()];
    if include_cancel_process {
        definitions.push(process_cancel_tool_definition());
    }
    definitions
}

fn process_control_tool_definition(
    name: &str,
    include_cancel_process: bool,
) -> Option<ToolDefinition> {
    match name {
        "list_process_handles" => Some(process_list_tool_definition()),
        "cancel_process" if include_cancel_process => Some(process_cancel_tool_definition()),
        _ => None,
    }
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
        serde_json::json!({
            "type": "object",
            "properties": {
                "process_id": { "type": "string" },
                "terminal": { "type": "string" }
            },
            "required": ["process_id", "terminal"],
            "additionalProperties": false
        }),
    )
    .with_examples(vec![
        r#"cancel_process(process_id="tool:call-01JZK7G4QP9Q4J7W3Q2E1H6M9C")"#.into(),
        r#"cancel_process(process_id="subagent:session-01JZK7G4QP9Q4J7W3Q2E1H6M9C")"#.into(),
    ])
    .with_availability(ToolAvailabilityConfig::callable())
    .with_scheduling(ToolScheduling::Parallel)
}

pub async fn execute_process_list_tool_call(context: &lash_core::ToolContext<'_>) -> ToolResult {
    match context.list_process_handles().await {
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
    context: &lash_core::ToolContext<'_>,
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
    if let Err(err) = context.validate_process_handles(&[id.to_string()]).await {
        return ToolResult::err_fmt(err.to_string());
    }
    match context.cancel_process(id).await {
        Ok(status) => ToolResult::ok(serde_json::json!({
            "process_id": status.id,
            "terminal": terminal_label(status.terminal.as_ref()),
        })),
        Err(err) => ToolResult::err_fmt(err.to_string()),
    }
}

fn terminal_label(terminal: Option<&lash_core::ProcessTerminalSemantics>) -> &'static str {
    match terminal.map(|terminal| terminal.state) {
        None => "running",
        Some(lash_core::ProcessTerminalState::Completed) => "completed",
        Some(lash_core::ProcessTerminalState::Failed) => "failed",
        Some(lash_core::ProcessTerminalState::Cancelled) => "cancelled",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_definitions_expose_process_control_tools() {
        let names = process_control_tool_definitions(true)
            .into_iter()
            .map(|tool| tool.name)
            .collect::<Vec<_>>();

        assert_eq!(names, vec!["list_process_handles", "cancel_process"]);
    }

    #[test]
    fn cancel_process_definition_is_callable_when_registered() {
        assert_eq!(
            process_cancel_tool_definition().effective_availability(),
            lash_core::ToolAvailability::Callable
        );
    }

    #[test]
    fn plugin_registers_cancel_when_configured_and_omits_it_otherwise() {
        let standard_session =
            lash_core::PluginHost::new(
                std::iter::once(
                    Arc::new(ProcessControlsPluginFactory::new()) as Arc<dyn PluginFactory>
                )
                .chain(lash_core::testing::test_standard_protocol_factories())
                .collect(),
            )
            .build_session("standard", None)
            .expect("standard session");
        let standard_names = standard_session
            .tool_surface("standard")
            .expect("standard surface")
            .tool_names()
            .as_ref()
            .clone();

        let rlm_session = lash_core::PluginHost::new(
            std::iter::once(
                Arc::new(ProcessControlsPluginFactory::without_cancel_process())
                    as Arc<dyn PluginFactory>,
            )
            .chain(lash_core::testing::test_rlm_protocol_factories())
            .collect(),
        )
        .build_session("rlm", None)
        .expect("rlm session");
        let rlm_names = rlm_session
            .tool_surface("rlm")
            .expect("rlm surface")
            .tool_names()
            .as_ref()
            .clone();

        assert!(standard_names.contains(&"list_process_handles".to_string()));
        assert!(standard_names.contains(&"cancel_process".to_string()));
        assert!(rlm_names.contains(&"list_process_handles".to_string()));
        assert!(!rlm_names.contains(&"cancel_process".to_string()));
    }
}
