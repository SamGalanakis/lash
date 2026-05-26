//! Mode-agnostic runtime-control tools (`list_process_handles`,
//! `cancel_process`).
//!
//! Dedicated plugins register these tools into the native-tools surface,
//! so mode crates do not own or duplicate runtime control behavior. RLM
//! hides process-control tools when a mode exposes the same control through
//! native process handles.

use std::sync::Arc;

use serde_json::Value;

use lash_core::plugin::{
    ModeNativeToolsPlugin, PluginError, PluginFactory, PluginRegistrar, PluginSessionContext,
    SessionPlugin,
};
use lash_core::tool_dispatch::ToolDispatchContext;
use lash_core::{
    ProgressSender, ToolContract, ToolDefinition, ToolExecutionMode, ToolManifest, ToolResult,
};

/// Exposure policy for mode-agnostic process-control tools.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProcessControlsConfig {
    pub expose_list: bool,
    pub expose_cancel: bool,
    pub include_rlm: bool,
}

impl Default for ProcessControlsConfig {
    fn default() -> Self {
        Self {
            expose_list: true,
            expose_cancel: true,
            include_rlm: false,
        }
    }
}

/// Plugin factory for mode-agnostic process-control tools.
#[derive(Clone, Copy, Debug)]
pub struct ProcessControlsPluginFactory {
    config: ProcessControlsConfig,
}

impl ProcessControlsPluginFactory {
    pub fn new() -> Self {
        Self {
            config: ProcessControlsConfig::default(),
        }
    }

    pub fn with_config(config: ProcessControlsConfig) -> Self {
        Self { config }
    }

    pub fn list_only_for_rlm() -> Self {
        Self::with_config(ProcessControlsConfig {
            expose_list: true,
            expose_cancel: false,
            include_rlm: true,
        })
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

    fn build(&self, ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        Ok(Arc::new(ProcessControlsPlugin {
            config: self.config,
            enabled: self.config.include_rlm
                || ctx.execution_mode != lash_core::ExecutionMode::new("rlm"),
        }))
    }
}

struct ProcessControlsPlugin {
    config: ProcessControlsConfig,
    enabled: bool,
}

impl SessionPlugin for ProcessControlsPlugin {
    fn id(&self) -> &'static str {
        "process_controls"
    }

    fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
        if self.enabled {
            reg.mode()
                .native_tools(Arc::new(ProcessControlsNativeTools {
                    config: self.config,
                }))?;
        }
        Ok(())
    }
}

struct ProcessControlsNativeTools {
    config: ProcessControlsConfig,
}

#[async_trait::async_trait]
impl ModeNativeToolsPlugin for ProcessControlsNativeTools {
    fn tool_manifests(&self) -> Vec<ToolManifest> {
        process_control_tool_definitions(self.config)
            .into_iter()
            .map(|tool| tool.manifest())
            .collect()
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<ToolContract>> {
        process_control_tool_definitions(self.config)
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
            "list_process_handles" if self.config.expose_list => {
                Some(execute_process_list_tool_call(context).await)
            }
            "cancel_process" if self.config.expose_cancel => {
                Some(execute_process_cancel_tool_call(context, args).await)
            }
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
    .with_execution_mode(ToolExecutionMode::Parallel)
}

fn process_control_tool_definitions(config: ProcessControlsConfig) -> Vec<ToolDefinition> {
    let mut tools = Vec::new();
    if config.expose_list {
        tools.push(process_list_tool_definition());
    }
    if config.expose_cancel {
        tools.push(process_cancel_tool_definition());
    }
    tools
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
    .with_execution_mode(ToolExecutionMode::Parallel)
}

pub async fn execute_process_list_tool_call(context: &ToolDispatchContext<'_>) -> ToolResult {
    match context
        .processes
        .list_visible(&context.session_id, context.process_scope())
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
        .processes
        .validate_visible(&context.session_id, &[id.to_string()])
        .await
    {
        return ToolResult::err_fmt(err.to_string());
    }
    match context
        .processes
        .cancel(&context.session_id, id, context.process_scope())
        .await
    {
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
        let names = process_control_tool_definitions(ProcessControlsConfig::default())
            .into_iter()
            .map(|tool| tool.name)
            .collect::<Vec<_>>();

        assert_eq!(names, vec!["list_process_handles", "cancel_process"]);
    }

    #[test]
    fn list_only_for_rlm_exposes_only_canonical_list_tool() {
        let names = process_control_tool_definitions(
            ProcessControlsPluginFactory::list_only_for_rlm().config,
        )
        .into_iter()
        .map(|tool| tool.name)
        .collect::<Vec<_>>();

        assert_eq!(names, vec!["list_process_handles"]);
    }

    #[test]
    fn plugin_registers_controls_for_standard_mode_only() {
        let standard_session =
            lash_core::PluginHost::new(
                std::iter::once(
                    Arc::new(ProcessControlsPluginFactory::new()) as Arc<dyn PluginFactory>
                )
                .chain(lash_core::testing::test_mode_factories())
                .collect(),
            )
            .build_standard_session("standard", None)
            .expect("standard session");
        let standard_names = standard_session
            .tool_surface("standard", lash_core::ExecutionMode::standard())
            .expect("standard surface")
            .tool_names()
            .as_ref()
            .clone();

        let rlm_session =
            lash_core::PluginHost::new(
                std::iter::once(
                    Arc::new(ProcessControlsPluginFactory::new()) as Arc<dyn PluginFactory>
                )
                .chain(lash_core::testing::test_mode_factories())
                .collect(),
            )
            .build_session("rlm", lash_core::ExecutionMode::new("rlm"), None, None)
            .expect("rlm session");
        let rlm_names = rlm_session
            .tool_surface("rlm", lash_core::ExecutionMode::new("rlm"))
            .expect("rlm surface")
            .tool_names()
            .as_ref()
            .clone();

        assert!(standard_names.contains(&"list_process_handles".to_string()));
        assert!(standard_names.contains(&"cancel_process".to_string()));
        assert!(!rlm_names.contains(&"list_process_handles".to_string()));
        assert!(!rlm_names.contains(&"cancel_process".to_string()));
    }

    #[test]
    fn list_only_for_rlm_registers_list_without_cancel_in_rlm() {
        let rlm_session = lash_core::PluginHost::new(
            std::iter::once(Arc::new(ProcessControlsPluginFactory::list_only_for_rlm())
                as Arc<dyn PluginFactory>)
            .chain(lash_core::testing::test_mode_factories())
            .collect(),
        )
        .build_session("rlm", lash_core::ExecutionMode::new("rlm"), None, None)
        .expect("rlm session");
        let rlm_names = rlm_session
            .tool_surface("rlm", lash_core::ExecutionMode::new("rlm"))
            .expect("rlm surface")
            .tool_names()
            .as_ref()
            .clone();

        assert!(rlm_names.contains(&"list_process_handles".to_string()));
        assert!(!rlm_names.contains(&"cancel_process".to_string()));
    }
}
