//! Protocol-stack runtime-control tools (`processes.list`,
//! `processes.cancel`).
//!
//! Dedicated plugins register these tools into the normal tool-provider
//! surface, so protocol crates do not own or duplicate runtime control behavior.

use std::sync::Arc;

use serde_json::Value;

use lash_core::plugin::{
    PluginError, PluginFactory, PluginSessionContext, PluginSpec, SessionPlugin,
    StaticPluginFactory,
};
use lash_core::{
    LashlangToolBinding, ToolAvailabilityConfig, ToolCall, ToolDefinition, ToolProvider,
    ToolResult, ToolScheduling,
};
use lash_tool_support::{StaticToolExecute, StaticToolProvider};

/// Plugin factory for process-control tools.
///
/// Declares its provider through a [`PluginSpec`] driven by
/// [`StaticPluginFactory`], so it does not hand-roll the `SessionPlugin` +
/// `register` ceremony.
pub struct SessionProcessAdminPluginFactory {
    inner: StaticPluginFactory,
}

impl SessionProcessAdminPluginFactory {
    pub fn new() -> Self {
        Self::with_cancel_process(true)
    }

    pub fn without_cancel_process() -> Self {
        Self::with_cancel_process(false)
    }

    fn with_cancel_process(include_cancel_process: bool) -> Self {
        let provider = StaticToolProvider::new(
            processes_tool_definitions(include_cancel_process),
            SessionProcessAdminTools {
                include_cancel_process,
            },
        );
        let spec =
            PluginSpec::new().with_tool_provider(Arc::new(provider) as Arc<dyn ToolProvider>);
        Self {
            inner: StaticPluginFactory::new("processess", spec),
        }
    }
}

impl Default for SessionProcessAdminPluginFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginFactory for SessionProcessAdminPluginFactory {
    fn id(&self) -> &'static str {
        self.inner.id()
    }

    fn build(&self, ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
        self.inner.build(ctx)
    }
}

struct SessionProcessAdminTools {
    include_cancel_process: bool,
}

#[async_trait::async_trait]
impl StaticToolExecute for SessionProcessAdminTools {
    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        match call.name {
            "list_process_handles" => execute_process_list_tool_call(call.context, call.args).await,
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
        "List process runs visible to this session, including `shell.start` runs, with process id, descriptor, optional definition name, and lifecycle status. Filters are optional; the default returns running runs.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["running", "completed", "failed", "cancelled", "any"],
                    "description": "Lifecycle status to list. The default is `running`; `any` includes historical runs."
                },
                "definition": {
                    "type": "object",
                    "description": "A Lashlang process definition value, for example `on_button`."
                }
            },
            "additionalProperties": false
        }),
        process_list_output_schema(),
    )
    .with_examples(vec![
        "await processes.list({})?".into(),
        r#"await processes.list({ status: "any" })?"#.into(),
        "await processes.list({ definition: on_button })?".into(),
    ])
    .with_lashlang_binding(LashlangToolBinding::new(["processes"], "list"))
    .with_availability(ToolAvailabilityConfig::callable())
    .with_scheduling(ToolScheduling::Parallel)
}

fn processes_tool_definitions(include_cancel_process: bool) -> Vec<ToolDefinition> {
    let mut definitions = vec![process_list_tool_definition()];
    if include_cancel_process {
        definitions.push(process_cancel_tool_definition());
    }
    definitions
}

pub fn process_cancel_tool_definition() -> ToolDefinition {
    ToolDefinition::raw(
        "tool:cancel_process",
        "cancel_process",
        "Request cancellation for a durable process, including a running `shell.start` process, by `process_id`.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "process_id": {
                    "type": "string",
                    "description": "Process id returned by a process handle or `processes.list(...)`."
                }
            },
            "required": ["process_id"],
            "additionalProperties": false
        }),
        serde_json::json!({
            "type": "object",
            "properties": {
                "process_id": { "type": "string" },
                "status": {
                    "type": "string",
                    "enum": ["running", "completed", "failed", "cancelled"]
                }
            },
            "required": ["process_id", "status"],
            "additionalProperties": false
        }),
    )
    .with_examples(vec![
        r#"await processes.cancel({ process_id: "tool:call-01JZK7G4QP9Q4J7W3Q2E1H6M9C" })?"#.into(),
        r#"await processes.cancel({ process_id: "subagent:session-01JZK7G4QP9Q4J7W3Q2E1H6M9C" })?"#.into(),
    ])
    .with_lashlang_binding(LashlangToolBinding::new(["processes"], "cancel"))
    .with_availability(ToolAvailabilityConfig::callable())
    .with_scheduling(ToolScheduling::Parallel)
}

pub async fn execute_process_list_tool_call(
    context: &lash_core::ToolContext<'_>,
    args: &Value,
) -> ToolResult {
    let filter = match lash_core::ProcessListFilter::decode(args) {
        Ok(filter) => filter,
        Err(err) => return ToolResult::err_fmt(err),
    };
    let processes = context.processes();
    let result = processes.list_handles_filtered(&filter).await;
    match result {
        Ok(entries) => ToolResult::ok(serde_json::json!(entries)),
        Err(err) => ToolResult::err_fmt(err.to_string()),
    }
}

fn process_list_output_schema() -> Value {
    serde_json::json!({
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "__handle__": {
                    "type": "string",
                    "enum": ["process"],
                    "description": "Handle marker; pass the whole record where a process handle is needed."
                },
                "id": {
                    "type": "string",
                    "description": "Process handle id."
                },
                "process_id": {
                    "type": "string",
                    "description": "Same process id, repeated for tools that ask for process_id."
                },
                "descriptor": {
                    "type": "object",
                    "properties": {
                        "kind": { "type": "string" },
                        "label": { "type": "string" }
                    },
                    "additionalProperties": false
                },
                "definition": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" }
                    },
                    "required": ["name"],
                    "additionalProperties": false
                },
                "status": {
                    "type": "string",
                    "enum": ["running", "completed", "failed", "cancelled"]
                }
            },
            "required": ["__handle__", "id", "process_id", "descriptor", "status"],
            "additionalProperties": false
        }
    })
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
    let processes = context.processes();
    match processes.cancel(id).await {
        Ok(summary) => ToolResult::ok(serde_json::json!(summary)),
        Err(err) => ToolResult::err_fmt(err.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[derive(Default)]
    struct DenyCancelAbility {
        calls: Mutex<Vec<(lash_core::ProcessCancelSource, String)>>,
    }

    impl DenyCancelAbility {
        fn calls(&self) -> Vec<(lash_core::ProcessCancelSource, String)> {
            self.calls.lock().expect("cancel calls").clone()
        }
    }

    #[async_trait::async_trait]
    impl lash_core::ProcessCancelAbility for DenyCancelAbility {
        async fn cancel(
            &self,
            _processes: &dyn lash_core::ProcessService,
            request: lash_core::ProcessCancelRequest<'_>,
        ) -> Result<lash_core::ProcessRecord, PluginError> {
            self.calls
                .lock()
                .expect("cancel calls")
                .push((request.source, request.process_id.to_string()));
            Err(PluginError::Session("denied by host".to_string()))
        }
    }

    fn context_with_cancel_ability(
        ability: Arc<dyn lash_core::ProcessCancelAbility>,
    ) -> lash_core::ToolContext<'static> {
        let manager = Arc::new(lash_core::testing::MockSessionManager::default());
        lash_core::ToolContext::__for_testing_with_process_cancel_ability(
            "session".to_string(),
            manager.clone(),
            manager.clone(),
            manager,
            Arc::new(lash_core::UnavailableProcessService),
            ability,
            Arc::new(lash_core::InMemoryAttachmentStore::new()),
            lash_core::DirectCompletionClient::from_fn(|_, _| {
                Err(PluginError::Session(
                    "direct completions are unavailable in this test context".to_string(),
                ))
            }),
            None,
        )
    }

    #[test]
    fn tool_definitions_expose_processes_tools() {
        let names = processes_tool_definitions(true)
            .into_iter()
            .map(|tool| tool.name().to_string())
            .collect::<Vec<_>>();

        assert_eq!(names, vec!["list_process_handles", "cancel_process"]);
    }

    #[test]
    fn cancel_process_definition_is_callable_when_registered() {
        let definition = process_cancel_tool_definition();
        assert_eq!(
            definition.effective_availability(),
            lash_core::ToolAvailability::Callable
        );
        let rendered = definition.compact_contract().render_signature();
        assert!(rendered.contains("status: enum["), "{rendered}");
        assert!(!rendered.contains("terminal:"), "{rendered}");
    }

    #[test]
    fn list_process_contract_returns_handle_array() {
        let definition = process_list_tool_definition();

        assert_eq!(
            definition.contract.output_schema["type"],
            serde_json::json!("array")
        );
        let rendered = definition.compact_contract().render_signature();
        assert!(rendered.contains("-> list[record{"), "{rendered}");
        assert!(rendered.contains("__handle__"), "{rendered}");
        assert!(rendered.contains("process_id"), "{rendered}");
        assert!(rendered.contains("definition"), "{rendered}");
        assert!(rendered.contains("status: enum["), "{rendered}");
        assert!(rendered.contains("status?: enum["), "{rendered}");
        assert!(rendered.contains("definition?: record"), "{rendered}");
        assert!(!rendered.contains("history"), "{rendered}");
        assert!(!rendered.contains("terminal:"), "{rendered}");
    }

    #[test]
    fn plugin_registers_cancel_when_configured_and_omits_it_otherwise() {
        let standard_session = lash_core::PluginHost::new(
            std::iter::once(
                Arc::new(SessionProcessAdminPluginFactory::new()) as Arc<dyn PluginFactory>
            )
            .chain(lash_core::testing::test_standard_protocol_factories())
            .collect(),
        )
        .build_session("standard", None)
        .expect("standard session");
        let standard_names = standard_session
            .resolved_tool_catalog("standard")
            .expect("standard tool catalog")
            .tool_names()
            .as_ref()
            .clone();

        let rlm_session = lash_core::PluginHost::new(
            std::iter::once(
                Arc::new(SessionProcessAdminPluginFactory::without_cancel_process())
                    as Arc<dyn PluginFactory>,
            )
            .chain(lash_core::testing::test_rlm_protocol_factories())
            .collect(),
        )
        .build_session("rlm", None)
        .expect("rlm session");
        let rlm_names = rlm_session
            .resolved_tool_catalog("rlm")
            .expect("rlm tool catalog")
            .tool_names()
            .as_ref()
            .clone();

        assert!(standard_names.contains(&"list_process_handles".to_string()));
        assert!(standard_names.contains(&"cancel_process".to_string()));
        assert!(rlm_names.contains(&"list_process_handles".to_string()));
        assert!(!rlm_names.contains(&"cancel_process".to_string()));
    }

    #[tokio::test]
    async fn cancel_process_tool_uses_host_cancel_ability() {
        let ability = Arc::new(DenyCancelAbility::default());
        let context = context_with_cancel_ability(ability.clone());

        let result = execute_process_cancel_tool_call(
            &context,
            &serde_json::json!({ "process_id": "process-1" }),
        )
        .await;

        assert!(!result.is_success());
        assert_eq!(
            result.value_for_projection(),
            serde_json::json!("plugin session error: denied by host")
        );
        assert_eq!(
            ability.calls(),
            vec![(
                lash_core::ProcessCancelSource::Tool,
                "process-1".to_string()
            )]
        );
    }
}
