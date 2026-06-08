use std::future::Future;
use std::sync::Arc;

use crate::runtime::AssembledTurn;
use crate::{
    MessageRole, ProtocolTurnOptions, SessionPolicy, ToolAvailability, ToolDefinition,
    ToolManifest, ToolProvider, ToolResult, TurnInput,
};

pub use lash_sansio::{
    CheckpointKind, PluginMessage, PluginRuntimeEvent, PromptContribution, ToolSurfaceContribution,
    ToolSurfaceOverride,
};

mod actions;
mod error;
mod history;
mod hooks;
mod protocol;
mod registrar;
mod registry;
pub mod runtime_host;
mod runtime_impl;
mod services;
mod session_obj;
mod session_types;
mod snapshot;
mod surface;
mod trigger_registry;

pub(crate) use actions::RegisteredPluginAction;
pub use actions::{
    PluginAction, PluginActionContext, PluginActionDef, PluginActionFailure, PluginActionFuture,
    PluginActionHandler, PluginActionInvokeFuture, PluginActionKind, SessionParam,
    plugin_action_def,
};
pub use error::PluginError;
pub use history::{
    HistoryError, HistoryRewriteMetadata, HistoryRewriter, HistoryState, RewriteContext,
    RewriteTrigger, SessionReadView, TurnContextTransform, TurnTransformContext,
};
pub use hooks::{
    AfterToolCallHook, AfterTurnHook, AssistantResponseHook, AssistantResponseHookContext,
    AssistantResponseTransform, AssistantStreamHook, AssistantStreamHookContext,
    AssistantStreamTransform, BeforeToolCallHook, BeforeTurnHook, CheckpointHook,
    CheckpointHookContext, PluginFuture, PluginLifecycleEvent, PluginLifecycleEventHook,
    PluginLifecycleFuture, PluginSessionTask, PromptContributor, PromptHookContext,
    SessionConfigChangedContext, SessionConfigMutator, SessionStateChangedContext,
    ToolCallHookContext, ToolDiscoveryContributor, ToolResultHookContext,
    ToolResultProjectionContext, ToolResultProjector, ToolSurfaceContributor, TurnHookContext,
    TurnResultHookContext, TurnResultSummary,
};
pub use protocol::{
    AssistantProseProjectorPlugin, CodeExecutorPlugin, PluginOptions, ProtocolBeforeLlmCallContext,
    ProtocolDriverPlugin, ProtocolLlmCallAction, ProtocolRuntimeContext, ProtocolSessionContext,
    ProtocolSessionPlugin,
};
pub use registrar::{
    ExecutionRegistrations, HistoryRegistrations, HostEventRegistrations, OutputRegistrations,
    PluginActionRegistrations, PluginRegistrar, PromptRegistrations, ProtocolRegistrations,
    SessionRegistrations, SurfaceRegistrations, ToolCallRegistrations, ToolRegistrations,
    ToolResultRegistrations, TurnRegistrations,
};
pub(crate) use registrar::{PluginContributions, RegisteredHook};
pub use registry::{
    PluginFactory, PluginSessionContext, PluginSpec, PluginSpecBuilder, PluginSpecFactory,
    SessionPlugin, SessionReadyContext, StaticPluginFactory,
};
pub use runtime_host::{
    AppendSessionNodesRequest, AppendSessionNodesResult, DirectCompletion, DirectLlmCompletion,
    SessionGraphService, SessionLifecycleService, SessionStateService, SessionTurnInput,
    SessionTurnRequest,
};
pub use runtime_impl::{PluginHost, SessionAuthorityContext};
pub(crate) use services::NoopSessionManager;
pub use services::{PersistentRuntimeServices, PluginActionInvokeError, RuntimeServices};
pub use session_obj::PluginSession;
pub use session_types::{
    AgentFrameAssignment, AgentFrameId, AgentFrameReason, AgentFrameRecord, AgentFrameStatus,
    PluginOwned, SessionAppendNode, SessionContextSurface, SessionCreateRequest, SessionHandle,
    SessionPluginSource, SessionRelation, SessionSnapshot, SessionStartPoint, SessionToolAccess,
    SubagentSessionContext,
};
pub(crate) use snapshot::{InMemorySnapshotReader, InMemorySnapshotWriter};
pub use snapshot::{
    PluginSessionSnapshot, PluginSnapshotArtifact, PluginSnapshotEntry, PluginSnapshotMeta,
    SnapshotReader, SnapshotWriter,
};
pub use surface::{
    CheckpointApplication, PluginAbort, PluginDirective, PrepareTurnRequest, ToolDiscoveryContext,
    ToolDiscoveryContribution, ToolDiscoveryToolContribution, ToolSurfaceContext, TurnFinalization,
    TurnPreparation,
};
pub(crate) use surface::{emit_plugin_runtime_events, plugin_runtime_session_events};
pub(crate) use trigger_registry::{trigger_handle_json, validate_target_process};
pub(crate) fn builtin_plugin_factories() -> Vec<Arc<dyn PluginFactory>> {
    // Protocol plugins must be registered by the embedder before calling
    // `PluginHost::build_session`. Unit tests use an in-tree fake to avoid
    // a dev-dep cycle through the protocol crates.
    let factories: Vec<Arc<dyn PluginFactory>> =
        vec![Arc::new(trigger_registry::TriggerResourcePluginFactory)];
    #[cfg(not(test))]
    return factories;

    #[cfg(test)]
    {
        factories
            .into_iter()
            .chain(crate::testing::test_standard_protocol_factories())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};
    use serde_json::json;

    use super::*;
    use crate::{SessionSnapshot, ToolDefinition};

    struct MockToolProvider;

    #[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
    struct TypedEchoArgs {
        value: String,
    }

    #[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
    struct TypedEchoOutput {
        value: String,
        session_id: Option<String>,
    }

    struct TypedEchoOp;

    impl PluginAction for TypedEchoOp {
        const NAME: &'static str = "mock.typed_echo";
        const DESCRIPTION: &'static str = "typed echo";
        const KIND: PluginActionKind = PluginActionKind::Query;
        const SESSION_PARAM: SessionParam = SessionParam::Optional;
        type Args = TypedEchoArgs;
        type Output = TypedEchoOutput;
    }

    #[async_trait::async_trait]
    impl ToolProvider for MockToolProvider {
        fn tool_manifests(&self) -> Vec<ToolManifest> {
            self.tool_definitions()
                .into_iter()
                .map(|tool| tool.manifest())
                .collect()
        }

        fn resolve_contract(&self, name: &str) -> Option<Arc<crate::ToolContract>> {
            self.tool_definitions()
                .into_iter()
                .find(|tool| tool.name() == name)
                .map(|tool| Arc::new(tool.contract()))
        }

        async fn execute(&self, call: crate::ToolCall<'_>) -> ToolResult {
            ToolResult::ok(call.args.clone())
        }
    }

    impl MockToolProvider {
        fn tool_definitions(&self) -> Vec<ToolDefinition> {
            vec![
                ToolDefinition::raw(
                    "tool:mock_tool",
                    "mock_tool",
                    "",
                    json!({
                        "type": "object",
                        "properties": { "value": { "type": "string" } },
                        "required": ["value"],
                        "additionalProperties": false
                    }),
                    json!({ "type": "string" }),
                )
                .with_availability(crate::ToolAvailabilityConfig::callable()),
            ]
        }
    }

    struct MockPluginFactory;

    impl PluginFactory for MockPluginFactory {
        fn id(&self) -> &'static str {
            "mock"
        }

        fn build(&self, ctx: &PluginSessionContext) -> Result<Arc<dyn SessionPlugin>, PluginError> {
            Ok(Arc::new(MockPlugin {
                session_id: ctx.session_id.clone(),
            }))
        }
    }

    struct TriggerResourcePluginFactory;

    impl PluginFactory for TriggerResourcePluginFactory {
        fn id(&self) -> &'static str {
            "trigger_resource"
        }

        fn lashlang_resources(&self) -> lashlang::ResourceCatalog {
            let mut resources = lashlang::ResourceCatalog::new();
            resources
                .add_trigger_source_constructor(
                    ["clock", "Alarm"],
                    lashlang::TypeExpr::Object(vec![lashlang::TypeField {
                        name: "at".into(),
                        ty: lashlang::TypeExpr::Str,
                        optional: false,
                    }]),
                    lashlang::NamedDataType::object(
                        "clock.Tick",
                        vec![lashlang::TypeField {
                            name: "fired_at".into(),
                            ty: lashlang::TypeExpr::Str,
                            optional: false,
                        }],
                    )
                    .expect("valid clock tick type"),
                )
                .expect("valid clock trigger source");
            resources
        }

        fn build(
            &self,
            _ctx: &PluginSessionContext,
        ) -> Result<Arc<dyn SessionPlugin>, PluginError> {
            Ok(Arc::new(TriggerResourcePlugin))
        }
    }

    struct TriggerResourcePlugin;

    impl SessionPlugin for TriggerResourcePlugin {
        fn id(&self) -> &'static str {
            "trigger_resource"
        }

        fn register(&self, _reg: &mut PluginRegistrar) -> Result<(), PluginError> {
            Ok(())
        }
    }

    struct MockPlugin {
        session_id: String,
    }

    use crate::testing::MockSessionManager;

    impl SessionPlugin for MockPlugin {
        fn id(&self) -> &'static str {
            "mock"
        }

        fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
            reg.tools().provider(Arc::new(MockToolProvider))?;
            reg.prompt().contribute(Arc::new(|_ctx| {
                Box::pin(async move {
                    Ok(vec![
                        PromptContribution::guidance("Plugin Prompt", "Structured plugin prompt"),
                        PromptContribution::guidance("Dynamic Note", "dynamic note")
                            .with_priority(1),
                    ])
                })
            }));
            let session_id = self.session_id.clone();
            reg.actions().op(
                PluginActionDef {
                    name: "mock.echo".to_string(),
                    description: "echo".to_string(),
                    kind: PluginActionKind::Query,
                    session_param: SessionParam::Optional,
                    input_schema: json!({}),
                    output_schema: json!({}),
                },
                Arc::new(move |ctx, args| {
                    let session_id = session_id.clone();
                    Box::pin(async move {
                        ToolResult::ok(json!({
                            "session_id": ctx.session_id,
                            "plugin_session_id": session_id,
                            "args": args,
                        }))
                    })
                }),
            )?;
            reg.actions()
                .typed::<TypedEchoOp, _, _>(move |ctx, args| async move {
                    Ok(TypedEchoOutput {
                        value: args.value,
                        session_id: ctx.session_id,
                    })
                })?;
            Ok(())
        }

        fn snapshot(
            &self,
            _writer: &mut dyn SnapshotWriter,
        ) -> Result<PluginSnapshotMeta, PluginError> {
            Ok(PluginSnapshotMeta {
                plugin_id: self.id().to_string(),
                plugin_version: self.version().to_string(),
                revision: self.snapshot_revision(),
                state: Some(json!({"session_id": self.session_id})),
            })
        }
    }

    #[test]
    fn plugin_host_collects_factory_lashlang_resources() {
        let host = PluginHost::new(vec![Arc::new(TriggerResourcePluginFactory)]);

        assert!(
            host.lashlang_resources()
                .resolve_value_constructor(&["clock", "Alarm"])
                .is_some()
        );
        let session = host.build_session("root", None).expect("session");
        assert!(
            session
                .lashlang_resources()
                .resolve_value_constructor(&["clock", "Alarm"])
                .is_some()
        );
    }

    #[test]
    fn declared_host_events_enter_session_lashlang_resources() {
        struct HostEventOnlyFactory;

        impl PluginFactory for HostEventOnlyFactory {
            fn id(&self) -> &'static str {
                "host_event_only"
            }

            fn build(
                &self,
                _ctx: &PluginSessionContext,
            ) -> Result<Arc<dyn SessionPlugin>, PluginError> {
                Ok(Arc::new(HostEventOnlyPlugin))
            }
        }

        struct HostEventOnlyPlugin;

        impl SessionPlugin for HostEventOnlyPlugin {
            fn id(&self) -> &'static str {
                "host_event_only"
            }

            fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
                reg.host_events().declare(crate::HostEvent::new(
                    "Button",
                    "ui.button",
                    "pressed",
                    lashlang::NamedDataType::object(
                        "ui.button.Pressed",
                        vec![lashlang::TypeField {
                            name: "button".into(),
                            ty: lashlang::TypeExpr::Str,
                            optional: false,
                        }],
                    )
                    .expect("valid button payload type"),
                ))
            }
        }

        let host = PluginHost::new(vec![Arc::new(HostEventOnlyFactory)]);

        assert!(
            host.lashlang_resources()
                .resolve_trigger_source("ui.button.pressed")
                .is_none()
        );
        let session = host.build_session("root", None).expect("session");
        assert!(
            session
                .host_events()
                .get("Button", "ui.button", "pressed")
                .is_some()
        );
        assert!(
            session
                .lashlang_resources()
                .resolve_trigger_source("ui.button.pressed")
                .is_some()
        );
        assert!(
            session
                .lashlang_resources()
                .resolve_value_constructor(&["ui", "button", "pressed"])
                .is_some()
        );
    }

    #[tokio::test]
    async fn session_collects_tools_and_prompts() {
        let host = PluginHost::new(vec![Arc::new(MockPluginFactory)]);
        let session = host.build_session("root", None).expect("session");
        let tool_names = session
            .tools()
            .tool_manifests()
            .into_iter()
            .map(|manifest| manifest.name)
            .collect::<std::collections::BTreeSet<_>>();
        assert!(tool_names.contains("mock_tool"));
        assert!(tool_names.contains("batch"));
        let contributions = session
            .collect_prompt_contributions(PromptHookContext {
                session_id: "root".to_string(),
                sessions: Arc::new(MockSessionManager::default()),
                state: SessionReadView::from_snapshot(&SessionSnapshot::default()),
                protocol_turn_options: ProtocolTurnOptions::default(),
                turn_context: crate::TurnContext::default(),
            })
            .await
            .expect("prompt contributions");
        assert_eq!(
            contributions,
            vec![
                PromptContribution::guidance("Plugin Prompt", "Structured plugin prompt"),
                PromptContribution::guidance("Dynamic Note", "dynamic note").with_priority(1),
            ]
        );
    }

    #[tokio::test]
    async fn external_invoke_defaults_to_current_session_when_requested() {
        let host = PluginHost::new(vec![Arc::new(MockPluginFactory)]);
        let session = host.build_session("root", None).expect("session");
        let result = session
            .invoke_plugin_action(
                "mock.echo",
                json!({"ok":true}),
                None,
                true,
                Arc::new(MockSessionManager::default()),
                Arc::new(MockSessionManager::default()),
                Arc::new(MockSessionManager::default()),
                Arc::new(crate::UnavailableProcessService),
            )
            .await
            .expect("invoke");
        assert!(result.is_success());
        assert_eq!(
            result
                .value_for_projection()
                .get("session_id")
                .and_then(|v| v.as_str()),
            Some("root")
        );
    }

    #[tokio::test]
    async fn plugin_action_generates_schema_and_invokes_typed_output() {
        let host = PluginHost::new(vec![Arc::new(MockPluginFactory)]);
        let session = host.build_session("root", None).expect("session");

        let def = session
            .plugin_actions()
            .into_iter()
            .find(|def| def.name == TypedEchoOp::NAME)
            .expect("typed op definition");
        assert_eq!(def.kind, PluginActionKind::Query);
        assert_eq!(def.session_param, SessionParam::Optional);
        let value_type = def
            .input_schema
            .pointer("/schema/properties/value/type")
            .or_else(|| def.input_schema.pointer("/properties/value/type"))
            .and_then(serde_json::Value::as_str);
        assert_eq!(value_type, Some("string"));

        let output = session
            .call_plugin_action::<TypedEchoOp>(
                TypedEchoArgs {
                    value: "hello".to_string(),
                },
                None,
                true,
                Arc::new(MockSessionManager::default()),
                Arc::new(MockSessionManager::default()),
                Arc::new(MockSessionManager::default()),
                Arc::new(crate::UnavailableProcessService),
            )
            .await
            .expect("typed invoke");
        assert_eq!(output.value, "hello");
        assert_eq!(output.session_id.as_deref(), Some("root"));
    }

    #[test]
    fn plugin_action_rejects_duplicate_names() {
        struct DuplicatePlugin;

        impl SessionPlugin for DuplicatePlugin {
            fn id(&self) -> &'static str {
                "duplicate"
            }

            fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
                reg.actions()
                    .typed::<TypedEchoOp, _, _>(move |ctx, args| async move {
                        Ok(TypedEchoOutput {
                            value: args.value,
                            session_id: ctx.session_id,
                        })
                    })?;
                reg.actions()
                    .typed::<TypedEchoOp, _, _>(move |ctx, args| async move {
                        Ok(TypedEchoOutput {
                            value: args.value,
                            session_id: ctx.session_id,
                        })
                    })
            }
        }

        struct DuplicateFactory;
        impl PluginFactory for DuplicateFactory {
            fn id(&self) -> &'static str {
                "duplicate"
            }

            fn build(
                &self,
                _ctx: &PluginSessionContext,
            ) -> Result<Arc<dyn SessionPlugin>, PluginError> {
                Ok(Arc::new(DuplicatePlugin))
            }
        }

        let err =
            match PluginHost::new(vec![Arc::new(DuplicateFactory)]).build_session("root", None) {
                Ok(_) => panic!("duplicate typed plugin action should fail"),
                Err(err) => err,
            };
        assert!(err.to_string().contains("duplicate plugin action name"));
    }

    #[tokio::test]
    async fn typed_external_invoke_errors_on_failed_or_invalid_output() {
        struct BadOp;
        impl PluginAction for BadOp {
            const NAME: &'static str = "mock.echo";
            const DESCRIPTION: &'static str = "bad typed projection over raw op";
            const KIND: PluginActionKind = PluginActionKind::Query;
            const SESSION_PARAM: SessionParam = SessionParam::Optional;
            type Args = TypedEchoArgs;
            type Output = TypedEchoOutput;
        }

        let host = PluginHost::new(vec![Arc::new(MockPluginFactory)]);
        let session = host.build_session("root", None).expect("session");
        let err = session
            .call_plugin_action::<BadOp>(
                TypedEchoArgs {
                    value: "hello".to_string(),
                },
                None,
                true,
                Arc::new(MockSessionManager::default()),
                Arc::new(MockSessionManager::default()),
                Arc::new(MockSessionManager::default()),
                Arc::new(crate::UnavailableProcessService),
            )
            .await
            .expect_err("raw output shape should not match typed output");
        assert!(err.to_string().contains("invalid mock.echo output"));
    }

    #[tokio::test]
    async fn plugin_host_can_invoke_plugin_action_for_registered_session() {
        let host = PluginHost::new(vec![Arc::new(MockPluginFactory)]);
        let _session = host.build_session("root", None).expect("session");

        let result = host
            .invoke_plugin_action_for_session(
                "root",
                "mock.echo",
                json!({"ok":true}),
                Arc::new(MockSessionManager::default()),
                Arc::new(MockSessionManager::default()),
                Arc::new(MockSessionManager::default()),
                Arc::new(crate::UnavailableProcessService),
            )
            .await
            .expect("invoke");
        assert!(result.is_success());
        assert_eq!(
            result
                .value_for_projection()
                .get("session_id")
                .and_then(|v| v.as_str()),
            Some("root")
        );
        assert_eq!(
            result
                .value_for_projection()
                .get("plugin_session_id")
                .and_then(|v| v.as_str()),
            Some("root")
        );
    }

    #[tokio::test]
    async fn plugin_host_can_invoke_plugin_action_for_forked_session() {
        let host = PluginHost::new(vec![Arc::new(MockPluginFactory)]);
        let root = host.build_session("root", None).expect("root");
        let child = root.fork_for_session("child").expect("child");

        let result = host
            .invoke_plugin_action_for_session(
                "child",
                "mock.echo",
                json!({"ok":true}),
                Arc::new(MockSessionManager::default()),
                Arc::new(MockSessionManager::default()),
                Arc::new(MockSessionManager::default()),
                Arc::new(crate::UnavailableProcessService),
            )
            .await
            .expect("invoke");
        assert!(result.is_success());
        assert_eq!(
            result
                .value_for_projection()
                .get("session_id")
                .and_then(|v| v.as_str()),
            Some("child")
        );
        assert_eq!(
            result
                .value_for_projection()
                .get("plugin_session_id")
                .and_then(|v| v.as_str()),
            Some("child")
        );

        drop(child);
    }

    #[test]
    fn plugin_host_unregisters_sessions() {
        let host = PluginHost::new(vec![Arc::new(MockPluginFactory)]);
        let _session = host.build_session("root", None).expect("session");
        assert!(host.session("root").is_ok());
        host.unregister_session("root").expect("unregister");
        match host.session("root") {
            Err(PluginActionInvokeError::UnknownSession(id)) => assert_eq!(id, "root"),
            Ok(_) => panic!("expected missing session"),
            Err(other) => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn snapshot_round_trip_preserves_plugin_entries() {
        let host = PluginHost::new(vec![Arc::new(MockPluginFactory)]);
        let session = host.build_session("root", None).expect("session");
        let snapshot = session.snapshot().expect("snapshot");
        assert!(snapshot.plugins.contains_key("mock"));
        let restored = host
            .build_session("child", Some(&snapshot))
            .expect("restored");
        let restored_snapshot = restored.snapshot().expect("snapshot");
        assert!(restored_snapshot.plugins.contains_key("mock"));
    }

    #[test]
    fn runtime_services_are_backed_by_plugin_sessions() {
        let host = PluginHost::new(vec![Arc::new(StaticPluginFactory::new(
            "mock_tool",
            PluginSpec::new()
                .with_tool_provider(Arc::new(MockToolProvider) as Arc<dyn ToolProvider>),
        ))]);
        let services = RuntimeServices::new(host.build_session("root", None).expect("session"));
        assert_eq!(services.plugins.session_id(), "root");
        assert!(
            services
                .plugins
                .tools()
                .tool_manifests()
                .iter()
                .any(|tool| tool.name == "mock_tool")
        );
    }

    struct ProjectorPluginFactory {
        plugin_id: &'static str,
    }

    impl PluginFactory for ProjectorPluginFactory {
        fn id(&self) -> &'static str {
            self.plugin_id
        }

        fn build(
            &self,
            _ctx: &PluginSessionContext,
        ) -> Result<Arc<dyn SessionPlugin>, PluginError> {
            Ok(Arc::new(ProjectorPlugin {
                plugin_id: self.plugin_id,
            }))
        }
    }

    struct ProjectorPlugin {
        plugin_id: &'static str,
    }

    impl SessionPlugin for ProjectorPlugin {
        fn id(&self) -> &'static str {
            self.plugin_id
        }

        fn register(&self, reg: &mut PluginRegistrar) -> Result<(), PluginError> {
            reg.tool_results().projector(Arc::new(|ctx| {
                Box::pin(async move {
                    Ok(crate::ModelToolReturn::from_output(
                        ctx.call_id,
                        ctx.tool_name,
                        &ctx.output,
                    ))
                })
            }))
        }
    }

    #[test]
    fn duplicate_tool_result_projectors_are_rejected() {
        let host = PluginHost::new(vec![
            Arc::new(ProjectorPluginFactory {
                plugin_id: "projector-a",
            }),
            Arc::new(ProjectorPluginFactory {
                plugin_id: "projector-b",
            }),
        ]);
        let err = match host.build_session("root", None) {
            Ok(_) => panic!("duplicate projector"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("duplicate tool result projector"));
        assert!(err.to_string().contains("projector-a"));
        assert!(err.to_string().contains("projector-b"));
    }
}
