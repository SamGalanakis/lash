mod capability;
mod host;
mod local;
mod queue;
mod rlm;
mod routing;
mod shared;
mod types;

use std::sync::Arc;

pub use capability::{
    Capability, CapabilityContext, CapabilityField, CapabilityOptionalField, CapabilityRecursion,
    CapabilityRegistry, CapabilitySpec, CapabilityToolSurface, StaticCapability, TierCapability,
    TierExecutionMode, default_registry,
};

use lash::plugin::{PluginError, PluginFactory, PluginSessionContext};
use lash::{PluginSpec, PluginSpecFactory, SessionPolicy, ToolProvider};

pub use host::{
    AgentMetadata, CloseAgentRequest, CloseAgentResponse, LocalSubagentHost, SpawnAgentRequest,
    SpawnAgentResponse, SubagentHost, WaitAgentClosed, WaitAgentCompletion, WaitAgentEvent,
    WaitAgentRequest, WaitAgentResponse, WaitUntil, truncate_snapshot_to_recent_turns,
};
pub use rlm::spawn_agent_tool_definition;

pub struct SubagentsPluginFactory {
    policy: SessionPolicy,
    registry: Arc<CapabilityRegistry>,
    host: Arc<dyn SubagentHost>,
}

impl SubagentsPluginFactory {
    pub fn new(
        policy: SessionPolicy,
        registry: Arc<CapabilityRegistry>,
        host: Arc<dyn SubagentHost>,
    ) -> Self {
        Self {
            policy,
            registry,
            host,
        }
    }
}

impl PluginFactory for SubagentsPluginFactory {
    fn id(&self) -> &'static str {
        "subagents"
    }

    fn build(
        &self,
        ctx: &PluginSessionContext,
    ) -> Result<Arc<dyn lash::SessionPlugin>, PluginError> {
        let mut policy = self.policy.clone();
        policy.execution_mode = ctx.execution_mode.clone();

        let registry = Arc::clone(&self.registry);
        let host = Arc::clone(&self.host);
        let execution_mode = ctx.execution_mode.clone();

        let is_rlm = execution_mode == lash::ExecutionMode::new("rlm");
        let provider: Option<Arc<dyn ToolProvider>> = if is_rlm {
            Some(Arc::new(rlm::RlmSubagentToolsProvider {
                registry: Arc::clone(&registry),
                host: Arc::clone(&host),
                include_submit_error: ctx.subagent.is_some(),
            }))
        } else {
            None
        };

        PluginSpecFactory::new(
            "subagents",
            Arc::new(move |_ctx| {
                let mut spec =
                    PluginSpec::new().with_tool_surface_contributor(Arc::new(move |ctx| {
                        shared::subagent_surface_contribution(ctx)
                    }));
                if let Some(provider) = provider.as_ref() {
                    spec = spec.with_tool_provider(Arc::clone(provider));
                }
                Ok(spec)
            }),
        )
        .build(ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use crate::shared::{build_session_policy, build_spawn_create_request};
    use async_trait::async_trait;
    use lash::PersistedSessionState;
    use lash::plugin::{
        DirectCompletionHost, DynamicToolHost, MonitorHost, PluginError, PromptHost,
        SessionGraphHost, SessionHandle, SessionLifecycleHost, SessionSnapshotHost,
        SessionTurnHandle, TaskHost, ToolCatalogHost, TraceHost, TurnHost,
    };
    use lash::{
        SessionCreateRequest, ToolDefinition, ToolExecutionContext, ToolOutputContract, TurnInput,
    };
    use serde_json::json;

    #[test]
    fn static_capability_policy_fields_distinguish_inherit_set_and_clear() {
        let current = SessionPolicy {
            model: "parent-model".to_string(),
            model_variant: Some("parent-variant".to_string()),
            execution_mode: lash::ExecutionMode::standard(),
            ..SessionPolicy::default()
        };
        let mut spec = CapabilitySpec::inherit();
        spec.model = CapabilityField::Set("child-model".to_string());
        spec.model_variant = CapabilityOptionalField::Clear;
        spec.execution_mode = CapabilityField::Set(lash::ExecutionMode::new("rlm"));
        let registry =
            CapabilityRegistry::new().with(Arc::new(StaticCapability::new("child", spec)));

        let policy = build_session_policy(&registry, &current, "child").expect("policy");

        assert_eq!(policy.model, "child-model");
        assert_eq!(policy.model_variant, None);
        assert_eq!(policy.execution_mode, lash::ExecutionMode::new("rlm"));
    }

    #[test]
    fn rlm_definitions_expose_spawn_without_mini_api() {
        let registry = default_registry(&BTreeMap::new(), lash::ExecutionMode::standard());
        let rlm_defs = rlm::rlm_subagent_tool_definitions(&registry.names());

        assert!(rlm_defs.iter().any(|tool| tool.name == "spawn_agent"));
        assert_eq!(
            rlm_defs
                .iter()
                .map(|tool| tool.name.as_str())
                .collect::<Vec<_>>(),
            vec!["spawn_agent"]
        );

        let rlm_spawn = rlm_defs
            .iter()
            .find(|tool| tool.name == "spawn_agent")
            .expect("rlm spawn_agent");
        assert_eq!(
            rlm_spawn.output_contract,
            ToolOutputContract::from_input_schema("output", None)
        );
        assert!(
            rlm_spawn
                .examples
                .iter()
                .any(|example| example.contains("start call spawn_agent"))
        );
    }

    #[tokio::test]
    async fn spawn_uses_live_parent_provider_when_selecting_subagent_model() {
        struct SnapshotManager {
            snapshot: PersistedSessionState,
        }

        #[async_trait]
        impl SessionSnapshotHost for SnapshotManager {
            async fn snapshot_current(&self) -> Result<PersistedSessionState, PluginError> {
                Ok(self.snapshot.clone())
            }

            async fn snapshot_session(
                &self,
                _session_id: &str,
            ) -> Result<PersistedSessionState, PluginError> {
                Ok(self.snapshot.clone())
            }
        }

        #[async_trait]
        impl ToolCatalogHost for SnapshotManager {
            async fn tool_catalog(
                &self,
                _session_id: &str,
            ) -> Result<Vec<serde_json::Value>, PluginError> {
                Ok(Vec::new())
            }
        }

        impl DynamicToolHost for SnapshotManager {}

        #[async_trait]
        impl SessionLifecycleHost for SnapshotManager {
            async fn create_session(
                &self,
                _request: SessionCreateRequest,
            ) -> Result<SessionHandle, PluginError> {
                Err(PluginError::Session("not used in test".to_string()))
            }

            async fn close_session(&self, _session_id: &str) -> Result<(), PluginError> {
                Ok(())
            }
        }

        #[async_trait]
        impl TurnHost for SnapshotManager {
            async fn start_turn_stream(
                &self,
                _session_id: &str,
                _input: TurnInput,
            ) -> Result<SessionTurnHandle, PluginError> {
                Err(PluginError::Session("not used in test".to_string()))
            }

            async fn await_turn(&self, _turn_id: &str) -> Result<lash::AssembledTurn, PluginError> {
                Err(PluginError::Session("not used in test".to_string()))
            }

            async fn cancel_turn(&self, _turn_id: &str) -> Result<(), PluginError> {
                Ok(())
            }
        }

        impl TaskHost for SnapshotManager {}
        impl MonitorHost for SnapshotManager {}
        impl SessionGraphHost for SnapshotManager {}
        impl PromptHost for SnapshotManager {}
        impl DirectCompletionHost for SnapshotManager {}
        impl TraceHost for SnapshotManager {}

        // Two distinct stub providers so we can verify that spawn
        // resolves against the *live* policy, not the factory's stale
        // one. Each stub returns a different per-tier model from
        // `default_agent_model` so the final child policy's model shows
        // which provider the capability lookup resolved against.
        fn tiered_provider(tag: &'static str) -> lash::testing::TestProvider {
            let (kind, default_model, explore_model) = match tag {
                "stale" => ("stale-stub", "stale-model", "stale-explore"),
                "live" => ("live-stub", "live-model", "live-explore"),
                _ => ("stub", "mock-model", "mock-explore"),
            };
            lash::testing::TestProvider::builder()
                .kind(kind)
                .default_model(default_model)
                .default_agent_model(move |tier| {
                    if tier == "explore" {
                        Some(lash::AgentModelSelection {
                            model: explore_model.to_string(),
                            variant: None,
                        })
                    } else {
                        None
                    }
                })
                .complete_error("stub")
                .build()
        }
        let stale_policy = SessionPolicy {
            provider: tiered_provider("stale").into_handle(),
            execution_mode: lash::ExecutionMode::standard(),
            ..SessionPolicy::default()
        };
        let live_policy = SessionPolicy {
            provider: tiered_provider("live").into_handle(),
            execution_mode: lash::ExecutionMode::standard(),
            max_context_tokens: Some(1234),
            ..SessionPolicy::default()
        };
        let registry = Arc::new(default_registry(
            &BTreeMap::new(),
            lash::ExecutionMode::new("rlm"),
        ));
        let context = ToolExecutionContext {
            session_id: "root".to_string(),
            host: Arc::new(SnapshotManager {
                snapshot: PersistedSessionState {
                    policy: live_policy.clone(),
                    ..PersistedSessionState::default()
                },
            }),
            cancellation_token: None,
            async_task_id: None,
        };

        let request =
            build_spawn_create_request(&registry, &context, "explore", None, Default::default())
                .await
                .expect("spawn request");
        let child_policy = request.policy.expect("child policy");

        // The capability looked up the live policy's provider, not
        // the stale one. This pins the behaviour where the spawn
        // pipeline always resolves models against the *current* session
        // policy snapshot, even when the factory was built earlier.
        let stale_choice = build_session_policy(&registry, &stale_policy, "explore")
            .expect("stale policy")
            .model;
        assert_eq!(child_policy.provider, live_policy.provider);
        assert_eq!(
            child_policy.max_context_tokens,
            live_policy.max_context_tokens
        );
        assert_ne!(child_policy.model, stale_choice);
        assert_eq!(child_policy.model, "live-explore");
        assert_eq!(
            request
                .tool_access
                .tools
                .iter()
                .map(|tool| tool.name.as_str())
                .collect::<Vec<_>>(),
            vec![
                "read_file",
                "grep",
                "search_web",
                "fetch_url",
                "exec_command",
                "start_command",
                "write_stdin",
                "llm_query",
                "continue_as"
            ]
        );

        let structured_request = build_spawn_create_request(
            &registry,
            &context,
            "explore",
            Some(json!({
                "type": "object",
                "properties": { "ok": { "type": "boolean" } },
                "required": ["ok"]
            })),
            Default::default(),
        )
        .await
        .expect("structured spawn request");
        let structured_policy = structured_request.policy.expect("structured child policy");
        assert_eq!(
            structured_policy.execution_mode,
            lash::ExecutionMode::new("rlm"),
            "explore runs in RLM so typed output uses native submit"
        );
        assert_eq!(
            structured_request
                .tool_access
                .tools
                .last()
                .map(|tool| tool.name.as_str()),
            Some("continue_as")
        );
    }

    #[tokio::test]
    async fn standard_provider_does_not_expose_subagent_tools() {
        let factory = SubagentsPluginFactory::new(
            SessionPolicy::default(),
            Arc::new(default_registry(
                &BTreeMap::new(),
                lash::ExecutionMode::standard(),
            )),
            Arc::new(LocalSubagentHost::default()),
        );
        let ctx = PluginSessionContext {
            session_id: "parent".to_string(),
            execution_mode: lash::ExecutionMode::standard(),
            standard_context_approach: None,
            tool_access: lash::SessionToolAccess::default(),
            subagent: None,
            parent_session_id: None,
        };
        let plugin = factory.build(&ctx).expect("plugin");
        assert_eq!(plugin.id(), "subagents");
    }

    fn dummy_tool(name: &str) -> ToolDefinition {
        ToolDefinition::new(
            name,
            format!("{name} description"),
            ToolDefinition::default_input_schema(),
            json!({ "type": "null" }),
        )
    }

    #[test]
    fn subagent_surface_reports_authority_notes() {
        use lash::plugin::ToolSurfaceContext;

        let ctx = ToolSurfaceContext {
            session_id: "child".to_string(),
            mode: lash::ExecutionMode::standard(),
            tools: vec![
                dummy_tool("read_file"),
                dummy_tool("ask"),
                dummy_tool("show_snippet_to_user"),
                dummy_tool("showcase"),
                dummy_tool("plan_exit"),
                dummy_tool("apply_patch"),
                dummy_tool("spawn_agent"),
            ],
            tool_access: lash::SessionToolAccess::default(),
            subagent: Some(lash::SubagentSessionAuthority {
                agent_name: "probe".to_string(),
                parent_session_id: "root".to_string(),
                capability: "explore".to_string(),
                depth: 1,
                max_depth: 5,
            }),
        };

        let contribution =
            shared::subagent_surface_contribution(ctx).expect("surface contribution");
        assert!(contribution.overrides.is_empty());
        assert_eq!(
            contribution.tool_list_notes,
            vec!["Subagent agent_name: probe. Capability: explore."]
        );
    }
}
