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
    Capability, CapabilityContext, CapabilityRegistry, CapabilitySpec, DenyList, TierCapability,
    TierExecutionMode, default_registry,
};

use lash::plugin::{PluginError, PluginFactory, PluginSessionContext};
use lash::{PluginSpec, PluginSpecFactory, PromptContribution, SessionPolicy, ToolProvider};

pub use host::{
    AgentMetadata, AgentSummary, CloseAgentRequest, CloseAgentResponse, DeliveryMode,
    FollowupTaskRequest, FollowupTaskResponse, ListAgentsRequest, ListAgentsResponse,
    LocalSubagentHost, SendMessageRequest, SendMessageResponse, SpawnAgentRequest,
    SpawnAgentResponse, SubagentHost, WaitAgentClosed, WaitAgentCompletion, WaitAgentEvent,
    WaitAgentMessage, WaitAgentRequest, WaitAgentResponse, WaitUntil,
    truncate_snapshot_to_recent_turns,
};

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
            }))
        } else {
            None
        };

        let prompt_contributions = if is_rlm {
            rlm::rlm_subagent_prompt_contributions()
        } else {
            Vec::new()
        };
        let continue_as_prompt_contributions = if is_rlm {
            rlm::rlm_continue_as_prompt_contributions()
        } else {
            Vec::new()
        };

        PluginSpecFactory::new(
            "subagents",
            Arc::new(move |ctx| {
                let contributions = subagent_prompt_contributions_for_context(
                    &ctx,
                    &prompt_contributions,
                    &continue_as_prompt_contributions,
                );
                let mut spec = PluginSpec::new()
                    .with_prompt_contributor(Arc::new(move |_ctx| {
                        let contributions = contributions.clone();
                        Box::pin(async move { Ok(contributions) })
                    }))
                    .with_tool_surface_contributor(Arc::new(move |ctx| {
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

fn subagent_prompt_contributions_for_context(
    ctx: &lash::plugin::PluginSessionContext,
    full: &[PromptContribution],
    continue_only: &[PromptContribution],
) -> Vec<PromptContribution> {
    if tool_callable_from_authority(&ctx.tool_access, "spawn_agent") {
        full.to_vec()
    } else if tool_callable_from_authority(&ctx.tool_access, "continue_as") {
        continue_only.to_vec()
    } else {
        Vec::new()
    }
}

fn tool_callable_from_authority(access: &lash::SessionToolAccess, name: &str) -> bool {
    if access.hides(name) {
        return false;
    }
    access.tools.is_empty() || access.tools.iter().any(|tool| tool.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::Mutex;

    use crate::shared::{
        ForkTurns, build_session_policy, build_spawn_create_request, parse_fork_turns,
        parse_output_schema,
    };
    use async_trait::async_trait;
    use lash::PersistedSessionState;
    use lash::plugin::{PluginError, SessionHandle, SessionManager, SessionTurnHandle};
    use lash::session_model::SessionEventRecord;
    use lash::{
        SessionAppendNode, SessionCreateRequest, SessionPluginMode, SessionStartPoint,
        ToolDefinition, ToolExecutionContext, TurnInput,
    };
    use lash_rlm_types::{RlmCreateExtras, RlmModeEvent, RlmTermination};
    use serde_json::{Value, json};

    #[test]
    fn output_schema_supports_scalars_and_lists() {
        let schema = parse_output_schema(Some(&json!({
            "answer": "str",
            "count": "int",
            "items": "list[str]"
        })))
        .expect("schema")
        .expect("present");
        assert_eq!(schema["properties"]["answer"]["type"], json!("string"));
        assert_eq!(schema["properties"]["count"]["type"], json!("integer"));
        assert_eq!(schema["properties"]["items"]["type"], json!("array"));
    }

    #[test]
    fn output_schema_passes_through_lash_type_wrapper() {
        let inner_schema = json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "tags": { "type": "array", "items": { "type": "string" } },
                "status": { "type": "string", "enum": ["ok", "err"] }
            },
            "required": ["name", "tags", "status"],
            "additionalProperties": false
        });
        let wrapped = json!({ lashlang::LASH_TYPE_KEY: inner_schema.clone() });
        let schema = parse_output_schema(Some(&wrapped))
            .expect("schema")
            .expect("present");
        assert_eq!(schema, inner_schema);
    }

    #[test]
    fn output_schema_rejects_lash_type_without_type_field() {
        let wrapped = json!({ lashlang::LASH_TYPE_KEY: {"properties": {}} });
        let err = parse_output_schema(Some(&wrapped)).expect_err("missing type");
        assert!(err.contains("type"), "error: {err}");
    }

    #[test]
    fn output_schema_accepts_array_top_level_type() {
        let wrapped = json!({
            lashlang::LASH_TYPE_KEY: {
                "type": "array",
                "items": {"type": "string"}
            }
        });
        let schema = parse_output_schema(Some(&wrapped))
            .expect("schema")
            .expect("present");
        assert_eq!(schema["type"], json!("array"));
    }

    #[test]
    fn fork_turns_defaults_to_none() {
        assert!(matches!(
            parse_fork_turns(None).expect("fork"),
            ForkTurns::None
        ));
        assert!(matches!(
            parse_fork_turns(Some(&json!("all"))).expect("fork"),
            ForkTurns::All
        ));
        assert!(matches!(
            parse_fork_turns(Some(&json!(3))).expect("fork"),
            ForkTurns::Recent(3)
        ));
    }

    #[derive(Default)]
    struct DirectCompletionManager {
        snapshot: PersistedSessionState,
        requests: Mutex<Vec<(lash::DirectRequest, String)>>,
        response_text: String,
    }

    #[async_trait]
    impl SessionManager for DirectCompletionManager {
        async fn snapshot_current(&self) -> Result<PersistedSessionState, PluginError> {
            Ok(self.snapshot.clone())
        }

        async fn snapshot_session(
            &self,
            _session_id: &str,
        ) -> Result<PersistedSessionState, PluginError> {
            Ok(self.snapshot.clone())
        }

        async fn tool_catalog(
            &self,
            _session_id: &str,
        ) -> Result<Vec<serde_json::Value>, PluginError> {
            Ok(Vec::new())
        }

        async fn create_session(
            &self,
            _request: SessionCreateRequest,
        ) -> Result<SessionHandle, PluginError> {
            Err(PluginError::Session("not used".to_string()))
        }

        async fn close_session(&self, _session_id: &str) -> Result<(), PluginError> {
            Ok(())
        }

        async fn start_turn_stream(
            &self,
            _session_id: &str,
            _input: TurnInput,
        ) -> Result<SessionTurnHandle, PluginError> {
            Err(PluginError::Session("not used".to_string()))
        }

        async fn await_turn(&self, _turn_id: &str) -> Result<lash::AssembledTurn, PluginError> {
            Err(PluginError::Session("not used".to_string()))
        }

        async fn cancel_turn(&self, _turn_id: &str) -> Result<(), PluginError> {
            Ok(())
        }

        async fn direct_completion(
            &self,
            request: lash::DirectRequest,
            usage_source: &str,
        ) -> Result<lash::DirectCompletion, PluginError> {
            self.requests
                .lock()
                .expect("requests")
                .push((request, usage_source.to_string()));
            Ok(lash::DirectCompletion {
                text: self.response_text.clone(),
                usage: lash::TokenUsage::default(),
            })
        }
    }

    #[test]
    fn rlm_definitions_expose_spawn_without_mini_api() {
        let registry = default_registry(&BTreeMap::new(), lash::ExecutionMode::standard());
        let rlm_defs = rlm::rlm_subagent_tool_definitions(&registry.names());

        assert!(rlm_defs.iter().any(|tool| tool.name == "llm_query"));
        assert!(rlm_defs.iter().any(|tool| tool.name == "continue_as"));
        assert!(rlm_defs.iter().any(|tool| tool.name == "spawn_agent"));
        assert!(rlm_defs.iter().all(|tool| !matches!(
            tool.name.as_str(),
            "send_message" | "followup_task" | "wait_agent" | "close_agent" | "list_agents"
        )));

        let rlm_spawn = rlm_defs
            .iter()
            .find(|tool| tool.name == "spawn_agent")
            .expect("rlm spawn_agent");
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
        impl SessionManager for SnapshotManager {
            async fn snapshot_current(&self) -> Result<PersistedSessionState, PluginError> {
                Ok(self.snapshot.clone())
            }

            async fn snapshot_session(
                &self,
                _session_id: &str,
            ) -> Result<PersistedSessionState, PluginError> {
                Ok(self.snapshot.clone())
            }

            async fn tool_catalog(
                &self,
                _session_id: &str,
            ) -> Result<Vec<serde_json::Value>, PluginError> {
                Ok(Vec::new())
            }

            async fn create_session(
                &self,
                _request: SessionCreateRequest,
            ) -> Result<SessionHandle, PluginError> {
                Err(PluginError::Session("not used in test".to_string()))
            }

            async fn close_session(&self, _session_id: &str) -> Result<(), PluginError> {
                Ok(())
            }

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
            build_spawn_create_request(&registry, &context, "explore", ForkTurns::None, None)
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
            ForkTurns::None,
            Some(json!({
                "type": "object",
                "properties": { "ok": { "type": "boolean" } },
                "required": ["ok"]
            })),
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
    async fn llm_query_uses_explore_model_and_direct_completion() {
        fn tiered_provider() -> lash::testing::TestProvider {
            lash::testing::TestProvider::builder()
                .kind("direct-stub")
                .default_model("root-model")
                .default_agent_model(|tier| {
                    (tier == "explore").then(|| lash::AgentModelSelection {
                        model: "explore-model".to_string(),
                        variant: Some("low".to_string()),
                    })
                })
                .complete_error("stub")
                .build()
        }

        let manager = Arc::new(DirectCompletionManager {
            snapshot: PersistedSessionState {
                policy: SessionPolicy {
                    provider: tiered_provider().into_handle(),
                    model: "root-model".to_string(),
                    execution_mode: lash::ExecutionMode::new("rlm"),
                    ..SessionPolicy::default()
                },
                ..PersistedSessionState::default()
            },
            requests: Mutex::new(Vec::new()),
            response_text:
                r#"{"kind":"value","value":{"root_cause":"missing config","confidence":0.8},"error":null}"#
                    .to_string(),
        });
        let provider = rlm::RlmSubagentToolsProvider {
            registry: Arc::new(default_registry(
                &BTreeMap::new(),
                lash::ExecutionMode::new("rlm"),
            )),
            host: Arc::new(LocalSubagentHost::default()),
        };
        let context = ToolExecutionContext {
            session_id: "parent".to_string(),
            host: manager.clone(),
            cancellation_token: None,
            async_task_id: None,
        };

        let result = provider
            .execute_with_context(
                "llm_query",
                &json!({
                    "task": "extract root cause",
                    "inputs": { "log": "failed" },
                    "output": { "root_cause": "str", "confidence": "float" }
                }),
                &context,
            )
            .await;

        assert!(result.success, "{:?}", result.result);
        assert_eq!(result.result["root_cause"], json!("missing config"));
        assert_eq!(result.result["confidence"], json!(0.8));

        let requests = manager.requests.lock().expect("requests");
        assert_eq!(requests.len(), 1);
        let (request, usage_source) = &requests[0];
        assert_eq!(usage_source, "llm_query");
        assert_eq!(request.model, "explore-model");
        assert_eq!(request.model_variant.as_deref(), Some("low"));
        assert!(matches!(
            request.output,
            lash::DirectOutputSpec::JsonSchema(_)
        ));
        let prompt = request
            .messages
            .iter()
            .flat_map(|message| message.parts.iter())
            .filter_map(|part| match part {
                lash::DirectPart::Text(text) => Some(text.as_str()),
                lash::DirectPart::Image(_) => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(prompt.contains("extract root cause"));
        assert!(prompt.contains("\"log\": \"failed\""));
    }

    #[tokio::test]
    async fn llm_query_error_result_fails_tool_call() {
        let manager = Arc::new(DirectCompletionManager {
            snapshot: PersistedSessionState {
                policy: SessionPolicy {
                    execution_mode: lash::ExecutionMode::new("rlm"),
                    ..SessionPolicy::default()
                },
                ..PersistedSessionState::default()
            },
            requests: Mutex::new(Vec::new()),
            response_text: r#"{"kind":"error","value":null,"error":"missing required evidence"}"#
                .to_string(),
        });
        let provider = rlm::RlmSubagentToolsProvider {
            registry: Arc::new(default_registry(
                &BTreeMap::new(),
                lash::ExecutionMode::new("rlm"),
            )),
            host: Arc::new(LocalSubagentHost::default()),
        };
        let context = ToolExecutionContext {
            session_id: "parent".to_string(),
            host: manager,
            cancellation_token: None,
            async_task_id: None,
        };

        let result = provider
            .execute_with_context(
                "llm_query",
                &json!({ "task": "answer from missing evidence" }),
                &context,
            )
            .await;

        assert!(!result.success);
        assert_eq!(result.result, json!("missing required evidence"));
    }

    #[tokio::test]
    async fn continue_as_creates_empty_rlm_successor_with_seed_and_task() {
        #[derive(Default)]
        struct BatonManager {
            snapshot: PersistedSessionState,
            created: Mutex<Vec<SessionCreateRequest>>,
        }

        #[async_trait]
        impl SessionManager for BatonManager {
            async fn snapshot_current(&self) -> Result<PersistedSessionState, PluginError> {
                Ok(self.snapshot.clone())
            }

            async fn snapshot_session(
                &self,
                _session_id: &str,
            ) -> Result<PersistedSessionState, PluginError> {
                Ok(self.snapshot.clone())
            }

            async fn tool_catalog(
                &self,
                _session_id: &str,
            ) -> Result<Vec<serde_json::Value>, PluginError> {
                Ok(Vec::new())
            }

            async fn create_session(
                &self,
                request: SessionCreateRequest,
            ) -> Result<SessionHandle, PluginError> {
                self.created.lock().expect("created").push(request.clone());
                Ok(SessionHandle {
                    session_id: request.session_id.unwrap_or_else(|| "child".to_string()),
                    parent_session_id: request.parent_session_id,
                    policy: request.policy.unwrap_or_default(),
                })
            }

            async fn close_session(&self, _session_id: &str) -> Result<(), PluginError> {
                Ok(())
            }

            async fn start_turn_stream(
                &self,
                _session_id: &str,
                _input: TurnInput,
            ) -> Result<SessionTurnHandle, PluginError> {
                Err(PluginError::Session("not used".to_string()))
            }

            async fn await_turn(&self, _turn_id: &str) -> Result<lash::AssembledTurn, PluginError> {
                Err(PluginError::Session("not used".to_string()))
            }

            async fn cancel_turn(&self, _turn_id: &str) -> Result<(), PluginError> {
                Ok(())
            }
        }

        let manager = Arc::new(BatonManager {
            snapshot: PersistedSessionState {
                policy: SessionPolicy {
                    execution_mode: lash::ExecutionMode::new("rlm"),
                    model: "model".to_string(),
                    max_context_tokens: Some(200_000),
                    standard_context_approach: Some(lash::StandardContextApproach::default()),
                    ..SessionPolicy::default()
                },
                mode_turn_options: lash::ModeTurnOptions::rlm(RlmTermination::Finish {
                    schema: Some(json!({
                        "type": "object",
                        "properties": { "answer": { "type": "string" } },
                        "required": ["answer"]
                    })),
                    include_submit_prompt: true,
                }),
                ..PersistedSessionState::default()
            },
            created: Mutex::new(Vec::new()),
        });
        let registry = Arc::new(default_registry(
            &BTreeMap::new(),
            lash::ExecutionMode::new("rlm"),
        ));
        let provider = rlm::RlmSubagentToolsProvider {
            registry,
            host: Arc::new(LocalSubagentHost::default()),
        };
        let context = ToolExecutionContext {
            session_id: "parent".to_string(),
            host: manager.clone(),
            cancellation_token: None,
            async_task_id: None,
        };

        let result = provider
            .execute_with_context(
                "continue_as",
                &json!({
                    "task": "finish from here",
                    "seed": { "x": 1, "query": "original" }
                }),
                &context,
            )
            .await;

        assert!(result.success, "{:?}", result.result);
        assert!(
            result
                .result
                .get("_continue_as")
                .and_then(Value::as_str)
                .is_some()
        );
        let created = manager.created.lock().expect("created");
        assert_eq!(created.len(), 1);
        let request = &created[0];
        assert!(matches!(request.start, SessionStartPoint::Empty));
        assert_eq!(request.plugin_mode, SessionPluginMode::InheritCurrent);
        assert_eq!(request.parent_session_id.as_deref(), Some("parent"));
        assert_eq!(
            request
                .policy
                .as_ref()
                .and_then(|policy| policy.standard_context_approach.clone()),
            None,
            "continue_as successors run in RLM and must not inherit standard-only context policy"
        );
        assert_eq!(
            request
                .first_turn_input
                .as_ref()
                .map(|message| message.content.as_str()),
            Some("finish from here")
        );
        assert_eq!(request.initial_nodes.len(), 1);
        let SessionAppendNode::Event {
            event: SessionEventRecord::Mode(mode_event),
        } = &request.initial_nodes[0]
        else {
            panic!("expected seed globals event");
        };
        let Some(RlmModeEvent::RlmGlobalsPatch(seed)) = mode_event.rlm_event() else {
            panic!("expected RlmGlobalsPatch");
        };
        assert_eq!(seed.set["x"], json!(1));
        assert_eq!(seed.set["query"], json!("original"));
        let extras = request
            .mode_extras
            .decode::<RlmCreateExtras>(&lash::ExecutionMode::new("rlm"))
            .expect("decode extras")
            .expect("rlm extras");
        assert!(matches!(
            extras.termination,
            RlmTermination::Finish {
                schema: Some(_),
                ..
            }
        ));
    }

    #[tokio::test]
    async fn standard_provider_does_not_expose_continue_as() {
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

    fn plugin_session_context_with_access(access: lash::SessionToolAccess) -> PluginSessionContext {
        PluginSessionContext {
            session_id: "child".to_string(),
            execution_mode: lash::ExecutionMode::new("rlm"),
            standard_context_approach: None,
            tool_access: access,
            subagent: None,
            parent_session_id: Some("parent".to_string()),
        }
    }

    #[test]
    fn rlm_prompt_uses_continue_only_guidance_when_spawn_is_unavailable() {
        let mut hidden_tools = BTreeSet::new();
        hidden_tools.insert("spawn_agent".to_string());
        let ctx = plugin_session_context_with_access(lash::SessionToolAccess {
            tools: vec![dummy_tool("read_file"), dummy_tool("continue_as")],
            hidden_tools,
        });

        let contributions = subagent_prompt_contributions_for_context(
            &ctx,
            &rlm::rlm_subagent_prompt_contributions(),
            &rlm::rlm_continue_as_prompt_contributions(),
        );

        assert_eq!(contributions.len(), 1);
        assert_eq!(contributions[0].title.as_deref(), Some("Exploration"));
        assert!(contributions[0].content.contains("llm_query"));
        assert!(contributions[0].content.contains("`continue_as`"));
        assert!(contributions[0].content.contains("submit <expr>"));
        assert!(!contributions[0].content.contains("subagent"));
        assert!(!contributions[0].content.contains("spawn"));
        assert!(!contributions[0].content.contains("wait_agent"));
    }

    #[test]
    fn rlm_prompt_uses_full_subagent_guidance_when_spawn_is_available() {
        let ctx = plugin_session_context_with_access(lash::SessionToolAccess {
            tools: vec![dummy_tool("spawn_agent"), dummy_tool("continue_as")],
            hidden_tools: BTreeSet::new(),
        });

        let contributions = subagent_prompt_contributions_for_context(
            &ctx,
            &rlm::rlm_subagent_prompt_contributions(),
            &rlm::rlm_continue_as_prompt_contributions(),
        );

        assert_eq!(contributions.len(), 1);
        assert_eq!(
            contributions[0].title.as_deref(),
            Some("Subagents and lightweight LLM calls")
        );
        assert!(contributions[0].content.contains("llm_query"));
        assert!(contributions[0].content.contains("spawn_agent"));
        assert!(contributions[0].content.contains("list_async_handles"));
        assert!(!contributions[0].content.contains("wait_agent"));
    }

    #[test]
    fn rlm_prompt_omits_subagent_guidance_when_spawn_and_continue_are_unavailable() {
        let mut hidden_tools = BTreeSet::new();
        hidden_tools.insert("spawn_agent".to_string());
        hidden_tools.insert("continue_as".to_string());
        let ctx = plugin_session_context_with_access(lash::SessionToolAccess {
            tools: vec![dummy_tool("read_file")],
            hidden_tools,
        });

        let contributions = subagent_prompt_contributions_for_context(
            &ctx,
            &rlm::rlm_subagent_prompt_contributions(),
            &rlm::rlm_continue_as_prompt_contributions(),
        );

        assert!(contributions.is_empty());
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
