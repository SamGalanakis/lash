use async_trait::async_trait;
use lash::session_model::{ModeEvent, SessionEventRecord};
use lash::{
    MessageRole, ModeExtras, PluginMessage, ProgressSender, SessionAppendNode,
    SessionCreateRequest, SessionPluginMode, SessionPolicy, SessionRelation, SessionStartPoint,
    ToolControl, ToolDefinition, ToolExecutionContext, ToolExecutionMode, ToolProvider, ToolResult,
};
use serde_json::{Value, json};

pub(crate) struct RlmControlToolsProvider;

impl RlmControlToolsProvider {
    async fn continue_as(
        &self,
        args: &Value,
        context: &ToolExecutionContext,
    ) -> Result<ContinueAsResult, String> {
        let task = required_string(args, "task")?;
        let seed = match args.get("seed") {
            None | Some(Value::Null) => serde_json::Map::new(),
            Some(Value::Object(map)) => map.clone(),
            Some(_) => return Err("continue_as `seed` must be a record/dict".to_string()),
        };

        let current_snapshot = context
            .host
            .snapshot_session(&context.session_id)
            .await
            .map_err(|err| format!("failed to snapshot current session: {err}"))?;
        let termination = current_snapshot
            .mode_turn_options
            .decode(&lash::ExecutionMode::new("rlm"))
            .ok()
            .flatten()
            .unwrap_or_default();
        let mut policy = current_snapshot.policy.clone();
        policy.execution_mode = lash::ExecutionMode::new("rlm");
        normalize_context_policy(&mut policy);

        let mode_extras = ModeExtras::typed(
            lash::ExecutionMode::new("rlm"),
            lash_rlm_types::RlmCreateExtras { termination },
        )
        .map_err(|err| format!("failed to encode rlm mode extras: {err}"))?;
        let mut request = fresh_successor_request(
            context.session_id.clone(),
            policy,
            mode_extras,
            "continue_as",
        );
        request.plugin_mode = SessionPluginMode::InheritCurrent;
        let successor_session_id = request
            .session_id
            .clone()
            .expect("fresh successor request sets session id");
        request.initial_nodes = rlm_seed_initial_nodes(seed);
        request.first_turn_input = Some(PluginMessage::text(MessageRole::User, task.clone()));
        context
            .host
            .create_session(request)
            .await
            .map_err(|err| format!("failed to create continue_as successor: {err}"))?;

        Ok(ContinueAsResult {
            value: json!({
                "ok": true,
                "session_id": successor_session_id.clone(),
                "task": task,
            }),
            control: ToolControl::Handoff {
                session_id: successor_session_id,
            },
        })
    }
}

#[async_trait]
impl ToolProvider for RlmControlToolsProvider {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![continue_as_tool_definition()]
    }

    async fn execute(&self, name: &str, _args: &Value) -> ToolResult {
        ToolResult::err_fmt(format_args!(
            "`{name}` requires session context and cannot run without it"
        ))
    }

    async fn execute_with_context(
        &self,
        name: &str,
        args: &Value,
        context: &ToolExecutionContext,
    ) -> ToolResult {
        let result = match name {
            "continue_as" => self.continue_as(args, context).await,
            _ => Err(format!("Unknown tool: {name}")),
        };
        finalise_tool_result(result)
    }

    async fn execute_streaming_with_context(
        &self,
        name: &str,
        args: &Value,
        context: &ToolExecutionContext,
        _progress: Option<&ProgressSender>,
    ) -> ToolResult {
        self.execute_with_context(name, args, context).await
    }
}

pub fn continue_as_tool_definition() -> ToolDefinition {
    ToolDefinition::new(
        "continue_as",
        "Tail-call into a fresh RLM successor with a clean window.\n\n- Use when the current trajectory is stale, dominated by failed attempts, or the context budget is tight.\n- Treat `continue_as` as a terminal control action: make it the last meaningful statement in the lashlang block, and do not call `submit` or perform more work after it.\n- `task` packs the concrete goal, constraints, and next steps the successor must act on.\n- `seed` packs the concrete state (paths, facts already learned, partial results) the successor needs in scope; leave bulky raw output behind.",
        continue_as_input_schema(),
        json!({ "type": "object", "additionalProperties": true }),
    )
    .with_examples(vec![
        r#"call continue_as { task: "continue the audit from the summarized findings", seed: { findings: findings } }"#.into(),
    ])
    .with_execution_mode(ToolExecutionMode::Parallel)
}

fn continue_as_input_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Task for the successor session."
            },
            "seed": {
                "type": "object",
                "additionalProperties": true,
                "description": "Optional record/dict of concrete state for the successor."
            }
        },
        "required": ["task"],
        "additionalProperties": false
    })
}

fn fresh_successor_request(
    parent_session_id: String,
    policy: SessionPolicy,
    mode_extras: ModeExtras,
    usage_source: impl Into<String>,
) -> SessionCreateRequest {
    SessionCreateRequest {
        session_id: Some(uuid::Uuid::new_v4().to_string()),
        relation: SessionRelation::Handoff {
            parent_session_id,
            reason: "continue_as".to_string(),
            metadata: serde_json::Map::new(),
        },
        start: SessionStartPoint::Empty,
        policy: Some(policy),
        plugin_mode: SessionPluginMode::Fresh,
        initial_nodes: Vec::new(),
        first_turn_input: None,
        tool_access: lash::SessionToolAccess::default(),
        subagent: None,
        context_surface: lash::SessionContextSurface::default(),
        mode_extras,
        usage_source: Some(usage_source.into()),
    }
}

fn rlm_seed_initial_nodes(seed: serde_json::Map<String, Value>) -> Vec<SessionAppendNode> {
    if seed.is_empty() {
        return Vec::new();
    }
    vec![SessionAppendNode::event(SessionEventRecord::Mode(
        ModeEvent::rlm(lash_rlm_types::RlmModeEvent::RlmGlobalsPatch(
            lash_rlm_types::RlmGlobalsPatchPluginBody { set_default: seed },
        )),
    ))]
}

fn normalize_context_policy(policy: &mut SessionPolicy) {
    if policy.execution_mode != lash::ExecutionMode::standard() {
        policy.standard_context_approach = None;
    }
}

fn required_string(args: &Value, key: &str) -> Result<String, String> {
    args.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| format!("missing required parameter: {key}"))
}

struct ContinueAsResult {
    value: Value,
    control: ToolControl,
}

fn finalise_tool_result(result: Result<ContinueAsResult, String>) -> ToolResult {
    match result {
        Ok(result) => ToolResult::ok(result.value).with_control(result.control),
        Err(err) => ToolResult::err(json!(err)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    use lash::plugin::{
        DirectCompletionHost, DynamicToolHost, MonitorHost, PluginError, PromptHost,
        SessionGraphHost, SessionHandle, SessionLifecycleHost, SessionSnapshotHost,
        SessionTurnHandle, TaskHost, ToolCatalogHost, TraceHost, TurnHost,
    };
    use lash::{PersistedSessionState, TurnInput};
    use lash_rlm_types::{RlmCreateExtras, RlmModeEvent, RlmTermination};

    #[derive(Default)]
    struct BatonManager {
        snapshot: PersistedSessionState,
        created: Mutex<Vec<SessionCreateRequest>>,
    }

    #[async_trait]
    impl SessionSnapshotHost for BatonManager {
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
    impl ToolCatalogHost for BatonManager {
        async fn tool_catalog(
            &self,
            _session_id: &str,
        ) -> Result<Vec<serde_json::Value>, PluginError> {
            Ok(Vec::new())
        }
    }

    impl DynamicToolHost for BatonManager {}

    #[async_trait]
    impl SessionLifecycleHost for BatonManager {
        async fn create_session(
            &self,
            request: SessionCreateRequest,
        ) -> Result<SessionHandle, PluginError> {
            self.created.lock().expect("created").push(request.clone());
            Ok(SessionHandle {
                session_id: request.session_id.unwrap_or_else(|| "child".to_string()),
                parent_session_id: request.relation.parent_session_id().map(ToOwned::to_owned),
                policy: request.policy.unwrap_or_default(),
            })
        }

        async fn close_session(&self, _session_id: &str) -> Result<(), PluginError> {
            Ok(())
        }
    }

    #[async_trait]
    impl TurnHost for BatonManager {
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

    impl TaskHost for BatonManager {}
    impl MonitorHost for BatonManager {}
    impl SessionGraphHost for BatonManager {}
    impl PromptHost for BatonManager {}
    impl DirectCompletionHost for BatonManager {}
    impl TraceHost for BatonManager {}

    #[test]
    fn rlm_control_definitions_include_continue_as_only() {
        let provider = RlmControlToolsProvider;
        let names = provider
            .definitions()
            .into_iter()
            .map(|tool| tool.name)
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["continue_as"]);
    }

    #[tokio::test]
    async fn continue_as_creates_empty_rlm_successor_with_seed_and_task() {
        let mut session_graph = lash::SessionGraph::default();
        session_graph.append_event(SessionEventRecord::Mode(ModeEvent::rlm(
            RlmModeEvent::RlmGlobalsPatch(lash_rlm_types::RlmGlobalsPatchPluginBody {
                set_default: serde_json::Map::from_iter([("diary".to_string(), json!([]))]),
            }),
        )));
        let manager = Arc::new(BatonManager {
            snapshot: PersistedSessionState {
                policy: SessionPolicy {
                    execution_mode: lash::ExecutionMode::new("rlm"),
                    model: "model".to_string(),
                    max_context_tokens: Some(200_000),
                    standard_context_approach: Some(lash::StandardContextApproach::default()),
                    ..SessionPolicy::default()
                },
                mode_turn_options: lash::ModeTurnOptions::typed(
                    lash::ExecutionMode::new("rlm"),
                    RlmTermination::Finish {
                        schema: Some(json!({
                            "type": "object",
                            "properties": { "answer": { "type": "string" } },
                            "required": ["answer"]
                        })),
                        include_submit_prompt: true,
                    },
                )
                .expect("valid rlm turn options"),
                session_graph,
                ..PersistedSessionState::default()
            },
            created: Mutex::new(Vec::new()),
        });
        let provider = RlmControlToolsProvider;
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
                .get("session_id")
                .and_then(Value::as_str)
                .is_some()
        );
        assert!(matches!(result.control, Some(ToolControl::Handoff { .. })));
        let created = manager.created.lock().expect("created");
        assert_eq!(created.len(), 1);
        let request = &created[0];
        assert!(matches!(request.start, SessionStartPoint::Empty));
        assert_eq!(request.plugin_mode, SessionPluginMode::InheritCurrent);
        assert_eq!(request.relation.parent_session_id(), Some("parent"));
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
        assert_eq!(seed.set_default["x"], json!(1));
        assert_eq!(seed.set_default["query"], json!("original"));
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
}
