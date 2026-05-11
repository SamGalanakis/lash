use async_trait::async_trait;
use lash_core::session_model::SessionEventRecord;
use lash_core::{
    MessageRole, ModeExtras, PluginMessage, SessionAppendNode, SessionCreateRequest,
    SessionLifecycleHost, SessionPluginMode, SessionPolicy, SessionRelation, SessionSnapshotHost,
    SessionStartPoint, ToolCall, ToolContext, ToolControl, ToolDefinition, ToolExecutionMode,
    ToolProvider, ToolResult,
};
use serde_json::{Value, json};

pub(crate) struct RlmControlToolsProvider;

impl RlmControlToolsProvider {
    async fn continue_as(
        &self,
        args: &Value,
        context: &ToolContext,
    ) -> Result<ContinueAsResult, String> {
        let task = required_string(args, "task")?;
        // Projected entries (sourced from the parent's host bindings) get
        // re-projected on the successor; global entries land as RLM globals
        // via the existing `RlmGlobalsPatch` plumbing.
        let seed =
            lash_rlm_types::classify_seed(args).map_err(|err| format!("continue_as {err}"))?;

        let successor_session_id = create_continue_as_successor(
            context.host().as_ref(),
            context.session_id(),
            task.clone(),
            seed,
            ContinueAsHandoff {
                relation_reason: "continue_as".to_string(),
                usage_source: "continue_as".to_string(),
                metadata: serde_json::Map::new(),
            },
        )
        .await?;

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

    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        let result = match call.name {
            "continue_as" => self.continue_as(call.args, call.context).await,
            _ => Err(format!("Unknown tool: {}", call.name)),
        };
        finalise_tool_result(result)
    }
}

pub fn continue_as_tool_definition() -> ToolDefinition {
    ToolDefinition::raw(
        "continue_as",
        "Tail-call into a fresh RLM successor with a clean window.\n\nThe successor inherits **nothing** automatically — no globals, no projected bindings, no message history. Pass everything it needs via `seed: { name: value, ... }`. Each entry's kind is preserved: if the value's lashlang source root is a host-projected binding (e.g. `seed: { problem: input.prompt }`), it stays projected on the successor (read-only `Host Projected Variables`); other sources land as regular RLM globals. Computed expressions default to global.\n\n- Use when the current trajectory is stale, dominated by failed attempts, or the context budget is tight.\n- Treat `continue_as` as a terminal control action: make it the last meaningful statement in the lashlang block, and do not call `submit` or perform more work after it.\n- `task` packs the concrete goal, constraints, and next steps the successor must act on.\n- `seed` packs the concrete state (paths, facts already learned, partial results, projected sources) the successor needs in scope; leave bulky raw output behind.",
        continue_as_input_schema(),
        json!({ "type": "object", "additionalProperties": true }),
    )
    .with_examples(vec![
        r#"call continue_as { task: "continue the audit from the summarized findings", seed: { problem: input.prompt, findings: findings } }"#.into(),
    ])
    .with_execution_mode(ToolExecutionMode::Parallel)
}

pub fn continue_as_input_schema() -> Value {
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

pub(crate) struct ContinueAsHandoff {
    pub relation_reason: String,
    pub usage_source: String,
    pub metadata: serde_json::Map<String, Value>,
}

pub(crate) async fn create_continue_as_successor<H>(
    host: &H,
    parent_session_id: &str,
    task: String,
    seed: lash_rlm_types::ClassifiedSeed,
    handoff: ContinueAsHandoff,
) -> Result<String, String>
where
    H: SessionSnapshotHost + SessionLifecycleHost + ?Sized,
{
    let current_snapshot = host
        .snapshot_session(parent_session_id)
        .await
        .map_err(|err| format!("failed to snapshot current session: {err}"))?;
    let termination = current_snapshot
        .mode_turn_options
        .decode(&lash_core::ExecutionMode::new("rlm"))
        .ok()
        .flatten()
        .unwrap_or_default();
    let mut policy = current_snapshot.policy.clone();
    policy.execution_mode = lash_core::ExecutionMode::new("rlm");
    normalize_context_policy(&mut policy);

    let mode_extras = ModeExtras::typed(
        lash_core::ExecutionMode::new("rlm"),
        lash_rlm_types::RlmCreateExtras {
            termination,
            projected_seed: (!seed.projected.is_empty()).then(|| seed.projected.clone()),
        },
    )
    .map_err(|err| format!("failed to encode rlm mode extras: {err}"))?;
    let mut request = fresh_successor_request(
        parent_session_id.to_string(),
        policy,
        mode_extras,
        handoff.relation_reason,
        handoff.metadata,
        handoff.usage_source,
    );
    request.plugin_mode = SessionPluginMode::InheritCurrent;
    let successor_session_id = request
        .session_id
        .clone()
        .expect("fresh successor request sets session id");
    request.initial_nodes = rlm_seed_initial_nodes(seed.globals);
    request.first_turn_input = Some(PluginMessage::text(MessageRole::User, task));
    host.create_session(request)
        .await
        .map_err(|err| format!("failed to create continue_as successor: {err}"))?;
    Ok(successor_session_id)
}

fn fresh_successor_request(
    parent_session_id: String,
    policy: SessionPolicy,
    mode_extras: ModeExtras,
    relation_reason: String,
    metadata: serde_json::Map<String, Value>,
    usage_source: impl Into<String>,
) -> SessionCreateRequest {
    SessionCreateRequest {
        session_id: Some(uuid::Uuid::new_v4().to_string()),
        relation: SessionRelation::Handoff {
            parent_session_id,
            reason: relation_reason,
            metadata,
        },
        start: SessionStartPoint::Empty,
        policy: Some(policy),
        plugin_mode: SessionPluginMode::Fresh,
        initial_nodes: Vec::new(),
        first_turn_input: None,
        tool_access: lash_core::SessionToolAccess::default(),
        subagent: None,
        context_surface: lash_core::SessionContextSurface::default(),
        mode_extras,
        usage_source: Some(usage_source.into()),
    }
}

/// Build the `initial_nodes` payload that seeds RLM globals on a new session.
///
/// Used by `continue_as` for its own seed split, and re-exported for
/// `spawn_agent` (lash-subagents) so both tools share the same RLM-mode patch
/// emission. Each entry of `seed` becomes a `let <name> = <value>` global on
/// the child via `RlmGlobalsPatch`.
pub fn rlm_seed_initial_nodes(seed: serde_json::Map<String, Value>) -> Vec<SessionAppendNode> {
    if seed.is_empty() {
        return Vec::new();
    }
    vec![SessionAppendNode::event(SessionEventRecord::Mode(
        crate::rlm_mode_event(lash_rlm_types::RlmModeEvent::RlmGlobalsPatch(
            lash_rlm_types::RlmGlobalsPatchPluginBody { set_default: seed },
        )),
    ))]
}

fn normalize_context_policy(policy: &mut SessionPolicy) {
    if policy.execution_mode != lash_core::ExecutionMode::standard() {
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

    use lash_core::plugin::{
        DirectCompletionHost, MonitorHost, PluginError, SessionGraphHost, SessionHandle,
        SessionLifecycleHost, SessionSnapshotHost, SessionTurnHandle, TaskHost, ToolCatalogHost,
        ToolStateHost, TraceHost, TurnHost,
    };
    use lash_core::{PersistedSessionState, TurnInput};
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

    impl ToolStateHost for BatonManager {}

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

        async fn await_turn(
            &self,
            _turn_id: &str,
        ) -> Result<lash_core::AssembledTurn, PluginError> {
            Err(PluginError::Session("not used".to_string()))
        }

        async fn cancel_turn(&self, _turn_id: &str) -> Result<(), PluginError> {
            Ok(())
        }
    }

    impl TaskHost for BatonManager {}
    impl MonitorHost for BatonManager {}
    impl SessionGraphHost for BatonManager {}
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
        let mut session_graph = lash_core::SessionGraph::default();
        session_graph.append_event(SessionEventRecord::Mode(crate::rlm_mode_event(
            RlmModeEvent::RlmGlobalsPatch(lash_rlm_types::RlmGlobalsPatchPluginBody {
                set_default: serde_json::Map::from_iter([("diary".to_string(), json!([]))]),
            }),
        )));
        let manager = Arc::new(BatonManager {
            snapshot: PersistedSessionState {
                policy: SessionPolicy {
                    execution_mode: lash_core::ExecutionMode::new("rlm"),
                    model: "model".to_string(),
                    max_context_tokens: Some(200_000),
                    standard_context_approach: Some(lash_core::StandardContextApproach::default()),
                    ..SessionPolicy::default()
                },
                mode_turn_options: lash_core::ModeTurnOptions::typed(
                    lash_core::ExecutionMode::new("rlm"),
                    RlmTermination::SubmitRequired {
                        schema: Some(json!({
                            "type": "object",
                            "properties": { "answer": { "type": "string" } },
                            "required": ["answer"]
                        })),
                    },
                )
                .expect("valid rlm turn options"),
                session_graph,
                ..PersistedSessionState::default()
            },
            created: Mutex::new(Vec::new()),
        });
        let provider = RlmControlToolsProvider;
        let context = lash_core::testing::mock_tool_context_with_host(manager.clone());

        let args = json!({
            "task": "finish from here",
            "seed": { "x": 1, "query": "original" }
        });
        let result = provider
            .execute(ToolCall {
                name: "continue_as",
                args: &args,
                context: &context,
                progress: None,
            })
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
        assert_eq!(request.relation.parent_session_id(), Some("test-session"));
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
        let Some(RlmModeEvent::RlmGlobalsPatch(seed)) = crate::decode_rlm_mode_event(mode_event)
        else {
            panic!("expected RlmGlobalsPatch");
        };
        assert_eq!(seed.set_default["x"], json!(1));
        assert_eq!(seed.set_default["query"], json!("original"));
        let extras = request
            .mode_extras
            .decode::<RlmCreateExtras>(&lash_core::ExecutionMode::new("rlm"))
            .expect("decode extras")
            .expect("rlm extras");
        assert!(matches!(
            extras.termination,
            RlmTermination::SubmitRequired { schema: Some(_) }
        ));
        assert!(
            extras.projected_seed.is_none(),
            "no projected entries → projected_seed should be absent"
        );
    }

    #[tokio::test]
    async fn continue_as_routes_projected_entries_to_mode_extras_and_globals_to_initial_nodes() {
        // Mixed seed: `proj` was a projected source on the parent (encoded with
        // the canonical `__projected__` JSON wrapper), `glob` was a regular
        // global. The successor must receive `proj` as a projected binding via
        // mode_extras.projected_seed and `glob` via the RlmGlobalsPatch event.
        let manager = Arc::new(BatonManager {
            snapshot: PersistedSessionState {
                policy: SessionPolicy {
                    execution_mode: lash_core::ExecutionMode::new("rlm"),
                    model: "model".to_string(),
                    max_context_tokens: Some(200_000),
                    ..SessionPolicy::default()
                },
                ..PersistedSessionState::default()
            },
            created: Mutex::new(Vec::new()),
        });
        let provider = RlmControlToolsProvider;
        let context = lash_core::testing::mock_tool_context_with_host(manager.clone());

        let args = json!({
            "task": "finish from here",
            "seed": {
                "proj": { "__projected__": "carry-over" },
                "glob": 7
            }
        });
        let result = provider
            .execute(ToolCall {
                name: "continue_as",
                args: &args,
                context: &context,
                progress: None,
            })
            .await;
        assert!(result.success, "{:?}", result.result);

        let created = manager.created.lock().expect("created");
        let request = &created[0];

        // Globals: the regular entry lands as RlmGlobalsPatch; the projected
        // entry must NOT appear here.
        assert_eq!(request.initial_nodes.len(), 1);
        let SessionAppendNode::Event {
            event: SessionEventRecord::Mode(mode_event),
        } = &request.initial_nodes[0]
        else {
            panic!("expected seed globals event");
        };
        let Some(RlmModeEvent::RlmGlobalsPatch(seed)) = crate::decode_rlm_mode_event(mode_event)
        else {
            panic!("expected RlmGlobalsPatch");
        };
        assert_eq!(
            seed.set_default.len(),
            1,
            "only `glob` should land as a global"
        );
        assert_eq!(seed.set_default["glob"], json!(7));
        assert!(!seed.set_default.contains_key("proj"));

        // Projected: rides on mode_extras as the snapshot.
        let extras = request
            .mode_extras
            .decode::<RlmCreateExtras>(&lash_core::ExecutionMode::new("rlm"))
            .expect("decode extras")
            .expect("rlm extras");
        let snapshot = extras
            .projected_seed
            .expect("projected entries should ride on mode_extras");
        assert_eq!(snapshot.entries.len(), 1);
        assert_eq!(snapshot.entries[0].0, "proj");
        assert_eq!(snapshot.entries[0].1, json!("carry-over"));
    }
}
