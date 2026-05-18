use async_trait::async_trait;
use lash_core::plugin::runtime_host::RuntimeSessionHost;
use lash_core::{
    MessageRole, ModeExtras, PluginMessage, SessionCreateRequest, SessionPluginMode,
    SessionSnapshot, ToolArgumentProjectionPolicy, ToolCall, ToolContext, ToolContract,
    ToolControl, ToolDefinition, ToolExecutionMode, ToolManifest, ToolProvider, ToolResult,
};
use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::sync::Arc;

use crate::RlmSeed;

pub(crate) struct RlmControlToolsProvider;

impl RlmControlToolsProvider {
    async fn continue_as(
        &self,
        args: &Value,
        context: &ToolContext<'_>,
    ) -> Result<ContinueAsResult, String> {
        let task = required_string(args, "task")?;
        let seed = RlmSeed::from_tool_args(args).map_err(|err| format!("continue_as {err}"))?;
        let referenced_handles = collect_seed_async_handle_ids(args.get("seed"));
        let referenced_handles_vec = referenced_handles.into_iter().collect::<Vec<_>>();
        context
            .tasks()
            .validate_async_handles_visible(&referenced_handles_vec)
            .await
            .map_err(|err| format!("continue_as async handle validation failed: {err}"))?;

        let current_snapshot = context
            .session_snapshot()
            .await
            .map_err(|err| format!("failed to snapshot current session: {err}"))?;
        let successor_session_id = create_continue_as_successor(
            &context.sessions(),
            context.session_id(),
            current_snapshot,
            task.clone(),
            seed,
            ContinueAsHandoff {
                relation_reason: "continue_as".to_string(),
                usage_source: "continue_as".to_string(),
                metadata: serde_json::Map::new(),
            },
        )
        .await?;
        if let Err(err) = context
            .tasks()
            .transfer_async_handles_to_session(&successor_session_id, &referenced_handles_vec)
            .await
        {
            let _ = context
                .sessions()
                .close_session(&successor_session_id)
                .await;
            return Err(format!("continue_as async handle transfer failed: {err}"));
        }
        if let Err(err) = context
            .tasks()
            .cancel_unreferenced_async_handles(&referenced_handles_vec)
            .await
        {
            let _ = context
                .sessions()
                .close_session(&successor_session_id)
                .await;
            return Err(format!(
                "continue_as async handle cleanup failed after successor creation: {err}"
            ));
        }

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
    fn tool_manifests(&self) -> Vec<ToolManifest> {
        vec![continue_as_tool_definition().manifest()]
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<ToolContract>> {
        (name == "continue_as").then(|| Arc::new(continue_as_tool_definition().contract()))
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
        "Tail-call into a fresh RLM successor with a clean window.\n\nThe successor inherits **nothing** automatically — no globals, no projected bindings, no message history. Pass everything it needs via `seed: { name: value, ... }`. Each entry's kind is preserved: if the value's lashlang source root is a host-projected binding (e.g. `seed: { problem: input.prompt }`), it stays projected on the successor (read-only `Host Projected Variables`); other sources land as regular RLM globals. Computed expressions default to global.\n\n- Use when the current trajectory is stale, dominated by failed attempts, or the context budget is tight.\n- Treat `continue_as` as a terminal control action: make it the last meaningful statement in the lashlang block, and do not call `submit` or perform more work after it.\n- `task` packs the concrete goal, constraints, and next steps the successor must act on.\n- `seed` packs the concrete state (paths, facts already learned, partial results, projected sources) the successor needs in scope; leave bulky raw output behind.\n- If live async work is needed after handoff, include its handle in `seed` (for example from `list_async_handles`). Referenced handles transfer to the successor and can be awaited there. Live handles not included in `seed` are cancelled when `continue_as` succeeds.",
        continue_as_input_schema(),
        json!({ "type": "object", "additionalProperties": true }),
    )
    .with_examples(vec![
        r#"call continue_as { task: "continue the audit from the summarized findings", seed: { problem: input.prompt, findings: findings } }"#.into(),
    ])
    .with_argument_projection(ToolArgumentProjectionPolicy::preserve_projected_refs_in_field(
        "seed",
    ))
    .with_execution_mode(ToolExecutionMode::Parallel)
}

pub(crate) fn collect_seed_async_handle_ids(seed: Option<&Value>) -> BTreeSet<String> {
    fn visit(value: &Value, out: &mut BTreeSet<String>) {
        if let Some(id) = value
            .get("__handle__")
            .and_then(Value::as_str)
            .filter(|kind| *kind == "task")
            .and_then(|_| value.get("id"))
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .map(str::to_string)
        {
            out.insert(id);
        }
        match value {
            Value::Array(items) => {
                for item in items {
                    visit(item, out);
                }
            }
            Value::Object(map) => {
                for value in map.values() {
                    visit(value, out);
                }
            }
            _ => {}
        }
    }

    let mut out = BTreeSet::new();
    if let Some(seed) = seed {
        visit(seed, &mut out);
    }
    out
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

pub(crate) async fn create_continue_as_successor(
    sessions: &dyn RuntimeSessionHost,
    parent_session_id: &str,
    current_snapshot: SessionSnapshot,
    task: String,
    seed: RlmSeed,
    handoff: ContinueAsHandoff,
) -> Result<String, String> {
    let termination = current_snapshot
        .mode_turn_options
        .decode(&lash_core::ExecutionMode::new("rlm"))
        .ok()
        .flatten()
        .unwrap_or_default();
    let mut policy = current_snapshot.policy.clone();
    policy.execution_mode = lash_core::ExecutionMode::new("rlm");

    let mode_extras = ModeExtras::typed(
        lash_core::ExecutionMode::new("rlm"),
        lash_rlm_types::RlmCreateExtras { termination },
    )
    .map_err(|err| format!("failed to encode rlm mode extras: {err}"))?;
    let request = SessionCreateRequest::handoff(
        parent_session_id,
        policy,
        mode_extras,
        handoff.relation_reason,
        handoff.metadata,
        handoff.usage_source,
    )
    .with_plugin_mode(SessionPluginMode::InheritCurrent)
    .with_initial_nodes(crate::rlm_seed_initial_nodes(seed))
    .with_first_turn_input(PluginMessage::text(MessageRole::User, task));
    let successor_session_id = request
        .session_id
        .clone()
        .expect("fresh successor request sets session id");
    sessions
        .create_session(request)
        .await
        .map_err(|err| format!("failed to create continue_as successor: {err}"))?;
    Ok(successor_session_id)
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
    use std::collections::BTreeSet;
    use std::sync::{Arc, Mutex};

    use lash_core::plugin::runtime_host::RuntimeSessionHost;
    use lash_core::plugin::{PluginError, SessionHandle, SessionTurnHandle};
    use lash_core::{
        PersistedSessionState, SessionAppendNode, SessionPolicy, SessionStartPoint, TurnInput,
    };
    use lash_rlm_types::{RlmCreateExtras, RlmModeEvent, RlmTermination};

    #[derive(Default)]
    struct BatonManager {
        snapshot: PersistedSessionState,
        created: Mutex<Vec<SessionCreateRequest>>,
        closed: Mutex<Vec<String>>,
        visible_handles: Mutex<BTreeSet<String>>,
        transferred: Mutex<Vec<(String, String, Vec<String>)>>,
        cleanup_keep: Mutex<Vec<Vec<String>>>,
    }

    #[test]
    fn continue_as_tool_definition_preserves_projected_seed_refs_by_metadata() {
        assert_eq!(
            continue_as_tool_definition().argument_projection,
            ToolArgumentProjectionPolicy::preserve_projected_refs_in_field("seed")
        );
    }

    #[async_trait]
    impl RuntimeSessionHost for BatonManager {
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
                parent_session_id: request.relation.parent_session_id().map(ToOwned::to_owned),
                policy: request.policy.unwrap_or_default(),
            })
        }

        async fn close_session(&self, session_id: &str) -> Result<(), PluginError> {
            self.closed
                .lock()
                .expect("closed")
                .push(session_id.to_string());
            Ok(())
        }
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
        async fn validate_async_handles_visible(
            &self,
            _session_id: &str,
            handle_ids: &[String],
        ) -> Result<(), PluginError> {
            let visible = self.visible_handles.lock().expect("visible handles");
            if let Some(missing) = handle_ids.iter().find(|id| !visible.contains(*id)) {
                return Err(PluginError::Session(format!("missing {missing}")));
            }
            Ok(())
        }

        async fn transfer_async_handles(
            &self,
            from_session_id: &str,
            to_session_id: &str,
            handle_ids: &[String],
        ) -> Result<(), PluginError> {
            self.transferred.lock().expect("transferred").push((
                from_session_id.to_string(),
                to_session_id.to_string(),
                handle_ids.to_vec(),
            ));
            Ok(())
        }

        async fn cancel_unreferenced_async_handles(
            &self,
            _session_id: &str,
            keep_handle_ids: &[String],
        ) -> Result<Vec<lash_core::BackgroundTaskRecord>, PluginError> {
            self.cleanup_keep
                .lock()
                .expect("cleanup keep")
                .push(keep_handle_ids.to_vec());
            Ok(Vec::new())
        }
    }

    #[test]
    fn rlm_control_definitions_include_continue_as_only() {
        let provider = RlmControlToolsProvider;
        let names = provider
            .tool_manifests()
            .into_iter()
            .map(|tool| tool.name)
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["continue_as"]);
    }

    #[tokio::test]
    async fn continue_as_creates_empty_rlm_successor_with_seed_and_task() {
        let mut session_graph = lash_core::SessionGraph::default();
        session_graph.append_mode_event(crate::rlm_mode_event(RlmModeEvent::RlmGlobalsPatch(
            lash_rlm_types::RlmGlobalsPatchPluginBody {
                set_default: serde_json::Map::from_iter([("diary".to_string(), json!([]))]),
            },
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
            ..BatonManager::default()
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

        assert!(result.is_success(), "{:?}", result.value_for_projection());
        let value = result.value_for_projection();
        assert!(value.get("session_id").and_then(Value::as_str).is_some());
        assert!(matches!(
            result.as_output().control,
            Some(ToolControl::Handoff { .. })
        ));
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
        let SessionAppendNode::ModeEvent { event: mode_event } = &request.initial_nodes[0] else {
            panic!("expected seed globals event");
        };
        let Some(RlmModeEvent::RlmSeed(seed)) = crate::decode_rlm_mode_event(mode_event) else {
            panic!("expected RlmSeed");
        };
        assert_eq!(seed.globals["x"], json!(1));
        assert_eq!(seed.globals["query"], json!("original"));
        assert!(seed.projected.is_empty());
        let extras = request
            .mode_extras
            .decode::<RlmCreateExtras>(&lash_core::ExecutionMode::new("rlm"))
            .expect("decode extras")
            .expect("rlm extras");
        assert!(matches!(
            extras.termination,
            RlmTermination::SubmitRequired { schema: Some(_) }
        ));
    }

    #[tokio::test]
    async fn continue_as_routes_projected_entries_and_globals_to_one_seed_event() {
        // Mixed seed: `proj` was a projected source on the parent (encoded with
        // the canonical `__projected__` JSON wrapper), `glob` was a regular
        // global. The successor receives both through one durable RLM seed event.
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
            ..BatonManager::default()
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
        assert!(result.is_success(), "{:?}", result.value_for_projection());

        let created = manager.created.lock().expect("created");
        let request = &created[0];

        assert_eq!(request.initial_nodes.len(), 1);
        let SessionAppendNode::ModeEvent { event: mode_event } = &request.initial_nodes[0] else {
            panic!("expected seed globals event");
        };
        let Some(RlmModeEvent::RlmSeed(seed)) = crate::decode_rlm_mode_event(mode_event) else {
            panic!("expected RlmSeed");
        };
        assert_eq!(seed.globals.len(), 1, "only `glob` should land as a global");
        assert_eq!(seed.globals["glob"], json!(7));
        assert!(!seed.globals.contains_key("proj"));
        assert_eq!(seed.projected.entries.len(), 1);
        assert_eq!(seed.projected.entries[0].0, "proj");
        assert_eq!(seed.projected.entries[0].1, json!("carry-over"));
        let extras = request
            .mode_extras
            .decode::<RlmCreateExtras>(&lash_core::ExecutionMode::new("rlm"))
            .expect("decode extras")
            .expect("rlm extras");
        assert_eq!(extras.termination, RlmTermination::default());
    }

    #[tokio::test]
    async fn continue_as_transfers_handles_found_recursively_in_seed() {
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
            visible_handles: Mutex::new(BTreeSet::from_iter(["h1".to_string(), "h2".to_string()])),
            ..BatonManager::default()
        });
        let provider = RlmControlToolsProvider;
        let context = lash_core::testing::mock_tool_context_with_host(manager.clone());

        let result = provider
            .execute(ToolCall {
                name: "continue_as",
                args: &json!({
                    "task": "continue with background work",
                    "seed": {
                        "one": { "__handle__": "task", "id": "h1", "tool": "slow" },
                        "nested": [{ "h": { "__handle__": "task", "id": "h2", "tool": "slow" } }]
                    }
                }),
                context: &context,
                progress: None,
            })
            .await;

        assert!(result.is_success(), "{:?}", result.value_for_projection());
        let value = result.value_for_projection();
        let successor = value
            .get("session_id")
            .and_then(Value::as_str)
            .expect("successor")
            .to_string();
        assert_eq!(
            *manager.transferred.lock().expect("transferred"),
            vec![(
                "test-session".to_string(),
                successor,
                vec!["h1".to_string(), "h2".to_string()]
            )]
        );
        assert_eq!(
            *manager.cleanup_keep.lock().expect("cleanup keep"),
            vec![vec!["h1".to_string(), "h2".to_string()]]
        );
    }

    #[tokio::test]
    async fn continue_as_rejects_unknown_seed_handle_before_creating_successor() {
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
            ..BatonManager::default()
        });
        let provider = RlmControlToolsProvider;
        let context = lash_core::testing::mock_tool_context_with_host(manager.clone());

        let result = provider
            .execute(ToolCall {
                name: "continue_as",
                args: &json!({
                    "task": "continue",
                    "seed": { "h": { "__handle__": "task", "id": "missing", "tool": "slow" } }
                }),
                context: &context,
                progress: None,
            })
            .await;

        assert!(!result.is_success());
        assert!(manager.created.lock().expect("created").is_empty());
        assert!(manager.transferred.lock().expect("transferred").is_empty());
        assert!(
            manager
                .cleanup_keep
                .lock()
                .expect("cleanup keep")
                .is_empty()
        );
    }
}
