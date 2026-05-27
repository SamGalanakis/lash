use async_trait::async_trait;
use lash_core::plugin::runtime_host::RuntimeSessionHost;
use lash_core::{
    MessageRole, PluginMessage, PluginOptions, SessionCreateRequest, SessionPluginSource,
    SessionSnapshot, ToolArgumentProjectionPolicy, ToolAvailabilityConfig, ToolCall, ToolContext,
    ToolContract, ToolControl, ToolDefinition, ToolManifest, ToolProvider, ToolResult,
    ToolScheduling,
};
use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::sync::Arc;

use crate::plugin::RLM_PROTOCOL_PLUGIN_ID;
use crate::projection::RlmSeed;

pub(crate) struct RlmControlToolsProvider;

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
            "continue_as" => continue_as_handoff(call.args, call.context).await,
            _ => return ToolResult::err_fmt(format_args!("Unknown tool: {}", call.name)),
        };
        finalise_tool_result(result)
    }
}

pub fn continue_as_tool_definition() -> ToolDefinition {
    ToolDefinition::raw(
        "tool:continue_as",
        "continue_as",
        "Tail-call into a fresh RLM successor with a clean window.\n\nThe successor inherits **nothing** automatically — no globals, no projected bindings, no message history. Pass everything it needs via `seed: { name: value, ... }`. Each entry's kind is preserved: if the value's lashlang source root is a host-projected binding (e.g. `seed: { problem: input.prompt }`), it stays projected on the successor (read-only `Host Projected Variables`); other sources land as regular RLM globals. Computed expressions default to global.\n\n- Use when the current trajectory is stale, dominated by failed attempts, or the context budget is tight.\n- Treat `continue_as` as a terminal control action: make it the last meaningful statement in the lashlang block, and do not call `submit` or perform more work after it.\n- `task` packs the concrete goal, constraints, and next steps the successor must act on.\n- `seed` packs the concrete state (paths, facts already learned, partial results, projected sources) the successor needs in scope; leave bulky raw output behind.\n- If live async work is needed after handoff, include its handle in `seed` (for example from `list_process_handles`). Referenced handles transfer to the successor and can be awaited there. Live handles not included in `seed` are cancelled when `continue_as` succeeds.",
        continue_as_input_schema(),
        json!({ "type": "object", "additionalProperties": true }),
    )
    .with_examples(vec![
        r#"await TOOL.default.continue_as({ task: "continue the audit from the summarized findings", seed: { problem: input.prompt, findings: findings } })?"#.into(),
    ])
    .with_argument_projection(ToolArgumentProjectionPolicy::preserve_projected_refs_in_field(
        "seed",
    ))
    .with_availability(ToolAvailabilityConfig::callable())
    .with_scheduling(ToolScheduling::Parallel)
}

pub(crate) fn collect_seed_process_handle_ids(seed: Option<&Value>) -> BTreeSet<String> {
    fn visit(value: &Value, out: &mut BTreeSet<String>) {
        if let Some(id) = value
            .get("__handle__")
            .and_then(Value::as_str)
            .filter(|kind| *kind == "process")
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

async fn continue_as_handoff(
    args: &Value,
    context: &ToolContext<'_>,
) -> Result<ContinueAsResult, String> {
    let task = required_string(args, "task")?;
    let seed = RlmSeed::from_tool_args(args).map_err(|err| format!("continue_as {err}"))?;
    let referenced_handles = collect_seed_process_handle_ids(args.get("seed"));
    let referenced_handles_vec = referenced_handles.into_iter().collect::<Vec<_>>();
    let processes = context.processes();
    processes
        .validate_handles(&referenced_handles_vec)
        .await
        .map_err(|err| format!("continue_as process handle validation failed: {err}"))?;

    let sessions = context.sessions();
    let current_snapshot = sessions
        .snapshot_current()
        .await
        .map_err(|err| format!("failed to snapshot current session: {err}"))?;
    let successor_session_id = create_continue_as_successor(
        &sessions,
        context.session_id(),
        current_snapshot,
        task.clone(),
        seed,
    )
    .await?;
    if let Err(err) = processes
        .transfer_handles(&successor_session_id, referenced_handles_vec.clone())
        .await
    {
        let _ = sessions.close_session(&successor_session_id).await;
        return Err(format!("continue_as process handle transfer failed: {err}"));
    }
    if let Err(err) = processes
        .cancel_unreferenced_handles(referenced_handles_vec.clone())
        .await
    {
        let _ = sessions.close_session(&successor_session_id).await;
        return Err(format!(
            "continue_as process handle cleanup failed after successor creation: {err}"
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

async fn create_continue_as_successor(
    sessions: &dyn RuntimeSessionHost,
    parent_session_id: &str,
    current_snapshot: SessionSnapshot,
    task: String,
    seed: RlmSeed,
) -> Result<String, String> {
    let termination = current_snapshot
        .protocol_turn_options
        .decode()
        .ok()
        .unwrap_or_default();

    let plugin_options = PluginOptions::typed(
        RLM_PROTOCOL_PLUGIN_ID,
        lash_rlm_types::RlmCreateExtras { termination },
    )
    .map_err(|err| format!("failed to encode rlm plugin options: {err}"))?;
    let request = SessionCreateRequest::handoff(
        parent_session_id,
        current_snapshot.policy.clone(),
        plugin_options,
        "continue_as".to_string(),
        serde_json::Map::new(),
        "continue_as".to_string(),
    )
    .with_plugin_source(SessionPluginSource::CurrentSessionFork)
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
    use crate::projection::{decode_rlm_protocol_event, rlm_protocol_event};
    use std::collections::BTreeSet;
    use std::sync::{Arc, Mutex};

    use lash_core::plugin::runtime_host::RuntimeSessionHost;
    use lash_core::plugin::{PluginError, SessionHandle};
    use lash_core::{
        RuntimeSessionState, SessionAppendNode, SessionPolicy, SessionStartPoint, ToolProvider,
    };
    use lash_rlm_types::{RlmCreateExtras, RlmProtocolEvent, RlmTermination};

    fn model_spec(model: &str) -> lash_core::ModelSpec {
        lash_core::ModelSpec::from_token_limits(model, None, 200_000, None, None)
            .expect("valid test model spec")
    }

    #[derive(Default)]
    struct BatonManager {
        snapshot: RuntimeSessionState,
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
        async fn snapshot_current(&self) -> Result<RuntimeSessionState, PluginError> {
            Ok(self.snapshot.clone())
        }

        async fn snapshot_session(
            &self,
            _session_id: &str,
        ) -> Result<RuntimeSessionState, PluginError> {
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
    }

    #[async_trait]
    impl lash_core::ProcessService for BatonManager {
        async fn start(
            &self,
            _session_id: &str,
            _registration: lash_core::ProcessRegistration,
            _options: lash_core::ProcessStartOptions,
            _scope: lash_core::ProcessOpScope<'_>,
        ) -> Result<lash_core::ProcessRecord, PluginError> {
            Err(PluginError::Session(
                "process starts are unavailable in this test".to_string(),
            ))
        }

        async fn await_process(
            &self,
            _process_id: &str,
            _scope: lash_core::ProcessOpScope<'_>,
        ) -> Result<lash_core::ProcessAwaitOutput, PluginError> {
            Err(PluginError::Session(
                "process awaiting is unavailable in this test".to_string(),
            ))
        }

        async fn list_visible(
            &self,
            _session_id: &str,
            _scope: lash_core::ProcessOpScope<'_>,
        ) -> Result<Vec<lash_core::ProcessHandleGrantEntry>, PluginError> {
            Ok(Vec::new())
        }

        async fn validate_visible(
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

        async fn cancel(
            &self,
            _session_id: &str,
            _process_id: &str,
            _scope: lash_core::ProcessOpScope<'_>,
        ) -> Result<lash_core::ProcessRecord, PluginError> {
            Err(PluginError::Session(
                "process cancellation is unavailable in this test".to_string(),
            ))
        }

        async fn cancel_all(
            &self,
            _session_id: &str,
            _scope: lash_core::ProcessOpScope<'_>,
        ) -> Result<Vec<lash_core::ProcessRecord>, PluginError> {
            Ok(Vec::new())
        }

        async fn transfer(
            &self,
            from_session_id: &str,
            to_session_id: &str,
            process_ids: Vec<String>,
            _scope: lash_core::ProcessOpScope<'_>,
        ) -> Result<(), PluginError> {
            self.transferred.lock().expect("transferred").push((
                from_session_id.to_string(),
                to_session_id.to_string(),
                process_ids,
            ));
            Ok(())
        }

        async fn cancel_unreferenced(
            &self,
            _session_id: &str,
            keep_process_ids: Vec<String>,
            _scope: lash_core::ProcessOpScope<'_>,
        ) -> Result<Vec<lash_core::ProcessRecord>, PluginError> {
            self.cleanup_keep
                .lock()
                .expect("cleanup keep")
                .push(keep_process_ids);
            Ok(Vec::new())
        }
    }

    async fn run_continue_as(
        provider: &RlmControlToolsProvider,
        manager: Arc<BatonManager>,
        args: &Value,
    ) -> ToolResult {
        let host: Arc<dyn RuntimeSessionHost> = manager.clone();
        let processes: Arc<dyn lash_core::ProcessService> = manager;
        let context = lash_core::ToolContext::__for_testing(
            "test-session".to_string(),
            host,
            processes,
            Arc::new(lash_core::InMemoryAttachmentStore::new()),
            lash_core::DirectCompletionClient::from_fn(|_, _| {
                Err(lash_core::PluginError::Session(
                    "direct completions are unavailable in continue_as tests".to_string(),
                ))
            }),
            Some("continue-as-test".to_string()),
        );
        provider
            .execute(lash_core::ToolCall {
                name: "continue_as",
                args,
                context: &context,
                progress: None,
            })
            .await
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
        session_graph.append_protocol_event(rlm_protocol_event(RlmProtocolEvent::RlmGlobalsPatch(
            lash_rlm_types::RlmGlobalsPatchPluginBody {
                set_default: serde_json::Map::from_iter([("diary".to_string(), json!([]))]),
            },
        )));
        let manager = Arc::new(BatonManager {
            snapshot: RuntimeSessionState {
                policy: SessionPolicy {
                    model: model_spec("model"),
                    ..SessionPolicy::default()
                },
                protocol_turn_options: lash_core::ProtocolTurnOptions::typed(
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
                ..RuntimeSessionState::default()
            },
            created: Mutex::new(Vec::new()),
            ..BatonManager::default()
        });
        let provider = RlmControlToolsProvider;

        let args = json!({
            "task": "finish from here",
            "seed": { "x": 1, "query": "original" }
        });
        let result = run_continue_as(&provider, manager.clone(), &args).await;

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
        assert_eq!(
            request.plugin_source,
            SessionPluginSource::CurrentSessionFork
        );
        assert_eq!(request.relation.parent_session_id(), Some("test-session"));
        assert_eq!(
            request
                .first_turn_input
                .as_ref()
                .map(|message| message.content.as_str()),
            Some("finish from here")
        );
        assert_eq!(request.initial_nodes.len(), 1);
        let SessionAppendNode::ProtocolEvent {
            event: protocol_event,
        } = &request.initial_nodes[0]
        else {
            panic!("expected seed globals event");
        };
        let Some(RlmProtocolEvent::RlmSeed(seed)) = decode_rlm_protocol_event(protocol_event)
        else {
            panic!("expected RlmSeed");
        };
        assert_eq!(seed.globals["x"], json!(1));
        assert_eq!(seed.globals["query"], json!("original"));
        assert!(seed.projected.is_empty());
        let extras = request
            .plugin_options
            .decode::<RlmCreateExtras>(RLM_PROTOCOL_PLUGIN_ID)
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
            snapshot: RuntimeSessionState {
                policy: SessionPolicy {
                    model: model_spec("model"),
                    ..SessionPolicy::default()
                },
                ..RuntimeSessionState::default()
            },
            created: Mutex::new(Vec::new()),
            ..BatonManager::default()
        });
        let provider = RlmControlToolsProvider;

        let args = json!({
            "task": "finish from here",
            "seed": {
                "proj": { "__projected__": "carry-over" },
                "glob": 7
            }
        });
        let result = run_continue_as(&provider, manager.clone(), &args).await;
        assert!(result.is_success(), "{:?}", result.value_for_projection());

        let created = manager.created.lock().expect("created");
        let request = &created[0];

        assert_eq!(request.initial_nodes.len(), 1);
        let SessionAppendNode::ProtocolEvent {
            event: protocol_event,
        } = &request.initial_nodes[0]
        else {
            panic!("expected seed globals event");
        };
        let Some(RlmProtocolEvent::RlmSeed(seed)) = decode_rlm_protocol_event(protocol_event)
        else {
            panic!("expected RlmSeed");
        };
        assert_eq!(seed.globals.len(), 1, "only `glob` should land as a global");
        assert_eq!(seed.globals["glob"], json!(7));
        assert!(!seed.globals.contains_key("proj"));
        assert_eq!(seed.projected.entries.len(), 1);
        assert_eq!(seed.projected.entries[0].0, "proj");
        assert_eq!(seed.projected.entries[0].1, json!("carry-over"));
        let extras = request
            .plugin_options
            .decode::<RlmCreateExtras>(RLM_PROTOCOL_PLUGIN_ID)
            .expect("decode extras")
            .expect("rlm extras");
        assert_eq!(extras.termination, RlmTermination::default());
    }

    #[tokio::test]
    async fn continue_as_transfers_handles_found_recursively_in_seed() {
        let manager = Arc::new(BatonManager {
            snapshot: RuntimeSessionState {
                policy: SessionPolicy {
                    model: model_spec("model"),
                    ..SessionPolicy::default()
                },
                ..RuntimeSessionState::default()
            },
            created: Mutex::new(Vec::new()),
            visible_handles: Mutex::new(BTreeSet::from_iter(["h1".to_string(), "h2".to_string()])),
            ..BatonManager::default()
        });
        let provider = RlmControlToolsProvider;

        let args = json!({
            "task": "continue with background work",
            "seed": {
                "one": { "__handle__": "process", "id": "h1", "tool": "slow" },
                "nested": [{ "h": { "__handle__": "process", "id": "h2", "tool": "slow" } }]
            }
        });
        let result = run_continue_as(&provider, manager.clone(), &args).await;

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
            snapshot: RuntimeSessionState {
                policy: SessionPolicy {
                    model: model_spec("model"),
                    ..SessionPolicy::default()
                },
                ..RuntimeSessionState::default()
            },
            created: Mutex::new(Vec::new()),
            ..BatonManager::default()
        });
        let provider = RlmControlToolsProvider;

        let args = json!({
            "task": "continue",
            "seed": { "h": { "__handle__": "process", "id": "missing", "tool": "slow" } }
        });
        let result = run_continue_as(&provider, manager.clone(), &args).await;

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
