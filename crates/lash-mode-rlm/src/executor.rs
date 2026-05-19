mod host_bridge;
mod projections;

use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use std::sync::Arc;

use lash_core::{
    ExecRequest, ExecResponse, ModeExecutionContext, SessionError, ToolOutputBudgetConfig,
};
use lashlang::{CompiledProgramCache, ExecutionOutcome, ExecutionScratch, State as FlowState};
use serde_json::json;

use self::host_bridge::HostBridge;
use self::projections::{
    projected_bindings, prune_projected_binding_names, prune_protected_bindings,
    prune_reserved_projected_bindings, rehydrate_projected_globals,
};
use crate::projected_bindings::{
    ProjectionResolver, RLM_TURN_INPUT_PLUGIN_ID, RlmProjectedBindings, RlmProjectionExtension,
};
use crate::projection_codec::{flow_to_json_value, json_to_flow_value};

const RLM_SNAPSHOT_VERSION: u32 = 3;

pub struct RlmExecutionState {
    rlm: FlowState,
    program_cache: CompiledProgramCache,
    scratch: ExecutionScratch,
    scratch_dir: tempfile::TempDir,
    observe_projection: ToolOutputBudgetConfig,
    dirty: bool,
}

impl RlmExecutionState {
    pub fn new(config: ToolOutputBudgetConfig) -> Result<Self, SessionError> {
        Ok(Self {
            rlm: FlowState::new(),
            program_cache: CompiledProgramCache::default(),
            scratch: ExecutionScratch::new(),
            scratch_dir: tempfile::TempDir::new()?,
            observe_projection: config,
            dirty: true,
        })
    }

    pub fn execution_state_dirty(&self) -> bool {
        self.dirty
    }

    pub fn snapshot_execution_state(&mut self) -> Result<Option<Vec<u8>>, SessionError> {
        let vars = snapshot_runtime(&self.rlm).map_err(SessionError::Protocol)?;
        let files = collect_files(self.scratch_dir.path()).unwrap_or_default();
        let combined = json!({
            "version": RLM_SNAPSHOT_VERSION,
            "engine": "lashlang",
            "vars": vars,
            "files": files,
        });
        self.dirty = false;
        Ok(Some(serde_json::to_vec(&combined)?))
    }

    pub fn restore_execution_state(&mut self, data: &[u8]) -> Result<(), SessionError> {
        let parsed: serde_json::Value = serde_json::from_slice(data).unwrap_or(json!({}));

        if parsed.get("version").is_none() || parsed.get("engine").is_none() {
            return Err(SessionError::Protocol(
                "unsupported RLM snapshot format".to_string(),
            ));
        }
        if parsed.get("version").and_then(|v| v.as_u64()) != Some(RLM_SNAPSHOT_VERSION as u64) {
            return Err(SessionError::Protocol(
                "unsupported RLM snapshot version".to_string(),
            ));
        }
        if parsed.get("engine").and_then(|v| v.as_str()) != Some("lashlang") {
            return Err(SessionError::Protocol(
                "unsupported RLM snapshot engine".to_string(),
            ));
        }

        let vars_str = parsed
            .get("vars")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        self.rlm = restore_runtime(&vars_str)
            .map_err(|err| SessionError::Protocol(format!("executor restore failed: {err}")))?;
        prune_reserved_projected_bindings(&mut self.rlm);

        if let Some(files_val) = parsed.get("files")
            && let Ok(files) = serde_json::from_value::<HashMap<String, String>>(files_val.clone())
        {
            clear_dir(self.scratch_dir.path());
            let _ = restore_files(self.scratch_dir.path(), &files);
        }
        self.dirty = true;
        Ok(())
    }

    pub fn prune_protected_globals(&mut self, protected_names: &BTreeSet<String>) {
        prune_protected_bindings(&mut self.rlm, protected_names);
    }

    pub fn patch_globals(
        &mut self,
        patch: &lash_rlm_types::RlmGlobalsPatchPluginBody,
        protected_names: &BTreeSet<String>,
    ) -> Result<(), SessionError> {
        if patch.is_empty() {
            return Ok(());
        }
        apply_global_defaults(&mut self.rlm, patch, protected_names)
            .map_err(SessionError::Protocol)?;
        self.dirty = true;
        Ok(())
    }
}

pub async fn execute_code(
    mut state: RlmExecutionState,
    ctx: ModeExecutionContext<'_>,
    request: ExecRequest,
    session_projected_bindings: RlmProjectedBindings,
    projection_resolver: Arc<dyn ProjectionResolver>,
) -> Result<(RlmExecutionState, ExecResponse), SessionError> {
    let start = std::time::Instant::now();
    let clean_code = clean_model_code(&request.code);
    let response = Box::pin(execute_code_inner(
        &mut state,
        ctx,
        &clean_code,
        start,
        session_projected_bindings,
        projection_resolver,
    ))
    .await;
    Ok((state, response))
}

fn clean_model_code(code: &str) -> String {
    code.lines()
        .filter(|line| {
            let trimmed = line.trim();
            trimmed.is_empty()
                || trimmed
                    .trim_matches('-')
                    .chars()
                    .any(|c| !c.is_whitespace())
        })
        .collect::<Vec<_>>()
        .join("\n")
}

async fn execute_code_inner(
    state: &mut RlmExecutionState,
    ctx: ModeExecutionContext<'_>,
    code: &str,
    start: std::time::Instant,
    session_projected_bindings: RlmProjectedBindings,
    projection_resolver: Arc<dyn ProjectionResolver>,
) -> ExecResponse {
    state.dirty = true;
    let compiled = match state.program_cache.get_or_compile(code) {
        Ok(compiled) => compiled,
        Err(err) => {
            return ExecResponse {
                output: String::new(),
                observations: Vec::new(),
                observation_truncation: Vec::new(),
                tool_calls: Vec::new(),
                images: Vec::new(),
                printed_images: Vec::new(),
                error: Some(lashlang::format_parse_diagnostic(code, &err)),
                duration_ms: start.elapsed().as_millis() as u64,
                terminal_finish: None,
            };
        }
    };

    if let Err(err) =
        rehydrate_projected_globals(&mut state.rlm, Arc::clone(&projection_resolver)).await
    {
        return ExecResponse {
            output: String::new(),
            observations: Vec::new(),
            observation_truncation: Vec::new(),
            tool_calls: Vec::new(),
            images: Vec::new(),
            printed_images: Vec::new(),
            error: Some(err),
            duration_ms: start.elapsed().as_millis() as u64,
            terminal_finish: None,
        };
    }

    let projected =
        match projected_bindings(&ctx, session_projected_bindings, projection_resolver).await {
            Ok(projected) => projected,
            Err(err) => {
                return ExecResponse {
                    output: String::new(),
                    observations: Vec::new(),
                    observation_truncation: Vec::new(),
                    tool_calls: Vec::new(),
                    images: Vec::new(),
                    printed_images: Vec::new(),
                    error: Some(err),
                    duration_ms: start.elapsed().as_millis() as u64,
                    terminal_finish: None,
                };
            }
        };
    let projected_names = projected.names().collect::<Vec<_>>();
    prune_projected_binding_names(&mut state.rlm, projected_names.iter().map(String::as_str));
    let tool_result_projectors = tool_result_projectors(&ctx);
    let host = HostBridge::new(
        ctx,
        state.observe_projection.clone(),
        tool_result_projectors,
    );

    let result = Box::pin(
        lashlang::execute_compiled_traced_with_scratch_and_projected_bindings(
            &compiled,
            &mut state.rlm,
            &host,
            &mut state.scratch,
            &projected,
        ),
    )
    .await;
    let terminal_finish = match result {
        Ok(ExecutionOutcome::Finished(value)) => Some(flow_to_json_value(&value).await),
        Ok(ExecutionOutcome::Continued) => None,
        Err(failure) => {
            let collected = host.into_collected();
            return ExecResponse {
                output: String::new(),
                observations: collected.observations,
                observation_truncation: collected.observation_truncation,
                tool_calls: collected.tool_calls,
                images: collected.tool_images,
                printed_images: collected.printed_images,
                error: Some(lashlang::format_runtime_diagnostic(
                    code,
                    &failure.error,
                    failure.span,
                )),
                duration_ms: start.elapsed().as_millis() as u64,
                terminal_finish: None,
            };
        }
    };
    let collected = host.into_collected();
    ExecResponse {
        output: String::new(),
        observations: collected.observations,
        observation_truncation: collected.observation_truncation,
        tool_calls: collected.tool_calls,
        images: collected.tool_images,
        printed_images: collected.printed_images,
        error: None,
        duration_ms: start.elapsed().as_millis() as u64,
        terminal_finish,
    }
}

fn tool_result_projectors(ctx: &ModeExecutionContext<'_>) -> Vec<crate::RlmToolResultProjector> {
    ctx.turn_context()
        .plugin_input::<RlmProjectionExtension>(RLM_TURN_INPUT_PLUGIN_ID)
        .map(|extension| extension.tool_result_projectors.clone())
        .unwrap_or_default()
}

fn snapshot_runtime(rlm: &FlowState) -> Result<String, String> {
    serde_json::to_string(&rlm.snapshot()).map_err(|err| format!("failed to snapshot RLM: {err}"))
}

fn restore_runtime(data: &str) -> Result<FlowState, String> {
    let snapshot: lashlang::Snapshot =
        serde_json::from_str(data).map_err(|err| format!("failed to restore RLM: {err}"))?;
    Ok(FlowState::from_snapshot(snapshot))
}

fn apply_global_defaults(
    rlm: &mut FlowState,
    patch: &lash_rlm_types::RlmGlobalsPatchPluginBody,
    protected_names: &BTreeSet<String>,
) -> Result<(), String> {
    if patch.set_default.is_empty() {
        return Ok(());
    }
    let mut snapshot = rlm.snapshot();
    for (key, value) in &patch.set_default {
        if is_reserved_global_name(key) || protected_names.contains(key) {
            return Err(format!(
                "`{key}` is a read-only projected host binding; choose a different Lashlang variable name for `set_default`"
            ));
        }
        if snapshot.globals.get(key).is_none() {
            snapshot
                .globals
                .insert(key.clone(), json_to_flow_value(value.clone()));
        }
    }
    *rlm = FlowState::from_snapshot(snapshot);
    Ok(())
}

fn is_reserved_global_name(key: &str) -> bool {
    key == "history"
}

fn collect_files(root: &Path) -> std::io::Result<HashMap<String, String>> {
    let mut files = HashMap::new();
    walk_dir(root, root, &mut files)?;
    Ok(files)
}

fn walk_dir(root: &Path, dir: &Path, files: &mut HashMap<String, String>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            walk_dir(root, &path, files)?;
        } else {
            let rel = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .to_string();
            let contents = std::fs::read_to_string(&path).unwrap_or_default();
            files.insert(rel, contents);
        }
    }
    Ok(())
}

fn restore_files(root: &Path, files: &HashMap<String, String>) -> std::io::Result<()> {
    for (rel, contents) in files {
        let path = root.join(rel);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, contents)?;
    }
    Ok(())
}

fn clear_dir(root: &Path) {
    if let Ok(entries) = std::fs::read_dir(root) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let _ = std::fs::remove_dir_all(&path);
            } else {
                let _ = std::fs::remove_file(&path);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::projections::projected_index;
    use super::*;
    use crate::projected_bindings::ProjectionRef;
    use crate::projection_codec::{
        flow_record_to_json_value, flow_record_to_tool_args, flow_to_json_value,
    };
    use lash_rlm_types::PROJECTED_JSON_TAG;
    use lashlang::{
        ProjectedBindings, ProjectedFuture, ProjectedHostValue, ProjectedReadRequest,
        ProjectedReadResponse, ProjectedValue, Record as FlowRecord, ToolHost, ToolHostError,
        Value as FlowValue,
    };
    use serde_json::Value;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Default)]
    struct NoopHost;

    impl ToolHost for NoopHost {
        async fn call(&self, name: String, _args: FlowRecord) -> Result<FlowValue, ToolHostError> {
            Err(ToolHostError::new(format!("unknown tool: {name}")))
        }
    }

    fn block_on<T>(future: impl std::future::Future<Output = T>) -> T {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime")
            .block_on(future)
    }

    struct TestProjectedValue(Vec<FlowValue>);

    #[derive(Default)]
    struct SnapshotProjectedToolText {
        materialize_count: AtomicUsize,
        render_count: AtomicUsize,
    }

    impl ProjectedHostValue for SnapshotProjectedToolText {
        fn type_name(&self) -> &str {
            "string"
        }

        fn read_one(
            &self,
            request: ProjectedReadRequest,
        ) -> ProjectedFuture<'_, ProjectedReadResponse> {
            Box::pin(async move {
                match request {
                    ProjectedReadRequest::Render => {
                        self.render_count.fetch_add(1, Ordering::SeqCst);
                        ProjectedReadResponse::Text("rendered tool text".to_string())
                    }
                    ProjectedReadRequest::Materialize => {
                        self.materialize_count.fetch_add(1, Ordering::SeqCst);
                        ProjectedReadResponse::Value(FlowValue::String(
                            "materialized tool text".into(),
                        ))
                    }
                    _ => ProjectedReadResponse::Missing,
                }
            })
        }
    }

    impl ProjectedHostValue for TestProjectedValue {
        fn type_name(&self) -> &str {
            "list"
        }

        fn read_one(
            &self,
            request: ProjectedReadRequest,
        ) -> ProjectedFuture<'_, ProjectedReadResponse> {
            Box::pin(async move {
                let ProjectedReadRequest::Index(index) = request else {
                    return match request {
                        ProjectedReadRequest::Len => ProjectedReadResponse::Len(self.0.len()),
                        ProjectedReadRequest::Materialize => {
                            ProjectedReadResponse::Value(FlowValue::List(self.0.clone().into()))
                        }
                        _ => ProjectedReadResponse::Missing,
                    };
                };
                let Ok(Some(index)) = projected_index(&index, self.0.len()) else {
                    return ProjectedReadResponse::Missing;
                };
                self.0
                    .get(index)
                    .cloned()
                    .map(ProjectedReadResponse::Value)
                    .unwrap_or(ProjectedReadResponse::Missing)
            })
        }
    }

    fn projected_history(values: Vec<FlowValue>) -> ProjectedBindings {
        let mut projected = ProjectedBindings::new();
        projected.insert(
            "history",
            ProjectedValue::custom("history", Arc::new(TestProjectedValue(values))),
        );
        projected
    }

    #[test]
    fn projected_history_is_available_without_clobbering_executor_globals() {
        block_on(async {
            let mut state =
                RlmExecutionState::new(ToolOutputBudgetConfig::default()).expect("state");
            let mut set_default = serde_json::Map::new();
            set_default.insert("diary".to_string(), serde_json::json!(["kept"]));
            state
                .patch_globals(
                    &lash_rlm_types::RlmGlobalsPatchPluginBody { set_default },
                    &BTreeSet::new(),
                )
                .expect("patch diary");

            let projected = projected_history(vec![FlowValue::String("hello".into())]);
            let compiled = lashlang::compile_source(
                "submit { history_len: len(history), diary_len: len(diary) }",
            )
            .expect("compile");
            let outcome = lashlang::execute_compiled_with_projected_bindings(
                &compiled,
                &mut state.rlm,
                &NoopHost,
                &projected,
            )
            .await
            .expect("execute");
            let ExecutionOutcome::Finished(FlowValue::Record(record)) = outcome else {
                panic!("expected submitted record");
            };
            assert_eq!(record["history_len"], FlowValue::Number(1.0));
            assert_eq!(record["diary_len"], FlowValue::Number(1.0));
            assert!(state.rlm.snapshot().globals.get("history").is_none());
        });
    }

    #[test]
    fn projected_history_defaults_to_empty_list_when_missing() {
        block_on(async {
            let mut state =
                RlmExecutionState::new(ToolOutputBudgetConfig::default()).expect("state");

            let projected = projected_history(Vec::new());
            let compiled =
                lashlang::compile_source("submit { history_len: len(history) }").expect("compile");
            let outcome = lashlang::execute_compiled_with_projected_bindings(
                &compiled,
                &mut state.rlm,
                &NoopHost,
                &projected,
            )
            .await
            .expect("execute");
            let ExecutionOutcome::Finished(FlowValue::Record(record)) = outcome else {
                panic!("expected submitted record");
            };
            assert_eq!(record["history_len"], FlowValue::Number(0.0));
        });
    }

    #[test]
    fn set_default_initializes_once_and_does_not_mutate_projected_globals() {
        let mut state = RlmExecutionState::new(ToolOutputBudgetConfig::default()).expect("state");
        let projected = BTreeSet::from_iter(["current_query".to_string()]);

        state
            .patch_globals(
                &lash_rlm_types::RlmGlobalsPatchPluginBody {
                    set_default: serde_json::Map::from_iter([(
                        "diary".to_string(),
                        serde_json::json!(["initial"]),
                    )]),
                },
                &projected,
            )
            .expect("apply defaults");
        assert_eq!(
            state.rlm.snapshot().globals.get("diary"),
            Some(&FlowValue::List(
                vec![FlowValue::String("initial".into())].into()
            ))
        );
        assert!(state.rlm.snapshot().globals.get("current_query").is_none());

        state
            .patch_globals(
                &lash_rlm_types::RlmGlobalsPatchPluginBody {
                    set_default: serde_json::Map::from_iter([(
                        "diary".to_string(),
                        serde_json::json!(["clobber"]),
                    )]),
                },
                &projected,
            )
            .expect("reapply defaults");
        assert_eq!(
            state.rlm.snapshot().globals.get("diary"),
            Some(&FlowValue::List(
                vec![FlowValue::String("initial".into())].into()
            ))
        );
    }

    #[test]
    fn set_default_rejects_projected_host_bindings() {
        let mut state = RlmExecutionState::new(ToolOutputBudgetConfig::default()).expect("state");
        let projected = BTreeSet::from_iter(["current_query".to_string()]);

        let err = state
            .patch_globals(
                &lash_rlm_types::RlmGlobalsPatchPluginBody {
                    set_default: serde_json::Map::from_iter([(
                        "current_query".to_string(),
                        serde_json::json!("bad"),
                    )]),
                },
                &projected,
            )
            .expect_err("projected default should fail");
        assert!(err.to_string().contains("read-only projected host binding"));

        let err = state
            .patch_globals(
                &lash_rlm_types::RlmGlobalsPatchPluginBody {
                    set_default: serde_json::Map::from_iter([(
                        "history".to_string(),
                        serde_json::json!([]),
                    )]),
                },
                &BTreeSet::new(),
            )
            .expect_err("history default should fail");
        assert!(err.to_string().contains("read-only projected host binding"));
    }

    #[test]
    fn projected_scalar_bindings_are_read_only_and_not_snapshotted() {
        block_on(async {
            let mut state =
                RlmExecutionState::new(ToolOutputBudgetConfig::default()).expect("state");
            let mut projected = ProjectedBindings::new();
            projected.insert(
                "current_query",
                ProjectedValue::scalar("current_query", FlowValue::String("host".into())),
            );

            let compiled = lashlang::compile_source(
                "submit { chars: len(current_query), value: current_query }",
            )
            .expect("compile read");
            let outcome = lashlang::execute_compiled_with_projected_bindings(
                &compiled,
                &mut state.rlm,
                &NoopHost,
                &projected,
            )
            .await
            .expect("execute read");
            let ExecutionOutcome::Finished(FlowValue::Record(record)) = outcome else {
                panic!("expected submitted record");
            };
            assert_eq!(record["chars"], FlowValue::Number(4.0));
            assert_eq!(record["value"], FlowValue::String("host".into()));
            assert!(state.rlm.snapshot().globals.get("current_query").is_none());

            let compiled =
                lashlang::compile_source("current_query = \"local\"").expect("compile write");
            let failure = lashlang::execute_compiled_traced_with_projected_bindings(
                &compiled,
                &mut state.rlm,
                &NoopHost,
                &projected,
            )
            .await
            .expect_err("projected write should fail");
            assert!(
                failure
                    .error
                    .to_string()
                    .contains("read-only projected binding")
            );
        });
    }

    #[test]
    fn executor_snapshot_does_not_materialize_projected_tool_result_globals() {
        let projected = Arc::new(SnapshotProjectedToolText::default());
        let mut state = RlmExecutionState::new(ToolOutputBudgetConfig::default()).expect("state");
        let mut snapshot = state.rlm.snapshot();
        snapshot.globals.insert(
            "m".to_string(),
            FlowValue::Projected(ProjectedValue::custom(
                "search.matches[0].text",
                projected.clone(),
            )),
        );
        state.rlm = FlowState::from_snapshot(snapshot);

        let bytes = state
            .snapshot_execution_state()
            .expect("executor snapshot")
            .expect("snapshot bytes");

        assert_eq!(projected.render_count.load(Ordering::SeqCst), 0);
        assert_eq!(projected.materialize_count.load(Ordering::SeqCst), 0);
        let outer: Value = serde_json::from_slice(&bytes).expect("snapshot json");
        let vars = outer
            .get("vars")
            .and_then(Value::as_str)
            .expect("vars string");
        assert!(!vars.contains("rendered tool text"));
        assert!(!vars.contains("materialized tool text"));
        assert!(vars.contains("__lashlang_snapshot_projected__"));
        assert!(vars.contains("search.matches[0].text"));

        let restored = restore_runtime(vars).expect("restore runtime");
        assert!(matches!(
            restored.snapshot().globals.get("m"),
            Some(FlowValue::Projected(_))
        ));
    }

    #[test]
    fn flow_to_json_value_emits_projected_marker_for_projected_values() {
        block_on(async {
            let projected = ProjectedValue::scalar("input", FlowValue::String("hello".into()));
            let value = flow_to_json_value(&FlowValue::Projected(projected)).await;
            let obj = value
                .as_object()
                .expect("expected projected wrapper object");
            assert_eq!(obj.len(), 1, "wrapper should have exactly one key");
            assert_eq!(
                obj.get(PROJECTED_JSON_TAG)
                    .and_then(|v| v.as_str())
                    .expect("inner string"),
                "hello"
            );
        });
    }

    #[test]
    fn flow_to_json_value_preserves_projection_ref_without_materializing() {
        block_on(async {
            let host = Arc::new(SnapshotProjectedToolText::default());
            let reference = ProjectionRef::new("memory", serde_json::json!("doc"));
            let projected = ProjectedValue::custom_with_projection_ref(
                "doc",
                host.clone(),
                serde_json::json!(reference),
            );
            let value = flow_to_json_value(&FlowValue::Projected(projected)).await;
            assert_eq!(host.render_count.load(Ordering::SeqCst), 0);
            assert_eq!(host.materialize_count.load(Ordering::SeqCst), 0);
            assert_eq!(
                value,
                serde_json::json!({
                    PROJECTED_JSON_TAG: {
                        lash_rlm_types::PROJECTION_REF_JSON_TAG: {
                            "kind": "memory",
                            "key": "doc",
                        }
                    }
                })
            );
        });
    }

    #[test]
    fn executor_snapshot_round_trips_projection_ref_metadata() {
        let reference = ProjectionRef::new("memory", serde_json::json!("doc"));
        let mut state = RlmExecutionState::new(ToolOutputBudgetConfig::default()).expect("state");
        let mut snapshot = state.rlm.snapshot();
        snapshot.globals.insert(
            "doc".to_string(),
            FlowValue::Projected(ProjectedValue::custom_with_projection_ref(
                "doc",
                Arc::new(SnapshotProjectedToolText::default()),
                serde_json::json!(reference),
            )),
        );
        state.rlm = FlowState::from_snapshot(snapshot);

        let bytes = state
            .snapshot_execution_state()
            .expect("executor snapshot")
            .expect("snapshot bytes");
        let outer: Value = serde_json::from_slice(&bytes).expect("snapshot json");
        let vars = outer
            .get("vars")
            .and_then(Value::as_str)
            .expect("vars string");
        assert!(vars.contains("projection_ref"));
        assert!(vars.contains("\"kind\":\"memory\""));

        let restored = restore_runtime(vars).expect("restore runtime");
        let restored_snapshot = restored.snapshot();
        let Some(FlowValue::Projected(projected)) = restored_snapshot.globals.get("doc") else {
            panic!("expected restored projected value");
        };
        assert_eq!(
            projected.projection_ref(),
            Some(&serde_json::json!({"kind": "memory", "key": "doc"}))
        );
    }

    #[test]
    fn flow_record_to_json_value_marks_only_projected_entries() {
        block_on(async {
            let projected = ProjectedValue::scalar("input", FlowValue::String("p".into()));
            let mut record = FlowRecord::default();
            record.insert("proj".to_string(), FlowValue::Projected(projected));
            record.insert("glob".to_string(), FlowValue::String("g".into()));

            let value = flow_record_to_json_value(&record).await;
            let obj = value.as_object().expect("record object");
            // proj entry must be wrapped in {"__projected__": ...}
            let proj = obj
                .get("proj")
                .and_then(|v| v.as_object())
                .expect("proj entry is an object");
            assert!(proj.contains_key(PROJECTED_JSON_TAG));
            // glob entry stays a bare string
            assert_eq!(obj.get("glob").and_then(|v| v.as_str()).expect("glob"), "g");
        });
    }

    #[test]
    fn flow_record_to_tool_args_materializes_ordinary_tools() {
        block_on(async {
            let projected = ProjectedValue::scalar("input", FlowValue::String("p".into()));
            let mut record = FlowRecord::default();
            record.insert("query".to_string(), FlowValue::Projected(projected));

            let value = flow_record_to_tool_args(
                &record,
                &lash_core::ToolArgumentProjectionPolicy::MaterializeProjectedValues,
            )
            .await;

            assert_eq!(value, serde_json::json!({ "query": "p" }));
        });
    }

    #[test]
    fn flow_record_to_tool_args_preserves_only_seed_projected_roots() {
        block_on(async {
            let reference = ProjectionRef::new("memory", serde_json::json!("doc"));
            let projected_root = ProjectedValue::custom_with_projection_ref(
                "doc",
                Arc::new(SnapshotProjectedToolText::default()),
                serde_json::json!(reference),
            );
            let mut computed = FlowRecord::default();
            computed.insert(
                "summary".to_string(),
                FlowValue::Projected(ProjectedValue::scalar(
                    "summary",
                    FlowValue::String("materialized summary".into()),
                )),
            );
            let mut seed = FlowRecord::default();
            seed.insert("problem".to_string(), FlowValue::Projected(projected_root));
            seed.insert(
                "computed".to_string(),
                FlowValue::Record(Arc::new(computed)),
            );
            let mut record = FlowRecord::default();
            record.insert(
                "task".to_string(),
                FlowValue::Projected(ProjectedValue::scalar(
                    "task",
                    FlowValue::String("inspect".into()),
                )),
            );
            record.insert("seed".to_string(), FlowValue::Record(Arc::new(seed)));

            let value = flow_record_to_tool_args(
                &record,
                &lash_core::ToolArgumentProjectionPolicy::preserve_projected_refs_in_field("seed"),
            )
            .await;

            assert_eq!(
                value,
                serde_json::json!({
                    "task": "inspect",
                    "seed": {
                        "problem": {
                            "__projected__": {
                                "__projection_ref__": {
                                    "kind": "memory",
                                    "key": "doc"
                                }
                            }
                        },
                        "computed": {
                            "summary": "materialized summary"
                        }
                    }
                })
            );
        });
    }

    #[test]
    fn parse_error_for_unsupported_while_points_at_while() {
        let source = r#"pool_i = 0
while len(final_ids) < 100 && pool_i < len(candidate_pools) {
  for m in candidate_pools[pool_i].matches {
    print m
  }
}"#;
        let err = match lashlang::compile_source(source) {
            Ok(_) => panic!("while should not compile"),
            Err(err) => err,
        };

        let message = lashlang::format_parse_diagnostic(source, &err);
        assert!(message.contains("unsupported `while` loop"), "{message}");
        assert!(message.contains("--> line 2, column 1"), "{message}");
        assert!(
            message.contains("use bounded `for` loops over ranges or lists"),
            "{message}"
        );
        assert!(
            message.contains("hint: use bounded `for` loops over ranges or lists"),
            "{message}"
        );
        assert!(message.contains("while len(final_ids) < 100"), "{message}");
        assert!(
            !message.contains("expected `:`, found identifier `m`"),
            "{message}"
        );
    }
}
