use std::collections::{BTreeSet, HashMap};

use lash_core::SessionError;
use lashlang::{ExecutionScratch, State as FlowState, Value as FlowValue};
use serde_json::json;

use crate::projection::{prune_protected_bindings, prune_reserved_projected_bindings};

use super::apply_global_defaults;
use super::files::{clear_dir, collect_files, restore_files};
use super::snapshot::{RLM_SNAPSHOT_VERSION, restore_runtime, snapshot_runtime};

pub struct RlmExecutionState {
    pub(super) rlm: FlowState,
    pub(super) scratch: ExecutionScratch,
    pub(super) linked_programs: lashlang::LinkedProgramCache,
    pub(super) stored_lashlang_modules: BTreeSet<lashlang::ModuleRef>,
    pub(super) scratch_dir: tempfile::TempDir,
    pub(super) dirty: bool,
}

impl RlmExecutionState {
    pub fn new() -> Result<Self, SessionError> {
        Ok(Self {
            rlm: FlowState::new(),
            scratch: ExecutionScratch::new(),
            linked_programs: lashlang::LinkedProgramCache::new(),
            stored_lashlang_modules: BTreeSet::new(),
            scratch_dir: tempfile::TempDir::new()?,
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

    /// The live top-level variable namespace as JSON for the "Bound Variables"
    /// prompt section: the model's own scratch variables plus any seeded
    /// computed globals, which are the same kind of value and render the same
    /// way.
    ///
    /// Excludes the reserved `history` binding, the supplied `exclude` names
    /// (read-only values, which get their own type-only section), and any
    /// value that contains read-only projected data. Those are never
    /// materialized for a value preview here.
    pub(crate) fn bound_variable_values(
        &self,
        exclude: &BTreeSet<String>,
    ) -> Vec<(String, FlowValue)> {
        let mut out = Vec::new();
        for (name, value) in self.rlm.globals().iter() {
            if name == "history" || exclude.contains(name) || value.contains_projected() {
                continue;
            }
            out.push((name.to_string(), value.clone()));
        }
        out
    }
}

#[cfg(test)]
mod bound_variable_value_tests {
    use super::*;
    use lashlang::{
        ProjectedFuture, ProjectedHostDescriptor, ProjectedReadRequest, ProjectedReadResponse,
        ProjectedValue, Record as FlowRecord, Value as FlowValue,
    };
    use serde_json::json;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn includes_globals_excludes_history_and_named() {
        let mut state = RlmExecutionState::new().unwrap();
        let mut set_default = serde_json::Map::new();
        set_default.insert("inventory".to_string(), json!(["lantern"]));
        set_default.insert("secret".to_string(), json!(1));
        state
            .patch_globals(
                &lash_rlm_types::RlmGlobalsPatchPluginBody { set_default },
                &BTreeSet::new(),
            )
            .unwrap();

        let exclude: BTreeSet<String> = ["secret".to_string()].into_iter().collect();
        let vars = state.bound_variable_values(&exclude);
        assert!(vars.iter().any(|(name, _)| name == "inventory"), "{vars:?}");
        assert!(
            !vars.iter().any(|(name, _)| name == "secret"),
            "excluded name leaked: {vars:?}"
        );
        assert!(
            !vars.iter().any(|(name, _)| name == "history"),
            "history leaked: {vars:?}"
        );
    }

    #[test]
    fn excludes_direct_projected_globals() {
        let mut state = RlmExecutionState::new().unwrap();
        let mut snapshot = state.rlm.snapshot();
        snapshot.globals.insert(
            "projected".to_string(),
            FlowValue::Projected(ProjectedValue::scalar(
                "projected",
                FlowValue::String("host".into()),
            )),
        );
        snapshot
            .globals
            .insert("plain".to_string(), FlowValue::String("local".into()));
        state.rlm = FlowState::from_snapshot(snapshot);

        let vars = state.bound_variable_values(&BTreeSet::new());

        assert!(vars.iter().any(
            |(name, value)| name == "plain" && value == &FlowValue::String("local".into())
        ));
        assert!(
            !vars.iter().any(|(name, _)| name == "projected"),
            "{vars:?}"
        );
    }

    #[test]
    fn excludes_top_level_globals_containing_nested_projected_values() {
        let mut state = RlmExecutionState::new().unwrap();
        let mut record = FlowRecord::new();
        record.insert(
            "body".to_string(),
            FlowValue::Projected(ProjectedValue::scalar(
                "body",
                FlowValue::String("host".into()),
            )),
        );
        record.insert("title".to_string(), FlowValue::String("local".into()));
        let mut snapshot = state.rlm.snapshot();
        snapshot
            .globals
            .insert("doc".to_string(), FlowValue::Record(Arc::new(record)));
        snapshot.globals.insert(
            "plain".to_string(),
            FlowValue::List(vec![FlowValue::Number(1.0)].into()),
        );
        state.rlm = FlowState::from_snapshot(snapshot);

        let vars = state.bound_variable_values(&BTreeSet::new());

        assert!(vars.iter().any(|(name, _)| name == "plain"));
        assert!(!vars.iter().any(|(name, _)| name == "doc"), "{vars:?}");
    }

    #[derive(Default)]
    struct CountingProjectedValue {
        materialize_count: AtomicUsize,
        render_count: AtomicUsize,
    }

    impl ProjectedHostDescriptor for CountingProjectedValue {
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
                        ProjectedReadResponse::Text("rendered".to_string())
                    }
                    ProjectedReadRequest::Materialize => {
                        self.materialize_count.fetch_add(1, Ordering::SeqCst);
                        ProjectedReadResponse::Value(FlowValue::String("materialized".into()))
                    }
                    _ => ProjectedReadResponse::Missing,
                }
            })
        }
    }

    #[test]
    fn excludes_custom_projected_globals_without_rendering_or_materializing() {
        let projected = Arc::new(CountingProjectedValue::default());
        let mut state = RlmExecutionState::new().unwrap();
        let mut snapshot = state.rlm.snapshot();
        snapshot.globals.insert(
            "projected".to_string(),
            FlowValue::Projected(ProjectedValue::custom("projected", projected.clone())),
        );
        state.rlm = FlowState::from_snapshot(snapshot);

        let vars = state.bound_variable_values(&BTreeSet::new());

        assert!(vars.is_empty(), "{vars:?}");
        assert_eq!(projected.render_count.load(Ordering::SeqCst), 0);
        assert_eq!(projected.materialize_count.load(Ordering::SeqCst), 0);
    }
}
