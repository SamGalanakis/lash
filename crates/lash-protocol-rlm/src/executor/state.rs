use std::collections::{BTreeSet, HashMap};

use lash_core::SessionError;
use lash_plugin_tool_output_budget::ToolOutputBudgetConfig;
use lashlang::{ExecutionScratch, State as FlowState};
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
    pub(super) observe_projection: ToolOutputBudgetConfig,
    pub(super) dirty: bool,
}

impl RlmExecutionState {
    pub fn new(config: ToolOutputBudgetConfig) -> Result<Self, SessionError> {
        Ok(Self {
            rlm: FlowState::new(),
            scratch: ExecutionScratch::new(),
            linked_programs: lashlang::LinkedProgramCache::new(),
            stored_lashlang_modules: BTreeSet::new(),
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
