//! Plan-mode persistent state: enable/generation/plan-path snapshot, the
//! on-disk plan file template, and plan-report reading helpers.

use super::*;

pub(crate) const PLAN_TEMPLATE: &str = r#"# Plan

## Goal
- TBD

## Steps
- TBD

## Files
- TBD

## Risks
- TBD

## Verification
- TBD
"#;

pub(crate) fn plan_display_path(path: &Path) -> String {
    let display = std::env::current_dir()
        .ok()
        .and_then(|cwd| path.strip_prefix(&cwd).ok().map(PathBuf::from))
        .unwrap_or_else(|| path.to_path_buf());
    let rendered = display.display().to_string();
    if rendered.is_empty() {
        ".".to_string()
    } else {
        rendered.replace('\\', "/")
    }
}

pub(crate) fn resolve_plan_path(run_session_id: &str) -> Result<PathBuf, String> {
    let cwd = std::env::current_dir().map_err(|err| format!("Failed to determine cwd: {err}"))?;
    Ok(cwd
        .join(".lash")
        .join("plans")
        .join(format!("{run_session_id}.md")))
}

pub(crate) fn effective_run_session_id<'a>(
    session_id: &'a str,
    policy: &'a lash_core::SessionPolicy,
) -> &'a str {
    policy.session_id.as_deref().unwrap_or(session_id)
}

pub(crate) fn seed_plan_template(path: &Path) -> Result<bool, String> {
    if path.is_file() {
        return Ok(false);
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "Failed to create plan directory `{}`: {err}",
                plan_display_path(parent)
            )
        })?;
    }
    fs::write(path, PLAN_TEMPLATE).map_err(|err| {
        format!(
            "Failed to seed plan template `{}`: {err}",
            plan_display_path(path)
        )
    })?;
    Ok(true)
}

#[derive(Clone, Debug, Default)]
pub(crate) struct PlanReport {
    pub(crate) display_path: String,
    pub(crate) content: Option<String>,
}

impl PlanReport {
    pub(crate) fn preview_content(&self) -> String {
        format!("Path: `{}`", self.display_path)
    }

    pub(crate) fn approval_content(&self) -> String {
        self.content
            .as_deref()
            .map(str::trim_end)
            .filter(|value| !value.trim().is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| self.preview_content())
    }
}

pub(crate) fn read_plan_report(path: &Path) -> Result<PlanReport, String> {
    let display_path = plan_display_path(path);
    if !path.is_file() {
        return Ok(PlanReport {
            display_path,
            ..Default::default()
        });
    }

    let content = fs::read_to_string(path).map_err(|err| {
        format!(
            "Failed to read plan file `{}`: {err}",
            plan_display_path(path)
        )
    })?;
    Ok(PlanReport {
        display_path,
        content: Some(content),
    })
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct PlanModeSnapshot {
    #[serde(default)]
    pub(crate) enabled: bool,
    #[serde(default)]
    pub(crate) generation: u64,
    #[serde(default)]
    pub(crate) plan_path: Option<String>,
}

#[derive(Debug, Default)]
pub(crate) struct PlanModeState {
    pub(crate) enabled: bool,
    pub(crate) generation: u64,
    pub(crate) plan_path: Option<PathBuf>,
    pub(crate) active_turn_applied_generation: Option<u64>,
}

impl PlanModeState {
    pub(crate) fn snapshot(&self) -> PlanModeSnapshot {
        PlanModeSnapshot {
            enabled: self.enabled,
            generation: self.generation,
            plan_path: self
                .plan_path
                .as_ref()
                .map(|path| path.to_string_lossy().to_string()),
        }
    }

    pub(crate) fn set_enabled(&mut self, enabled: bool) -> PlanModeSnapshot {
        if self.enabled != enabled {
            self.enabled = enabled;
            self.generation = self.generation.wrapping_add(1).max(1);
            self.active_turn_applied_generation = None;
        }
        self.snapshot()
    }

    pub(crate) fn prepare_turn(&mut self) -> bool {
        if !self.enabled {
            return false;
        }
        if self.active_turn_applied_generation == Some(self.generation) {
            return false;
        }
        self.active_turn_applied_generation = Some(self.generation);
        true
    }

    pub(crate) fn checkpoint_injection_needed(&mut self) -> bool {
        if !self.enabled || self.active_turn_applied_generation == Some(self.generation) {
            return false;
        }
        self.active_turn_applied_generation = Some(self.generation);
        true
    }

    pub(crate) fn finish_turn(&mut self) {}

    pub(crate) fn plan_path(&self) -> Option<PathBuf> {
        self.plan_path.clone()
    }

    pub(crate) fn ensure_plan_path_from_state(
        &mut self,
        state: &lash_core::SessionSnapshot,
    ) -> Result<PathBuf, PluginError> {
        if let Some(path) = self.plan_path() {
            return Ok(path);
        }
        let path = resolve_plan_path(effective_run_session_id(&state.session_id, &state.policy))
            .map_err(PluginError::Session)?;
        self.plan_path = Some(path.clone());
        Ok(path)
    }

    pub(crate) fn set_plan_path(&mut self, path: PathBuf) {
        self.plan_path = Some(path);
    }

    pub(crate) fn restore_snapshot(&mut self, snapshot: PlanModeSnapshot) {
        self.enabled = snapshot.enabled;
        self.generation = snapshot.generation;
        self.plan_path = snapshot.plan_path.map(PathBuf::from);
        self.active_turn_applied_generation = None;
    }
}
