use lash_sansio::{ModeProtocol, ToolCallRecord};

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct RlmTrajectoryEntry {
    pub id: String,
    pub iteration: usize,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub reasoning: String,
    pub code: String,
    pub output: String,
    #[serde(default)]
    pub observations: Vec<String>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCallRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_output: Option<serde_json::Value>,
    #[serde(default)]
    pub output_raw_len: usize,
}

impl RlmTrajectoryEntry {
    pub fn output_preview(&self, max_chars: usize) -> String {
        lash_sansio::head_tail_truncate(&self.output, max_chars).0
    }
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RlmGlobalsPatchPluginBody {
    #[serde(default)]
    pub set: serde_json::Map<String, serde_json::Value>,
    #[serde(default)]
    pub unset: Vec<String>,
}

impl RlmGlobalsPatchPluginBody {
    pub fn is_empty(&self) -> bool {
        self.set.is_empty() && self.unset.is_empty()
    }
}

pub fn apply_globals_patch(
    globals: &mut serde_json::Map<String, serde_json::Value>,
    patch: &RlmGlobalsPatchPluginBody,
) {
    for key in &patch.unset {
        globals.remove(key);
    }
    for (key, value) in &patch.set {
        globals.insert(key.clone(), value.clone());
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum RlmModeEvent {
    RlmTrajectoryEntry(RlmTrajectoryEntry),
    RlmGlobalsPatch(RlmGlobalsPatchPluginBody),
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RlmProjection {
    pub globals: serde_json::Map<String, serde_json::Value>,
    pub trajectory: Vec<RlmTrajectoryEntry>,
}

impl RlmProjection {
    pub fn from_events(events: impl IntoIterator<Item = RlmModeEvent>) -> Self {
        let mut projection = Self::default();
        for event in events {
            projection.apply_event(event);
        }
        projection
    }

    pub fn apply_event(&mut self, event: RlmModeEvent) {
        match event {
            RlmModeEvent::RlmTrajectoryEntry(entry) => {
                self.trajectory.push(entry);
            }
            RlmModeEvent::RlmGlobalsPatch(patch) => {
                apply_globals_patch(&mut self.globals, &patch);
            }
        }
    }
}

pub fn project_globals(
    events: impl IntoIterator<Item = RlmModeEvent>,
) -> serde_json::Map<String, serde_json::Value> {
    RlmProjection::from_events(events).globals
}

pub fn project_trajectory(
    events: impl IntoIterator<Item = RlmModeEvent>,
) -> Vec<RlmTrajectoryEntry> {
    RlmProjection::from_events(events).trajectory
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RlmTermination {
    #[default]
    ProseWithoutFence,
    Finish {
        schema: Option<serde_json::Value>,
    },
}

/// RLM-mode session config. Carries the choice of how the model
/// terminates the session (prose vs `submit`-with-optional-schema).
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct RlmCreateExtras {
    #[serde(default)]
    pub termination: RlmTermination,
}

/// Pure-RLM-mode session config. Uses the same termination contract as
/// RLM while projecting history as a structured REPL trajectory.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct RlmpureCreateExtras {
    #[serde(default)]
    pub termination: RlmTermination,
}

#[derive(Clone, Debug)]
pub struct RlmModeProtocol;

impl ModeProtocol for RlmModeProtocol {
    type Event = RlmModeEvent;
    type Termination = RlmTermination;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_derives_globals_and_trajectory_from_mode_events() {
        let first = RlmGlobalsPatchPluginBody {
            set: serde_json::Map::from_iter([
                ("alpha".to_string(), serde_json::json!(1)),
                ("beta".to_string(), serde_json::json!("old")),
            ]),
            unset: Vec::new(),
        };
        let second = RlmGlobalsPatchPluginBody {
            set: serde_json::Map::from_iter([("beta".to_string(), serde_json::json!("new"))]),
            unset: vec!["alpha".to_string()],
        };
        let entry = RlmTrajectoryEntry {
            id: "step-1".to_string(),
            iteration: 1,
            code: "submit 1".to_string(),
            ..Default::default()
        };

        let projection = RlmProjection::from_events([
            RlmModeEvent::RlmGlobalsPatch(first),
            RlmModeEvent::RlmTrajectoryEntry(entry.clone()),
            RlmModeEvent::RlmGlobalsPatch(second),
        ]);

        assert_eq!(projection.globals.get("alpha"), None);
        assert_eq!(
            projection.globals.get("beta"),
            Some(&serde_json::json!("new"))
        );
        assert_eq!(projection.trajectory, vec![entry]);
    }
}
