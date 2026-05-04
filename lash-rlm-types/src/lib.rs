use lash_sansio::{AttachmentRef, ModeProtocol, ToolCallRecord};

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct RlmTrajectoryEntry {
    pub id: String,
    pub iteration: usize,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub reasoning: String,
    pub code: String,
    /// One entry per `print` (and any raw stdout-style emission from the
    /// lashlang executor). Replaces the old split between a combined
    /// `output: String` and `observations: Vec<String>` — those carried
    /// the same content twice, wasting tokens on every history-bearing
    /// iteration.
    #[serde(default, alias = "observations")]
    pub output: Vec<String>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCallRecord>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<AttachmentRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_output: Option<serde_json::Value>,
}

impl RlmTrajectoryEntry {
    /// Total characters across every `print`/output entry, summed.
    pub fn output_chars(&self) -> usize {
        self.output.iter().map(|s| s.chars().count()).sum()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RlmHistoryRole {
    User,
    System,
    Assistant,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RlmAttachmentRef {
    pub id: String,
    pub media_type: lash_sansio::MediaType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub reference: String,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RlmImageRef {
    pub id: String,
    pub media_type: lash_sansio::MediaType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    pub bytes: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RlmHistoryItem {
    Message {
        id: String,
        role: RlmHistoryRole,
        content: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        attachments: Vec<RlmAttachmentRef>,
    },
    ToolCall {
        id: String,
        tool: String,
        args: serde_json::Value,
        result: serde_json::Value,
        success: bool,
        duration_ms: u64,
    },
    RlmStep {
        id: String,
        iteration: usize,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        reasoning: String,
        code: String,
        #[serde(default, alias = "observations")]
        output: Vec<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tool_calls: Vec<ToolCallRecord>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        images: Vec<RlmImageRef>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        final_output: Option<serde_json::Value>,
    },
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
    RlmDiagnostic(RlmDiagnosticEvent),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct RlmDiagnosticEvent {
    pub phase: String,
    #[serde(default)]
    pub payload: serde_json::Value,
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
            RlmModeEvent::RlmDiagnostic(_) => {}
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

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RlmTermination {
    Finish {
        schema: Option<serde_json::Value>,
        #[serde(default = "default_true", skip_serializing_if = "is_true")]
        include_submit_prompt: bool,
    },
}

impl Default for RlmTermination {
    fn default() -> Self {
        Self::Finish {
            schema: None,
            include_submit_prompt: true,
        }
    }
}

fn default_true() -> bool {
    true
}

fn is_true(value: &bool) -> bool {
    *value
}

/// RLM-mode session config. RLM turns terminate through `submit`,
/// optionally validated against a schema.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct RlmCreateExtras {
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
