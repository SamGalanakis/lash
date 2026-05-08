use lash_sansio::{AttachmentRef, ModeProtocol, ToolCallRecord};

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct RlmTrajectoryEntry {
    pub id: String,
    pub mode_iteration: usize,
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
        mode_iteration: usize,
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
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub set_default: serde_json::Map<String, serde_json::Value>,
}

impl RlmGlobalsPatchPluginBody {
    pub fn is_empty(&self) -> bool {
        self.set_default.is_empty()
    }
}

pub fn apply_globals_patch(
    _globals: &mut serde_json::Map<String, serde_json::Value>,
    _patch: &RlmGlobalsPatchPluginBody,
) {
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
    SubmitRequired { schema: Option<serde_json::Value> },
    ProseOrSubmit,
}

impl Default for RlmTermination {
    fn default() -> Self {
        Self::SubmitRequired { schema: None }
    }
}

/// RLM-mode session config. RLM turns terminate through `submit`,
/// optionally validated against a schema, unless prose completion is
/// explicitly enabled for the turn.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct RlmCreateExtras {
    #[serde(default)]
    pub termination: RlmTermination,
    /// Optional projected-binding seed for the new session. Each entry becomes
    /// a host-projected (read-only) binding visible in the child's RLM
    /// system-prompt under `Host Projected Variables`. Used by `spawn_agent`
    /// and `continue_as` when the parent passes a projected source via `seed:`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projected_seed: Option<RlmProjectedSeedSnapshot>,
}

/// Wire-format snapshot of a set of projected bindings. Pairs of
/// `(name, json_value)` that get re-projected as host bindings on the child
/// session at creation time. This is the serializable form of
/// `lash_mode_rlm::RlmProjectedBindings`; lash-rlm-types stays free of any
/// runtime dependency on lashlang itself.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct RlmProjectedSeedSnapshot {
    pub entries: Vec<(String, serde_json::Value)>,
}

impl RlmProjectedSeedSnapshot {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, name: impl Into<String>, value: serde_json::Value) {
        self.entries.push((name.into(), value));
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Classified `seed:` argument: each entry has been routed to either the
/// projected-bindings snapshot (if its lashlang source was a projected
/// binding, encoded as a single-key `{"__projected__": <inner>}` object) or
/// the regular RLM-globals patch.
#[derive(Default, Debug, Clone)]
pub struct ClassifiedSeed {
    pub projected: RlmProjectedSeedSnapshot,
    pub globals: serde_json::Map<String, serde_json::Value>,
}

/// Reserved JSON key used as the canonical wire encoding for
/// `lashlang::Value::Projected` across the lashlang→host bridge. When the
/// model passes a projected source as a tool argument, lashlang serializes it
/// as `{"__projected__": <inner>}`.
pub const PROJECTED_JSON_TAG: &str = "__projected__";

/// Returns the inner JSON value if `value` is the canonical projection wrapper
/// (a single-key object whose key is [`PROJECTED_JSON_TAG`]), else `None`.
pub fn projection_inner(value: &serde_json::Value) -> Option<&serde_json::Value> {
    let obj = value.as_object()?;
    if obj.len() != 1 {
        return None;
    }
    obj.get(PROJECTED_JSON_TAG)
}

/// Walk a `seed:` JSON object and split each entry by lashlang-source kind.
/// Projected sources (encoded `{"__projected__": <inner>}`) land in the
/// snapshot; everything else lands in the globals map. Returns an empty
/// classification when no seed is provided. Errors on a non-object value.
pub fn classify_seed(args: &serde_json::Value) -> Result<ClassifiedSeed, String> {
    let raw = match args.get("seed") {
        None | Some(serde_json::Value::Null) => return Ok(ClassifiedSeed::default()),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => return Err("`seed` must be a record/dict".to_string()),
    };
    let mut out = ClassifiedSeed::default();
    for (name, value) in raw.iter() {
        if let Some(inner) = projection_inner(value) {
            out.projected.push(name.clone(), inner.clone());
        } else {
            out.globals.insert(name.clone(), value.clone());
        }
    }
    Ok(out)
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
    fn projection_ignores_executor_defaults_and_derives_trajectory_from_mode_events() {
        let first = RlmGlobalsPatchPluginBody {
            set_default: serde_json::Map::from_iter([(
                "executor_only".to_string(),
                serde_json::json!("ignored by projection"),
            )]),
        };
        let second = RlmGlobalsPatchPluginBody {
            set_default: serde_json::Map::from_iter([(
                "diary".to_string(),
                serde_json::json!("ignored by projection"),
            )]),
        };
        let entry = RlmTrajectoryEntry {
            id: "step-1".to_string(),
            mode_iteration: 1,
            code: "submit 1".to_string(),
            ..Default::default()
        };

        let projection = RlmProjection::from_events([
            RlmModeEvent::RlmGlobalsPatch(first),
            RlmModeEvent::RlmTrajectoryEntry(entry.clone()),
            RlmModeEvent::RlmGlobalsPatch(second),
        ]);

        assert!(projection.globals.is_empty());
        assert_eq!(projection.trajectory, vec![entry]);
    }
}
