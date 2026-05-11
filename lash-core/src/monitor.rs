use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

const fn default_monitor_timeout_ms() -> u64 {
    300_000
}

pub const MAX_MONITOR_TIMEOUT_MS: u64 = 3_600_000;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MonitorArmOn {
    #[default]
    Manual,
    SessionStart,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MonitorWakePolicy {
    Notify,
    #[default]
    QueueTurn,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct MonitorSpec {
    pub id: String,
    pub command: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub env: BTreeMap<String, String>,
    #[serde(default)]
    pub persistent: bool,
    #[serde(default = "default_monitor_timeout_ms")]
    pub timeout_ms: u64,
    #[serde(default)]
    pub arm_on: MonitorArmOn,
    #[serde(default)]
    pub wake_policy: MonitorWakePolicy,
    #[serde(default)]
    pub restart_on_restore: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MonitorRunState {
    #[default]
    Idle,
    Running,
    Stopped,
    Exited,
    Failed,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct MonitorStatus {
    pub spec: MonitorSpec,
    #[serde(default)]
    pub armed: bool,
    #[serde(default)]
    pub run_state: MonitorRunState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_event: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_exit_status: Option<i32>,
    #[serde(default)]
    pub event_count: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct MonitorEvent {
    pub sequence: u64,
    pub monitor_id: String,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queue_turn_input: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct MonitorSnapshot {
    #[serde(default)]
    pub revision: u64,
    #[serde(default)]
    pub active_count: usize,
    #[serde(default)]
    pub monitors: Vec<MonitorStatus>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct MonitorUpdateBatch {
    #[serde(default)]
    pub revision: u64,
    #[serde(default)]
    pub active_count: usize,
    #[serde(default)]
    pub events: Vec<MonitorEvent>,
}
