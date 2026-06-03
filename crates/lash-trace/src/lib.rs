use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

mod lashlang_graph;
#[cfg(feature = "otel")]
pub mod otel;

pub use lashlang_graph::{
    TraceLashlangEdgeSelection, TraceLashlangGraph, TraceLashlangGraphChildLink,
    TraceLashlangGraphEdge, TraceLashlangGraphNode, TraceLashlangGraphStore,
    TraceLashlangNodeStatus,
};

pub const TRACE_SCHEMA_VERSION: u32 = 2;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceLevel {
    #[default]
    Standard,
    Extended,
}

impl TraceLevel {
    pub fn is_extended(self) -> bool {
        matches!(self, Self::Extended)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceContext {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub experiment_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_parent_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub example_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub split: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph_node_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_graph_node_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_index: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub protocol_iteration: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effect_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llm_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

impl TraceContext {
    pub fn for_session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    pub fn for_turn_index(mut self, turn_index: usize) -> Self {
        self.turn_index = Some(turn_index);
        self
    }

    pub fn for_turn(mut self, turn_id: impl Into<String>) -> Self {
        self.turn_id = Some(turn_id.into());
        self
    }

    pub fn for_protocol_iteration(mut self, protocol_iteration: usize) -> Self {
        self.protocol_iteration = Some(protocol_iteration);
        self
    }

    pub fn for_llm_call(mut self, llm_call_id: impl Into<String>) -> Self {
        self.llm_call_id = Some(llm_call_id.into());
        self
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TraceRecord {
    pub schema_version: u32,
    pub id: String,
    pub timestamp: String,
    pub context: TraceContext,
    #[serde(flatten)]
    pub event: TraceEvent,
}

impl TraceRecord {
    pub fn new(context: TraceContext, event: TraceEvent) -> Self {
        Self::new_with_timestamp(context, event, chrono::Utc::now())
    }

    pub fn new_with_timestamp(
        context: TraceContext,
        event: TraceEvent,
        timestamp: chrono::DateTime<chrono::Utc>,
    ) -> Self {
        Self {
            schema_version: TRACE_SCHEMA_VERSION,
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: timestamp.to_rfc3339(),
            context,
            event,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[allow(
    clippy::large_enum_variant,
    reason = "TraceEvent is a public DTO; keeping event payloads inline preserves ergonomic pattern matching"
)]
pub enum TraceEvent {
    SessionStarted {
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        metadata: BTreeMap<String, Value>,
    },
    TurnStarted {
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        metadata: BTreeMap<String, Value>,
    },
    PromptBuilt {
        prompt_hash: String,
        prompt_chars: usize,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        components: Vec<TracePromptComponent>,
    },
    LlmCallStarted {
        request: TraceLlmRequest,
    },
    LlmCallCompleted {
        response: TraceLlmResponse,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<TraceTokenUsage>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_usage: Option<Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        stream_summary: Option<Value>,
    },
    LlmCallFailed {
        error: TraceError,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        stream_summary: Option<Value>,
    },
    ProviderStreamEvent {
        event: TraceProviderStreamEvent,
    },
    RuntimeStreamEvent {
        event: TraceRuntimeStreamEvent,
    },
    ToolCallStarted {
        call_id: Option<String>,
        name: String,
        args: Value,
    },
    ToolCallCompleted {
        call_id: Option<String>,
        name: String,
        args: Value,
        output: TraceToolCallOutput,
        duration_ms: u64,
    },
    ProtocolStep {
        plugin_id: String,
        payload: Value,
    },
    TokenUsage {
        usage: TraceTokenUsage,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cumulative: Option<TraceTokenUsage>,
    },
    LashlangExecution {
        event: TraceLashlangExecutionEvent,
    },
    TurnCompleted {
        status: String,
        done_reason: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        agent_frame_switch: Option<TraceAgentFrameSwitch>,
    },
    Custom {
        name: String,
        payload: Value,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TraceToolCallOutput {
    pub outcome: TraceToolCallOutcome,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control: Option<Value>,
}

impl TraceToolCallOutput {
    pub fn status(&self) -> TraceToolCallStatus {
        match self.outcome {
            TraceToolCallOutcome::Success(_) => TraceToolCallStatus::Success,
            TraceToolCallOutcome::Failure(_) => TraceToolCallStatus::Failure,
            TraceToolCallOutcome::Cancelled(_) => TraceToolCallStatus::Cancelled,
        }
    }

    pub fn is_success(&self) -> bool {
        self.status() == TraceToolCallStatus::Success
    }

    pub fn value_for_projection(&self) -> Value {
        match &self.outcome {
            TraceToolCallOutcome::Success(value)
            | TraceToolCallOutcome::Failure(value)
            | TraceToolCallOutcome::Cancelled(value) => value.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", content = "payload", rename_all = "snake_case")]
pub enum TraceToolCallOutcome {
    Success(Value),
    Failure(Value),
    Cancelled(Value),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceToolCallStatus {
    Success,
    Failure,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TracePromptComponent {
    pub id: String,
    pub kind: String,
    pub hash: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chars: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TraceLlmRequest {
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_variant: Option<String>,
    pub messages: Vec<TraceLlmMessage>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attachments: Vec<TraceAttachment>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<TraceToolSpec>,
    pub tool_choice: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_spec: Option<Value>,
    pub stream: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TraceLlmMessage {
    pub role: String,
    pub blocks: Vec<TraceContentBlock>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TraceContentBlock {
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "is_false")]
        cache_breakpoint: bool,
    },
    Image {
        attachment_idx: usize,
    },
    ToolCall {
        call_id: Option<String>,
        tool_name: String,
        input_json: Value,
        item_id: Option<String>,
        has_signature: bool,
    },
    ToolResult {
        call_id: Option<String>,
        tool_name: Option<String>,
        content: String,
    },
    Reasoning {
        text: String,
        item_id: Option<String>,
        summary: Vec<String>,
        has_encrypted: bool,
        redacted: bool,
    },
}

fn is_false(value: &bool) -> bool {
    !*value
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TraceAttachment {
    pub mime: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes_len: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TraceToolSpec {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
    pub output_schema: Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TraceLlmResponse {
    pub text: String,
    pub duration_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parts: Option<Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TraceProviderStreamEvent {
    pub provider: String,
    pub sequence: u64,
    pub elapsed_ms: u64,
    pub event_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub item_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_index: Option<i64>,
    pub raw_len: usize,
    pub raw_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_json: Option<Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TraceRuntimeStreamEvent {
    pub sequence: u64,
    pub elapsed_ms: u64,
    pub event_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub visible_text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub item_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_index: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_json: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<TraceTokenUsage>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceTokenUsage {
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cached_input_tokens: i64,
    pub reasoning_tokens: i64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceAgentFrameSwitch {
    pub frame_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceRuntimeScope {
    pub session_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_index: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub protocol_iteration: Option<usize>,
}

impl TraceRuntimeScope {
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            turn_id: None,
            turn_index: None,
            protocol_iteration: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TraceRuntimeSubject {
    Effect { effect_id: String, kind: String },
    Process { process_id: String },
}

impl TraceRuntimeSubject {
    pub fn graph_key(&self, scope: &TraceRuntimeScope) -> String {
        match self {
            Self::Effect { effect_id, .. } => match scope.turn_id.as_deref() {
                Some(turn_id) if !turn_id.is_empty() => {
                    format!("effect:{}:{turn_id}:{effect_id}", scope.session_id)
                }
                _ => format!("effect:{}:{effect_id}", scope.session_id),
            },
            Self::Process { process_id } => format!("process:{process_id}"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceLashlangExecutionIdentity {
    pub scope: TraceRuntimeScope,
    pub subject: TraceRuntimeSubject,
    pub module_ref: String,
    pub entry_kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry_ref: Option<String>,
    pub entry_name: String,
}

impl TraceLashlangExecutionIdentity {
    pub fn graph_key(&self) -> String {
        self.subject.graph_key(&self.scope)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TraceLashlangExecutionEvent {
    ExecutionStarted {
        event_key: String,
        identity: TraceLashlangExecutionIdentity,
        execution_map: TraceLashlangMap,
    },
    ExecutionFinished {
        event_key: String,
        identity: TraceLashlangExecutionIdentity,
        status: TraceLashlangStatus,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    NodeStarted {
        event_key: String,
        identity: TraceLashlangExecutionIdentity,
        node_id: String,
        node_kind: String,
        label: String,
        occurrence: u64,
    },
    NodeCompleted {
        event_key: String,
        identity: TraceLashlangExecutionIdentity,
        node_id: String,
        node_kind: String,
        label: String,
        occurrence: u64,
    },
    NodeFailed {
        event_key: String,
        identity: TraceLashlangExecutionIdentity,
        node_id: String,
        node_kind: String,
        label: String,
        occurrence: u64,
        error: String,
    },
    BranchSelected {
        event_key: String,
        identity: TraceLashlangExecutionIdentity,
        node_id: String,
        occurrence: u64,
        edge_id: String,
        selected: TraceBranchSelection,
    },
    ChildStarted {
        event_key: String,
        identity: TraceLashlangExecutionIdentity,
        parent_node_id: String,
        occurrence: u64,
        child: TraceLashlangChildExecution,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceLashlangChildExecution {
    pub scope: TraceRuntimeScope,
    pub subject: TraceRuntimeSubject,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub module_ref: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry_ref: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry_name: Option<String>,
}

impl TraceLashlangChildExecution {
    pub fn graph_key(&self) -> String {
        self.subject.graph_key(&self.scope)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceLashlangStatus {
    Running,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceBranchSelection {
    Then,
    Else,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceLashlangMap {
    pub module_ref: String,
    pub entry_kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry_ref: Option<String>,
    pub entry_name: String,
    #[serde(default)]
    pub nodes: Vec<TraceLashlangMapNode>,
    #[serde(default)]
    pub edges: Vec<TraceLashlangMapEdge>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceLashlangMapNode {
    pub id: String,
    pub kind: String,
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label_metadata: Option<TraceLabelMetadata>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceLabelMetadata {
    pub title: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceLashlangMapEdge {
    pub id: String,
    pub from: String,
    pub to: String,
    pub label: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceError {
    pub message: String,
    pub retryable: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum TraceSinkError {
    #[error("failed to serialize trace record: {0}")]
    Serialize(#[from] serde_json::Error),
    #[error("trace sink lock poisoned")]
    LockPoisoned,
    #[error("failed to create trace directory {path}: {source}")]
    CreateDir { path: PathBuf, source: io::Error },
    #[error("failed to open trace file {path}: {source}")]
    Open { path: PathBuf, source: io::Error },
    #[error("failed to write trace file {path}: {source}")]
    Write { path: PathBuf, source: io::Error },
}

pub trait TraceSink: Send + Sync {
    fn append(&self, record: &TraceRecord) -> Result<(), TraceSinkError>;
}

pub struct JsonlTraceSink {
    path: PathBuf,
    lock: Mutex<()>,
}

impl JsonlTraceSink {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            lock: Mutex::new(()),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl TraceSink for JsonlTraceSink {
    fn append(&self, record: &TraceRecord) -> Result<(), TraceSinkError> {
        let line = serde_json::to_string(record)?;
        let _guard = self.lock.lock().map_err(|_| TraceSinkError::LockPoisoned)?;
        if let Some(parent) = self.path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).map_err(|source| TraceSinkError::CreateDir {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|source| TraceSinkError::Open {
                path: self.path.clone(),
                source,
            })?;
        writeln!(file, "{line}").map_err(|source| TraceSinkError::Write {
            path: self.path.clone(),
            source,
        })
    }
}

/// Writes each trace record as one JSON line to stderr — handy for `cargo run`
/// debugging without a trace file.
#[derive(Default)]
pub struct StderrTraceSink {
    lock: Mutex<()>,
}

impl TraceSink for StderrTraceSink {
    fn append(&self, record: &TraceRecord) -> Result<(), TraceSinkError> {
        let line = serde_json::to_string(record)?;
        let _guard = self.lock.lock().map_err(|_| TraceSinkError::LockPoisoned)?;
        eprintln!("{line}");
        Ok(())
    }
}

/// Fans each trace record out to several sinks in order (e.g. stderr + a JSONL
/// file). Stops at the first sink that errors.
pub struct TeeTraceSink {
    sinks: Vec<Arc<dyn TraceSink>>,
}

impl TeeTraceSink {
    pub fn new(sinks: impl IntoIterator<Item = Arc<dyn TraceSink>>) -> Self {
        Self {
            sinks: sinks.into_iter().collect(),
        }
    }
}

impl TraceSink for TeeTraceSink {
    fn append(&self, record: &TraceRecord) -> Result<(), TraceSinkError> {
        for sink in &self.sinks {
            sink.append(record)?;
        }
        Ok(())
    }
}

pub fn sha256_hex(input: impl AsRef<[u8]>) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_ref());
    format!("{:x}", hasher.finalize())
}

pub fn json_hash(value: &Value) -> String {
    sha256_hex(serde_json::to_vec(value).unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jsonl_sink_writes_record() {
        let dir = std::env::temp_dir().join(format!("lash-trace-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("trace.jsonl");
        let sink = JsonlTraceSink::new(&path);
        sink.append(&TraceRecord::new(
            TraceContext::default().for_session("root"),
            TraceEvent::Custom {
                name: "test.event".to_string(),
                payload: serde_json::json!({"ok": true}),
            },
        ))
        .unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("\"type\":\"custom\""));
        assert!(text.contains("\"session_id\":\"root\""));
    }

    #[test]
    fn tool_start_and_frame_switch_records_are_jsonl_shaped() {
        let started = TraceRecord::new(
            TraceContext::default().for_session("root"),
            TraceEvent::ToolCallStarted {
                call_id: Some("call-1".to_string()),
                name: "read_file".to_string(),
                args: serde_json::json!({"path": "README.md"}),
            },
        );
        let completed = TraceRecord::new(
            TraceContext::default().for_session("root"),
            TraceEvent::TurnCompleted {
                status: "completed".to_string(),
                done_reason: "modelstop".to_string(),
                agent_frame_switch: Some(TraceAgentFrameSwitch {
                    frame_id: "frame-1".to_string(),
                }),
            },
        );

        let started_json = serde_json::to_value(started).unwrap();
        assert_eq!(started_json["type"], "tool_call_started");
        assert_eq!(started_json["call_id"], "call-1");

        let completed_json = serde_json::to_value(completed).unwrap();
        assert_eq!(completed_json["type"], "turn_completed");
        assert_eq!(completed_json["agent_frame_switch"]["frame_id"], "frame-1");
    }

    #[test]
    fn lashlang_execution_records_are_jsonl_shaped() {
        let identity = TraceLashlangExecutionIdentity {
            scope: TraceRuntimeScope::new("s1"),
            subject: TraceRuntimeSubject::Process {
                process_id: "p1".to_string(),
            },
            module_ref: "module".to_string(),
            entry_kind: "process".to_string(),
            entry_ref: Some("component:0".to_string()),
            entry_name: "main".to_string(),
        };
        let event = TraceLashlangExecutionEvent::NodeStarted {
            event_key: "process:p1:node:n1:1:started".to_string(),
            identity,
            node_id: "n1".to_string(),
            node_kind: "resource_operation".to_string(),
            label: "read_file".to_string(),
            occurrence: 1,
        };
        let record = TraceRecord::new(
            TraceContext::default().for_session("s1"),
            TraceEvent::LashlangExecution { event },
        );

        let json = serde_json::to_value(&record).expect("serialize lashlang execution");
        assert_eq!(json["type"], "lashlang_execution");
        assert_eq!(json["event"]["kind"], "node_started");
        assert_eq!(json["event"]["event_key"], "process:p1:node:n1:1:started");

        let round_trip =
            serde_json::from_value::<TraceRecord>(json).expect("deserialize lashlang execution");
        assert!(matches!(
            round_trip.event,
            TraceEvent::LashlangExecution {
                event: TraceLashlangExecutionEvent::NodeStarted { .. }
            }
        ));
    }

    #[test]
    fn tool_completion_serializes_typed_failure_output() {
        let record = TraceRecord::new(
            TraceContext::default().for_session("root"),
            TraceEvent::ToolCallCompleted {
                call_id: Some("call-1".to_string()),
                name: "read_file".to_string(),
                args: serde_json::json!({"path": "missing"}),
                output: TraceToolCallOutput {
                    outcome: TraceToolCallOutcome::Failure(serde_json::json!({
                        "class": "invalid_request",
                        "code": "invalid_tool_args",
                        "message": "bad args",
                        "source": "runtime",
                        "retry": { "type": "never" },
                        "raw": { "path": "missing" }
                    })),
                    control: None,
                },
                duration_ms: 3,
            },
        );

        let json = serde_json::to_value(record).unwrap();
        assert_eq!(json["type"], "tool_call_completed");
        assert_eq!(json["output"]["outcome"]["status"], "failure");
        assert_eq!(
            json["output"]["outcome"]["payload"]["code"],
            "invalid_tool_args"
        );
        assert_eq!(
            json["output"]["outcome"]["payload"]["raw"]["path"],
            "missing"
        );
    }

    #[test]
    fn jsonl_sink_creates_parent_directories() {
        let dir = std::env::temp_dir().join(format!("lash-trace-{}", uuid::Uuid::new_v4()));
        let path = dir.join("nested").join("trace.jsonl");
        let sink = JsonlTraceSink::new(&path);
        sink.append(&TraceRecord::new(
            TraceContext::default().for_session("root"),
            TraceEvent::RuntimeStreamEvent {
                event: TraceRuntimeStreamEvent {
                    sequence: 1,
                    elapsed_ms: 0,
                    event_name: "delta".to_string(),
                    raw_text: Some("hello".to_string()),
                    visible_text: Some("hello".to_string()),
                    item_id: None,
                    output_index: None,
                    call_id: None,
                    tool_name: None,
                    input_json: None,
                    usage: None,
                },
            },
        ))
        .unwrap();
        assert!(path.exists());
        let _ = std::fs::remove_dir_all(dir);
    }
}
