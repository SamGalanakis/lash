use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

#[cfg(feature = "otel")]
pub mod otel;

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
    pub iteration: Option<usize>,
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

    pub fn for_iteration(mut self, iteration: usize) -> Self {
        self.iteration = Some(iteration);
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
        Self {
            schema_version: TRACE_SCHEMA_VERSION,
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            context,
            event,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
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
        result: Value,
        success: bool,
        duration_ms: u64,
    },
    ModeStep {
        mode: String,
        payload: Value,
    },
    TokenUsage {
        usage: TraceTokenUsage,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cumulative: Option<TraceTokenUsage>,
    },
    TurnCompleted {
        status: String,
        done_reason: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        handoff: Option<TraceHandoff>,
    },
    Custom {
        name: String,
        payload: Value,
    },
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
pub struct TraceHandoff {
    pub successor_session_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceError {
    pub message: String,
    pub retryable: bool,
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
    fn tool_start_and_handoff_records_are_jsonl_shaped() {
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
                handoff: Some(TraceHandoff {
                    successor_session_id: "child-1".to_string(),
                }),
            },
        );

        let started_json = serde_json::to_value(started).unwrap();
        assert_eq!(started_json["type"], "tool_call_started");
        assert_eq!(started_json["call_id"], "call-1");

        let completed_json = serde_json::to_value(completed).unwrap();
        assert_eq!(completed_json["type"], "turn_completed");
        assert_eq!(completed_json["handoff"]["successor_session_id"], "child-1");
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
