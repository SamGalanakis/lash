use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

pub const TRACE_SCHEMA_VERSION: u32 = 1;

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

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceTokenUsage {
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cached_input_tokens: i64,
    pub reasoning_tokens: i64,
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

pub trait TraceSink: Send + Sync {
    fn append(&self, record: &TraceRecord);
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
    fn append(&self, record: &TraceRecord) {
        let Ok(line) = serde_json::to_string(record) else {
            return;
        };
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path);
        let Ok(mut file) = file else {
            return;
        };
        let _guard = self.lock.lock().ok();
        let _ = writeln!(file, "{line}");
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
        ));
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("\"type\":\"custom\""));
        assert!(text.contains("\"session_id\":\"root\""));
    }
}
