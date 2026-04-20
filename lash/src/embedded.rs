use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::mpsc as std_mpsc;
use std::thread::JoinHandle;

use lashlang::{
    CompiledProgram, ExecutionOutcome, ExecutionScratch, ParseError as FlowParseError,
    Record as FlowRecord, Snapshot as FlowSnapshot, State as FlowState, ToolHost, ToolHostError,
    Value as FlowValue,
};
use serde_json::Value;

use crate::plugin::{ToolResultProjectionPluginConfig, truncate_observation_text};

#[derive(Debug)]
pub enum LashlangRequest {
    Init {
        tools_json: String,
        session_id: String,
        observe_projection: ToolResultProjectionPluginConfig,
    },
    Exec {
        id: String,
        code: String,
        /// When true, a `submit <expr>` from inside the lashlang
        /// program is captured (instead of being treated as an error
        /// like the chat-style contract). The captured value flows
        /// back through `LashlangResponse::ExecResult::terminal_finish`.
        accept_finish: bool,
    },
    Snapshot {
        id: String,
    },
    Restore {
        id: String,
        data: String,
    },
    PatchGlobals {
        id: String,
        set: serde_json::Map<String, Value>,
        unset: Vec<String>,
    },
    Reset {
        id: String,
    },
    Reconfigure {
        tools_json: String,
        generation: u64,
        observe_projection: ToolResultProjectionPluginConfig,
    },
    CheckComplete {
        code: String,
    },
    Shutdown,
}

#[derive(Debug)]
pub enum LashlangResponse {
    Ready,
    ToolCall {
        id: String,
        name: String,
        args: Value,
        result_tx: std_mpsc::Sender<LashlangToolReply>,
    },
    StartToolCall {
        id: String,
        name: String,
        args: Value,
        result_tx: std_mpsc::Sender<LashlangToolReply>,
    },
    AwaitToolHandle {
        id: String,
        handle: Value,
        result_tx: std_mpsc::Sender<LashlangToolReply>,
    },
    CancelToolHandle {
        id: String,
        handle: Value,
        result_tx: std_mpsc::Sender<LashlangToolReply>,
    },
    ExecResult {
        id: String,
        output: String,
        observations: Vec<String>,
        error: Option<String>,
        /// `Some(value)` only when the surrounding session was started
        /// with `accept_finish: true` AND the lashlang program ended
        /// with `submit <expr>`. The value is JSON-encoded.
        terminal_finish: Option<Value>,
    },
    SnapshotResult {
        id: String,
        data: String,
    },
    PatchGlobalsResult {
        id: String,
        error: Option<String>,
    },
    ResetResult {
        id: String,
    },
    ReconfigureResult {
        generation: u64,
        error: Option<String>,
    },
    CheckCompleteResult {
        is_complete: bool,
    },
}

#[derive(Clone, Debug)]
pub struct LashlangToolReply {
    pub success: bool,
    pub result: Value,
}

impl LashlangToolReply {
    pub(crate) fn success(result: Value) -> Self {
        Self {
            success: true,
            result,
        }
    }

    pub(crate) fn error(result: Value) -> Self {
        Self {
            success: false,
            result,
        }
    }
}

pub struct LashlangRuntime {
    request_tx: std_mpsc::Sender<LashlangRequest>,
    response_rx: std_mpsc::Receiver<LashlangResponse>,
    thread: Option<JoinHandle<()>>,
}

impl LashlangRuntime {
    pub fn start() -> Result<Self, std::io::Error> {
        let (request_tx, request_rx) = std_mpsc::channel::<LashlangRequest>();
        let (response_tx, response_rx) = std_mpsc::channel::<LashlangResponse>();

        let thread = std::thread::Builder::new()
            .name("lashlang-runtime".into())
            .spawn(move || runtime_thread_main(request_rx, response_tx))?;

        Ok(Self {
            request_tx,
            response_rx,
            thread: Some(thread),
        })
    }

    pub fn send(&self, request: LashlangRequest) -> Result<(), std::io::Error> {
        self.request_tx.send(request).map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::BrokenPipe, "executor thread gone")
        })
    }

    pub fn recv(&self) -> Result<LashlangResponse, std::io::Error> {
        self.response_rx.recv().map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::BrokenPipe, "executor thread gone")
        })
    }

    pub fn try_recv(&self) -> Result<Option<LashlangResponse>, std::io::Error> {
        match self.response_rx.try_recv() {
            Ok(msg) => Ok(Some(msg)),
            Err(std_mpsc::TryRecvError::Empty) => Ok(None),
            Err(std_mpsc::TryRecvError::Disconnected) => Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "executor thread gone",
            )),
        }
    }
}

impl Drop for LashlangRuntime {
    fn drop(&mut self) {
        let _ = self.request_tx.send(LashlangRequest::Shutdown);
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

#[derive(Clone, Debug, Default)]
struct RuntimeConfig {
    session_id: String,
    observe_projection: ToolResultProjectionPluginConfig,
}

impl RuntimeConfig {
    fn apply(
        &mut self,
        _tools_json: &str,
        session_id: String,
        observe_projection: ToolResultProjectionPluginConfig,
    ) {
        self.session_id = session_id;
        self.observe_projection = observe_projection;
    }
}

struct RuntimeState {
    config: RuntimeConfig,
    rlm: FlowState,
    program_cache: ProgramCache,
    scratch: ExecutionScratch,
}

impl RuntimeState {
    fn new() -> Self {
        lashlang::prewarm();
        Self {
            config: RuntimeConfig::default(),
            rlm: FlowState::new(),
            program_cache: ProgramCache::default(),
            scratch: ExecutionScratch::new(),
        }
    }
}

const PROGRAM_CACHE_CAPACITY: usize = 64;

#[derive(Default)]
struct ProgramCache {
    entries: VecDeque<CachedProgram>,
}

struct CachedProgram {
    source: String,
    compiled: Arc<CompiledProgram>,
}

impl ProgramCache {
    fn get_or_compile(&mut self, source: &str) -> Result<Arc<CompiledProgram>, FlowParseError> {
        if let Some(index) = self.entries.iter().position(|entry| entry.source == source) {
            let entry = self
                .entries
                .remove(index)
                .expect("cache index came from existing entry");
            let compiled = entry.compiled.clone();
            self.entries.push_back(entry);
            return Ok(compiled);
        }

        let compiled = Arc::new(lashlang::compile_source(source)?);
        if self.entries.len() == PROGRAM_CACHE_CAPACITY {
            self.entries.pop_front();
        }
        self.entries.push_back(CachedProgram {
            source: source.to_string(),
            compiled: compiled.clone(),
        });
        Ok(compiled)
    }
}

fn runtime_thread_main(
    request_rx: std_mpsc::Receiver<LashlangRequest>,
    response_tx: std_mpsc::Sender<LashlangResponse>,
) {
    let mut state = RuntimeState::new();

    while let Ok(request) = request_rx.recv() {
        match request {
            LashlangRequest::Init {
                tools_json,
                session_id,
                observe_projection,
            } => {
                state
                    .config
                    .apply(&tools_json, session_id, observe_projection);
                let _ = response_tx.send(LashlangResponse::Ready);
            }
            LashlangRequest::Exec {
                id,
                code,
                accept_finish,
            } => {
                let result = execute_code(&mut state, &code, accept_finish, &response_tx);
                let _ = response_tx.send(LashlangResponse::ExecResult {
                    id,
                    output: result.output,
                    observations: result.observations,
                    error: result.error,
                    terminal_finish: result.terminal_finish,
                });
            }
            LashlangRequest::Snapshot { id } => {
                let _ = response_tx.send(LashlangResponse::SnapshotResult {
                    id,
                    data: snapshot_runtime(&state.rlm).unwrap_or_default(),
                });
            }
            LashlangRequest::Restore { id, data } => {
                let error = match restore_runtime(&data) {
                    Ok(rlm) => {
                        state.rlm = rlm;
                        None
                    }
                    Err(err) => Some(err),
                };
                let _ = response_tx.send(LashlangResponse::ExecResult {
                    id,
                    output: String::new(),
                    observations: Vec::new(),
                    error,
                    terminal_finish: None,
                });
            }
            LashlangRequest::PatchGlobals { id, set, unset } => {
                let error = patch_globals(&mut state.rlm, set, unset).err();
                let _ = response_tx.send(LashlangResponse::PatchGlobalsResult { id, error });
            }
            LashlangRequest::Reset { id } => {
                state.rlm = FlowState::new();
                let _ = response_tx.send(LashlangResponse::ResetResult { id });
            }
            LashlangRequest::Reconfigure {
                tools_json,
                generation,
                observe_projection,
            } => {
                state.config.apply(
                    &tools_json,
                    state.config.session_id.clone(),
                    observe_projection,
                );
                let _ = response_tx.send(LashlangResponse::ReconfigureResult {
                    generation,
                    error: None,
                });
            }
            LashlangRequest::CheckComplete { code } => {
                let _ = response_tx.send(LashlangResponse::CheckCompleteResult {
                    is_complete: is_code_complete(&code),
                });
            }
            LashlangRequest::Shutdown => break,
        }
    }
}

struct ExecOutcome {
    output: String,
    observations: Vec<String>,
    error: Option<String>,
    terminal_finish: Option<Value>,
}

fn execute_code(
    state: &mut RuntimeState,
    code: &str,
    accept_finish: bool,
    response_tx: &std_mpsc::Sender<LashlangResponse>,
) -> ExecOutcome {
    let compiled = match state.program_cache.get_or_compile(code) {
        Ok(compiled) => compiled,
        Err(err) => {
            return ExecOutcome {
                output: String::new(),
                observations: Vec::new(),
                error: Some(format_parse_error(code, &err)),
                terminal_finish: None,
            };
        }
    };

    let observations = Mutex::new(Vec::new());
    let host = HostBridge {
        response_tx,
        session_id: state.config.session_id.clone(),
        observe_projection: &state.config.observe_projection,
        observations: &observations,
    };

    let _ = accept_finish; // schema validation lives upstream in mode.rs
    match lashlang::execute_compiled_traced_with_scratch(
        &compiled,
        &mut state.rlm,
        &host,
        &mut state.scratch,
    ) {
        Ok(ExecutionOutcome::Finished(value)) => ExecOutcome {
            output: String::new(),
            observations: observations.into_inner().unwrap_or_default(),
            error: None,
            terminal_finish: Some(flow_to_json_value(&value)),
        },
        Ok(ExecutionOutcome::Continued) => ExecOutcome {
            output: String::new(),
            observations: observations.into_inner().unwrap_or_default(),
            error: None,
            terminal_finish: None,
        },
        Err(failure) => ExecOutcome {
            output: String::new(),
            observations: observations.into_inner().unwrap_or_default(),
            error: Some(lashlang::format_runtime_diagnostic(
                code,
                &failure.error,
                failure.span,
            )),
            terminal_finish: None,
        },
    }
}

struct HostBridge<'a> {
    response_tx: &'a std_mpsc::Sender<LashlangResponse>,
    session_id: String,
    observe_projection: &'a ToolResultProjectionPluginConfig,
    observations: &'a Mutex<Vec<String>>,
}

impl ToolHost for HostBridge<'_> {
    fn call(&self, name: &str, args: &FlowRecord) -> Result<FlowValue, ToolHostError> {
        let mut payload = flow_to_json_value(&FlowValue::Record(Arc::new(args.clone())));
        if let Some(obj) = payload.as_object_mut() {
            obj.entry("__session_id__".to_string())
                .or_insert_with(|| Value::String(self.session_id.clone()));
        }

        let result_rx = send_tool_call(self.response_tx, name, payload)?;
        wait_tool_result(result_rx)
    }

    fn start_call(&self, name: &str, args: &FlowRecord) -> Result<FlowValue, ToolHostError> {
        let mut payload = flow_to_json_value(&FlowValue::Record(Arc::new(args.clone())));
        if let Some(obj) = payload.as_object_mut() {
            obj.entry("__session_id__".to_string())
                .or_insert_with(|| Value::String(self.session_id.clone()));
        }

        let result_rx = send_start_tool_call(self.response_tx, name, payload)?;
        wait_tool_result(result_rx)
    }

    fn await_handle(&self, handle: &FlowValue) -> Result<FlowValue, ToolHostError> {
        let result_rx = send_await_tool_handle(self.response_tx, flow_to_json_value(handle))?;
        wait_tool_result(result_rx)
    }

    fn cancel_handle(&self, handle: &FlowValue) -> Result<FlowValue, ToolHostError> {
        let result_rx = send_cancel_tool_handle(self.response_tx, flow_to_json_value(handle))?;
        wait_tool_result(result_rx)
    }

    fn print(&self, value: &FlowValue) -> Result<(), ToolHostError> {
        let text = truncate_observation_text(&format_output_value(value), self.observe_projection);
        self.observations
            .lock()
            .map_err(|_| ToolHostError::new("observation buffer poisoned"))?
            .push(text);
        Ok(())
    }
}

fn send_tool_call(
    response_tx: &std_mpsc::Sender<LashlangResponse>,
    name: &str,
    payload: Value,
) -> Result<std_mpsc::Receiver<LashlangToolReply>, ToolHostError> {
    let (result_tx, result_rx) = std_mpsc::channel();
    response_tx
        .send(LashlangResponse::ToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name: name.to_string(),
            args: payload,
            result_tx,
        })
        .map_err(|_| ToolHostError::new(format!("failed to dispatch tool `{name}`")))?;
    Ok(result_rx)
}

fn send_start_tool_call(
    response_tx: &std_mpsc::Sender<LashlangResponse>,
    name: &str,
    payload: Value,
) -> Result<std_mpsc::Receiver<LashlangToolReply>, ToolHostError> {
    let (result_tx, result_rx) = std_mpsc::channel();
    response_tx
        .send(LashlangResponse::StartToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name: name.to_string(),
            args: payload,
            result_tx,
        })
        .map_err(|_| ToolHostError::new(format!("failed to start tool `{name}`")))?;
    Ok(result_rx)
}

fn send_await_tool_handle(
    response_tx: &std_mpsc::Sender<LashlangResponse>,
    handle: Value,
) -> Result<std_mpsc::Receiver<LashlangToolReply>, ToolHostError> {
    let (result_tx, result_rx) = std_mpsc::channel();
    response_tx
        .send(LashlangResponse::AwaitToolHandle {
            id: uuid::Uuid::new_v4().to_string(),
            handle,
            result_tx,
        })
        .map_err(|_| ToolHostError::new("failed to await async tool handle"))?;
    Ok(result_rx)
}

fn send_cancel_tool_handle(
    response_tx: &std_mpsc::Sender<LashlangResponse>,
    handle: Value,
) -> Result<std_mpsc::Receiver<LashlangToolReply>, ToolHostError> {
    let (result_tx, result_rx) = std_mpsc::channel();
    response_tx
        .send(LashlangResponse::CancelToolHandle {
            id: uuid::Uuid::new_v4().to_string(),
            handle,
            result_tx,
        })
        .map_err(|_| ToolHostError::new("failed to cancel async tool handle"))?;
    Ok(result_rx)
}

fn wait_tool_result(
    result_rx: std_mpsc::Receiver<LashlangToolReply>,
) -> Result<FlowValue, ToolHostError> {
    let reply = result_rx
        .recv()
        .map_err(|_| ToolHostError::new("tool result channel closed"))?;
    decode_tool_reply(reply)
}

fn decode_tool_reply(reply: LashlangToolReply) -> Result<FlowValue, ToolHostError> {
    if reply.success {
        Ok(json_to_flow_value(reply.result))
    } else {
        Err(ToolHostError::new(tool_error_message(reply.result)))
    }
}

fn tool_error_message(value: Value) -> String {
    match value {
        Value::String(text) => text,
        other => serde_json::to_string(&other).unwrap_or_else(|_| "tool call failed".to_string()),
    }
}

fn snapshot_runtime(rlm: &FlowState) -> Result<String, String> {
    serde_json::to_string(&rlm.snapshot()).map_err(|err| format!("failed to snapshot RLM: {err}"))
}

fn restore_runtime(data: &str) -> Result<FlowState, String> {
    let snapshot: FlowSnapshot =
        serde_json::from_str(data).map_err(|err| format!("failed to restore RLM: {err}"))?;
    Ok(FlowState::from_snapshot(snapshot))
}

fn patch_globals(
    rlm: &mut FlowState,
    set: serde_json::Map<String, Value>,
    unset: Vec<String>,
) -> Result<(), String> {
    let mut snapshot = rlm.snapshot();
    for key in unset {
        snapshot.globals.remove(&key);
    }
    for (key, value) in set {
        snapshot.globals.insert(key, json_to_flow_value(value));
    }
    *rlm = FlowState::from_snapshot(snapshot);
    Ok(())
}

fn is_code_complete(code: &str) -> bool {
    let trimmed = code.trim();
    if trimmed.is_empty() {
        return true;
    }

    match lashlang::parse(trimmed) {
        Ok(_) => true,
        Err(FlowParseError::Lex(lashlang::LexError::UnterminatedString { .. })) => false,
        Err(FlowParseError::Expected { found, .. } | FlowParseError::Unexpected { found, .. }) => {
            found != "end of input"
        }
        Err(FlowParseError::Lex(_)) => true,
    }
}

fn flow_to_json_value(value: &FlowValue) -> Value {
    match value {
        FlowValue::Null => Value::Null,
        FlowValue::Bool(value) => Value::Bool(*value),
        FlowValue::Number(value) => json_number(*value),
        FlowValue::String(value) => Value::String(value.to_string()),
        FlowValue::List(values) => Value::Array(values.iter().map(flow_to_json_value).collect()),
        FlowValue::Record(record) => Value::Object(
            record
                .iter()
                .map(|(key, value)| (key.to_string(), flow_to_json_value(value)))
                .collect(),
        ),
    }
}

fn json_number(value: f64) -> Value {
    if value.is_finite() && value.fract() == 0.0 {
        let as_i64 = value as i64 as f64;
        if as_i64 == value {
            return Value::Number(serde_json::Number::from(value as i64));
        }
        let as_u64 = value as u64 as f64;
        if as_u64 == value {
            return Value::Number(serde_json::Number::from(value as u64));
        }
    }
    serde_json::Number::from_f64(value)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

fn json_to_flow_value(value: Value) -> FlowValue {
    match value {
        Value::Null => FlowValue::Null,
        Value::Bool(value) => FlowValue::Bool(value),
        Value::Number(value) => FlowValue::Number(value.as_f64().unwrap_or_default()),
        Value::String(value) => FlowValue::String(value.into()),
        Value::Array(values) => {
            FlowValue::List(values.into_iter().map(json_to_flow_value).collect())
        }
        Value::Object(map) => FlowValue::Record(Arc::new(
            map.into_iter()
                .map(|(key, value)| (key, json_to_flow_value(value)))
                .collect::<FlowRecord>(),
        )),
    }
}

fn format_output_value(value: &FlowValue) -> String {
    match value {
        FlowValue::Null => "null".to_string(),
        FlowValue::String(text) => text.to_string(),
        FlowValue::Bool(value) => value.to_string(),
        FlowValue::Number(value) => value.to_string(),
        FlowValue::List(_) | FlowValue::Record(_) => {
            serde_json::to_string(&flow_to_json_value(value)).unwrap_or_else(|_| value.to_string())
        }
    }
}

fn format_parse_error(source: &str, err: &FlowParseError) -> String {
    let offset = err.offset();
    let (line_no, column, line_text) = line_context(source, offset);
    let caret_pad = " ".repeat(column.saturating_sub(1));
    let mut message = format!("line {line_no}, column {column}: {err}\n{line_text}\n{caret_pad}^");
    if matches!(
        err,
        FlowParseError::Unexpected { found, .. } if found == "`if`"
    ) {
        message.push_str("\nhint: use `cond ? yes : no` for inline conditionals");
    }
    if matches!(
        err,
        FlowParseError::Lex(lashlang::LexError::UnexpectedChar { ch: '&', .. })
    ) {
        message.push_str("\nhint: use `and` or `&&`");
    }
    if matches!(
        err,
        FlowParseError::Lex(lashlang::LexError::UnexpectedChar { ch: '|', .. })
    ) {
        message.push_str("\nhint: use `or` or `||`");
    }
    message
}

fn line_context(source: &str, offset: usize) -> (usize, usize, String) {
    let clamped = offset.min(source.len());
    let prefix = &source[..clamped];
    let line_no = prefix.bytes().filter(|b| *b == b'\n').count() + 1;
    let line_start = prefix.rfind('\n').map(|idx| idx + 1).unwrap_or(0);
    let line_end = source[clamped..]
        .find('\n')
        .map(|idx| clamped + idx)
        .unwrap_or(source.len());
    let column = source[line_start..clamped].chars().count() + 1;
    let line_text = source[line_start..line_end].to_string();
    (line_no, column, line_text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::time::Duration;

    fn execute_code_with_tool_reply(
        code: &str,
        tool_name: &str,
        tool_result: serde_json::Value,
    ) -> ExecOutcome {
        let mut state = RuntimeState::new();
        let (tx, rx) = std_mpsc::channel();
        let expected_tool_name = tool_name.to_string();

        let tool_thread = std::thread::spawn(move || {
            let message = rx
                .recv_timeout(Duration::from_secs(1))
                .expect("tool call should be sent");
            let LashlangResponse::ToolCall {
                name, result_tx, ..
            } = message
            else {
                panic!("expected tool call");
            };
            assert_eq!(name, expected_tool_name);
            result_tx
                .send(LashlangToolReply::success(tool_result))
                .expect("tool result should be delivered");
        });

        let result = execute_code(&mut state, code, false, &tx);
        tool_thread.join().expect("tool thread should finish");
        result
    }

    #[test]
    fn tool_reply_success_round_trips_json() {
        let reply = LashlangToolReply::success(json!({"id": 7, "name": "lash"}));

        let value = decode_tool_reply(reply).expect("reply should parse");
        let FlowValue::Record(record) = value else {
            panic!("expected record");
        };
        assert_eq!(record["id"], FlowValue::Number(7.0));
        assert_eq!(record["name"], FlowValue::String("lash".into()));
    }

    #[test]
    fn tool_reply_error_uses_payload_text() {
        let reply = LashlangToolReply::error(json!("missing path"));

        let err = decode_tool_reply(reply).expect_err("reply should fail");
        assert_eq!(err.to_string(), "missing path");
    }

    #[test]
    fn whole_numbers_serialize_as_json_integers_for_tool_args() {
        assert_eq!(json_number(1.0), serde_json::json!(1));
        assert_eq!(json_number(1.5), serde_json::json!(1.5));
    }

    #[test]
    fn completion_check_distinguishes_incomplete_inputs() {
        assert!(!is_code_complete("if true {"));
        assert!(!is_code_complete("submit \"unterminated"));
        assert!(is_code_complete("submit"));
        assert!(is_code_complete("submit 1"));
        assert!(is_code_complete("oops ]"));
    }

    #[test]
    fn program_cache_reuses_exact_source() {
        let mut cache = ProgramCache::default();
        let first = cache
            .get_or_compile("submit 1")
            .expect("program should compile");
        let second = cache
            .get_or_compile("submit 1")
            .expect("program should compile");
        let other = cache
            .get_or_compile("submit 2")
            .expect("program should compile");

        assert!(Arc::ptr_eq(&first, &second));
        assert!(!Arc::ptr_eq(&first, &other));
        assert!(cache.get_or_compile("submit @").is_err());
    }

    #[test]
    fn parse_error_format_includes_line_context_and_hint() {
        let err = lashlang::parse("x = if true { 1 } else { 2 }").expect_err("parse should fail");
        let formatted = format_parse_error("x = if true { 1 } else { 2 }", &err);
        assert!(formatted.starts_with("line 1, column 5: unexpected `if`"));
        assert!(formatted.contains("line 1, column 5"));
        assert!(formatted.contains("x = if true { 1 } else { 2 }"));
        assert!(formatted.contains("^"));
        assert!(formatted.contains("use `cond ? yes : no`"));
    }

    #[test]
    fn parse_error_format_handles_lex_offsets() {
        let err = lashlang::parse("a = @").expect_err("parse should fail");
        let formatted = format_parse_error("a = @", &err);
        assert!(formatted.starts_with("line 1, column 5: unexpected `@`"));
        assert!(formatted.contains("line 1, column 5"));
        assert!(formatted.contains("a = @"));
    }

    #[test]
    fn parse_error_format_adds_operator_hints() {
        let err = lashlang::parse("x = true & false").expect_err("parse should fail");
        let formatted = format_parse_error("x = true & false", &err);
        assert!(formatted.contains("hint: use `and` or `&&`"));
    }

    #[test]
    fn execute_code_runtime_errors_include_source_context() {
        let mut state = RuntimeState::new();
        let (tx, _rx) = std_mpsc::channel();

        let result = execute_code(&mut state, "value = 1\nsubmit value.name", false, &tx);

        let error = result.error.expect("runtime error");
        assert!(error.contains("can't read `.name` from number"), "{error}");
        assert!(error.contains("--> line 2, column 1"), "{error}");
        assert!(error.contains("submit value.name"), "{error}");
    }

    #[test]
    fn execute_code_captures_observations_separately_from_final_output() {
        let mut state = RuntimeState::new();
        let (tx, _rx) = std_mpsc::channel();

        let result = execute_code(
            &mut state,
            r#"
            print { ok: true, count: 2 }
            "#,
            false,
            &tx,
        );

        assert_eq!(result.output, "");
        assert_eq!(result.error, None);
        assert_eq!(result.observations.len(), 1);
        let parsed: serde_json::Value =
            serde_json::from_str(&result.observations[0]).expect("observation should be json");
        assert_eq!(parsed, serde_json::json!({"ok": true, "count": 2}));
    }

    #[test]
    fn execute_code_observe_null_is_not_silent() {
        let mut state = RuntimeState::new();
        let (tx, _rx) = std_mpsc::channel();

        let result = execute_code(
            &mut state,
            r#"
            print null
            "#,
            false,
            &tx,
        );

        assert_eq!(result.error, None);
        assert_eq!(result.observations, vec!["null".to_string()]);
    }

    #[test]
    fn execute_code_can_index_listing_items_directly_under_result_value() {
        let result = execute_code_with_tool_reply(
            r#"
            listing = call glob { pattern: "*.rs" }
            print {
              count: len(listing.value.items),
              first: listing.value.items[0].path
            }
            "#,
            "glob",
            json!({
                "items": [
                    { "path": "src/lib.rs", "kind": "file" },
                    { "path": "src/main.rs", "kind": "file" }
                ],
                "truncated": null,
            }),
        );

        assert_eq!(result.error, None);
        let parsed: serde_json::Value =
            serde_json::from_str(&result.observations[0]).expect("observation should be json");
        assert_eq!(
            parsed,
            json!({
                "count": 2,
                "first": "src/lib.rs",
            })
        );
    }

    #[test]
    fn finish_returns_terminal_value_regardless_of_mode() {
        // `submit <expr>` always terminates the program and delivers the
        // value via `terminal_finish`. Upstream (`lash-sansio/src/mode.rs`)
        // renders it as the turn's final assistant message and validates
        // against a typed schema when one is declared.
        for accept_finish in [false, true] {
            let mut state = RuntimeState::new();
            let (tx, _rx) = std_mpsc::channel();
            let result = execute_code(&mut state, "submit \"all done\"", accept_finish, &tx);
            assert_eq!(result.error, None);
            assert_eq!(result.terminal_finish, Some(serde_json::json!("all done")));
            assert!(result.observations.is_empty());
        }
    }

    #[test]
    fn finish_is_captured_in_typed_mode() {
        let mut state = RuntimeState::new();
        let (tx, _rx) = std_mpsc::channel();

        let result = execute_code(
            &mut state,
            r#"submit { answer: "yes", confidence: 0.9 }"#,
            true,
            &tx,
        );

        assert_eq!(result.error, None);
        let captured = result.terminal_finish.expect("typed finish");
        assert_eq!(captured["answer"], serde_json::json!("yes"));
        assert_eq!(captured["confidence"], serde_json::json!(0.9));
    }

    #[test]
    fn typed_finish_with_string_value_round_trips() {
        let mut state = RuntimeState::new();
        let (tx, _rx) = std_mpsc::channel();

        let result = execute_code(&mut state, r#"submit "all done""#, true, &tx);
        assert_eq!(result.error, None);
        assert_eq!(result.terminal_finish, Some(serde_json::json!("all done")));
    }

    #[test]
    fn typed_mode_continues_when_program_doesnt_finish() {
        let mut state = RuntimeState::new();
        let (tx, _rx) = std_mpsc::channel();

        let result = execute_code(&mut state, "x = 1", true, &tx);
        assert_eq!(result.error, None);
        assert!(result.terminal_finish.is_none());
    }

    #[test]
    fn snapshot_round_trips_state() {
        let mut state = FlowState::new();
        let host = TestHost;
        lashlang::execute("value = 41", &mut state, &host).expect("exec succeeds");

        let snap = snapshot_runtime(&state).expect("snapshot succeeds");
        let restored = restore_runtime(&snap).expect("restore succeeds");
        assert_eq!(state.snapshot(), restored.snapshot());
    }

    struct TestHost;

    impl ToolHost for TestHost {
        fn call(&self, _name: &str, _args: &FlowRecord) -> Result<FlowValue, ToolHostError> {
            Ok(FlowValue::Null)
        }
    }
}
