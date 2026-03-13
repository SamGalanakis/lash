use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::mpsc as std_mpsc;
use std::thread::JoinHandle;

use lashlang::{
    ExecuteError, ParseError as FlowParseError, Snapshot as FlowSnapshot, State as FlowState,
    ToolHost, ToolHostError, Value as FlowValue,
};
use serde_json::{Value, json};

#[derive(Debug)]
pub enum PythonRequest {
    Init {
        tools_json: String,
        agent_id: String,
        headless: bool,
        capabilities_json: String,
    },
    Exec {
        id: String,
        code: String,
    },
    Snapshot {
        id: String,
    },
    Restore {
        id: String,
        data: String,
    },
    Reset {
        id: String,
    },
    Reconfigure {
        tools_json: String,
        capabilities_json: String,
        generation: u64,
    },
    CheckComplete {
        code: String,
    },
    Shutdown,
}

#[derive(Debug)]
pub enum PythonResponse {
    Ready,
    ToolCall {
        id: String,
        name: String,
        args: String,
        result_tx: std_mpsc::Sender<String>,
    },
    Message {
        text: String,
        kind: String,
    },
    ExecResult {
        id: String,
        output: String,
        observations: Vec<String>,
        error: Option<String>,
    },
    SnapshotResult {
        id: String,
        data: String,
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
    AskUser {
        question: String,
        options: Vec<String>,
        result_tx: std_mpsc::Sender<String>,
    },
}

pub struct PythonRuntime {
    request_tx: std_mpsc::Sender<PythonRequest>,
    response_rx: std_mpsc::Receiver<PythonResponse>,
    thread: Option<JoinHandle<()>>,
}

impl PythonRuntime {
    pub fn start() -> Result<Self, std::io::Error> {
        let (request_tx, request_rx) = std_mpsc::channel::<PythonRequest>();
        let (response_tx, response_rx) = std_mpsc::channel::<PythonResponse>();

        let thread = std::thread::Builder::new()
            .name("lashlang-runtime".into())
            .spawn(move || runtime_thread_main(request_rx, response_tx))?;

        Ok(Self {
            request_tx,
            response_rx,
            thread: Some(thread),
        })
    }

    pub fn send(&self, request: PythonRequest) -> Result<(), std::io::Error> {
        self.request_tx.send(request).map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::BrokenPipe, "executor thread gone")
        })
    }

    pub fn recv(&self) -> Result<PythonResponse, std::io::Error> {
        self.response_rx.recv().map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::BrokenPipe, "executor thread gone")
        })
    }

    pub fn try_recv(&self) -> Result<Option<PythonResponse>, std::io::Error> {
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

impl Drop for PythonRuntime {
    fn drop(&mut self) {
        let _ = self.request_tx.send(PythonRequest::Shutdown);
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

#[derive(Clone, Debug, Default)]
struct RuntimeConfig {
    agent_id: String,
    headless: bool,
}

impl RuntimeConfig {
    fn apply(
        &mut self,
        _tools_json: &str,
        agent_id: String,
        headless: bool,
        _capabilities_json: &str,
    ) {
        self.agent_id = agent_id;
        self.headless = headless;
    }
}

struct RuntimeState {
    config: RuntimeConfig,
    repl: FlowState,
}

impl RuntimeState {
    fn new() -> Self {
        Self {
            config: RuntimeConfig::default(),
            repl: FlowState::new(),
        }
    }
}

fn runtime_thread_main(
    request_rx: std_mpsc::Receiver<PythonRequest>,
    response_tx: std_mpsc::Sender<PythonResponse>,
) {
    let mut state = RuntimeState::new();

    while let Ok(request) = request_rx.recv() {
        match request {
            PythonRequest::Init {
                tools_json,
                agent_id,
                headless,
                capabilities_json,
            } => {
                state
                    .config
                    .apply(&tools_json, agent_id, headless, &capabilities_json);
                let _ = response_tx.send(PythonResponse::Ready);
            }
            PythonRequest::Exec { id, code } => {
                let result = execute_code(&mut state, &code, &response_tx);
                let _ = response_tx.send(PythonResponse::ExecResult {
                    id,
                    output: result.output,
                    observations: result.observations,
                    error: result.error,
                });
            }
            PythonRequest::Snapshot { id } => {
                let _ = response_tx.send(PythonResponse::SnapshotResult {
                    id,
                    data: snapshot_runtime(&state.repl).unwrap_or_default(),
                });
            }
            PythonRequest::Restore { id, data } => {
                let error = match restore_runtime(&data) {
                    Ok(repl) => {
                        state.repl = repl;
                        None
                    }
                    Err(err) => Some(err),
                };
                let _ = response_tx.send(PythonResponse::ExecResult {
                    id,
                    output: String::new(),
                    observations: Vec::new(),
                    error,
                });
            }
            PythonRequest::Reset { id } => {
                state.repl = FlowState::new();
                let _ = response_tx.send(PythonResponse::ResetResult { id });
            }
            PythonRequest::Reconfigure {
                tools_json,
                capabilities_json,
                generation,
            } => {
                state.config.apply(
                    &tools_json,
                    state.config.agent_id.clone(),
                    state.config.headless,
                    &capabilities_json,
                );
                let _ = response_tx.send(PythonResponse::ReconfigureResult {
                    generation,
                    error: None,
                });
            }
            PythonRequest::CheckComplete { code } => {
                let _ = response_tx.send(PythonResponse::CheckCompleteResult {
                    is_complete: is_code_complete(&code),
                });
            }
            PythonRequest::Shutdown => break,
        }
    }
}

struct ExecOutcome {
    output: String,
    observations: Vec<String>,
    error: Option<String>,
}

fn execute_code(
    state: &mut RuntimeState,
    code: &str,
    response_tx: &std_mpsc::Sender<PythonResponse>,
) -> ExecOutcome {
    let observations = Mutex::new(Vec::new());
    let host = HostBridge {
        response_tx,
        agent_id: state.config.agent_id.clone(),
        headless: state.config.headless,
        observations: &observations,
    };

    match lashlang::execute(code, &mut state.repl, &host) {
        Ok(value) => {
            let final_text = format_output_value(&value);
            if !final_text.is_empty() {
                let _ = response_tx.send(PythonResponse::Message {
                    text: final_text,
                    kind: "final".into(),
                });
            }
            ExecOutcome {
                output: String::new(),
                observations: observations.into_inner().unwrap_or_default(),
                error: None,
            }
        }
        Err(ExecuteError::Parse(err)) => ExecOutcome {
            output: String::new(),
            observations: observations.into_inner().unwrap_or_default(),
            error: Some(format_parse_error(code, &err)),
        },
        Err(ExecuteError::Runtime(err)) => ExecOutcome {
            output: String::new(),
            observations: observations.into_inner().unwrap_or_default(),
            error: Some(err.to_string()),
        },
    }
}

struct HostBridge<'a> {
    response_tx: &'a std_mpsc::Sender<PythonResponse>,
    agent_id: String,
    headless: bool,
    observations: &'a Mutex<Vec<String>>,
}

impl ToolHost for HostBridge<'_> {
    fn call(
        &self,
        name: &str,
        args: &HashMap<String, FlowValue>,
    ) -> Result<FlowValue, ToolHostError> {
        if name == "ask" {
            if self.headless {
                return Err(ToolHostError::new(
                    "ask is unavailable in headless sessions",
                ));
            }
            let question = args
                .get("question")
                .and_then(as_flow_string)
                .ok_or_else(|| ToolHostError::new("ask requires a string `question`"))?;
            let options = args
                .get("options")
                .map(parse_string_list)
                .transpose()?
                .unwrap_or_default();
            let answer = ask_user(self.response_tx, question, options)?;
            return Ok(FlowValue::String(answer));
        }

        let mut payload = flow_to_json_value(&FlowValue::Record(args.clone()));
        if let Some(obj) = payload.as_object_mut() {
            obj.entry("__agent_id__".to_string())
                .or_insert_with(|| Value::String(self.agent_id.clone()));
        }

        let result_rx = send_tool_call(self.response_tx, name, payload)?;
        wait_tool_result(result_rx)
    }

    fn observe(&self, value: &FlowValue) -> Result<(), ToolHostError> {
        let text = format_output_value(value);
        self.observations
            .lock()
            .map_err(|_| ToolHostError::new("observation buffer poisoned"))?
            .push(text);
        Ok(())
    }
}

fn ask_user(
    response_tx: &std_mpsc::Sender<PythonResponse>,
    question: String,
    options: Vec<String>,
) -> Result<String, ToolHostError> {
    let (result_tx, result_rx) = std_mpsc::channel();
    response_tx
        .send(PythonResponse::AskUser {
            question,
            options,
            result_tx,
        })
        .map_err(|_| ToolHostError::new("failed to send prompt to host"))?;
    result_rx
        .recv()
        .map_err(|_| ToolHostError::new("ask_user channel closed"))
}

fn send_tool_call(
    response_tx: &std_mpsc::Sender<PythonResponse>,
    name: &str,
    payload: Value,
) -> Result<std_mpsc::Receiver<String>, ToolHostError> {
    let (result_tx, result_rx) = std_mpsc::channel();
    response_tx
        .send(PythonResponse::ToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name: name.to_string(),
            args: payload.to_string(),
            result_tx,
        })
        .map_err(|_| ToolHostError::new(format!("failed to dispatch tool `{name}`")))?;
    Ok(result_rx)
}

fn wait_tool_result(result_rx: std_mpsc::Receiver<String>) -> Result<FlowValue, ToolHostError> {
    let result_json = result_rx
        .recv()
        .map_err(|_| ToolHostError::new("tool result channel closed"))?;
    parse_tool_reply(&result_json)
}

fn parse_tool_reply(result_json: &str) -> Result<FlowValue, ToolHostError> {
    let parsed: Value = serde_json::from_str(result_json)
        .map_err(|err| ToolHostError::new(format!("invalid tool reply: {err}")))?;
    let success = parsed
        .get("success")
        .and_then(Value::as_bool)
        .ok_or_else(|| ToolHostError::new("tool reply missing `success`"))?;
    let raw_result = parsed
        .get("result")
        .and_then(Value::as_str)
        .ok_or_else(|| ToolHostError::new("tool reply missing `result`"))?;

    let decoded = serde_json::from_str::<Value>(raw_result)
        .unwrap_or_else(|_| Value::String(raw_result.to_string()));
    if success {
        Ok(json_to_flow_value(decoded))
    } else {
        Err(ToolHostError::new(tool_error_message(decoded)))
    }
}

fn tool_error_message(value: Value) -> String {
    match value {
        Value::String(text) => text,
        other => serde_json::to_string(&other).unwrap_or_else(|_| "tool call failed".to_string()),
    }
}

fn snapshot_runtime(repl: &FlowState) -> Result<String, String> {
    serde_json::to_string(&repl.snapshot()).map_err(|err| format!("failed to snapshot REPL: {err}"))
}

fn restore_runtime(data: &str) -> Result<FlowState, String> {
    let snapshot: FlowSnapshot =
        serde_json::from_str(data).map_err(|err| format!("failed to restore REPL: {err}"))?;
    Ok(FlowState::from_snapshot(snapshot))
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

fn as_flow_string(value: &FlowValue) -> Option<String> {
    match value {
        FlowValue::String(text) => Some(text.clone()),
        _ => None,
    }
}

fn parse_string_list(value: &FlowValue) -> Result<Vec<String>, ToolHostError> {
    match value {
        FlowValue::List(values) => values
            .iter()
            .map(|value| {
                as_flow_string(value)
                    .ok_or_else(|| ToolHostError::new("ask `options` must contain only strings"))
            })
            .collect(),
        _ => Err(ToolHostError::new(
            "ask `options` must be a list of strings",
        )),
    }
}

fn flow_to_json_value(value: &FlowValue) -> Value {
    match value {
        FlowValue::Null => Value::Null,
        FlowValue::Bool(value) => Value::Bool(*value),
        FlowValue::Number(value) => json!(value),
        FlowValue::String(value) => Value::String(value.clone()),
        FlowValue::List(values) => Value::Array(values.iter().map(flow_to_json_value).collect()),
        FlowValue::Record(record) => Value::Object(
            record
                .iter()
                .map(|(key, value)| (key.clone(), flow_to_json_value(value)))
                .collect(),
        ),
    }
}

fn json_to_flow_value(value: Value) -> FlowValue {
    match value {
        Value::Null => FlowValue::Null,
        Value::Bool(value) => FlowValue::Bool(value),
        Value::Number(value) => FlowValue::Number(value.as_f64().unwrap_or_default()),
        Value::String(value) => FlowValue::String(value),
        Value::Array(values) => {
            FlowValue::List(values.into_iter().map(json_to_flow_value).collect())
        }
        Value::Object(map) => FlowValue::Record(
            map.into_iter()
                .map(|(key, value)| (key, json_to_flow_value(value)))
                .collect(),
        ),
    }
}

fn format_output_value(value: &FlowValue) -> String {
    match value {
        FlowValue::Null => String::new(),
        FlowValue::String(text) => text.clone(),
        FlowValue::Bool(value) => value.to_string(),
        FlowValue::Number(value) => value.to_string(),
        FlowValue::List(_) | FlowValue::Record(_) => {
            serde_json::to_string(&flow_to_json_value(value)).unwrap_or_default()
        }
    }
}

fn format_parse_error(source: &str, err: &FlowParseError) -> String {
    let offset = match err {
        FlowParseError::Lex(lashlang::LexError::UnexpectedChar { offset, .. })
        | FlowParseError::Lex(lashlang::LexError::UnterminatedString { offset })
        | FlowParseError::Lex(lashlang::LexError::InvalidNumber { offset, .. }) => *offset,
        FlowParseError::Expected { span, .. } | FlowParseError::Unexpected { span, .. } => {
            span.start
        }
    };
    let (line_no, column, line_text) = line_context(source, offset);
    let caret_pad = " ".repeat(column.saturating_sub(1));
    let mut message = format!(
        "{} at line {}, column {}\n{}\n{}^",
        err, line_no, column, line_text, caret_pad
    );
    if matches!(
        err,
        FlowParseError::Unexpected { found, .. } if found == "`if`"
    ) {
        message.push_str("\nhint: statement `if { ... } else { ... }` is not valid inside expressions; use `cond ? yes : no`");
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

    #[test]
    fn tool_reply_success_round_trips_json() {
        let reply = json!({
            "success": true,
            "result": json!({"id": 7, "name": "lash"}).to_string(),
        })
        .to_string();

        let value = parse_tool_reply(&reply).expect("reply should parse");
        let FlowValue::Record(record) = value else {
            panic!("expected record");
        };
        assert_eq!(record["id"], FlowValue::Number(7.0));
        assert_eq!(record["name"], FlowValue::String("lash".into()));
    }

    #[test]
    fn tool_reply_error_uses_payload_text() {
        let reply = json!({
            "success": false,
            "result": json!("missing path").to_string(),
        })
        .to_string();

        let err = parse_tool_reply(&reply).expect_err("reply should fail");
        assert_eq!(err.to_string(), "missing path");
    }

    #[test]
    fn completion_check_distinguishes_incomplete_inputs() {
        assert!(!is_code_complete("if true {"));
        assert!(!is_code_complete("finish \"unterminated"));
        assert!(is_code_complete("finish 1"));
        assert!(is_code_complete("oops ]"));
    }

    #[test]
    fn parse_error_format_includes_line_context_and_hint() {
        let err = lashlang::parse("x = if true { 1 } else { 2 }").expect_err("parse should fail");
        let formatted = format_parse_error("x = if true { 1 } else { 2 }", &err);
        assert!(formatted.contains("line 1, column 5"));
        assert!(formatted.contains("x = if true { 1 } else { 2 }"));
        assert!(formatted.contains("^"));
        assert!(formatted.contains("use `cond ? yes : no`"));
    }

    #[test]
    fn parse_error_format_handles_lex_offsets() {
        let err = lashlang::parse("a = @").expect_err("parse should fail");
        let formatted = format_parse_error("a = @", &err);
        assert!(formatted.contains("line 1, column 5"));
        assert!(formatted.contains("a = @"));
    }

    #[test]
    fn execute_code_captures_observations_separately_from_final_output() {
        let mut state = RuntimeState::new();
        let (tx, _rx) = std_mpsc::channel();

        let result = execute_code(
            &mut state,
            r#"
            observe { ok: true, count: 2 }
            finish "done"
            "#,
            &tx,
        );

        assert_eq!(result.output, "");
        assert_eq!(result.error, None);
        assert_eq!(result.observations.len(), 1);
        let parsed: serde_json::Value =
            serde_json::from_str(&result.observations[0]).expect("observation should be json");
        assert_eq!(parsed, serde_json::json!({"ok": true, "count": 2.0}));
    }

    #[test]
    fn snapshot_round_trips_state() {
        let mut state = FlowState::new();
        let host = TestHost;
        lashlang::execute("value = 41 finish value", &mut state, &host).expect("exec succeeds");

        let snap = snapshot_runtime(&state).expect("snapshot succeeds");
        let restored = restore_runtime(&snap).expect("restore succeeds");
        assert_eq!(state.snapshot(), restored.snapshot());
    }

    struct TestHost;

    impl ToolHost for TestHost {
        fn call(
            &self,
            _name: &str,
            _args: &HashMap<String, FlowValue>,
        ) -> Result<FlowValue, ToolHostError> {
            Ok(FlowValue::Null)
        }
    }
}
