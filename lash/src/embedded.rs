use std::ffi::CString;
use std::sync::mpsc as std_mpsc;
use std::thread::JoinHandle;

use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::python_home;

const REPL_PY: &str = include_str!("../python/repl.py");

// ── Request/Response types ──

#[derive(Debug)]
pub enum PythonRequest {
    Init { tools_json: String },
    Exec { id: String, code: String },
    Snapshot { id: String },
    Restore { id: String, data: String },
    Reset { id: String },
    CheckComplete { code: String },
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
        error: Option<String>,
    },
    SnapshotResult {
        id: String,
        data: String,
    },
    ResetResult {
        id: String,
    },
    CheckCompleteResult { is_complete: bool },
    AskUser {
        question: String,
        options: Vec<String>,
        result_tx: std_mpsc::Sender<String>,
    },
}

/// Wrapper to mark a `!Sync` type as `Sync` for GIL release.
/// Safety: only used when the inner value is accessed by a single thread.
struct SyncWrapper<T>(T);
unsafe impl<T> Sync for SyncWrapper<T> {}

// ── RustBridge: injected into Python as _rust_bridge ──

#[pyclass]
struct RustBridge {
    response_tx: std_mpsc::Sender<PythonResponse>,
}

#[pymethods]
impl RustBridge {
    /// Called by Python to send messages/results back to Rust.
    fn send_message(&self, json_str: &str) -> PyResult<()> {
        let msg: serde_json::Value = serde_json::from_str(json_str).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("invalid JSON: {e}"))
        })?;

        let msg_type = msg
            .get("type")
            .and_then(|t| t.as_str())
            .unwrap_or("");

        let response = match msg_type {
            "message" => PythonResponse::Message {
                text: msg["text"].as_str().unwrap_or("").to_string(),
                kind: msg["kind"].as_str().unwrap_or("progress").to_string(),
            },
            "exec_result" => PythonResponse::ExecResult {
                id: msg["id"].as_str().unwrap_or("").to_string(),
                output: msg["output"].as_str().unwrap_or("").to_string(),
                error: msg["error"].as_str().map(String::from),
            },
            "snapshot_result" => PythonResponse::SnapshotResult {
                id: msg["id"].as_str().unwrap_or("").to_string(),
                data: msg["data"].as_str().unwrap_or("").to_string(),
            },
            "reset_result" => PythonResponse::ResetResult {
                id: msg["id"].as_str().unwrap_or("").to_string(),
            },
            "ready" => PythonResponse::Ready,
            other => {
                tracing::warn!("unknown message type from Python: {other}");
                return Ok(());
            }
        };

        let _ = self.response_tx.send(response);
        Ok(())
    }

    /// Called by Python to invoke a tool. Blocks until the tool result is ready.
    /// GIL is released while waiting so other Python threads can run.
    fn call_tool(
        &self,
        py: Python<'_>,
        call_id: String,
        name: String,
        args_json: String,
    ) -> PyResult<String> {
        let (result_tx, result_rx) = std_mpsc::channel();

        let _ = self.response_tx.send(PythonResponse::ToolCall {
            id: call_id,
            name,
            args: args_json,
            result_tx,
        });

        // Release GIL while waiting for the tool result.
        // Safety: result_rx is only accessed from this thread — wrapping
        // in SyncWrapper is sound because no concurrent access occurs.
        let wrapper = SyncWrapper(result_rx);
        py.detach(move || {
            wrapper.0.recv().map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "tool result channel closed: {e}"
                ))
            })
        })
    }

    /// Called by Python to ask the user a question. Blocks until the user responds.
    /// GIL is released while waiting.
    fn ask_user(&self, py: Python<'_>, payload_json: String) -> PyResult<String> {
        let payload: serde_json::Value =
            serde_json::from_str(&payload_json).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("invalid JSON: {e}"))
            })?;

        let question = payload
            .get("question")
            .and_then(|q| q.as_str())
            .unwrap_or("")
            .to_string();
        let options: Vec<String> = payload
            .get("options")
            .and_then(|o| o.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let (result_tx, result_rx) = std_mpsc::channel();

        let _ = self.response_tx.send(PythonResponse::AskUser {
            question,
            options,
            result_tx,
        });

        // Release GIL while waiting for the user's answer.
        let wrapper = SyncWrapper(result_rx);
        py.detach(move || {
            wrapper.0.recv().map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "ask_user channel closed: {e}"
                ))
            })
        })
    }
}

// ── PythonRuntime ──

pub struct PythonRuntime {
    request_tx: std_mpsc::Sender<PythonRequest>,
    response_rx: std_mpsc::Receiver<PythonResponse>,
    thread: Option<JoinHandle<()>>,
}

impl PythonRuntime {
    /// Start the embedded Python runtime on a dedicated OS thread.
    pub fn start() -> Result<Self, std::io::Error> {
        // Extract stdlib to cache before starting the thread
        let lib_dir = python_home::ensure_python_home()?;
        let home_dir = python_home::python_home(&lib_dir);

        let (request_tx, request_rx) = std_mpsc::channel::<PythonRequest>();
        let (response_tx, response_rx) = std_mpsc::channel::<PythonResponse>();

        let thread = std::thread::Builder::new()
            .name("python-runtime".into())
            .spawn(move || {
                python_thread_main(home_dir, request_rx, response_tx);
            })?;

        Ok(Self {
            request_tx,
            response_rx,
            thread: Some(thread),
        })
    }

    pub fn send(&self, request: PythonRequest) -> Result<(), std::io::Error> {
        self.request_tx.send(request).map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::BrokenPipe, "python thread gone")
        })
    }

    pub fn recv(&self) -> Result<PythonResponse, std::io::Error> {
        self.response_rx.recv().map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::BrokenPipe, "python thread gone")
        })
    }

    /// Non-blocking try_recv.
    pub fn try_recv(&self) -> Result<Option<PythonResponse>, std::io::Error> {
        match self.response_rx.try_recv() {
            Ok(msg) => Ok(Some(msg)),
            Err(std_mpsc::TryRecvError::Empty) => Ok(None),
            Err(std_mpsc::TryRecvError::Disconnected) => Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "python thread gone",
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

// ── Python thread ──

fn python_thread_main(
    home_dir: std::path::PathBuf,
    request_rx: std_mpsc::Receiver<PythonRequest>,
    response_tx: std_mpsc::Sender<PythonResponse>,
) {
    // Set PYTHONHOME before Python initializes
    unsafe {
        std::env::set_var("PYTHONHOME", &home_dir);
        // Disable user site-packages
        std::env::set_var("PYTHONNOUSERSITE", "1");
    }

    Python::initialize();

    Python::attach(|py| {
        // Load repl.py as a module — PyModule::from_code takes &CStr
        let code_cstr =
            CString::new(REPL_PY).expect("repl.py contains null byte");
        let repl = match PyModule::from_code(
            py,
            &code_cstr,
            c"repl.py",
            c"repl",
        ) {
            Ok(m) => m,
            Err(e) => {
                tracing::error!("Failed to load repl.py: {e}");
                e.print(py);
                return;
            }
        };

        // Inject the RustBridge
        let bridge = Py::new(
            py,
            RustBridge {
                response_tx: response_tx.clone(),
            },
        )
        .expect("Failed to create RustBridge");

        if let Err(e) = repl.setattr("_rust_bridge", bridge) {
            tracing::error!("Failed to inject _rust_bridge: {e}");
            return;
        }

        // Main request loop
        loop {
            let request = match request_rx.recv() {
                Ok(r) => r,
                Err(_) => break, // Channel closed
            };

            match request {
                PythonRequest::Init { tools_json } => {
                    if let Err(e) = repl.call_method1("_register_tools", (&tools_json,)) {
                        tracing::error!("_register_tools failed: {e}");
                        e.print(py);
                    }
                    let _ = response_tx.send(PythonResponse::Ready);
                }
                PythonRequest::Exec { id, code } => {
                    // Run _handle_exec via asyncio.run()
                    let asyncio = py.import("asyncio").expect("asyncio import failed");
                    let coro = match repl.call_method1("_handle_exec", (&id, &code)) {
                        Ok(c) => c,
                        Err(e) => {
                            tracing::error!("_handle_exec failed: {e}");
                            e.print(py);
                            let _ = response_tx.send(PythonResponse::ExecResult {
                                id,
                                output: String::new(),
                                error: Some(format!("Python error: {e}")),
                            });
                            continue;
                        }
                    };
                    if let Err(e) = asyncio.call_method1("run", (coro,)) {
                        tracing::error!("asyncio.run failed: {e}");
                        e.print(py);
                    }
                }
                PythonRequest::Snapshot { id } => {
                    if let Err(e) = repl.call_method1("_handle_snapshot", (&id,)) {
                        tracing::error!("_handle_snapshot failed: {e}");
                        e.print(py);
                    }
                }
                PythonRequest::Restore { id, data } => {
                    if let Err(e) = repl.call_method1("_handle_restore", (&id, &data)) {
                        tracing::error!("_handle_restore failed: {e}");
                        e.print(py);
                    }
                }
                PythonRequest::Reset { id } => {
                    if let Err(e) = repl.call_method1("_handle_reset", (&id,)) {
                        tracing::error!("_handle_reset failed: {e}");
                        e.print(py);
                    }
                }
                PythonRequest::CheckComplete { code } => {
                    let ast = py.import("ast").expect("ast import");
                    let is_complete = ast.call_method1("parse", (&code,)).is_ok();
                    let _ = response_tx.send(PythonResponse::CheckCompleteResult { is_complete });
                }
                PythonRequest::Shutdown => break,
            }
        }
    });
}
