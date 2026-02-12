# kaml — Architecture Spec

## Overview

kaml is a Rust library providing a complete agent kernel: a native Python REPL subprocess with pluggable tools, BAML-powered LLM integration, and optional syd syscall sandboxing.

**Architecture**: Native CPython subprocess communicating over JSONL (stdin/stdout). No Wasm, no componentize-py, no wasmtime.

**Why native Python over Wasm**: A coding agent needs filesystem access, bash, and broad host interaction. Wasm becomes an expensive enforcement layer over a mostly-open policy. Native Python + syd gives the same isolation with better performance, full stdlib, and simpler architecture.

**Why BAML**: All LLM calls go through BAML — typed prompts, multi-provider abstraction, streaming, runtime client configuration.

## Components

```
┌──────────────────────────────────────────────────┐
│  Agent (CodeAct loop)                            │
│  - BAML streaming LLM calls                      │
│  - Extract ```python blocks from LLM response    │
│  - Feed execution results back as context         │
│  - Emit AgentEvents (text, tool calls, code, etc) │
├──────────────────────────────────────────────────┤
│  Session (Python REPL subprocess)                 │
│  - Spawn Python with JSONL I/O                    │
│  - Execute code blocks                            │
│  - Bridge tool calls (Python → host → Python)     │
│  - Snapshot/restore state                         │
├──────────────────────────────────────────────────┤
│  SessionManager (pooling)                         │
│  - take/put/destroy by UUID                       │
│  - Idle timeout auto-cleanup                      │
└──────────────────────────────────────────────────┘
```

## JSONL Protocol

All messages are single-line JSON objects with a `"type"` discriminator field.

### Host → Python (stdin)

| Type | Fields | Description |
|------|--------|-------------|
| `init` | `tools: string` | Tool definitions JSON. Sent once after spawn. |
| `exec` | `id: string`, `code: string` | Execute a code block. |
| `tool_result` | `id: string`, `success: bool`, `result: string` | Response to a `tool_call`. |
| `snapshot` | `id: string` | Request serialized state. |
| `restore` | `id: string`, `data: string` | Restore serialized state. |
| `shutdown` | *(none)* | Clean exit. |

### Python → Host (stdout)

| Type | Fields | Description |
|------|--------|-------------|
| `ready` | *(none)* | Init complete, tools registered. |
| `tool_call` | `id: string`, `name: string`, `args: string` | Request tool execution from host. |
| `message` | `text: string`, `kind: string` | Progress/final message for the user. |
| `exec_result` | `id: string`, `output: string`, `response: string`, `error: string?` | Code execution complete. |
| `snapshot_result` | `id: string`, `data: string` | Serialized state (JSON). |

### Flow: Code Execution

```
Host                         Python
  │                            │
  ├── exec {id, code} ───────►│
  │                            ├── (executes code)
  │                            │
  │◄── tool_call {id, ...} ───┤  (tool needed)
  ├── tool_result {id, ...} ──►│  (result from host)
  │                            │
  │◄── tool_call {id, ...} ───┤  (another tool, concurrent ok)
  ├── tool_result {id, ...} ──►│
  │                            │
  │◄── message {text, kind} ──┤  (optional progress/final)
  │                            │
  │◄── exec_result {id, ...} ─┤  (done)
```

### Flow: Concurrent Tool Calls (asyncio.gather)

When Python code uses `asyncio.gather(tool1(), tool2())`, multiple `tool_call` messages arrive before any `tool_result` is sent back. The host spawns concurrent tokio tasks and sends results as they complete. Python resolves futures by matching `id`.

## Session Lifecycle

1. **Spawn**: Host writes `repl.py` to a temp file, spawns `python3 <path>` (or `uv run --python 3.13 <path>`, or `syd -c <config> -- python3 <path>`)
2. **Init**: Host sends `init` with tool definitions JSON. Python registers tool wrappers, sends `ready`.
3. **Execute**: Host sends `exec` messages. Python executes, bridges tool calls, sends `exec_result`.
4. **Snapshot**: Host sends `snapshot`. Python serializes namespace + scratch files, sends `snapshot_result`.
5. **Restore**: Host sends `restore` with snapshot data. Python deserializes.
6. **Shutdown**: Host sends `shutdown`. Python exits cleanly. (Also: Drop kills child process.)

## Tool System

### ToolProvider Trait

```rust
#[async_trait]
pub trait ToolProvider: Send + Sync + 'static {
    fn definitions(&self) -> Vec<ToolDefinition>;
    async fn execute(&self, name: &str, args: &serde_json::Value) -> ToolResult;
}
```

Consumers implement this trait to provide domain-specific tools. The session bridges calls between Python and the host via JSONL.

### Tool Wrappers in Python

On `init`, the REPL dynamically generates async Python functions for each tool:

```python
async def read(path, start=None, end=None):
    return await _call("read", {"path": path, "start": start, "end": end})
```

Functions support positional args, keyword args, type annotations, docstrings, and `inspect.Signature` for `help()`.

## Agent Loop (CodeAct)

The agent implements the CodeAct pattern: LLM writes prose + Python code blocks, code executes in the REPL, output feeds back to the LLM.

1. Build BAML `ClientRegistry` with model/api_key/base_url
2. Loop (up to `max_iterations`):
   a. Stream LLM response via BAML
   b. Extract ```python code blocks
   c. If no code: done (prose-only response)
   d. Execute via `session.run_code(code)`
   e. If `message(kind="final")` called: done
   f. Feed output/errors back into messages for next iteration
3. Emit `Done` event

## Sandbox Model

Optional syd syscall sandboxing. If `SessionConfig::syd_config` is set, the Python process is spawned under syd:

```
syd -c <config> -- python3 repl.py
```

This restricts syscalls (no network, limited filesystem, etc.) while keeping full Python stdlib available.

## Python Management

Session startup tries Python interpreters in order:
1. `uv run --python 3.13 <path>` — preferred, exact version control
2. `python3 <path>` — fallback
3. `SessionConfig::python` — explicit override
