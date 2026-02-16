import builtins
import json
import io
import re
import sys
import os
import ast
import inspect
import traceback
import types
import typing
import asyncio
import uuid
from enum import StrEnum

import dill


# ─── Turn History ───

class ToolName(StrEnum):
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    EDIT_FILE = "edit_file"
    FIND_REPLACE = "find_replace"
    DIFF_FILE = "diff_file"
    GLOB = "glob"
    GREP = "grep"
    LS = "ls"
    SHELL = "shell"
    WEB_SEARCH = "web_search"
    FETCH_URL = "fetch_url"
    DELEGATE_TASK = "delegate_task"
    DELEGATE_SEARCH = "delegate_search"
    DELEGATE_DEEP = "delegate_deep"
    TASKS = "tasks"
    SKILLS = "skills"
    HASHLINE = "hashline"
    OTHER = "other"

_TOOL_NAME_MAP = {v.value: v for v in ToolName}

# Tools whose "path" arg counts as a file read
_READ_TOOLS = {ToolName.READ_FILE, ToolName.GLOB, ToolName.GREP}
# Tools whose "path" arg counts as a file write
_WRITE_TOOLS = {ToolName.WRITE_FILE, ToolName.EDIT_FILE, ToolName.FIND_REPLACE, ToolName.DIFF_FILE}


class ToolCall:
    __slots__ = ("tool", "args", "result", "success", "duration_ms")

    def __init__(self, tool, args, result, success, duration_ms):
        self.tool = tool
        self.args = args
        self.result = result
        self.success = success
        self.duration_ms = duration_ms

    def __repr__(self):
        sym = "\u2713" if self.success else "\u2717"
        parts = [f"ToolCall({self.tool.value}"]
        path = self.args.get("path") or self.args.get("pattern")
        if path:
            parts.append(f' path="{path}"')
        parts.append(f" {sym} {self.duration_ms}ms)")
        return "".join(parts)


class Turn:
    __slots__ = ("index", "code", "output", "error", "tool_calls", "files_read", "files_written")

    def __init__(self, index, code, output, error, tool_calls, files_read, files_written):
        self.index = index
        self.code = code
        self.output = output
        self.error = error
        self.tool_calls = tool_calls
        self.files_read = files_read
        self.files_written = files_written

    def __repr__(self):
        tc = len(self.tool_calls)
        out_len = len(self.output)
        if out_len >= 1024:
            out_str = f"{out_len / 1024:.1f}k output"
        else:
            out_str = f"{out_len} output"
        sym = "\u2717" if self.error else "\u2713"
        return f"Turn(#{self.index}: {tc} tool calls, {out_str}, {sym})"


class TurnHistory:
    def __init__(self):
        self._turns = []

    def __getitem__(self, key):
        return self._turns[key]

    def __len__(self):
        return len(self._turns)

    def __iter__(self):
        return iter(self._turns)

    def __repr__(self):
        n = len(self._turns)
        files = len(self.files_modified())
        return f"TurnHistory({n} turns, {files} files touched)"

    def search(self, pattern):
        """Regex search over code+output of all turns."""
        rx = re.compile(pattern, re.IGNORECASE)
        return [t for t in self._turns if rx.search(t.code) or rx.search(t.output)]

    def tool_calls(self, tool=None):
        """All tool calls, optionally filtered by tool name."""
        out = []
        for t in self._turns:
            for tc in t.tool_calls:
                if tool is None or tc.tool.value == tool or tc.tool == tool:
                    out.append(tc)
        return out

    def files_read(self):
        """All unique files read across all turns."""
        s = set()
        for t in self._turns:
            s.update(t.files_read)
        return sorted(s)

    def files_modified(self):
        """All unique files written/edited across all turns."""
        s = set()
        for t in self._turns:
            s.update(t.files_written)
        return sorted(s)

    def errors(self):
        """Turns that had errors."""
        return [t for t in self._turns if t.error]

    def summary(self):
        """Auto-generated brief summary."""
        n = len(self._turns)
        tc = sum(len(t.tool_calls) for t in self._turns)
        errs = len(self.errors())
        fr = self.files_read()
        fm = self.files_modified()
        lines = [f"{n} turns, {tc} tool calls, {errs} errors"]
        if fr:
            lines.append(f"Files read: {', '.join(fr[:10])}" + (f" (+{len(fr)-10} more)" if len(fr) > 10 else ""))
        if fm:
            lines.append(f"Files modified: {', '.join(fm[:10])}" + (f" (+{len(fm)-10} more)" if len(fm) > 10 else ""))
        return "\n".join(lines)

    def _add_turn(self, json_str):
        """Deserialize a turn from JSON (called from Rust)."""
        data = json.loads(json_str)
        tcs = []
        files_read = []
        files_written = []
        for tc_data in data.get("tool_calls", []):
            tool_name = _TOOL_NAME_MAP.get(tc_data.get("tool", ""), ToolName.OTHER)
            args = tc_data.get("args", {})
            # Extract args dict from serde_json::Value
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            tc = ToolCall(
                tool=tool_name,
                args=args if isinstance(args, dict) else {},
                result=tc_data.get("result"),
                success=tc_data.get("success", False),
                duration_ms=tc_data.get("duration_ms", 0),
            )
            tcs.append(tc)
            path = tc.args.get("path")
            if path:
                if tool_name in _READ_TOOLS:
                    files_read.append(path)
                if tool_name in _WRITE_TOOLS:
                    files_written.append(path)
        turn = Turn(
            index=data.get("index", len(self._turns)),
            code=data.get("code", ""),
            output=data.get("output", ""),
            error=data.get("error"),
            tool_calls=tcs,
            files_read=files_read,
            files_written=files_written,
        )
        self._turns.append(turn)


# _rust_bridge is injected by the Rust runtime before any functions are called.
# It provides:
#   send_message(json_str) -> None
#   call_tool(py, call_id, name, args_json) -> str
_rust_bridge = None

# --- Persistent REPL namespace ---
_ns = {}
_tools_initialized = False

# --- Tool call resolution ---
_pending_calls = {}  # id -> asyncio.Future
_loop = None


class _Awaitable:
    """Immediately-resolved awaitable that optionally carries a return value."""
    def __init__(self, value=None):
        self._value = value

    def __await__(self):
        return self._value
        yield  # makes this a generator; unreachable but required

    def __repr__(self):
        return repr(self._value)

    def __str__(self):
        return str(self._value) if self._value is not None else ""


def _send(msg):
    """Send a message to Rust via the bridge."""
    _rust_bridge.send_message(json.dumps(msg))


_MAX_OUTPUT_LEN = 20_000
_MAX_STDOUT_LEN = 20_000

def _truncate_output(text, limit=_MAX_STDOUT_LEN):
    """Head+tail truncation for stdout captured during exec."""
    if len(text) <= limit:
        return text
    half = limit // 2
    omitted = len(text) - limit
    return f"{text[:half]}\n\n... ({omitted:,} chars omitted) ...\n\n{text[-half:]}"

def _format_value(value):
    """Format a value as the REPL would: strings as-is, others via repr()."""
    if isinstance(value, str):
        text = value.strip()
    elif value is None:
        return None  # skip, like REPL
    else:
        text = repr(value)
    if not text:
        return None
    if len(text) > _MAX_OUTPUT_LEN:
        text = text[:_MAX_OUTPUT_LEN] + f"\n... [truncated, {len(text)} chars total]"
    return text

def _respond(value):
    """Send a final response to the user. Ends the turn."""
    text = _format_value(value)
    if text is None:
        text = ""
    _send({"type": "message", "text": text, "kind": "final"})
    return _Awaitable()

def _say(value):
    """Show text to the user. Non-blocking — execution continues."""
    text = _format_value(value)
    if text is not None:
        _send({"type": "message", "text": text, "kind": "say"})
    return _Awaitable()


def _observe():
    """Stop and view output. Handled by the runtime — this is a no-op fallback."""
    return None


async def _ask(question, options=None):
    """Ask the user a question. Blocks until they respond."""
    payload = json.dumps({"question": str(question), "options": options})
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _rust_bridge.ask_user, payload)
    # Print the answer so it appears in execution output, ensuring the agent
    # continues to the next LLM iteration with the user's response in context.
    print(f"[User response: {result}]")
    return result


class BashHandle:
    """Handle to a running bash process (PTY-backed).

    Returned by bash(command=...). Provides:
      .result(timeout=None) — wait for exit, return output
      .write(text)         — send stdin input
      .output()            — read accumulated output (non-blocking)
      .kill()              — send SIGTERM
    """
    def __init__(self, id):
        self.id = id

    async def result(self, timeout=None):
        """Wait for the process to exit and return its full output."""
        return await _call("bash_result", {"id": self.id, "timeout": timeout})

    async def write(self, text):
        """Send input to the process's stdin."""
        return await _call("bash_write", {"id": self.id, "input": text})

    async def output(self):
        """Read accumulated output so far (non-blocking)."""
        return await _call("bash_output", {"id": self.id})

    async def kill(self):
        """Send SIGTERM to the process."""
        return await _call("bash_kill", {"id": self.id})

    def __repr__(self):
        return f"BashHandle(id='{self.id}')"

    def __str__(self):
        return f"BashHandle({self.id})"


class Task:
    """A task in the task management system.

    Returned by create_task(), get_task(), update_task(), and items in tasks().
    Provides convenience methods that call the underlying tools.

    repr: Task(a1b2 ~ 'Fix auth bug' high)
    str:  [~ in_progress] Fix auth bug  (a1b2, high)
          description text here
          blocked_by: c3d4, e5f6
    """

    _STATUS_SYMBOLS = {
        "pending": "\u25cb",      # ○
        "in_progress": "~",
        "completed": "\u2713",    # ✓
        "cancelled": "\u2717",    # ✗
    }

    def __init__(self, data):
        self.id = data.get("id", "")
        self.subject = data.get("subject", "")
        self.description = data.get("description", "")
        self.status = data.get("status", "pending")
        self.priority = data.get("priority", "medium")
        self.active_form = data.get("active_form", "")
        self.owner = data.get("owner", "")
        self.blocks = data.get("blocks", [])
        self.blocked_by = data.get("blocked_by", [])
        self.metadata = data.get("metadata", {})

    def _sym(self):
        return self._STATUS_SYMBOLS.get(self.status, "?")

    def __repr__(self):
        return f"Task({self.id} {self._sym()} '{self.subject}' {self.priority})"

    def __str__(self):
        lines = [f"[{self._sym()} {self.status}] {self.subject}  ({self.id}, {self.priority})"]
        if self.description:
            lines.append(f"  {self.description}")
        if self.blocked_by:
            lines.append(f"  blocked_by: {', '.join(self.blocked_by)}")
        if self.blocks:
            lines.append(f"  blocks: {', '.join(self.blocks)}")
        return "\n".join(lines)

    async def claim(self):
        """Claim this task for the current agent."""
        return await _call("claim_task", {"id": self.id, "owner": _ns.get("__agent_id__", "")})

    async def start(self):
        """Set status to in_progress."""
        return await _call("update_task", {"id": self.id, "status": "in_progress"})

    async def done(self):
        """Set status to completed."""
        return await _call("update_task", {"id": self.id, "status": "completed"})

    async def cancel(self):
        """Set status to cancelled."""
        return await _call("update_task", {"id": self.id, "status": "cancelled"})

    async def delete(self):
        """Permanently remove this task."""
        await _call("delete_task", {"id": self.id})

    async def block(self, *ids):
        """Mark task IDs that this task blocks."""
        return await _call("update_task", {"id": self.id, "add_blocks": list(ids)})

    async def wait_on(self, *ids):
        """Mark task IDs that block this task."""
        return await _call("update_task", {"id": self.id, "add_blocked_by": list(ids)})

    async def update(self, **kw):
        """General update — pass any updatable fields as keyword args."""
        kw["id"] = self.id
        return await _call("update_task", kw)


class SkillSummary:
    """Summary of a skill (name + description only).

    repr: Skill('deploy')
    str:  deploy -- Deploy to production (2 files)
    """
    def __init__(self, data):
        self.name = data.get("name", "")
        self.description = data.get("description", "")
        self.file_count = data.get("file_count", 0)

    def __repr__(self):
        return f"Skill('{self.name}')"

    def __str__(self):
        parts = [self.name]
        if self.description:
            parts.append(f"-- {self.description}")
        if self.file_count > 0:
            parts.append(f"({self.file_count} files)")
        return " ".join(parts)

    async def load(self):
        return await _call("load_skill", {"name": self.name})


class Skill:
    """A loaded skill with full instructions.

    repr: Skill('deploy', 2 files)
    str:  deploy -- Deploy to production (2 files)
    Access .instructions for the full markdown body.
    """
    def __init__(self, data):
        self.name = data.get("name", "")
        self.description = data.get("description", "")
        self.instructions = data.get("instructions", "")
        self.files = data.get("files", [])

    def __repr__(self):
        if self.files:
            return f"Skill('{self.name}', {len(self.files)} files)"
        return f"Skill('{self.name}')"

    def __str__(self):
        parts = [self.name]
        if self.description:
            parts.append(f"-- {self.description}")
        if self.files:
            parts.append(f"({len(self.files)} files)")
        return " ".join(parts)

    async def read_file(self, path):
        return await _call("read_skill_file", {"skill_name": self.name, "path": path})


class ToolError(Exception):
    """Raised when a tool call fails.

    Extends Exception so it stops execution at the failing statement.
    Use asyncio.gather(..., return_exceptions=True) to collect failures
    without stopping.
    """
    def __init__(self, name, error):
        self.name = name
        self.error = error
        super().__init__(f"{name}: {error}")

    def __repr__(self):
        return f"ToolError({self.name!r}, {self.error!r})"

    def __bool__(self):
        return False


async def _call(name, params):
    """Call a tool by name. Returns parsed JSON result or ToolError on failure."""
    call_id = str(uuid.uuid4())
    args_json = json.dumps(params)

    # Use run_in_executor to offload the blocking Rust call to a thread pool.
    # This keeps the asyncio loop responsive for concurrent tool calls.
    loop = asyncio.get_event_loop()
    result_json = await loop.run_in_executor(
        None, _rust_bridge.call_tool, call_id, name, args_json
    )

    result = json.loads(result_json)
    if result["success"]:
        value = json.loads(result["result"]) if result["result"] else None
        # Wrap bash handles automatically
        if isinstance(value, dict) and value.get("__handle__") == "bash":
            return BashHandle(value["id"])
        # Wrap typed objects
        if isinstance(value, dict) and "__type__" in value:
            t = value["__type__"]
            if t == "task":
                return Task(value)
            if t == "task_list":
                return [Task(item) for item in value.get("items", [])]
            if t == "skill":
                return Skill(value)
            if t == "skill_summary":
                return SkillSummary(value)
            if t == "skill_list":
                return [SkillSummary(item) for item in value.get("items", [])]
        return value
    else:
        error = json.loads(result["result"]) if result["result"] else "Tool call failed"
        raise ToolError(name, error)


_TYPE_MAP = {
    "str": str, "string": str,
    "int": int, "integer": int,
    "float": float, "number": float,
    "bool": bool, "boolean": bool,
    "list": list, "dict": dict,
    "any": typing.Any, "None": type(None),
}

_tool_defs = []


def _list_tools():
    """List all available tools with their signatures."""
    lines = []
    for t in (t for t in _tool_defs if not t.get("hidden", False)):
        params = t.get("params", [])
        if params:
            parts = []
            for p in params:
                if isinstance(p, dict):
                    ty = p.get("type", "any")
                    name = p["name"]
                    part = f"{name}: {ty}"
                    if not p.get("required", True):
                        part += " = None"
                    parts.append(part)
                else:
                    parts.append(str(p))
            sig = ", ".join(parts)
        else:
            sig = ""
        ret = t.get("returns", "any")
        lines.append(f"  {t['name']}({sig}) -> {ret}")
        desc = t.get("description", "")
        if desc:
            lines.append(f"      {desc}")
    result = "Available tools:\n" + "\n".join(lines)
    print(result)
    return _Awaitable(result)


def _reset_repl():
    """Reset the REPL namespace and re-register tools."""
    global _tools_initialized
    # Preserve the stored tool definitions
    saved_defs = json.dumps(_tool_defs)
    _ns.clear()
    _tools_initialized = False
    _register_tools(saved_defs)
    print("REPL reset: namespace cleared, tools re-registered.")
    return _Awaitable("REPL reset complete")


def _register_tools(tools_json, agent_id=""):
    """Register tool wrappers from JSON tool definitions."""
    global _tools_initialized, _tool_defs
    if _tools_initialized:
        return
    _tools_initialized = True
    _ns["__agent_id__"] = agent_id
    _tool_defs = json.loads(tools_json)
    for tool in _tool_defs:
        name = tool["name"]
        desc = tool.get("description", "")
        raw_params = tool.get("params", [])
        returns = tool.get("returns", "any")

        # Normalize params
        param_info = []
        for p in raw_params:
            if isinstance(p, str):
                param_info.append({"name": p, "type": "any", "required": True, "description": ""})
            else:
                param_info.append(p)

        def make_fn(n, d, pinfo, ret):
            pnames = [p["name"] for p in pinfo]

            async def fn(*args, **kw):
                params = {}
                for i, arg in enumerate(args):
                    if isinstance(arg, dict):
                        params.update(arg)
                    elif i < len(pnames):
                        params[pnames[i]] = arg
                params.update(kw)
                return await _call(n, params)

            fn.__name__ = n
            fn.__qualname__ = n

            # Build docstring with param types
            sig_parts = []
            doc_parts = [d, ""] if d else []
            for p in pinfo:
                ty = p.get("type", "any")
                req = p.get("required", True)
                part = f"{p['name']}: {ty}"
                if not req:
                    part += " = None"
                sig_parts.append(part)
                pdesc = p.get("description", "")
                if pdesc:
                    doc_parts.append(f"  {p['name']}: {pdesc}")
            if any(p.get("description") for p in pinfo):
                doc_parts.insert(1, "Args:")
            if ret and ret != "any":
                doc_parts.append(f"Returns: {ret}")
            fn.__doc__ = "\n".join(doc_parts) if doc_parts else d

            # Set inspect.Signature for help()
            params_list = []
            for p in pinfo:
                annotation = _TYPE_MAP.get(p.get("type", "any"), typing.Any)
                default = inspect.Parameter.empty if p.get("required", True) else None
                params_list.append(inspect.Parameter(
                    p["name"],
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=annotation,
                    default=default,
                ))
            ret_annotation = _TYPE_MAP.get(ret, typing.Any)
            fn.__signature__ = inspect.Signature(params_list, return_annotation=ret_annotation)
            return fn

        # Hidden tools are callable via _call() but not exposed in the REPL namespace
        if not tool.get("hidden", False):
            _ns[name] = make_fn(name, desc, param_info, returns)
    _async_tool_names.update(name for name in (t["name"] for t in _tool_defs))
    _async_tool_names.add("ask")

    # Override claim_task: auto-fill owner from __agent_id__, id is optional
    async def _claim_task(id=None):
        """Claim a task. If id is omitted, claims the next available task.
        Owner is automatically set to this agent's identity."""
        params = {"owner": _ns.get("__agent_id__", "")}
        if id is not None:
            params["id"] = id
        return await _call("claim_task", params)
    _claim_task.__name__ = "claim_task"
    _claim_task.__qualname__ = "claim_task"
    _ns["claim_task"] = _claim_task

    _ns["_history"] = TurnHistory()
    _ns.update({
        "json": json, "print": print, "respond": _respond, "say": _say, "observe": _observe,
        "asyncio": asyncio, "list_tools": _list_tools, "reset_repl": _reset_repl, "ask": _ask,
        "Task": Task, "Skill": Skill, "SkillSummary": SkillSummary, "ToolError": ToolError,
        "TurnHistory": TurnHistory, "Turn": Turn, "ToolCall": ToolCall, "ToolName": ToolName,
    })


# Flag that lets exec/eval accept top-level `await` (CPython 3.10+).
_ASYNC_FLAG = ast.PyCF_ALLOW_TOP_LEVEL_AWAIT

# Names of async tool functions — auto-awaited if the LLM forgets `await`.
_async_tool_names = set()


# Async method names on wrapper objects (Task, BashHandle, Skill, etc.)
_async_method_names = {
    # Task
    "claim", "start", "done", "cancel", "delete", "block", "wait_on", "update",
    # BashHandle
    "result", "write", "output", "kill",
    # SkillSummary
    "load",
    # Skill
    "read_file",
}


class _AutoAwait(ast.NodeTransformer):
    """Inject `await` around calls to known async tool functions and methods.

    Skips calls already inside an `await` or passed directly to
    asyncio.gather / create_task / ensure_future / wait.
    """

    def __init__(self, names):
        self._names = names
        self._inside_await = False
        self._suppress = False

    def visit_Await(self, node):
        old = self._inside_await
        self._inside_await = True
        node = self.generic_visit(node)
        self._inside_await = old
        return node

    @staticmethod
    def _is_async_passthrough(node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "asyncio":
                return node.func.attr in ("gather", "create_task", "ensure_future", "wait")
        return False

    def _should_auto_await(self, node):
        """Check if a call node should be auto-awaited."""
        if isinstance(node.func, ast.Name):
            return node.func.id in self._names
        if isinstance(node.func, ast.Attribute):
            return node.func.attr in _async_method_names
        return False

    def visit_Call(self, node):
        if self._is_async_passthrough(node):
            old = self._suppress
            self._suppress = True
            node = self.generic_visit(node)
            self._suppress = old
            return node
        node = self.generic_visit(node)
        if (not self._inside_await
                and not self._suppress
                and self._should_auto_await(node)):
            return ast.Await(value=node)
        return node


def _displayhook(value):
    """Custom displayhook matching interactive Python behavior.

    Skips None and _Awaitable sentinels, sets builtins._, prints repr.
    """
    if value is None or isinstance(value, _Awaitable):
        return
    builtins._ = value
    print(repr(value))


async def _handle_exec(exec_id, code):
    """Execute code using real REPL semantics (ast.Interactive + "single" mode)."""
    stdout_buf = io.StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    old_displayhook = sys.displayhook
    sys.stdout = sys.stderr = stdout_buf
    sys.displayhook = _displayhook
    error = None

    try:
        tree = ast.parse(code)
    except SyntaxError:
        error = traceback.format_exc()
        tree = None

    if tree and _async_tool_names:
        tree = _AutoAwait(_async_tool_names).visit(tree)
        ast.fix_missing_locations(tree)

    if tree:
        for node in tree.body:
            try:
                mod = ast.Interactive(body=[node])
                ast.fix_missing_locations(mod)
                co = compile(mod, "<repl>", "single", flags=_ASYNC_FLAG)
                if co.co_flags & inspect.CO_COROUTINE:
                    await types.FunctionType(co, _ns)()
                else:
                    exec(co, _ns)
            except Exception:
                error = traceback.format_exc()
                break

    sys.stdout, sys.stderr = old_stdout, old_stderr
    sys.displayhook = old_displayhook

    output = _truncate_output(stdout_buf.getvalue())
    _send({
        "type": "exec_result",
        "id": exec_id,
        "output": output,
        "response": "",
        "error": error,
    })


def _handle_snapshot(snap_id):
    """Serialize the REPL namespace using dill."""
    skip = {"json", "asyncio", "dill", "print", "respond", "say", "observe", "list_tools", "reset_repl", "ask"}
    skip.update(t["name"] for t in _tool_defs)

    data = {}
    for k, v in _ns.items():
        if k.startswith("_") or k in skip:
            continue
        try:
            dill.dumps(v)
            data[k] = v
        except Exception:
            continue
    _send({"type": "snapshot_result", "id": snap_id, "data": dill.dumps(data).hex()})


def _handle_restore(restore_id, data_str):
    """Restore namespace from a dill snapshot."""
    saved = dill.loads(bytes.fromhex(data_str))
    _ns.update(saved)
    _send({"type": "exec_result", "id": restore_id, "output": "", "response": "", "error": None})


def _handle_reset(reset_id):
    """Reset the REPL namespace via the protocol."""
    _reset_repl()
    _send({"type": "reset_result", "id": reset_id})
