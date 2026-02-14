import builtins
import json
import io
import sys
import os
import ast
import inspect
import traceback
import types
import typing
import asyncio
import uuid

import dill

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


def _respond(text):
    """Send a final response to the user. Ends the turn."""
    _send({"type": "message", "text": str(text), "kind": "final"})
    return _Awaitable()

def _status(text):
    """Show a status update to the user. Non-blocking — execution continues."""
    _send({"type": "message", "text": str(text), "kind": "progress"})
    return _Awaitable()


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


class ToolError:
    """Returned instead of raising when a tool call fails.

    This allows asyncio.gather() to complete all calls even if some fail.
    """
    def __init__(self, name, error):
        self.name = name
        self.error = error

    def __repr__(self):
        return f"ToolError({self.name!r}, {self.error!r})"

    def __str__(self):
        return f"Error calling {self.name}: {self.error}"

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
        # Wrap task types
        if isinstance(value, dict) and "__type__" in value:
            t = value["__type__"]
            if t == "task":
                return Task(value)
            if t == "task_list":
                return [Task(item) for item in value.get("items", [])]
        return value
    else:
        error = json.loads(result["result"]) if result["result"] else "Tool call failed"
        return ToolError(name, error)


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


def _register_tools(tools_json):
    """Register tool wrappers from JSON tool definitions."""
    global _tools_initialized, _tool_defs
    if _tools_initialized:
        return
    _tools_initialized = True
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
    _ns.update({
        "json": json, "print": print, "respond": _respond, "status": _status, "asyncio": asyncio,
        "list_tools": _list_tools, "reset_repl": _reset_repl, "ask": _ask,
        "Task": Task,
    })


# Flag that lets exec/eval accept top-level `await` (CPython 3.10+).
_ASYNC_FLAG = ast.PyCF_ALLOW_TOP_LEVEL_AWAIT

# Names of async tool functions — auto-awaited if the LLM forgets `await`.
_async_tool_names = set()


class _AutoAwait(ast.NodeTransformer):
    """Inject `await` around calls to known async tool functions.

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
                and isinstance(node.func, ast.Name)
                and node.func.id in self._names):
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
        traceback.print_exc()
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
                traceback.print_exc()

    sys.stdout, sys.stderr = old_stdout, old_stderr
    sys.displayhook = old_displayhook

    output = stdout_buf.getvalue()
    _send({
        "type": "exec_result",
        "id": exec_id,
        "output": output,
        "response": "",
        "error": None,
    })


def _handle_snapshot(snap_id):
    """Serialize the REPL namespace using dill."""
    skip = {"json", "asyncio", "dill", "print", "respond", "status", "list_tools", "reset_repl", "ask"}
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
