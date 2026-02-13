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
import threading
import uuid
import queue

import dill

# --- Save real stdio before we redirect anything ---
_real_stdout = sys.stdout
_real_stdin = sys.stdin

# --- Persistent REPL namespace ---
_ns = {}
_tools_initialized = False

# --- Tool call resolution ---
_pending_calls = {}  # id -> asyncio.Future
_loop = None
# Queue for incoming messages read by the background reader thread.
# During exec, tool_result messages are pulled from here.
_inbox = queue.Queue()
# Event signaling that exec is active and reader should feed _inbox
_exec_active = threading.Event()


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
    """Send a JSONL message to the host via real stdout."""
    _real_stdout.write(json.dumps(msg) + "\n")
    _real_stdout.flush()


def _recv_raw():
    """Read a JSONL message from real stdin (blocking)."""
    line = _real_stdin.readline()
    if not line:
        sys.exit(0)
    return json.loads(line.strip())


def _message(text, *, kind):
    """Send a message to the user.

    kind="progress" — streams immediately, execution continues.
    kind="final" — streams to user, stops the turn.
    """
    _send({"type": "message", "text": str(text), "kind": kind})
    return _Awaitable()


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
    _send({"type": "tool_call", "id": call_id, "name": name, "args": json.dumps(params)})

    # Create a future that will be resolved when we get the tool_result back
    future = _loop.create_future()
    _pending_calls[call_id] = future
    result = await future

    if result["success"]:
        return json.loads(result["result"]) if result["result"] else None
    else:
        error = json.loads(result["result"]) if result["result"] else "Tool call failed"
        return ToolError(name, error)


def _resolve_tool_result(msg):
    """Resolve a pending tool call future with the result."""
    call_id = msg["id"]
    if call_id in _pending_calls:
        future = _pending_calls.pop(call_id)
        if not future.done():
            future.set_result(msg)


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
    for t in _tool_defs:
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

        _ns[name] = make_fn(name, desc, param_info, returns)
    _async_tool_names.update(name for name in (t["name"] for t in _tool_defs))
    _ns.update({
        "json": json, "print": print, "message": _message, "asyncio": asyncio,
        "list_tools": _list_tools, "reset_repl": _reset_repl,
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


async def _drain_inbox():
    """Async task: drains _inbox and resolves tool call futures."""
    while True:
        try:
            msg = _inbox.get_nowait()
            if msg.get("type") == "tool_result":
                _resolve_tool_result(msg)
            _inbox.task_done()
        except queue.Empty:
            await asyncio.sleep(0.005)


async def _handle_exec(exec_id, code):
    """Execute code using real REPL semantics (ast.Interactive + "single" mode)."""
    global _loop
    _loop = asyncio.get_event_loop()

    # Start inbox drainer that resolves tool_result futures
    drainer = asyncio.create_task(_drain_inbox())

    # Signal the reader thread to feed messages into _inbox
    _exec_active.set()

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

    # Stop the inbox drainer and reader feeding
    _exec_active.clear()
    drainer.cancel()
    try:
        await drainer
    except asyncio.CancelledError:
        pass

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
    skip = {"json", "asyncio", "dill", "print", "message", "list_tools", "reset_repl"}
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


def _reader_thread():
    """Background thread: reads all stdin messages.

    During exec (_exec_active is set), tool_result messages go to _inbox.
    Other messages are queued for the main loop.
    """
    while True:
        try:
            line = _real_stdin.readline()
            if not line:
                break
            msg = json.loads(line.strip())
            if _exec_active.is_set() and msg.get("type") == "tool_result":
                _inbox.put(msg)
            else:
                # Put non-tool-result messages into _inbox for main loop to pick up
                _inbox.put(msg)
        except Exception:
            break


def main():
    """Main REPL loop: reads JSONL commands from stdin, dispatches."""
    scratch = os.environ.get("SCRATCH_DIR", "/tmp/scratch")
    os.makedirs(scratch, exist_ok=True)

    # Wait for init (read directly, before starting reader thread)
    msg = _recv_raw()
    assert msg["type"] == "init", f"Expected init, got {msg['type']}"
    _register_tools(msg["tools"])
    _send({"type": "ready"})

    # Start persistent reader thread
    reader = threading.Thread(target=_reader_thread, daemon=True)
    reader.start()

    # Create an event loop for async execution
    loop = asyncio.new_event_loop()

    while True:
        # Read next command from inbox (reader thread fills it)
        msg = _inbox.get()
        msg_type = msg["type"]

        if msg_type == "exec":
            loop.run_until_complete(_handle_exec(msg["id"], msg["code"]))
        elif msg_type == "snapshot":
            _handle_snapshot(msg["id"])
        elif msg_type == "restore":
            _handle_restore(msg["id"], msg["data"])
        elif msg_type == "reset":
            _handle_reset(msg["id"])
        elif msg_type == "shutdown":
            break


if __name__ == "__main__":
    main()
