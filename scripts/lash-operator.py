#!/usr/bin/env python3
"""Drive the Lash interactive CLI through a child PTY.

This is meant for agent-operated smoke testing: it launches the real `lash`
binary in an isolated pseudo-terminal, then accepts simple commands on stdin
for typing, key presses, waits, and output assertions.
"""

from __future__ import annotations

import argparse
import codecs
import os
import pty
import re
import select
import shlex
import signal
import struct
import subprocess
import sys
import tempfile
import termios
import threading
import time
from pathlib import Path


ANSI_RE = re.compile(
    rb"\x1b\[[0-?]*[ -/]*[@-~]"
    rb"|\x1b\][^\x07]*(?:\x07|\x1b\\)"
    rb"|\x1b[@-_]"
)

KEYS = {
    "enter": b"\r",
    "return": b"\r",
    "tab": b"\t",
    "backtab": b"\x1b[Z",
    "esc": b"\x1b",
    "escape": b"\x1b",
    "ctrl-c": b"\x03",
    "ctrl-d": b"\x04",
    "backspace": b"\x7f",
    "delete": b"\x1b[3~",
    "up": b"\x1b[A",
    "down": b"\x1b[B",
    "right": b"\x1b[C",
    "left": b"\x1b[D",
    "alt-up": b"\x1b\x1b[A",
    "alt-down": b"\x1b\x1b[B",
    "pageup": b"\x1b[5~",
    "pagedown": b"\x1b[6~",
    "home": b"\x1b[H",
    "end": b"\x1b[F",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch Lash in a child PTY and drive it with text commands."
    )
    parser.add_argument(
        "--provider",
        choices=["test", "real"],
        default="test",
        help="test writes an isolated deterministic config; real uses existing config/env.",
    )
    parser.add_argument(
        "--scenario",
        default="standard-echo",
        help="test-provider scenario to use when --provider test.",
    )
    parser.add_argument(
        "--lash-home",
        type=Path,
        help="LASH_HOME to use. Defaults to a temp dir for --provider test.",
    )
    parser.add_argument(
        "--build",
        dest="build",
        action="store_true",
        default=True,
        help="build the lash binary before launching (default).",
    )
    parser.add_argument(
        "--no-build",
        dest="build",
        action="store_false",
        help="launch the existing target/debug/lash binary.",
    )
    parser.add_argument(
        "--trace",
        type=Path,
        help="pass --debug-ui-trace PATH to the child lash process.",
    )
    parser.add_argument("--cols", type=int, default=100)
    parser.add_argument("--rows", type=int, default=28)
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="mirror raw child PTY bytes to this process's stdout.",
    )
    parser.add_argument(
        "--script",
        type=Path,
        help="read operator commands from a file instead of stdin.",
    )
    parser.add_argument(
        "lash_args",
        nargs=argparse.REMAINDER,
        help="arguments passed to lash after an optional -- separator.",
    )
    args = parser.parse_args()
    if args.lash_args and args.lash_args[0] == "--":
        args.lash_args = args.lash_args[1:]
    return args


def build_binary(use_test_provider: bool) -> None:
    cmd = ["cargo", "build", "-p", "lash-cli"]
    if use_test_provider:
        cmd.extend(["--features", "test-provider"])
    subprocess.run(cmd, cwd=repo_root(), check=True)


def lash_bin() -> Path:
    return repo_root() / "target" / "debug" / "lash"


def write_test_provider_config(lash_home: Path, scenario: str) -> None:
    import json

    lash_home.mkdir(parents=True, exist_ok=True)
    config = {
        "active_provider": "test",
        "providers": {"test": {"type": "test", "scenario": scenario}},
    }
    (lash_home / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    cache = lash_home / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    catalog = {
        "test": {
            "models": {
                "cli-e2e-model": {"limit": {"context": 64000, "output": 4096}}
            }
        }
    }
    (cache / "models.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")


class LashOperator:
    def __init__(self, args: argparse.Namespace, lash_home: Path | None) -> None:
        self.args = args
        self.lash_home = lash_home
        self.master_fd: int | None = None
        self.proc: subprocess.Popen[bytes] | None = None
        self.output = bytearray()
        self.output_lock = threading.Lock()
        self.reader_thread: threading.Thread | None = None
        self.stop_reader = threading.Event()

    def start(self) -> None:
        master_fd, slave_fd = pty.openpty()
        self.master_fd = master_fd
        winsize = struct.pack("HHHH", self.args.rows, self.args.cols, 0, 0)
        try:
            termios.tcsetwinsize(slave_fd, (self.args.rows, self.args.cols))
        except AttributeError:
            import fcntl

            fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)

        env = os.environ.copy()
        env.setdefault("LASH_LOG", "warn")
        env.setdefault("TERM", "xterm-256color")
        env.setdefault("NO_COLOR", "1")
        if self.lash_home is not None:
            env["LASH_HOME"] = str(self.lash_home)

        child_args = list(self.args.lash_args)
        if self.args.provider == "test" and not has_model_arg(child_args):
            child_args = ["--model", "test/cli-e2e-model", *child_args]
        if self.args.trace is not None:
            child_args.extend(["--debug-ui-trace", str(self.args.trace)])

        argv = [str(lash_bin()), *child_args]
        self.proc = subprocess.Popen(
            argv,
            cwd=repo_root(),
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            env=env,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            close_fds=True,
        )
        os.close(slave_fd)
        self.reader_thread = threading.Thread(target=self.read_loop, daemon=True)
        self.reader_thread.start()
        print(
            f"LASH_OPERATOR_READY pid={self.proc.pid} provider={self.args.provider} "
            f"home={self.lash_home or '<real>'} argv={shlex.join(argv)}",
            flush=True,
        )

    def read_loop(self) -> None:
        assert self.master_fd is not None
        while not self.stop_reader.is_set():
            ready, _, _ = select.select([self.master_fd], [], [], 0.1)
            if not ready:
                continue
            try:
                data = os.read(self.master_fd, 8192)
            except OSError:
                break
            if not data:
                break
            with self.output_lock:
                self.output.extend(data)
                if len(self.output) > 4_000_000:
                    del self.output[:1_000_000]
            if self.args.mirror:
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()

    def write(self, data: bytes) -> None:
        assert self.master_fd is not None
        os.write(self.master_fd, data)

    def current_output(self, clean: bool = False) -> str:
        with self.output_lock:
            data = bytes(self.output)
        if clean:
            data = ANSI_RE.sub(b"", data)
            data = data.replace(b"\r", b"\n")
        return data.decode("utf-8", errors="replace")

    def wait_for(self, needle: str, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while True:
            raw = self.current_output(clean=False)
            clean = self.current_output(clean=True)
            if needle in raw or needle in clean:
                print(f"OK expect {needle!r}", flush=True)
                return
            status = self.poll_status()
            if status is not None:
                raise RuntimeError(
                    f"lash exited with {status} before {needle!r} appeared\n"
                    f"{tail_text(clean, 80)}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for {needle!r}\n{tail_text(clean, 80)}")
            time.sleep(0.05)

    def poll_status(self) -> int | None:
        assert self.proc is not None
        return self.proc.poll()

    def stop(self, timeout: float = 5.0) -> int | None:
        status = self.poll_status()
        if status is None and self.proc is not None:
            try:
                if hasattr(os, "killpg"):
                    os.killpg(self.proc.pid, signal.SIGTERM)
                else:
                    self.proc.terminate()
                status = self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                if hasattr(os, "killpg"):
                    os.killpg(self.proc.pid, signal.SIGKILL)
                else:
                    self.proc.kill()
                status = self.proc.wait(timeout=timeout)
        self.stop_reader.set()
        if self.master_fd is not None:
            try:
                os.close(self.master_fd)
            except OSError:
                pass
            self.master_fd = None
        if self.reader_thread is not None:
            self.reader_thread.join(timeout=1)
        return status


def has_model_arg(args: list[str]) -> bool:
    return any(arg == "--model" or arg.startswith("--model=") for arg in args)


def tail_text(text: str, lines: int) -> str:
    return "\n".join(text.splitlines()[-lines:])


def decode_escapes(text: str) -> bytes:
    decoded = codecs.decode(text, "unicode_escape")
    return decoded.encode("utf-8")


def parse_timeout_prefixed(rest: str, default: float) -> tuple[float, str]:
    rest = rest.strip()
    if not rest:
        return default, ""
    first, _, tail = rest.partition(" ")
    try:
        return float(first), tail
    except ValueError:
        return default, rest


def run_command(op: LashOperator, line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return True
    name, _, rest = stripped.partition(" ")
    name = name.lower()

    if name in {"help", "?"}:
        print(
            "commands: type TEXT | send ESCAPED | key NAME [COUNT] | expect [SECS] TEXT | "
            "expect-re [SECS] REGEX | wait SECS | screen [LINES] | raw [LINES] | "
            "clear | status | lash-exit [SECS] | kill | quit",
            flush=True,
        )
    elif name in {"type", "paste"}:
        op.write(rest.encode("utf-8"))
        print(f"OK typed {len(rest)} chars", flush=True)
    elif name == "send":
        data = decode_escapes(rest)
        op.write(data)
        print(f"OK sent {len(data)} bytes", flush=True)
    elif name == "key":
        parts = rest.split()
        if not parts:
            raise ValueError("key requires a key name")
        key = parts[0].lower()
        count = int(parts[1]) if len(parts) > 1 else 1
        if key not in KEYS:
            raise ValueError(f"unknown key {key!r}; known: {', '.join(sorted(KEYS))}")
        op.write(KEYS[key] * count)
        print(f"OK key {key} x{count}", flush=True)
    elif name == "expect":
        timeout, needle = parse_timeout_prefixed(rest, 10.0)
        if not needle:
            raise ValueError("expect requires text")
        op.wait_for(needle, timeout)
    elif name == "expect-re":
        timeout, pattern = parse_timeout_prefixed(rest, 10.0)
        if not pattern:
            raise ValueError("expect-re requires a regex")
        deadline = time.monotonic() + timeout
        compiled = re.compile(pattern, re.MULTILINE)
        while True:
            clean = op.current_output(clean=True)
            if compiled.search(clean):
                print(f"OK expect-re {pattern!r}", flush=True)
                break
            status = op.poll_status()
            if status is not None:
                raise RuntimeError(
                    f"lash exited with {status} before regex {pattern!r} matched\n"
                    f"{tail_text(clean, 80)}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for regex {pattern!r}\n{tail_text(clean, 80)}"
                )
            time.sleep(0.05)
    elif name == "wait":
        seconds = float(rest.strip() or "1")
        time.sleep(seconds)
        print(f"OK waited {seconds:g}s", flush=True)
    elif name == "screen":
        lines = int(rest.strip() or "80")
        print(tail_text(op.current_output(clean=True), lines), flush=True)
    elif name == "raw":
        lines = int(rest.strip() or "80")
        print(tail_text(op.current_output(clean=False), lines), flush=True)
    elif name == "clear":
        with op.output_lock:
            op.output.clear()
        print("OK cleared output buffer", flush=True)
    elif name == "status":
        print(f"STATUS {op.poll_status()}", flush=True)
    elif name == "lash-exit":
        timeout = float(rest.strip() or "10")
        op.write(b"/exit\r")
        deadline = time.monotonic() + timeout
        while True:
            status = op.poll_status()
            if status is not None:
                print(f"OK lash exited {status}", flush=True)
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for lash to exit after /exit")
            time.sleep(0.05)
    elif name == "kill":
        status = op.stop()
        print(f"OK killed status={status}", flush=True)
        return False
    elif name in {"quit", "driver-exit"}:
        return False
    else:
        raise ValueError(f"unknown command {name!r}; run `help`")
    return True


def main() -> int:
    args = parse_args()
    tmp_home: tempfile.TemporaryDirectory[str] | None = None
    lash_home = args.lash_home
    if args.provider == "test" and lash_home is None:
        tmp_home = tempfile.TemporaryDirectory(prefix="lash-operator-")
        lash_home = Path(tmp_home.name)
    if args.provider == "test" and lash_home is not None:
        write_test_provider_config(lash_home, args.scenario)

    if args.build:
        build_binary(use_test_provider=args.provider == "test")

    op = LashOperator(args, lash_home)
    op.start()
    source = args.script.open(encoding="utf-8") if args.script else sys.stdin
    exit_code = 0
    try:
        for line in source:
            try:
                keep_going = run_command(op, line)
            except Exception as exc:
                exit_code = 1
                print(f"ERROR {exc}", file=sys.stderr, flush=True)
                print(tail_text(op.current_output(clean=True), 100), file=sys.stderr, flush=True)
                break
            if not keep_going:
                break
    finally:
        if source is not sys.stdin:
            source.close()
        if op.poll_status() is None:
            op.stop()
        if tmp_home is not None:
            tmp_home.cleanup()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
