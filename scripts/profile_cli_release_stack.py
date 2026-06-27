#!/usr/bin/env python3
"""Run release-binary Lash CLI startup scenarios across Tokio stack sizes.

Unlike profile_cli_stack.py, this harness does not use the bench feature or the
hidden UI benchmark path. It exercises ordinary `lash` startup/error paths with
`LASH_TOKIO_STACK_BYTES` set, so stack overflows in the shipped binary are
captured as failed samples.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import pty
import shutil
import select
import struct
import subprocess
import sys
import termios
import time
from pathlib import Path


DEFAULT_STACKS = [
    512 * 1024,
    1024 * 1024,
    1536 * 1024,
    2 * 1024 * 1024,
    4 * 1024 * 1024,
    8 * 1024 * 1024,
]
STACK_BUDGET_BYTES = 2 * 1024 * 1024

SCENARIOS: dict[str, dict[str, object]] = {
    "info_standard": {
        "args": ["--info"],
        "expected_returncode": 0,
        "stdout_contains": "not configured",
    },
    "info_rlm": {
        "args": ["--info", "--execution-mode", "rlm"],
        "expected_returncode": 0,
        "stdout_contains": "not configured",
    },
    "json_requires_print": {
        "args": ["--mode", "json"],
        "expected_returncode": 1,
        "stderr_contains": "--mode json",
    },
    "print_standard_echo": {
        "args": [
            "--model",
            "test/cli-e2e-model",
            "--print",
            "hello from pty",
        ],
        "expected_returncode": 0,
        "stdout_contains": "test-provider echo: hello from pty",
        "provider_scenario": "standard-echo",
        "required_features": ["test-provider"],
    },
    "interactive_steer_escape": {
        "args": ["--model", "test/cli-e2e-model"],
        "expected_returncode": 0,
        "pty": True,
        "pty_expected_text": "test-provider echo: queued after escape",
        "pty_interrupt_text": "Manually interrupted.",
        "provider_scenario": "standard-slow-echo",
        "required_features": ["test-provider"],
    },
}
KNOWN_SCENARIOS = list(SCENARIOS)
DEFAULT_SCENARIOS = KNOWN_SCENARIOS.copy()
STACK_OVERFLOW_MARKERS = (
    "stack overflow",
    "has overflowed its stack",
    "fatal runtime error: stack overflow",
    "fatal runtime error: stack overflow, aborting",
    "aborted (core dumped)",
)


def parse_size(value: str) -> int:
    raw = value.strip().lower().replace("_", "")
    multiplier = 1
    for suffix, factor in (
        ("kib", 1024),
        ("kb", 1024),
        ("k", 1024),
        ("mib", 1024 * 1024),
        ("mb", 1024 * 1024),
        ("m", 1024 * 1024),
    ):
        if raw.endswith(suffix):
            raw = raw[: -len(suffix)]
            multiplier = factor
            break
    try:
        return int(raw) * multiplier
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid stack size `{value}`") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binary",
        type=Path,
        help="Path to lash. Defaults to CARGO_TARGET_DIR/{release,debug}/lash.",
    )
    parser.add_argument(
        "--build",
        dest="build",
        action="store_true",
        default=True,
        help="Build lash before running the matrix (default: enabled).",
    )
    parser.add_argument(
        "--no-build",
        dest="build",
        action="store_false",
        help="Skip rebuilding and use the existing binary.",
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="Use target/release/lash instead of target/debug/lash.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Release CLI stack scenario to run. Defaults to all known scenarios.",
    )
    parser.add_argument(
        "--stack-bytes",
        action="append",
        type=parse_size,
        default=[],
        help="Worker stack size to test, e.g. 768k, 2m, 4m. May be repeated.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=120,
        help="Timeout for each stack-size sample.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Write matrix JSON here. Defaults under .benchmarks/cli-release-stack/.",
    )
    parser.add_argument(
        "--strict-failures",
        action="store_true",
        help=(
            "Exit non-zero when any stack-size sample fails. By default the matrix "
            "exits successfully if every scenario has at least one passing stack size."
        ),
    )
    parser.add_argument(
        "--cargo-feature",
        action="append",
        default=[],
        help="Cargo feature to enable when building lash-cli. May be repeated.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def cargo_target_dir(root: Path) -> Path:
    value = os.environ.get("CARGO_TARGET_DIR")
    if value:
        path = Path(value)
        return path if path.is_absolute() else root / path
    return root / "target"


def default_out(root: Path) -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / ".benchmarks" / "cli-release-stack" / f"cli-release-stack-{stamp}.json"


def resolve_binary(args: argparse.Namespace, root: Path) -> Path:
    if args.binary:
        return args.binary
    profile = "release" if args.release else "debug"
    return cargo_target_dir(root) / profile / "lash"


def effective_cargo_features(args: argparse.Namespace, scenarios: list[str]) -> list[str]:
    features = list(args.cargo_feature)
    for feature in required_features_for_scenarios(scenarios):
        if feature not in features:
            features.append(feature)
    return features


def required_features_for_scenarios(scenarios: list[str]) -> list[str]:
    features: list[str] = []
    for scenario in scenarios:
        for feature in SCENARIOS[scenario].get("required_features", []):
            if isinstance(feature, str) and feature not in features:
                features.append(feature)
    return features


def maybe_build(args: argparse.Namespace, root: Path, scenarios: list[str]) -> list[str]:
    features = effective_cargo_features(args, scenarios)
    if not args.build:
        return features
    cmd = ["cargo", "build", "-q", "-p", "lash-cli"]
    if features:
        cmd.extend(["--features", ",".join(features)])
    if args.release:
        cmd.append("--release")
    print(f"Building release-path lash binary: {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, cwd=root, check=True)
    return features


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def binary_metadata(binary: Path) -> dict[str, object]:
    stat = binary.stat() if binary.exists() else None
    return {
        "path": str(binary),
        "exists": binary.exists(),
        "size_bytes": stat.st_size if stat else None,
        "mtime_ns": stat.st_mtime_ns if stat else None,
        "sha256": file_sha256(binary),
    }


def git_metadata(root: Path) -> dict[str, object]:
    def run_git(args: list[str]) -> str | None:
        proc = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return None
        return proc.stdout.strip()

    status = run_git(["status", "--porcelain"])
    return {
        "sha": run_git(["rev-parse", "HEAD"]),
        "dirty": bool(status),
    }


def validate_scenarios(requested: list[str]) -> list[str]:
    scenarios = requested or DEFAULT_SCENARIOS
    unknown = sorted(set(scenarios) - set(KNOWN_SCENARIOS))
    if unknown:
        expected = ", ".join(KNOWN_SCENARIOS)
        raise SystemExit(f"error: unknown scenario(s) {', '.join(unknown)}; expected: {expected}")
    return scenarios


def result_home(out: Path, scenario: str, stack_bytes: int) -> Path:
    safe_scenario = scenario.replace("/", "_")
    return out.with_name(f"{out.stem}-{safe_scenario}-{stack_bytes}.home")


def prepare_lash_home(home: Path, scenario: str) -> None:
    spec = SCENARIOS[scenario]
    provider_scenario = spec.get("provider_scenario")
    if not isinstance(provider_scenario, str):
        return
    config = {
        "active_provider": "test",
        "providers": {
            "test": {
                "type": "test",
                "scenario": provider_scenario,
            }
        },
    }
    catalog = {
        "test": {
            "models": {
                "cli-e2e-model": {
                    "limit": {
                        "context": 64000,
                        "output": 4096,
                    }
                }
            }
        }
    }
    cache_dir = home / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (home / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (cache_dir / "models.json").write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n")


def contains_stack_overflow(stdout: str, stderr: str) -> bool:
    combined = f"{stdout}\n{stderr}".lower()
    return any(marker in combined for marker in STACK_OVERFLOW_MARKERS)


def set_pty_size(fd: int, rows: int = 32, cols: int = 120) -> None:
    try:
        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
    except OSError:
        pass


def decode_pty(buffer: bytearray) -> str:
    return buffer.decode("utf-8", errors="replace")


def drain_pty(master_fd: int, buffer: bytearray) -> None:
    while True:
        try:
            chunk = os.read(master_fd, 8192)
        except BlockingIOError:
            return
        except OSError:
            return
        if not chunk:
            return
        buffer.extend(chunk)


def read_pty_until(
    master_fd: int,
    proc: subprocess.Popen[bytes],
    buffer: bytearray,
    needle: str,
    deadline: float,
) -> bool:
    while time.monotonic() < deadline:
        drain_pty(master_fd, buffer)
        if needle in decode_pty(buffer):
            return True
        if proc.poll() is not None:
            drain_pty(master_fd, buffer)
            return needle in decode_pty(buffer)
        timeout = min(0.05, max(0.0, deadline - time.monotonic()))
        if timeout > 0:
            select.select([master_fd], [], [], timeout)
    drain_pty(master_fd, buffer)
    return needle in decode_pty(buffer)


def write_pty(master_fd: int, value: str) -> None:
    os.write(master_fd, value.encode("utf-8"))


def run_pty_sample(
    *,
    root: Path,
    binary: Path,
    scenario: str,
    stack_bytes: int,
    out: Path,
    timeout_seconds: int,
) -> dict[str, object]:
    spec = SCENARIOS[scenario]
    sample_home = result_home(out, scenario, stack_bytes)
    if sample_home.exists():
        shutil.rmtree(sample_home)
    sample_home.mkdir(parents=True, exist_ok=True)
    prepare_lash_home(sample_home, scenario)
    cmd = [str(binary), *list(spec["args"])]
    env = dict(os.environ)
    env["LASH_TOKIO_STACK_BYTES"] = str(stack_bytes)
    env["LASH_HOME"] = str(sample_home)
    env["NO_COLOR"] = "1"
    env["TERM"] = env.get("TERM") or "xterm-256color"
    env.setdefault("LASH_LOG", "warn")

    started = dt.datetime.now(dt.timezone.utc)
    deadline = time.monotonic() + timeout_seconds
    master_fd, slave_fd = pty.openpty()
    set_pty_size(slave_fd)
    buffer = bytearray()
    proc: subprocess.Popen[bytes] | None = None
    timed_out = False
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=root,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            env=env,
            close_fds=True,
        )
        os.close(slave_fd)
        slave_fd = -1
        os.set_blocking(master_fd, False)

        read_pty_until(
            master_fd,
            proc,
            buffer,
            "Message",
            min(deadline, time.monotonic() + 5),
        )
        write_pty(master_fd, "slow initial prompt\r")
        read_pty_until(
            master_fd,
            proc,
            buffer,
            "slow initial prompt",
            min(deadline, time.monotonic() + 5),
        )
        time.sleep(0.2)
        write_pty(master_fd, "queued after escape\r")
        time.sleep(0.2)
        write_pty(master_fd, "\x1b")
        interrupt_text = str(spec.get("pty_interrupt_text", ""))
        saw_interrupt = read_pty_until(master_fd, proc, buffer, interrupt_text, deadline)
        expected_text = str(spec.get("pty_expected_text", ""))
        saw_expected = read_pty_until(master_fd, proc, buffer, expected_text, deadline)
        write_pty(master_fd, "/exit\r")
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)
        drain_pty(master_fd, buffer)
    finally:
        if slave_fd >= 0:
            os.close(slave_fd)
        try:
            os.close(master_fd)
        except OSError:
            pass
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2)

    finished = dt.datetime.now(dt.timezone.utc)
    output = decode_pty(buffer)
    returncode = proc.returncode if proc is not None else None
    failure_reasons = []
    if timed_out:
        failure_reasons.append("timeout")
    if returncode != spec["expected_returncode"]:
        failure_reasons.append("unexpected_returncode")
    if not saw_interrupt:
        failure_reasons.append("missing_interrupt_text")
    if not saw_expected:
        failure_reasons.append("missing_expected_pty_text")
    if contains_stack_overflow(output, ""):
        failure_reasons.append("stack_overflow")
    stack_env_accounted = env["LASH_TOKIO_STACK_BYTES"] == str(stack_bytes)
    if not stack_env_accounted:
        failure_reasons.append("stack_env_not_accounted")

    status = "ok" if not failure_reasons else ("timeout" if timed_out else "failed")
    sample: dict[str, object] = {
        "scenario": scenario,
        "stack_bytes": stack_bytes,
        "status": status,
        "returncode": returncode,
        "expected_returncode": spec["expected_returncode"],
        "started_at": started.isoformat(),
        "duration_ms": round((finished - started).total_seconds() * 1000, 3),
        "lash_home": str(sample_home),
        "stack_env_accounted": stack_env_accounted,
        "stdout_tail": "" if status == "ok" else output[-4000:],
        "stderr_tail": "",
    }
    if failure_reasons:
        sample["failure_reason"] = ",".join(failure_reasons)
        sample["failure_reasons"] = failure_reasons
    return sample


def run_sample(
    *,
    root: Path,
    binary: Path,
    scenario: str,
    stack_bytes: int,
    out: Path,
    timeout_seconds: int,
) -> dict[str, object]:
    spec = SCENARIOS[scenario]
    if spec.get("pty") is True:
        return run_pty_sample(
            root=root,
            binary=binary,
            scenario=scenario,
            stack_bytes=stack_bytes,
            out=out,
            timeout_seconds=timeout_seconds,
        )
    sample_home = result_home(out, scenario, stack_bytes)
    if sample_home.exists():
        shutil.rmtree(sample_home)
    sample_home.mkdir(parents=True, exist_ok=True)
    prepare_lash_home(sample_home, scenario)
    cmd = [str(binary), *list(spec["args"])]
    env = dict(os.environ)
    env["LASH_TOKIO_STACK_BYTES"] = str(stack_bytes)
    env["LASH_HOME"] = str(sample_home)
    env["NO_COLOR"] = "1"

    started = dt.datetime.now(dt.timezone.utc)
    try:
        proc = subprocess.run(
            cmd,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "scenario": scenario,
            "stack_bytes": stack_bytes,
            "status": "timeout",
            "returncode": None,
            "expected_returncode": spec["expected_returncode"],
            "started_at": started.isoformat(),
            "duration_ms": timeout_seconds * 1000,
            "lash_home": str(sample_home),
            "stack_env_accounted": True,
            "stdout_tail": (exc.stdout or "")[-4000:],
            "stderr_tail": (exc.stderr or "")[-4000:],
        }

    finished = dt.datetime.now(dt.timezone.utc)
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    failure_reasons = []
    if proc.returncode != spec["expected_returncode"]:
        failure_reasons.append("unexpected_returncode")
    stdout_contains = spec.get("stdout_contains")
    if isinstance(stdout_contains, str) and stdout_contains not in stdout:
        failure_reasons.append("missing_expected_stdout")
    stderr_contains = spec.get("stderr_contains")
    if isinstance(stderr_contains, str) and stderr_contains not in stderr:
        failure_reasons.append("missing_expected_stderr")
    if contains_stack_overflow(stdout, stderr):
        failure_reasons.append("stack_overflow")
    stack_env_accounted = env["LASH_TOKIO_STACK_BYTES"] == str(stack_bytes)
    if not stack_env_accounted:
        failure_reasons.append("stack_env_not_accounted")

    status = "ok" if not failure_reasons else "failed"
    sample: dict[str, object] = {
        "scenario": scenario,
        "stack_bytes": stack_bytes,
        "status": status,
        "returncode": proc.returncode,
        "expected_returncode": spec["expected_returncode"],
        "started_at": started.isoformat(),
        "duration_ms": round((finished - started).total_seconds() * 1000, 3),
        "lash_home": str(sample_home),
        "stack_env_accounted": stack_env_accounted,
        "stdout_tail": "" if status == "ok" else stdout[-4000:],
        "stderr_tail": "" if status == "ok" else stderr[-4000:],
    }
    if failure_reasons:
        sample["failure_reason"] = ",".join(failure_reasons)
        sample["failure_reasons"] = failure_reasons
    return sample


def first_success(samples: list[dict[str, object]], scenario: str) -> int | None:
    for sample in sorted(
        (sample for sample in samples if sample["scenario"] == scenario),
        key=lambda sample: int(sample["stack_bytes"]),
    ):
        if sample["status"] == "ok" and sample.get("stack_env_accounted") is True:
            return int(sample["stack_bytes"])
    return None


def main() -> int:
    args = parse_args()
    root = repo_root()
    out = args.out or default_out(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    binary = resolve_binary(args, root)
    scenarios = validate_scenarios(args.scenario)
    stacks = sorted(set(args.stack_bytes or DEFAULT_STACKS))

    cargo_features = maybe_build(args, root, scenarios)
    if not binary.exists():
        raise SystemExit(f"error: binary not found: {binary}")

    samples: list[dict[str, object]] = []
    for scenario in scenarios:
        for stack_bytes in stacks:
            print(f"stack={stack_bytes} scenario={scenario}", file=sys.stderr)
            sample = run_sample(
                root=root,
                binary=binary,
                scenario=scenario,
                stack_bytes=stack_bytes,
                out=out,
                timeout_seconds=args.timeout_seconds,
            )
            samples.append(sample)
            print(f"  -> {sample['status']} rc={sample['returncode']}", file=sys.stderr)

    payload = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "kind": "cli-release-stack",
        "binary": str(binary),
        "binary_metadata": binary_metadata(binary),
        "git": git_metadata(root),
        "release": args.release,
        "cargo_features": cargo_features,
        "scenarios": scenarios,
        "known_scenarios": KNOWN_SCENARIOS,
        "missing_known_scenarios": sorted(set(KNOWN_SCENARIOS) - set(scenarios)),
        "stack_bytes": stacks,
        "stack_budget_bytes": STACK_BUDGET_BYTES,
        "first_success_stack_bytes": {
            scenario: first_success(samples, scenario) for scenario in scenarios
        },
        "samples": samples,
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Release CLI stack matrix: {out}", file=sys.stderr)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.strict_failures:
        return 0 if all(sample["status"] == "ok" for sample in samples) else 1
    return 0 if all(first_success(samples, scenario) is not None for scenario in scenarios) else 1


if __name__ == "__main__":
    raise SystemExit(main())
