#!/usr/bin/env python3
"""Run runtime perf scenarios across Tokio worker stack sizes.

This is a crash-boundary harness for embedded Tokio stack regressions. Each
stack size runs in a fresh process with lash-cli's hidden runtime-perf Tokio
runtime builder knob, so stack overflows are recorded as failed samples instead
of killing the whole matrix.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_STACKS = [
    64 * 1024,
    80 * 1024,
    96 * 1024,
    128 * 1024,
    192 * 1024,
    256 * 1024,
    384 * 1024,
    512 * 1024,
    768 * 1024,
    1024 * 1024,
    1536 * 1024,
    2 * 1024 * 1024,
    3 * 1024 * 1024,
    4 * 1024 * 1024,
    6 * 1024 * 1024,
    8 * 1024 * 1024,
]

DEFAULT_SCENARIOS = [
    "standard",
    "rlm",
    "rlm_globals",
    "rlm_tool_calls",
    "rlm_process_handles",
    "rlm_large_tool_surface",
    "tool_discovery_search",
    "scoped_effect_controller",
    "turn_checkpoint",
]


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
        help="Path to lash. Defaults to target/release/lash or target/debug/lash.",
    )
    parser.add_argument(
        "--build",
        dest="build",
        action="store_true",
        default=True,
        help="Build lash-cli before running the matrix (default: enabled).",
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
        help=(
            "Runtime perf scenario to run. Defaults to the stack-sensitive runtime "
            "coverage set. May be repeated."
        ),
    )
    parser.add_argument(
        "--stack-bytes",
        action="append",
        type=parse_size,
        default=[],
        help="Worker stack size to test, e.g. 768k, 2m, 8388608. May be repeated.",
    )
    parser.add_argument("--runs", type=int, default=1, help="Measured runs per sample.")
    parser.add_argument("--warmups", type=int, default=0, help="Warmups per sample.")
    parser.add_argument("--turns", type=int, default=1, help="Turns per measured run.")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=120,
        help="Timeout for each stack-size sample.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Write matrix JSON here. Defaults under .benchmarks/runtime-stack/.",
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
        help="Additional Cargo feature to enable when building lash-cli.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_out(root: Path) -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / ".benchmarks" / "runtime-stack" / f"runtime-stack-{stamp}.json"


def resolve_binary(args: argparse.Namespace, root: Path) -> Path:
    if args.binary:
        return args.binary
    profile = "release" if args.release else "debug"
    return root / "target" / profile / "lash"


def maybe_build(args: argparse.Namespace, root: Path) -> None:
    if not args.build:
        return
    cmd = ["cargo", "build", "-q", "-p", "lash-cli"]
    features = ["bench", *args.cargo_feature]
    cmd.extend(["--features", ",".join(features)])
    if args.release:
        cmd.append("--release")
    print(f"Building lash binary: {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, cwd=root, check=True)


def result_path(out: Path, scenario: str, stack_bytes: int) -> Path:
    safe_scenario = scenario.replace("/", "_")
    return out.with_name(f"{out.stem}-{safe_scenario}-{stack_bytes}.runtime-perf.json")


def run_sample(
    *,
    root: Path,
    binary: Path,
    scenario: str,
    stack_bytes: int,
    out: Path,
    runs: int,
    warmups: int,
    turns: int,
    timeout_seconds: int,
) -> dict[str, object]:
    sample_out = result_path(out, scenario, stack_bytes)
    cmd = [
        str(binary),
        "--runtime-perf-benchmark",
        f"--runtime-perf-runs={max(runs, 1)}",
        f"--runtime-perf-warmups={max(warmups, 0)}",
        f"--runtime-perf-turns={max(turns, 1)}",
        f"--runtime-perf-worker-stack-bytes={stack_bytes}",
        "--runtime-perf-scenario",
        scenario,
        f"--runtime-perf-out={sample_out}",
    ]
    started = dt.datetime.now(dt.timezone.utc)
    try:
        proc = subprocess.run(
            cmd,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "scenario": scenario,
            "stack_bytes": stack_bytes,
            "status": "timeout",
            "returncode": None,
            "started_at": started.isoformat(),
            "duration_ms": timeout_seconds * 1000,
            "runtime_perf_out": str(sample_out),
            "stdout_tail": (exc.stdout or "")[-4000:],
            "stderr_tail": (exc.stderr or "")[-4000:],
        }

    finished = dt.datetime.now(dt.timezone.utc)
    status = "ok" if proc.returncode == 0 else "failed"
    payload: dict[str, object] | None = None
    if proc.returncode == 0 and sample_out.exists():
        payload = json.loads(sample_out.read_text())
    return {
        "scenario": scenario,
        "stack_bytes": stack_bytes,
        "status": status,
        "returncode": proc.returncode,
        "started_at": started.isoformat(),
        "duration_ms": round((finished - started).total_seconds() * 1000, 3),
        "runtime_perf_out": str(sample_out),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "summary": (payload or {}).get("summary"),
    }


def first_success(samples: list[dict[str, object]], scenario: str) -> int | None:
    for sample in sorted(
        (sample for sample in samples if sample["scenario"] == scenario),
        key=lambda sample: int(sample["stack_bytes"]),
    ):
        if sample["status"] == "ok":
            return int(sample["stack_bytes"])
    return None


def main() -> int:
    args = parse_args()
    root = repo_root()
    out = args.out or default_out(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    binary = resolve_binary(args, root)
    scenarios = args.scenario or DEFAULT_SCENARIOS
    stacks = sorted(set(args.stack_bytes or DEFAULT_STACKS))

    maybe_build(args, root)
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
                runs=args.runs,
                warmups=args.warmups,
                turns=args.turns,
                timeout_seconds=args.timeout_seconds,
            )
            samples.append(sample)
            print(f"  -> {sample['status']} rc={sample['returncode']}", file=sys.stderr)

    payload = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "binary": str(binary),
        "release": args.release,
        "runs": max(args.runs, 1),
        "warmups": max(args.warmups, 0),
        "turns": max(args.turns, 1),
        "scenarios": scenarios,
        "stack_bytes": stacks,
        "first_success_stack_bytes": {
            scenario: first_success(samples, scenario) for scenario in scenarios
        },
        "samples": samples,
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Runtime stack matrix: {out}", file=sys.stderr)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.strict_failures:
        return 0 if all(sample["status"] == "ok" for sample in samples) else 1
    return 0 if all(first_success(samples, scenario) is not None for scenario in scenarios) else 1


if __name__ == "__main__":
    raise SystemExit(main())
