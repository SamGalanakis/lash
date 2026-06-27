#!/usr/bin/env python3
"""Run the real lash CLI/UI benchmark across Tokio worker stack sizes.

This is the crash-boundary harness for the shipped `lash` binary. Each stack
size runs in a fresh process with `LASH_TOKIO_STACK_BYTES`, so a Tokio worker
stack overflow is recorded as a failed matrix sample instead of taking down the
whole guard run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_STACKS = [
    256 * 1024,
    384 * 1024,
    512 * 1024,
    768 * 1024,
    1024 * 1024,
    1536 * 1024,
    2 * 1024 * 1024,
    3 * 1024 * 1024,
    4 * 1024 * 1024,
    8 * 1024 * 1024,
]
STACK_BUDGET_BYTES = 2 * 1024 * 1024

KNOWN_UI_SCENARIOS = [
    "history_render",
    "workspace_surface",
    "workspace_overlay",
    "streaming_reactor",
    "slow_snapshot",
    "file_index_storm",
    "timeline_projection",
    "activity_projection",
    "html_export",
    "turn_interrupt_steer_reconciliation",
]
DEFAULT_SCENARIOS = KNOWN_UI_SCENARIOS.copy()
STACK_OVERFLOW_MARKERS = (
    "stack overflow",
    "fatal runtime error",
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
        help="Path to lash. Defaults to target/release/lash or target/debug/lash.",
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
        help="UI perf scenario to run. Defaults to all known UI perf scenarios.",
    )
    parser.add_argument(
        "--stack-bytes",
        action="append",
        type=parse_size,
        default=[],
        help="Worker stack size to test, e.g. 768k, 2m, 4m. May be repeated.",
    )
    parser.add_argument("--runs", type=int, default=1, help="Measured runs per sample.")
    parser.add_argument("--warmups", type=int, default=0, help="Warmups per sample.")
    parser.add_argument(
        "--profile",
        choices=("quick", "full", "stress"),
        default="quick",
        help="UI perf workload profile.",
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
        help="Write matrix JSON here. Defaults under .benchmarks/cli-stack/.",
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
    return root / ".benchmarks" / "cli-stack" / f"cli-stack-{stamp}.json"


def resolve_binary(args: argparse.Namespace, root: Path) -> Path:
    if args.binary:
        return args.binary
    profile = "release" if args.release else "debug"
    return cargo_target_dir(root) / profile / "lash"


def cargo_target_dir(root: Path) -> Path:
    value = os.environ.get("CARGO_TARGET_DIR")
    if value:
        path = Path(value)
        return path if path.is_absolute() else root / path
    return root / "target"


def maybe_build(args: argparse.Namespace, root: Path) -> None:
    if not args.build:
        return
    cmd = ["cargo", "build", "-q", "-p", "lash-cli"]
    features = ["bench", *args.cargo_feature]
    if features:
        cmd.extend(["--features", ",".join(features)])
    if args.release:
        cmd.append("--release")
    print(f"Building lash binary: {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, cwd=root, check=True)


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


def result_path(out: Path, scenario: str, stack_bytes: int) -> Path:
    safe_scenario = scenario.replace("/", "_")
    return out.with_name(f"{out.stem}-{safe_scenario}-{stack_bytes}.ui-perf.json")


def _profile_worker_stack(profile: object) -> object:
    if not isinstance(profile, dict):
        return None
    return profile.get("worker_stack_bytes")


def contains_stack_overflow(stdout: str, stderr: str) -> bool:
    combined = f"{stdout}\n{stderr}".lower()
    return any(marker in combined for marker in STACK_OVERFLOW_MARKERS)


def report_metadata(
    sample_out: Path,
    *,
    scenario: str,
    stack_bytes: int,
) -> dict[str, object]:
    if not sample_out.exists():
        return {
            "status": "failed",
            "failure_reason": "missing_ui_perf_report",
            "reported_worker_stack_bytes": None,
            "stack_accounted": False,
            "reported_scenarios": [],
        }

    try:
        payload = json.loads(sample_out.read_text())
    except json.JSONDecodeError as exc:
        return {
            "status": "failed",
            "failure_reason": "malformed_ui_perf_report",
            "json_error": str(exc),
            "reported_worker_stack_bytes": None,
            "stack_accounted": False,
            "reported_scenarios": [],
        }

    if not isinstance(payload, dict):
        return {
            "status": "failed",
            "failure_reason": "non_object_ui_perf_report",
            "reported_worker_stack_bytes": None,
            "stack_accounted": False,
            "reported_scenarios": [],
        }

    stack_profile = payload.get("stack_profile")
    reported_stack_bytes = _profile_worker_stack(stack_profile)
    parameters = payload.get("parameters")
    parameter_stack_profile = (
        parameters.get("stack_profile") if isinstance(parameters, dict) else None
    )
    parameter_stack_bytes = _profile_worker_stack(parameter_stack_profile)

    reported_scenarios: list[str] = []
    scenario_stack_accounted = False
    result_stack_accounted = False
    scenarios = payload.get("scenarios")
    if isinstance(scenarios, list):
        for item in scenarios:
            if not isinstance(item, dict):
                continue
            item_scenario = item.get("scenario")
            if isinstance(item_scenario, str):
                reported_scenarios.append(item_scenario)
            if item_scenario != scenario:
                continue
            scenario_stack_accounted = _profile_worker_stack(item.get("stack_profile")) == stack_bytes
            results = item.get("results")
            if isinstance(results, list) and results:
                result_stack_accounted = all(
                    isinstance(result, dict)
                    and _profile_worker_stack(result.get("stack_profile")) == stack_bytes
                    for result in results
                )

    stack_accounted = (
        reported_stack_bytes == stack_bytes
        and parameter_stack_bytes == stack_bytes
        and scenario_stack_accounted
        and result_stack_accounted
    )
    scenario_reported = scenario in reported_scenarios
    failure_reasons = []
    if reported_stack_bytes != stack_bytes:
        failure_reasons.append("top_level_stack_size_not_accounted")
    if parameter_stack_bytes != stack_bytes:
        failure_reasons.append("parameter_stack_size_not_accounted")
    if not scenario_stack_accounted:
        failure_reasons.append("scenario_stack_size_not_accounted")
    if not result_stack_accounted:
        failure_reasons.append("result_stack_size_not_accounted")
    if not stack_accounted:
        failure_reasons.append("stack_size_not_accounted")
    if not scenario_reported:
        failure_reasons.append("missing_ui_perf_scenario")

    metadata: dict[str, object] = {
        "status": "ok" if not failure_reasons else "failed",
        "reported_worker_stack_bytes": reported_stack_bytes,
        "reported_stack_profile": stack_profile if isinstance(stack_profile, dict) else None,
        "parameter_stack_accounted": parameter_stack_bytes == stack_bytes,
        "scenario_stack_accounted": scenario_stack_accounted,
        "result_stack_accounted": result_stack_accounted,
        "stack_accounted": stack_accounted,
        "reported_scenarios": reported_scenarios,
    }
    if failure_reasons:
        metadata["failure_reason"] = ",".join(failure_reasons)
        metadata["failure_reasons"] = failure_reasons
    return metadata


def run_sample(
    *,
    root: Path,
    binary: Path,
    scenario: str,
    stack_bytes: int,
    out: Path,
    runs: int,
    warmups: int,
    profile: str,
    timeout_seconds: int,
) -> dict[str, object]:
    sample_out = result_path(out, scenario, stack_bytes)
    cmd = [
        str(binary),
        "--ui-perf-benchmark",
        f"--ui-perf-runs={max(runs, 1)}",
        f"--ui-perf-warmups={max(warmups, 0)}",
        f"--ui-perf-profile={profile}",
        "--ui-perf-scenario",
        scenario,
        f"--ui-perf-out={sample_out}",
    ]
    env = dict(os.environ)
    env["LASH_TOKIO_STACK_BYTES"] = str(stack_bytes)
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
            "started_at": started.isoformat(),
            "duration_ms": timeout_seconds * 1000,
            "ui_perf_out": str(sample_out),
            "stdout_tail": (exc.stdout or "")[-4000:],
            "stderr_tail": (exc.stderr or "")[-4000:],
        }

    finished = dt.datetime.now(dt.timezone.utc)
    metadata: dict[str, object] = {}
    status = "failed"
    if proc.returncode == 0:
        metadata = report_metadata(sample_out, scenario=scenario, stack_bytes=stack_bytes)
        status = str(metadata.pop("status"))
    if contains_stack_overflow(proc.stdout or "", proc.stderr or ""):
        reasons = list(metadata.get("failure_reasons", []))
        if "stack_overflow" not in reasons:
            reasons.append("stack_overflow")
        metadata["failure_reasons"] = reasons
        metadata["failure_reason"] = ",".join(reasons)
        status = "failed"

    sample_failed = status != "ok"
    sample = {
        "scenario": scenario,
        "stack_bytes": stack_bytes,
        "status": status,
        "returncode": proc.returncode,
        "started_at": started.isoformat(),
        "duration_ms": round((finished - started).total_seconds() * 1000, 3),
        "ui_perf_out": str(sample_out),
        "stdout_tail": proc.stdout[-4000:] if sample_failed else "",
        "stderr_tail": proc.stderr[-4000:] if sample_failed else "",
    }
    sample.update(metadata)
    return sample


def first_success(samples: list[dict[str, object]], scenario: str) -> int | None:
    for sample in sorted(
        (sample for sample in samples if sample["scenario"] == scenario),
        key=lambda sample: int(sample["stack_bytes"]),
    ):
        reported_scenarios = sample.get("reported_scenarios", [])
        if (
            sample["status"] == "ok"
            and sample.get("stack_accounted") is True
            and isinstance(reported_scenarios, list)
            and scenario in reported_scenarios
        ):
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
                profile=args.profile,
                timeout_seconds=args.timeout_seconds,
            )
            samples.append(sample)
            print(f"  -> {sample['status']} rc={sample['returncode']}", file=sys.stderr)

    payload = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "kind": "cli-stack",
        "binary": str(binary),
        "binary_metadata": binary_metadata(binary),
        "git": git_metadata(root),
        "release": args.release,
        "cargo_features": ["bench", *args.cargo_feature],
        "profile": args.profile,
        "runs": max(args.runs, 1),
        "warmups": max(args.warmups, 0),
        "scenarios": scenarios,
        "known_scenarios": KNOWN_UI_SCENARIOS,
        "missing_known_scenarios": sorted(set(KNOWN_UI_SCENARIOS) - set(scenarios)),
        "stack_bytes": stacks,
        "stack_budget_bytes": STACK_BUDGET_BYTES,
        "first_success_stack_bytes": {
            scenario: first_success(samples, scenario) for scenario in scenarios
        },
        "samples": samples,
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"CLI stack matrix: {out}", file=sys.stderr)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.strict_failures:
        return 0 if all(sample["status"] == "ok" for sample in samples) else 1
    return 0 if all(first_success(samples, scenario) is not None for scenario in scenarios) else 1


if __name__ == "__main__":
    raise SystemExit(main())
