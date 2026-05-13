#!/usr/bin/env python3
"""Run Lashlang perf/profile sweeps and write a structured JSON report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCENARIOS = [
    "baseline",
    "language_surface",
    "async_await",
    "direct_unwrap",
    "general_parallel",
    "loop_control",
    "indexed_assignment",
    "projected_values",
    "large_data",
    "cache_pressure",
    "projected_operations",
    "type_system_stress",
    "wrapped_error_paths",
    "tool_control_surface",
    "snapshot_projected_state",
    "continue_as_seed_surface",
]

PERF_MODES = [
    "execute",
    "parse",
    "compile",
    "block",
    "cached_block",
    "cached_session_block",
    "cold_once",
    "prewarmed_once",
    "cached_cold_once",
    "snapshot",
]

DEFAULT_PERF_MODES = ["execute", "parse", "compile", "cached_session_block", "snapshot"]
ONCE_MODES = {"cold_once", "prewarmed_once", "cached_cold_once"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", action="append", default=[], help="Scenario to run; defaults to all.")
    parser.add_argument("--mode", action="append", default=[], help="Perf mode to run; defaults to the standard sweep.")
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--profile-iterations", type=int, default=10_000)
    parser.add_argument("--profile-scenario", action="append", default=[], help="Profile scenario; defaults to aggregate all.")
    parser.add_argument("--skip-perf", action="store_true")
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--build", dest="build", action="store_true", default=True)
    parser.add_argument("--no-build", dest="build", action="store_false")
    parser.add_argument("--out", type=Path, help="Defaults under .benchmarks/lashlang-perf/.")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_out(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / ".benchmarks" / "lashlang-perf" / f"{stamp}.json"


def resolve_requested(values: list[str], known: list[str], default: list[str]) -> list[str]:
    requested = default if not values else values
    resolved: list[str] = []
    for value in requested:
        if value == "all":
            for item in known:
                if item not in resolved:
                    resolved.append(item)
            continue
        if value not in known:
            expected = ", ".join([*known, "all"])
            raise SystemExit(f"error: unknown value `{value}`; expected one of: {expected}")
        if value not in resolved:
            resolved.append(value)
    return resolved


def resolve_profile_scenarios(values: list[str]) -> list[str]:
    requested = values or ["all"]
    resolved: list[str] = []
    for value in requested:
        if value != "all" and value not in SCENARIOS:
            expected = ", ".join([*SCENARIOS, "all"])
            raise SystemExit(f"error: unknown profile scenario `{value}`; expected one of: {expected}")
        if value not in resolved:
            resolved.append(value)
    return resolved


def maybe_build(root: Path, debug: bool, build: bool) -> None:
    if not build:
        return
    cmd = ["cargo", "build", "-q", "-p", "lashlang", "--example", "perf", "--example", "profile"]
    if not debug:
        cmd.append("--release")
    print(f"Building Lashlang profiling examples: {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, cwd=root, check=True)


def example_path(root: Path, debug: bool, name: str) -> Path:
    return root / "target" / ("debug" if debug else "release") / "examples" / name


def run_command(root: Path, cmd: list[str]) -> str:
    proc = subprocess.run(cmd, cwd=root, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout, file=sys.stderr, end="")
        if proc.stderr:
            print(proc.stderr, file=sys.stderr, end="")
        raise SystemExit(proc.returncode)
    return proc.stdout


def parse_scalar(value: str) -> str | int | float:
    value = value.strip()
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_perf_output(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for line in text.splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        result[key.strip()] = parse_scalar(value)
    return result


def parse_hotspot_line(line: str) -> dict[str, Any] | None:
    parts = line.split()
    if len(parts) < 4:
        return None
    stat: dict[str, Any] = {"name": parts[0]}
    for part in parts[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        stat[key] = parse_scalar(value)
    return stat


def parse_profile_output(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {"instruction_hotspots": [], "builtin_hotspots": []}
    section: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped == "lashlang profile":
            continue
        if stripped == "instruction_hotspots:":
            section = "instruction_hotspots"
            continue
        if stripped == "builtin_hotspots:":
            section = "builtin_hotspots"
            continue
        if section:
            stat = parse_hotspot_line(stripped)
            if stat:
                result[section].append(stat)
            continue
        if ": " in stripped:
            key, value = stripped.split(": ", 1)
            result[key] = parse_scalar(value)
    return result


def git_info(root: Path) -> dict[str, Any]:
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "diff", "--quiet", "--ignore-submodules", "--"], cwd=root).returncode != 0
    return {"sha": sha or "unknown", "dirty": dirty}


def main() -> int:
    args = parse_args()
    root = repo_root()
    scenarios = resolve_requested(args.scenario, SCENARIOS, SCENARIOS)
    modes = resolve_requested(args.mode, PERF_MODES, DEFAULT_PERF_MODES)
    profile_scenarios = resolve_profile_scenarios(args.profile_scenario)

    maybe_build(root, args.debug, args.build)
    perf_bin = example_path(root, args.debug, "perf")
    profile_bin = example_path(root, args.debug, "profile")

    perf_results = []
    if not args.skip_perf:
        if not perf_bin.exists():
            raise SystemExit(f"error: perf example not found: {perf_bin}")
        for mode in modes:
            iterations = 1 if mode in ONCE_MODES else max(args.iterations, 1)
            for scenario in scenarios:
                parsed = parse_perf_output(run_command(root, [str(perf_bin), mode, scenario, str(iterations)]))
                parsed["mode_arg"] = mode
                parsed["scenario_arg"] = scenario
                perf_results.append(parsed)

    profile_results = []
    if not args.skip_profile:
        if not profile_bin.exists():
            raise SystemExit(f"error: profile example not found: {profile_bin}")
        for scenario in profile_scenarios:
            parsed = parse_profile_output(run_command(root, [str(profile_bin), scenario, str(max(args.profile_iterations, 1))]))
            parsed["scenario_arg"] = scenario
            profile_results.append(parsed)

    out_path = args.out or default_out(root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "lashlang-perf",
        "git": git_info(root),
        "build_mode": "debug" if args.debug else "release",
        "parameters": {
            "scenarios": scenarios,
            "modes": modes,
            "iterations": max(args.iterations, 1),
            "profile_scenarios": profile_scenarios,
            "profile_iterations": max(args.profile_iterations, 1),
            "skip_perf": args.skip_perf,
            "skip_profile": args.skip_profile,
        },
        "perf_results": perf_results,
        "profile_results": profile_results,
    }
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({"out": str(out_path), "perf_results": len(perf_results), "profile_results": len(profile_results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
