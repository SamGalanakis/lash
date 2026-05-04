#!/usr/bin/env python3
"""Run the synthetic non-provider UI perf benchmark and write structured results."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binary",
        type=Path,
        help="Path to the lash binary. Defaults to target/release/lash or target/debug/lash.",
    )
    parser.add_argument(
        "--build",
        dest="build",
        action="store_true",
        default=True,
        help="Build the selected binary before running the benchmark (default: enabled).",
    )
    parser.add_argument(
        "--no-build",
        dest="build",
        action="store_false",
        help="Skip rebuilding and use the existing binary as-is.",
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="Use target/release/lash instead of target/debug/lash.",
    )
    parser.add_argument("--runs", type=int, default=5, help="Measured runs (default: 5).")
    parser.add_argument("--warmups", type=int, default=1, help="Warm-up runs (default: 1).")
    parser.add_argument(
        "--profile",
        choices=("quick", "full", "stress"),
        default="quick",
        help="Workload profile (default: quick).",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Limit to one or more UI perf scenarios, or use all.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Write JSON results to this file. Defaults under .benchmarks/ui-perf/.",
    )
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        type=Path,
        help="Include one or more baseline reports as comparison inputs.",
    )
    parser.add_argument(
        "--enforce-budgets",
        action="store_true",
        help="Exit non-zero when a UI perf budget is exceeded.",
    )
    parser.add_argument(
        "--dhat",
        action="store_true",
        help="Build with the dhat heap profiler and record a measured-window heap profile.",
    )
    parser.add_argument(
        "--dhat-out",
        type=Path,
        help="Write the dhat heap profile to this file. Defaults next to --out.",
    )
    parser.add_argument(
        "--dhat-frames",
        type=int,
        default=16,
        help="Trim dhat backtraces to this many frames (default: 16).",
    )
    parser.add_argument(
        "--cargo-feature",
        action="append",
        default=[],
        help="Enable one or more Cargo features when building lash-cli.",
    )
    return parser.parse_args()


def resolve_binary(args: argparse.Namespace, repo_root: Path) -> Path:
    if args.binary:
        return args.binary
    profile = "release" if args.release else "debug"
    return repo_root / "target" / profile / "lash"


def maybe_build(
    binary: Path,
    release: bool,
    repo_root: Path,
    force: bool,
    dhat: bool,
    cargo_features: list[str],
) -> None:
    if not force and binary.exists():
        return
    cmd = ["cargo", "build", "-q", "-p", "lash-cli"]
    feature_list = list(cargo_features)
    if dhat:
        feature_list.append("dhat-heap")
    if feature_list:
        cmd.extend(["--features", ",".join(feature_list)])
    if release:
        cmd.append("--release")
    env = None
    if dhat and release:
        env = dict(os.environ)
        env["CARGO_PROFILE_RELEASE_STRIP"] = "none"
        env["CARGO_PROFILE_RELEASE_DEBUG"] = "1"
    print(f"Building lash binary: {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, cwd=repo_root, check=True, env=env)


def default_dhat_out(out: Path) -> Path:
    return out.with_name(f"{out.stem}.dhat.json")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    binary = resolve_binary(args, repo_root)
    maybe_build(
        binary,
        args.release,
        repo_root,
        args.build,
        args.dhat,
        args.cargo_feature,
    )
    if not binary.exists():
        raise SystemExit(f"error: binary not found: {binary}")

    cmd = [
        str(binary),
        "--ui-perf-benchmark",
        f"--ui-perf-runs={max(args.runs, 1)}",
        f"--ui-perf-warmups={max(args.warmups, 0)}",
        f"--ui-perf-profile={args.profile}",
    ]
    if args.out:
        cmd.append(f"--ui-perf-out={args.out}")
    if args.enforce_budgets:
        cmd.append("--ui-perf-enforce-budgets")
    if args.dhat:
        cmd.append("--ui-perf-dhat")
        if args.dhat_out:
            dhat_out = args.dhat_out
        elif args.out:
            dhat_out = default_dhat_out(args.out)
        else:
            dhat_out = None
        if dhat_out:
            cmd.append(f"--ui-perf-dhat-out={dhat_out}")
        cmd.append(f"--ui-perf-dhat-frames={max(args.dhat_frames, 1)}")
    for scenario in args.scenario:
        cmd.extend(["--ui-perf-scenario", scenario])
    for compare in args.compare:
        cmd.extend(["--ui-perf-compare", str(compare)])

    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr, file=sys.stderr, end="")
        raise SystemExit(proc.returncode)

    payload = json.loads(proc.stdout)
    out_path = Path(payload["out"])
    dhat_out = payload.get("dhat_out")
    print(f"UI perf report: {out_path}", file=sys.stderr)
    if dhat_out:
        print(f"dhat heap profile: {dhat_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
