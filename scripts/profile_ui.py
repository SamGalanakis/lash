#!/usr/bin/env python3
"""Run the synthetic non-provider UI perf benchmark and write structured results."""

from __future__ import annotations

import argparse
import json
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
        action="store_true",
        help="Build the selected binary first if it does not exist.",
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="Use target/release/lash instead of target/debug/lash.",
    )
    parser.add_argument("--runs", type=int, default=5, help="Measured runs (default: 5).")
    parser.add_argument("--warmups", type=int, default=1, help="Warm-up runs (default: 1).")
    parser.add_argument(
        "--out",
        type=Path,
        help="Write JSON results to this file. Defaults under .benchmarks/ui-perf/.",
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


def maybe_build(binary: Path, release: bool, repo_root: Path, cargo_features: list[str]) -> None:
    if binary.exists():
        return
    cmd = ["cargo", "build", "-q", "-p", "lash-cli"]
    if cargo_features:
        cmd.extend(["--features", ",".join(cargo_features)])
    if release:
        cmd.append("--release")
    print(f"Building lash binary: {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, cwd=repo_root, check=True)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    binary = resolve_binary(args, repo_root)
    if args.build:
        maybe_build(binary, args.release, repo_root, args.cargo_feature)
    if not binary.exists():
        raise SystemExit(f"error: binary not found: {binary}")

    cmd = [
        str(binary),
        "--ui-perf-benchmark",
        f"--ui-perf-runs={max(args.runs, 1)}",
        f"--ui-perf-warmups={max(args.warmups, 0)}",
    ]
    if args.out:
        cmd.append(f"--ui-perf-out={args.out}")

    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr, file=sys.stderr, end="")
        raise SystemExit(proc.returncode)

    payload = json.loads(proc.stdout)
    out_path = Path(payload["out"])
    print(f"UI perf report: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
