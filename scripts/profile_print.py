#!/usr/bin/env python3
"""Profile the autonomous `lash -p` path and write structured results."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", help="Prompt to run with `lash -p`.")
    parser.add_argument("--prompt-file", type=Path, help="Read the prompt from a file.")
    parser.add_argument("--runs", type=int, default=5, help="Measured runs (default: 5).")
    parser.add_argument("--warmups", type=int, default=1, help="Warm-up runs (default: 1).")
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
    parser.add_argument(
        "--out",
        type=Path,
        help="Write JSON results to this file. Defaults under .benchmarks/print-profile/.",
    )
    parser.add_argument(
        "--cargo-feature",
        action="append",
        default=[],
        help="Enable one or more Cargo features when building lash-cli.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra environment variables for the lash process.",
    )
    parser.add_argument(
        "--cwd",
        type=Path,
        default=Path.cwd(),
        help="Working directory for the lash command (default: current directory).",
    )
    return parser.parse_args()


def resolve_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return args.prompt_file.read_text()
    if args.prompt:
        return args.prompt
    raise SystemExit("error: provide a prompt or --prompt-file")


def resolve_binary(args: argparse.Namespace, repo_root: Path) -> Path:
    if args.binary:
        return args.binary
    profile = "release" if args.release else "debug"
    return repo_root / "target" / profile / "lash"


def maybe_build(
    binary: Path,
    release: bool,
    repo_root: Path,
    cargo_features: list[str],
) -> None:
    if binary.exists():
        return
    cmd = ["cargo", "build", "-q", "-p", "lash-cli"]
    if cargo_features:
        cmd.extend(["--features", ",".join(cargo_features)])
    if release:
        cmd.append("--release")
    print(f"Building lash binary: {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, cwd=repo_root, check=True)


def parse_env(items: list[str]) -> dict[str, str]:
    env = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"error: invalid --env value `{item}`; expected KEY=VALUE")
        key, value = item.split("=", 1)
        env[key] = value
    return env


def make_output_path(repo_root: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return repo_root / ".benchmarks" / "print-profile" / f"{stamp}.json"


def run_once(
    binary: Path,
    prompt: str,
    cwd: Path,
    env: dict[str, str],
) -> dict[str, object]:
    time_bin = Path("/usr/bin/time")
    has_time = time_bin.exists()
    base_cmd = [str(binary), "-p", prompt]
    cmd = base_cmd
    if has_time:
        cmd = [str(time_bin), "-f", "__TIME__ %e %M"] + base_cmd

    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    max_rss_kb = None
    stderr = proc.stderr
    if has_time:
        lines = stderr.splitlines()
        if lines and lines[-1].startswith("__TIME__ "):
            _, wall_s, rss_kb = lines[-1].split()
            elapsed_ms = float(wall_s) * 1000.0
            max_rss_kb = int(rss_kb)
            stderr = "\n".join(lines[:-1])

    return {
        "returncode": proc.returncode,
        "elapsed_ms": round(elapsed_ms, 3),
        "max_rss_kb": max_rss_kb,
        "stdout_chars": len(proc.stdout),
        "stderr_chars": len(stderr),
        "stdout_preview": proc.stdout[:240],
        "stderr_preview": stderr[:240],
    }


def summarize(results: list[dict[str, object]]) -> dict[str, object]:
    elapsed = [float(item["elapsed_ms"]) for item in results]
    rss_values = [item["max_rss_kb"] for item in results if item["max_rss_kb"] is not None]
    return {
        "runs": len(results),
        "elapsed_ms": {
            "min": round(min(elapsed), 3),
            "median": round(statistics.median(elapsed), 3),
            "max": round(max(elapsed), 3),
            "mean": round(statistics.mean(elapsed), 3),
        },
        "max_rss_kb": {
            "min": min(rss_values),
            "median": int(statistics.median(rss_values)),
            "max": max(rss_values),
            "mean": round(statistics.mean(rss_values), 1),
        }
        if rss_values
        else None,
        "nonzero_exit_codes": sum(1 for item in results if int(item["returncode"]) != 0),
    }


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    prompt = resolve_prompt(args)
    binary = resolve_binary(args, repo_root)
    if args.build:
        maybe_build(binary, args.release, repo_root, args.cargo_feature)
    if not binary.exists():
        raise SystemExit(f"error: binary not found: {binary}")

    env = os.environ.copy()
    env.update(parse_env(args.env))

    warmups = max(args.warmups, 0)
    runs = max(args.runs, 1)

    print(
        f"Profiling {binary} in {args.cwd} with {warmups} warmup(s) and {runs} measured run(s).",
        file=sys.stderr,
    )
    print(f"Prompt: {shlex.quote(prompt[:120])}", file=sys.stderr)

    for _ in range(warmups):
        result = run_once(binary, prompt, args.cwd, env)
        if int(result["returncode"]) != 0:
            raise SystemExit(f"error: warmup run failed: {result}")

    results = []
    for idx in range(runs):
        result = run_once(binary, prompt, args.cwd, env)
        results.append(result)
        print(
            f"run {idx + 1}/{runs}: {result['elapsed_ms']} ms"
            + (
                f", {result['max_rss_kb']} KB rss"
                if result["max_rss_kb"] is not None
                else ""
            ),
            file=sys.stderr,
        )

    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "binary": str(binary),
        "cwd": str(args.cwd),
        "prompt": prompt,
        "warmups": warmups,
        "results": results,
        "summary": summarize(results),
    }

    out_path = args.out or make_output_path(repo_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"out": str(out_path), "summary": payload["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
