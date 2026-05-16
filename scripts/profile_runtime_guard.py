#!/usr/bin/env python3
"""Run the runtime performance guard suite.

The guard records three complementary signals for Lash/RLM runtime changes:

- normal runtime-perf timing, allocation, RSS, and phase metrics
- Tokio worker stack sensitivity
- optional DHAT heap profiles for call-stack attribution
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_SCENARIOS = ["rlm_large_tool_surface", "turn_checkpoint"]
DEFAULT_STACKS = ["64k", "128k", "256k", "320k", "512k", "1m", "8m"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", action="store_true", help="Run release binaries.")
    parser.add_argument(
        "--profile",
        choices=["quick", "full"],
        default="quick",
        help="Runtime perf size preset.",
    )
    parser.add_argument("--runs", type=int, help="Measured runs for each runtime lane.")
    parser.add_argument("--warmups", type=int, help="Warmups for each runtime lane.")
    parser.add_argument("--turns", type=int, help="Turns for each runtime lane.")
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help=(
            "Scenario to track. Defaults to rlm_large_tool_surface and "
            "turn_checkpoint. May be repeated."
        ),
    )
    parser.add_argument(
        "--stack-bytes",
        action="append",
        default=[],
        help="Worker stack size for the stack lane, e.g. 256k or 8388608.",
    )
    parser.add_argument(
        "--skip-dhat",
        action="store_true",
        help="Skip DHAT heap profiling.",
    )
    parser.add_argument(
        "--dhat-frames",
        type=int,
        default=24,
        help="Trim DHAT backtraces to this many frames.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=180,
        help="Timeout for each stack matrix sample.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Write the guard JSON here. Defaults under .benchmarks/runtime-guard/.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Compare against a previous runtime guard JSON.",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero when baseline comparison finds regressions.",
    )
    parser.add_argument(
        "--time-regression",
        type=float,
        default=0.15,
        help="Allowed median runtime increase vs baseline before failing.",
    )
    parser.add_argument(
        "--alloc-regression",
        type=float,
        default=0.15,
        help="Allowed median allocation/live-byte increase vs baseline before failing.",
    )
    parser.add_argument(
        "--rss-regression",
        type=float,
        default=0.25,
        help="Allowed median RSS growth increase vs baseline before failing.",
    )
    parser.add_argument(
        "--stack-regression-bytes",
        type=int,
        default=64 * 1024,
        help="Allowed first-success stack increase vs baseline before failing.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_out(root: Path) -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / ".benchmarks" / "runtime-guard" / f"runtime-guard-{stamp}.json"


def run_json(cmd: list[str], root: Path, *, allow_failure: bool = False) -> tuple[int, dict[str, Any], str]:
    proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")
    if proc.returncode != 0 and not allow_failure:
        if proc.stdout:
            print(proc.stdout, end="")
        raise SystemExit(proc.returncode)
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"error: command did not emit JSON: {' '.join(cmd)}\n{proc.stdout[-4000:]}"
        ) from exc
    return proc.returncode, payload, proc.stderr


def script_cmd(script: str, root: Path) -> list[str]:
    return [sys.executable, str(root / "scripts" / script)]


def runtime_common_args(args: argparse.Namespace) -> list[str]:
    values = ["--profile", args.profile]
    if args.release:
        values.append("--release")
    if args.runs is not None:
        values.append(f"--runs={args.runs}")
    if args.warmups is not None:
        values.append(f"--warmups={args.warmups}")
    if args.turns is not None:
        values.append(f"--turns={args.turns}")
    for scenario in args.scenario or DEFAULT_SCENARIOS:
        values.extend(["--scenario", scenario])
    return values


def metric(summary: dict[str, Any], name: str) -> float | None:
    value = summary.get(name)
    if not isinstance(value, dict):
        return None
    median = value.get("median")
    if isinstance(median, int | float):
        return float(median)
    return None


def runtime_summaries(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        item["scenario"]: item
        for item in report.get("summary", [])
        if isinstance(item, dict) and isinstance(item.get("scenario"), str)
    }


def compare_ratio(
    *,
    findings: list[dict[str, Any]],
    scenario: str,
    metric_name: str,
    baseline_summary: dict[str, Any],
    current_summary: dict[str, Any],
    tolerance: float,
) -> None:
    baseline = metric(baseline_summary, metric_name)
    current = metric(current_summary, metric_name)
    if baseline is None or current is None:
        return
    allowed = baseline * (1.0 + tolerance)
    if current > allowed:
        findings.append(
            {
                "kind": "runtime_metric_regression",
                "scenario": scenario,
                "metric": metric_name,
                "baseline": baseline,
                "current": current,
                "allowed": allowed,
            }
        )


def compare_reports(
    baseline: dict[str, Any],
    current: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    baseline_runtime = runtime_summaries(baseline.get("normal_runtime", {}).get("report", {}))
    current_runtime = runtime_summaries(current.get("normal_runtime", {}).get("report", {}))
    for scenario, current_summary in current_runtime.items():
        baseline_summary = baseline_runtime.get(scenario)
        if not baseline_summary:
            continue
        compare_ratio(
            findings=findings,
            scenario=scenario,
            metric_name="run_turn_ms",
            baseline_summary=baseline_summary,
            current_summary=current_summary,
            tolerance=args.time_regression,
        )
        for metric_name in ("run_turn_alloc_bytes", "total_alloc_bytes", "total_live_bytes"):
            compare_ratio(
                findings=findings,
                scenario=scenario,
                metric_name=metric_name,
                baseline_summary=baseline_summary,
                current_summary=current_summary,
                tolerance=args.alloc_regression,
            )
        compare_ratio(
            findings=findings,
            scenario=scenario,
            metric_name="rss_growth_kb",
            baseline_summary=baseline_summary,
            current_summary=current_summary,
            tolerance=args.rss_regression,
        )

    baseline_stack = baseline.get("stack_sensitivity", {}).get("first_success_stack_bytes", {})
    current_stack = current.get("stack_sensitivity", {}).get("first_success_stack_bytes", {})
    for scenario, current_value in current_stack.items():
        baseline_value = baseline_stack.get(scenario)
        if not isinstance(baseline_value, int) or not isinstance(current_value, int):
            continue
        allowed = baseline_value + args.stack_regression_bytes
        if current_value > allowed:
            findings.append(
                {
                    "kind": "stack_threshold_regression",
                    "scenario": scenario,
                    "baseline": baseline_value,
                    "current": current_value,
                    "allowed": allowed,
                }
            )
    return findings


def main() -> int:
    args = parse_args()
    root = repo_root()
    out = args.out or default_out(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    scenarios = args.scenario or DEFAULT_SCENARIOS

    normal_out = out.with_name(f"{out.stem}.runtime-perf.json")
    normal_cmd = [
        *script_cmd("profile_runtime.py", root),
        *runtime_common_args(args),
        f"--out={normal_out}",
    ]
    print("Running runtime perf lane", file=sys.stderr)
    normal_rc, normal_report, _ = run_json(normal_cmd, root)

    stack_out = out.with_name(f"{out.stem}.runtime-stack.json")
    stack_cmd = [
        *script_cmd("profile_runtime_stack.py", root),
        "--no-build",
        f"--out={stack_out}",
        f"--runs={args.runs if args.runs is not None else 1}",
        f"--warmups={args.warmups if args.warmups is not None else 0}",
        f"--turns={args.turns if args.turns is not None else 1}",
        f"--timeout-seconds={args.timeout_seconds}",
    ]
    if args.release:
        stack_cmd.append("--release")
    for scenario in scenarios:
        stack_cmd.extend(["--scenario", scenario])
    for stack in args.stack_bytes or DEFAULT_STACKS:
        stack_cmd.extend(["--stack-bytes", stack])
    print("Running stack sensitivity lane", file=sys.stderr)
    stack_rc, stack_report, _ = run_json(stack_cmd, root, allow_failure=True)

    dhat_report: dict[str, Any] | None = None
    dhat_rc: int | None = None
    if not args.skip_dhat:
        dhat_runtime_out = out.with_name(f"{out.stem}.runtime-perf-dhat.json")
        dhat_out = out.with_name(f"{out.stem}.dhat.json")
        dhat_cmd = [
            *script_cmd("profile_runtime.py", root),
            *runtime_common_args(args),
            "--dhat",
            f"--dhat-frames={args.dhat_frames}",
            f"--out={dhat_runtime_out}",
            f"--dhat-out={dhat_out}",
        ]
        print("Running DHAT heap lane", file=sys.stderr)
        dhat_rc, dhat_report, _ = run_json(dhat_cmd, root)

    payload: dict[str, Any] = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scenarios": scenarios,
        "profile": args.profile,
        "release": args.release,
        "normal_runtime": {
            "returncode": normal_rc,
            "out": str(normal_out),
            "report": normal_report,
        },
        "stack_sensitivity": {
            "returncode": stack_rc,
            "out": str(stack_out),
            "first_success_stack_bytes": stack_report.get("first_success_stack_bytes", {}),
            "samples": stack_report.get("samples", []),
        },
        "dhat": None,
        "comparison": None,
    }
    if dhat_report is not None:
        payload["dhat"] = {
            "returncode": dhat_rc,
            "runtime_perf_out": dhat_report.get("out"),
            "dhat_out": dhat_report.get("dhat_out"),
            "report": dhat_report,
        }

    if args.baseline:
        baseline = json.loads(args.baseline.read_text())
        findings = compare_reports(baseline, payload, args)
        payload["comparison"] = {
            "baseline": str(args.baseline),
            "failed": bool(findings),
            "findings": findings,
            "thresholds": {
                "time_regression": args.time_regression,
                "alloc_regression": args.alloc_regression,
                "rss_regression": args.rss_regression,
                "stack_regression_bytes": args.stack_regression_bytes,
            },
        }

    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Runtime guard report: {out}", file=sys.stderr)
    print(json.dumps(payload, indent=2, sort_keys=True))

    if args.fail_on_regression and payload.get("comparison", {}).get("failed"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
