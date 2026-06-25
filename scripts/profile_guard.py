#!/usr/bin/env python3
"""Run the canonical Lash performance guard suite."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import profile_runtime_stack

DEFAULT_STACK_SCENARIOS = profile_runtime_stack.DEFAULT_SCENARIOS
QUICK_STACKS = ["2m"]
FULL_STACKS = ["64k", "128k", "256k", "320k", "512k", "1m", "2m", "8m"]
STACK_BUDGET_BYTES = 2 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", action="store_true", help="Run release binaries.")
    parser.add_argument(
        "--profile",
        choices=["quick", "full"],
        default="quick",
        help="Guard workload size preset.",
    )
    parser.add_argument("--runtime-runs", type=int, help="Measured runtime runs.")
    parser.add_argument("--runtime-warmups", type=int, help="Runtime warmups.")
    parser.add_argument("--runtime-turns", type=int, help="Turns per runtime run.")
    parser.add_argument(
        "--runtime-scenario",
        action="append",
        default=[],
        help="Runtime scenario filter. Defaults to runtime benchmark defaults.",
    )
    parser.add_argument(
        "--ui-scenario",
        action="append",
        default=[],
        help="UI scenario filter. Defaults to UI benchmark defaults.",
    )
    parser.add_argument("--ui-runs", type=int, help="Measured UI runs.")
    parser.add_argument("--ui-warmups", type=int, help="UI warmups.")
    parser.add_argument(
        "--lashlang-scenario",
        action="append",
        default=[],
        help="Lashlang scenario filter. Defaults to all benchmark scenarios.",
    )
    parser.add_argument("--lashlang-iterations", type=int, help="Lashlang perf iterations.")
    parser.add_argument(
        "--lashlang-profile-iterations",
        type=int,
        help="Lashlang profile iterations.",
    )
    parser.add_argument(
        "--cargo-feature",
        action="append",
        default=[],
        help="Additional Cargo feature to enable for all built benchmark binaries.",
    )
    parser.add_argument(
        "--cli-cargo-feature",
        action="append",
        default=[],
        help="Additional Cargo feature to enable only when building lash-cli/UI benchmarks.",
    )
    parser.add_argument(
        "--stack-scenario",
        action="append",
        default=[],
        help="Runtime stack-sensitivity scenario. Defaults to the guard stack set.",
    )
    parser.add_argument(
        "--stack-bytes",
        action="append",
        default=[],
        help="Worker stack size for the stack lane, e.g. 2m. Defaults by profile.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=180,
        help="Timeout for each stack matrix sample.",
    )
    parser.add_argument("--skip-dhat", action="store_true", help="Skip DHAT heap profiling.")
    parser.add_argument(
        "--dhat-frames",
        type=int,
        default=24,
        help="Trim DHAT backtraces to this many frames.",
    )
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="Exit non-zero when coverage or budgets fail.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Write the guard JSON here. Defaults under .benchmarks/perf-guard/.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_out(root: Path) -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / ".benchmarks" / "perf-guard" / f"perf-guard-{stamp}.json"


def script_cmd(script: str, root: Path) -> list[str]:
    return [sys.executable, str(root / "scripts" / script)]


def run_json(cmd: list[str], root: Path) -> tuple[int, dict[str, Any], str]:
    proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")
    payload: dict[str, Any] = {}
    if proc.stdout.strip():
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError:
            # Some wrappers echo the underlying JSON before exiting non-zero.
            start = proc.stdout.find("{")
            end = proc.stdout.rfind("}")
            if start >= 0 and end > start:
                try:
                    payload = json.loads(proc.stdout[start : end + 1])
                except json.JSONDecodeError:
                    payload = {}
    if proc.returncode != 0 and not payload:
        if proc.stdout:
            print(proc.stdout, end="")
        payload = {
            "error": "command did not emit parseable JSON",
            "stdout_tail": proc.stdout[-4000:],
        }
    return proc.returncode, payload, proc.stderr


def load_report_file(path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return fallback


def append_common_binary_args(cmd: list[str], args: argparse.Namespace) -> None:
    if args.release:
        cmd.append("--release")
    for feature in args.cargo_feature:
        cmd.extend(["--cargo-feature", feature])


def runtime_cmd(args: argparse.Namespace, root: Path, out: Path, *, dhat: bool = False) -> list[str]:
    cmd = [*script_cmd("profile_runtime.py", root), "--profile", args.profile, f"--out={out}"]
    append_common_binary_args(cmd, args)
    if args.runtime_runs is not None:
        cmd.append(f"--runs={args.runtime_runs}")
    if args.runtime_warmups is not None:
        cmd.append(f"--warmups={args.runtime_warmups}")
    if args.runtime_turns is not None:
        cmd.append(f"--turns={args.runtime_turns}")
    for scenario in args.runtime_scenario:
        cmd.extend(["--scenario", scenario])
    if args.enforce and not dhat:
        cmd.append("--enforce-budgets")
    if dhat:
        cmd.append("--dhat")
        cmd.append(f"--dhat-frames={max(args.dhat_frames, 1)}")
        cmd.append(f"--dhat-out={out.with_name(f'{out.stem}.dhat.json')}")
    return cmd


def stack_cmd(args: argparse.Namespace, root: Path, out: Path) -> list[str]:
    cmd = [
        *script_cmd("profile_runtime_stack.py", root),
        "--no-build",
        f"--out={out}",
        f"--runs={args.runtime_runs if args.runtime_runs is not None else 1}",
        f"--warmups={args.runtime_warmups if args.runtime_warmups is not None else 0}",
        f"--turns={args.runtime_turns if args.runtime_turns is not None else 1}",
        f"--timeout-seconds={args.timeout_seconds}",
    ]
    append_common_binary_args(cmd, args)
    for scenario in args.stack_scenario or DEFAULT_STACK_SCENARIOS:
        cmd.extend(["--scenario", scenario])
    for stack in args.stack_bytes or (QUICK_STACKS if args.profile == "quick" else FULL_STACKS):
        cmd.extend(["--stack-bytes", stack])
    return cmd


def ui_cmd(args: argparse.Namespace, root: Path, out: Path) -> list[str]:
    cmd = [*script_cmd("profile_ui.py", root), "--profile", args.profile, f"--out={out}"]
    append_common_binary_args(cmd, args)
    for feature in args.cli_cargo_feature:
        cmd.extend(["--cargo-feature", feature])
    if args.ui_runs is not None:
        cmd.append(f"--runs={args.ui_runs}")
    if args.ui_warmups is not None:
        cmd.append(f"--warmups={args.ui_warmups}")
    for scenario in args.ui_scenario:
        cmd.extend(["--scenario", scenario])
    if args.enforce:
        cmd.append("--enforce-budgets")
    return cmd


def lashlang_cmd(args: argparse.Namespace, root: Path, out: Path) -> list[str]:
    iterations = args.lashlang_iterations
    profile_iterations = args.lashlang_profile_iterations
    if iterations is None:
        iterations = 500 if args.profile == "quick" else 2500
    if profile_iterations is None:
        profile_iterations = 500 if args.profile == "quick" else 2500
    cmd = [
        *script_cmd("profile_lashlang.py", root),
        f"--iterations={iterations}",
        f"--profile-iterations={profile_iterations}",
        f"--out={out}",
    ]
    for scenario in args.lashlang_scenario:
        cmd.extend(["--scenario", scenario])
    if args.enforce:
        cmd.append("--enforce-budgets")
    return cmd


def runtime_summary_names(report: dict[str, Any]) -> set[str]:
    return {
        item["scenario"]
        for item in report.get("summary", [])
        if isinstance(item, dict) and isinstance(item.get("scenario"), str)
    }


def ui_summary_names(report: dict[str, Any]) -> set[str]:
    return {
        item["scenario"]
        for item in report.get("scenarios", [])
        if isinstance(item, dict) and isinstance(item.get("scenario"), str)
    }


def failed_budget_results(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in report.get("budget_results", [])
        if isinstance(item, dict) and not item.get("passed")
    ]


def stack_profile_is_measured(stack_profile: Any) -> bool:
    if not isinstance(stack_profile, dict):
        return False
    return any(
        isinstance(stack_profile.get(key), int)
        for key in (
            "measured_stack_bytes",
            "worker_stack_bytes",
            "rust_min_stack_bytes",
            "process_stack_soft_limit_bytes",
        )
    )


def require_measured_stack_profile(
    findings: list[dict[str, Any]],
    *,
    stack_profile: Any,
    missing_kind: str,
    unmeasured_kind: str,
    missing_budget_kind: str,
    budget_failed_kind: str,
    section: str,
    context: dict[str, Any] | None = None,
    enforce_budget: bool = True,
) -> None:
    context = context or {}
    if not isinstance(stack_profile, dict):
        findings.append({"kind": missing_kind, "section": section, **context})
        return
    if not stack_profile_is_measured(stack_profile):
        findings.append({"kind": unmeasured_kind, "section": section, **context})
    if not isinstance(stack_profile.get("stack_budget_bytes"), int):
        findings.append({"kind": missing_budget_kind, "section": section, **context})
    if enforce_budget and stack_profile.get("within_stack_budget") is False:
        findings.append(
            {
                "kind": budget_failed_kind,
                "section": section,
                **context,
                "actual": stack_profile.get("measured_stack_bytes"),
                "budget": stack_profile.get("stack_budget_bytes"),
            }
        )


def runtime_report_stack_findings(report: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    worker_stack_bytes = report.get("worker_stack_bytes")
    stack_profile = report.get("stack_profile")
    if not isinstance(stack_profile, dict):
        findings.append(
            {
                "kind": "runtime_stack_profile_missing",
                "section": "runtime",
            }
        )
    elif stack_profile.get("worker_stack_bytes") != worker_stack_bytes:
        findings.append(
            {
                "kind": "runtime_stack_profile_mismatch",
                "section": "runtime",
                "worker_stack_bytes": worker_stack_bytes,
                "profile_worker_stack_bytes": stack_profile.get("worker_stack_bytes"),
            }
        )
    elif not isinstance(stack_profile.get("stack_budget_bytes"), int):
        findings.append({"kind": "runtime_stack_budget_missing", "section": "runtime"})
    elif stack_profile.get("within_stack_budget") is False:
        findings.append(
            {
                "kind": "runtime_stack_budget_failed",
                "section": "runtime",
                "actual": stack_profile.get("measured_stack_bytes"),
                "budget": stack_profile.get("stack_budget_bytes"),
            }
        )

    for summary in report.get("summary", []):
        if not isinstance(summary, dict):
            continue
        summary_stack_profile = summary.get("stack_profile")
        if not isinstance(summary_stack_profile, dict):
            findings.append(
                {
                    "kind": "runtime_summary_stack_profile_missing",
                    "section": "runtime",
                    "scenario": summary.get("scenario"),
                }
            )
        elif summary_stack_profile.get("worker_stack_bytes") != worker_stack_bytes:
            findings.append(
                {
                    "kind": "runtime_summary_stack_profile_mismatch",
                    "section": "runtime",
                    "scenario": summary.get("scenario"),
                    "worker_stack_bytes": worker_stack_bytes,
                    "summary_worker_stack_bytes": summary_stack_profile.get(
                        "worker_stack_bytes"
                    ),
                }
            )
        elif not isinstance(summary_stack_profile.get("stack_budget_bytes"), int):
            findings.append(
                {
                    "kind": "runtime_summary_stack_budget_missing",
                    "section": "runtime",
                    "scenario": summary.get("scenario"),
                }
            )
        elif summary_stack_profile.get("within_stack_budget") is False:
            findings.append(
                {
                    "kind": "runtime_summary_stack_budget_failed",
                    "section": "runtime",
                    "scenario": summary.get("scenario"),
                    "actual": summary_stack_profile.get("measured_stack_bytes"),
                    "budget": summary_stack_profile.get("stack_budget_bytes"),
                }
            )
    for result in report.get("results", []):
        if not isinstance(result, dict):
            continue
        result_stack_profile = result.get("stack_profile")
        if not isinstance(result_stack_profile, dict):
            findings.append(
                {
                    "kind": "runtime_result_stack_profile_missing",
                    "section": "runtime",
                    "scenario": result.get("scenario"),
                }
            )
        elif result_stack_profile.get("worker_stack_bytes") != worker_stack_bytes:
            findings.append(
                {
                    "kind": "runtime_result_stack_profile_mismatch",
                    "section": "runtime",
                    "scenario": result.get("scenario"),
                    "worker_stack_bytes": worker_stack_bytes,
                    "result_worker_stack_bytes": result_stack_profile.get("worker_stack_bytes"),
                }
            )
        elif not isinstance(result_stack_profile.get("stack_budget_bytes"), int):
            findings.append(
                {
                    "kind": "runtime_result_stack_budget_missing",
                    "section": "runtime",
                    "scenario": result.get("scenario"),
                }
            )
        elif result_stack_profile.get("within_stack_budget") is False:
            findings.append(
                {
                    "kind": "runtime_result_stack_budget_failed",
                    "section": "runtime",
                    "scenario": result.get("scenario"),
                    "actual": result_stack_profile.get("measured_stack_bytes"),
                    "budget": result_stack_profile.get("stack_budget_bytes"),
                }
            )
    return findings


def ui_report_stack_findings(report: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    require_measured_stack_profile(
        findings,
        stack_profile=report.get("stack_profile"),
        missing_kind="ui_stack_profile_missing",
        unmeasured_kind="ui_stack_profile_unmeasured",
        missing_budget_kind="ui_stack_budget_missing",
        budget_failed_kind="ui_stack_budget_failed",
        section="ui",
        enforce_budget=False,
    )
    parameters = report.get("parameters")
    parameter_stack_profile = (
        parameters.get("stack_profile") if isinstance(parameters, dict) else None
    )
    require_measured_stack_profile(
        findings,
        stack_profile=parameter_stack_profile,
        missing_kind="ui_parameter_stack_profile_missing",
        unmeasured_kind="ui_parameter_stack_profile_unmeasured",
        missing_budget_kind="ui_parameter_stack_budget_missing",
        budget_failed_kind="ui_parameter_stack_budget_failed",
        section="ui",
        enforce_budget=False,
    )
    for scenario in report.get("scenarios", []):
        if not isinstance(scenario, dict):
            continue
        context = {"scenario": scenario.get("scenario")}
        require_measured_stack_profile(
            findings,
            stack_profile=scenario.get("stack_profile"),
            missing_kind="ui_scenario_stack_profile_missing",
            unmeasured_kind="ui_scenario_stack_profile_unmeasured",
            missing_budget_kind="ui_scenario_stack_budget_missing",
            budget_failed_kind="ui_scenario_stack_budget_failed",
            section="ui",
            context=context,
            enforce_budget=False,
        )
        for result in scenario.get("results", []):
            if not isinstance(result, dict):
                continue
            require_measured_stack_profile(
                findings,
                stack_profile=result.get("stack_profile"),
                missing_kind="ui_result_stack_profile_missing",
                unmeasured_kind="ui_result_stack_profile_unmeasured",
                missing_budget_kind="ui_result_stack_budget_missing",
                budget_failed_kind="ui_result_stack_budget_failed",
                section="ui",
                context=context,
                enforce_budget=False,
            )
    return findings


def lashlang_report_stack_findings(report: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    require_measured_stack_profile(
        findings,
        stack_profile=report.get("stack_profile"),
        missing_kind="lashlang_stack_profile_missing",
        unmeasured_kind="lashlang_stack_profile_unmeasured",
        missing_budget_kind="lashlang_stack_budget_missing",
        budget_failed_kind="lashlang_stack_budget_failed",
        section="lashlang",
    )
    parameters = report.get("parameters")
    parameter_stack_profile = (
        parameters.get("stack_profile") if isinstance(parameters, dict) else None
    )
    require_measured_stack_profile(
        findings,
        stack_profile=parameter_stack_profile,
        missing_kind="lashlang_parameter_stack_profile_missing",
        unmeasured_kind="lashlang_parameter_stack_profile_unmeasured",
        missing_budget_kind="lashlang_parameter_stack_budget_missing",
        budget_failed_kind="lashlang_parameter_stack_budget_failed",
        section="lashlang",
    )
    for result in report.get("perf_results", []):
        if not isinstance(result, dict):
            continue
        require_measured_stack_profile(
            findings,
            stack_profile=result.get("stack_profile"),
            missing_kind="lashlang_perf_result_stack_profile_missing",
            unmeasured_kind="lashlang_perf_result_stack_profile_unmeasured",
            missing_budget_kind="lashlang_perf_result_stack_budget_missing",
            budget_failed_kind="lashlang_perf_result_stack_budget_failed",
            section="lashlang",
            context={
                "scenario": result.get("scenario_arg"),
                "mode": result.get("mode_arg"),
            },
        )
    for result in report.get("profile_results", []):
        if not isinstance(result, dict):
            continue
        require_measured_stack_profile(
            findings,
            stack_profile=result.get("stack_profile"),
            missing_kind="lashlang_profile_result_stack_profile_missing",
            unmeasured_kind="lashlang_profile_result_stack_profile_unmeasured",
            missing_budget_kind="lashlang_profile_result_stack_budget_missing",
            budget_failed_kind="lashlang_profile_result_stack_budget_failed",
            section="lashlang",
            context={"scenario": result.get("scenario_arg")},
        )
    return findings


def evaluate_guard_coverage(payload: dict[str, Any]) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    required_sections = ["runtime", "runtime_stack", "ui", "lashlang"]
    for section in required_sections:
        value = payload.get(section)
        if not isinstance(value, dict):
            findings.append(
                {"kind": "missing_section", "section": section, "message": "section missing"}
            )
            continue
        if value.get("returncode") != 0:
            findings.append(
                {
                    "kind": "lane_failed",
                    "section": section,
                    "returncode": value.get("returncode"),
                }
            )

    runtime = payload.get("runtime", {}).get("report", {})
    expected_runtime = set(payload.get("runtime", {}).get("expected_scenarios", []))
    if expected_runtime:
        missing_runtime = sorted(expected_runtime - runtime_summary_names(runtime))
        for scenario in missing_runtime:
            findings.append(
                {
                    "kind": "missing_runtime_scenario",
                    "section": "runtime",
                    "scenario": scenario,
                }
            )
    for budget in failed_budget_results(runtime):
        findings.append({"kind": "runtime_budget_failed", "section": "runtime", "budget": budget})
    findings.extend(runtime_report_stack_findings(runtime))

    ui = payload.get("ui", {}).get("report", {})
    if not ui_summary_names(ui):
        findings.append({"kind": "missing_ui_scenarios", "section": "ui"})
    findings.extend(ui_report_stack_findings(ui))

    lashlang = payload.get("lashlang", {}).get("report", {})
    if not lashlang.get("perf_results"):
        findings.append({"kind": "missing_lashlang_perf_results", "section": "lashlang"})
    if not lashlang.get("profile_results"):
        findings.append({"kind": "missing_lashlang_profile_results", "section": "lashlang"})
    findings.extend(lashlang_report_stack_findings(lashlang))
    for budget in failed_budget_results(lashlang):
        findings.append(
            {"kind": "lashlang_budget_failed", "section": "lashlang", "budget": budget}
        )

    stack = payload.get("runtime_stack", {})
    first_success = stack.get("first_success_stack_bytes", {})
    expected_stack = set(stack.get("expected_scenarios", []))
    for scenario in sorted(expected_stack):
        value = first_success.get(scenario)
        if not isinstance(value, int):
            findings.append(
                {
                    "kind": "missing_stack_success",
                    "section": "runtime_stack",
                    "scenario": scenario,
                }
            )
        elif value > STACK_BUDGET_BYTES:
            findings.append(
                {
                    "kind": "stack_budget_failed",
                    "section": "runtime_stack",
                    "scenario": scenario,
                    "actual": value,
                    "budget": STACK_BUDGET_BYTES,
                }
            )
    for sample in stack.get("samples", []):
        if not isinstance(sample, dict):
            continue
        if sample.get("status") == "ok" and sample.get("stack_accounted") is not True:
            findings.append(
                {
                    "kind": "stack_size_not_accounted",
                    "section": "runtime_stack",
                    "scenario": sample.get("scenario"),
                    "stack_bytes": sample.get("stack_bytes"),
                    "reported_worker_stack_bytes": sample.get("reported_worker_stack_bytes"),
                }
            )
        summary_scenarios = sample.get("summary_scenarios")
        scenario = sample.get("scenario")
        if (
            sample.get("status") == "ok"
            and isinstance(scenario, str)
            and (not isinstance(summary_scenarios, list) or scenario not in summary_scenarios)
        ):
            findings.append(
                {
                    "kind": "stack_scenario_not_reported",
                    "section": "runtime_stack",
                    "scenario": scenario,
                    "stack_bytes": sample.get("stack_bytes"),
                }
            )

    return {
        "passed": not findings,
        "findings": findings,
        "thresholds": {"stack_budget_bytes": STACK_BUDGET_BYTES},
        "required_sections": required_sections,
    }


def main() -> int:
    args = parse_args()
    root = repo_root()
    out = args.out or default_out(root)
    out.parent.mkdir(parents=True, exist_ok=True)

    runtime_out = out.with_name(f"{out.stem}.runtime.json")
    stack_out = out.with_name(f"{out.stem}.runtime-stack.json")
    ui_out = out.with_name(f"{out.stem}.ui.json")
    lashlang_out = out.with_name(f"{out.stem}.lashlang.json")

    print("Running runtime perf lane", file=sys.stderr)
    runtime_rc, runtime_payload, _ = run_json(runtime_cmd(args, root, runtime_out), root)
    runtime_report = load_report_file(runtime_out, runtime_payload)

    print("Running runtime stack lane", file=sys.stderr)
    stack_rc, stack_payload, _ = run_json(stack_cmd(args, root, stack_out), root)
    stack_report = load_report_file(stack_out, stack_payload)

    print("Running UI perf lane", file=sys.stderr)
    ui_rc, ui_payload, _ = run_json(ui_cmd(args, root, ui_out), root)
    ui_report = load_report_file(ui_out, ui_payload)

    print("Running Lashlang perf lane", file=sys.stderr)
    lashlang_rc, lashlang_payload, _ = run_json(lashlang_cmd(args, root, lashlang_out), root)
    lashlang_report = load_report_file(lashlang_out, lashlang_payload)

    dhat_payload = None
    dhat_rc = None
    if not args.skip_dhat:
        dhat_runtime_out = out.with_name(f"{out.stem}.runtime-dhat.json")
        print("Running runtime DHAT lane", file=sys.stderr)
        dhat_rc, dhat_payload, _ = run_json(
            runtime_cmd(args, root, dhat_runtime_out, dhat=True),
            root,
        )
        dhat_payload = load_report_file(dhat_runtime_out, dhat_payload)

    payload: dict[str, Any] = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "kind": "perf-guard",
        "profile": args.profile,
        "release": args.release,
        "runtime": {
            "returncode": runtime_rc,
            "out": str(runtime_out),
            "expected_scenarios": [],
            "report": runtime_report,
        },
        "runtime_stack": {
            "returncode": stack_rc,
            "out": str(stack_out),
            "expected_scenarios": stack_report.get(
                "scenarios", args.stack_scenario or DEFAULT_STACK_SCENARIOS
            ),
            "first_success_stack_bytes": stack_report.get("first_success_stack_bytes", {}),
            "samples": stack_report.get("samples", []),
            "report": stack_report,
        },
        "ui": {
            "returncode": ui_rc,
            "out": str(ui_out),
            "report": ui_report,
        },
        "lashlang": {
            "returncode": lashlang_rc,
            "out": str(lashlang_out),
            "report": lashlang_report,
        },
        "coverage": None,
        "dhat": None,
    }
    runtime_scenarios = runtime_report.get("scenarios")
    if isinstance(runtime_scenarios, list):
        payload["runtime"]["expected_scenarios"] = runtime_scenarios
    else:
        payload["runtime"]["expected_scenarios"] = list(runtime_summary_names(runtime_report))

    if dhat_payload is not None:
        payload["dhat"] = {
            "returncode": dhat_rc,
            "runtime_perf_out": dhat_payload.get("out"),
            "dhat_out": dhat_payload.get("dhat_out"),
            "report": dhat_payload,
        }

    payload["coverage"] = evaluate_guard_coverage(payload)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Perf guard report: {out}", file=sys.stderr)
    print(
        json.dumps(
            {
                "out": str(out),
                "coverage_passed": payload["coverage"]["passed"],
                "findings": payload["coverage"]["findings"],
                "runtime_out": str(runtime_out),
                "runtime_stack_out": str(stack_out),
                "ui_out": str(ui_out),
                "lashlang_out": str(lashlang_out),
            },
            indent=2,
            sort_keys=True,
        )
    )

    if args.enforce and not payload["coverage"]["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
