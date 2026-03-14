#!/usr/bin/env python3
"""Export a Harbor Terminal Bench job into a structured persistent run directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from terminalbench_results import ExportArgs, export_run, format_duration, load_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_dir", type=Path)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--execution-mode", required=True)
    parser.add_argument("--requested-model")
    parser.add_argument("--variant")
    parser.add_argument("--harbor-env", required=True)
    parser.add_argument("--registry-url", required=True)
    parser.add_argument("--n-concurrent", type=int, required=True)
    parser.add_argument("--attempts", type=int, required=True)
    parser.add_argument("--timeout-multiplier", type=float, required=True)
    parser.add_argument("--binary-path", required=True)
    parser.add_argument("--provider-config", type=Path)
    parser.add_argument("--delete-after-run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--task-pattern", action="append", default=[])
    parser.add_argument("--exact-task", action="append", default=[])
    parser.add_argument("--exclude-pattern", action="append", default=[])
    parser.add_argument("--extra-arg", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    ns = parse_args()
    run_dir = export_run(
        ExportArgs(
            job_dir=ns.job_dir,
            results_dir=ns.results_dir,
            dataset=ns.dataset,
            execution_mode=ns.execution_mode,
            requested_model=ns.requested_model,
            variant=ns.variant,
            harbor_env=ns.harbor_env,
            registry_url=ns.registry_url,
            n_concurrent=ns.n_concurrent,
            attempts=ns.attempts,
            timeout_multiplier=ns.timeout_multiplier,
            delete_after_run=ns.delete_after_run,
            debug=ns.debug,
            binary_path=ns.binary_path,
            task_patterns=ns.task_pattern,
            exact_tasks=ns.exact_task,
            exclude_patterns=ns.exclude_pattern,
            extra_args=ns.extra_arg,
            provider_config=ns.provider_config,
        )
    )
    run = load_run(run_dir)
    stats = run.get("global_stats") or {}
    timing = run.get("timing") or {}
    print()
    print(f"Exported benchmark run: {run_dir}")
    print(
        "Run stats: "
        f"passed={stats.get('trials_passed', 0)}/{stats.get('trials_total', 0)} "
        f"pass_rate={(stats.get('pass_rate', 0.0) * 100):.1f}% "
        f"avg_duration={format_duration(stats.get('duration_seconds_avg'))} "
        f"tokens_total={(stats.get('tokens_total') or {}).get('total', 0)} "
        f"wall={format_duration(timing.get('duration_seconds'))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
