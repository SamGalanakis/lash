#!/usr/bin/env python3
"""Summarize a Harbor Terminal Bench job directory."""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


def parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def format_duration(started_at: str | None, finished_at: str | None) -> str:
    start = parse_ts(started_at)
    end = parse_ts(finished_at)
    if not start or not end:
        return "-"
    total = int((end - start).total_seconds())
    minutes, seconds = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02}m{seconds:02}s"
    return f"{minutes}m{seconds:02}s"


def trial_rows(job_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for result_path in sorted(job_dir.glob("*__*/result.json")):
        data = json.loads(result_path.read_text())
        verifier = data.get("verifier_result") or {}
        rewards = verifier.get("rewards") or {}
        exception = data.get("exception_info") or {}
        agent_result = data.get("agent_result") or {}

        reward = rewards.get("reward")
        if reward is None:
            reward = "-"

        if exception:
            status = f"error:{exception.get('exception_type', 'unknown')}"
        elif reward == 1 or reward == 1.0:
            status = "pass"
        elif reward == "-":
            status = "no-reward"
        else:
            status = "fail"

        tokens = []
        for key, label in (
            ("n_input_tokens", "in"),
            ("n_output_tokens", "out"),
            ("n_cache_tokens", "cache"),
        ):
            value = agent_result.get(key)
            if value is not None:
                tokens.append(f"{label}={value}")

        rows.append(
            {
                "task": data.get("task_name", result_path.parent.name),
                "trial": data.get("trial_name", result_path.parent.name),
                "status": status,
                "reward": str(reward),
                "duration": format_duration(data.get("started_at"), data.get("finished_at")),
                "tokens": " ".join(tokens) or "-",
                "path": str(result_path.parent),
            }
        )
    return rows


def print_table(rows: list[dict[str, str]]) -> None:
    headers = ("task", "status", "reward", "duration", "tokens")
    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(row[header]))

    line = "  ".join(header.ljust(widths[header]) for header in headers)
    print(line)
    print("  ".join("-" * widths[header] for header in headers))
    for row in rows:
        print("  ".join(row[header].ljust(widths[header]) for header in headers))


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: summarize_terminalbench.py <job-dir>", file=sys.stderr)
        return 2

    job_dir = Path(sys.argv[1])
    if not job_dir.exists():
        print(f"error: job dir not found: {job_dir}", file=sys.stderr)
        return 1

    job_result_path = job_dir / "result.json"
    if job_result_path.exists():
        job_result = json.loads(job_result_path.read_text())
    else:
        job_result = {}

    rows = trial_rows(job_dir)
    if not rows:
        print(f"No trial results found under {job_dir}")
        return 1

    print()
    print(f"Summary: {job_dir}")
    if job_result:
        stats = job_result.get("stats") or {}
        print(
            "Overall: "
            f"trials={stats.get('n_trials', '-')} "
            f"errors={stats.get('n_errors', '-')} "
            f"started={job_result.get('started_at', '-')} "
            f"finished={job_result.get('finished_at', '-')}"
        )

    print_table(rows)

    counts = Counter(row["status"] for row in rows)
    print()
    print(
        "Status counts: "
        + ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
    )
    print("Trial dirs:")
    for row in rows:
        print(f"- {row['trial']}: {row['path']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
