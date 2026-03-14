#!/usr/bin/env python3
"""Structured export helpers for Harbor Terminal Bench runs."""

from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def duration_seconds(started_at: str | None, finished_at: str | None) -> float | None:
    start = parse_ts(started_at)
    finish = parse_ts(finished_at)
    if not start or not finish:
        return None
    return max((finish - start).total_seconds(), 0.0)


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    total = int(round(seconds))
    minutes, seconds = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02}m{seconds:02}s"
    return f"{minutes}m{seconds:02}s"


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip().lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or "run"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def read_text(path: Path) -> str:
    return path.read_text(errors="replace") if path.exists() else ""


def shorten(text: str, limit: int = 2400, from_end: bool = False) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None
    if len(stripped) <= limit:
        return stripped
    if from_end:
        return "..." + stripped[-limit:]
    return stripped[:limit] + "..."


def first_meaningful_line(text: str) -> str | None:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return None


def summarize_failure(
    status: str,
    exception_info: dict[str, Any] | None,
    verifier_stdout: str,
    command_stdout: str,
    command_stderr: str,
    reward: float | None,
) -> str | None:
    if exception_info:
        exc_type = exception_info.get("exception_type") or "error"
        message = (
            exception_info.get("exception_message")
            or exception_info.get("message")
            or exception_info.get("detail")
        )
        return f"{exc_type}: {message}" if message else exc_type

    if status == "fail":
        for line in verifier_stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("E       "):
                return stripped[8:].strip()
            if stripped.startswith("FAILED ") or "AssertionError" in stripped:
                return stripped
        if reward is not None:
            return f"Verifier reward was {reward:.1f}"
        return "Verifier failed"

    stderr_line = first_meaningful_line(command_stderr)
    if stderr_line:
        return stderr_line

    stdout_line = first_meaningful_line(command_stdout)
    if stdout_line:
        return stdout_line

    return None


def numeric_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def resolve_run_id(job_dir: Path, started_at: str | None) -> str:
    stamp = parse_ts(started_at)
    prefix = (
        stamp.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if stamp
        else datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    return f"{prefix}__{slugify(job_dir.name)}"


def copy_artifact(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def safe_relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def load_provider_metadata(config_path: Path | None) -> dict[str, Any]:
    if not config_path or not config_path.exists():
        return {"active_provider": None, "active_provider_type": None, "available_providers": []}

    data = load_json(config_path)
    active_key = data.get("active_provider")
    providers = data.get("providers") or {}
    active = providers.get(active_key) or {}
    available = []
    for name, provider in providers.items():
        available.append(
            {
                "name": name,
                "type": provider.get("type"),
            }
        )

    return {
        "active_provider": active_key,
        "active_provider_type": active.get("type"),
        "available_providers": available,
    }


def load_llm_metadata(llm_path: Path) -> dict[str, Any]:
    models: list[str] = []
    request_count = 0
    for line in llm_path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        request_count += 1
        raw_request = record.get("request")
        if not isinstance(raw_request, str):
            continue
        try:
            request = json.loads(raw_request)
        except json.JSONDecodeError:
            continue
        model = request.get("model")
        if isinstance(model, str) and model not in models:
            models.append(model)
    return {
        "request_count": request_count,
        "models": models,
    }


@dataclass
class ExportArgs:
    job_dir: Path
    results_dir: Path
    dataset: str
    execution_mode: str
    requested_model: str | None
    variant: str | None
    harbor_env: str
    registry_url: str
    n_concurrent: int
    attempts: int
    timeout_multiplier: float
    delete_after_run: bool
    debug: bool
    binary_path: str
    task_patterns: list[str]
    exact_tasks: list[str]
    exclude_patterns: list[str]
    extra_args: list[str]
    provider_config: Path | None


def build_trial_record(
    trial_dir: Path,
    run_dir: Path,
    args: ExportArgs,
) -> dict[str, Any]:
    result = load_json(trial_dir / "result.json")
    config = load_json(trial_dir / "config.json")
    agent_result = result.get("agent_result") or {}
    agent_metadata = (
        agent_result.get("metadata") if isinstance(agent_result.get("metadata"), dict) else {}
    )
    verifier_result = result.get("verifier_result") or {}
    exception_info = result.get("exception_info") or None
    reward = (verifier_result.get("rewards") or {}).get("reward")
    if exception_info:
        status = "error"
    elif reward == 1 or reward == 1.0:
        status = "pass"
    elif reward is None:
        status = "no-reward"
    else:
        status = "fail"

    tokens = {
        "input": int(agent_result.get("n_input_tokens") or 0),
        "output": int(agent_result.get("n_output_tokens") or 0),
        "cache": int(agent_result.get("n_cache_tokens") or 0),
        "total": int(agent_result.get("n_input_tokens") or 0)
        + int(agent_result.get("n_output_tokens") or 0)
        + int(agent_result.get("n_cache_tokens") or 0),
    }

    timing = {
        "started_at": result.get("started_at"),
        "finished_at": result.get("finished_at"),
        "trial_seconds": duration_seconds(result.get("started_at"), result.get("finished_at")),
        "environment_setup_seconds": duration_seconds(
            (result.get("environment_setup") or {}).get("started_at"),
            (result.get("environment_setup") or {}).get("finished_at"),
        ),
        "agent_setup_seconds": duration_seconds(
            (result.get("agent_setup") or {}).get("started_at"),
            (result.get("agent_setup") or {}).get("finished_at"),
        ),
        "agent_execution_seconds": duration_seconds(
            (result.get("agent_execution") or {}).get("started_at"),
            (result.get("agent_execution") or {}).get("finished_at"),
        ),
        "verifier_seconds": duration_seconds(
            (result.get("verifier") or {}).get("started_at"),
            (result.get("verifier") or {}).get("finished_at"),
        ),
    }

    snapshot_trial_dir = run_dir / "trials" / result.get("trial_name", trial_dir.name)
    artifacts_dir = snapshot_trial_dir / "artifacts"
    copied_artifacts: dict[str, str] = {}
    copied_files = {
        "result_json": trial_dir / "result.json",
        "config_json": trial_dir / "config.json",
        "trial_log": trial_dir / "trial.log",
        "exception_txt": trial_dir / "exception.txt",
        "verifier_reward": trial_dir / "verifier" / "reward.txt",
        "verifier_stdout": trial_dir / "verifier" / "test-stdout.txt",
        "lash_log": trial_dir / "agent" / "lash-home" / "lash.log",
    }
    for key, src in copied_files.items():
        if copy_artifact(src, artifacts_dir / f"{key}{src.suffix or '.txt'}"):
            copied_artifacts[key] = safe_relative(artifacts_dir / f"{key}{src.suffix or '.txt'}", run_dir)

    command_records: list[dict[str, Any]] = []
    for command_dir in sorted((trial_dir / "agent").glob("command-*")):
        command_idx = command_dir.name
        record: dict[str, Any] = {"name": command_idx}
        for label, filename in (
            ("command", "command.txt"),
            ("stdout", "stdout.txt"),
            ("stderr", "stderr.txt"),
            ("return_code", "return-code.txt"),
        ):
            src = command_dir / filename
            if not src.exists():
                continue
            dst = artifacts_dir / command_idx / filename
            copy_artifact(src, dst)
            record[label] = safe_relative(dst, run_dir)
        command_records.append(record)

    llm_candidates = sorted((trial_dir / "agent" / "lash-home" / "sessions").glob("*.llm.jsonl"))
    llm_path = llm_candidates[0] if llm_candidates else None
    llm_metadata = load_llm_metadata(llm_path) if llm_path else {"request_count": 0, "models": []}

    verifier_stdout = read_text(trial_dir / "verifier" / "test-stdout.txt")
    command_stdout = ""
    command_stderr = ""
    if command_records:
        first = command_records[0]
        if "stdout" in first:
            command_stdout = read_text(run_dir / first["stdout"])
        if "stderr" in first:
            command_stderr = read_text(run_dir / first["stderr"])
    if not command_stdout:
        command_stdout = str(agent_metadata.get("assistant_response") or "")

    failure_reason = summarize_failure(
        status=status,
        exception_info=exception_info,
        verifier_stdout=verifier_stdout,
        command_stdout=command_stdout,
        command_stderr=command_stderr,
        reward=reward if isinstance(reward, (float, int)) else None,
    )

    return {
        "trial_name": result.get("trial_name", trial_dir.name),
        "task_name": result.get("task_name", trial_dir.name),
        "task_source": result.get("source"),
        "status": status,
        "reward": float(reward) if isinstance(reward, (float, int)) else None,
        "timing": timing,
        "duration_display": format_duration(timing["trial_seconds"]),
        "tokens": tokens,
        "cost_usd": agent_result.get("cost_usd"),
        "metadata": {
            "execution_mode": args.execution_mode,
            "requested_model": args.requested_model,
            "resolved_models": llm_metadata["models"],
            "variant": args.variant or None,
            "provider": load_provider_metadata(args.provider_config),
            "task_path": ((result.get("task_id") or {}).get("path")),
            "task_git_url": ((result.get("task_id") or {}).get("git_url")),
            "task_git_commit_id": ((result.get("task_id") or {}).get("git_commit_id")),
            "llm_request_count": llm_metadata["request_count"],
            "assistant_response_present": bool(agent_metadata.get("assistant_response")),
        },
        "failure_reason": failure_reason,
        "logs": {
            "assistant_excerpt": shorten(command_stdout, from_end=True),
            "verifier_excerpt": shorten(verifier_stdout),
            "stderr_excerpt": shorten(command_stderr, from_end=True),
        },
        "artifacts": {
            "files": copied_artifacts,
            "commands": command_records,
        },
    }


def build_global_stats(trials: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = defaultdict(int)
    rewards: list[float] = []
    trial_seconds: list[float] = []
    agent_exec_seconds: list[float] = []
    total_tokens = {"input": 0, "output": 0, "cache": 0, "total": 0}

    for trial in trials:
        statuses[trial["status"]] += 1
        reward = trial.get("reward")
        if isinstance(reward, (float, int)):
            rewards.append(float(reward))
        duration = (trial.get("timing") or {}).get("trial_seconds")
        if isinstance(duration, (float, int)):
            trial_seconds.append(float(duration))
        agent_duration = (trial.get("timing") or {}).get("agent_execution_seconds")
        if isinstance(agent_duration, (float, int)):
            agent_exec_seconds.append(float(agent_duration))
        for key in total_tokens:
            total_tokens[key] += int((trial.get("tokens") or {}).get(key) or 0)

    trial_count = len(trials)
    passed = statuses["pass"]
    return {
        "trials_total": trial_count,
        "trials_passed": passed,
        "trials_failed": statuses["fail"],
        "trials_errors": statuses["error"],
        "trials_without_reward": statuses["no-reward"],
        "pass_rate": (passed / trial_count) if trial_count else 0.0,
        "reward_mean": numeric_mean(rewards),
        "duration_seconds_sum": sum(trial_seconds),
        "duration_seconds_avg": numeric_mean(trial_seconds),
        "duration_seconds_min": min(trial_seconds) if trial_seconds else None,
        "duration_seconds_max": max(trial_seconds) if trial_seconds else None,
        "agent_execution_seconds_avg": numeric_mean(agent_exec_seconds),
        "tokens_total": total_tokens,
        "tokens_avg": {
            key: (value / trial_count if trial_count else 0.0)
            for key, value in total_tokens.items()
        },
        "status_counts": dict(sorted(statuses.items())),
    }


def build_task_rollups(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trial in trials:
        grouped[trial["task_name"]].append(trial)

    rollups: list[dict[str, Any]] = []
    for task_name, task_trials in sorted(grouped.items()):
        stats = build_global_stats(task_trials)
        rollups.append(
            {
                "task_name": task_name,
                "attempts": len(task_trials),
                "pass_rate": stats["pass_rate"],
                "status_counts": stats["status_counts"],
                "reward_mean": stats["reward_mean"],
                "duration_seconds_avg": stats["duration_seconds_avg"],
                "duration_seconds_sum": stats["duration_seconds_sum"],
                "tokens_total": stats["tokens_total"],
                "tokens_avg": stats["tokens_avg"],
                "trial_names": [trial["trial_name"] for trial in task_trials],
            }
        )
    return rollups


def export_run(args: ExportArgs) -> Path:
    job_result = load_json(args.job_dir / "result.json")
    run_id = resolve_run_id(args.job_dir, job_result.get("started_at"))
    run_dir = args.results_dir / "runs" / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    job_artifacts_dir = run_dir / "job-artifacts"
    for src in (args.job_dir / "config.json", args.job_dir / "result.json", args.job_dir / "job.log"):
        if src.exists():
            copy_artifact(src, job_artifacts_dir / src.name)

    trials = []
    for trial_result in sorted(args.job_dir.glob("*__*/result.json")):
        trials.append(build_trial_record(trial_result.parent, run_dir, args))

    run_payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "exported_at": iso_utc_now(),
        "job_name": args.job_dir.name,
        "source_job_dir": str(args.job_dir.resolve()),
        "params": {
            "dataset": args.dataset,
            "execution_mode": args.execution_mode,
            "requested_model": args.requested_model,
            "variant": args.variant or None,
            "provider": load_provider_metadata(args.provider_config),
            "harbor_env": args.harbor_env,
            "registry_url": args.registry_url,
            "n_concurrent": args.n_concurrent,
            "attempts": args.attempts,
            "timeout_multiplier": args.timeout_multiplier,
            "delete_after_run": args.delete_after_run,
            "debug": args.debug,
            "binary_path": args.binary_path,
            "task_patterns": args.task_patterns,
            "exact_tasks": args.exact_tasks,
            "exclude_patterns": args.exclude_patterns,
            "extra_args": args.extra_args,
        },
        "timing": {
            "started_at": job_result.get("started_at"),
            "finished_at": job_result.get("finished_at"),
            "duration_seconds": duration_seconds(
                job_result.get("started_at"),
                job_result.get("finished_at"),
            ),
        },
        "global_stats": build_global_stats(trials),
        "task_rollups": build_task_rollups(trials),
        "trials": trials,
    }

    (run_dir / "run.json").write_text(json.dumps(run_payload, indent=2) + "\n")
    return run_dir


def load_run(run_dir: Path) -> dict[str, Any]:
    return load_json(run_dir / "run.json")


def load_run_summaries(results_dir: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for run_json in sorted((results_dir / "runs").glob("*/run.json"), reverse=True):
        run = load_json(run_json)
        if not run:
            continue
        stats = run.get("global_stats") or {}
        timing = run.get("timing") or {}
        params = run.get("params") or {}
        runs.append(
            {
                "run_id": run.get("run_id"),
                "job_name": run.get("job_name"),
                "started_at": timing.get("started_at"),
                "finished_at": timing.get("finished_at"),
                "duration_seconds": timing.get("duration_seconds"),
                "dataset": params.get("dataset"),
                "execution_mode": params.get("execution_mode"),
                "requested_model": params.get("requested_model"),
                "variant": params.get("variant"),
                "provider": (params.get("provider") or {}).get("active_provider"),
                "trials_total": stats.get("trials_total", 0),
                "trials_passed": stats.get("trials_passed", 0),
                "trials_failed": stats.get("trials_failed", 0),
                "trials_errors": stats.get("trials_errors", 0),
                "pass_rate": stats.get("pass_rate", 0.0),
                "tokens_total": (stats.get("tokens_total") or {}).get("total", 0),
                "run_dir": str(run_json.parent.resolve()),
            }
        )
    runs.sort(key=lambda item: item.get("started_at") or "", reverse=True)
    return runs


def delete_run(results_dir: Path, run_id: str) -> bool:
    run_dir = results_dir / "runs" / run_id
    if not run_dir.exists():
        return False
    run = load_run(run_dir)
    source_job_dir_raw = run.get("source_job_dir")
    job_name = run.get("job_name")
    if isinstance(source_job_dir_raw, str) and source_job_dir_raw:
        source_job_dir = Path(source_job_dir_raw).resolve()
        source_result = source_job_dir / "result.json"
        source_config = source_job_dir / "config.json"
        # Only remove directories that still look like Harbor job outputs for the
        # same recorded job, so a malformed run.json cannot point deletion at an
        # arbitrary unrelated directory.
        if (
            source_job_dir.exists()
            and source_job_dir.is_dir()
            and (source_result.exists() or source_config.exists())
            and (job_name is None or source_job_dir.name == job_name)
            and source_job_dir != run_dir
        ):
            shutil.rmtree(source_job_dir)
    shutil.rmtree(run_dir)
    return True
