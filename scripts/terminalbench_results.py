#!/usr/bin/env python3
"""Structured export helpers for Harbor Terminal Bench runs."""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 8

PRESET_TASKS: dict[str, tuple[str, ...]] = {
    "trivial": ("log-summary-date-ranges",),
    "smoke": (
        "log-summary-date-ranges",
        "fix-code-vulnerability",
    ),
    "fast-3": (
        "log-summary-date-ranges",
        "fix-code-vulnerability",
        "regex-log",
    ),
    "fast-medium": (
        "regex-log",
        "log-summary-date-ranges",
        "fix-code-vulnerability",
        "sqlite-with-gcov",
    ),
    "representative-10": (
        "build-cython-ext",
        "configure-git-webserver",
        "db-wal-recovery",
        "fix-code-vulnerability",
        "git-leak-recovery",
        "log-summary-date-ranges",
        "nginx-request-logging",
        "polyglot-c-py",
        "regex-log",
        "sqlite-with-gcov",
    ),
}


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


def normalize_task_names(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        stripped = value.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        normalized.append(stripped)
    return normalized


def infer_preset(exact_tasks: Any) -> str | None:
    normalized = sorted(normalize_task_names(exact_tasks))
    if not normalized:
        return None
    for name, tasks in PRESET_TASKS.items():
        if normalized == sorted(tasks):
            return name
    return None


def resolve_preset(
    preset: Any,
    exact_tasks: Any,
) -> tuple[str | None, str | None]:
    if isinstance(preset, str):
        stripped = preset.strip()
        if stripped:
            return stripped, "explicit"
    inferred = infer_preset(exact_tasks)
    if inferred:
        return inferred, "inferred"
    return None, None


def build_task_scope(
    exact_tasks: Any,
    task_patterns: Any,
    trials: list[dict[str, Any]],
) -> dict[str, Any]:
    requested = normalize_task_names(exact_tasks)
    executed = sorted(
        {
            task_name.strip()
            for trial in trials
            for task_name in [trial.get("task_name")]
            if isinstance(task_name, str) and task_name.strip()
        }
    )
    requested_set = set(requested)
    executed_set = set(executed)
    task_patterns_list = normalize_task_names(task_patterns)
    if requested:
        selection_mode = "exact"
    elif task_patterns_list:
        selection_mode = "pattern"
    else:
        selection_mode = "dataset"
    return {
        "selection_mode": selection_mode,
        "requested_tasks": requested,
        "requested_task_count": len(requested),
        "executed_tasks": executed,
        "executed_task_count": len(executed),
        "missing_requested_tasks": [task for task in requested if task not in executed_set],
        "unexpected_executed_tasks": [task for task in executed if task not in requested_set],
        "scope_mismatch": bool(requested) and requested_set != executed_set,
    }


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


def parse_time_duration(value: str | None) -> float | None:
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None

    parts = raw.split(":")
    try:
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        return float(raw)
    except ValueError:
        return None


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


def load_llm_metadata(llm_paths: list[Path]) -> dict[str, Any]:
    models: list[str] = []
    record_count = 0
    turns: set[int] = set()
    files: list[dict[str, Any]] = []
    token_totals = {
        "raw_input": 0,
        "output": 0,
        "cache": 0,
        "reasoning": 0,
    }
    usage_record_count = 0
    for llm_path in llm_paths:
        file_record_count = 0
        file_models: list[str] = []
        file_turns: set[int] = set()
        for line in llm_path.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_count += 1
            file_record_count += 1
            usage = record.get("usage")
            if isinstance(usage, dict):
                token_totals["raw_input"] += int(usage.get("input_tokens") or 0)
                token_totals["output"] += int(usage.get("output_tokens") or 0)
                token_totals["cache"] += int(usage.get("cached_input_tokens") or 0)
                token_totals["reasoning"] += int(usage.get("reasoning_tokens") or 0)
                usage_record_count += 1
            turn = record.get("turn")
            if isinstance(turn, int):
                turns.add(turn)
                file_turns.add(turn)
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
            if isinstance(model, str) and model not in file_models:
                file_models.append(model)
        files.append(
            {
                "name": llm_path.name,
                "record_count": file_record_count,
                "turn_count": len(file_turns),
                "models": file_models,
            }
        )
    return {
        "record_count": record_count,
        "turn_count": len(turns),
        "models": models,
        "files": files,
        "usage_record_count": usage_record_count,
        "tokens": token_totals,
    }


def load_activity_metadata(lash_log_path: Path | None) -> dict[str, Any]:
    empty = {
        "exec_result_count": 0,
        "tool_call_count": 0,
        "tool_call_breakdown": {},
    }
    if not lash_log_path or not lash_log_path.exists():
        return empty

    tool_call_count = 0
    exec_result_count = 0
    tool_call_breakdown: dict[str, int] = defaultdict(int)

    for raw_line in lash_log_path.read_text(errors="replace").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        message = record.get("message")
        if not isinstance(message, str):
            continue
        if message.startswith("PARALLEL: ToolCall #"):
            tool_call_count += 1
            match = re.search(r"'([^']+)'", message)
            if match:
                tool_call_breakdown[match.group(1)] += 1
        elif message.startswith("PARALLEL: ExecResult received"):
            exec_result_count += 1

    return {
        "exec_result_count": exec_result_count,
        "tool_call_count": tool_call_count,
        "tool_call_breakdown": dict(sorted(tool_call_breakdown.items())),
    }


def load_trajectory_metadata(trajectory_path: Path | None) -> dict[str, Any]:
    empty = {
        "models": [],
        "llm_turn_count": 0,
        "llm_call_count": 0,
        "tool_call_count": 0,
        "batch_call_count": 0,
        "tool_call_breakdown": {},
        "assistant_response": None,
    }
    if not trajectory_path or not trajectory_path.exists():
        return empty

    data = load_json(trajectory_path)
    steps = data.get("steps")
    if not isinstance(steps, list):
        return empty

    models: list[str] = []
    tool_call_breakdown: dict[str, int] = defaultdict(int)
    assistant_parts: list[str] = []
    turn_count = 0
    tool_call_count = 0
    batch_call_count = 0

    def add_model(value: Any) -> None:
        if isinstance(value, str) and value and value not in models:
            models.append(value)

    agent = data.get("agent")
    if isinstance(agent, dict):
        add_model(agent.get("model_name"))

    for step in steps:
        if not isinstance(step, dict) or step.get("source") != "agent":
            continue
        turn_count += 1
        add_model(step.get("model_name"))
        message = step.get("message")
        if isinstance(message, str):
            stripped = message.strip()
            if stripped and stripped != "(tool use)":
                assistant_parts.append(stripped)
        tool_calls = step.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            tool_name = tool_call.get("function_name") or tool_call.get("name")
            if not isinstance(tool_name, str) or not tool_name:
                continue
            tool_call_count += 1
            tool_call_breakdown[tool_name] += 1
            if tool_name == "batch":
                batch_call_count += 1

    return {
        "models": models,
        "llm_turn_count": turn_count,
        "llm_call_count": turn_count,
        "tool_call_count": tool_call_count,
        "batch_call_count": batch_call_count,
        "tool_call_breakdown": dict(sorted(tool_call_breakdown.items())),
        "assistant_response": "\n\n".join(assistant_parts).strip() or None,
    }


def load_opencode_metadata(opencode_path: Path | None) -> dict[str, Any]:
    empty = {
        "models": [],
        "llm_turn_count": 0,
        "llm_call_count": 0,
        "tool_call_count": 0,
        "batch_call_count": 0,
        "tool_call_breakdown": {},
        "assistant_response": None,
        "tokens": {
            "input": 0,
            "output": 0,
            "reasoning": 0,
            "cache": 0,
            "cache_read": 0,
            "cache_write": 0,
            "provider_total": 0,
        },
    }
    if not opencode_path or not opencode_path.exists():
        return empty

    models: list[str] = []
    tool_call_breakdown: dict[str, int] = defaultdict(int)
    assistant_parts: list[str] = []
    turn_message_ids: set[str] = set()
    llm_call_count = 0
    tool_call_count = 0
    tokens = {
        "input": 0,
        "output": 0,
        "reasoning": 0,
        "cache": 0,
        "cache_read": 0,
        "cache_write": 0,
        "provider_total": 0,
    }

    def add_model(value: Any) -> None:
        if isinstance(value, str) and value and value not in models:
            models.append(value)

    for raw_line in opencode_path.read_text(errors="replace").splitlines():
        raw_line = raw_line.strip()
        if not raw_line or not raw_line.startswith("{"):
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError:
            continue

        part = record.get("part")
        if not isinstance(part, dict):
            continue

        event_type = record.get("type")
        part_type = part.get("type")
        message_id = part.get("messageID")

        if event_type == "step_start" and isinstance(message_id, str) and message_id:
            turn_message_ids.add(message_id)

        if event_type == "text" and part_type == "text":
            text = part.get("text")
            if isinstance(text, str):
                stripped = text.strip()
                if stripped:
                    assistant_parts.append(stripped)

        if event_type == "tool_use" and part_type == "tool":
            tool_name = part.get("tool")
            if isinstance(tool_name, str) and tool_name:
                tool_call_count += 1
                tool_call_breakdown[tool_name] += 1

        if event_type != "step_finish" or part_type != "step-finish":
            continue

        llm_call_count += 1
        token_info = part.get("tokens")
        if isinstance(token_info, dict):
            tokens["input"] += int(token_info.get("input") or 0)
            tokens["output"] += int(token_info.get("output") or 0)
            tokens["reasoning"] += int(token_info.get("reasoning") or 0)
            cache = token_info.get("cache")
            if isinstance(cache, dict):
                cache_read = int(cache.get("read") or 0)
                cache_write = int(cache.get("write") or 0)
                tokens["cache_read"] += cache_read
                tokens["cache_write"] += cache_write
                tokens["cache"] += cache_read + cache_write
            tokens["provider_total"] += int(token_info.get("total") or 0)

    return {
        "models": models,
        "llm_turn_count": len(turn_message_ids),
        "llm_call_count": llm_call_count,
        "tool_call_count": tool_call_count,
        "batch_call_count": 0,
        "tool_call_breakdown": dict(sorted(tool_call_breakdown.items())),
        "assistant_response": "\n\n".join(assistant_parts).strip() or None,
        "tokens": tokens,
    }


def _load_codex_metadata(codex_path: Path | None) -> dict[str, Any]:
    """Parse Codex JSONL output to extract token usage and activity metadata."""
    empty: dict[str, Any] = {
        "models": [],
        "llm_turn_count": 0,
        "llm_call_count": 0,
        "tool_call_count": 0,
        "batch_call_count": 0,
        "tool_call_breakdown": {},
        "assistant_response": None,
        "tokens": {
            "input": 0,
            "output": 0,
            "reasoning": 0,
            "cache": 0,
            "cache_read": 0,
            "cache_write": 0,
            "provider_total": 0,
        },
    }
    if not codex_path or not codex_path.exists():
        return empty

    models: list[str] = []
    tool_call_breakdown: dict[str, int] = defaultdict(int)
    assistant_parts: list[str] = []
    llm_call_count = 0
    tool_call_count = 0
    tokens = {
        "input": 0,
        "output": 0,
        "reasoning": 0,
        "cache": 0,
        "cache_read": 0,
        "cache_write": 0,
        "provider_total": 0,
    }

    def add_model(value: Any) -> None:
        if isinstance(value, str) and value and value not in models:
            models.append(value)

    for raw_line in codex_path.read_text(errors="replace").splitlines():
        raw_line = raw_line.strip()
        if not raw_line or not raw_line.startswith("{"):
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError:
            continue

        event_type = record.get("type", "")

        if event_type in ("response.completed", "message"):
            llm_call_count += 1
            add_model(record.get("model"))
            usage = record.get("usage") or {}
            tokens["input"] += int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
            tokens["output"] += int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
            tokens["reasoning"] += int(usage.get("reasoning_tokens") or 0)
            cached = int(
                usage.get("cached_tokens")
                or usage.get("cache_read_input_tokens")
                or usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                or 0
            )
            tokens["cache"] += cached
            tokens["cache_read"] += cached
            total = int(usage.get("total_tokens") or 0)
            tokens["provider_total"] += total

        if event_type in ("function_call", "tool_use", "exec"):
            tool_call_count += 1
            tool_name = record.get("name") or record.get("function", {}).get("name") or "unknown"
            tool_call_breakdown[tool_name] += 1

        if event_type == "message" and record.get("role") == "assistant":
            content = record.get("content")
            if isinstance(content, str) and content.strip():
                assistant_parts.append(content.strip())
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = part.get("text", "").strip()
                        if text:
                            assistant_parts.append(text)

    return {
        "models": models,
        "llm_turn_count": llm_call_count,
        "llm_call_count": llm_call_count,
        "tool_call_count": tool_call_count,
        "batch_call_count": 0,
        "tool_call_breakdown": dict(sorted(tool_call_breakdown.items())),
        "assistant_response": "\n\n".join(assistant_parts).strip() or None,
        "tokens": tokens,
    }


def token_input_includes_cache(provider_metadata: dict[str, Any], requested_model: str | None) -> bool:
    provider_type = str(provider_metadata.get("active_provider_type") or "").strip().lower()
    requested = str(requested_model or "").strip().lower()
    if provider_type == "claude" or requested.startswith("anthropic/claude"):
        return False
    return True


def normalize_token_usage(
    *,
    raw_input: int,
    output: int,
    reasoning: int,
    cache_total: int,
    input_includes_cache: bool,
    provider_total: int | None = None,
    cache_read: int | None = None,
    cache_write: int | None = None,
) -> dict[str, Any]:
    normalized_input = raw_input - cache_total if input_includes_cache else raw_input
    normalized_input = max(normalized_input, 0)
    non_cache_total = normalized_input + output + reasoning
    total = non_cache_total + cache_total
    return {
        "input": normalized_input,
        "output": output,
        "reasoning": reasoning,
        "cache": cache_total,
        "cache_read": cache_read,
        "cache_write": cache_write,
        "non_cache_total": non_cache_total,
        "total": total,
        "raw_input": raw_input,
        "provider_total": provider_total,
        "input_includes_cache": input_includes_cache,
    }


def parse_resource_usage(path: Path | None) -> dict[str, Any]:
    empty = {
        "user_cpu_seconds": None,
        "system_cpu_seconds": None,
        "cpu_seconds_total": None,
        "cpu_percent": None,
        "wall_clock_seconds": None,
        "max_rss_kb": None,
        "major_page_faults": None,
        "minor_page_faults": None,
        "voluntary_context_switches": None,
        "involuntary_context_switches": None,
        "file_system_inputs": None,
        "file_system_outputs": None,
        "swaps": None,
    }
    if not path or not path.exists():
        return empty

    metrics = dict(empty)
    for line in path.read_text(errors="replace").splitlines():
        if ": " not in line:
            continue
        key, raw_value = line.split(": ", 1)
        value = raw_value.strip()
        key = key.strip()
        try:
            if key == "User time (seconds)":
                metrics["user_cpu_seconds"] = float(value)
            elif key == "System time (seconds)":
                metrics["system_cpu_seconds"] = float(value)
            elif key == "Percent of CPU this job got":
                metrics["cpu_percent"] = float(value.rstrip("%"))
            elif key == "Elapsed (wall clock) time (h:mm:ss or m:ss)":
                metrics["wall_clock_seconds"] = parse_time_duration(value)
            elif key == "Maximum resident set size (kbytes)":
                metrics["max_rss_kb"] = int(value)
            elif key == "Major (requiring I/O) page faults":
                metrics["major_page_faults"] = int(value)
            elif key == "Minor (reclaiming a frame) page faults":
                metrics["minor_page_faults"] = int(value)
            elif key == "Voluntary context switches":
                metrics["voluntary_context_switches"] = int(value)
            elif key == "Involuntary context switches":
                metrics["involuntary_context_switches"] = int(value)
            elif key == "File system inputs":
                metrics["file_system_inputs"] = int(value)
            elif key == "File system outputs":
                metrics["file_system_outputs"] = int(value)
            elif key == "Swaps":
                metrics["swaps"] = int(value)
        except ValueError:
            continue

    user_cpu = metrics["user_cpu_seconds"] or 0.0
    system_cpu = metrics["system_cpu_seconds"] or 0.0
    total_cpu = user_cpu + system_cpu
    metrics["cpu_seconds_total"] = total_cpu if total_cpu else None
    return metrics


def aggregate_resource_usage(items: list[dict[str, Any]]) -> dict[str, Any]:
    cpu_totals: list[float] = []
    wall_clock_seconds: list[float] = []
    cpu_percents: list[float] = []
    rss_values: list[int] = []

    for item in items:
        cpu_total = item.get("cpu_seconds_total")
        if isinstance(cpu_total, (float, int)):
            cpu_totals.append(float(cpu_total))
        wall = item.get("wall_clock_seconds")
        if isinstance(wall, (float, int)):
            wall_clock_seconds.append(float(wall))
        cpu_percent = item.get("cpu_percent")
        if isinstance(cpu_percent, (float, int)):
            cpu_percents.append(float(cpu_percent))
        rss = item.get("max_rss_kb")
        if isinstance(rss, int):
            rss_values.append(rss)

    return {
        "sample_count": len(items),
        "cpu_seconds_sum": sum(cpu_totals),
        "cpu_seconds_avg": numeric_mean(cpu_totals),
        "cpu_seconds_max": max(cpu_totals) if cpu_totals else None,
        "wall_clock_seconds_sum": sum(wall_clock_seconds),
        "wall_clock_seconds_avg": numeric_mean(wall_clock_seconds),
        "wall_clock_seconds_max": max(wall_clock_seconds) if wall_clock_seconds else None,
        "cpu_percent_avg": numeric_mean(cpu_percents),
        "cpu_percent_max": max(cpu_percents) if cpu_percents else None,
        "max_rss_kb_avg": numeric_mean([float(value) for value in rss_values]),
        "max_rss_kb_max": max(rss_values) if rss_values else None,
    }


def load_session_activity_metadata(session_db_paths: list[Path]) -> dict[str, Any]:
    turn_count = 0
    tool_call_count = 0
    batch_call_count = 0
    tool_call_breakdown: dict[str, int] = defaultdict(int)

    for db_path in session_db_paths:
        try:
            resolved = db_path.resolve()
            connection = sqlite3.connect(
                f"file:{resolved.as_posix()}?mode=ro&immutable=1",
                uri=True,
            )
        except sqlite3.Error:
            continue
        with connection:
            try:
                cursor = connection.execute("SELECT tool_calls_json FROM history_turns")
            except sqlite3.Error:
                continue
            for (tool_calls_json,) in cursor.fetchall():
                turn_count += 1
                try:
                    tool_calls = json.loads(tool_calls_json or "[]")
                except json.JSONDecodeError:
                    continue
                if not isinstance(tool_calls, list):
                    continue
                for item in tool_calls:
                    if not isinstance(item, dict):
                        continue
                    tool_name = item.get("tool")
                    if not isinstance(tool_name, str) or not tool_name:
                        continue
                    tool_call_count += 1
                    tool_call_breakdown[tool_name] += 1
                    if tool_name == "batch":
                        batch_call_count += 1

    return {
        "turn_count": turn_count,
        "tool_call_count": tool_call_count,
        "batch_call_count": batch_call_count,
        "tool_call_breakdown": dict(sorted(tool_call_breakdown.items())),
    }


def combine_activity_metadata(
    llm_metadata: dict[str, Any],
    session_activity: dict[str, Any],
    lash_log_activity: dict[str, Any],
    trajectory_metadata: dict[str, Any],
) -> dict[str, Any]:
    llm_record_count = int(llm_metadata.get("record_count") or 0)
    llm_turn_count = int(llm_metadata.get("turn_count") or 0)
    session_turn_count = int(session_activity.get("turn_count") or 0)
    session_tool_count = int(session_activity.get("tool_call_count") or 0)
    session_batch_count = int(session_activity.get("batch_call_count") or 0)
    exec_result_count = int(lash_log_activity.get("exec_result_count") or 0)
    trajectory_turn_count = int(trajectory_metadata.get("llm_turn_count") or 0)
    trajectory_call_count = int(trajectory_metadata.get("llm_call_count") or 0)
    trajectory_tool_count = int(trajectory_metadata.get("tool_call_count") or 0)
    trajectory_batch_count = int(trajectory_metadata.get("batch_call_count") or 0)

    tool_breakdown = trajectory_metadata.get("tool_call_breakdown") or {}
    if not tool_breakdown:
        tool_breakdown = session_activity.get("tool_call_breakdown") or {}
    if not tool_breakdown:
        tool_breakdown = lash_log_activity.get("tool_call_breakdown") or {}

    return {
        "llm_record_count": llm_record_count,
        "llm_turn_count": max(llm_turn_count, session_turn_count, trajectory_turn_count),
        "llm_call_count": max(llm_record_count, exec_result_count, trajectory_call_count),
        "tool_call_count": max(
            session_tool_count,
            int(lash_log_activity.get("tool_call_count") or 0),
            trajectory_tool_count,
        ),
        "batch_call_count": max(session_batch_count, trajectory_batch_count),
        "tool_call_breakdown": tool_breakdown,
    }


def copy_named_artifact(
    copied_artifacts: dict[str, str],
    key: str,
    src: Path,
    dst: Path,
    run_dir: Path,
) -> None:
    if copy_artifact(src, dst):
        copied_artifacts[key] = safe_relative(dst, run_dir)


def copy_directory_artifacts(src_dir: Path, dst_dir: Path, run_dir: Path) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    if not src_dir.exists():
        return copied
    for src in sorted(path for path in src_dir.rglob("*") if path.is_file()):
        dst = dst_dir / src.relative_to(src_dir)
        copy_artifact(src, dst)
        copied.append(
            {
                "name": src.name,
                "path": safe_relative(dst, run_dir),
            }
        )
    return copied


def infer_command_metadata(
    command_dir: Path,
    command_text: str,
    *,
    command_index: int,
    command_count: int,
) -> dict[str, Any]:
    metadata_path = command_dir / "metadata.json"
    if metadata_path.exists():
        data = load_json(metadata_path)
        if isinstance(data, dict):
            return data

    normalized = command_text.strip()
    if "opencode" in normalized and " run" in normalized:
        return {"phase": "main", "purpose": "agent_run", "family": "opencode", "is_main": True}
    if "opencode.json" in normalized:
        return {"phase": "bootstrap", "purpose": "config", "family": "opencode", "is_main": False}
    if "skills" in normalized and "cp -r" in normalized:
        return {"phase": "bootstrap", "purpose": "skills", "family": "opencode", "is_main": False}
    if "codex" in normalized and "exec" in normalized:
        return {"phase": "main", "purpose": "agent_run", "family": "codex", "is_main": True}
    if normalized.startswith("lash ") or " lash " in f" {normalized} ":
        return {"phase": "main", "purpose": "agent_run", "family": "lash", "is_main": True}
    return {
        "phase": "main" if command_index == command_count - 1 else "bootstrap",
        "purpose": "agent_run" if command_index == command_count - 1 else "setup",
        "family": "unknown",
        "is_main": command_index == command_count - 1,
    }


def select_resource_usage_scope(
    command_records: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    all_items = [
        record["resource_usage"]
        for record in command_records
        if isinstance(record.get("resource_usage"), dict)
    ]
    main_items = [
        record["resource_usage"]
        for record in command_records
        if isinstance(record.get("resource_usage"), dict)
        and ((record.get("metadata") or {}).get("phase") == "main")
    ]
    overhead_items = [
        record["resource_usage"]
        for record in command_records
        if isinstance(record.get("resource_usage"), dict)
        and ((record.get("metadata") or {}).get("phase") != "main")
    ]
    return (
        aggregate_resource_usage(all_items),
        aggregate_resource_usage(main_items),
        aggregate_resource_usage(overhead_items),
    )


def resource_usage_bucket(resource_usage: dict[str, Any], bucket: str) -> dict[str, Any]:
    nested = resource_usage.get(bucket)
    if isinstance(nested, dict):
        return nested
    if bucket in {"main_commands", "all_commands"}:
        return resource_usage
    return {}


def summarize_trial_resource_usage(
    trials: list[dict[str, Any]],
    bucket: str,
) -> dict[str, Any]:
    resource_cpu_seconds: list[float] = []
    resource_rss_kb: list[float] = []
    resource_wall_seconds: list[float] = []
    sampled_trials = 0

    for trial in trials:
        resource_usage = resource_usage_bucket(trial.get("resource_usage") or {}, bucket)
        if int(resource_usage.get("sample_count") or 0) > 0:
            sampled_trials += 1
        cpu_seconds = resource_usage.get("cpu_seconds_sum")
        if isinstance(cpu_seconds, (float, int)):
            resource_cpu_seconds.append(float(cpu_seconds))
        wall_seconds = resource_usage.get("wall_clock_seconds_sum")
        if isinstance(wall_seconds, (float, int)):
            resource_wall_seconds.append(float(wall_seconds))
        max_rss_kb = resource_usage.get("max_rss_kb_max")
        if isinstance(max_rss_kb, (float, int)):
            resource_rss_kb.append(float(max_rss_kb))

    return {
        "sampled_trials": sampled_trials,
        "cpu_seconds_sum": sum(resource_cpu_seconds),
        "cpu_seconds_avg": numeric_mean(resource_cpu_seconds),
        "wall_clock_seconds_sum": sum(resource_wall_seconds),
        "wall_clock_seconds_avg": numeric_mean(resource_wall_seconds),
        "max_rss_kb_avg": numeric_mean(resource_rss_kb),
        "max_rss_kb_max": max(resource_rss_kb) if resource_rss_kb else None,
    }


@dataclass
class ExportArgs:
    job_dir: Path
    results_dir: Path
    agent: str
    dataset: str
    execution_mode: str
    preset: str | None
    requested_model: str | None
    variant: str | None
    context_strategy: str | None
    harbor_env: str
    registry_url: str
    n_concurrent: int
    attempts: int
    timeout_multiplier: float
    delete_after_run: bool
    debug: bool
    binary_path: str | None
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
    preset, preset_source = resolve_preset(args.preset, args.exact_tasks)
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

    provider_metadata = load_provider_metadata(args.provider_config)
    raw_agent_tokens = {
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
    agent_dir = trial_dir / "agent"
    lash_home_dir = agent_dir / "lash-home"
    trajectory_path = agent_dir / "trajectory.json"
    copied_files = {
        "result_json": trial_dir / "result.json",
        "config_json": trial_dir / "config.json",
        "trial_log": trial_dir / "trial.log",
        "exception_txt": trial_dir / "exception.txt",
        "verifier_reward": trial_dir / "verifier" / "reward.txt",
        "verifier_stdout": trial_dir / "verifier" / "test-stdout.txt",
        "trajectory_json": trajectory_path,
        "opencode_raw": agent_dir / "opencode.txt",
        "codex_raw": agent_dir / "codex.txt",
        "lash_log": lash_home_dir / "lash.log",
        "lash_config": lash_home_dir / "config.json",
        "models_cache": lash_home_dir / "cache" / "models.json",
    }
    for key, src in copied_files.items():
        copy_named_artifact(
            copied_artifacts,
            key,
            src,
            artifacts_dir / f"{key}{src.suffix or '.txt'}",
            run_dir,
        )

    command_records: list[dict[str, Any]] = []
    command_dirs = sorted((trial_dir / "agent").glob("command-*"))
    for index, command_dir in enumerate(command_dirs):
        command_idx = command_dir.name
        record: dict[str, Any] = {"name": command_idx}
        command_text = read_text(command_dir / "command.txt")
        metadata = infer_command_metadata(
            command_dir,
            command_text,
            command_index=index,
            command_count=len(command_dirs),
        )
        for label, filename in (
            ("command", "command.txt"),
            ("stdout", "stdout.txt"),
            ("stderr", "stderr.txt"),
            ("return_code", "return-code.txt"),
            ("metadata_path", "metadata.json"),
            ("resource_usage_path", "resource-usage.txt"),
        ):
            src = command_dir / filename
            if not src.exists():
                continue
            dst = artifacts_dir / command_idx / filename
            copy_artifact(src, dst)
            record[label] = safe_relative(dst, run_dir)
        resource_usage = parse_resource_usage(command_dir / "resource-usage.txt")
        if any(value is not None for value in resource_usage.values()):
            record["resource_usage"] = resource_usage
        record["metadata"] = metadata
        command_records.append(record)

    setup_record: dict[str, str] = {}
    for label, filename in (("stdout", "stdout.txt"), ("stderr", "stderr.txt"), ("return_code", "return-code.txt")):
        src = trial_dir / "agent" / "setup" / filename
        if not src.exists():
            continue
        dst = artifacts_dir / "setup" / filename
        copy_artifact(src, dst)
        setup_record[label] = safe_relative(dst, run_dir)

    sessions_dir = lash_home_dir / "sessions"
    session_artifacts = copy_directory_artifacts(sessions_dir, artifacts_dir / "sessions", run_dir)
    llm_candidates = sorted(sessions_dir.glob("*.llm.jsonl"))
    llm_metadata = (
        load_llm_metadata(llm_candidates)
        if llm_candidates
        else {
            "record_count": 0,
            "turn_count": 0,
            "models": [],
            "files": [],
            "usage_record_count": 0,
            "tokens": {
                "raw_input": 0,
                "output": 0,
                "cache": 0,
                "reasoning": 0,
            },
        }
    )
    session_db_candidates = sorted(sessions_dir.glob("*.db"))
    trajectory_metadata = load_trajectory_metadata(trajectory_path)
    opencode_metadata = load_opencode_metadata(agent_dir / "opencode.txt")
    codex_metadata = _load_codex_metadata(agent_dir / "codex.txt")
    activity_metadata = combine_activity_metadata(
        llm_metadata,
        load_session_activity_metadata(session_db_candidates),
        load_activity_metadata(lash_home_dir / "lash.log"),
        trajectory_metadata,
    )
    if args.agent == "opencode":
        activity_metadata = {
            "llm_record_count": opencode_metadata["llm_call_count"],
            "llm_turn_count": opencode_metadata["llm_turn_count"],
            "llm_call_count": opencode_metadata["llm_call_count"],
            "tool_call_count": opencode_metadata["tool_call_count"],
            "batch_call_count": opencode_metadata["batch_call_count"],
            "tool_call_breakdown": opencode_metadata["tool_call_breakdown"],
        }
    elif args.agent == "codex":
        activity_metadata = {
            "llm_record_count": codex_metadata["llm_call_count"],
            "llm_turn_count": codex_metadata["llm_turn_count"],
            "llm_call_count": codex_metadata["llm_call_count"],
            "tool_call_count": codex_metadata["tool_call_count"],
            "batch_call_count": codex_metadata["batch_call_count"],
            "tool_call_breakdown": codex_metadata["tool_call_breakdown"],
        }

    token_source = "agent_result"
    token_details = normalize_token_usage(
        raw_input=raw_agent_tokens["input"],
        output=raw_agent_tokens["output"],
        reasoning=0,
        cache_total=raw_agent_tokens["cache"],
        input_includes_cache=token_input_includes_cache(provider_metadata, args.requested_model),
    )
    if args.agent == "opencode" and (agent_dir / "opencode.txt").exists():
        opencode_tokens = opencode_metadata["tokens"]
        token_details = normalize_token_usage(
            raw_input=int(opencode_tokens.get("input") or 0),
            output=int(opencode_tokens.get("output") or 0),
            reasoning=int(opencode_tokens.get("reasoning") or 0),
            cache_total=int(opencode_tokens.get("cache") or 0),
            input_includes_cache=False,
            provider_total=int(opencode_tokens.get("provider_total") or 0),
            cache_read=int(opencode_tokens.get("cache_read") or 0),
            cache_write=int(opencode_tokens.get("cache_write") or 0),
        )
        token_source = "opencode_log"
    elif args.agent == "codex" and (agent_dir / "codex.txt").exists():
        codex_tokens = codex_metadata["tokens"]
        token_details = normalize_token_usage(
            raw_input=int(codex_tokens.get("input") or 0),
            output=int(codex_tokens.get("output") or 0),
            reasoning=int(codex_tokens.get("reasoning") or 0),
            cache_total=int(codex_tokens.get("cache") or 0),
            input_includes_cache=False,
            provider_total=int(codex_tokens.get("provider_total") or 0),
            cache_read=int(codex_tokens.get("cache_read") or 0),
            cache_write=int(codex_tokens.get("cache_write") or 0),
        )
        token_source = "codex_log"
    elif int(llm_metadata.get("usage_record_count") or 0) > 0:
        llm_tokens = llm_metadata.get("tokens") or {}
        token_details = normalize_token_usage(
            raw_input=int(llm_tokens.get("raw_input") or 0),
            output=int(llm_tokens.get("output") or 0),
            reasoning=int(llm_tokens.get("reasoning") or 0),
            cache_total=int(llm_tokens.get("cache") or 0),
            input_includes_cache=token_input_includes_cache(
                provider_metadata, args.requested_model
            ),
        )
        token_source = "lash_llm_log"
    elif raw_agent_tokens["total"] > 0:
        token_source = "agent_result_fallback"

    tokens = {
        "input": token_details["input"],
        "output": token_details["output"],
        "reasoning": token_details["reasoning"],
        "cache": token_details["cache"],
        "cache_read": token_details["cache_read"],
        "cache_write": token_details["cache_write"],
        "non_cache_total": token_details["non_cache_total"],
        "total": token_details["total"],
    }
    resolved_models: list[str] = []
    for model in [
        *llm_metadata["models"],
        *trajectory_metadata["models"],
        *opencode_metadata["models"],
    ]:
        if isinstance(model, str) and model and model not in resolved_models:
            resolved_models.append(model)
    if not resolved_models and args.requested_model:
        resolved_models.append(args.requested_model)
    resource_usage_all, resource_usage_main, resource_usage_overhead = (
        select_resource_usage_scope(command_records)
    )
    resource_usage = dict(
        resource_usage_main
        if int(resource_usage_main.get("sample_count") or 0) > 0
        else resource_usage_all
    )
    resource_usage["scope"] = (
        "main_commands"
        if int(resource_usage_main.get("sample_count") or 0) > 0
        else "all_commands"
    )
    resource_usage["all_commands"] = resource_usage_all
    resource_usage["main_commands"] = resource_usage_main
    resource_usage["overhead_commands"] = resource_usage_overhead
    resource_usage["command_phase_counts"] = {
        "main": sum(
            1
            for record in command_records
            if ((record.get("metadata") or {}).get("phase") == "main")
        ),
        "overhead": sum(
            1
            for record in command_records
            if ((record.get("metadata") or {}).get("phase") != "main")
        ),
        "total": len(command_records),
    }

    verifier_stdout = read_text(trial_dir / "verifier" / "test-stdout.txt")
    command_stdout = ""
    command_stderr = ""
    primary_command = next(
        (
            record
            for record in reversed(command_records)
            if "stdout" in record or "stderr" in record
        ),
        command_records[-1] if command_records else None,
    )
    if primary_command:
        if "stdout" in primary_command:
            command_stdout = read_text(run_dir / primary_command["stdout"])
        if "stderr" in primary_command:
            command_stderr = read_text(run_dir / primary_command["stderr"])
    assistant_response = (
        str(agent_metadata.get("assistant_response") or "").strip()
        or str(trajectory_metadata.get("assistant_response") or "").strip()
        or str(opencode_metadata.get("assistant_response") or "").strip()
    )
    if not assistant_response:
        assistant_response = command_stdout
    if not command_stdout:
        command_stdout = assistant_response

    failure_reason = summarize_failure(
        status=status,
        exception_info=exception_info,
        verifier_stdout=verifier_stdout,
        command_stdout=assistant_response,
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
        "resource_usage": resource_usage,
        "metadata": {
            "agent": args.agent,
            "execution_mode": args.execution_mode,
            "preset": preset,
            "preset_source": preset_source,
            "requested_model": args.requested_model,
            "resolved_models": resolved_models,
            "variant": args.variant or None,
            "context_strategy": args.context_strategy or None,
            "provider": provider_metadata,
            "task_path": ((result.get("task_id") or {}).get("path")),
            "task_git_url": ((result.get("task_id") or {}).get("git_url")),
            "task_git_commit_id": ((result.get("task_id") or {}).get("git_commit_id")),
            "llm_request_count": activity_metadata["llm_record_count"],
            "llm_record_count": activity_metadata["llm_record_count"],
            "llm_turn_count": activity_metadata["llm_turn_count"],
            "llm_call_count": activity_metadata["llm_call_count"],
            "tool_call_count": activity_metadata["tool_call_count"],
            "tool_batch_count": activity_metadata["batch_call_count"],
            "tool_call_breakdown": activity_metadata["tool_call_breakdown"],
            "llm_debug_files": llm_metadata["files"],
            "assistant_response_present": bool(assistant_response),
            "token_accounting": {
                "source": token_source,
                "raw_input": token_details["raw_input"],
                "input_includes_cache": token_details["input_includes_cache"],
                "provider_total": token_details["provider_total"],
                "non_cache_total": token_details["non_cache_total"],
                "cache_read": token_details["cache_read"],
                "cache_write": token_details["cache_write"],
            },
            "environment": {
                "type": ((config.get("environment") or {}).get("type")),
                "override_cpus": ((config.get("environment") or {}).get("override_cpus")),
                "override_memory_mb": ((config.get("environment") or {}).get("override_memory_mb")),
            },
        },
        "failure_reason": failure_reason,
        "logs": {
            "assistant_excerpt": shorten(assistant_response, from_end=True),
            "verifier_excerpt": shorten(verifier_stdout),
            "stderr_excerpt": shorten(command_stderr, from_end=True),
        },
        "artifacts": {
            "files": copied_artifacts,
            "commands": command_records,
            "setup": setup_record,
            "sessions": session_artifacts,
        },
    }


def build_global_stats(trials: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = defaultdict(int)
    rewards: list[float] = []
    trial_seconds: list[float] = []
    agent_exec_seconds: list[float] = []
    total_tokens = {
        "input": 0,
        "output": 0,
        "reasoning": 0,
        "cache": 0,
        "cache_read": 0,
        "cache_write": 0,
        "non_cache_total": 0,
        "total": 0,
    }
    activity_totals = {
        "llm_records": 0,
        "llm_turns": 0,
        "llm_calls": 0,
        "tool_calls": 0,
        "tool_batches": 0,
    }
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
        metadata = trial.get("metadata") or {}
        activity_totals["llm_records"] += int(metadata.get("llm_record_count") or 0)
        activity_totals["llm_turns"] += int(metadata.get("llm_turn_count") or 0)
        activity_totals["llm_calls"] += int(metadata.get("llm_call_count") or 0)
        activity_totals["tool_calls"] += int(metadata.get("tool_call_count") or 0)
        activity_totals["tool_batches"] += int(metadata.get("tool_batch_count") or 0)

    trial_count = len(trials)
    passed = statuses["pass"]
    resource_usage_main = summarize_trial_resource_usage(trials, "main_commands")
    resource_usage_all = summarize_trial_resource_usage(trials, "all_commands")
    resource_usage_overhead = summarize_trial_resource_usage(trials, "overhead_commands")
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
        "activity_total": activity_totals,
        "activity_avg": {
            key: (value / trial_count if trial_count else 0.0)
            for key, value in activity_totals.items()
        },
        "resource_usage": resource_usage_main,
        "resource_usage_all_commands": resource_usage_all,
        "resource_usage_overhead_commands": resource_usage_overhead,
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
                "activity_total": stats["activity_total"],
                "activity_avg": stats["activity_avg"],
                "resource_usage": stats["resource_usage"],
                "trial_names": [trial["trial_name"] for trial in task_trials],
            }
        )
    return rollups


def export_run(args: ExportArgs) -> Path:
    preset, preset_source = resolve_preset(args.preset, args.exact_tasks)
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

    task_scope = build_task_scope(args.exact_tasks, args.task_patterns, trials)

    run_payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "exported_at": iso_utc_now(),
        "job_name": args.job_dir.name,
        "source_job_dir": str(args.job_dir.resolve()),
        "params": {
            "agent": args.agent,
            "dataset": args.dataset,
            "execution_mode": args.execution_mode,
            "preset": preset,
            "preset_source": preset_source,
            "requested_model": args.requested_model,
            "variant": args.variant or None,
            "context_strategy": args.context_strategy or None,
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
            "task_scope": task_scope,
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
    run = load_json(run_dir / "run.json")
    params = run.get("params")
    if not isinstance(params, dict):
        return run

    preset, preset_source = resolve_preset(
        params.get("preset") or params.get("task_preset"),
        params.get("exact_tasks"),
    )
    if preset and not params.get("preset"):
        params["preset"] = preset
    if preset_source and not params.get("preset_source"):
        params["preset_source"] = preset_source

    task_scope = params.get("task_scope")
    if not isinstance(task_scope, dict):
        task_scope = build_task_scope(
            params.get("exact_tasks"),
            params.get("task_patterns"),
            run.get("trials") if isinstance(run.get("trials"), list) else [],
        )
        params["task_scope"] = task_scope

    trials = run.get("trials")
    if isinstance(trials, list):
        for trial in trials:
            if not isinstance(trial, dict):
                continue
            metadata = trial.get("metadata")
            if not isinstance(metadata, dict):
                continue
            if preset and not metadata.get("preset"):
                metadata["preset"] = preset
            if preset_source and not metadata.get("preset_source"):
                metadata["preset_source"] = preset_source

    return run


def load_run_summaries(results_dir: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for run_json in sorted((results_dir / "runs").glob("*/run.json"), reverse=True):
        run = load_run(run_json.parent)
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
                "agent": params.get("agent"),
                "execution_mode": params.get("execution_mode"),
                "preset": params.get("preset"),
                "requested_model": params.get("requested_model"),
                "variant": params.get("variant"),
                "context_strategy": params.get("context_strategy"),
                "provider": (params.get("provider") or {}).get("active_provider"),
                "requested_task_count": ((params.get("task_scope") or {}).get("requested_task_count")),
                "executed_task_count": ((params.get("task_scope") or {}).get("executed_task_count")),
                "scope_mismatch": bool((params.get("task_scope") or {}).get("scope_mismatch")),
                "requested_tasks": ((params.get("task_scope") or {}).get("requested_tasks") or []),
                "executed_tasks": ((params.get("task_scope") or {}).get("executed_tasks") or []),
                "trials_total": stats.get("trials_total", 0),
                "trials_passed": stats.get("trials_passed", 0),
                "trials_failed": stats.get("trials_failed", 0),
                "trials_errors": stats.get("trials_errors", 0),
                "pass_rate": stats.get("pass_rate", 0.0),
                "tokens_total": (stats.get("tokens_total") or {}).get("total", 0),
                "tokens_non_cache_total": (stats.get("tokens_total") or {}).get(
                    "non_cache_total", 0
                ),
                "tokens_reasoning_total": (stats.get("tokens_total") or {}).get("reasoning", 0),
                "tokens_cache_total": (stats.get("tokens_total") or {}).get("cache", 0),
                "llm_calls_total": (stats.get("activity_total") or {}).get("llm_calls", 0),
                "turns_total": (stats.get("activity_total") or {}).get("llm_turns", 0),
                "tool_calls_total": (stats.get("activity_total") or {}).get("tool_calls", 0),
                "cpu_seconds_total": (stats.get("resource_usage") or {}).get("cpu_seconds_sum", 0),
                "peak_rss_kb": (stats.get("resource_usage") or {}).get("max_rss_kb_max"),
                "cpu_seconds_all_total": (
                    stats.get("resource_usage_all_commands") or {}
                ).get("cpu_seconds_sum", 0),
                "peak_rss_all_kb": (
                    stats.get("resource_usage_all_commands") or {}
                ).get("max_rss_kb_max"),
                "cpu_seconds_overhead_total": (
                    stats.get("resource_usage_overhead_commands") or {}
                ).get("cpu_seconds_sum", 0),
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
