#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATE_ROOT = Path(".benchmarks/appworld")
DEFAULT_DATASET = "dev"
DEFAULT_EXECUTION_MODE = "rlm"
DEFAULT_CONTEXT_APPROACH = "rolling_history"
APPWORLD_DATASETS = ("train", "dev", "test_normal", "test_challenge")


@dataclass
class TaskRun:
    task_id: str
    status: str
    success: bool | None
    exit_code: int | None
    elapsed_seconds: float
    task_dir: str
    error: str | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Lash on AppWorld via AppWorld MCP.")
    p.add_argument("--dataset", default=DEFAULT_DATASET, choices=APPWORLD_DATASETS)
    p.add_argument("--task-id", action="append", default=[])
    p.add_argument("--limit", type=int)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--run-id")
    p.add_argument("--experiment-name")
    p.add_argument("--model")
    p.add_argument("--variant")
    p.add_argument("--provider-id")
    p.add_argument("--execution-mode", default=DEFAULT_EXECUTION_MODE)
    p.add_argument("--context-approach", default=DEFAULT_CONTEXT_APPROACH)
    p.add_argument("--lash-binary")
    p.add_argument("--appworld-root", type=Path)
    p.add_argument("--state-root", type=Path, default=STATE_ROOT)
    p.add_argument("--timeout-seconds", type=int, default=1800)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_http(url: str, timeout_seconds: float = 60.0) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status < 500:
                    return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"timed out waiting for {url}: {last_error}")


def wait_tcp(host: str, port: int, timeout_seconds: float = 60.0) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return
        except OSError as exc:
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"timed out waiting for {host}:{port}: {last_error}")


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} failed with HTTP {exc.code}: {detail}") from exc


def venv_bin(state_root: Path, name: str) -> Path:
    return state_root / "venv" / "bin" / name


def ensure_setup(args: argparse.Namespace, root: Path) -> tuple[Path, Path]:
    state_root = (root / args.state_root).resolve()
    appworld = venv_bin(state_root, "appworld")
    if not appworld.exists():
        raise SystemExit("AppWorld venv missing; run bench/appworld/setup.sh first")
    if shutil.which("npx") is None:
        raise SystemExit("npx is required for mcp-remote")
    host_config = Path.home() / ".lash" / "config.json"
    if not host_config.exists():
        raise SystemExit("~/.lash/config.json missing; configure Lash provider first")
    appworld_root = (args.appworld_root or state_root / "root").resolve()
    if not (appworld_root / "data" / "datasets").exists():
        raise SystemExit(
            f"AppWorld data missing under {appworld_root}; run bench/appworld/setup.sh"
        )
    return state_root, appworld_root


def load_task_ids(args: argparse.Namespace, appworld_root: Path) -> list[str]:
    if args.task_id:
        task_ids = list(dict.fromkeys(args.task_id))
    else:
        dataset_file = appworld_root / "data" / "datasets" / f"{args.dataset}.txt"
        if not dataset_file.exists():
            raise SystemExit(f"dataset file missing: {dataset_file}")
        task_ids = [
            line.strip().split()[0]
            for line in dataset_file.read_text().splitlines()
            if line.strip()
        ]
    if args.offset:
        task_ids = task_ids[args.offset :]
    if args.limit is not None:
        task_ids = task_ids[: args.limit]
    if not task_ids:
        raise SystemExit("No AppWorld tasks selected")
    return task_ids


def task_instruction(appworld_root: Path, task_id: str) -> str:
    specs = read_json(appworld_root / "data" / "tasks" / task_id / "specs.json")
    return str(specs["instruction"])


def task_scenario_id(task_id: str) -> str:
    return task_id if "_" not in task_id else task_id.rsplit("_", 1)[0]


def build_prompt(instruction: str) -> str:
    return f"""You are solving an AppWorld benchmark task on behalf of the supervisor.

Use the AppWorld MCP tools to inspect and update the simulated apps. Do not inspect AppWorld data files, task specs, or ground-truth files from the filesystem.

Rules:
- Act fully autonomously. Do not ask the user to confirm, clarify, or perform steps.
- You have permission to operate across the supervisor's connected AppWorld accounts.
- Never invent IDs, names, credentials, addresses, payment details, dates, or other values. Look them up through the relevant AppWorld APIs.
- Supervisor profile data, account passwords, addresses, cards, and other personal details are in the Supervisor app.
- References to friends, family, or relationships refer to contacts in the Phone app.
- Retrieve current date/time from the Phone app, not from the local machine clock or your internal clock.
- For temporal requests, use complete time ranges where appropriate, such as 00:00:00 through 23:59:59 for a full day.
- References to a file system mean the AppWorld file system app, not this machine's OS filesystem.
- For paginated APIs, inspect all pages instead of stopping after the first page.
- Avoid collateral damage. Only perform operations needed for the requested task.

Completion:
- After completing the task, call the Supervisor app's complete_task API.
- If the task asks a question, pass the minimal answer value to complete_task: only the entity, number, or direct value requested, not a full sentence.
- Numbers should be numeric, not words.
- If no answer is required, call complete_task without an answer, or with a null answer if the tool requires one.
- If you determine the task cannot be completed, call complete_task with failure status if that option is available.

Task instruction:
{instruction}
"""


def lash_command(args: argparse.Namespace, prompt: str, root: Path) -> list[str]:
    if args.lash_binary:
        cmd = [args.lash_binary]
    else:
        release_binary = root / "target" / "release" / "lash"
        if release_binary.exists():
            cmd = [str(release_binary)]
        else:
            cmd = ["cargo", "run", "--release", "-p", "lash-cli", "--"]
    if args.model:
        cmd.extend(["--model", args.model])
    if args.variant:
        cmd.extend(["--variant", args.variant])
    cmd.extend(["--execution-mode", args.execution_mode])
    cmd.extend(["--tool-surface", "appworld"])
    if args.context_approach:
        cmd.extend(["--context-approach", args.context_approach])
    cmd.extend(["--print", prompt])
    return cmd


def write_lash_config(run_dir: Path, mcp_url: str, provider_id: str | None) -> Path:
    host_config_path = Path.home() / ".lash" / "config.json"
    config = read_json(host_config_path)
    if provider_id:
        providers = config.get("providers")
        if not isinstance(providers, dict) or provider_id not in providers:
            raise SystemExit(f"provider {provider_id!r} is not configured in ~/.lash/config.json")
        config["active_provider"] = provider_id
    config["mcp_servers"] = {
        "appworld": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "mcp-remote", f"{mcp_url}/mcp/"],
            "startup_timeout_ms": 30000,
            "call_timeout_ms": 120000,
        },
    }
    lash_home = run_dir / "lash-home"
    lash_home.mkdir(parents=True, exist_ok=True)
    config_path = lash_home / "config.json"
    write_json(config_path, config)
    try:
        config_path.chmod(0o600)
    except OSError:
        pass
    return lash_home


def start_servers(state_root: Path, appworld_root: Path, run_dir: Path) -> tuple[subprocess.Popen[str], dict[str, str]]:
    env_port = free_port()
    apis_port = free_port()
    mcp_port = free_port()
    urls = {
        "environment": f"http://127.0.0.1:{env_port}",
        "apis": f"http://127.0.0.1:{apis_port}",
        "mcp": f"http://127.0.0.1:{mcp_port}",
    }
    appworld = venv_bin(state_root, "appworld")
    command = [
        str(appworld),
        "serve",
        "multiple",
        "--environment",
        f"--port {env_port}",
        "--apis",
        f"--port {apis_port}",
        "--mcp",
        f"http --port {mcp_port} --output-type content_only",
        "--root",
        str(appworld_root),
    ]
    logs_dir = run_dir / "server-logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout = (logs_dir / "stdout.txt").open("w")
    stderr = (logs_dir / "stderr.txt").open("w")
    proc = subprocess.Popen(
        command,
        cwd=repo_root(),
        stdout=stdout,
        stderr=stderr,
        text=True,
    )
    (logs_dir / "command.txt").write_text(" ".join(command) + "\n")
    try:
        wait_http(f"{urls['environment']}/")
        wait_http(f"{urls['apis']}/")
        wait_tcp("127.0.0.1", mcp_port)
    except Exception:
        proc.terminate()
        raise
    return proc, urls


def stop_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def run_task(
    args: argparse.Namespace,
    root: Path,
    run_dir: Path,
    appworld_root: Path,
    urls: dict[str, str],
    lash_home: Path,
    task_id: str,
) -> TaskRun:
    task_dir = run_dir / "tasks" / task_id
    workspace = task_dir / "workspace"
    task_dir.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        prompt = build_prompt(task_instruction(appworld_root, task_id))
        (task_dir / "prompt.txt").write_text(prompt)
        cmd = lash_command(args, prompt, root)
        (task_dir / "command.txt").write_text(" ".join(cmd) + "\n")
        if args.dry_run:
            return TaskRun(task_id, "dry-run", None, 0, time.time() - started, str(task_dir))

        initialize = post_json(
            f"{urls['environment']}/initialize",
            {
                "task_id": task_id,
                "experiment_name": args.experiment_name,
                "remote_apis_url": urls["apis"],
                "remote_mcp_url": urls["mcp"],
                "raise_on_failure": False,
                "ground_truth_mode": "full",
            },
        )
        write_json(task_dir / "initialize.json", initialize)

        env = os.environ.copy()
        env["LASH_HOME"] = str(lash_home)
        env.pop("APPWORLD_ROOT", None)
        proc = subprocess.run(
            cmd,
            cwd=workspace,
            env=env,
            capture_output=True,
            text=True,
            timeout=args.timeout_seconds,
        )
        (task_dir / "stdout.txt").write_text(proc.stdout)
        (task_dir / "stderr.txt").write_text(proc.stderr)
        (task_dir / "return-code.txt").write_text(str(proc.returncode) + "\n")

        save_response = post_json(f"{urls['environment']}/save", {"task_id": task_id})
        write_json(task_dir / "save.json", save_response)
        evaluation = post_json(
            f"{urls['environment']}/evaluate",
            {"task_id": task_id, "suppress_errors": True, "report": False},
        )
        write_json(task_dir / "evaluation.json", evaluation)
        post_json(f"{urls['environment']}/close", {"task_id": task_id})

        output = evaluation.get("output") if isinstance(evaluation, dict) else None
        success = output.get("success") if isinstance(output, dict) else None
        status = "ok" if proc.returncode == 0 else "agent-failed"
        return TaskRun(
            task_id=task_id,
            status=status,
            success=success if isinstance(success, bool) else None,
            exit_code=proc.returncode,
            elapsed_seconds=time.time() - started,
            task_dir=str(task_dir),
        )
    except Exception as exc:  # noqa: BLE001
        (task_dir / "error.txt").write_text(str(exc) + "\n")
        try:
            post_json(f"{urls['environment']}/close", {"task_id": task_id})
        except Exception:
            pass
        return TaskRun(
            task_id=task_id,
            status="error",
            success=False,
            exit_code=None,
            elapsed_seconds=time.time() - started,
            task_dir=str(task_dir),
            error=str(exc),
        )


def main() -> int:
    args = parse_args()
    root = repo_root()
    state_root, appworld_root = ensure_setup(args, root)
    task_ids = load_task_ids(args, appworld_root)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.experiment_name = args.experiment_name or f"lash-{run_id}"
    run_dir = state_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "created_at": now_iso(),
        "dataset": args.dataset,
        "task_ids": task_ids,
        "experiment_name": args.experiment_name,
        "model": args.model,
        "variant": args.variant,
        "provider_id": args.provider_id,
        "execution_mode": args.execution_mode,
        "context_approach": args.context_approach,
        "appworld_root": str(appworld_root),
    }
    write_json(run_dir / "manifest.json", manifest)

    server_proc: subprocess.Popen[str] | None = None
    results: list[TaskRun] = []
    try:
        if args.dry_run:
            urls = {"environment": "dry-run", "apis": "dry-run", "mcp": "dry-run"}
            lash_home = run_dir / "lash-home"
        else:
            server_proc, urls = start_servers(state_root, appworld_root, run_dir)
            lash_home = write_lash_config(run_dir, urls["mcp"], args.provider_id)
        write_json(run_dir / "server-urls.json", urls)

        for task_id in task_ids:
            result = run_task(args, root, run_dir, appworld_root, urls, lash_home, task_id)
            results.append(result)
            write_json(run_dir / "results.json", [r.__dict__ for r in results])
            print(
                f"{task_id}: {result.status} success={result.success} "
                f"elapsed={result.elapsed_seconds:.1f}s",
                flush=True,
            )
    finally:
        if server_proc is not None:
            stop_process(server_proc)

    passed = sum(1 for r in results if r.success is True)
    failed = sum(1 for r in results if r.success is False)
    scored = [r for r in results if r.success is not None]
    scenario_scores: dict[str, list[float]] = {}
    for result in scored:
        scenario_scores.setdefault(task_scenario_id(result.task_id), []).append(
            1.0 if result.success else 0.0
        )
    task_goal_completion = (
        round(100 * sum(scores for values in scenario_scores.values() for scores in values) / len(scored), 1)
        if scored
        else None
    )
    scenario_goal_completion = (
        round(100 * sum(min(values) for values in scenario_scores.values()) / len(scenario_scores), 1)
        if scenario_scores
        else None
    )
    summary = {
        "run_id": run_id,
        "completed_at": now_iso(),
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "unknown": len(results) - passed - failed,
        "aggregate": {
            "task_goal_completion": task_goal_completion,
            "scenario_goal_completion": scenario_goal_completion,
        },
        "results": [r.__dict__ for r in results],
    }
    write_json(run_dir / "results.json", summary)
    print(f"Run written to {run_dir}")
    print(f"Passed {passed}/{len(results)}")
    if args.dry_run:
        return 0
    return 0 if all(r.success is True for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
