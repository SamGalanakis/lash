"""Lash CLI system adapter for Continual Learning Bench."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

from ...interface import (
    ContinualLearningSystem,
    Observation,
    Query,
    Response,
    observation_marks_instance_complete,
)
from ...registry import register_system
from ...usage import build_usage_event


@register_system("lash")
class LashSystem(ContinualLearningSystem):
    """Run Lash once per CLBench query, resuming one local Lash session."""

    supports_baseline = True
    parallel_safe = True

    def __init__(
        self,
        model: str = "gpt-5.5",
        name: str = "lash",
        provider_id: str | None = None,
        variant: str | None = "high",
        execution_mode: str = "rlm",
        context_approach: str | None = None,
        timeout: int = 600,
        lash_binary: str | None = None,
        runner_binary: str | None = None,
        repo_root: str | None = None,
        single_session: bool = True,
        repair_attempts: int = 1,
    ):
        self._name = name
        self._model = model
        self._provider_id = provider_id
        self._variant = variant
        self._execution_mode = execution_mode
        self._context_approach = context_approach
        self._timeout = timeout
        self._runner_binary = (
            runner_binary
            or os.environ.get("LASH_CLBENCH_RUNNER_BINARY")
            or lash_binary
            or os.environ.get("LASH_CLBENCH_LASH_BINARY")
        )
        self._repo_root = Path(
            repo_root or os.environ.get("LASH_CLBENCH_REPO_ROOT", ".")
        ).expanduser().resolve()
        self._single_session = single_session
        self._repair_attempts = max(0, int(repair_attempts))

        state_root = Path(
            os.environ.get(
                "LASH_CLBENCH_STATE_ROOT",
                str(Path.home() / ".cache" / "lash_clbench"),
            )
        ).expanduser()
        run_root = state_root / "runs" / "lash-system"
        run_root.mkdir(parents=True, exist_ok=True)
        self._tmp_dir = Path(tempfile.mkdtemp(prefix="run_", dir=str(run_root)))
        self._lash_home = self._tmp_dir / "lash-home"
        self._workspace = self._tmp_dir / "workspace"
        self._turns_dir = self._tmp_dir / "turns"
        self._session_db = self._tmp_dir / "session.db"
        self._session_file: str | None = str(self._session_db)
        self._interaction_count = 0
        self._pending_feedback: str | None = None
        self._turn_history: list[dict[str, Any]] = []

        self._workspace.mkdir(parents=True, exist_ok=True)
        self._turns_dir.mkdir(parents=True, exist_ok=True)
        self._write_lash_config()

    @property
    def name(self) -> str:
        return self._name

    def respond(self, query: Query) -> Response:
        prompt = self._build_prompt(query)
        feedback = self._feedback_text(query)
        usage_path = self._turns_dir / f"turn-{self._interaction_count + 1:04d}-usage.json"

        action, metadata = self._run_and_parse(
            prompt=prompt,
            feedback=feedback,
            response_schema=query.response_schema,
            usage_path=usage_path,
        )
        self._interaction_count += 1
        self._pending_feedback = None
        return Response(action=action, metadata=metadata)

    def observe(
        self, observation: Observation, next_query: Optional[Query] = None
    ) -> None:
        _ = next_query
        content = observation.content.strip()
        self._pending_feedback = content or None
        if not self._single_session and observation_marks_instance_complete(observation):
            self._session_file = None

    def reset(self) -> None:
        self._session_file = str(self._session_db)
        self._interaction_count = 0
        self._pending_feedback = None
        self._turn_history = []
        if self._session_db.exists():
            self._session_db.unlink()
        self._clear_dir(self._workspace)
        self._clear_dir(self._turns_dir)
        self._write_lash_config()

    def get_run_artifacts(self) -> dict[str, Any]:
        return {
            "artifact_type": "lash",
            "lash_home": str(self._lash_home),
            "workspace": str(self._workspace),
            "session_file": self._session_file,
            "interaction_count": self._interaction_count,
            "turn_history": list(self._turn_history),
        }

    def _write_lash_config(self) -> None:
        host_config_path = Path.home() / ".lash" / "config.json"
        if not host_config_path.exists():
            raise FileNotFoundError(
                f"~/.lash/config.json missing; configure Lash provider first: {host_config_path}"
            )
        config = json.loads(host_config_path.read_text())
        if self._provider_id is not None:
            providers = config.get("providers")
            if not isinstance(providers, dict) or self._provider_id not in providers:
                available = ", ".join(sorted(providers)) if isinstance(providers, dict) else "<none>"
                raise ValueError(
                    f"provider_id {self._provider_id!r} is not configured in ~/.lash/config.json; "
                    f"available: {available}"
                )
            config["active_provider"] = self._provider_id

        self._lash_home.mkdir(parents=True, exist_ok=True)
        config_path = self._lash_home / "config.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        try:
            config_path.chmod(0o600)
        except OSError:
            pass

    def _runner_command(self, request_path: Path) -> list[str]:
        if self._runner_binary:
            cmd = [self._runner_binary]
        else:
            release_binary = self._repo_root / "target" / "release" / "bench-clbench-lash"
            if release_binary.exists():
                cmd = [str(release_binary)]
            else:
                cmd = ["cargo", "run", "--release", "-p", "bench-clbench-lash", "--"]

        cmd.extend(["--request", str(request_path)])
        return cmd

    def _build_prompt(self, query: Query) -> str:
        return query.prompt if query.prompt else "(no prompt content)"

    def _feedback_text(self, query: Query) -> str | None:
        feedback_parts = []
        if self._pending_feedback:
            feedback_parts.append(self._pending_feedback)
        if query.feedback and query.feedback.content.strip():
            feedback_parts.append(query.feedback.content.strip())
        return "\n".join(feedback_parts) if feedback_parts else None

    def _run_and_parse(
        self,
        *,
        prompt: str,
        feedback: str | None,
        response_schema: type[BaseModel],
        usage_path: Path,
    ) -> tuple[BaseModel, dict[str, Any]]:
        attempts: list[dict[str, Any]] = []
        proc, elapsed = self._run_runner(
            prompt,
            feedback=feedback,
            response_schema=response_schema,
            usage_path=usage_path,
            attempt=0,
        )
        usage_artifact = self._read_usage_artifact(usage_path)
        self._record_usage_events(usage_artifact, attempt=0)
        output_text = proc.stdout.strip()
        attempts.append(
            {
                "attempt": 0,
                "exit_code": proc.returncode,
                "elapsed_seconds": elapsed,
                "stdout_chars": len(proc.stdout),
                "stderr_chars": len(proc.stderr),
                "usage": usage_artifact,
            }
        )

        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip()
            raise RuntimeError(
                f"LLM call failed: Lash exited with code {proc.returncode}: {detail[:1000]}"
            )

        try:
            runner_response = json.loads(output_text)
        except Exception as exc:
            raise RuntimeError(f"Lash runner returned non-JSON output: {output_text[:1000]}") from exc

        try:
            action = response_schema.model_validate(runner_response["action"])
        except Exception as exc:
            raise RuntimeError(
                "Lash submitted a value that passed through the runner but could not "
                "be instantiated as the CLBench response model"
            ) from exc

        metadata = self._metadata(usage_path, attempts)
        self._turn_history.append(metadata)
        return action, metadata

    def _run_runner(
        self,
        prompt: str,
        *,
        feedback: str | None,
        response_schema: type[BaseModel],
        usage_path: Path,
        attempt: int,
    ) -> tuple[subprocess.CompletedProcess[str], float]:
        request_path = self._turns_dir / (
            f"turn-{self._interaction_count + 1:04d}-attempt-{attempt:02d}-request.json"
        )
        request = {
            "session_id": "root",
            "session_db": str(self._session_db),
            "trace_path": str(self._turns_dir / "session.trace.jsonl"),
            "model": self._model,
            "provider_id": self._provider_id,
            "variant": self._variant,
            "iteration": self._interaction_count + 1,
            "prompt": prompt,
            "feedback": feedback,
            "response_schema": response_schema.model_json_schema(),
            "init_diary": self._interaction_count == 0
            and attempt == 0
            and not self._session_db.exists(),
        }
        request_path.write_text(json.dumps(request, indent=2) + "\n")
        cmd = self._runner_command(request_path)
        env = os.environ.copy()
        env["LASH_HOME"] = str(self._lash_home)
        started = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=self._workspace,
            env=env,
            capture_output=True,
            text=True,
            timeout=self._timeout,
        )
        elapsed = time.perf_counter() - started
        self._session_file = str(self._session_db)
        if proc.stdout.strip():
            try:
                usage_path.write_text(json.dumps(json.loads(proc.stdout)["usage"], indent=2) + "\n")
            except Exception:
                pass
        return proc, elapsed

    def _metadata(
        self,
        usage_path: Path,
        attempts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        usage = None
        if usage_path.exists():
            try:
                usage = json.loads(usage_path.read_text())
            except Exception:
                usage = None
        return {
            "interaction_count": self._interaction_count + 1,
            "system_type": "lash",
            "model": self._model,
            "provider_id": self._provider_id,
            "variant": self._variant,
            "execution_mode": self._execution_mode,
            "context_approach": self._context_approach,
            "session_file": self._session_file,
            "attempts": attempts,
            "turn_usage": usage,
        }

    @staticmethod
    def _clear_dir(path: Path) -> None:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _read_usage_artifact(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    def _usage_provider(self) -> str | None:
        if self._provider_id == "codex":
            return "openai"
        return self._provider_id

    def _record_usage_events(
        self, usage_artifact: dict[str, Any] | None, *, attempt: int
    ) -> None:
        if not usage_artifact:
            return
        entries = usage_artifact.get("delta_entries")
        if isinstance(entries, list):
            iterable = entries
        else:
            iterable = usage_artifact.get("by_source_model")
        if not isinstance(iterable, list):
            return
        for entry in iterable:
            if not isinstance(entry, dict):
                continue
            usage = entry.get("usage")
            if not isinstance(usage, dict):
                continue
            self.record_usage_event(
                build_usage_event(
                    model=str(entry.get("model") or self._model),
                    provider=self._usage_provider(),
                    input_tokens=usage.get("input_tokens"),
                    output_tokens=usage.get("output_tokens"),
                    reasoning_tokens=usage.get("reasoning_tokens"),
                    cached_input_tokens=usage.get("cached_input_tokens"),
                    call_type="completion",
                    raw_usage=usage,
                    metadata={
                        "source": entry.get("source"),
                        "attempt": attempt,
                        "session_file": self._session_file,
                    },
                )
            )

    @staticmethod
    def _repair_prompt(original_prompt: str, assistant_text: str, validation_error: str) -> str:
        return "\n".join(
            [
                "Your previous benchmark action did not match the required schema.",
                f"Validation error: {validation_error}",
                "Previous output:",
                assistant_text,
                "Return only a corrected action for the same task.",
                "Original prompt:",
                original_prompt,
            ]
        )
