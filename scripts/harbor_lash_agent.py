"""Harbor adapter for running lash headlessly inside benchmark environments."""

from __future__ import annotations

import os
import shlex
import json
from pathlib import Path

from harbor.agents.installed.base import BaseInstalledAgent, ExecInput
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths
from harbor.utils.templating import render_prompt_template

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LASH_BINARY = REPO_ROOT / "target" / "release" / "lash"
OPTIONAL_LIBS_DIR = REPO_ROOT / "bench" / "libs"
HOST_LASH_CONFIG = Path.home() / ".lash" / "config.json"

REMOTE_HOME = "/installed-agent/home"
REMOTE_LASH_HOME = (EnvironmentPaths.agent_dir / "lash-home").as_posix()
REMOTE_LASH_CONFIG = f"{REMOTE_LASH_HOME}/config.json"

BENCHMARK_GUIDELINES_APPEND = """## Benchmark Constraints

- You are being graded by exact verifier checks, not by partial progress.
- Do exactly what the task asks. Match required filenames, file contents, output formats, ports, protocols, process state, and side effects exactly.
- Do not stop at an approximate solution. If the task asks for a concrete final state, keep going until that exact state exists.
- Treat extra files, leftover build products, debug artifacts, temporary scripts, and stray outputs as failures unless the task explicitly requires them.
- Before finishing, remove temporary/debug artifacts that are not part of the required final state.
- Before finishing, re-read the task and verify each concrete requirement against the current environment.
- If the task implies that a service, VM, server, or port must be reachable, verify it yourself before stopping.
- Prefer direct verification over assumption. Re-open files, re-run checks, and inspect the exact final outputs before returning.
- Optimize for correctness and task completion, not for narration.
"""


class LashAgent(BaseInstalledAgent):
    @staticmethod
    def name() -> str:
        return "lash"

    @property
    def _install_agent_template_path(self) -> Path:
        return Path(__file__).resolve().parent / "install-lash.sh.j2"

    async def setup(self, environment: BaseEnvironment) -> None:
        await environment.exec(
            command=f"mkdir -p /installed-agent/libs {REMOTE_HOME} {REMOTE_LASH_HOME}"
        )

        # Optional host-provided libs are disabled by default because they may
        # be ABI-incompatible with task images (e.g. older glibc in benchmark containers).
        use_optional_libs = os.environ.get("LASH_BENCH_USE_OPTIONAL_LIBS") == "1"
        self._use_optional_libs = use_optional_libs

        binary_path = Path(os.environ.get("LASH_BENCH_BINARY", str(DEFAULT_LASH_BINARY)))
        if not binary_path.exists():
            raise FileNotFoundError(
                f"Expected lash binary at {binary_path}. Build it before running Harbor."
            )

        await environment.upload_file(
            source_path=str(binary_path),
            target_path="/installed-agent/lash",
        )

        if use_optional_libs and OPTIONAL_LIBS_DIR.exists():
            for lib in OPTIONAL_LIBS_DIR.iterdir():
                if lib.is_file():
                    await environment.upload_file(
                        source_path=str(lib),
                        target_path=f"/installed-agent/libs/{lib.name}",
                    )

        if HOST_LASH_CONFIG.exists():
            await environment.upload_file(
                source_path=str(HOST_LASH_CONFIG),
                target_path=REMOTE_LASH_CONFIG,
            )
        else:
            self.logger.warning(
                "No local lash config found at %s; run may require env-based provider auth.",
                HOST_LASH_CONFIG,
            )

        await super().setup(environment)

    def create_run_agent_commands(self, instruction: str) -> list[ExecInput]:
        execution_mode = os.environ.get("LASH_BENCH_EXECUTION_MODE", "").strip()
        if execution_mode not in {"repl", "native-tools"}:
            raise ValueError(
                "LASH_BENCH_EXECUTION_MODE must be set to 'repl' or 'native-tools'"
            )

        env: dict[str, str] = {
            "HOME": REMOTE_HOME,
            "LASH_HOME": REMOTE_LASH_HOME,
            # Bench tasks can involve long thinking phases with sparse stream chunks.
            # Use a higher default than interactive runs; allow override from host env.
            "LASH_LLM_STREAM_TIMEOUT_SECS": os.environ.get(
                "LASH_LLM_STREAM_TIMEOUT_SECS", "300"
            ),
        }

        if getattr(self, "_use_optional_libs", False):
            env["LD_LIBRARY_PATH"] = "/installed-agent/libs"

        for key in (
            "OPENROUTER_API_KEY",
            "ANTHROPIC_API_KEY",
            "TAVILY_API_KEY",
            "LASH_LOG",
            "LASH_ALLOW_UNKNOWN_MODELS",
            "LASH_LLM_STREAM_TIMEOUT_SECS",
        ):
            value = os.environ.get(key, "")
            if value:
                env[key] = value

        provider_flag = "--provider " if os.environ.get("LASH_PROVIDER_SETUP") == "1" else ""
        model_flag = (
            f"--model {shlex.quote(self.model_name)} " if self.model_name else ""
        )
        execution_mode_flag = f"--execution-mode {shlex.quote(execution_mode)} "
        prompt_flags = ""
        for env_key, section in (
            ("LASH_PROMPT_REPLACE_IDENTITY", "identity"),
            ("LASH_PROMPT_REPLACE_GUIDELINES", "guidelines"),
            ("LASH_PROMPT_REPLACE_TOOL_GUIDES", "tool_guides"),
        ):
            value = os.environ.get(env_key)
            if value:
                prompt_flags += (
                    f"--prompt-replace {shlex.quote(f'{section}={value}')} "
                )

        benchmark_guidelines = os.environ.get(
            "LASH_BENCH_PROMPT_APPEND_GUIDELINES", BENCHMARK_GUIDELINES_APPEND
        )
        if benchmark_guidelines.strip():
            prompt_flags += (
                f"--prompt-append {shlex.quote(f'guidelines={benchmark_guidelines}')} "
            )

        disable_sections = os.environ.get("LASH_PROMPT_DISABLE", "").strip()
        if disable_sections:
            for section in disable_sections.split(","):
                sec = section.strip()
                if sec:
                    prompt_flags += f"--prompt-disable {shlex.quote(sec)} "
        prompt = shlex.quote(instruction)

        return [
            ExecInput(
                command=(
                    f"lash {provider_flag}{model_flag}{execution_mode_flag}"
                    f"{prompt_flags}--print {prompt}"
                ),
                env=env,
                timeout_sec=None,
            )
        ]

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        rendered_instruction = (
            render_prompt_template(self._prompt_template_path, instruction)
            if self._prompt_template_path
            else instruction
        )
        for i, exec_input in enumerate(self.create_run_agent_commands(rendered_instruction)):
            command_dir = self.logs_dir / f"command-{i}"
            command_dir.mkdir(parents=True, exist_ok=True)
            (command_dir / "command.txt").write_text(exec_input.command)

            result = await environment.exec(
                command=exec_input.command,
                cwd=exec_input.cwd,
                env=exec_input.env,
                timeout_sec=exec_input.timeout_sec,
            )

            (command_dir / "return-code.txt").write_text(str(result.return_code))
            if result.stdout:
                (command_dir / "stdout.txt").write_text(result.stdout)
            if result.stderr:
                (command_dir / "stderr.txt").write_text(result.stderr)
        self.populate_context_post_run(context)

    def populate_context_post_run(self, context: AgentContext) -> None:
        sessions_dir = self.logs_dir / "lash-home" / "sessions"
        if not sessions_dir.exists():
            return

        n_input_tokens = 0
        n_output_tokens = 0
        n_cache_tokens = 0
        saw_usage = False

        for path in sorted(sessions_dir.glob("*.llm.jsonl")):
            try:
                with path.open() as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        record = json.loads(line)
                        usage = record.get("usage")
                        if not isinstance(usage, dict):
                            continue
                        n_input_tokens += int(usage.get("input_tokens") or 0)
                        n_output_tokens += int(usage.get("output_tokens") or 0)
                        n_cache_tokens += int(usage.get("cached_input_tokens") or 0)
                        saw_usage = True
            except Exception as exc:  # pragma: no cover - defensive, non-fatal
                self.logger.warning("Failed to parse lash usage from %s: %s", path, exc)

        if saw_usage:
            context.n_input_tokens = n_input_tokens
            context.n_output_tokens = n_output_tokens
            context.n_cache_tokens = n_cache_tokens
