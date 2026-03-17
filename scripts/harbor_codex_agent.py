"""Harbor adapter for running Codex CLI headlessly inside benchmark environments."""

from __future__ import annotations

import json
import os
import shlex
from collections import defaultdict
from pathlib import Path
from typing import Any

from harbor.agents.installed.base import BaseInstalledAgent, ExecInput
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.utils.templating import render_prompt_template

def _find_codex_binary() -> Path:
    """Locate the codex binary on the host, checking common locations."""
    import shutil

    found = shutil.which("codex")
    if found:
        return Path(found)
    return Path("/usr/bin/codex")


DEFAULT_CODEX_BINARY = _find_codex_binary()
HOST_CA_CERT_BUNDLE = Path("/etc/ssl/certs/ca-certificates.crt")
REMOTE_CA_CERT_DIR = "/etc/ssl/certs"
REMOTE_CA_CERT_BUNDLE = f"{REMOTE_CA_CERT_DIR}/ca-certificates.crt"
REMOTE_CODEX_BINARY = "/installed-agent/codex"
REMOTE_CODEX_CONFIG_DIR = "/root/.codex"

INSTALL_GNU_TIME_COMMAND = """
if [ ! -x /usr/bin/time ]; then
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update && apt-get install -y time
  elif command -v apk >/dev/null 2>&1; then
    apk add --no-cache time
  elif command -v dnf >/dev/null 2>&1; then
    dnf install -y time
  elif command -v yum >/dev/null 2>&1; then
    yum install -y time
  elif command -v microdnf >/dev/null 2>&1; then
    microdnf install -y time
  elif command -v zypper >/dev/null 2>&1; then
    zypper --non-interactive install time
  fi
fi
"""


def load_codex_metadata(codex_path: Path | None) -> dict[str, Any]:
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

        # Count LLM responses
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

        # Count tool calls from function_call or tool_use events
        if event_type in ("function_call", "tool_use", "exec"):
            tool_call_count += 1
            tool_name = record.get("name") or record.get("function", {}).get("name") or "unknown"
            tool_call_breakdown[tool_name] += 1

        # Capture assistant text
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


class BenchCodexAgent(BaseInstalledAgent):
    @staticmethod
    def name() -> str:
        return "codex"

    @property
    def _install_agent_template_path(self) -> Path:
        # Codex uses a pre-built binary; no install template needed.
        return Path(__file__).resolve().parent / "install-codex.sh.j2"

    @staticmethod
    def _command_metadata(command: str) -> dict[str, str]:
        normalized = command.strip()
        if "codex" in normalized and "exec" in normalized:
            return {
                "phase": "main",
                "purpose": "agent_run",
                "family": "codex",
                "is_main": "true",
            }
        return {
            "phase": "bootstrap",
            "purpose": "setup",
            "family": "codex",
            "is_main": "false",
        }

    @staticmethod
    def _timed_command(command: str, index: int) -> str:
        output_path = f"/logs/agent/command-{index}/resource-usage.txt"
        escaped_command = shlex.quote(f"set -o pipefail; {command}")
        return (
            f"if [ -x /usr/bin/time ]; then "
            f"mkdir -p /logs/agent/command-{index} && "
            f"/usr/bin/time -v -o {shlex.quote(output_path)} bash -lc {escaped_command}; "
            f"else bash -lc {escaped_command}; fi"
        )

    def create_run_agent_commands(self, instruction: str) -> list[ExecInput]:
        escaped_instruction = shlex.quote(instruction)

        env: dict[str, str] = {}

        # Codex uses OPENAI_API_KEY by default
        for key in (
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
        ):
            value = os.environ.get(key)
            if value:
                env[key] = value

        env["SSL_CERT_FILE"] = REMOTE_CA_CERT_BUNDLE
        env["CURL_CA_BUNDLE"] = REMOTE_CA_CERT_BUNDLE
        env["REQUESTS_CA_BUNDLE"] = REMOTE_CA_CERT_BUNDLE
        env["NODE_EXTRA_CA_CERTS"] = REMOTE_CA_CERT_BUNDLE

        model_flag = f" -m {shlex.quote(self.model_name)}" if self.model_name else ""

        commands: list[ExecInput] = []

        commands.append(
            ExecInput(
                command=(
                    f"chmod +x {shlex.quote(REMOTE_CODEX_BINARY)} && "
                    f"{shlex.quote(REMOTE_CODEX_BINARY)} exec"
                    f"{model_flag}"
                    f" --dangerously-bypass-approvals-and-sandbox"
                    f" --skip-git-repo-check"
                    f" --json"
                    f" {escaped_instruction}"
                    f" 2>&1 </dev/null | stdbuf -oL tee /logs/agent/codex.txt"
                ),
                env=env,
            )
        )

        return commands

    async def setup(self, environment: BaseEnvironment) -> None:
        await environment.exec(
            command=(
                f"mkdir -p /installed-agent {REMOTE_CODEX_CONFIG_DIR} {REMOTE_CA_CERT_DIR}"
            )
        )

        if HOST_CA_CERT_BUNDLE.exists():
            await environment.upload_file(
                source_path=str(HOST_CA_CERT_BUNDLE.resolve()),
                target_path=REMOTE_CA_CERT_BUNDLE,
            )
            await environment.exec(
                command=(
                    "if command -v update-ca-certificates >/dev/null 2>&1; then "
                    "update-ca-certificates >/dev/null 2>&1 || true; "
                    "fi"
                )
            )
        else:
            self.logger.warning(
                "No host CA bundle found at %s; benchmark containers may fail TLS checks.",
                HOST_CA_CERT_BUNDLE,
            )

        await environment.exec(command=INSTALL_GNU_TIME_COMMAND)

        binary_path = Path(
            os.environ.get("CODEX_BENCH_BINARY", str(DEFAULT_CODEX_BINARY))
        )
        if not binary_path.exists():
            raise FileNotFoundError(
                f"Expected codex binary at {binary_path}. Install it before running Harbor."
            )

        await environment.upload_file(
            source_path=str(binary_path),
            target_path=REMOTE_CODEX_BINARY,
        )

        # Upload codex config if it exists
        host_codex_config = Path.home() / ".codex" / "config.toml"
        if host_codex_config.exists():
            await environment.upload_file(
                source_path=str(host_codex_config),
                target_path=f"{REMOTE_CODEX_CONFIG_DIR}/config.toml",
            )

        setup_dir = self.logs_dir / "setup"
        setup_dir.mkdir(parents=True, exist_ok=True)
        (setup_dir / "return-code.txt").write_text("0")
        (setup_dir / "stdout.txt").write_text(
            "Skipped Harbor Codex installer; using uploaded local codex binary.\n"
        )

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
            (command_dir / "metadata.json").write_text(
                json.dumps(self._command_metadata(exec_input.command), indent=2) + "\n"
            )

            env = exec_input.env
            extra_env = getattr(self, "_extra_env", None)
            if extra_env:
                env = dict(exec_input.env) if exec_input.env else {}
                env.update(extra_env)

            result = await environment.exec(
                command=self._timed_command(exec_input.command, i),
                cwd=exec_input.cwd,
                env=env,
                timeout_sec=exec_input.timeout_sec,
            )

            (command_dir / "return-code.txt").write_text(str(result.return_code))

            if result.stdout:
                (command_dir / "stdout.txt").write_text(result.stdout)

            if result.stderr:
                (command_dir / "stderr.txt").write_text(result.stderr)

            try:
                await environment.download_file(
                    source_path=f"/logs/agent/command-{i}/resource-usage.txt",
                    target_path=command_dir / "resource-usage.txt",
                )
            except Exception:
                self.logger.debug(
                    "Failed to download resource usage for command-%s", i, exc_info=True
                )

        self.populate_context_post_run(context)

    def populate_context_post_run(self, context: AgentContext) -> None:
        codex_output = self.logs_dir / "command-0" / "stdout.txt"
        metadata = load_codex_metadata(codex_output if codex_output.exists() else None)

        tokens = metadata["tokens"]
        if tokens["input"] or tokens["output"]:
            context.n_input_tokens = tokens["input"]
            context.n_output_tokens = tokens["output"]
            context.n_cache_tokens = tokens.get("cache_read", 0)

        assistant_response = metadata.get("assistant_response")
        if assistant_response:
            ctx_metadata = dict(context.metadata or {})
            ctx_metadata["assistant_response"] = assistant_response
            context.metadata = ctx_metadata
