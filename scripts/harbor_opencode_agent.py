"""Harbor adapter for running OpenCode headlessly inside benchmark environments."""

from __future__ import annotations

import os
import shlex
import json
from pathlib import Path

from harbor.agents.installed.base import ExecInput
from harbor.agents.installed.opencode import OpenCode
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.utils.templating import render_prompt_template

DEFAULT_OPENCODE_BINARY = Path("/usr/bin/opencode")
HOST_OPENCODE_AUTH = Path.home() / ".local" / "share" / "opencode" / "auth.json"
HOST_OPENCODE_CONFIG = Path.home() / ".config" / "opencode" / "opencode.json"
HOST_CA_CERT_BUNDLE = Path("/etc/ssl/certs/ca-certificates.crt")
REMOTE_OPENCODE_AUTH = "/root/.local/share/opencode/auth.json"
REMOTE_OPENCODE_CONFIG = "/root/.config/opencode/opencode.json"
REMOTE_OPENCODE_BINARY = "/installed-agent/opencode"
REMOTE_CA_CERT_DIR = "/etc/ssl/certs"
REMOTE_CA_CERT_BUNDLE = f"{REMOTE_CA_CERT_DIR}/ca-certificates.crt"
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


class BenchOpenCodeAgent(OpenCode):
    @staticmethod
    def _command_metadata(command: str) -> dict[str, str]:
        normalized = command.strip()
        if "opencode" in normalized and " run" in normalized:
            return {
                "phase": "main",
                "purpose": "agent_run",
                "family": "opencode",
                "is_main": "true",
            }
        if "opencode.json" in normalized:
            return {
                "phase": "bootstrap",
                "purpose": "config",
                "family": "opencode",
                "is_main": "false",
            }
        return {
            "phase": "bootstrap",
            "purpose": "setup",
            "family": "opencode",
            "is_main": "false",
        }

    def _build_register_skills_command(self) -> str | None:
        skills_dir = getattr(self, "skills_dir", None)
        if not skills_dir:
            return None
        return (
            f"mkdir -p /root/.config/opencode/skills && "
            f"cp -r {shlex.quote(str(skills_dir))}/* "
            "/root/.config/opencode/skills/ 2>/dev/null || true"
        )

    def _build_register_config_command(self) -> str | None:
        config: dict[str, object] = {
            "autoupdate": False,
            "formatter": False,
            # Benchmark containers are already isolated. Avoid interactive permission
            # gates or external-directory denials skewing task outcomes.
            "permission": "allow",
        }

        mcp_servers = getattr(self, "mcp_servers", None) or []
        if mcp_servers:
            mcp: dict[str, dict[str, object]] = {}
            for server in mcp_servers:
                if getattr(server, "transport", None) == "stdio":
                    cmd_list = [server.command] + server.args if server.command else []
                    mcp[server.name] = {"type": "local", "command": cmd_list}
                else:
                    mcp[server.name] = {"type": "remote", "url": server.url}
            config["mcp"] = mcp

        if self.model_name and "/" in self.model_name:
            provider, model_id = self.model_name.split("/", 1)
            config["provider"] = {provider: {"models": {model_id: {}}}}

        if not config:
            return None

        config_json = json.dumps(config, indent=2)
        escaped = shlex.quote(config_json)
        return (
            "mkdir -p /root/.config/opencode && "
            f"echo {escaped} > {shlex.quote(REMOTE_OPENCODE_CONFIG)}"
        )

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

        if not self.model_name or "/" not in self.model_name:
            raise ValueError("Model name must be in the format provider/model_name")

        provider, _ = self.model_name.split("/", 1)

        env: dict[str, str] = {}
        keys: list[str] = []

        if provider == "amazon-bedrock":
            keys.extend(["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"])
        elif provider == "anthropic":
            keys.append("ANTHROPIC_API_KEY")
        elif provider == "azure":
            keys.extend(["AZURE_RESOURCE_NAME", "AZURE_API_KEY"])
        elif provider == "deepseek":
            keys.append("DEEPSEEK_API_KEY")
        elif provider == "github-copilot":
            keys.append("GITHUB_TOKEN")
        elif provider == "google":
            keys.extend(
                [
                    "GEMINI_API_KEY",
                    "GOOGLE_GENERATIVE_AI_API_KEY",
                    "GOOGLE_APPLICATION_CREDENTIALS",
                    "GOOGLE_CLOUD_PROJECT",
                    "GOOGLE_CLOUD_LOCATION",
                    "GOOGLE_GENAI_USE_VERTEXAI",
                    "GOOGLE_API_KEY",
                ]
            )
        elif provider == "groq":
            keys.append("GROQ_API_KEY")
        elif provider == "huggingface":
            keys.append("HF_TOKEN")
        elif provider == "llama":
            keys.append("LLAMA_API_KEY")
        elif provider == "mistral":
            keys.append("MISTRAL_API_KEY")
        elif provider == "openai":
            keys.append("OPENAI_API_KEY")
        elif provider == "openrouter":
            keys.append("OPENROUTER_API_KEY")
        elif provider == "xai":
            keys.append("XAI_API_KEY")
        else:
            raise ValueError(
                f"Unknown provider {provider}. If you believe this provider "
                "should be supported, please contact the maintainers."
            )

        for key in keys:
            value = os.environ.get(key)
            if value:
                env[key] = value

        env["OPENCODE_FAKE_VCS"] = "git"
        env["SSL_CERT_FILE"] = REMOTE_CA_CERT_BUNDLE
        env["CURL_CA_BUNDLE"] = REMOTE_CA_CERT_BUNDLE
        env["REQUESTS_CA_BUNDLE"] = REMOTE_CA_CERT_BUNDLE
        env["NODE_EXTRA_CA_CERTS"] = REMOTE_CA_CERT_BUNDLE

        commands: list[ExecInput] = []

        skills_command = self._build_register_skills_command()
        if skills_command:
            commands.append(ExecInput(command=skills_command, env=env))

        config_command = self._build_register_config_command()
        if config_command:
            commands.append(ExecInput(command=config_command, env=env))

        variant = os.environ.get("OPENCODE_BENCH_MODEL_VARIANT", "").strip()
        variant_flag = f" --variant={shlex.quote(variant)}" if variant else ""

        commands.append(
            ExecInput(
                command=(
                    f"chmod +x {shlex.quote(REMOTE_OPENCODE_BINARY)} && "
                    f"{shlex.quote(REMOTE_OPENCODE_BINARY)} "
                    f"--agent=build --model={self.model_name} "
                    f"run{variant_flag} --format=json -- "
                    f"{escaped_instruction} 2>&1 </dev/null | stdbuf -oL tee /logs/agent/opencode.txt"
                ),
                env=env,
            )
        )

        return commands

    async def setup(self, environment: BaseEnvironment) -> None:
        await environment.exec(
            command=(
                "mkdir -p /installed-agent /root/.local/share/opencode "
                f"/root/.config/opencode {REMOTE_CA_CERT_DIR}"
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
            os.environ.get("OPENCODE_BENCH_BINARY", str(DEFAULT_OPENCODE_BINARY))
        )
        if not binary_path.exists():
            raise FileNotFoundError(
                f"Expected opencode binary at {binary_path}. Install it before running Harbor."
            )

        await environment.upload_file(
            source_path=str(binary_path),
            target_path=REMOTE_OPENCODE_BINARY,
        )

        if HOST_OPENCODE_AUTH.exists():
            await environment.upload_file(
                source_path=str(HOST_OPENCODE_AUTH),
                target_path=REMOTE_OPENCODE_AUTH,
            )
        else:
            self.logger.warning(
                "No local OpenCode auth found at %s; run may require provider env vars.",
                HOST_OPENCODE_AUTH,
            )

        if HOST_OPENCODE_CONFIG.exists():
            await environment.upload_file(
                source_path=str(HOST_OPENCODE_CONFIG),
                target_path=REMOTE_OPENCODE_CONFIG,
            )

        setup_dir = self.logs_dir / "setup"
        setup_dir.mkdir(parents=True, exist_ok=True)
        (setup_dir / "return-code.txt").write_text("0")
        (setup_dir / "stdout.txt").write_text(
            "Skipped Harbor OpenCode installer; using uploaded local opencode binary.\n"
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
            except Exception:  # pragma: no cover - best effort
                self.logger.debug(
                    "Failed to download resource usage for command-%s", i, exc_info=True
                )

        self.populate_context_post_run(context)
