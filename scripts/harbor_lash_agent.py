"""Harbor adapter for running lash headlessly inside benchmark environments."""

from __future__ import annotations

import os
import shlex
from pathlib import Path

from harbor.agents.installed.base import BaseInstalledAgent, ExecInput
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LASH_BINARY = REPO_ROOT / "target" / "release" / "lash"
OPTIONAL_LIBS_DIR = REPO_ROOT / "bench" / "libs"
HOST_LASH_CONFIG = Path.home() / ".lash" / "config.json"

REMOTE_HOME = "/installed-agent/home"
REMOTE_LASH_CONFIG = f"{REMOTE_HOME}/.lash/config.json"


class LashAgent(BaseInstalledAgent):
    @staticmethod
    def name() -> str:
        return "lash"

    @property
    def _install_agent_template_path(self) -> Path:
        return Path(__file__).resolve().parent / "install-lash.sh.j2"

    async def setup(self, environment: BaseEnvironment) -> None:
        await environment.exec(
            command=f"mkdir -p /installed-agent/libs {REMOTE_HOME}/.lash"
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
        env: dict[str, str] = {
            "HOME": REMOTE_HOME,
        }

        if getattr(self, "_use_optional_libs", False):
            env["LD_LIBRARY_PATH"] = "/installed-agent/libs"

        for key in (
            "OPENROUTER_API_KEY",
            "ANTHROPIC_API_KEY",
            "TAVILY_API_KEY",
            "LASH_LOG",
            "LASH_ALLOW_UNKNOWN_MODELS",
            "LASH_PREAMBLE",
            "LASH_SOUL",
        ):
            value = os.environ.get(key, "")
            if value:
                env[key] = value

        provider_flag = "--provider " if os.environ.get("LASH_PROVIDER_SETUP") == "1" else ""
        model_flag = (
            f"--model {shlex.quote(self.model_name)} " if self.model_name else ""
        )
        prompt = shlex.quote(instruction)

        return [
            ExecInput(
                command=f"lash {provider_flag}{model_flag}--print {prompt}",
                env=env,
                timeout_sec=600,
            )
        ]

    def populate_context_post_run(self, context: AgentContext) -> None:
        pass
