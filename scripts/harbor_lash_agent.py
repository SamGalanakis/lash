"""Harbor adapter for running lash headlessly inside benchmark environments."""

from __future__ import annotations

import os
import shlex
import json
import tomllib
from pathlib import Path

from harbor.agents.installed.base import BaseInstalledAgent, ExecInput
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths
from harbor.utils.templating import render_prompt_template

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LASH_BINARY_CANDIDATES = (
    REPO_ROOT / "target" / "release" / "lash",
    REPO_ROOT / "target-bookworm" / "release" / "lash",
    REPO_ROOT / "target-bullseye" / "release" / "lash",
)
HOST_LASH_CONFIG = Path(
    os.environ.get("LASH_BENCH_CONFIG", str(Path.home() / ".lash" / "config.json"))
).expanduser()
HOST_CA_CERT_BUNDLE = Path("/etc/ssl/certs/ca-certificates.crt")

REMOTE_HOME = "/installed-agent/home"
REMOTE_LASH_HOME = (EnvironmentPaths.agent_dir / "lash-home").as_posix()
REMOTE_LASH_CONFIG = f"{REMOTE_LASH_HOME}/config.json"
REMOTE_CA_CERT_DIR = "/etc/ssl/certs"
REMOTE_CA_CERT_BUNDLE = f"{REMOTE_CA_CERT_DIR}/ca-certificates.crt"

BENCHMARK_GUIDELINES_APPEND = """## What "finishing" means for this task

Most terminal-bench tasks are graded by inspecting the environment
after you stop — files, services, running processes, configuration
state. For these tasks:

- Make the required changes directly to the filesystem / services.
- Verify the end state (re-open files, re-run the service's own check,
  probe the port, read the expected output).
- Submit the final result with `submit <expr>` once you've confirmed the
  required end state.

Only `submit <value>` when the task explicitly asks you to return a
computed value (e.g. "print the total count"). Never submit
exploration output (file listings, version strings, tree dumps) as if
it were the answer — the grader is not reading your response.

## Strict verifier rules

- You are graded by exact checks. Match required filenames, file
  contents, output formats, ports, protocols, and process state
  precisely. Approximate solutions fail.
- If the task implies a service or port must be reachable, confirm it
  yourself (curl / nc / the service's own healthcheck) before stopping.
- Prefer direct verification over assumption. Re-open the exact file
  and check the exact bytes; re-run the check the task describes.
- Hidden verifier tests may check details not shown in the task. Do not
  treat `test -f`, `ls`, `sed`, or a successful command exit as enough
  verification. Your final check should assert the task-specific
  invariants: exact counts, exact JSON values, imports from the installed
  location, service behavior, permissions, and process state as applicable.
- For nontrivial tasks, do not call `submit` in the same lashlang block
  as the command that performs your final verification. Print the
  verification result first, inspect it on the next iteration, then submit.
  The exception is a verification script that exits nonzero on every
  task-specific mismatch and checks the exact required values, not just
  file existence or command success.
- Leave the workspace in exactly the state the task asks for. Keep
  required build outputs, installed packages, generated files, services,
  and other artifacts when the task asks for them. Do not leave unrelated
  scratch files, downloaded archives, backup copies, or exploratory output
  in directories the task names. Use `/tmp` for intermediate artifacts and
  clean them up before submitting.
- For recovery or forensics tasks, copy the original evidence to `/tmp`
  before opening it with tools that may mutate, checkpoint, normalize, or
  delete sidecar files.

## Working efficiently across iterations

- If a setup, build, or verifier script is longer than ~10 lines and
  you might need to retry it, save it to disk once
  (`cat > /tmp/verify.sh && chmod +x /tmp/verify.sh`) and re-run on
  subsequent iterations rather than re-emitting the entire script.
- The `?` operator on `exec_command` aborts the lashlang block on any
  non-zero exit (including SIGPIPE 141 from `cmd | head`-style
  pipelines and timeouts). When you expect a non-zero exit is fine, set
  `allow_nonzero_exit: true` on the `exec_command` call so you can
  inspect the result and decide what to do.
- Any string containing backslashes or backticks (heredocs, regexes,
  file contents, code excerpts) should use a raw string
  (`r\"\"\"…\"\"\"` or `r'''…'''`) — over-escaped `\\\\\"` in plain
  `\"…\"` strings is a frequent cause of lashlang parse errors. Pick
  the delimiter the body does *not* contain: if your content has `\"\"\"`
  (e.g. a Python docstring), use `r'''…'''`; if it has `'''`, use
  `r\"\"\"…\"\"\"`. The first matching closer ends the raw string, so a
  collision terminates it early. If your lashlang body itself needs to
  contain literal triple-backticks (e.g. embedding markdown), open the
  fence with four backticks (` ````lashlang `) and close with the same.
"""

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


class LashAgent(BaseInstalledAgent):
    @staticmethod
    def _default_binary_path() -> Path:
        existing = [candidate for candidate in DEFAULT_LASH_BINARY_CANDIDATES if candidate.exists()]
        if existing:
            return max(existing, key=lambda path: path.stat().st_mtime_ns)
        return DEFAULT_LASH_BINARY_CANDIDATES[0]

    @staticmethod
    def _command_metadata(command: str) -> dict[str, str]:
        return {
            "phase": "main",
            "purpose": "agent_run",
            "family": "lash",
            "is_main": "true",
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

    def version(self) -> str | None:
        version = super().version()
        if version is not None:
            return version

        cargo_toml = REPO_ROOT / "Cargo.toml"
        try:
            with cargo_toml.open("rb") as f:
                workspace = tomllib.load(f).get("workspace", {})
        except (FileNotFoundError, tomllib.TOMLDecodeError):
            return None

        package = workspace.get("package")
        if isinstance(package, dict):
            version_value = package.get("version")
            if isinstance(version_value, str):
                return version_value
        return None

    @staticmethod
    def name() -> str:
        return "lash"

    @property
    def _install_agent_template_path(self) -> Path:
        return REPO_ROOT / "bench" / "terminalbench2" / "install-lash.sh.j2"

    async def setup(self, environment: BaseEnvironment) -> None:
        await environment.exec(
            command=f"mkdir -p {REMOTE_HOME} {REMOTE_LASH_HOME} {REMOTE_CA_CERT_DIR}"
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
            os.environ.get("LASH_BENCH_BINARY", str(self._default_binary_path()))
        )
        if not binary_path.exists():
            raise FileNotFoundError(
                f"Expected lash binary at {binary_path}. Build it before running Harbor."
            )

        await environment.upload_file(
            source_path=str(binary_path),
            target_path="/installed-agent/lash",
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
        if execution_mode not in {"rlm", "standard"}:
            raise ValueError(
                "LASH_BENCH_EXECUTION_MODE must be set to 'rlm' or 'standard'"
            )

        env: dict[str, str] = {
            "HOME": REMOTE_HOME,
            "LASH_HOME": REMOTE_LASH_HOME,
            "SSL_CERT_FILE": REMOTE_CA_CERT_BUNDLE,
            "CURL_CA_BUNDLE": REMOTE_CA_CERT_BUNDLE,
            "REQUESTS_CA_BUNDLE": REMOTE_CA_CERT_BUNDLE,
            "NODE_EXTRA_CA_CERTS": REMOTE_CA_CERT_BUNDLE,
            # Bench tasks can involve long thinking phases with sparse stream chunks.
            # Use a higher default than interactive runs; allow override from host env.
            "LASH_LLM_STREAM_TIMEOUT_SECS": os.environ.get(
                "LASH_LLM_STREAM_TIMEOUT_SECS", "300"
            ),
        }

        for key in (
            "OPENROUTER_API_KEY",
            "ANTHROPIC_API_KEY",
            "TAVILY_API_KEY",
            "LASH_LOG",
            "LASH_ALLOW_UNKNOWN_MODELS",
            "LASH_LLM_STREAM_TIMEOUT_SECS",
            "LASH_AUTONOMOUS_SETTLE_MS",
        ):
            value = os.environ.get(key, "")
            if value:
                env[key] = value

        provider_flag = "--provider " if os.environ.get("LASH_PROVIDER_SETUP") == "1" else ""
        model_flag = (
            f"--model {shlex.quote(self.model_name)} " if self.model_name else ""
        )
        variant = os.environ.get("LASH_BENCH_MODEL_VARIANT", "").strip()
        variant_flag = f"--variant {shlex.quote(variant)} " if variant else ""
        context_approach = os.environ.get("LASH_BENCH_CONTEXT_APPROACH", "").strip()
        if context_approach and execution_mode != "standard":
            raise ValueError(
                "LASH_BENCH_CONTEXT_APPROACH only applies to standard execution mode"
            )
        context_approach_flag = (
            f"--context-approach {shlex.quote(context_approach)} "
            if context_approach
            else ""
        )
        execution_mode_flag = f"--execution-mode {shlex.quote(execution_mode)} "
        # The old `--prompt-replace` / `--prompt-append` /
        # `--prompt-disable` flags were removed from the lash CLI.
        # Instead, we own one benchmark-specific guidance block
        # (`BENCHMARK_GUIDELINES_APPEND`) and fold it into the user
        # prompt. The block covers the terminal-bench task shape
        # (environment-state, not submit-value) and strict verifier
        # rules. `LASH_BENCH_PROMPT_APPEND_GUIDELINES` overrides the
        # default for ad-hoc runs.
        bench_guidelines = os.environ.get(
            "LASH_BENCH_PROMPT_APPEND_GUIDELINES", BENCHMARK_GUIDELINES_APPEND
        ).strip()
        augmented_instruction = (
            f"{instruction}\n\n{bench_guidelines}" if bench_guidelines else instruction
        )
        prompt = shlex.quote(augmented_instruction)

        lash_binary = "/installed-agent/lash"

        return [
            ExecInput(
                command=(
                    f"chmod +x {shlex.quote(lash_binary)} && "
                    f"{shlex.quote(lash_binary)} {provider_flag}{model_flag}{variant_flag}"
                    f"{context_approach_flag}{execution_mode_flag}"
                    f"--print {prompt}"
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
            (command_dir / "metadata.json").write_text(
                json.dumps(self._command_metadata(exec_input.command), indent=2) + "\n"
            )

            result = await environment.exec(
                command=self._timed_command(exec_input.command, i),
                cwd=exec_input.cwd,
                env=exec_input.env,
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

    def _read_assistant_response(self) -> str | None:
        outputs: list[str] = []
        for path in sorted(self.logs_dir.glob("command-*/stdout.txt")):
            try:
                text = path.read_text().strip()
            except Exception as exc:  # pragma: no cover - defensive, non-fatal
                self.logger.warning("Failed to read lash stdout from %s: %s", path, exc)
                continue
            if text:
                outputs.append(text)
        if not outputs:
            return None
        return "\n\n".join(outputs)

    def populate_context_post_run(self, context: AgentContext) -> None:
        sessions_dir = self.logs_dir / "lash-home" / "sessions"
        n_input_tokens = 0
        n_output_tokens = 0
        n_cache_tokens = 0
        saw_usage = False

        if sessions_dir.exists():
            for path in sorted(sessions_dir.glob("*.trace.jsonl")):
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

        assistant_response = self._read_assistant_response()
        if assistant_response:
            metadata = dict(context.metadata or {})
            metadata["assistant_response"] = assistant_response
            context.metadata = metadata
