#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import profile_guard  # noqa: E402
import profile_cli_release_stack  # noqa: E402
import profile_cli_stack  # noqa: E402
import profile_lashlang  # noqa: E402
import profile_runtime  # noqa: E402
import profile_runtime_stack  # noqa: E402
import profile_ui  # noqa: E402


def stack_profile(
    stack_bytes: int = 2 * 1024 * 1024,
    stack_budget_bytes: int = 2 * 1024 * 1024,
) -> dict[str, object]:
    return {
        "worker_stack_bytes": None,
        "rust_min_stack_bytes": None,
        "process_stack_soft_limit_bytes": stack_bytes,
        "process_stack_hard_limit_bytes": None,
        "process_stack_hard_limit_unlimited": True,
        "measured_stack_bytes": stack_bytes,
        "measured_stack_source": "process_stack_soft_limit",
        "stack_budget_bytes": stack_budget_bytes,
        "within_stack_budget": stack_bytes <= stack_budget_bytes,
    }


def stack_lane_report(
    release: bool = False,
    cargo_features: list[str] | None = None,
) -> dict[str, object]:
    return {
        "binary_metadata": {"sha256": "abc123"},
        "cargo_features": cargo_features or [],
        "git": {"sha": "deadbeef", "dirty": False},
        "release": release,
    }


class RuntimeStackProfilerTests(unittest.TestCase):
    def run_fake_sample(
        self,
        mode: str,
        *,
        scenario: str = "standard",
        stack_bytes: int = 128 * 1024,
    ) -> dict[str, object]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "fake_lash_perf"
            binary.write_text(
                f"""#!{sys.executable}
import json
import sys
from pathlib import Path

mode = {mode!r}
args = sys.argv[1:]

def arg_value(prefix):
    for arg in args:
        if arg.startswith(prefix):
            return arg.split("=", 1)[1]
    raise SystemExit(f"missing {{prefix}}")

scenario_index = args.index("--runtime-perf-scenario")
scenario = args[scenario_index + 1]
stack_bytes = int(arg_value("--runtime-perf-worker-stack-bytes="))
out = Path(arg_value("--runtime-perf-out="))
out.parent.mkdir(parents=True, exist_ok=True)
print(f"fake mode {{mode}}", file=sys.stderr)

if mode == "missing_report":
    raise SystemExit(0)
if mode == "malformed_report":
    out.write_text("{{not json")
    raise SystemExit(0)

reported_stack = stack_bytes + 1 if mode == "wrong_stack" else stack_bytes
reported_scenario = f"{{scenario}}_other" if mode == "missing_scenario" else scenario
if mode == "missing_stack_profile":
    out.write_text(json.dumps({{
        "worker_stack_bytes": reported_stack,
        "summary": [{{"scenario": reported_scenario}}],
    }}))
    raise SystemExit(0)
out.write_text(json.dumps({{
    "worker_stack_bytes": reported_stack,
    "stack_profile": {{
        "worker_stack_bytes": reported_stack,
        "stack_budget_bytes": 2097152,
        "within_stack_budget": reported_stack <= 2097152,
    }},
    "summary": [{{
        "scenario": reported_scenario,
        "stack_profile": {{
            "worker_stack_bytes": reported_stack,
            "stack_budget_bytes": 2097152,
            "within_stack_budget": reported_stack <= 2097152,
        }},
    }}],
    "results": [{{
        "scenario": reported_scenario,
        "stack_profile": {{
            "worker_stack_bytes": reported_stack,
            "stack_budget_bytes": 2097152,
            "within_stack_budget": reported_stack <= 2097152,
        }},
    }}],
}}))
"""
            )
            binary.chmod(0o755)
            return profile_runtime_stack.run_sample(
                root=root,
                binary=binary,
                scenario=scenario,
                stack_bytes=stack_bytes,
                out=root / "matrix.json",
                runs=1,
                warmups=0,
                turns=1,
                timeout_seconds=5,
            )

    def test_parse_stack_sizes_accepts_common_suffixes(self) -> None:
        self.assertEqual(profile_runtime_stack.parse_size("64k"), 64 * 1024)
        self.assertEqual(profile_runtime_stack.parse_size("2m"), 2 * 1024 * 1024)
        self.assertEqual(profile_runtime_stack.parse_size("1_024"), 1024)

    def test_first_success_requires_accounted_reported_scenario(self) -> None:
        samples = [
            {
                "scenario": "standard",
                "stack_bytes": 64 * 1024,
                "status": "ok",
                "stack_accounted": False,
                "summary_scenarios": ["standard"],
            },
            {
                "scenario": "standard",
                "stack_bytes": 96 * 1024,
                "status": "ok",
                "stack_accounted": True,
                "summary_scenarios": ["other"],
            },
            {
                "scenario": "standard",
                "stack_bytes": 128 * 1024,
                "status": "ok",
                "stack_accounted": True,
                "summary_scenarios": ["standard"],
            },
        ]

        self.assertEqual(
            profile_runtime_stack.first_success(samples, "standard"),
            128 * 1024,
        )

    def test_run_sample_rejects_success_without_report(self) -> None:
        sample = self.run_fake_sample("missing_report")

        self.assertEqual(sample["status"], "failed")
        self.assertEqual(sample["failure_reason"], "missing_runtime_perf_report")
        self.assertIn("fake mode missing_report", sample["stderr_tail"])

    def test_run_sample_rejects_success_with_malformed_report(self) -> None:
        sample = self.run_fake_sample("malformed_report")

        self.assertEqual(sample["status"], "failed")
        self.assertEqual(sample["failure_reason"], "malformed_runtime_perf_report")
        self.assertIn("json_error", sample)

    def test_run_sample_rejects_unaccounted_stack_report(self) -> None:
        sample = self.run_fake_sample("wrong_stack")

        self.assertEqual(sample["status"], "failed")
        self.assertIn("stack_size_not_accounted", sample["failure_reasons"])
        self.assertIs(sample["stack_accounted"], False)

    def test_run_sample_rejects_missing_scenario_report(self) -> None:
        sample = self.run_fake_sample("missing_scenario")

        self.assertEqual(sample["status"], "failed")
        self.assertIn("missing_runtime_perf_scenario", sample["failure_reasons"])

    def test_run_sample_rejects_missing_stack_profile(self) -> None:
        sample = self.run_fake_sample("missing_stack_profile")

        self.assertEqual(sample["status"], "failed")
        self.assertIn("stack_profile_size_not_accounted", sample["failure_reasons"])
        self.assertIn("summary_stack_size_not_accounted", sample["failure_reasons"])
        self.assertIn("result_stack_size_not_accounted", sample["failure_reasons"])

    def test_run_sample_accepts_accounted_scenario_report(self) -> None:
        sample = self.run_fake_sample("ok")

        self.assertEqual(sample["status"], "ok")
        self.assertEqual(sample["reported_worker_stack_bytes"], 128 * 1024)
        self.assertEqual(
            sample["reported_stack_profile"]["worker_stack_bytes"],
            128 * 1024,
        )
        self.assertIs(sample["summary_stack_accounted"], True)
        self.assertIs(sample["result_stack_accounted"], True)
        self.assertIs(sample["stack_accounted"], True)
        self.assertEqual(sample["summary_scenarios"], ["standard"])
        self.assertEqual(sample["stderr_tail"], "")


class CliStackProfilerTests(unittest.TestCase):
    def run_fake_sample(
        self,
        mode: str,
        *,
        scenario: str = "history_render",
        stack_bytes: int = 128 * 1024,
    ) -> dict[str, object]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "fake_lash"
            binary.write_text(
                f"""#!{sys.executable}
import json
import os
import sys
from pathlib import Path

mode = {mode!r}
args = sys.argv[1:]

def arg_value(prefix):
    for arg in args:
        if arg.startswith(prefix):
            return arg.split("=", 1)[1]
    raise SystemExit(f"missing {{prefix}}")

def stack_profile(stack_bytes):
    return {{
        "worker_stack_bytes": stack_bytes,
        "measured_stack_bytes": stack_bytes,
        "measured_stack_source": "tokio_worker",
        "stack_budget_bytes": 2097152,
        "within_stack_budget": stack_bytes <= 2097152,
    }}

scenario_index = args.index("--ui-perf-scenario")
scenario = args[scenario_index + 1]
stack_bytes = int(os.environ["LASH_TOKIO_STACK_BYTES"])
out = Path(arg_value("--ui-perf-out="))
out.parent.mkdir(parents=True, exist_ok=True)
print(f"fake mode {{mode}}", file=sys.stderr)

if mode == "missing_report":
    raise SystemExit(0)
if mode == "malformed_report":
    out.write_text("{{not json")
    raise SystemExit(0)

reported_stack = stack_bytes + 1 if mode == "wrong_stack" else stack_bytes
reported_scenario = f"{{scenario}}_other" if mode == "missing_scenario" else scenario
if mode == "missing_stack_profile":
    out.write_text(json.dumps({{
        "parameters": {{"scenarios": [reported_scenario]}},
        "scenarios": [{{"scenario": reported_scenario}}],
    }}))
    raise SystemExit(0)
out.write_text(json.dumps({{
    "stack_profile": stack_profile(reported_stack),
    "parameters": {{
        "scenarios": [reported_scenario],
        "stack_profile": stack_profile(reported_stack),
    }},
    "scenarios": [{{
        "scenario": reported_scenario,
        "stack_profile": stack_profile(reported_stack),
        "results": [{{"stack_profile": stack_profile(reported_stack)}}],
    }}],
}}))
"""
            )
            binary.chmod(0o755)
            return profile_cli_stack.run_sample(
                root=root,
                binary=binary,
                scenario=scenario,
                stack_bytes=stack_bytes,
                out=root / "matrix.json",
                runs=1,
                warmups=0,
                profile="quick",
                timeout_seconds=5,
            )

    def test_parse_stack_sizes_accepts_common_suffixes(self) -> None:
        self.assertEqual(profile_cli_stack.parse_size("64k"), 64 * 1024)
        self.assertEqual(profile_cli_stack.parse_size("2m"), 2 * 1024 * 1024)
        self.assertEqual(profile_cli_stack.parse_size("1_024"), 1024)

    def test_first_success_requires_accounted_reported_scenario(self) -> None:
        samples = [
            {
                "scenario": "history_render",
                "stack_bytes": 64 * 1024,
                "status": "ok",
                "stack_accounted": False,
                "reported_scenarios": ["history_render"],
            },
            {
                "scenario": "history_render",
                "stack_bytes": 96 * 1024,
                "status": "ok",
                "stack_accounted": True,
                "reported_scenarios": ["other"],
            },
            {
                "scenario": "history_render",
                "stack_bytes": 128 * 1024,
                "status": "ok",
                "stack_accounted": True,
                "reported_scenarios": ["history_render"],
            },
        ]

        self.assertEqual(
            profile_cli_stack.first_success(samples, "history_render"),
            128 * 1024,
        )

    def test_run_sample_rejects_success_without_report(self) -> None:
        sample = self.run_fake_sample("missing_report")

        self.assertEqual(sample["status"], "failed")
        self.assertEqual(sample["failure_reason"], "missing_ui_perf_report")
        self.assertIn("fake mode missing_report", sample["stderr_tail"])

    def test_run_sample_rejects_success_with_malformed_report(self) -> None:
        sample = self.run_fake_sample("malformed_report")

        self.assertEqual(sample["status"], "failed")
        self.assertEqual(sample["failure_reason"], "malformed_ui_perf_report")
        self.assertIn("json_error", sample)

    def test_run_sample_rejects_unaccounted_stack_report(self) -> None:
        sample = self.run_fake_sample("wrong_stack")

        self.assertEqual(sample["status"], "failed")
        self.assertIn("stack_size_not_accounted", sample["failure_reasons"])
        self.assertIs(sample["stack_accounted"], False)

    def test_run_sample_rejects_missing_scenario_report(self) -> None:
        sample = self.run_fake_sample("missing_scenario")

        self.assertEqual(sample["status"], "failed")
        self.assertIn("missing_ui_perf_scenario", sample["failure_reasons"])

    def test_run_sample_rejects_missing_stack_profile(self) -> None:
        sample = self.run_fake_sample("missing_stack_profile")

        self.assertEqual(sample["status"], "failed")
        self.assertIn("top_level_stack_size_not_accounted", sample["failure_reasons"])
        self.assertIn("parameter_stack_size_not_accounted", sample["failure_reasons"])
        self.assertIn("scenario_stack_size_not_accounted", sample["failure_reasons"])
        self.assertIn("result_stack_size_not_accounted", sample["failure_reasons"])

    def test_run_sample_accepts_accounted_scenario_report(self) -> None:
        sample = self.run_fake_sample("ok")

        self.assertEqual(sample["status"], "ok")
        self.assertEqual(sample["reported_worker_stack_bytes"], 128 * 1024)
        self.assertEqual(
            sample["reported_stack_profile"]["worker_stack_bytes"],
            128 * 1024,
        )
        self.assertIs(sample["parameter_stack_accounted"], True)
        self.assertIs(sample["scenario_stack_accounted"], True)
        self.assertIs(sample["result_stack_accounted"], True)
        self.assertIs(sample["stack_accounted"], True)
        self.assertEqual(sample["reported_scenarios"], ["history_render"])
        self.assertEqual(sample["stderr_tail"], "")


class CliReleaseStackProfilerTests(unittest.TestCase):
    def run_fake_sample(
        self,
        mode: str,
        *,
        scenario: str = "info_standard",
        stack_bytes: int = 128 * 1024,
    ) -> dict[str, object]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "fake_lash"
            binary.write_text(
                f"""#!{sys.executable}
import os
import sys

mode = {mode!r}
args = sys.argv[1:]
assert os.environ["LASH_TOKIO_STACK_BYTES"]
assert os.environ["LASH_HOME"]
print(f"fake mode {{mode}}", file=sys.stderr)

if mode == "stack_overflow":
    print("thread 'tokio-rt-worker' has overflowed its stack", file=sys.stderr)
    raise SystemExit(134)
if "--mode" in args and "json" in args:
    print("Error: `--mode json` requires `--print <prompt>`.", file=sys.stderr)
    raise SystemExit(1)
if "--print" in args:
    print("test-provider echo: hello from pty")
    raise SystemExit(0)
if mode == "missing_stdout":
    print("configured")
    raise SystemExit(0)
print("not configured")
"""
            )
            binary.chmod(0o755)
            return profile_cli_release_stack.run_sample(
                root=root,
                binary=binary,
                scenario=scenario,
                stack_bytes=stack_bytes,
                out=root / "matrix.json",
                timeout_seconds=5,
            )

    def test_parse_stack_sizes_accepts_common_suffixes(self) -> None:
        self.assertEqual(profile_cli_release_stack.parse_size("64k"), 64 * 1024)
        self.assertEqual(profile_cli_release_stack.parse_size("2m"), 2 * 1024 * 1024)
        self.assertEqual(profile_cli_release_stack.parse_size("1_024"), 1024)

    def test_first_success_requires_expected_result(self) -> None:
        samples = [
            {
                "scenario": "info_standard",
                "stack_bytes": 64 * 1024,
                "status": "failed",
                "stack_env_accounted": True,
            },
            {
                "scenario": "info_standard",
                "stack_bytes": 128 * 1024,
                "status": "ok",
                "stack_env_accounted": True,
            },
        ]

        self.assertEqual(
            profile_cli_release_stack.first_success(samples, "info_standard"),
            128 * 1024,
        )

    def test_run_sample_accepts_expected_zero_and_nonzero_exits(self) -> None:
        info = self.run_fake_sample("ok", scenario="info_standard")
        json_mode = self.run_fake_sample("ok", scenario="json_requires_print")
        print_mode = self.run_fake_sample("ok", scenario="print_standard_echo")

        self.assertEqual(info["status"], "ok")
        self.assertEqual(info["returncode"], 0)
        self.assertIs(info["stack_env_accounted"], True)
        self.assertEqual(json_mode["status"], "ok")
        self.assertEqual(json_mode["returncode"], 1)
        self.assertEqual(print_mode["status"], "ok")
        self.assertEqual(print_mode["returncode"], 0)

    def test_print_scenario_requires_test_provider_feature(self) -> None:
        self.assertEqual(
            profile_cli_release_stack.required_features_for_scenarios(
                ["info_standard", "print_standard_echo", "interactive_steer_escape"]
            ),
            ["test-provider"],
        )

    def test_run_sample_rejects_missing_expected_output(self) -> None:
        sample = self.run_fake_sample("missing_stdout")

        self.assertEqual(sample["status"], "failed")
        self.assertIn("missing_expected_stdout", sample["failure_reasons"])

    def test_run_sample_rejects_stack_overflow_text(self) -> None:
        sample = self.run_fake_sample("stack_overflow")

        self.assertEqual(sample["status"], "failed")
        self.assertIn("stack_overflow", sample["failure_reasons"])


class ProfileGuardCoverageTests(unittest.TestCase):
    def test_cli_cargo_feature_only_routes_to_ui_lane(self) -> None:
        args = argparse.Namespace(
            release=True,
            profile="quick",
            cargo_feature=["common"],
            cli_cargo_feature=["fff-zlob"],
            runtime_runs=None,
            runtime_warmups=None,
            runtime_turns=None,
            runtime_scenario=[],
            ui_runs=None,
            ui_warmups=None,
            ui_scenario=[],
            cli_stack_scenario=[],
            cli_stack_bytes=[],
            cli_release_stack_scenario=[],
            cli_release_stack_bytes=[],
            timeout_seconds=180,
            enforce=False,
            dhat_frames=24,
        )
        root = Path("/repo")

        runtime = profile_guard.runtime_cmd(args, root, Path("/tmp/runtime.json"))
        ui = profile_guard.ui_cmd(args, root, Path("/tmp/ui.json"))
        cli_stack = profile_guard.cli_stack_cmd(args, root, Path("/tmp/cli-stack.json"))
        cli_release_stack = profile_guard.cli_release_stack_cmd(
            args, root, Path("/tmp/cli-release-stack.json")
        )

        self.assertIn("common", runtime)
        self.assertNotIn("fff-zlob", runtime)
        self.assertIn("common", ui)
        self.assertIn("fff-zlob", ui)
        self.assertIn("common", cli_stack)
        self.assertIn("fff-zlob", cli_stack)
        self.assertIn("common", cli_release_stack)
        self.assertIn("fff-zlob", cli_release_stack)

    def test_async_completion_scenarios_are_in_default_perf_coverage(self) -> None:
        self.assertIn(
            "rlm_process_async_tool_completion",
            profile_guard.DEFAULT_STACK_SCENARIOS,
        )
        self.assertIn(
            "rlm_process_async_tool_completion",
            profile_runtime_stack.DEFAULT_SCENARIOS,
        )

    def test_workbench_trigger_mail_pipeline_is_in_default_stack_coverage(self) -> None:
        self.assertIn(
            "rlm_trigger_mail_pipeline",
            profile_guard.DEFAULT_STACK_SCENARIOS,
        )
        self.assertIn(
            "rlm_trigger_mail_pipeline",
            profile_runtime_stack.DEFAULT_SCENARIOS,
        )

    def test_typed_facade_scenarios_are_in_default_perf_coverage(self) -> None:
        required = {
            "standard",
            "rlm",
            "embed_standard",
            "embed_rlm",
            "trace_jsonl_standard",
            "trace_jsonl_extended",
        }

        self.assertTrue(required.issubset(profile_guard.DEFAULT_STACK_SCENARIOS))
        self.assertTrue(required.issubset(profile_runtime_stack.DEFAULT_SCENARIOS))

    def test_runtime_stack_defaults_cover_every_known_runtime_scenario(self) -> None:
        self.assertEqual(
            profile_runtime_stack.DEFAULT_SCENARIOS,
            profile_runtime_stack.KNOWN_RUNTIME_SCENARIOS,
        )
        self.assertEqual(
            len(profile_runtime_stack.DEFAULT_SCENARIOS),
            len(set(profile_runtime_stack.DEFAULT_SCENARIOS)),
        )
        self.assertEqual(
            profile_guard.DEFAULT_STACK_SCENARIOS,
            profile_runtime_stack.DEFAULT_SCENARIOS,
        )

    def test_runtime_stack_defaults_match_rust_runtime_perf_scenarios(self) -> None:
        scenarios_rs = (
            SCRIPT_DIR.parent
            / "crates"
            / "lash-perf"
            / "src"
            / "runtime_perf"
            / "scenarios.rs"
        ).read_text()
        rust_scenarios = re.findall(r'"([a-z0-9_]+)" => Some\(Self::', scenarios_rs)

        self.assertEqual(profile_runtime_stack.KNOWN_RUNTIME_SCENARIOS, rust_scenarios)
        self.assertIn(
            "turn_input_ingress_interrupt",
            profile_runtime_stack.DEFAULT_SCENARIOS,
        )

    def test_cli_stack_defaults_cover_every_known_ui_scenario(self) -> None:
        self.assertEqual(
            profile_cli_stack.DEFAULT_SCENARIOS,
            profile_cli_stack.KNOWN_UI_SCENARIOS,
        )
        self.assertEqual(
            profile_guard.DEFAULT_CLI_STACK_SCENARIOS,
            profile_cli_stack.DEFAULT_SCENARIOS,
        )
        self.assertEqual(
            len(profile_cli_stack.DEFAULT_SCENARIOS),
            len(set(profile_cli_stack.DEFAULT_SCENARIOS)),
        )

    def test_cli_stack_defaults_match_rust_ui_perf_scenarios(self) -> None:
        scenarios_rs = (
            SCRIPT_DIR.parent / "crates" / "lash-cli" / "src" / "ui_perf" / "scenarios.rs"
        ).read_text()
        impl_start = scenarios_rs.index("impl UiPerfScenario")
        impl_end = scenarios_rs.index("#[derive", impl_start)
        rust_scenarios = re.findall(
            r'Self::[A-Za-z0-9]+ => "([a-z0-9_]+)"',
            scenarios_rs[impl_start:impl_end],
        )

        self.assertEqual(profile_cli_stack.KNOWN_UI_SCENARIOS, rust_scenarios)
        self.assertIn(
            "turn_interrupt_steer_reconciliation",
            profile_cli_stack.DEFAULT_SCENARIOS,
        )

    def test_cli_release_stack_defaults_cover_every_known_release_scenario(self) -> None:
        self.assertEqual(
            profile_cli_release_stack.DEFAULT_SCENARIOS,
            profile_cli_release_stack.KNOWN_SCENARIOS,
        )
        self.assertEqual(
            profile_guard.DEFAULT_CLI_RELEASE_STACK_SCENARIOS,
            profile_cli_release_stack.DEFAULT_SCENARIOS,
        )
        self.assertIn(
            "interactive_steer_escape",
            profile_cli_release_stack.DEFAULT_SCENARIOS,
        )

    def test_profile_scripts_respect_cargo_target_dir(self) -> None:
        root = Path("/repo")
        args = argparse.Namespace(binary=None, release=False)

        with patch.dict(os.environ, {"CARGO_TARGET_DIR": "/tmp/cargo-target"}):
            self.assertEqual(
                profile_runtime.resolve_binary(args, root),
                Path("/tmp/cargo-target/debug/lash-perf"),
            )
            self.assertEqual(
                profile_runtime_stack.resolve_binary(args, root),
                Path("/tmp/cargo-target/debug/lash-perf"),
            )
            self.assertEqual(
                profile_ui.resolve_binary(args, root),
                Path("/tmp/cargo-target/debug/lash"),
            )
            self.assertEqual(
                profile_cli_stack.resolve_binary(args, root),
                Path("/tmp/cargo-target/debug/lash"),
            )
            self.assertEqual(
                profile_cli_release_stack.resolve_binary(args, root),
                Path("/tmp/cargo-target/debug/lash"),
            )
            self.assertEqual(
                profile_lashlang.example_path(root, True, "perf"),
                Path("/tmp/cargo-target/debug/examples/perf"),
            )

    def test_coverage_passes_with_required_sections_and_stack_budget(self) -> None:
        payload = {
            "runtime": {
                "returncode": 0,
                "expected_scenarios": ["standard"],
                "report": {
                    "worker_stack_bytes": 2 * 1024 * 1024,
                    "stack_profile": {
                        "worker_stack_bytes": 2 * 1024 * 1024,
                        "stack_budget_bytes": 2 * 1024 * 1024,
                        "within_stack_budget": True,
                    },
                    "summary": [
                        {
                            "scenario": "standard",
                            "stack_profile": {
                                "worker_stack_bytes": 2 * 1024 * 1024,
                                "stack_budget_bytes": 2 * 1024 * 1024,
                                "within_stack_budget": True,
                            },
                        }
                    ],
                    "results": [
                        {
                            "scenario": "standard",
                            "stack_profile": {
                                "worker_stack_bytes": 2 * 1024 * 1024,
                                "stack_budget_bytes": 2 * 1024 * 1024,
                                "within_stack_budget": True,
                            },
                        }
                    ],
                    "budget_results": [{"passed": True}],
                },
            },
            "runtime_stack": {
                "returncode": 0,
                "expected_scenarios": ["standard"],
                "first_success_stack_bytes": {"standard": 2 * 1024 * 1024},
                "report": stack_lane_report(),
            },
            "ui": {
                "returncode": 0,
                "report": {
                    "stack_profile": stack_profile(),
                    "parameters": {"stack_profile": stack_profile()},
                    "scenarios": [
                        {
                            "scenario": "history_render",
                            "stack_profile": stack_profile(),
                            "results": [{"stack_profile": stack_profile()}],
                        }
                    ],
                },
            },
            "cli_stack": {
                "returncode": 0,
                "expected_scenarios": ["history_render"],
                "first_success_stack_bytes": {"history_render": 2 * 1024 * 1024},
                "report": stack_lane_report(),
                "samples": [
                    {
                        "scenario": "history_render",
                        "stack_bytes": 2 * 1024 * 1024,
                        "status": "ok",
                        "stack_accounted": True,
                        "reported_scenarios": ["history_render"],
                    }
                ],
            },
            "cli_release_stack": {
                "returncode": 0,
                "expected_scenarios": ["info_standard"],
                "first_success_stack_bytes": {"info_standard": 2 * 1024 * 1024},
                "report": stack_lane_report(),
                "samples": [
                    {
                        "scenario": "info_standard",
                        "stack_bytes": 2 * 1024 * 1024,
                        "status": "ok",
                        "stack_env_accounted": True,
                    }
                ],
            },
            "lashlang": {
                "returncode": 0,
                "report": {
                    "stack_profile": stack_profile(),
                    "parameters": {"stack_profile": stack_profile()},
                    "perf_results": [
                        {
                            "scenario_arg": "baseline",
                            "stack_profile": stack_profile(),
                        }
                    ],
                    "profile_results": [
                        {
                            "scenario_arg": "baseline",
                            "stack_profile": stack_profile(),
                        }
                    ],
                    "budget_results": [{"passed": True}],
                },
            },
        }

        coverage = profile_guard.evaluate_guard_coverage(payload)

        self.assertTrue(coverage["passed"], coverage["findings"])

    def test_coverage_reports_missing_section_and_budget_failures(self) -> None:
        payload = {
            "runtime": {
                "returncode": 0,
                "expected_scenarios": ["standard", "trace_jsonl_standard"],
                "report": {
                    "summary": [{"scenario": "standard"}],
                    "results": [{"scenario": "standard"}],
                    "budget_results": [{"passed": False, "metric": "total_alloc_bytes"}],
                },
            },
            "runtime_stack": {
                "returncode": 0,
                "expected_scenarios": ["standard"],
                "first_success_stack_bytes": {"standard": 4 * 1024 * 1024},
                "report": stack_lane_report(),
                "samples": [
                    {
                        "scenario": "standard",
                        "stack_bytes": 4 * 1024 * 1024,
                        "status": "ok",
                        "stack_accounted": False,
                    }
                ],
            },
            "lashlang": {
                "returncode": 0,
                "report": {
                    "perf_results": [],
                    "profile_results": [{"scenario_arg": "baseline"}],
                    "budget_results": [],
                },
            },
        }

        coverage = profile_guard.evaluate_guard_coverage(payload)
        kinds = {finding["kind"] for finding in coverage["findings"]}

        self.assertFalse(coverage["passed"])
        self.assertIn("missing_section", kinds)
        self.assertIn("missing_runtime_scenario", kinds)
        self.assertIn("runtime_budget_failed", kinds)
        self.assertIn("runtime_stack_profile_missing", kinds)
        self.assertIn("runtime_summary_stack_profile_missing", kinds)
        self.assertIn("runtime_result_stack_profile_missing", kinds)
        self.assertIn("stack_budget_failed", kinds)
        self.assertIn("stack_size_not_accounted", kinds)
        self.assertIn("missing_lashlang_perf_results", kinds)
        self.assertIn("lashlang_stack_profile_missing", kinds)
        self.assertIn("lashlang_parameter_stack_profile_missing", kinds)
        self.assertIn("lashlang_profile_result_stack_profile_missing", kinds)

    def test_coverage_reports_cli_stack_budget_and_accounting_failures(self) -> None:
        payload = {
            "runtime": {
                "returncode": 0,
                "expected_scenarios": [],
                "report": {
                    "worker_stack_bytes": 2 * 1024 * 1024,
                    "stack_profile": {
                        "worker_stack_bytes": 2 * 1024 * 1024,
                        "stack_budget_bytes": 2 * 1024 * 1024,
                        "within_stack_budget": True,
                    },
                    "summary": [],
                    "results": [],
                    "budget_results": [{"passed": True}],
                },
            },
            "runtime_stack": {
                "returncode": 0,
                "expected_scenarios": [],
                "first_success_stack_bytes": {},
            },
            "ui": {
                "returncode": 0,
                "report": {
                    "stack_profile": stack_profile(),
                    "parameters": {"stack_profile": stack_profile()},
                    "scenarios": [
                        {
                            "scenario": "history_render",
                            "stack_profile": stack_profile(),
                            "results": [{"stack_profile": stack_profile()}],
                        }
                    ],
                },
            },
            "cli_stack": {
                "returncode": 0,
                "expected_scenarios": ["history_render", "workspace_surface"],
                "first_success_stack_bytes": {
                    "history_render": 4 * 1024 * 1024,
                },
                "report": stack_lane_report(),
                "samples": [
                    {
                        "scenario": "history_render",
                        "stack_bytes": 4 * 1024 * 1024,
                        "status": "ok",
                        "stack_accounted": False,
                        "reported_scenarios": ["other"],
                    }
                ],
            },
            "cli_release_stack": {
                "returncode": 0,
                "expected_scenarios": ["info_standard", "info_rlm"],
                "first_success_stack_bytes": {
                    "info_standard": 4 * 1024 * 1024,
                },
                "report": stack_lane_report(),
                "samples": [
                    {
                        "scenario": "info_standard",
                        "stack_bytes": 4 * 1024 * 1024,
                        "status": "ok",
                        "stack_env_accounted": False,
                    }
                ],
            },
            "lashlang": {
                "returncode": 0,
                "report": {
                    "stack_profile": stack_profile(),
                    "parameters": {"stack_profile": stack_profile()},
                    "perf_results": [{"scenario_arg": "baseline", "stack_profile": stack_profile()}],
                    "profile_results": [
                        {"scenario_arg": "baseline", "stack_profile": stack_profile()}
                    ],
                    "budget_results": [{"passed": True}],
                },
            },
        }

        coverage = profile_guard.evaluate_guard_coverage(payload)
        kinds = {finding["kind"] for finding in coverage["findings"]}

        self.assertFalse(coverage["passed"])
        self.assertIn("cli_stack_budget_failed", kinds)
        self.assertIn("missing_cli_stack_success", kinds)
        self.assertIn("cli_stack_size_not_accounted", kinds)
        self.assertIn("cli_stack_scenario_not_reported", kinds)
        self.assertIn("cli_release_stack_budget_failed", kinds)
        self.assertIn("missing_cli_release_stack_success", kinds)
        self.assertIn("cli_release_stack_env_not_accounted", kinds)

    def test_coverage_reports_stack_budget_gaps_and_failures(self) -> None:
        unbudgeted = stack_profile()
        unbudgeted.pop("stack_budget_bytes")
        oversized = stack_profile(4 * 1024 * 1024)
        payload = {
            "runtime": {
                "returncode": 0,
                "expected_scenarios": ["standard"],
                "report": {
                    "worker_stack_bytes": 4 * 1024 * 1024,
                    "stack_profile": {
                        "worker_stack_bytes": 4 * 1024 * 1024,
                        "measured_stack_bytes": 4 * 1024 * 1024,
                        "stack_budget_bytes": 2 * 1024 * 1024,
                        "within_stack_budget": False,
                    },
                    "summary": [
                        {
                            "scenario": "standard",
                            "stack_profile": {
                                "worker_stack_bytes": 4 * 1024 * 1024,
                                "measured_stack_bytes": 4 * 1024 * 1024,
                            },
                        }
                    ],
                    "results": [
                        {
                            "scenario": "standard",
                            "stack_profile": {
                                "worker_stack_bytes": 4 * 1024 * 1024,
                                "measured_stack_bytes": 4 * 1024 * 1024,
                                "stack_budget_bytes": 2 * 1024 * 1024,
                                "within_stack_budget": False,
                            },
                        }
                    ],
                    "budget_results": [{"passed": True}],
                },
            },
            "runtime_stack": {
                "returncode": 0,
                "expected_scenarios": ["standard"],
                "first_success_stack_bytes": {"standard": 2 * 1024 * 1024},
                "report": stack_lane_report(),
            },
            "cli_stack": {
                "returncode": 0,
                "expected_scenarios": ["history_render"],
                "first_success_stack_bytes": {"history_render": 2 * 1024 * 1024},
                "report": stack_lane_report(),
            },
            "cli_release_stack": {
                "returncode": 0,
                "expected_scenarios": ["info_standard"],
                "first_success_stack_bytes": {"info_standard": 2 * 1024 * 1024},
                "report": stack_lane_report(),
            },
            "ui": {
                "returncode": 0,
                "report": {
                    "stack_profile": unbudgeted,
                    "parameters": {"stack_profile": stack_profile()},
                    "scenarios": [{"scenario": "history_render", "stack_profile": stack_profile()}],
                },
            },
            "lashlang": {
                "returncode": 0,
                "report": {
                    "stack_profile": oversized,
                    "parameters": {"stack_profile": stack_profile()},
                    "perf_results": [{"scenario_arg": "baseline", "stack_profile": stack_profile()}],
                    "profile_results": [
                        {"scenario_arg": "baseline", "stack_profile": oversized}
                    ],
                    "budget_results": [{"passed": True}],
                },
            },
        }

        coverage = profile_guard.evaluate_guard_coverage(payload)
        kinds = {finding["kind"] for finding in coverage["findings"]}

        self.assertFalse(coverage["passed"])
        self.assertIn("runtime_stack_budget_failed", kinds)
        self.assertIn("runtime_summary_stack_budget_missing", kinds)
        self.assertIn("runtime_result_stack_budget_failed", kinds)
        self.assertIn("ui_stack_budget_missing", kinds)
        self.assertIn("lashlang_stack_budget_failed", kinds)
        self.assertIn("lashlang_profile_result_stack_budget_failed", kinds)


class LashlangBudgetTests(unittest.TestCase):
    def test_lashlang_budget_evaluator_checks_allocations_and_instructions(self) -> None:
        budgets = {
            "lashlang": {
                "perf": {
                    "default": {
                        "allocated_bytes_per_iter_max": 64,
                        "allocations_per_iter_max": 8,
                    }
                },
                "profile": {"default": {"instructions_per_iter_max": 6}},
            }
        }
        report = {
            "perf_results": [
                {
                    "scenario_arg": "baseline",
                    "mode_arg": "one_shot",
                    "allocated_bytes_per_iter": 32,
                    "allocations_per_iter": 4,
                }
            ],
            "profile_results": [
                {
                    "scenario_arg": "baseline",
                    "iterations": 5,
                    "instruction_hotspots": [{"count": 20}, {"count": 5}],
                }
            ],
        }

        results = profile_lashlang.evaluate_lashlang_budgets(report, budgets)

        self.assertTrue(all(result["passed"] for result in results), results)

    def test_lashlang_budget_evaluator_reports_excessive_allocations(self) -> None:
        budgets = {
            "lashlang": {
                "perf": {
                    "default": {
                        "allocated_bytes_per_iter_max": 10,
                        "allocations_per_iter_max": 8,
                    }
                },
                "profile": {"default": {"instructions_per_iter_max": 6}},
            }
        }
        report = {
            "perf_results": [
                {
                    "scenario_arg": "baseline",
                    "mode_arg": "one_shot",
                    "allocated_bytes_per_iter": 32,
                    "allocations_per_iter": 4,
                }
            ],
            "profile_results": [
                {
                    "scenario_arg": "baseline",
                    "iterations": 5,
                    "instruction_hotspots": [{"count": 20}],
                }
            ],
        }

        results = profile_lashlang.evaluate_lashlang_budgets(report, budgets)

        self.assertTrue(any(not result["passed"] for result in results), results)


if __name__ == "__main__":
    unittest.main()
