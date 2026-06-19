#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import profile_guard  # noqa: E402
import profile_lashlang  # noqa: E402
import profile_runtime_stack  # noqa: E402


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
out.write_text(json.dumps({{
    "worker_stack_bytes": reported_stack,
    "summary": [{{"scenario": reported_scenario}}],
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

    def test_run_sample_accepts_accounted_scenario_report(self) -> None:
        sample = self.run_fake_sample("ok")

        self.assertEqual(sample["status"], "ok")
        self.assertEqual(sample["reported_worker_stack_bytes"], 128 * 1024)
        self.assertIs(sample["stack_accounted"], True)
        self.assertEqual(sample["summary_scenarios"], ["standard"])
        self.assertEqual(sample["stderr_tail"], "")


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
            enforce=False,
            dhat_frames=24,
        )
        root = Path("/repo")

        runtime = profile_guard.runtime_cmd(args, root, Path("/tmp/runtime.json"))
        ui = profile_guard.ui_cmd(args, root, Path("/tmp/ui.json"))

        self.assertIn("common", runtime)
        self.assertNotIn("fff-zlob", runtime)
        self.assertIn("common", ui)
        self.assertIn("fff-zlob", ui)

    def test_async_completion_scenarios_are_in_default_perf_coverage(self) -> None:
        self.assertIn(
            "rlm_process_async_tool_completion",
            profile_guard.DEFAULT_STACK_SCENARIOS,
        )
        self.assertIn(
            "rlm_process_async_tool_completion",
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

    def test_coverage_passes_with_required_sections_and_stack_budget(self) -> None:
        payload = {
            "runtime": {
                "returncode": 0,
                "expected_scenarios": ["standard"],
                "report": {
                    "summary": [{"scenario": "standard"}],
                    "budget_results": [{"passed": True}],
                },
            },
            "runtime_stack": {
                "returncode": 0,
                "expected_scenarios": ["standard"],
                "first_success_stack_bytes": {"standard": 2 * 1024 * 1024},
            },
            "ui": {
                "returncode": 0,
                "report": {"scenarios": [{"scenario": "history_render"}]},
            },
            "lashlang": {
                "returncode": 0,
                "report": {
                    "perf_results": [{"scenario_arg": "baseline"}],
                    "profile_results": [{"scenario_arg": "baseline"}],
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
                    "budget_results": [{"passed": False, "metric": "total_alloc_bytes"}],
                },
            },
            "runtime_stack": {
                "returncode": 0,
                "expected_scenarios": ["standard"],
                "first_success_stack_bytes": {"standard": 4 * 1024 * 1024},
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
        self.assertIn("stack_budget_failed", kinds)
        self.assertIn("stack_size_not_accounted", kinds)
        self.assertIn("missing_lashlang_perf_results", kinds)


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
