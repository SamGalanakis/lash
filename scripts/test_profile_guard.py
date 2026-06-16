#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import profile_guard  # noqa: E402
import profile_lashlang  # noqa: E402
import profile_runtime_stack  # noqa: E402


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
