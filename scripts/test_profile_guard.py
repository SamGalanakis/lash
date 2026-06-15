#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import profile_guard  # noqa: E402
import profile_lashlang  # noqa: E402
import profile_runtime_stack  # noqa: E402


class ProfileGuardCoverageTests(unittest.TestCase):
    def test_async_completion_scenarios_are_in_default_perf_coverage(self) -> None:
        self.assertIn(
            "rlm_process_async_tool_completion",
            profile_guard.DEFAULT_STACK_SCENARIOS,
        )
        self.assertIn(
            "rlm_process_async_tool_completion",
            profile_runtime_stack.DEFAULT_SCENARIOS,
        )

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
