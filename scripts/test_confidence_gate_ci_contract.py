#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
GATE = ROOT / "scripts" / "confidence-gate.sh"
BROAD_STEP_NAME = "Run bounded broad replay/backend confidence"


def shell_int_constant(script: str, name: str) -> int:
    match = re.search(rf"^{re.escape(name)}=([0-9]+)$", script, re.MULTILINE)
    if match is None:
        raise AssertionError(f"missing shell constant {name}")
    return int(match.group(1))


def broad_step_env(workflow: str) -> dict[str, str]:
    marker = f"- name: {BROAD_STEP_NAME}"
    try:
        section = workflow.split(marker, 1)[1]
    except IndexError as exc:
        raise AssertionError(f"missing workflow step {BROAD_STEP_NAME!r}") from exc
    section = section.split("\n      - name:", 1)[0]
    env: dict[str, str] = {}
    for name, value in re.findall(r"^\s{10}([A-Z0-9_]+):\s*(\S+)\s*$", section, re.MULTILINE):
        env[name] = value
    return env


class ConfidenceGateCiContractTest(unittest.TestCase):
    def test_ci_broad_scheduled_depth_matches_gate_thresholds(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        gate = GATE.read_text(encoding="utf-8")
        env = broad_step_env(workflow)

        min_seeds = shell_int_constant(gate, "BROAD_SCHEDULED_DEPTH_MIN_SEEDS")
        min_boundaries = shell_int_constant(gate, "BROAD_SCHEDULED_DEPTH_MIN_MAX_BOUNDARIES")

        self.assertGreaterEqual(int(env["LASH_SCHEDULED_SIM_SEEDS"]), min_seeds)
        self.assertGreaterEqual(
            int(env["LASH_SCHEDULED_SIM_MAX_BOUNDARIES"]),
            min_boundaries,
        )

    def test_ci_broad_lane_is_explicitly_bounded_not_full_confidence(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        env = broad_step_env(workflow)

        self.assertEqual(env["LASH_CONFIDENCE_MUTATION_SCOPE"], "none")
        self.assertEqual(env["LASH_CONFIDENCE_COVERAGE_SCOPE"], "none")


if __name__ == "__main__":
    unittest.main()
