#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
GATE = ROOT / "scripts" / "confidence-gate.sh"
BROAD_STEP_NAME = "Run bounded broad replay/backend confidence"
BOUNDED_BROAD_JOB_ID = "bounded-broad-replay-backend"
BOUNDED_BROAD_ARTIFACT = "bounded-broad-replay-backend-confidence"
BOUNDED_BROAD_OUT_ROOT = "target/confidence-ci/bounded-broad-replay-backend"


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
        gate = GATE.read_text(encoding="utf-8")
        env = broad_step_env(workflow)

        self.assertIn(
            f"  {BOUNDED_BROAD_JOB_ID}:\n    name: Bounded Broad Replay/Backend",
            workflow,
        )
        self.assertNotIn("confidence-" + "broad-replay-backend", workflow)
        self.assertNotIn("Confidence " + "Broad Replay/Backend", workflow)
        self.assertEqual(env["LASH_CONFIDENCE_OUT_DIR"], BOUNDED_BROAD_OUT_ROOT)
        self.assertEqual(env["LASH_CONFIDENCE_MUTATION_SCOPE"], "none")
        self.assertEqual(env["LASH_CONFIDENCE_COVERAGE_SCOPE"], "none")
        self.assertIn(f"name: {BOUNDED_BROAD_ARTIFACT}", workflow)
        self.assertIn(f"path: {BOUNDED_BROAD_OUT_ROOT}/broad", workflow)
        self.assertIn('"bounded_broad_ci": {', gate)
        self.assertIn(f'"workflow_job": "{BOUNDED_BROAD_JOB_ID}"', gate)
        self.assertIn(f'"artifact_name": "{BOUNDED_BROAD_ARTIFACT}"', gate)
        self.assertIn('"coverage_evidence_status": "not_run_by_scope"', gate)
        self.assertIn('"mutation_evidence_status": "not_run_by_scope"', gate)

    def test_full_lane_artifact_contract_requires_true_full_evidence(self) -> None:
        gate = GATE.read_text(encoding="utf-8")

        required_snippets = [
            'if [ "$lane" = "full" ] && [ "$mutation_scope" != "full" ]; then',
            'if [ "$lane" = "full" ] && [ "$coverage_scope" != "run" ]; then',
            "full_mutation_suites_complete()",
            "mutation_evidence_status()",
            "coverage_evidence_status()",
            "restate_postgres_workers_e2e_status()",
            '"artifact_contract": {',
            '"schema": "lash.confidence.summary-artifact-contract.v1"',
            '"full_lane": {',
            '"confidence_class": "true_full"',
            '"required_coverage_scope": "run"',
            '"effective_coverage_scope": "${coverage_scope}"',
            '"coverage_evidence_status": "$(coverage_evidence_status)"',
            '"required_mutation_scope": "full"',
            '"effective_mutation_scope": "${mutation_scope}"',
            '"mutation_evidence": "$(mutation_evidence_path)"',
            '"mutation_evidence_status": "$(mutation_evidence_status)"',
            '"full_mutation_status": "$(full_mutation_status)"',
            '"required_restate_postgres_workers_e2e": "sim/restate-postgres-workers-e2e.json"',
            '"restate_postgres_workers_e2e_status": "$(restate_postgres_workers_e2e_status)"',
            "run_restate_postgres_workers_e2e",
            '"status": "not_run"',
            '"reason": "distributed Restate/Postgres/MinIO worker e2e is full-lane-only"',
        ]

        for snippet in required_snippets:
            self.assertIn(snippet, gate)


if __name__ == "__main__":
    unittest.main()
