#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
CONFIDENCE_WORKFLOW = ROOT / ".github" / "workflows" / "confidence.yml"
GATE = ROOT / "scripts" / "confidence-gate.sh"
FOCUSED_SQLITE_REPRO = ROOT / "scripts" / "lash-sim-focused-sqlite-repro.sh"
FAST_STEP_NAME = "Confidence gate fast lane"
OLD_BROAD_CI_STEP_NAME = "Run bounded broad " + "replay/backend confidence"
OLD_BROAD_CI_JOB_ID = "bounded-" + "broad-replay-backend"
OLD_BROAD_CI_ARTIFACT = "bounded-" + "broad-replay-backend-confidence"
OLD_BROAD_CI_OUT_ROOT = "target/confidence-ci/" + OLD_BROAD_CI_JOB_ID


def shell_int_constant(script: str, name: str) -> int:
    match = re.search(rf"^{re.escape(name)}=([0-9]+)$", script, re.MULTILINE)
    if match is None:
        raise AssertionError(f"missing shell constant {name}")
    return int(match.group(1))


def step_block(workflow: str, step_name: str) -> str:
    marker = f"- name: {step_name}"
    try:
        section = workflow.split(marker, 1)[1]
    except IndexError as exc:
        raise AssertionError(f"missing workflow step {step_name!r}") from exc
    return section.split("\n      - name:", 1)[0]


class ConfidenceGateCiContractTest(unittest.TestCase):
    def test_ci_runs_only_fast_confidence_not_broad_replay_backend(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        gate = GATE.read_text(encoding="utf-8")
        fast_step = step_block(workflow, FAST_STEP_NAME)

        self.assertIn("bash scripts/confidence-gate.sh fast", fast_step)
        self.assertNotIn(OLD_BROAD_CI_JOB_ID, workflow)
        self.assertNotIn("Bounded Broad " + "Replay/Backend", workflow)
        self.assertNotIn(OLD_BROAD_CI_STEP_NAME, workflow)
        self.assertNotIn(OLD_BROAD_CI_ARTIFACT, workflow)
        self.assertNotIn(OLD_BROAD_CI_OUT_ROOT, workflow)

        min_seeds = shell_int_constant(gate, "BROAD_SCHEDULED_DEPTH_MIN_SEEDS")
        min_boundaries = shell_int_constant(gate, "BROAD_SCHEDULED_DEPTH_MIN_MAX_BOUNDARIES")
        self.assertGreaterEqual(min_seeds, 4)
        self.assertGreaterEqual(min_boundaries, 256)

    def test_broad_lane_is_manual_or_scheduled_confidence_not_ci_cd(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        confidence_workflow = CONFIDENCE_WORKFLOW.read_text(encoding="utf-8")
        gate = GATE.read_text(encoding="utf-8")

        self.assertIn("- fast", confidence_workflow)
        self.assertIn("- default", confidence_workflow)
        self.assertIn("- broad", confidence_workflow)
        self.assertIn("- full", confidence_workflow)
        self.assertIn("run: bash scripts/confidence-gate.sh", confidence_workflow)
        self.assertIn("inputs.lane || 'full'", confidence_workflow)
        self.assertIn("schedule:", confidence_workflow)
        self.assertNotIn("bash scripts/confidence-gate.sh broad", workflow)

        self.assertIn('"bounded_broad_confidence": {', gate)
        self.assertIn('"workflow": "Confidence"', gate)
        self.assertIn('"lane": "broad"', gate)
        self.assertIn('"trigger": "workflow_dispatch_or_schedule"', gate)
        self.assertIn('"artifact_name": "confidence-artifacts"', gate)
        self.assertIn('"full_confidence_claim": "false"', gate)

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

    def test_focused_sqlite_seed_tail_repro_gate_is_named_and_exact(self) -> None:
        gate = GATE.read_text(encoding="utf-8")
        repro = FOCUSED_SQLITE_REPRO.read_text(encoding="utf-8")

        required_gate_snippets = [
            "run_focused_sqlite_seed_tail_repro()",
            'step "Focused generated SQLite seed-tail repro"',
            'scripts/lash-sim-focused-sqlite-repro.sh "$repro_dir"',
            "run_focused_sqlite_seed_tail_repro",
            '"focused_sqlite_seed_tail_repro": "$([ -f "${out_dir}/sim/focused-sqlite-seed-tail/focused-sqlite-seed-tail.json" ]',
        ]
        for snippet in required_gate_snippets:
            self.assertIn(snippet, gate)

        required_repro_snippets = [
            '"schema": "lash.confidence.focused-sqlite-seed-tail-repro.v1"',
            'focused_single_seed="4101155038242989457"',
            'focused_tail_previous_seed="17785827714152183977"',
            '--profile "$profile"',
            '--max-boundaries "$max_boundaries"',
            'run_case "single-seed-4101155038242989457" "$focused_single_seed"',
            '"tail-seeds-17785827714152183977-4101155038242989457"',
            '"sqlite_divergence_reports"',
        ]
        for snippet in required_repro_snippets:
            self.assertIn(snippet, repro)

    def test_static_replay_matrix_does_not_claim_generated_backend_static_replay(self) -> None:
        gate = GATE.read_text(encoding="utf-8")

        required_snippets = [
            'step "Static replay evidence matrix"',
            "generated_static_backend_skip_reason=",
            "generated_backend_fixture_static_skip_reason=",
            "generated_backend_regression_fixture",
            '"schema": "lash.confidence.static-replay-evidence-matrix.v1"',
            "backend equivalence is proved by per-seed generated workload rerun artifacts",
        ]
        for snippet in required_snippets:
            self.assertIn(snippet, gate)

        self.assertNotIn("backend_replayable_regression", gate)
        self.assertNotIn(
            "Every generated trace and every backend-replayable regression trace is replayed through model, SQLite, and Postgres",
            gate,
        )

    def test_generated_postgres_dynamic_rerun_is_bounded_and_artifacted(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        confidence_workflow = CONFIDENCE_WORKFLOW.read_text(encoding="utf-8")
        gate = GATE.read_text(encoding="utf-8")

        required_snippets = [
            "run_generated_postgres_dynamic_replay()",
            'step "Generated Postgres dynamic backend rerun"',
            "cargo run -p lash-sim --locked -- run-postgres",
            '--seed "$seed"',
            'LASH_POSTGRES_GENERATED_PROFILE:-full-random',
            'LASH_POSTGRES_GENERATED_MAX_BOUNDARIES:-128',
            '"confidence_lane": "generated_dynamic_postgres_backend_rerun"',
            '"generated_postgres_dynamic_replay": "$([ -f "${out_dir}/sim/postgres-generated-rerun/summary.json" ]',
        ]
        for snippet in required_snippets:
            self.assertIn(snippet, gate)

        self.assertIn("- broad", confidence_workflow)
        self.assertIn("inputs.lane || 'full'", confidence_workflow)
        self.assertNotIn("LASH_POSTGRES_GENERATED_PROFILE", workflow)
        self.assertNotIn("LASH_POSTGRES_GENERATED_MAX_BOUNDARIES", workflow)


if __name__ == "__main__":
    unittest.main()
