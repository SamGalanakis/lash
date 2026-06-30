#![allow(clippy::result_large_err)]

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use lash::scenario_contracts::AGENT_SCENARIO_CONTRACTS;
use lash_core::llm::transport::{LlmTransportError, ProviderFailureKind};
use lash_core::llm::types::{
    LlmEventSender, LlmMessage, LlmOutputPart, LlmRequest, LlmResponse, LlmRole, LlmStreamEvent,
    LlmTerminalReason, LlmToolChoice, LlmToolSpec,
};
use lash_core::provider::{
    DefaultProviderFailureClassifier, Provider, ProviderFailureClassifier, ProviderHandle,
    ProviderOptions,
};
use lash_core::runtime::{RUNTIME_SCENARIO_CONTRACTS, ScenarioContractSpec};
use lash_core::{SessionStoreFactory, TriggerStore};
use lash_llm_transport::LlmHttpTransport;
use lash_protocol_rlm::scenario_contracts::RLM_PROTOCOL_SCENARIO_CONTRACTS;
use lash_protocol_standard::scenario_contracts::STANDARD_PROTOCOL_SCENARIO_CONTRACTS;
use lash_provider_anthropic::AnthropicProvider;
use lash_provider_google::GoogleOAuthProvider;
use lash_provider_openai::{OpenAiCompatibleProvider, OpenAiProvider};
use lash_rlm_types::{RlmCreateExtras, RlmProtocolEvent, RlmTermination};
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use lash::rlm::RlmTurnBuilderExt as _;
use lash_lashlang_runtime::ToolDefinitionLashlangExt as _;

use crate::artifacts::*;
use crate::canonical_scripts::{
    ANTHROPIC_MESSAGES_TEXT, CANONICAL_SCRIPTS, GOOGLE_GENERATE_RATE_LIMIT, GOOGLE_GENERATE_TEXT,
    GOOGLE_STREAM_GENERATE_TEXT, OPENAI_COMPAT_DISCONNECT, OPENAI_COMPAT_RATE_LIMIT,
    OPENAI_COMPAT_RESPONSE_START_TIMEOUT, OPENAI_COMPAT_STREAM_CHUNK_TIMEOUT,
    OPENAI_COMPAT_TOOL_CALL, OPENAI_COMPAT_VALIDATION, OPENAI_RESPONSES_TEXT,
};
use crate::generator::{
    GENERATOR_VERSION, GeneratedWorkload, WorkloadProfileError, generate_workload,
    validate_workload_profile,
};
use crate::minimize::{MinimizeError, minimize_trace};
use crate::oracles::{
    LiveProviderFailureFacts, backend_failure_observed, cancellation_observed, combine_oracles,
    cross_session_isolation, durable_effect_exactly_once, exec_code_observed,
    generated_final_value_semantic_channel,
    generated_runtime_provider_matrix as generated_runtime_provider_matrix_oracle,
    generated_suspend_resume, ingress_sessions_opened, lease_time_monotonic,
    live_provider_failure_coverage, observer_convergence, observer_reconnect_observed,
    operational_coverage, peak_concurrent_live_turns, pending_tool_completion,
    process_wake_observed, provider_mutation_rejected, provider_transport_mutation_classified,
    provider_turn_interleaving_depth, queued_ingress_observed, replay_determinism,
    runtime_final_value_semantic, runtime_graph_acyclic, runtime_provider_turn,
    runtime_session_graph_contract, runtime_single_active_agent_frame, runtime_usage_monotonic,
    scenario_contract_generated_facts, scenario_contract_mini_oracles, scenario_contract_oracles,
    scheduler_controlled_delivery, scheduler_owned_runtime_completions,
    state_machine_semantic_invariants, tool_boundary_observed, trigger_delivery_observed,
    worker_failover_continues_work, worker_stale_completion_rejected,
};
use crate::provider::{
    ProviderWireEvent, ProviderWireHeader, ProviderWireScript, ScriptedLlmHttpExchange,
    ScriptedLlmHttpTransport, ScriptedTransportSchedule,
};
use crate::provider_mutations::{ProviderMutationMatrixCache, is_transport_provider_mutation};
use crate::replay::{ReplayError, replay_trace};
use crate::runtime_boundaries::{RuntimeBoundaryHarness, RuntimeEffectReplayStore};
use crate::runtime_contracts::{
    RuntimeTurnObservation, require_passed, runtime_agent_frame_invariant_facts,
    runtime_final_value_invariant_facts, runtime_graph_invariant_facts, runtime_turn_contract,
    runtime_usage_invariant_facts,
};
use crate::runtime_providers::{
    ANTHROPIC, LIVE_FAILURE_LEAK_PROSE, OPENAI_COMPATIBLE, live_failure_script,
    runtime_provider_components, runtime_script_for_text,
    runtime_scripts_for_texts as runtime_provider_scripts_for_texts, suspend_roundtrip_scripts,
};
use crate::scheduler::{
    BoundaryDeliveryLog, BoundaryEvent, BoundaryKind, BoundaryScheduler, RuntimeCompletionQueue,
    RuntimeCompletionUnit,
};
use crate::sqlite_replay::SqliteReplayError;
use crate::stack_policy::{
    SIM_HARNESS_STACK_LIMIT_BYTES, run_on_product_stack, run_on_sim_harness_stack,
};
use crate::store::{ModelStore, backend_fault_observation};
use crate::trace::{
    AbstractWorldSummary, OracleStatus, OracleVerdict, SimulationTrace, TraceEventLine,
    TraceIoError, write_event_lines, write_replay_report, write_trace,
};

/// Deterministic yield budgets that bound provider-event release polling and
/// turn-completion polling. These replace wall-clock timeouts: a turn that
/// drifts so it never reaches a gate (or never finishes) terminates the poll
/// after a fixed number of cooperative yields instead of hanging the runtime.
/// The budgets are large enough that no in-order, single-exchange-per-turn
/// generated turn ever reaches them.
pub(crate) const MAX_PROVIDER_EVENT_POLL_YIELDS: u64 = 200_000;
pub(crate) const MAX_TURN_FINISH_POLL_YIELDS: u64 = 2_000_000;

pub const FIXED_SCRIPT_PROFILE: &str = "tiny-fixed-provider-scripts";
pub const FIXED_SCRIPT_EVENTS: &str = "events.jsonl";
pub const FIXED_SCRIPT_MANIFEST: &str = "fixed-script-manifest.json";
pub const FIXED_SCRIPT_SUMMARY: &str = "summary.json";
pub const FIXED_SCRIPT_TIMELINE_AT_SEMANTICS: &str =
    "validated-monotonic-metadata-reserved-for-future-scheduler";
pub const GENERATED_SIM_EVENTS: &str = "events.jsonl";
pub const GENERATED_SIM_SUMMARY: &str = "summary.json";
pub const GENERATED_SIM_PROVIDER_MANIFEST: &str = "provider-script-manifest.json";
pub const GENERATED_SIM_FAILURE_SHAPE: &str = "failures/_shape.json";
pub const GENERATED_SIM_SCENARIO_SLICES: &str = "scenario-contract-slices";
pub const GENERATED_SIM_SCENARIO_PACKAGES: &str = "scenario-contract-packages";
pub const GENERATED_SIM_BACKEND_REGRESSION_FIXTURES: &str = "backend-regression-fixtures";

#[derive(Debug)]
pub enum FixedScriptRunnerError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Provider(LlmTransportError),
    Trace(TraceIoError),
    Replay(ReplayError),
    SqliteReplay(SqliteReplayError),
    Minimize(MinimizeError),
    Profile(WorkloadProfileError),
    Runtime(String),
    Assertion(String),
}

impl fmt::Display for FixedScriptRunnerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "fixed-script artifact I/O failed: {err}"),
            Self::Json(err) => write!(f, "fixed-script artifact JSON failed: {err}"),
            Self::Provider(err) => write!(f, "fixed-script provider proof failed: {err}"),
            Self::Trace(err) => write!(f, "simulation trace artifact failed: {err}"),
            Self::Replay(err) => write!(f, "simulation replay failed: {err}"),
            Self::SqliteReplay(err) => write!(f, "SQLite simulation replay failed: {err}"),
            Self::Minimize(err) => write!(f, "simulation trace minimization failed: {err}"),
            Self::Profile(err) => write!(f, "workload profile rejected: {err}"),
            Self::Runtime(err) => write!(f, "runtime/facade proof failed: {err}"),
            Self::Assertion(message) => write!(f, "fixed-script proof assertion failed: {message}"),
        }
    }
}

impl std::error::Error for FixedScriptRunnerError {}

impl From<std::io::Error> for FixedScriptRunnerError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for FixedScriptRunnerError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

impl From<LlmTransportError> for FixedScriptRunnerError {
    fn from(value: LlmTransportError) -> Self {
        Self::Provider(value)
    }
}

impl From<TraceIoError> for FixedScriptRunnerError {
    fn from(value: TraceIoError) -> Self {
        Self::Trace(value)
    }
}

impl From<ReplayError> for FixedScriptRunnerError {
    fn from(value: ReplayError) -> Self {
        Self::Replay(value)
    }
}

impl From<SqliteReplayError> for FixedScriptRunnerError {
    fn from(value: SqliteReplayError) -> Self {
        Self::SqliteReplay(value)
    }
}

impl From<MinimizeError> for FixedScriptRunnerError {
    fn from(value: MinimizeError) -> Self {
        Self::Minimize(value)
    }
}

impl From<WorkloadProfileError> for FixedScriptRunnerError {
    fn from(value: WorkloadProfileError) -> Self {
        Self::Profile(value)
    }
}

fn provider_transport_exclusions() -> Vec<ProviderTransportExclusion> {
    vec![
        ProviderTransportExclusion {
            path: "crates/lash-provider-openai/src/codex.rs",
            status: "reviewed_non_dst_exclusion",
            reason: "Codex OAuth provider execution still owns a device/OAuth-specific reqwest client path and is not in the no-live-LLM LlmHttpTransport script matrix.",
            replacement_lane: "future codex-oauth LlmHttpTransport scripted transcript lane; current confidence gate keeps OpenAI/OpenAI-compatible/Anthropic/Google provider execution in DST.",
            review_owner: "lash-sim provider matrix",
        },
        ProviderTransportExclusion {
            path: "crates/lash-provider-openai/src/codex/oauth.rs",
            status: "reviewed_non_dst_exclusion",
            reason: "OAuth device-code polling and token exchange are auth flows, not LLM provider execution; they use reqwest directly and require separate OAuth script fixtures.",
            replacement_lane: "auth-flow conformance lane, not live LLM DST",
            review_owner: "lash-sim provider matrix",
        },
        ProviderTransportExclusion {
            path: "crates/lash-provider-google/src/oauth.rs",
            status: "reviewed_non_dst_exclusion",
            reason: "Google OAuth authorization-code and refresh-token HTTP calls remain outside the LLM transport DST because the generated harness scripts provider execution, not credential issuance.",
            replacement_lane: "auth-flow conformance lane, not live LLM DST",
            review_owner: "lash-sim provider matrix",
        },
        ProviderTransportExclusion {
            path: "crates/lash-core/src/runtime/session_manager/direct.rs",
            status: "reviewed_non_dst_exclusion",
            reason: "Direct completion effects are runtime effect-controller behavior; provider execution behind the direct request is covered through scripted provider transports, while direct effect planning is covered by runtime tests.",
            replacement_lane: "runtime direct-effect scenario contracts plus scripted provider transport proofs",
            review_owner: "lash-sim runtime/provider boundary",
        },
    ]
}

fn model_only_boundary_reviews() -> Vec<ModelOnlyBoundaryReview> {
    vec![
        ModelOnlyBoundaryReview {
            boundary_kind: "durable_effect",
            status: "runtime_effect_controller_backed_with_reviewed_host_history_ceiling",
            production_abstraction_used: "RuntimeEffectEnvelope, RuntimeEffectCommand::DurableStep, RuntimeEffectLocalExecutor::durable_step, SqliteRuntimeEffectController, and PostgresRuntimeEffectController",
            model_only_scope: "workflow-host crash history outside store-backed effect replay remains excluded; generated memory runs and generated SQLite dynamic reruns execute production runtime effect replay controllers, while Postgres conformance/contention lanes cover native Postgres replay storage in lash_runtime_effect_replay",
            oracle_id: "sim.oracle.durable-effect-exactly-once.v1",
            artifact_evidence: "durable-effect observations include runtime_effect.controller=sqlite_runtime_effect_controller or postgres_runtime_effect_controller, local_executor_called false on replay, first completion, replay for the same durable key, Postgres effect_history_replay.status=native_postgres_runtime_effect_controller, and generated SQLite divergence artifacts on mismatch",
        },
        ModelOnlyBoundaryReview {
            boundary_kind: "worker",
            status: "runtime_persistence_lease_backed_with_reviewed_worker_task_ceiling",
            production_abstraction_used: "RuntimePersistence session execution lease claim/reclaim/renew/release, SessionExecutionLease, LeaseOwnerIdentity, and SessionExecutionLeaseCompletion",
            model_only_scope: "DurableProcessWorker task body launch remains excluded; generated memory runs, generated SQLite dynamic reruns, and Postgres backend contention validate stale completion rejection through real backend lease stores",
            oracle_id: "sim.oracle.worker-stale-completion-rejected.v1",
            artifact_evidence: "worker observed payload records runtime_active_lease, runtime_stale_completion, runtime_worker_store.session_execution_lease_reclaimed=true, and stale_completion_rejected=true",
        },
        ModelOnlyBoundaryReview {
            boundary_kind: "backend_failure",
            status: "production_store_error_classified_fault_injection_boundary",
            production_abstraction_used: "lash_core::StoreError variants plus generated SQLite divergence artifacts and Postgres backend contention lanes",
            model_only_scope: "connection corruption and live database fault injection remain excluded; generated boundaries classify retryable and terminal backend faults as concrete StoreError variants and fail on generated SQLite dynamic rerun divergence",
            oracle_id: "sim.oracle.backend-failure-observed.v1",
            artifact_evidence: "backend failure events include production_store_error.type=lash_core::StoreError, variant, message, retryable_class, retry attempt counts, and generated SQLite divergence artifacts on mismatch",
        },
        ModelOnlyBoundaryReview {
            boundary_kind: "provider_mutation",
            status: "runtime_backed_script_mutation_boundary_with_reviewed_live_call_exclusion",
            production_abstraction_used: "ProviderWireScript, ScriptedLlmHttpTransport, and provider failure envelopes",
            model_only_scope: "live provider calls remain excluded; generated mutation boundaries execute malformed/rate-limit scripts through OpenAI-compatible, direct OpenAI, Anthropic, and Google provider parsers using ScriptedLlmHttpTransport",
            oracle_id: "sim.oracle.provider-mutation-rejected.v1",
            artifact_evidence: "provider mutation events include provider_parser_matrix proofs with real_provider_parser_execution=true and all migrated provider kinds",
        },
        ModelOnlyBoundaryReview {
            boundary_kind: "tool",
            status: "runtime_effect_controller_backed_with_reviewed_tool_provider_ceiling",
            production_abstraction_used: "RuntimeEffectEnvelope, RuntimeEffectCommand::ToolAttempt, RuntimeEffectLocalExecutor, ToolAttemptLaunch, ToolCallRecord, and ToolCallOutput",
            model_only_scope: "app-specific ToolProvider implementation bodies remain excluded; generated memory runs and generated SQLite dynamic reruns execute the production runtime effect-controller boundary with scripted no-network tool outcomes",
            oracle_id: "sim.oracle.tool-boundary-observed.v1",
            artifact_evidence: "tool events carry runtime_effect.controller=sqlite_runtime_effect_controller or postgres_runtime_effect_controller, runtime_tool_record, runtime_tool_output, and generated SQLite divergence artifacts on mismatch",
        },
        ModelOnlyBoundaryReview {
            boundary_kind: "exec_code",
            status: "runtime_effect_controller_backed_with_reviewed_shell_launch_ceiling",
            production_abstraction_used: "RuntimeEffectEnvelope, RuntimeEffectCommand::ExecCode, RuntimeEffectLocalExecutor, RuntimeEffectOutcome::ExecCode, and ExecResponse",
            model_only_scope: "host shell/kernel process launch remains excluded; generated memory runs and generated SQLite dynamic reruns execute the production runtime effect-controller boundary with scripted no-shell ExecResponse outcomes",
            oracle_id: "sim.oracle.exec-code-observed.v1",
            artifact_evidence: "exec-code events carry runtime_effect.controller=sqlite_runtime_effect_controller or postgres_runtime_effect_controller, runtime_effect_outcome, exit-code data, and generated SQLite divergence artifacts on mismatch",
        },
        ModelOnlyBoundaryReview {
            boundary_kind: "process_wake",
            status: "runtime_persistence_queued_work_backed_with_reviewed_process_body_ceiling",
            production_abstraction_used: "process_wake_delivery, QueuedWorkBatchDraft, QueuedWorkPayload::process_wake, RuntimePersistence::enqueue_queued_work, and claim_ready_queued_work_by_batch_ids",
            model_only_scope: "the eventual process body that consumes the wake remains excluded; generated memory runs, generated SQLite dynamic reruns, and Postgres backend contention enqueue or claim wake-adjacent queued work through real queued-work/session-lease backend paths",
            oracle_id: "sim.oracle.process-wake-observed.v1",
            artifact_evidence: "process wake events include runtime_process_wake, runtime_queued_work claim evidence, claimed_once=true, and duplicate claimed_once=false dedupe evidence from real queued-work claims",
        },
    ]
}

fn scenario_contract_manifests() -> Vec<ScenarioContractManifest> {
    vec![
        scenario_contract_manifest(RUNTIME_SCENARIO_CONTRACTS),
        scenario_contract_manifest(STANDARD_PROTOCOL_SCENARIO_CONTRACTS),
        scenario_contract_manifest(RLM_PROTOCOL_SCENARIO_CONTRACTS),
        scenario_contract_manifest(AGENT_SCENARIO_CONTRACTS),
    ]
}

fn scenario_contract_manifest(
    contracts: &'static [ScenarioContractSpec],
) -> ScenarioContractManifest {
    let first = contracts
        .first()
        .expect("scenario contract manifest must not be empty");
    ScenarioContractManifest {
        suite: first.suite,
        oracle_id: first.oracle_id,
        contract_count: contracts.len(),
        required_sim_evidence: first.required_sim_evidence.to_vec(),
        contracts: contracts.to_vec(),
    }
}

fn all_scenario_contracts() -> impl Iterator<Item = &'static ScenarioContractSpec> {
    [
        RUNTIME_SCENARIO_CONTRACTS,
        STANDARD_PROTOCOL_SCENARIO_CONTRACTS,
        RLM_PROTOCOL_SCENARIO_CONTRACTS,
        AGENT_SCENARIO_CONTRACTS,
    ]
    .into_iter()
    .flat_map(|contracts| contracts.iter())
}

fn scenario_contract_oracle_id(contract: &ScenarioContractSpec) -> String {
    format!("{}:{}", contract.oracle_id, contract.test_name)
}

fn scenario_contract_backing_oracle_id(contract: &ScenarioContractSpec) -> String {
    scenario_contract_oracle_id(contract)
}

fn scenario_contract_classification(_contract: &ScenarioContractSpec) -> &'static str {
    "per_contract_oracle"
}

fn write_scenario_contract_slices(
    artifact_root: &Path,
    event_lines: &[TraceEventLine],
    oracle_verdicts: &[OracleVerdict],
) -> Result<Vec<ScenarioContractSliceManifest>, FixedScriptRunnerError> {
    let slice_root = artifact_root.join(GENERATED_SIM_SCENARIO_SLICES);
    std::fs::create_dir_all(&slice_root)?;
    let mut manifests = Vec::new();
    for contract in all_scenario_contracts() {
        let oracle_id = scenario_contract_backing_oracle_id(contract);
        let classification = scenario_contract_classification(contract);
        let verdicts = oracle_verdicts
            .iter()
            .filter(|verdict| verdict.oracle_id == oracle_id)
            .cloned()
            .collect::<Vec<_>>();
        if verdicts.is_empty() {
            return Err(FixedScriptRunnerError::Assertion(format!(
                "scenario contract `{}` had no generated backing oracle verdict ({oracle_id})",
                contract.test_name
            )));
        }
        let selected_evidence = select_scenario_contract_evidence(contract, event_lines)?;
        let mut selected_keys = BTreeSet::new();
        let mut events = Vec::new();
        for evidence in &selected_evidence {
            if selected_keys.insert((evidence.trace_alias.clone(), evidence.boundary_id.clone())) {
                let event_line = event_lines
                    .iter()
                    .find(|line| {
                        line.trace_alias == evidence.trace_alias
                            && line.event.boundary_id == evidence.boundary_id
                    })
                    .ok_or_else(|| {
                        FixedScriptRunnerError::Assertion(format!(
                            "scenario contract `{}` selected missing event `{}`",
                            contract.test_name, evidence.boundary_id
                        ))
                    })?;
                events.push(event_line.clone());
            }
        }
        let generated_shape = scenario_generated_shape(contract, &selected_evidence, &events)?;
        let suite_dir = slice_root.join(contract.suite);
        std::fs::create_dir_all(&suite_dir)?;
        let artifact_path = suite_dir.join(format!("{}.json", contract.test_name));
        let artifact = ScenarioContractSliceArtifact {
            schema: "lash.sim.scenario-contract-slice.v1",
            contract: *contract,
            oracle_id: oracle_id.clone(),
            classification,
            semantic_oracle: contract.semantic_oracle,
            generated_shape: generated_shape.clone(),
            selected_evidence: selected_evidence.clone(),
            events,
            verdicts,
        };
        std::fs::write(&artifact_path, serde_json::to_vec_pretty(&artifact)?)?;
        manifests.push(ScenarioContractSliceManifest {
            schema: "lash.sim.scenario-contract-slice-manifest.v1",
            suite: contract.suite,
            test_name: contract.test_name,
            semantic_oracle: contract.semantic_oracle,
            oracle_id,
            classification,
            status: "generated_trace_slice_written",
            artifact_path: relative_path(artifact_root, &artifact_path),
            generated_shape,
            selected_event_count: artifact.selected_evidence.len(),
            selected_evidence,
        });
    }
    Ok(manifests)
}

fn write_scenario_contract_packages(
    artifact_root: &Path,
    event_lines: &[TraceEventLine],
    oracle_verdicts: &[OracleVerdict],
    replay_reports: &[GeneratedReplayArtifact],
) -> Result<Vec<ScenarioContractPackageManifest>, FixedScriptRunnerError> {
    let package_root = artifact_root.join(GENERATED_SIM_SCENARIO_PACKAGES);
    std::fs::create_dir_all(&package_root)?;
    let replay_lookup = replay_artifact_lookup(replay_reports);
    let mut manifests = Vec::new();
    let mut backing_oracle_claims = BTreeMap::new();
    let mut package_fact_graph_claims = BTreeMap::new();
    for contract in all_scenario_contracts() {
        let oracle_id = scenario_contract_backing_oracle_id(contract);
        let classification = scenario_contract_classification(contract);
        if let Some(previous) = backing_oracle_claims.insert(oracle_id.clone(), contract.test_name)
        {
            return Err(FixedScriptRunnerError::Assertion(format!(
                "scenario contracts `{previous}` and `{}` share backing oracle `{oracle_id}` despite claiming per-contract generated semantics",
                contract.test_name
            )));
        }
        let verdicts = oracle_verdicts
            .iter()
            .filter(|verdict| verdict.oracle_id == oracle_id)
            .cloned()
            .collect::<Vec<_>>();
        if verdicts.is_empty() {
            return Err(FixedScriptRunnerError::Assertion(format!(
                "scenario contract `{}` had no generated backing oracle verdict for package ({oracle_id})",
                contract.test_name
            )));
        }
        let selected_evidence = select_scenario_contract_evidence(contract, event_lines)?;
        let events = selected_events(contract, event_lines, &selected_evidence)?;
        let generated_shape = scenario_generated_shape(contract, &selected_evidence, &events)?;
        assert_distinct_package_fact_graph(
            contract,
            &generated_shape,
            &mut package_fact_graph_claims,
        )?;
        let positive =
            scenario_positive_evidence(contract, &selected_evidence, &verdicts, &replay_lookup)?;
        let negative = scenario_negative_evidence(&generated_shape.negative_fixture);
        let operational_cases = scenario_operational_cases(contract, &selected_evidence);
        let package_id = format!("{}.{}", contract.suite, contract.test_name);
        let suite_dir = package_root.join(contract.suite).join(contract.test_name);
        std::fs::create_dir_all(&suite_dir)?;
        let package_path = suite_dir.join("package.json");
        let artifact = ScenarioContractPackageArtifact {
            schema: "lash.sim.scenario-contract-package.v1",
            package_id: package_id.clone(),
            contract: *contract,
            oracle_id: oracle_id.clone(),
            classification,
            semantic_oracle: contract.semantic_oracle,
            operational_cases: operational_cases.clone(),
            generated_shape: generated_shape.clone(),
            positive: positive.clone(),
            negative: negative.clone(),
            selected_evidence: selected_evidence.clone(),
            events,
            verdicts,
        };
        std::fs::write(&package_path, serde_json::to_vec_pretty(&artifact)?)?;
        manifests.push(ScenarioContractPackageManifest {
            schema: "lash.sim.scenario-contract-package-manifest.v1",
            package_id,
            suite: contract.suite,
            test_name: contract.test_name,
            semantic_oracle: contract.semantic_oracle,
            transition_kind: generated_shape.transition_kind,
            oracle_id,
            classification,
            status: "generated_replay_package_written",
            package_path: relative_path(artifact_root, &package_path),
            operational_cases,
            positive,
            negative,
        });
    }
    Ok(manifests)
}

fn assert_distinct_package_fact_graph(
    contract: &ScenarioContractSpec,
    generated_shape: &ScenarioGeneratedShape,
    seen: &mut BTreeMap<Vec<(String, Vec<String>)>, &'static str>,
) -> Result<(), FixedScriptRunnerError> {
    let fingerprint = transition_fact_graph_fingerprint(&generated_shape.transition_facts);
    if let Some(previous) = seen.insert(fingerprint, contract.test_name) {
        return Err(FixedScriptRunnerError::Assertion(format!(
            "scenario packages `{previous}` and `{}` produced identical generated transition fact graphs despite claiming distinct per-contract semantics",
            contract.test_name
        )));
    }
    Ok(())
}

fn transition_fact_graph_fingerprint(
    transition_facts: &[ScenarioTransitionFact],
) -> Vec<(String, Vec<String>)> {
    transition_facts
        .iter()
        .map(|fact| (fact.fact.clone(), fact.boundary_ids.clone()))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

struct BackendRegressionSpec {
    fixture_id: &'static str,
    required_boundary_kinds: &'static [&'static str],
    semantic_oracles: &'static [&'static str],
    regression_contract: &'static str,
    predicate: fn(&[&TraceEventLine]) -> bool,
}

fn write_generated_backend_regression_fixtures(
    artifact_root: &Path,
    event_lines: &[TraceEventLine],
    replay_reports: &[GeneratedReplayArtifact],
) -> Result<Vec<GeneratedBackendRegressionManifest>, FixedScriptRunnerError> {
    let fixture_root = artifact_root.join(GENERATED_SIM_BACKEND_REGRESSION_FIXTURES);
    std::fs::create_dir_all(&fixture_root)?;
    let replay_lookup = replay_artifact_lookup(replay_reports);
    let mut by_trace: BTreeMap<String, Vec<&TraceEventLine>> = BTreeMap::new();
    for line in event_lines {
        by_trace
            .entry(line.trace_alias.clone())
            .or_default()
            .push(line);
    }

    let specs = [
        BackendRegressionSpec {
            fixture_id: "queued-active-turn-cancel-race",
            required_boundary_kinds: &["queued_ingress", "cancellation", "provider"],
            semantic_oracles: &[
                "sim.oracle.state-machine-semantic-invariants.v1",
                "sim.oracle.scenario-mini.runtime.queued-input-hidden-while-live.v1",
                "sim.oracle.scenario-mini.runtime.cancellation-prevents-idle-claim.v1",
            ],
            regression_contract: "active-turn queued input stays hidden, then cancellation terminalizes the pending row before any later idle claim can surface it",
            predicate: trace_has_queued_cancel_race,
        },
        BackendRegressionSpec {
            fixture_id: "trigger-wakeup-routes-process",
            required_boundary_kinds: &["trigger"],
            semantic_oracles: &["sim.oracle.state-machine-semantic-invariants.v1"],
            regression_contract: "trigger occurrence records a stable source key, reserves a matching delivery, and starts process wake routing without live external input",
            predicate: trace_has_trigger_wakeup_route,
        },
        BackendRegressionSpec {
            fixture_id: "duplicate-process-wake-idempotency",
            required_boundary_kinds: &["process_wake"],
            semantic_oracles: &[
                "sim.oracle.state-machine-semantic-invariants.v1",
                "sim.oracle.process-wake-observed.v1",
            ],
            regression_contract: "duplicate process wake deliveries share a dedupe key, claim queued work once, and keep replay/idempotency evidence backed by generated dynamic replay",
            predicate: trace_has_duplicate_process_wake_idempotency,
        },
        BackendRegressionSpec {
            fixture_id: "worker-stale-completion-fenced",
            required_boundary_kinds: &["worker"],
            semantic_oracles: &[
                "sim.oracle.state-machine-semantic-invariants.v1",
                "sim.oracle.scenario-mini.runtime.stale-lease-commit-rejected.v1",
            ],
            regression_contract: "stale worker completion carries an older fence and is rejected while the live incarnation remains active",
            predicate: trace_has_worker_stale_completion,
        },
        BackendRegressionSpec {
            fixture_id: "durable-effect-crash-reopen-replay",
            required_boundary_kinds: &["durable_effect"],
            semantic_oracles: &["sim.oracle.state-machine-semantic-invariants.v1"],
            regression_contract: "durable effect first execution calls the local executor once and crash/reopen-style replay returns stored history without re-executing",
            predicate: trace_has_durable_effect_replay,
        },
        BackendRegressionSpec {
            fixture_id: "backend-retry-terminalization",
            required_boundary_kinds: &["backend_failure"],
            semantic_oracles: &["sim.oracle.state-machine-semantic-invariants.v1"],
            regression_contract: "retryable backend conflicts advance attempts and terminate on a non-retryable production StoreError class",
            predicate: trace_has_backend_retry_terminalization,
        },
        BackendRegressionSpec {
            fixture_id: "provider-protocol-terminalization",
            required_boundary_kinds: &["provider_mutation"],
            semantic_oracles: &["sim.oracle.state-machine-semantic-invariants.v1"],
            regression_contract: "scripted provider mutation matrices classify retryable 429 and dropped-terminal parser failures through every migrated provider parser",
            predicate: trace_has_provider_protocol_terminalization,
        },
        BackendRegressionSpec {
            fixture_id: "rlm-standard-protocol-terminal-boundaries",
            required_boundary_kinds: &["provider", "provider_mutation", "exec_code"],
            semantic_oracles: &[
                "sim.oracle.state-machine-semantic-invariants.v1",
                "sim.oracle.scenario.standard-contract.v1:standard_protocol_scenario_provider_error_stops_without_checkpoint",
                "sim.oracle.scenario.rlm-contract.v1:rlm_protocol_scenario_exec_any_tool_control_fail_is_terminal_error",
                "sim.oracle.scenario.rlm-contract.v1:rlm_protocol_scenario_exec_any_tool_control_frame_switch_is_terminal",
            ],
            regression_contract: "standard provider-error terminalization and RLM exec terminal boundaries stay represented by generated transitions with dynamic backend evidence",
            predicate: trace_has_protocol_terminal_boundary_mix,
        },
    ];

    let mut manifests = Vec::new();
    for spec in specs {
        let Some((trace_alias, _lines)) = by_trace
            .iter()
            .find(|(_alias, lines)| (spec.predicate)(lines))
        else {
            return Err(FixedScriptRunnerError::Assertion(format!(
                "generated backend regression fixture `{}` could not select a generated trace",
                spec.fixture_id
            )));
        };
        let replay = replay_lookup.get(trace_alias).ok_or_else(|| {
            FixedScriptRunnerError::Assertion(format!(
                "generated backend regression fixture `{}` selected trace `{trace_alias}` without replay artifact",
                spec.fixture_id
            ))
        })?;
        let package_dir = fixture_root.join(spec.fixture_id);
        std::fs::create_dir_all(&package_dir)?;
        let source_trace_path = artifact_root.join(&replay.trace_path);
        let fixture_trace_path = package_dir.join("trace.json");
        std::fs::copy(&source_trace_path, &fixture_trace_path)?;
        let package_path = package_dir.join("package.json");
        let static_backend_replay_policy = "not_claimed_for_generated_scheduler_traces";
        let backend_equivalence_contract = "source seed passed the dynamic generated workload rerun against the serialized in-memory reference and lash-sqlite-store; static SQLite/Postgres replay is a different fixed-order trace contract and is not inferred from this generated trace";
        let package = GeneratedBackendRegressionPackage {
            schema: "lash.sim.generated-backend-regression-package.v1",
            fixture_id: spec.fixture_id,
            status: "generated_cross_backend_valid_trace",
            trace: "trace.json",
            source_trace_path: replay.trace_path.clone(),
            source_trace_sha256: replay.trace_sha256.clone(),
            source_sqlite_replay_report_path: replay.sqlite_replay_report_path.clone(),
            source_sqlite_replay_report_sha256: replay.sqlite_replay_report_sha256.clone(),
            required_boundary_kinds: spec.required_boundary_kinds.to_vec(),
            semantic_oracles: spec.semantic_oracles.to_vec(),
            replay_backends: vec!["model"],
            static_backend_replay_policy,
            backend_equivalence_contract,
            regression_contract: spec.regression_contract,
            replay_command: format!(
                "cargo run -p lash-sim --locked -- replay {}",
                fixture_trace_path.display()
            ),
        };
        std::fs::write(&package_path, serde_json::to_vec_pretty(&package)?)?;
        manifests.push(GeneratedBackendRegressionManifest {
            schema: "lash.sim.generated-backend-regression-manifest.v1",
            fixture_id: spec.fixture_id,
            status: "generated_cross_backend_valid_trace",
            package_path: relative_path(artifact_root, &package_path),
            trace_path: relative_path(artifact_root, &fixture_trace_path),
            source_trace_path: replay.trace_path.clone(),
            source_trace_sha256: replay.trace_sha256.clone(),
            source_sqlite_replay_report_path: replay.sqlite_replay_report_path.clone(),
            source_sqlite_replay_report_sha256: replay.sqlite_replay_report_sha256.clone(),
            required_boundary_kinds: spec.required_boundary_kinds.to_vec(),
            semantic_oracles: spec.semantic_oracles.to_vec(),
            replay_backends: vec!["model"],
            static_backend_replay_policy,
            backend_equivalence_contract,
            regression_contract: spec.regression_contract,
        });
    }
    Ok(manifests)
}

fn trace_has_queued_cancel_race(lines: &[&TraceEventLine]) -> bool {
    lines
        .iter()
        .filter(|line| line.event.kind == BoundaryKind::QueuedIngress)
        .any(|queued| {
            let Some(source_key) = queued
                .event
                .payload
                .get("source_key")
                .and_then(Value::as_str)
            else {
                return false;
            };
            queued
                .event
                .payload
                .get("ingress_mode")
                .and_then(Value::as_str)
                == Some("active_turn")
                && queued
                    .event
                    .observed
                    .get("source_key")
                    .and_then(Value::as_str)
                    == Some(source_key)
                && queued
                    .event
                    .observed
                    .get("input_state")
                    .and_then(Value::as_str)
                    .is_some_and(|state| state.starts_with("pending"))
                && lines.iter().any(|line| {
                    line.event.kind == BoundaryKind::Cancellation
                        && line.event.sequence > queued.event.sequence
                        && line.event.observed.get("target").and_then(Value::as_str)
                            == Some(queued.event.boundary_id.as_str())
                        && line
                            .event
                            .observed
                            .get("cancelled")
                            .and_then(Value::as_bool)
                            == Some(true)
                })
        })
}

fn trace_has_trigger_wakeup_route(lines: &[&TraceEventLine]) -> bool {
    lines
        .iter()
        .filter(|line| line.event.kind == BoundaryKind::Trigger)
        .any(|line| {
            line.event
                .observed
                .get("trigger_delivered")
                .and_then(Value::as_bool)
                == Some(true)
                && line
                    .event
                    .observed
                    .get("started_process")
                    .and_then(Value::as_bool)
                    == Some(true)
                && line
                    .event
                    .observed
                    .get("reservation_count")
                    .and_then(Value::as_u64)
                    .is_some_and(|count| count > 0)
        })
}

fn trace_has_duplicate_process_wake_idempotency(lines: &[&TraceEventLine]) -> bool {
    let mut by_dedupe_key: BTreeMap<String, Vec<&TraceEventLine>> = BTreeMap::new();
    for line in lines
        .iter()
        .filter(|line| line.event.kind == BoundaryKind::ProcessWake)
    {
        let Some(dedupe_key) = line
            .event
            .observed
            .get("dedupe_key")
            .and_then(Value::as_str)
        else {
            continue;
        };
        by_dedupe_key
            .entry(dedupe_key.to_string())
            .or_default()
            .push(*line);
    }
    by_dedupe_key.values().any(|events| {
        let strict_claim_dedupe = events.len() >= 2
            && events.iter().any(|line| {
                line.event
                    .observed
                    .get("claimed_once")
                    .and_then(Value::as_bool)
                    == Some(true)
                    && line
                        .event
                        .observed
                        .pointer("/runtime_queued_work/claimed")
                        .and_then(Value::as_bool)
                        == Some(true)
            })
            && events.iter().all(|line| {
                line.event.observed.get("runtime_process_wake").is_some()
                    && line.event.observed.get("runtime_queued_work").is_some()
            });
        let in_flight_rejection = events.len() >= 2
            && events.iter().any(|line| {
                line.event
                    .observed
                    .get("lease_busy")
                    .and_then(Value::as_bool)
                    == Some(true)
                    && line
                        .event
                        .observed
                        .pointer("/runtime_queued_work/enqueued")
                        .and_then(Value::as_bool)
                        == Some(false)
                    && line.event.observed.get("runtime_process_wake").is_some()
                    && line.event.observed.get("runtime_queued_work").is_some()
            });
        strict_claim_dedupe || in_flight_rejection
    })
}

fn trace_has_worker_stale_completion(lines: &[&TraceEventLine]) -> bool {
    lines
        .iter()
        .filter(|line| line.event.kind == BoundaryKind::Worker)
        .any(|line| {
            line.event
                .observed
                .get("stale_completion_rejected")
                .and_then(Value::as_bool)
                == Some(true)
                && line.event.observed.get("runtime_active_lease").is_some()
                && line
                    .event
                    .observed
                    .get("runtime_stale_completion")
                    .is_some()
        })
}

fn trace_has_durable_effect_replay(lines: &[&TraceEventLine]) -> bool {
    let mut by_key: BTreeMap<String, Vec<&TraceEventLine>> = BTreeMap::new();
    for line in lines
        .iter()
        .filter(|line| line.event.kind == BoundaryKind::DurableEffect)
    {
        let Some(key) = line
            .event
            .observed
            .get("durable_key")
            .and_then(Value::as_str)
        else {
            continue;
        };
        by_key.entry(key.to_string()).or_default().push(*line);
    }
    by_key.values().any(|events| {
        let first = events.iter().any(|line| {
            line.event.observed.get("replayed").and_then(Value::as_bool) == Some(false)
                && line
                    .event
                    .observed
                    .pointer("/runtime_effect/local_executor_called")
                    .and_then(Value::as_bool)
                    == Some(true)
        });
        let replay = events.iter().any(|line| {
            line.event.observed.get("replayed").and_then(Value::as_bool) == Some(true)
                && line
                    .event
                    .observed
                    .pointer("/runtime_effect/local_executor_called")
                    .and_then(Value::as_bool)
                    == Some(false)
                && line
                    .event
                    .observed
                    .get("execution_count")
                    .and_then(Value::as_u64)
                    == Some(1)
        });
        first && replay
    })
}

fn trace_has_backend_retry_terminalization(lines: &[&TraceEventLine]) -> bool {
    let mut by_operation: BTreeMap<String, Vec<&TraceEventLine>> = BTreeMap::new();
    for line in lines
        .iter()
        .filter(|line| line.event.kind == BoundaryKind::BackendFailure)
    {
        let operation = line
            .event
            .observed
            .get("operation")
            .or_else(|| line.event.payload.get("operation"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        by_operation.entry(operation).or_default().push(*line);
    }
    by_operation.values().any(|events| {
        let retryable = events.iter().any(|line| {
            line.event
                .observed
                .get("retryable")
                .and_then(Value::as_bool)
                == Some(true)
                && line
                    .event
                    .observed
                    .get("attempt")
                    .and_then(Value::as_u64)
                    .is_some_and(|attempt| attempt > 0)
        });
        let terminal = events.iter().any(|line| {
            line.event
                .observed
                .get("retryable")
                .and_then(Value::as_bool)
                == Some(false)
                && line
                    .event
                    .observed
                    .get("store_error_class")
                    .and_then(Value::as_str)
                    == Some("terminal_backend_error")
        });
        retryable && terminal
    })
}

fn trace_has_protocol_terminal_boundary_mix(lines: &[&TraceEventLine]) -> bool {
    let has_provider_terminal = lines.iter().any(|line| {
        line.event.kind == BoundaryKind::Provider
            && line
                .event
                .observed
                .pointer("/runtime_contract/status")
                .and_then(Value::as_str)
                == Some("passed")
            && line.event.observed.get("success").and_then(Value::as_bool) == Some(true)
    });
    let has_provider_failure_terminal = trace_has_provider_protocol_terminalization(lines);
    let has_exec_terminal = lines.iter().any(|line| {
        line.event.kind == BoundaryKind::ExecCode
            && line.event.observed.get("runtime_effect_outcome").is_some()
            && line
                .event
                .observed
                .pointer("/runtime_effect/kind")
                .and_then(Value::as_str)
                == Some("exec_code")
    });
    has_provider_terminal && has_provider_failure_terminal && has_exec_terminal
}

fn trace_has_provider_protocol_terminalization(lines: &[&TraceEventLine]) -> bool {
    let mutations = lines
        .iter()
        .filter(|line| line.event.kind == BoundaryKind::ProviderMutation)
        .filter_map(|line| {
            line.event
                .observed
                .get("mutation")
                .or_else(|| line.event.payload.get("mutation"))
                .and_then(Value::as_str)
        })
        .collect::<BTreeSet<_>>();
    mutations.contains("rate_limit_error_envelope")
        && mutations.contains("dropped_terminal_event")
        && lines.iter().any(|line| {
            line.event.kind == BoundaryKind::ProviderMutation
                && line
                    .event
                    .observed
                    .pointer("/provider_parser_matrix/matrix/real_provider_parser_execution")
                    .and_then(Value::as_bool)
                    == Some(true)
        })
}

fn selected_events(
    contract: &ScenarioContractSpec,
    event_lines: &[TraceEventLine],
    selected_evidence: &[ScenarioEvidenceSelection],
) -> Result<Vec<TraceEventLine>, FixedScriptRunnerError> {
    let mut selected_keys = BTreeSet::new();
    let mut events = Vec::new();
    for evidence in selected_evidence {
        if selected_keys.insert((evidence.trace_alias.clone(), evidence.boundary_id.clone())) {
            let event_line = event_lines
                .iter()
                .find(|line| {
                    line.trace_alias == evidence.trace_alias
                        && line.event.boundary_id == evidence.boundary_id
                })
                .ok_or_else(|| {
                    FixedScriptRunnerError::Assertion(format!(
                        "scenario contract `{}` selected missing event `{}`",
                        contract.test_name, evidence.boundary_id
                    ))
                })?;
            events.push(event_line.clone());
        }
    }
    Ok(events)
}

fn replay_artifact_lookup(
    replay_reports: &[GeneratedReplayArtifact],
) -> BTreeMap<String, &GeneratedReplayArtifact> {
    replay_reports
        .iter()
        .map(|replay| (format!("seed-{:016x}", replay.seed), replay))
        .collect()
}

fn scenario_positive_evidence(
    contract: &ScenarioContractSpec,
    selected_evidence: &[ScenarioEvidenceSelection],
    verdicts: &[OracleVerdict],
    replay_lookup: &BTreeMap<String, &GeneratedReplayArtifact>,
) -> Result<ScenarioPositiveEvidence, FixedScriptRunnerError> {
    let mut source_trace_aliases = BTreeSet::new();
    let mut selected_boundary_ids = BTreeSet::new();
    for evidence in selected_evidence {
        source_trace_aliases.insert(evidence.trace_alias.clone());
        selected_boundary_ids.insert(evidence.boundary_id.clone());
    }
    let mut source_trace_paths = Vec::new();
    let mut replay_report_paths = Vec::new();
    let mut sqlite_replay_report_paths = Vec::new();
    for alias in &source_trace_aliases {
        let replay = replay_lookup.get(alias).ok_or_else(|| {
            FixedScriptRunnerError::Assertion(format!(
                "scenario contract `{}` selected trace `{alias}` without replay artifact",
                contract.test_name
            ))
        })?;
        source_trace_paths.push(replay.trace_path.clone());
        replay_report_paths.push(replay.replay_report_path.clone());
        sqlite_replay_report_paths.push(replay.sqlite_replay_report_path.clone());
    }
    let verdict = verdicts
        .iter()
        .find(|verdict| verdict.status == OracleStatus::Passed)
        .unwrap_or_else(|| {
            verdicts
                .first()
                .expect("scenario package requires at least one verdict")
        });
    Ok(ScenarioPositiveEvidence {
        schema: "lash.sim.scenario-contract-positive-evidence.v1",
        source_trace_aliases: source_trace_aliases.into_iter().collect(),
        source_trace_paths,
        replay_report_paths,
        sqlite_replay_report_paths,
        selected_boundary_ids: selected_boundary_ids.into_iter().collect(),
        selected_event_count: selected_evidence.len(),
        oracle_status: verdict.status.clone(),
        oracle_reason: verdict.message.clone(),
    })
}

fn scenario_negative_evidence(fixture: &ScenarioNegativeFixture) -> ScenarioNegativeEvidence {
    ScenarioNegativeEvidence {
        schema: "lash.sim.scenario-contract-negative-evidence.v1",
        fixture_id: fixture.fixture_id,
        fixture_path: fixture.fixture_path,
        expected_oracle_id: fixture.expected_oracle_id,
        expected_reason_contains: fixture.expected_reason_contains,
        minimize_command: format!(
            "cargo run -p lash-sim --locked -- minimize {} --out <artifact-root>/{}",
            fixture.fixture_path, fixture.fixture_id
        ),
        minimized_package_path: format!(
            "sim/failing-fixtures/{}/minimized-regression/package.json",
            fixture.fixture_id
        ),
    }
}

fn scenario_operational_cases(
    contract: &ScenarioContractSpec,
    selected_evidence: &[ScenarioEvidenceSelection],
) -> Vec<String> {
    let mut cases = BTreeSet::new();
    cases.insert(format!("scenario-suite:{}", contract.suite));
    cases.insert(format!("semantic:{}", contract.semantic_oracle));
    cases.insert("protocol-state-transition".to_string());
    for evidence in selected_evidence {
        for case in operational_cases_for_evidence(&evidence.evidence) {
            cases.insert(case.to_string());
        }
    }
    for case in operational_cases_for_semantic(contract.semantic_oracle) {
        cases.insert(case.to_string());
    }
    cases.into_iter().collect()
}

fn operational_cases_for_evidence(evidence: &str) -> &'static [&'static str] {
    match evidence {
        "queued_ingress" => &[
            "queueing-inputs",
            "active-turn-input-queueing",
            "duplicate-replayed-inputs",
        ],
        "cancellation" => &["cancellation"],
        "process_wake" => &["triggers-wakeups", "process-wake", "duplicate-delivery"],
        "worker_stale_completion" => &["worker-failover", "lease-fencing", "stale-completion"],
        "provider_turn" => &["provider-runtime-turn", "scripted-provider-transport"],
        "provider_event" => &["provider-runtime-turn", "scheduler-owned-provider-events"],
        "tool_result" => &["tool-boundary", "tool-loop"],
        "max_turn_stop" => &["tool-boundary", "max-turn-stop"],
        "final_value" => &["semantic-final-value", "final-value-event"],
        "exec_code" => &["exec-boundary", "rlm-lashlang-exec"],
        "trigger" => &["triggers-wakeups", "trigger-delivery"],
        "durable_effect" => &["durable-effect", "crash-reopen-effect-replay"],
        "provider_mutation" => &[
            "provider-failure",
            "provider-mutation",
            "real-provider-parser",
        ],
        "backend_failure" => &["backend-retry", "persistence-fault"],
        "observer_reconnect" => &["observer-reconnect"],
        "observer_convergence" => &["observer-convergence"],
        "lease_time" => &["lease-fencing", "scheduler-time"],
        "multi_session" => &["multi-session"],
        "runtime_session_graph" => &["runtime-session-graph"],
        _ => &[],
    }
}

fn operational_cases_for_semantic(semantic_oracle: &str) -> &'static [&'static str] {
    match semantic_oracle {
        "runtime.checkpoint_redrive_cancel" => &[
            "active-turn-input-queueing",
            "cancellation",
            "duplicate-replayed-inputs",
        ],
        "runtime.queued_work_keeps_pending_input" | "runtime.queued_turn_input_completion" => {
            &["queueing-inputs", "active-turn-input-queueing"]
        }
        "runtime.process_wake_claim" => &["triggers-wakeups", "duplicate-delivery"],
        "runtime.lease_release_rejects_commit" | "runtime.dead_lease_reclaim_rejects_stale" => {
            &["lease-fencing", "worker-failover", "stale-completion"]
        }
        "standard.empty_provider_response_error" | "standard.provider_error_without_checkpoint" => {
            &["provider-failure", "retry-exhaustion"]
        }
        "standard.streamed_text_finalizes_once" => &["duplicate-free-stream-finalization"],
        "standard.native_tool_loop_reenters_model"
        | "standard.tool_failure_feedback_reenters_model"
        | "standard.parallel_tool_results_checkpoint_once" => &["tool-loop"],
        "rlm.lashlang_cell_exec_continues" => &["rlm-lashlang-exec", "triggers-wakeups"],
        "rlm.typed_schema_mismatch_repair_loop" | "rlm.typed_schema_any_of_mismatch" => {
            &["provider-failure", "repair-loop"]
        }
        semantic if semantic.starts_with("rlm.") => &["rlm-protocol-transition"],
        "agent.durable_input_suspension_resolution" => {
            &["durable-effect", "process-wake", "observer-reconnect"]
        }
        semantic if semantic.starts_with("agent.") => &["agent-process-graph", "worker-failover"],
        _ => &[],
    }
}

fn scenario_generated_shape(
    contract: &ScenarioContractSpec,
    selected_evidence: &[ScenarioEvidenceSelection],
    selected_events: &[TraceEventLine],
) -> Result<ScenarioGeneratedShape, FixedScriptRunnerError> {
    let mut evidence_names = BTreeSet::new();
    let required_evidence = contract
        .required_sim_evidence
        .iter()
        .copied()
        .chain(semantic_scenario_evidence(contract.semantic_oracle))
        .filter(|evidence| evidence_names.insert(*evidence))
        .map(|evidence| ScenarioRequiredEvidence {
            evidence: evidence.to_string(),
            boundary_kind: scenario_evidence_boundary_kind(evidence).to_string(),
            assertion: scenario_evidence_assertion(evidence),
            selected_event_count: selected_evidence
                .iter()
                .filter(|selection| selection.evidence == evidence)
                .count(),
        })
        .collect();
    Ok(ScenarioGeneratedShape {
        schema: "lash.sim.scenario-contract-generated-shape.v1",
        transition_kind: scenario_transition_kind(contract).to_string(),
        semantic_oracle: contract.semantic_oracle,
        required_evidence,
        transition_facts: scenario_transition_facts(contract, selected_events)?,
        generated_backend_regression: scenario_backend_regression_reference(contract),
        negative_fixture: scenario_negative_fixture_for_contract(contract, selected_evidence),
    })
}

fn scenario_transition_facts(
    contract: &ScenarioContractSpec,
    selected_events: &[TraceEventLine],
) -> Result<Vec<ScenarioTransitionFact>, FixedScriptRunnerError> {
    if matches!(contract.suite, "standard" | "rlm" | "agent") {
        let events = selected_events
            .iter()
            .map(|line| line.event.clone())
            .collect::<Vec<_>>();
        return scenario_contract_generated_facts(contract, &events)
            .map(|facts| {
                facts
                    .into_iter()
                    .map(|fact| ScenarioTransitionFact {
                        fact: fact.fact.to_string(),
                        status: "passed",
                        assertion: fact.assertion,
                        boundary_ids: fact.boundary_ids,
                        observed: fact.observed,
                    })
                    .collect()
            })
            .map_err(|reason| {
                FixedScriptRunnerError::Assertion(format!(
                    "scenario contract `{}` could not prove generated semantic facts: {reason}",
                    contract.test_name
                ))
            });
    }
    let mut facts = Vec::new();
    match contract.semantic_oracle {
        "runtime.checkpoint_redrive_cancel" => {
            facts.push(queued_active_turn_fact(contract, selected_events)?);
            facts.push(cancellation_terminalization_fact(
                contract,
                selected_events,
            )?);
        }
        "runtime.queued_work_keeps_pending_input" => {
            facts.push(queued_active_turn_fact(contract, selected_events)?);
        }
        "runtime.queued_turn_input_completion" => {
            facts.push(queued_active_turn_fact(contract, selected_events)?);
            facts.push(queued_turn_followup_provider_fact(
                contract,
                selected_events,
            )?);
        }
        "runtime.command_only_queue_drain" => {
            facts.push(command_queue_drain_fact(contract, selected_events)?);
        }
        "runtime.command_before_turn_work" => {
            facts.push(trigger_wakeup_fact(contract, selected_events)?);
            facts.push(queued_active_turn_fact(contract, selected_events)?);
        }
        "runtime.process_wake_claim" => {
            facts.push(process_wake_duplicate_fact(contract, selected_events)?);
        }
        "runtime.lease_release_rejects_commit" => {
            facts.push(worker_stale_completion_rejection_fact(
                contract,
                selected_events,
            )?);
        }
        "runtime.dead_lease_reclaim_rejects_stale" => {
            facts.push(worker_dead_lease_reclaim_fact(contract, selected_events)?);
        }
        "runtime.observation_replay_preserves_input" => {
            facts.push(observer_reconnect_transition_fact(
                contract,
                selected_events,
            )?);
        }
        "standard.empty_provider_response_error" | "standard.provider_error_without_checkpoint" => {
            facts.push(provider_terminalization_fact(contract, selected_events)?);
        }
        "standard.streamed_text_finalizes_once" | "standard.initial_request_projection" => {
            facts.push(provider_success_terminal_fact(contract, selected_events)?);
        }
        "standard.native_tool_loop_reenters_model"
        | "standard.tool_failure_feedback_reenters_model"
        | "standard.max_turns_after_tool_result"
        | "standard.parallel_tool_results_checkpoint_once" => {
            facts.push(tool_result_fact(contract, selected_events)?);
            facts.push(provider_success_terminal_fact(contract, selected_events)?);
        }
        "rlm.lashlang_cell_exec_continues"
        | "rlm.exec_error_max_turn_stop"
        | "rlm.exec_tool_control_frame_switch_terminal"
        | "rlm.exec_tool_control_fail_terminal"
        | "rlm.exec_result_no_tool_call_replay" => {
            facts.push(exec_terminal_fact(contract, selected_events)?);
        }
        "rlm.typed_schema_mismatch_repair_loop" | "rlm.typed_schema_any_of_mismatch" => {
            facts.push(provider_terminalization_fact(contract, selected_events)?);
            facts.push(provider_success_terminal_fact(contract, selected_events)?);
        }
        semantic if semantic.starts_with("rlm.") => {
            facts.push(provider_success_terminal_fact(contract, selected_events)?);
        }
        "agent.durable_input_suspension_resolution" => {
            facts.push(durable_effect_replay_fact(contract, selected_events)?);
            facts.push(process_wake_duplicate_fact(contract, selected_events)?);
        }
        semantic if semantic.starts_with("agent.") => {
            facts.push(process_wake_duplicate_fact(contract, selected_events)?);
        }
        _ => {}
    }
    if facts.is_empty() {
        if contract.suite == "runtime" {
            return Err(FixedScriptRunnerError::Assertion(format!(
                "runtime scenario contract `{}` has no contract-owned transition fact; generic selected-event fallback is not allowed for runtime contracts",
                contract.test_name
            )));
        }
        facts.push(generic_transition_fact(contract, selected_events)?);
    }
    Ok(facts)
}

fn scenario_backend_regression_reference(
    contract: &ScenarioContractSpec,
) -> Option<ScenarioBackendRegressionReference> {
    let (fixture_id, regression_contract) = match contract.semantic_oracle {
        "runtime.checkpoint_redrive_cancel"
        | "runtime.queued_work_keeps_pending_input"
        | "runtime.queued_turn_input_completion" => (
            "queued-active-turn-cancel-race",
            "active-turn queued input stays hidden, then cancellation terminalizes the pending row before any later idle claim can surface it",
        ),
        "runtime.command_before_turn_work" => (
            "trigger-wakeup-routes-process",
            "trigger occurrence records a stable source key, reserves a matching delivery, and starts process wake routing without live external input",
        ),
        "runtime.process_wake_claim" => (
            "duplicate-process-wake-idempotency",
            "duplicate process wake deliveries share a dedupe key, claim queued work once, and keep replay/idempotency evidence backed by generated dynamic replay",
        ),
        "runtime.lease_release_rejects_commit" | "runtime.dead_lease_reclaim_rejects_stale" => (
            "worker-stale-completion-fenced",
            "stale worker completion carries an older fence and is rejected while the live incarnation remains active",
        ),
        "standard.empty_provider_response_error" | "standard.provider_error_without_checkpoint" => {
            (
                "provider-protocol-terminalization",
                "scripted provider mutation matrices classify retryable 429 and dropped-terminal parser failures through every migrated provider parser",
            )
        }
        "standard.streamed_text_finalizes_once"
        | "rlm.exec_tool_control_fail_terminal"
        | "rlm.exec_tool_control_frame_switch_terminal"
        | "rlm.exec_result_no_tool_call_replay" => (
            "rlm-standard-protocol-terminal-boundaries",
            "standard provider-error terminalization and RLM exec terminal boundaries stay represented by generated transitions with dynamic backend evidence",
        ),
        "standard.max_turns_after_tool_result" => (
            "backend-retry-terminalization",
            "retryable backend conflicts advance attempts and terminate on a non-retryable production StoreError class",
        ),
        _ => return None,
    };
    Some(ScenarioBackendRegressionReference {
        fixture_id,
        status: "generated_cross_backend_valid_trace",
        regression_contract,
    })
}

fn queued_active_turn_fact(
    contract: &ScenarioContractSpec,
    selected_events: &[TraceEventLine],
) -> Result<ScenarioTransitionFact, FixedScriptRunnerError> {
    let events = selected_events
        .iter()
        .filter(|line| {
            line.event.kind == BoundaryKind::QueuedIngress
                && line
                    .event
                    .observed
                    .get("ingress_mode")
                    .and_then(Value::as_str)
                    == Some("active_turn")
                && line
                    .event
                    .observed
                    .get("input_state")
                    .and_then(Value::as_str)
                    .is_some_and(|state| state.starts_with("pending"))
                && line
                    .event
                    .observed
                    .get("source_key")
                    .and_then(Value::as_str)
                    .is_some()
        })
        .collect::<Vec<_>>();
    let observed = json!({
        "queued_inputs": events.iter().map(|line| json!({
            "boundary_id": line.event.boundary_id,
            "source_key": line.event.observed.get("source_key").cloned().unwrap_or(Value::Null),
            "input_id": line.event.observed.get("input_id").cloned().unwrap_or(Value::Null),
            "input_state": line.event.observed.get("input_state").cloned().unwrap_or(Value::Null),
            "ingress_mode": line.event.observed.get("ingress_mode").cloned().unwrap_or(Value::Null),
        })).collect::<Vec<_>>(),
    });
    require_transition_fact(
        contract,
        "active_turn_input_queued_hidden",
        "active-turn queued input has stable source key and remains pending/hidden until terminalized",
        events,
        observed,
    )
}

fn queued_turn_followup_provider_fact(
    contract: &ScenarioContractSpec,
    selected_events: &[TraceEventLine],
) -> Result<ScenarioTransitionFact, FixedScriptRunnerError> {
    let mut fact_events = Vec::new();
    for queued in selected_events.iter().filter(|line| {
        line.event.kind == BoundaryKind::QueuedIngress
            && line
                .event
                .observed
                .get("source_key")
                .and_then(Value::as_str)
                .is_some()
    }) {
        if let Some(provider) = selected_events
            .iter()
            .filter(|line| {
                line.trace_alias == queued.trace_alias
                    && line.event.kind == BoundaryKind::Provider
                    && line.event.actor_alias == queued.event.actor_alias
                    && line.event.sequence > queued.event.sequence
                    && line.event.observed.get("success").and_then(Value::as_bool) == Some(true)
            })
            .min_by_key(|line| line.event.sequence)
        {
            fact_events.push(queued);
            fact_events.push(provider);
            let observed = json!({
                "queued_boundary": queued.event.boundary_id,
                "provider_boundary": provider.event.boundary_id,
                "trace_alias": queued.trace_alias,
                "actor": queued.event.actor_alias,
                "source_key": queued.event.observed.get("source_key").cloned().unwrap_or(Value::Null),
                "provider_exchange_count": provider.event.observed.get("provider_exchange_count").cloned().unwrap_or(Value::Null),
            });
            return require_transition_fact(
                contract,
                "queued_turn_input_followed_by_provider_completion",
                "queued turn input evidence is followed by a same-trace same-actor provider completion",
                fact_events,
                observed,
            );
        }
    }
    require_transition_fact(
        contract,
        "queued_turn_input_followed_by_provider_completion",
        "queued turn input evidence is followed by a same-trace same-actor provider completion",
        fact_events,
        json!({ "queued_turn_completion": false }),
    )
}

fn cancellation_terminalization_fact(
    contract: &ScenarioContractSpec,
    selected_events: &[TraceEventLine],
) -> Result<ScenarioTransitionFact, FixedScriptRunnerError> {
    let events = selected_events
        .iter()
        .filter(|line| {
            line.event.kind == BoundaryKind::Cancellation
                && line
                    .event
                    .observed
                    .get("cancelled")
                    .and_then(Value::as_bool)
                    == Some(true)
                && line
                    .event
                    .observed
                    .get("target")
                    .and_then(Value::as_str)
                    .is_some()
        })
        .collect::<Vec<_>>();
    let observed = json!({
        "terminalizations": events.iter().map(|line| json!({
            "boundary_id": line.event.boundary_id,
            "target": line.event.observed.get("target").cloned().unwrap_or(Value::Null),
            "cancel_outcome": line.event.observed.get("cancel_outcome").cloned().unwrap_or(Value::Null),
        })).collect::<Vec<_>>(),
    });
    require_transition_fact(
        contract,
        "cancellation_terminalized_pending_input",
        "cancellation targets a generated queued input and returns a terminal cancelled outcome",
        events,
        observed,
    )
}

fn trigger_wakeup_fact(
    contract: &ScenarioContractSpec,
    selected_events: &[TraceEventLine],
) -> Result<ScenarioTransitionFact, FixedScriptRunnerError> {
    let events = selected_events
        .iter()
        .filter(|line| {
            line.event.kind == BoundaryKind::Trigger
                && line
                    .event
                    .observed
                    .get("trigger_delivered")
                    .and_then(Value::as_bool)
                    == Some(true)
                && line
                    .event
                    .observed
                    .get("started_process")
                    .and_then(Value::as_bool)
                    == Some(true)
                && line
                    .event
                    .observed
                    .get("reservation_count")
                    .and_then(Value::as_u64)
                    .is_some_and(|count| count > 0)
        })
        .collect::<Vec<_>>();
    let observed = json!({
        "trigger_deliveries": events.iter().map(|line| json!({
            "boundary_id": line.event.boundary_id,
            "source_key": line.event.observed.get("source_key").cloned().unwrap_or(Value::Null),
            "occurrence_id": line.event.observed.get("occurrence_id").cloned().unwrap_or(Value::Null),
            "reservation_count": line.event.observed.get("reservation_count").cloned().unwrap_or(Value::Null),
            "started_process": true,
        })).collect::<Vec<_>>(),
    });
    require_transition_fact(
        contract,
        "trigger_routes_process_wakeup",
        "trigger occurrence records a stable source key, reserves matching delivery, and starts process routing",
        events,
        observed,
    )
}

fn command_queue_drain_fact(
    contract: &ScenarioContractSpec,
    selected_events: &[TraceEventLine],
) -> Result<ScenarioTransitionFact, FixedScriptRunnerError> {
    let queued_events = selected_events
        .iter()
        .filter(|line| {
            line.event.kind == BoundaryKind::QueuedIngress
                && line
                    .event
                    .observed
                    .get("source_key")
                    .and_then(Value::as_str)
                    .is_some_and(|source_key| !source_key.is_empty())
        })
        .collect::<Vec<_>>();
    let lease_events = selected_events
        .iter()
        .filter(|line| {
            line.event.kind == BoundaryKind::LeaseTime
                && line
                    .event
                    .observed
                    .pointer("/runtime_lease_probe/real_lease_store")
                    .and_then(Value::as_bool)
                    == Some(true)
                && line
                    .event
                    .observed
                    .pointer("/runtime_lease_probe/session_execution_lease_fencing_token")
                    .and_then(Value::as_u64)
                    .is_some()
        })
        .collect::<Vec<_>>();
    let mut events = Vec::new();
    events.extend(queued_events.iter().copied());
    events.extend(lease_events.iter().copied());
    let observed = json!({
        "queued_inputs": queued_events.iter().map(|line| json!({
            "boundary_id": line.event.boundary_id,
            "source_key": line.event.observed.get("source_key").cloned().unwrap_or(Value::Null),
            "input_state": line.event.observed.get("input_state").cloned().unwrap_or(Value::Null),
            "ingress_mode": line.event.observed.get("ingress_mode").cloned().unwrap_or(Value::Null),
        })).collect::<Vec<_>>(),
        "lease_fences": lease_events.iter().map(|line| json!({
            "boundary_id": line.event.boundary_id,
            "session": line.event.actor_alias,
            "fencing_token": line.event.observed.pointer("/runtime_lease_probe/session_execution_lease_fencing_token").cloned().unwrap_or(Value::Null),
            "real_lease_store": true,
        })).collect::<Vec<_>>(),
    });
    require_transition_fact(
        contract,
        "command_queue_drains_with_real_lease_fence",
        "command-only queued work carries scheduler-owned source keys and drains under real session-execution-lease fencing tokens",
        events,
        observed,
    )
}

fn process_wake_duplicate_fact(
    contract: &ScenarioContractSpec,
    selected_events: &[TraceEventLine],
) -> Result<ScenarioTransitionFact, FixedScriptRunnerError> {
    let mut by_dedupe_key: BTreeMap<String, Vec<&TraceEventLine>> = BTreeMap::new();
    for line in selected_events
        .iter()
        .filter(|line| line.event.kind == BoundaryKind::ProcessWake)
    {
        if let Some(dedupe_key) = line
            .event
            .observed
            .get("dedupe_key")
            .and_then(Value::as_str)
        {
            by_dedupe_key
                .entry(dedupe_key.to_string())
                .or_default()
                .push(line);
        }
    }
    let events = by_dedupe_key
        .values()
        .find(|events| {
            let strict_claim_dedupe = events.len() >= 2
                && events.iter().any(|line| {
                    line.event
                        .observed
                        .get("claimed_once")
                        .and_then(Value::as_bool)
                        == Some(true)
                });
            let in_flight_rejection = events.len() >= 2
                && events.iter().any(|line| {
                    line.event
                        .observed
                        .get("lease_busy")
                        .and_then(Value::as_bool)
                        == Some(true)
                        && line
                            .event
                            .observed
                            .pointer("/runtime_queued_work/enqueued")
                            .and_then(Value::as_bool)
                            == Some(false)
                });
            strict_claim_dedupe || in_flight_rejection
        })
        .cloned()
        .unwrap_or_default();
    let observed = json!({
        "dedupe_keys": by_dedupe_key.iter().map(|(dedupe_key, events)| json!({
            "dedupe_key": dedupe_key,
            "delivery_count": events.len(),
            "claimed_once": events.iter().any(|line| line.event.observed.get("claimed_once").and_then(Value::as_bool) == Some(true)),
            "lease_busy_rejected": events.iter().any(|line| {
                line.event.observed.get("lease_busy").and_then(Value::as_bool) == Some(true)
                    && line.event.observed.pointer("/runtime_queued_work/enqueued").and_then(Value::as_bool) == Some(false)
            }),
            "boundary_ids": boundary_ids(events),
        })).collect::<Vec<_>>(),
    });
    require_transition_fact(
        contract,
        "duplicate_process_wake_idempotent",
        "duplicate process wake deliveries share a dedupe key and are terminalized by queued-work claim dedupe or in-flight lease rejection",
        events,
        observed,
    )
}

fn observer_reconnect_transition_fact(
    contract: &ScenarioContractSpec,
    selected_events: &[TraceEventLine],
) -> Result<ScenarioTransitionFact, FixedScriptRunnerError> {
    let events = selected_events
        .iter()
        .filter(|line| {
            line.event.kind == BoundaryKind::Observer
                && line
                    .event
                    .observed
                    .get("reconnected")
                    .and_then(Value::as_bool)
                    == Some(true)
                && line
                    .event
                    .observed
                    .get("turn_index")
                    .and_then(Value::as_u64)
                    .is_some()
                && line
                    .event
                    .observed
                    .pointer("/observer_invariants/session_id")
                    .and_then(Value::as_bool)
                    == Some(true)
                && line
                    .event
                    .observed
                    .pointer("/observer_invariants/turn_index_converged")
                    .and_then(Value::as_bool)
                    == Some(true)
                && line
                    .event
                    .observed
                    .pointer("/observer_invariants/transcript_message_count_converged")
                    .and_then(Value::as_bool)
                    == Some(true)
        })
        .collect::<Vec<_>>();
    let observed = json!({
        "observer_reconnects": events.iter().map(|line| json!({
            "boundary_id": line.event.boundary_id,
            "session": line.event.actor_alias,
            "turn_index": line.event.observed.get("turn_index").cloned().unwrap_or(Value::Null),
            "graph_node_count": line.event.observed.get("graph_node_count").cloned().unwrap_or(Value::Null),
            "transcript_message_count": line.event.observed.get("transcript_message_count").cloned().unwrap_or(Value::Null),
            "observer_invariants": line.event.observed.get("observer_invariants").cloned().unwrap_or(Value::Null),
        })).collect::<Vec<_>>(),
    });
    require_transition_fact(
        contract,
        "observer_reconnect_replays_original_input_state",
        "observer reconnect boundary reads a concrete session observation with converged session id, turn index, graph, and transcript state",
        events,
        observed,
    )
}

fn worker_stale_completion_rejection_fact(
    contract: &ScenarioContractSpec,
    selected_events: &[TraceEventLine],
) -> Result<ScenarioTransitionFact, FixedScriptRunnerError> {
    let events = selected_events
        .iter()
        .filter(|line| {
            line.event.kind == BoundaryKind::Worker
                && line
                    .event
                    .observed
                    .get("stale_completion_rejected")
                    .and_then(Value::as_bool)
                    == Some(true)
                && line.event.observed.get("runtime_active_lease").is_some()
                && line
                    .event
                    .observed
                    .get("runtime_stale_completion")
                    .is_some()
                && line
                    .event
                    .observed
                    .pointer("/runtime_active_lease/fencing_token")
                    .and_then(Value::as_u64)
                    > line
                        .event
                        .observed
                        .pointer("/runtime_stale_completion/fencing_token")
                        .and_then(Value::as_u64)
        })
        .collect::<Vec<_>>();
    let observed = json!({
        "stale_completions": events.iter().map(|line| json!({
            "boundary_id": line.event.boundary_id,
            "active_fencing_token": line.event.observed.pointer("/runtime_active_lease/fencing_token").cloned().unwrap_or(Value::Null),
            "stale_fencing_token": line.event.observed.pointer("/runtime_stale_completion/fencing_token").cloned().unwrap_or(Value::Null),
            "stale_completion_rejected": true,
        })).collect::<Vec<_>>(),
    });
    require_transition_fact(
        contract,
        "lease_release_rejects_stale_completion",
        "stale worker completion carries an older fence and is rejected while the live lease remains active",
        events,
        observed,
    )
}

fn worker_dead_lease_reclaim_fact(
    contract: &ScenarioContractSpec,
    selected_events: &[TraceEventLine],
) -> Result<ScenarioTransitionFact, FixedScriptRunnerError> {
    let events = selected_events
        .iter()
        .filter(|line| {
            line.event.kind == BoundaryKind::Worker
                && line
                    .event
                    .observed
                    .get("lease_owner_changed")
                    .and_then(Value::as_bool)
                    == Some(true)
                && line
                    .event
                    .observed
                    .pointer("/runtime_worker_store/session_execution_lease_reclaimed")
                    .and_then(Value::as_bool)
                    == Some(true)
                && line
                    .event
                    .observed
                    .pointer("/runtime_worker_store/worker_owned_work/second_owner_resumed_work")
                    .and_then(Value::as_bool)
                    == Some(true)
                && line
                    .event
                    .observed
                    .pointer("/runtime_worker_store/worker_owned_work/second_owner_outranks_first")
                    .and_then(Value::as_bool)
                    == Some(true)
        })
        .collect::<Vec<_>>();
    let observed = json!({
        "dead_lease_reclaims": events.iter().map(|line| json!({
            "boundary_id": line.event.boundary_id,
            "initial_owner": line.event.observed.get("initial_owner").cloned().unwrap_or(Value::Null),
            "active_owner": line.event.observed.get("active_owner").cloned().unwrap_or(Value::Null),
            "source_key": line.event.observed.pointer("/runtime_worker_store/worker_owned_work/source_key").cloned().unwrap_or(Value::Null),
            "second_owner_resumed_work": true,
            "second_owner_outranks_first": true,
        })).collect::<Vec<_>>(),
    });
    require_transition_fact(
        contract,
        "dead_lease_reclaim_resumes_worker_owned_work",
        "successor worker owner reclaims the dead lease, outranks the first fence, and resumes the owned work",
        events,
        observed,
    )
}

fn provider_terminalization_fact(
    contract: &ScenarioContractSpec,
    selected_events: &[TraceEventLine],
) -> Result<ScenarioTransitionFact, FixedScriptRunnerError> {
    let events = selected_events
        .iter()
        .filter(|line| {
            line.event.kind == BoundaryKind::ProviderMutation
                && line
                    .event
                    .observed
                    .pointer("/provider_parser_matrix/matrix/real_provider_parser_execution")
                    .and_then(Value::as_bool)
                    == Some(true)
        })
        .collect::<Vec<_>>();
    let observed = json!({
        "provider_mutations": events.iter().map(|line| json!({
            "boundary_id": line.event.boundary_id,
            "mutation": line.event.observed.get("mutation").cloned().unwrap_or(Value::Null),
            "parser_matrix_digest": line.event.observed.pointer("/provider_parser_matrix/digest").cloned().unwrap_or(Value::Null),
            "real_provider_parser_execution": true,
        })).collect::<Vec<_>>(),
    });
    require_transition_fact(
        contract,
        "provider_failure_terminalized_by_parser_matrix",
        "scripted provider failure reaches real migrated provider parsers and terminal classification",
        events,
        observed,
    )
}

fn provider_success_terminal_fact(
    contract: &ScenarioContractSpec,
    selected_events: &[TraceEventLine],
) -> Result<ScenarioTransitionFact, FixedScriptRunnerError> {
    let events = selected_events
        .iter()
        .filter(|line| {
            line.event.kind == BoundaryKind::Provider
                && line.event.observed.get("success").and_then(Value::as_bool) == Some(true)
                && line
                    .event
                    .observed
                    .pointer("/runtime_contract/status")
                    .and_then(Value::as_str)
                    == Some("passed")
        })
        .collect::<Vec<_>>();
    let observed = json!({
        "provider_turns": events.iter().map(|line| json!({
            "boundary_id": line.event.boundary_id,
            "provider_kind": line.event.observed.get("provider_kind").cloned().unwrap_or(Value::Null),
            "turn_index": line.event.observed.get("turn_index").cloned().unwrap_or(Value::Null),
            "runtime_contract": line.event.observed.get("runtime_contract").cloned().unwrap_or(Value::Null),
        })).collect::<Vec<_>>(),
    });
    require_transition_fact(
        contract,
        "provider_success_terminal_boundary",
        "provider turn terminates through the scripted runtime provider path and passes runtime contract checks",
        events,
        observed,
    )
}

fn tool_result_fact(
    contract: &ScenarioContractSpec,
    selected_events: &[TraceEventLine],
) -> Result<ScenarioTransitionFact, FixedScriptRunnerError> {
    let events = selected_events
        .iter()
        .filter(|line| {
            line.event.kind == BoundaryKind::Tool
                && line.event.observed.get("runtime_tool_output").is_some()
        })
        .collect::<Vec<_>>();
    let observed = json!({
        "tool_results": events.iter().map(|line| json!({
            "boundary_id": line.event.boundary_id,
            "tool_name": line.event.observed.get("tool_name").cloned().unwrap_or(Value::Null),
            "tool_call_id": line.event.observed.get("tool_call_id").cloned().unwrap_or(Value::Null),
            "runtime_effect": line.event.observed.get("runtime_effect").cloned().unwrap_or(Value::Null),
        })).collect::<Vec<_>>(),
    });
    require_transition_fact(
        contract,
        "tool_result_checkpoint_boundary",
        "tool result crosses runtime effect-controller output and remains available for protocol re-entry",
        events,
        observed,
    )
}

fn exec_terminal_fact(
    contract: &ScenarioContractSpec,
    selected_events: &[TraceEventLine],
) -> Result<ScenarioTransitionFact, FixedScriptRunnerError> {
    let events = selected_events
        .iter()
        .filter(|line| {
            line.event.kind == BoundaryKind::ExecCode
                && line.event.observed.get("runtime_effect_outcome").is_some()
        })
        .collect::<Vec<_>>();
    let observed = json!({
        "exec_boundaries": events.iter().map(|line| json!({
            "boundary_id": line.event.boundary_id,
            "exit_code": line.event.observed.get("exit_code").cloned().unwrap_or(Value::Null),
            "runtime_effect": line.event.observed.get("runtime_effect").cloned().unwrap_or(Value::Null),
            "runtime_effect_outcome_type": line.event.observed.pointer("/runtime_effect_outcome/type").cloned().unwrap_or(Value::Null),
        })).collect::<Vec<_>>(),
    });
    require_transition_fact(
        contract,
        "rlm_exec_terminal_boundary",
        "exec-code result crosses runtime effect-controller outcome and is available to RLM terminal state-machine contracts",
        events,
        observed,
    )
}

fn durable_effect_replay_fact(
    contract: &ScenarioContractSpec,
    selected_events: &[TraceEventLine],
) -> Result<ScenarioTransitionFact, FixedScriptRunnerError> {
    let mut by_key: BTreeMap<String, Vec<&TraceEventLine>> = BTreeMap::new();
    for line in selected_events
        .iter()
        .filter(|line| line.event.kind == BoundaryKind::DurableEffect)
    {
        if let Some(key) = line
            .event
            .observed
            .get("durable_key")
            .and_then(Value::as_str)
        {
            by_key.entry(key.to_string()).or_default().push(line);
        }
    }
    let events = by_key
        .values()
        .find(|events| {
            events.iter().any(|line| {
                line.event.observed.get("replayed").and_then(Value::as_bool) == Some(false)
                    && line
                        .event
                        .observed
                        .pointer("/runtime_effect/local_executor_called")
                        .and_then(Value::as_bool)
                        == Some(true)
            }) && events.iter().any(|line| {
                line.event.observed.get("replayed").and_then(Value::as_bool) == Some(true)
                    && line
                        .event
                        .observed
                        .pointer("/runtime_effect/local_executor_called")
                        .and_then(Value::as_bool)
                        == Some(false)
            })
        })
        .cloned()
        .unwrap_or_default();
    let observed = json!({
        "durable_keys": by_key.iter().map(|(key, events)| json!({
            "durable_key": key,
            "boundary_ids": boundary_ids(events),
            "first_execution": events.iter().any(|line| line.event.observed.get("replayed").and_then(Value::as_bool) == Some(false)),
            "replay": events.iter().any(|line| line.event.observed.get("replayed").and_then(Value::as_bool) == Some(true)),
        })).collect::<Vec<_>>(),
    });
    require_transition_fact(
        contract,
        "durable_effect_replay_idempotent",
        "durable effect executes locally once and replay returns stored history without re-execution",
        events,
        observed,
    )
}

fn generic_transition_fact(
    contract: &ScenarioContractSpec,
    selected_events: &[TraceEventLine],
) -> Result<ScenarioTransitionFact, FixedScriptRunnerError> {
    let events = selected_events.iter().collect::<Vec<_>>();
    let observed = json!({
        "selected_event_count": events.len(),
        "boundary_kinds": events
            .iter()
            .map(|line| format!("{:?}", line.event.kind))
            .collect::<BTreeSet<_>>(),
    });
    require_transition_fact(
        contract,
        "generated_transition_evidence_present",
        "scenario contract selected generated trace events for its required state transition",
        events,
        observed,
    )
}

fn require_transition_fact(
    contract: &ScenarioContractSpec,
    fact: &'static str,
    assertion: &'static str,
    events: Vec<&TraceEventLine>,
    observed: Value,
) -> Result<ScenarioTransitionFact, FixedScriptRunnerError> {
    if events.is_empty() {
        return Err(FixedScriptRunnerError::Assertion(format!(
            "scenario contract `{}` could not prove transition fact `{fact}`",
            contract.test_name
        )));
    }
    Ok(ScenarioTransitionFact {
        fact: fact.to_string(),
        status: "passed",
        assertion,
        boundary_ids: boundary_ids(&events),
        observed,
    })
}

fn boundary_ids(events: &[&TraceEventLine]) -> Vec<String> {
    events
        .iter()
        .map(|line| line.event.boundary_id.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn scenario_transition_kind(contract: &ScenarioContractSpec) -> &'static str {
    match contract.semantic_oracle {
        "runtime.process_wake_claim" => "runtime.process-wake-claim-dedupe-transition",
        "runtime.lease_release_rejects_commit" => {
            "runtime.lease-release-stale-commit-rejection-transition"
        }
        "runtime.dead_lease_reclaim_rejects_stale" => {
            "runtime.dead-lease-reclaim-stale-completion-transition"
        }
        "runtime.checkpoint_redrive_cancel" => "runtime.checkpoint-redrive-cancellation-transition",
        "runtime.queued_work_keeps_pending_input" => {
            "runtime.live-turn-keeps-queued-input-hidden-transition"
        }
        "runtime.queued_turn_input_completion" => {
            "runtime.queued-turn-input-claim-completion-transition"
        }
        "runtime.command_only_queue_drain" => "runtime.command-only-queue-drain-transition",
        "runtime.command_before_turn_work" => "runtime.command-before-turn-work-transition",
        "runtime.observation_replay_preserves_input" => {
            "runtime.observer-reconnect-preserves-input-transition"
        }
        "standard.empty_provider_response_error" => {
            "standard.empty-provider-response-terminal-error-transition"
        }
        "standard.provider_error_without_checkpoint" => {
            "standard.provider-error-no-checkpoint-transition"
        }
        "standard.native_tool_loop_reenters_model" => {
            "standard.native-tool-result-model-reentry-transition"
        }
        "standard.tool_failure_feedback_reenters_model" => {
            "standard.tool-failure-feedback-model-reentry-transition"
        }
        "standard.max_turns_after_tool_result" => "standard.max-turns-after-tool-result-transition",
        "standard.parallel_tool_results_checkpoint_once" => {
            "standard.parallel-tool-results-single-checkpoint-transition"
        }
        "standard.streamed_text_finalizes_once" => {
            "standard.streamed-text-duplicate-free-finalization-transition"
        }
        "standard.initial_request_projection" => "standard.initial-request-projection-transition",
        "rlm.lashlang_cell_exec_continues" => "rlm.lashlang-cell-exec-continue-transition",
        "rlm.exec_error_max_turn_stop" => "rlm.exec-error-max-turn-stop-transition",
        "rlm.exec_tool_control_frame_switch_terminal" => {
            "rlm.exec-tool-control-frame-switch-terminal-transition"
        }
        "rlm.exec_tool_control_fail_terminal" => "rlm.exec-tool-control-fail-terminal-transition",
        "rlm.exec_result_no_tool_call_replay" => {
            "rlm.exec-result-without-tool-call-replay-transition"
        }
        "rlm.typed_schema_mismatch_repair_loop" => "rlm.typed-schema-mismatch-repair-transition",
        "rlm.typed_schema_any_of_mismatch" => "rlm.typed-anyof-mismatch-repair-transition",
        semantic if semantic.starts_with("rlm.") => "rlm.provider-repair-or-finish-transition",
        "agent.shell_results_are_data" => "agent.shell-result-data-transition",
        "agent.shell_output_print_projection_survives" => {
            "agent.shell-output-print-projection-transition"
        }
        "agent.foreground_tool_call_round_trip" => {
            "agent.foreground-tool-call-round-trip-transition"
        }
        "agent.durable_input_suspension_resolution" => {
            "agent.durable-input-suspend-resolve-transition"
        }
        "agent.tuple_values_finish_as_json_arrays" => "agent.tuple-finish-json-array-transition",
        semantic if semantic.starts_with("agent.") => "agent.process-graph-transition",
        _ => "scenario.generic-generated-transition",
    }
}

fn scenario_evidence_boundary_kind(evidence: &str) -> &'static str {
    match evidence {
        "queued_ingress" => "queued_ingress",
        "cancellation" => "cancellation",
        "process_wake" => "process_wake",
        "worker_stale_completion" => "worker",
        "lease_time" => "lease_time",
        "provider_turn" => "provider",
        "provider_event" => "provider_event",
        "tool_result" => "tool",
        "max_turn_stop" => "trigger",
        "final_value" => "trigger",
        "observer_convergence" | "observer_reconnect" => "observer",
        "runtime_session_graph" => "ingress/provider",
        "exec_code" => "exec_code",
        "trigger" => "trigger",
        "durable_effect" => "durable_effect",
        "multi_session" => "multi_session",
        "provider_mutation" => "provider_mutation",
        "backend_failure" => "backend_failure",
        _ => "unknown",
    }
}

fn scenario_evidence_assertion(evidence: &str) -> &'static str {
    match evidence {
        "queued_ingress" => "queued input has stable source key and remains explicit work",
        "cancellation" => "cancellation targets an existing generated boundary",
        "process_wake" => "process wake crosses runtime queued-work claim/dedupe DTOs",
        "worker_stale_completion" => {
            "worker failover advances lease fencing and rejects stale completion"
        }
        "lease_time" => "lease ticks remain monotonic under scheduler delivery",
        "provider_turn" => {
            "provider completion is delivered through scripted provider runtime path"
        }
        "provider_event" => "provider event release is scheduler-owned generated evidence",
        "tool_result" => "tool result crosses runtime effect-controller output DTO",
        "max_turn_stop" => {
            "semantic proof records explicit max-turn stopped outcome after a tool result"
        }
        "final_value" => {
            "semantic proof records typed final value outcome and terminal event facts"
        }
        "observer_convergence" => "observer sees the generated final provider turn",
        "runtime_session_graph" => "session graph advances with generated ingress/provider turns",
        "exec_code" => "exec-code result crosses runtime effect-controller outcome DTO",
        "trigger" => "trigger delivery carries stable trigger identity",
        "durable_effect" => "durable effect records first completion and replay evidence",
        "multi_session" => "trace slice contains at least two generated sessions",
        "observer_reconnect" => "observer reconnect converges to the same session state",
        "provider_mutation" => "mutated provider script executes through real provider parsers",
        "backend_failure" => "backend failure is classified as a production StoreError shape",
        _ => "scenario evidence is selected from generated trace events",
    }
}

fn scenario_negative_fixture_for_contract(
    contract: &ScenarioContractSpec,
    selected_evidence: &[ScenarioEvidenceSelection],
) -> ScenarioNegativeFixture {
    match contract.semantic_oracle {
        "runtime.checkpoint_redrive_cancel" => {
            return scenario_negative_fixture("operational_coverage_missing_cancellation");
        }
        "runtime.queued_work_keeps_pending_input" => {
            return scenario_negative_fixture("queued_input_operational_missing");
        }
        "runtime.command_before_turn_work" => {
            return scenario_negative_fixture("trigger_wakeup_operational_missing");
        }
        "runtime.process_wake_claim" => {
            return scenario_negative_fixture("process_wake_operational_missing");
        }
        "standard.max_turns_after_tool_result" => {
            return scenario_negative_fixture("standard_max_turn_stop_missing");
        }
        "standard.provider_error_without_checkpoint" => {
            return scenario_negative_fixture("standard_provider_error_missing_parser_matrix");
        }
        "rlm.typed_finish_emits_outcome_and_done" => {
            return scenario_negative_fixture("rlm_typed_finish_terminal_event_missing");
        }
        "rlm.empty_options_natural_default" => {
            return scenario_negative_fixture("rlm_empty_options_default_mode_broken");
        }
        "rlm.lashlang_cell_exec_continues" => {
            return scenario_negative_fixture("rlm_lashlang_cell_missing_exec_outcome");
        }
        "agent.tuple_values_finish_as_json_arrays" => {
            return scenario_negative_fixture("agent_tuple_json_array_shape_broken");
        }
        _ => {}
    }
    let has = |evidence: &str| {
        selected_evidence
            .iter()
            .any(|selection| selection.evidence == evidence)
            || semantic_scenario_evidence(contract.semantic_oracle).contains(&evidence)
    };
    if has("backend_failure") {
        return scenario_negative_fixture("backend_retry_runtime_completion_missing");
    }
    if has("provider_mutation") {
        if contract.suite == "standard"
            && contract
                .semantic_oracle
                .contains("provider_error_without_checkpoint")
        {
            return scenario_negative_fixture("standard_provider_error_missing_parser_matrix");
        }
        return scenario_negative_fixture("provider_mutation_runtime_completion_missing");
    }
    if has("worker_stale_completion") {
        return scenario_negative_fixture("worker_failover_stale_rejection_missing");
    }
    if has("cancellation") {
        return scenario_negative_fixture("operational_coverage_missing_cancellation");
    }
    if has("queued_ingress") {
        return scenario_negative_fixture("queued_input_operational_missing");
    }
    if has("trigger") {
        return scenario_negative_fixture("trigger_wakeup_operational_missing");
    }
    if has("process_wake") {
        if contract.suite == "agent" {
            return scenario_negative_fixture("agent_parallel_join_missing_wake_session");
        }
        return scenario_negative_fixture("process_wake_operational_missing");
    }
    if has("exec_code") || contract.suite == "rlm" {
        return scenario_negative_fixture("rlm_lashlang_cell_missing_exec_outcome");
    }
    if contract.suite == "standard" {
        return scenario_negative_fixture("standard_provider_error_missing_parser_matrix");
    }
    if contract.suite == "agent" {
        return scenario_negative_fixture("agent_parallel_join_missing_wake_session");
    }
    scenario_negative_fixture("scheduler_owned_provider_completion_missing_evidence")
}

fn scenario_negative_fixture(fixture_id: &str) -> ScenarioNegativeFixture {
    match fixture_id {
        "operational_coverage_missing_cancellation" => ScenarioNegativeFixture {
            fixture_id: "operational-coverage-missing-cancellation",
            fixture_path: "crates/lash-sim/failure-fixtures/operational-coverage-missing-cancellation.json",
            expected_oracle_id: "sim.oracle.scheduler-owned-runtime-completions.v1",
            expected_reason_contains: "Cancellation",
        },
        "queued_input_operational_missing" => ScenarioNegativeFixture {
            fixture_id: "queued-input-operational-missing",
            fixture_path: "crates/lash-sim/failure-fixtures/queued-input-operational-missing.json",
            expected_oracle_id: "sim.oracle.state-machine-semantic-invariants.v1",
            expected_reason_contains: "queued active-turn input",
        },
        "trigger_wakeup_operational_missing" => ScenarioNegativeFixture {
            fixture_id: "trigger-wakeup-operational-missing",
            fixture_path: "crates/lash-sim/failure-fixtures/trigger-wakeup-operational-missing.json",
            expected_oracle_id: "sim.oracle.state-machine-semantic-invariants.v1",
            expected_reason_contains: "trigger wakeup routes",
        },
        "process_wake_operational_missing" => ScenarioNegativeFixture {
            fixture_id: "process-wake-operational-missing",
            fixture_path: "crates/lash-sim/failure-fixtures/process-wake-operational-missing.json",
            expected_oracle_id: "sim.oracle.state-machine-semantic-invariants.v1",
            expected_reason_contains: "duplicate delivery/replay semantics",
        },
        "standard_provider_error_missing_parser_matrix" => ScenarioNegativeFixture {
            fixture_id: "standard-provider-error-missing-parser-matrix",
            fixture_path: "crates/lash-sim/failure-fixtures/standard-provider-error-missing-parser-matrix.json",
            expected_oracle_id: "sim.oracle.scenario-mini.standard.provider-error-without-checkpoint.v1",
            expected_reason_contains: "no scheduler-owned provider failure before checkpoint",
        },
        "standard_max_turn_stop_missing" => ScenarioNegativeFixture {
            fixture_id: "standard-max-turn-stop-missing",
            fixture_path: "crates/lash-sim/failure-fixtures/standard-max-turn-stop-missing.json",
            expected_oracle_id: "sim.oracle.scenario.standard-contract.v1:standard_protocol_scenario_max_turns_terminates_after_tool_result",
            expected_reason_contains: "fixed-source replay validation",
        },
        "rlm_lashlang_cell_missing_exec_outcome" => ScenarioNegativeFixture {
            fixture_id: "rlm-lashlang-cell-missing-exec-outcome",
            fixture_path: "crates/lash-sim/failure-fixtures/rlm-lashlang-cell-missing-exec-outcome.json",
            expected_oracle_id: "sim.oracle.scenario-mini.rlm.lashlang-cell-exec-continues.v1",
            expected_reason_contains: "did not continue after exec",
        },
        "rlm_typed_finish_terminal_event_missing" => ScenarioNegativeFixture {
            fixture_id: "rlm-typed-finish-terminal-event-missing",
            fixture_path: "crates/lash-sim/failure-fixtures/rlm-typed-finish-terminal-event-missing.json",
            expected_oracle_id: "sim.oracle.scenario.rlm-contract.v1:rlm_protocol_scenario_typed_finish_emits_turn_outcome_and_done",
            expected_reason_contains: "fixed-source replay validation",
        },
        "rlm_empty_options_default_mode_broken" => ScenarioNegativeFixture {
            fixture_id: "rlm-empty-options-default-mode-broken",
            fixture_path: "crates/lash-sim/failure-fixtures/rlm-empty-options-default-mode-broken.json",
            expected_oracle_id: "sim.oracle.scenario.rlm-contract.v1:rlm_protocol_scenario_empty_turn_options_use_natural_default",
            expected_reason_contains: "fixed-source replay validation",
        },
        "agent_parallel_join_missing_wake_session" => ScenarioNegativeFixture {
            fixture_id: "agent-parallel-join-missing-wake-session",
            fixture_path: "crates/lash-sim/failure-fixtures/agent-parallel-join-missing-wake-session.json",
            expected_oracle_id: "sim.oracle.scenario-mini.agent.parallel-spawn-join-determinism.v1",
            expected_reason_contains: "did not record deterministic process/worker ordering",
        },
        "agent_tuple_json_array_shape_broken" => ScenarioNegativeFixture {
            fixture_id: "agent-tuple-json-array-shape-broken",
            fixture_path: "crates/lash-sim/failure-fixtures/agent-tuple-json-array-shape-broken.json",
            expected_oracle_id: "sim.oracle.scenario.agent-contract.v1:agent_scenario_tuple_values_finish_as_json_arrays",
            expected_reason_contains: "fixed-source replay validation",
        },
        "provider_mutation_runtime_completion_missing" => ScenarioNegativeFixture {
            fixture_id: "provider-mutation-runtime-completion-missing",
            fixture_path: "crates/lash-sim/failure-fixtures/provider-mutation-runtime-completion-missing.json",
            expected_oracle_id: "sim.oracle.scheduler-owned-runtime-completions.v1",
            expected_reason_contains: "for ProviderMutation",
        },
        "worker_failover_stale_rejection_missing" => ScenarioNegativeFixture {
            fixture_id: "worker-failover-stale-rejection-missing",
            fixture_path: "crates/lash-sim/failure-fixtures/worker-failover-stale-rejection-missing.json",
            expected_oracle_id: "sim.oracle.scheduler-owned-runtime-completions.v1",
            expected_reason_contains: "for Worker",
        },
        "backend_retry_runtime_completion_missing" => ScenarioNegativeFixture {
            fixture_id: "backend-retry-runtime-completion-missing",
            fixture_path: "crates/lash-sim/failure-fixtures/backend-retry-runtime-completion-missing.json",
            expected_oracle_id: "sim.oracle.scheduler-owned-runtime-completions.v1",
            expected_reason_contains: "for BackendFailure",
        },
        _ => ScenarioNegativeFixture {
            fixture_id: "scheduler-owned-provider-completion-missing-evidence",
            fixture_path: "crates/lash-sim/failure-fixtures/scheduler-owned-provider-completion-missing-evidence.json",
            expected_oracle_id: "sim.oracle.scheduler-owned-runtime-completions.v1",
            expected_reason_contains: "delivered without pending runtime boundary evidence",
        },
    }
}

fn select_scenario_contract_evidence(
    contract: &ScenarioContractSpec,
    event_lines: &[TraceEventLine],
) -> Result<Vec<ScenarioEvidenceSelection>, FixedScriptRunnerError> {
    let mut selected = Vec::new();
    let mut generated_fact_trace_alias = None;
    if matches!(contract.suite, "standard" | "rlm" | "agent") {
        let (trace_alias, facts) = select_scenario_contract_fact_trace(contract, event_lines)
            .map_err(|reason| {
                FixedScriptRunnerError::Assertion(format!(
                    "scenario contract `{}` could not select generated semantic facts: {reason}",
                    contract.test_name
                ))
            })?;
        generated_fact_trace_alias = Some(trace_alias.clone());
        for fact in facts {
            for boundary_id in fact.boundary_ids {
                let line = event_lines
                    .iter()
                    .find(|line| {
                        line.trace_alias == trace_alias && line.event.boundary_id == boundary_id
                    })
                    .ok_or_else(|| {
                        FixedScriptRunnerError::Assertion(format!(
                            "scenario contract `{}` generated fact `{}` selected missing boundary `{boundary_id}` in trace `{trace_alias}`",
                            contract.test_name, fact.fact
                        ))
                    })?;
                selected.push(ScenarioEvidenceSelection {
                    evidence: format!("semantic_fact:{}", fact.fact),
                    trace_alias: line.trace_alias.clone(),
                    boundary_id: line.event.boundary_id.clone(),
                    boundary_kind: line.event.kind,
                    sequence: line.event.sequence,
                });
            }
        }
    }
    let mut evidence_names = BTreeSet::new();
    for evidence in contract
        .required_sim_evidence
        .iter()
        .copied()
        .chain(semantic_scenario_evidence(contract.semantic_oracle))
        .filter(|evidence| evidence_names.insert(*evidence))
    {
        if let Some(trace_alias) = generated_fact_trace_alias.as_deref() {
            let scoped_event_lines = event_lines
                .iter()
                .filter(|line| line.trace_alias == trace_alias)
                .cloned()
                .collect::<Vec<_>>();
            let matches = select_event_lines_for_evidence(evidence, &scoped_event_lines);
            if matches.is_empty() {
                return Err(FixedScriptRunnerError::Assertion(format!(
                    "scenario contract `{}` could not select generated trace evidence `{}` from fact-owning trace `{trace_alias}`",
                    contract.test_name, evidence
                )));
            }
            selected.extend(matches.into_iter().map(|line| ScenarioEvidenceSelection {
                evidence: evidence.to_string(),
                trace_alias: line.trace_alias.clone(),
                boundary_id: line.event.boundary_id.clone(),
                boundary_kind: line.event.kind,
                sequence: line.event.sequence,
            }));
        } else {
            let matches = select_event_lines_for_evidence(evidence, event_lines);
            if matches.is_empty() {
                return Err(FixedScriptRunnerError::Assertion(format!(
                    "scenario contract `{}` could not select generated trace evidence `{}`",
                    contract.test_name, evidence
                )));
            }
            selected.extend(matches.into_iter().map(|line| ScenarioEvidenceSelection {
                evidence: evidence.to_string(),
                trace_alias: line.trace_alias.clone(),
                boundary_id: line.event.boundary_id.clone(),
                boundary_kind: line.event.kind,
                sequence: line.event.sequence,
            }));
        }
    }
    if contract.semantic_oracle == "runtime.queued_turn_input_completion" {
        let matches = select_queued_turn_followup_provider_evidence(event_lines);
        if matches.is_empty() {
            return Err(FixedScriptRunnerError::Assertion(format!(
                "scenario contract `{}` could not select same-actor provider completion after queued input",
                contract.test_name
            )));
        }
        selected.extend(matches.into_iter().map(|line| ScenarioEvidenceSelection {
            evidence: "queued_turn_followup_provider".to_string(),
            trace_alias: line.trace_alias.clone(),
            boundary_id: line.event.boundary_id.clone(),
            boundary_kind: line.event.kind,
            sequence: line.event.sequence,
        }));
    }
    Ok(selected)
}

fn select_scenario_contract_fact_trace(
    contract: &ScenarioContractSpec,
    event_lines: &[TraceEventLine],
) -> Result<(String, Vec<crate::oracles::ScenarioContractGeneratedFact>), String> {
    let mut by_trace: BTreeMap<&str, Vec<crate::scheduler::DeliveredBoundary>> = BTreeMap::new();
    for line in event_lines {
        by_trace
            .entry(line.trace_alias.as_str())
            .or_default()
            .push(line.event.clone());
    }
    let mut failures = Vec::new();
    for (trace_alias, events) in by_trace {
        match scenario_contract_generated_facts(contract, &events) {
            Ok(facts) => return Ok((trace_alias.to_string(), facts)),
            Err(reason) => failures.push(format!("{trace_alias}: {reason}")),
        }
    }
    Err(format!(
        "no single generated trace contained the contract-owned fact graph; {}",
        failures.join("; ")
    ))
}

fn semantic_scenario_evidence(semantic_oracle: &str) -> Vec<&'static str> {
    match semantic_oracle {
        "runtime.process_wake_claim" => vec!["process_wake"],
        "runtime.lease_release_rejects_commit" | "runtime.dead_lease_reclaim_rejects_stale" => {
            vec!["worker_stale_completion"]
        }
        "runtime.checkpoint_redrive_cancel" => vec!["queued_ingress", "cancellation"],
        "runtime.queued_work_keeps_pending_input" | "runtime.queued_turn_input_completion" => {
            vec!["queued_ingress", "provider_turn"]
        }
        "runtime.command_only_queue_drain" => vec!["queued_ingress", "lease_time"],
        "runtime.command_before_turn_work" => vec!["trigger", "queued_ingress", "lease_time"],
        "runtime.observation_replay_preserves_input" => vec!["observer_reconnect"],
        _ => Vec::new(),
    }
}

fn select_event_lines_for_evidence<'a>(
    evidence: &str,
    event_lines: &'a [TraceEventLine],
) -> Vec<&'a TraceEventLine> {
    match evidence {
        "multi_session" => {
            let mut actors = BTreeSet::new();
            event_lines
                .iter()
                .filter(|line| {
                    line.event.kind == BoundaryKind::Ingress
                        && actors.insert(line.event.actor_alias.clone())
                })
                .take(2)
                .collect()
        }
        "runtime_session_graph" => event_lines
            .iter()
            .filter(|line| {
                matches!(
                    line.event.kind,
                    BoundaryKind::Ingress | BoundaryKind::Provider
                )
            })
            .take(2)
            .collect(),
        "process_wake" => select_process_wake_evidence(event_lines),
        "durable_effect" => select_durable_effect_evidence(event_lines),
        _ => event_lines
            .iter()
            .filter(|line| event_satisfies_scenario_evidence(&line.event, evidence))
            .take(2)
            .collect(),
    }
}

fn select_queued_turn_followup_provider_evidence(
    event_lines: &[TraceEventLine],
) -> Vec<&TraceEventLine> {
    for queued in event_lines.iter().filter(|line| {
        event_satisfies_scenario_evidence(&line.event, "queued_ingress")
            && line
                .event
                .observed
                .get("ingress_mode")
                .and_then(Value::as_str)
                == Some("active_turn")
    }) {
        if let Some(provider) = event_lines
            .iter()
            .filter(|line| {
                line.trace_alias == queued.trace_alias
                    && line.event.kind == BoundaryKind::Provider
                    && line.event.actor_alias == queued.event.actor_alias
                    && line.event.sequence > queued.event.sequence
                    && line.event.observed.get("success").and_then(Value::as_bool) == Some(true)
            })
            .min_by_key(|line| line.event.sequence)
        {
            return vec![queued, provider];
        }
    }
    Vec::new()
}

fn select_process_wake_evidence(event_lines: &[TraceEventLine]) -> Vec<&TraceEventLine> {
    let mut by_dedupe_key = BTreeMap::<String, Vec<&TraceEventLine>>::new();
    for line in event_lines
        .iter()
        .filter(|line| event_satisfies_scenario_evidence(&line.event, "process_wake"))
    {
        let Some(dedupe_key) = line
            .event
            .observed
            .get("dedupe_key")
            .and_then(Value::as_str)
        else {
            continue;
        };
        by_dedupe_key
            .entry(dedupe_key.to_string())
            .or_default()
            .push(line);
    }
    if let Some((_dedupe_key, events)) = by_dedupe_key
        .iter()
        .find(|(_dedupe_key, events)| events.len() >= 2)
    {
        return events.iter().copied().take(2).collect();
    }
    event_lines
        .iter()
        .filter(|line| event_satisfies_scenario_evidence(&line.event, "process_wake"))
        .take(2)
        .collect()
}

fn select_durable_effect_evidence(event_lines: &[TraceEventLine]) -> Vec<&TraceEventLine> {
    let mut by_durable_key = BTreeMap::<String, Vec<&TraceEventLine>>::new();
    for line in event_lines
        .iter()
        .filter(|line| event_satisfies_scenario_evidence(&line.event, "durable_effect"))
    {
        let Some(durable_key) = line
            .event
            .observed
            .get("durable_key")
            .and_then(Value::as_str)
        else {
            continue;
        };
        by_durable_key
            .entry(durable_key.to_string())
            .or_default()
            .push(line);
    }
    if let Some((_durable_key, events)) = by_durable_key.iter().find(|(_durable_key, events)| {
        events
            .iter()
            .any(|line| line.event.observed.get("replayed").and_then(Value::as_bool) == Some(false))
            && events.iter().any(|line| {
                line.event.observed.get("replayed").and_then(Value::as_bool) == Some(true)
            })
    }) {
        return events.iter().copied().take(2).collect();
    }
    event_lines
        .iter()
        .filter(|line| event_satisfies_scenario_evidence(&line.event, "durable_effect"))
        .take(2)
        .collect()
}

fn event_satisfies_scenario_evidence(
    event: &crate::scheduler::DeliveredBoundary,
    evidence: &str,
) -> bool {
    match evidence {
        "queued_ingress" => {
            event.kind == BoundaryKind::QueuedIngress
                && event
                    .observed
                    .get("source_key")
                    .and_then(Value::as_str)
                    .is_some()
        }
        "cancellation" => {
            event.kind == BoundaryKind::Cancellation
                && event
                    .observed
                    .get("target")
                    .and_then(Value::as_str)
                    .is_some()
        }
        "process_wake" => {
            event.kind == BoundaryKind::ProcessWake
                && event.observed.get("runtime_process_wake").is_some()
        }
        "worker_stale_completion" => {
            event.kind == BoundaryKind::Worker
                && event
                    .observed
                    .get("stale_completion_rejected")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
        }
        "lease_time" => event.kind == BoundaryKind::LeaseTime,
        "provider_turn" => event.kind == BoundaryKind::Provider,
        "provider_event" => {
            event.kind == BoundaryKind::ProviderEvent
                && event
                    .observed
                    .get("provider_event_release")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
        }
        "tool_result" => {
            event.kind == BoundaryKind::Tool && event.observed.get("runtime_tool_output").is_some()
        }
        "max_turn_stop" => {
            event.kind == BoundaryKind::Trigger
                && event
                    .observed
                    .pointer("/contract_execution/contract")
                    .and_then(Value::as_str)
                    == Some("standard.max_turns_after_tool_result")
                && event
                    .observed
                    .pointer("/contract_execution/source/kind")
                    .and_then(Value::as_str)
                    == Some("fixed_dst_api_execution")
                && event
                    .observed
                    .pointer("/contract_execution/result/turn_outcomes")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .any(|outcome| {
                        outcome.get("kind").and_then(Value::as_str) == Some("stopped")
                            && outcome.get("stop_reason").and_then(Value::as_str)
                                == Some("max_turns")
                    })
        }
        "final_value" => {
            event.kind == BoundaryKind::Trigger
                && event
                    .observed
                    .pointer("/contract_execution/source/kind")
                    .and_then(Value::as_str)
                    == Some("fixed_dst_api_execution")
                && event
                    .observed
                    .pointer("/contract_execution/result/runtime_final_value_facts/outcome_kind")
                    .and_then(Value::as_str)
                    == Some("final_value")
                && event
                    .observed
                    .pointer(
                        "/contract_execution/result/runtime_final_value_facts/semantic_channel_observed",
                    )
                    .and_then(Value::as_bool)
                    == Some(true)
        }
        "observer_convergence" => event.kind == BoundaryKind::Observer,
        "exec_code" => {
            event.kind == BoundaryKind::ExecCode
                && event.observed.get("runtime_effect_outcome").is_some()
        }
        "trigger" => {
            event.kind == BoundaryKind::Trigger
                && event
                    .observed
                    .get("trigger_delivered")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
        }
        "durable_effect" => {
            event.kind == BoundaryKind::DurableEffect
                && event.observed.get("runtime_effect").is_some()
        }
        "observer_reconnect" => {
            event.kind == BoundaryKind::Observer
                && event
                    .observed
                    .get("reconnected")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
        }
        "provider_mutation" => {
            event.kind == BoundaryKind::ProviderMutation
                && event
                    .observed
                    .pointer("/provider_parser_matrix/matrix/real_provider_parser_execution")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
        }
        "backend_failure" => event.kind == BoundaryKind::BackendFailure,
        _ => false,
    }
}

#[derive(Clone, Debug, Serialize)]
struct FixedScriptEvent {
    schema: &'static str,
    sequence: usize,
    event: &'static str,
    profile: &'static str,
    proof_alias: String,
    exchange_alias: String,
    proof_name: String,
    provider_kind: String,
    outcome: String,
    transcript_path: String,
    request: FixedScriptEventRequest,
    response: FixedScriptEventResponse,
    terminal_classification: &'static str,
}

#[derive(Clone, Debug, Serialize)]
struct FixedScriptEventRequest {
    method: String,
    path: String,
    header_names: Vec<String>,
    body_bytes: usize,
    body_shape: serde_json::Value,
}

#[derive(Clone, Debug, Serialize)]
struct FixedScriptEventResponse {
    status: Option<u16>,
    header_names: Vec<String>,
    event_names: Vec<String>,
}

#[derive(Clone, Debug)]
struct ProofRun {
    name: String,
    provider_kind: String,
    endpoint: String,
    outcome: String,
    observed: serde_json::Value,
    transcript: FixedScriptTranscript,
}

#[derive(Clone, Debug, Serialize)]
struct FixedScriptTranscript {
    schema: &'static str,
    proof: String,
    provider_kind: String,
    endpoint: TranscriptEndpoint,
    timeline_at_semantics: &'static str,
    request_match: TranscriptRequestMatch,
    http_exchanges: Vec<ScriptedLlmHttpExchange>,
    response_events: Vec<TranscriptResponseEvent>,
    terminal: TranscriptTerminal,
    observed: serde_json::Value,
}

#[derive(Clone, Debug, Serialize)]
struct TranscriptEndpoint {
    method: String,
    path: String,
}

#[derive(Clone, Debug, Serialize)]
struct TranscriptRequestMatch {
    body_paths: Vec<String>,
    headers: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct TranscriptResponseEvent {
    at: u64,
    event: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    provider_event: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    status: Option<u16>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    headers: Vec<TranscriptHeader>,
}

#[derive(Clone, Debug, Serialize)]
struct TranscriptHeader {
    name: String,
    value: String,
}

#[derive(Clone, Debug, Serialize)]
struct TranscriptTerminal {
    classification: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    provider_result: Option<TranscriptProviderResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error_envelope: Option<TranscriptErrorEnvelope>,
}

#[derive(Clone, Debug, Serialize)]
struct TranscriptProviderResult {
    terminal_reason: String,
    full_text_bytes: usize,
    part_count: usize,
    usage: serde_json::Value,
}

#[derive(Clone, Debug, Serialize)]
struct TranscriptErrorEnvelope {
    kind: String,
    code: Option<String>,
    status: Option<u16>,
    retryable: bool,
    terminal_reason: String,
    raw_body_bytes: Option<usize>,
    headers: Vec<TranscriptHeader>,
    retry_after_ms: Option<u64>,
    request_body_snapshot: bool,
}

pub async fn run_fixed_script_profile(
    artifact_root: impl AsRef<Path>,
) -> Result<FixedScriptManifest, FixedScriptRunnerError> {
    let artifact_root = artifact_root.as_ref();
    std::fs::create_dir_all(artifact_root)?;

    let scripts = script_hash_manifest()?;
    let script_bundle_hash = script_bundle_hash(&scripts);
    let proof_runs = vec![
        prove_openai_compatible_tool_stream().await?,
        prove_openai_responses_text_stream().await?,
        prove_anthropic_messages_text_stream().await?,
        prove_openai_compatible_rate_limit().await?,
        prove_openai_compatible_validation().await?,
        prove_openai_compatible_disconnect().await?,
        prove_openai_compatible_response_start_timeout().await?,
        prove_openai_compatible_stream_chunk_timeout().await?,
        prove_openai_compatible_cancel_before_response_start().await?,
        prove_openai_compatible_retry_exhaustion().await?,
        prove_google_stream_generate_text().await?,
        prove_google_generate_text().await?,
        prove_google_generate_rate_limit().await?,
    ];
    let provider_matrix = provider_matrix(&scripts, &proof_runs);
    let fixed_events = fixed_script_events(&proof_runs);
    let events_sha256 = write_events_artifact(artifact_root, &fixed_events)?;
    let proofs = write_proof_artifacts(artifact_root, proof_runs)?;
    let summary = FixedScriptSummary {
        total_scripts: scripts.len(),
        total_proofs: proofs.len(),
        total_events: fixed_events.len(),
        passed: proofs.len(),
    };
    let manifest_path = artifact_root.join(FIXED_SCRIPT_MANIFEST);
    let summary_path = artifact_root.join(FIXED_SCRIPT_SUMMARY);
    let manifest = FixedScriptManifest {
        schema: "lash.sim.fixed-script-manifest.v1",
        profile: FIXED_SCRIPT_PROFILE,
        timeline_at_semantics: FIXED_SCRIPT_TIMELINE_AT_SEMANTICS,
        script_bundle_hash,
        events_path: FIXED_SCRIPT_EVENTS,
        events_sha256,
        event_count: fixed_events.len(),
        scripts,
        provider_matrix,
        provider_transport_exclusions: provider_transport_exclusions(),
        proofs,
        summary,
        manifest_path: manifest_path.clone(),
        summary_path: summary_path.clone(),
    };
    let lane_summary = lane_summary_for_manifest(&manifest);
    let body = serde_json::to_vec_pretty(&manifest)?;
    std::fs::write(&manifest_path, body)?;
    let body = serde_json::to_vec_pretty(&lane_summary)?;
    std::fs::write(&summary_path, body)?;
    Ok(manifest)
}

/// Anti-vacuity: across the whole generated seed set, the interleaving,
/// suspend/resume, and transport-mutation boundary classes must EACH appear at
/// least once. Several class oracles pass when their class is absent (you cannot
/// interleave one session, a run with no suspend boundary has nothing to check,
/// etc.); this lane-level guard fails loudly if a generator regression silently
/// drops a class so those oracles can never pass vacuously across the run.
fn assert_generated_class_coverage(
    event_lines: &[TraceEventLine],
    interleaving_depth_max: usize,
) -> Result<(), FixedScriptRunnerError> {
    if interleaving_depth_max < 2 {
        return Err(FixedScriptRunnerError::Assertion(format!(
            "no generated seed interleaved >= 2 live provider turns (peak {interleaving_depth_max}); the interleaving class is absent across the seed set"
        )));
    }
    let suspend_resume = event_lines.iter().any(|line| {
        line.event.observed.get("runtime_suspend").is_some()
            || line
                .event
                .payload
                .get("suspend_resume")
                .and_then(Value::as_bool)
                == Some(true)
    });
    if !suspend_resume {
        return Err(FixedScriptRunnerError::Assertion(
            "no suspend-resume boundary appeared across the generated seed set; the suspend/resume class is absent".to_string(),
        ));
    }
    let transport_mutation = event_lines.iter().any(|line| {
        line.event.kind == BoundaryKind::ProviderMutation
            && line
                .event
                .observed
                .get("mutation")
                .or_else(|| line.event.payload.get("mutation"))
                .and_then(Value::as_str)
                .is_some_and(is_transport_provider_mutation)
    });
    if !transport_mutation {
        return Err(FixedScriptRunnerError::Assertion(
            "no transport provider mutation appeared across the generated seed set; the transport-mutation class is absent".to_string(),
        ));
    }
    Ok(())
}

pub async fn run_generated_sim_profile(
    artifact_root: impl AsRef<Path>,
    profile: &str,
    seeds: usize,
    max_boundaries: usize,
) -> Result<GeneratedSimProfileReport, FixedScriptRunnerError> {
    let seed_values = (0..seeds.max(1))
        .map(|seed_index| generated_seed(profile, seed_index))
        .collect::<Vec<_>>();
    run_generated_sim_profile_for_seeds(artifact_root, profile, &seed_values, max_boundaries).await
}

pub async fn run_generated_sim_profile_for_seeds(
    artifact_root: impl AsRef<Path>,
    profile: &str,
    seed_values: &[u64],
    max_boundaries: usize,
) -> Result<GeneratedSimProfileReport, FixedScriptRunnerError> {
    validate_workload_profile(profile)?;
    if seed_values.is_empty() {
        return Err(FixedScriptRunnerError::Assertion(
            "generated simulation requires at least one seed".to_string(),
        ));
    }
    let artifact_root = artifact_root.as_ref();
    std::fs::create_dir_all(artifact_root)?;

    let provider_dir = artifact_root.join("provider-corpus");
    let fixed_manifest = run_fixed_script_profile(&provider_dir).await?;
    write_provider_script_manifest(artifact_root, &fixed_manifest)?;

    let runtime_proof = prove_runtime_facade_turn().await?;
    let seed_count = seed_values.len();
    let boundary_limit = max_boundaries.max(1);
    let replay_dir = artifact_root.join("replays");
    std::fs::create_dir_all(&replay_dir)?;

    let mut event_lines = Vec::new();
    let mut replay_reports = Vec::new();
    let mut oracle_verdicts = Vec::new();
    let mut model_store_sessions = 0;
    let mut boundary_events = 0;
    let mut runtime_turn_proofs = 0;
    let mut interleaving_depth_max = 0;
    let mut interleaving_depth_min = usize::MAX;

    for seed in seed_values.iter().copied() {
        let workload = generate_workload(seed, profile, boundary_limit)?;
        let trace_path = replay_dir.join(format!("seed-{seed:016x}.trace.json"));
        let trace =
            run_generated_workload(workload, &fixed_manifest.script_bundle_hash, &trace_path)
                .await?;
        boundary_events += trace.events.len();
        runtime_turn_proofs += trace
            .events
            .iter()
            .filter(|event| event.kind == BoundaryKind::Provider)
            .count();
        let seed_interleaving_depth = peak_concurrent_live_turns(&trace.events);
        interleaving_depth_max = interleaving_depth_max.max(seed_interleaving_depth);
        interleaving_depth_min = interleaving_depth_min.min(seed_interleaving_depth);
        model_store_sessions += trace.final_summary.session_count;
        oracle_verdicts.extend(trace.oracles.iter().cloned());
        for event in trace.events.iter().cloned() {
            event_lines.push(TraceEventLine::new(
                format!("seed-{seed:016x}"),
                seed,
                profile,
                event,
            ));
        }
        write_trace(&trace_path, &trace)?;
        let trace_sha256 = file_sha256(&trace_path)?;
        let replay = replay_trace(&trace_path, &trace)?;
        oracle_verdicts.push(replay.terminal_verdict.clone());
        let replay_report_path = replay_dir.join(format!("seed-{seed:016x}.replay.json"));
        let replay_report_sha256 = write_replay_report(&replay_report_path, &replay)?;
        let minimize_dir = replay_dir.join(format!("seed-{seed:016x}.minimize"));
        let minimize = minimize_trace(&trace_path, &trace, &minimize_dir)?;
        let minimize_report_path = minimize_dir.join("minimize.json");
        std::fs::write(&minimize_report_path, serde_json::to_vec_pretty(&minimize)?)?;
        let minimize_report_sha256 = file_sha256(&minimize_report_path)?;
        let minimized_trace_sha256 = file_sha256(&minimize.minimized_trace_path)?;
        let sqlite_database_path = replay_dir.join(format!("seed-{seed:016x}.sqlite-store"));
        let sqlite_replay_report_path =
            replay_dir.join(format!("seed-{seed:016x}.sqlite-replay.json"));
        // Cross-backend check: re-drive the SAME generated workload through the
        // SAME dynamic, concurrency-faithful runtime driver, backed by the real
        // `lash-sqlite-store` (session store + SQLite durable-effect controller),
        // and require the resulting observable Lash STATE (the abstract world
        // summary) to match the in-memory reference run exactly. Because both runs
        // share one driver and one scheduling discipline and differ ONLY in the
        // store, the comparison is apples-to-apples by construction: there is no
        // separate fixed-order, provider-event-gated re-drive that can deadlock or
        // spuriously diverge on active-turn / next-turn ingress timing. The
        // workload is regenerated deterministically from the seed (the original
        // was consumed by the reference run).
        let sqlite_workload = generate_workload(seed, profile, boundary_limit)?;
        // The cross-backend equivalence reference is a SERIALIZED in-memory run of
        // the same workload: it shares the durable re-run's serialize-provider-turn
        // discipline and differs ONLY in the backend store, so equality is a
        // well-posed durable-state check. (The concurrency-preserving search-lane
        // summary in `trace.final_summary` runs a different scheduling discipline
        // and is not directly comparable to the serialized durable re-run.)
        let serialized_reference = replay_workload_serialized_reference(&sqlite_workload).await?;
        let sqlite_summary =
            replay_workload_on_sqlite(&sqlite_workload, &sqlite_database_path).await?;
        let backend_verdict = replay_determinism(&serialized_reference, &sqlite_summary);
        oracle_verdicts.push(backend_verdict.clone());
        let sqlite_report = if backend_verdict.is_passed() {
            serde_json::json!({
                "schema": "lash.sim.sqlite-cross-backend-rerun.v1",
                "seed": seed,
                "profile": profile,
                "backend": "lash_sqlite_store",
                "driver": "unified_generated_runtime_world",
                "matches_reference": true,
                "reference_digest": serialized_reference.digest.clone(),
                "actual_digest": sqlite_summary.digest.clone(),
                "verdict": backend_verdict.clone(),
                "final_summary": sqlite_summary,
            })
        } else {
            serde_json::json!({
                "schema": "lash.sim.sqlite-cross-backend-rerun.v1",
                "seed": seed,
                "profile": profile,
                "backend": "lash_sqlite_store",
                "driver": "unified_generated_runtime_world",
                "matches_reference": false,
                "reference_digest": serialized_reference.digest.clone(),
                "actual_digest": sqlite_summary.digest.clone(),
                "verdict": backend_verdict.clone(),
                "reference_summary": serialized_reference,
                "actual_summary": sqlite_summary,
            })
        };
        std::fs::write(
            &sqlite_replay_report_path,
            serde_json::to_vec_pretty(&sqlite_report)?,
        )?;
        if !backend_verdict.is_passed() {
            return Err(FixedScriptRunnerError::Assertion(format!(
                "cross-backend SQLite re-run for seed {seed} ({profile}) diverged from the serialized in-memory reference: {}; wrote {}",
                backend_verdict.message,
                sqlite_replay_report_path.display()
            )));
        }
        let sqlite_replay_report_sha256 = file_sha256(&sqlite_replay_report_path)?;
        replay_reports.push(GeneratedReplayArtifact {
            seed,
            trace_path: relative_path(artifact_root, &trace_path),
            trace_sha256,
            replay_report_path: relative_path(artifact_root, &replay_report_path),
            replay_report_sha256,
            minimized_trace_path: relative_path(artifact_root, &minimize.minimized_trace_path),
            minimized_trace_sha256,
            failure_package_path: relative_path(artifact_root, &minimize.failure_package_path),
            minimize_report_path: relative_path(artifact_root, &minimize_report_path),
            minimize_report_sha256,
            sqlite_database_path: relative_path(artifact_root, &sqlite_database_path),
            sqlite_replay_report_path: relative_path(artifact_root, &sqlite_replay_report_path),
            sqlite_replay_report_sha256,
            replay_command: trace.replay_command,
            sqlite_replay_command: format!(
                "cargo run -p lash-sim --locked -- replay-sqlite {} --out {}",
                trace_path.display(),
                replay_dir.display()
            ),
        });
    }

    oracle_verdicts.push(runtime_proof.runtime_invariant.clone());
    oracle_verdicts.push(runtime_proof.provider_output_invariant.clone());
    oracle_verdicts.push(
        runtime_proof
            .pending_tool_completion
            .turn_suspension_invariant
            .clone(),
    );
    oracle_verdicts.push(
        runtime_proof
            .pending_tool_completion
            .scheduler_resolution_invariant
            .clone(),
    );
    oracle_verdicts.push(
        runtime_proof
            .pending_tool_completion
            .final_result_invariant
            .clone(),
    );
    oracle_verdicts.push(
        runtime_proof
            .final_value_semantic_channel
            .semantic_channel_invariant
            .clone(),
    );

    assert_generated_class_coverage(&event_lines, interleaving_depth_max)?;
    let events_sha256 = write_event_lines(&artifact_root.join(GENERATED_SIM_EVENTS), &event_lines)?;
    write_failure_artifact_shape(artifact_root)?;

    let oracle_passes = oracle_verdicts
        .iter()
        .filter(|verdict| verdict.status == OracleStatus::Passed)
        .count();
    let oracle_failures = oracle_verdicts.len() - oracle_passes;
    let scheduler_controlled_boundaries = event_lines
        .iter()
        .filter(|line| line.event.scheduler.scheduler_controlled)
        .count();
    let scheduler_owned_runtime_completions = event_lines
        .iter()
        .filter(|line| line.event.payload.get("runtime_completion").is_some())
        .count();
    let scenario_contract_oracles = oracle_verdicts
        .iter()
        .filter(|verdict| verdict.oracle_id.starts_with("sim.oracle.scenario."))
        .count();
    let scenario_contract_mini_oracles = oracle_verdicts
        .iter()
        .filter(|verdict| verdict.oracle_id.starts_with("sim.oracle.scenario-mini."))
        .count();
    let scenario_contract_slices =
        write_scenario_contract_slices(artifact_root, &event_lines, &oracle_verdicts)?;
    let scenario_contract_slice_count = scenario_contract_slices.len();
    let scenario_contract_packages = write_scenario_contract_packages(
        artifact_root,
        &event_lines,
        &oracle_verdicts,
        &replay_reports,
    )?;
    let scenario_contract_package_count = scenario_contract_packages.len();
    let generated_backend_regression_fixtures =
        write_generated_backend_regression_fixtures(artifact_root, &event_lines, &replay_reports)?;
    let generated_backend_regression_fixture_count = generated_backend_regression_fixtures.len();
    let generated_runtime_provider_matrix = generated_runtime_provider_matrix(&event_lines);
    let summary_path = artifact_root.join(GENERATED_SIM_SUMMARY);
    let report = GeneratedSimProfileReport {
        schema: "lash.sim.profile-summary.v1",
        profile: profile.to_string(),
        generator_version: GENERATOR_VERSION,
        script_bundle_hash: fixed_manifest.script_bundle_hash.clone(),
        provider_manifest_path: GENERATED_SIM_PROVIDER_MANIFEST,
        provider_matrix: fixed_manifest.provider_matrix.clone(),
        generated_runtime_provider_matrix,
        events_path: GENERATED_SIM_EVENTS,
        events_sha256,
        replay_reports,
        runtime_proof,
        scenario_contracts: scenario_contract_manifests(),
        scenario_contract_slices,
        scenario_contract_packages,
        generated_backend_regression_fixtures,
        model_only_boundary_reviews: model_only_boundary_reviews(),
        provider_transport_exclusions: fixed_manifest.provider_transport_exclusions.clone(),
        counts: GeneratedSimCounts {
            generated_seeds: seed_count,
            boundary_events,
            scheduler_controlled_boundaries,
            runtime_completion_registrations: scheduler_owned_runtime_completions,
            scheduler_owned_runtime_completions,
            fixed_provider_proofs: fixed_manifest.summary.total_proofs,
            runtime_proofs: runtime_turn_proofs + 3,
            replay_reports: seed_count,
            minimized_replays: seed_count,
            backend_replays: seed_count,
            scenario_contract_oracles,
            scenario_contract_mini_oracles,
            scenario_contract_slices: scenario_contract_slice_count,
            scenario_contract_packages: scenario_contract_package_count,
            generated_backend_regression_fixtures: generated_backend_regression_fixture_count,
            oracle_passes,
            oracle_failures,
            model_store_sessions,
            interleaving_depth_max,
            interleaving_depth_min: if interleaving_depth_min == usize::MAX {
                0
            } else {
                interleaving_depth_min
            },
        },
        oracle_verdicts,
        failure_artifact_shape: GENERATED_SIM_FAILURE_SHAPE,
        summary_path: summary_path.clone(),
    };
    std::fs::write(&summary_path, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

pub(crate) async fn run_generated_workload_for_fixture(
    workload: GeneratedWorkload,
    script_bundle_hash: &str,
) -> Result<SimulationTrace, FixedScriptRunnerError> {
    run_generated_workload(workload, script_bundle_hash, Path::new("trace.json")).await
}

/// Drive a generated workload through the scheduler-driven, concurrency-faithful
/// runtime world and return the delivered boundary log plus the abstract world
/// summary. This is the single driver shared by the reference in-memory run and
/// the cross-backend SQLite re-run (`replay_workload_on_sqlite`): the only thing
/// that varies is the `world`'s backend. Driving the SAME workload through the
/// SAME dynamic scheduler — rather than re-deriving a recorded trace in fixed
/// order with provider events gated to recorded counts — is what makes the
/// cross-backend comparison apples-to-apples.
async fn drive_generated_workload(
    world: &mut GeneratedRuntimeWorld,
    workload: &GeneratedWorkload,
) -> Result<
    (
        Vec<crate::scheduler::DeliveredBoundary>,
        AbstractWorldSummary,
    ),
    FixedScriptRunnerError,
> {
    let (initial_boundaries, mut completion_queue) =
        split_runtime_completion_boundaries(workload.boundaries.clone());
    let mut scheduler = BoundaryScheduler::with_events(workload.seed, initial_boundaries);
    let mut completion_state = RuntimeCompletionState {
        serialize_provider_turns: world.serialize_provider_turns,
        ..RuntimeCompletionState::default()
    };
    let mut store = ModelStore::default();
    let mut log = BoundaryDeliveryLog::default();
    let mut suspend_ready_at = 1_000_000u64;
    loop {
        // Serialized cross-backend barrier: while a provider turn is live, never
        // deliver a boundary scheduled at or after that turn's completion time
        // until the completion itself has been scheduled. The turn's own provider
        // releases all fall strictly before `final_ready_at`, so they still flow
        // through and drive the turn forward; only boundaries that would otherwise
        // jump ahead of the not-yet-scheduled completion are held. This removes
        // the sole source of backend-dependent delivery drift (a slow async store
        // letting a later boundary overtake the completion), so the in-memory and
        // durable serialized runs produce a byte-identical delivery order. The
        // SEARCH lane keeps full concurrency and is unaffected.
        if world.serialize_provider_turns
            && let Some(barrier) = world.min_active_final_ready_at()
            && scheduler
                .min_pending_at()
                .is_none_or(|next_at| next_at >= barrier)
        {
            world
                .schedule_finished_provider_turns(&mut scheduler)
                .await?;
            suspend_ready_at += 1;
            world
                .schedule_parked_suspend_resolutions(&mut scheduler, suspend_ready_at)
                .await?;
            world.sample_live_turn_highwater();
            // Spin until the live turn finishes and lands its completion (lowering
            // `min_pending_at` below the barrier), or it is gone. The provider
            // release deliveries that unblock the turn run on later iterations
            // because they are scheduled strictly before the barrier.
            if world.active_provider_turn_count() > 0
                && scheduler
                    .min_pending_at()
                    .is_none_or(|next_at| next_at >= barrier)
            {
                continue;
            }
        }
        let Some(mut delivered) = scheduler.deliver_next(Value::Null) else {
            world
                .schedule_finished_provider_turns(&mut scheduler)
                .await?;
            suspend_ready_at += 1;
            world
                .schedule_parked_suspend_resolutions(&mut scheduler, suspend_ready_at)
                .await?;
            world.sample_live_turn_highwater();
            if world.active_provider_turn_count() > 0 || world.pending_suspend_turn_count() > 0 {
                continue;
            }
            if scheduler.is_empty() {
                break;
            }
            continue;
        };
        let event = delivered.as_event();
        let observed = world.deliver_boundary(&event).await?;
        store.apply_observed_boundary(&event, &observed);
        delivered.observed = observed;
        completion_state.observe(&delivered);
        completion_queue.mark_completed(&delivered.boundary_id);
        register_ready_runtime_completions(
            &mut completion_queue,
            &mut completion_state,
            &mut scheduler,
            &delivered,
            world,
        )?;
        world.sample_live_turn_highwater();
        world
            .schedule_finished_provider_turns(&mut scheduler)
            .await?;
        suspend_ready_at += 1;
        world
            .schedule_parked_suspend_resolutions(&mut scheduler, suspend_ready_at)
            .await?;
        world.sample_live_turn_highwater();
        log.push(delivered);
    }
    if !completion_queue.is_empty() {
        return Err(FixedScriptRunnerError::Assertion(format!(
            "runtime completion queue ended with {} unresolved pending completions {:?} after registering {} and completing {}",
            completion_queue.pending_len(),
            completion_queue.pending_ids(),
            completion_queue.registered_len(),
            completion_queue.completed_len()
        )));
    }
    let mut events = log.into_vec();
    append_contract_execution_boundaries(&mut events, &mut store, workload.seed).await?;
    let final_summary = store.summary();
    // The event-derived interleaving highwater is the canonical, replay-stable
    // measure (recomputed identically from `events` on every backend). The
    // runtime world tracks the spawned-future highwater, which can only be equal
    // or larger; a smaller runtime measure would mean the bookkeeping lost a live
    // turn, so assert the bound holds before trusting either number.
    let event_peak = peak_concurrent_live_turns(&events);
    if event_peak > world.peak_concurrent_live_turns {
        return Err(FixedScriptRunnerError::Assertion(format!(
            "event-derived interleaving highwater {event_peak} exceeded the runtime-observed live turn highwater {}",
            world.peak_concurrent_live_turns
        )));
    }
    Ok((events, final_summary))
}

async fn append_contract_execution_boundaries(
    events: &mut Vec<crate::scheduler::DeliveredBoundary>,
    store: &mut ModelStore,
    seed: u64,
) -> Result<(), FixedScriptRunnerError> {
    let start_sequence = events.len();
    let mut scheduler = BoundaryScheduler::with_events(
        seed ^ 0x5e_3a_11_ce_c0_de,
        contract_execution_boundaries(events).await?,
    );
    while let Some(mut delivered) = scheduler.deliver_next_with(|event| store.apply_boundary(event))
    {
        delivered.sequence += start_sequence;
        events.push(delivered);
    }
    Ok(())
}

async fn contract_execution_boundaries(
    events: &[crate::scheduler::DeliveredBoundary],
) -> Result<Vec<BoundaryEvent>, FixedScriptRunnerError> {
    let mut next_at = events
        .iter()
        .map(|event| event.at)
        .max()
        .unwrap_or(0)
        .saturating_add(1);
    let mut proof_events = Vec::new();
    for execution in standard_protocol_contract_executions()? {
        proof_events.push(standard_protocol_execution_boundary(
            events, next_at, execution,
        )?);
        next_at = next_at.saturating_add(1);
    }
    for execution in rlm_protocol_contract_executions()? {
        proof_events.push(rlm_protocol_execution_boundary(events, next_at, execution)?);
        next_at = next_at.saturating_add(1);
    }
    for execution in agent_contract_executions().await? {
        proof_events.push(agent_contract_execution_boundary(
            events, next_at, execution,
        )?);
        next_at = next_at.saturating_add(1);
    }
    Ok(proof_events)
}

fn standard_protocol_execution_boundary(
    events: &[crate::scheduler::DeliveredBoundary],
    at: u64,
    mut execution: Value,
) -> Result<BoundaryEvent, FixedScriptRunnerError> {
    let contract = execution
        .get("contract")
        .and_then(Value::as_str)
        .unwrap_or("standard.protocol.contract");
    let proof_id = contract.replace('.', "-").replace('_', "-");
    match contract {
        "standard.initial_request_projection" | "standard.streamed_text_finalizes_once" => {
            let provider = first_successful_provider(events).ok_or_else(|| {
                FixedScriptRunnerError::Assertion(format!(
                    "could not anchor {contract} execution to a successful generated provider boundary"
                ))
            })?;
            execution
                .as_object_mut()
                .expect("contract execution object")
                .insert(
                    "generated_anchor".to_string(),
                    json!({
                        "provider_boundary": provider.boundary_id,
                        "actor": provider.actor_alias,
                        "provider_sequence": provider.sequence,
                    }),
                );
            Ok(contract_execution_boundary(
                &provider.actor_alias,
                &proof_id,
                at,
                execution,
            ))
        }
        "standard.empty_provider_response_error" | "standard.provider_error_without_checkpoint" => {
            let mutation = match contract {
                "standard.empty_provider_response_error" => "dropped_terminal_event",
                "standard.provider_error_without_checkpoint" => "rate_limit_error_envelope",
                _ => unreachable!(),
            };
            let provider = first_successful_provider(events).ok_or_else(|| {
                FixedScriptRunnerError::Assertion(format!(
                    "could not anchor {contract} execution to a successful generated provider boundary"
                ))
            })?;
            let parser = events
                .iter()
                .find(|event| {
                    event.kind == BoundaryKind::ProviderMutation
                        && event
                            .observed
                            .get("mutation")
                            .or_else(|| event.payload.get("mutation"))
                            .and_then(Value::as_str)
                            == Some(mutation)
                        && event
                            .observed
                            .pointer(
                                "/provider_parser_matrix/matrix/real_provider_parser_execution",
                            )
                            .and_then(Value::as_bool)
                            == Some(true)
                })
                .ok_or_else(|| {
                    FixedScriptRunnerError::Assertion(format!(
                        "could not anchor {contract} execution to real parser mutation `{mutation}`"
                    ))
                })?;
            execution
                .as_object_mut()
                .expect("contract execution object")
                .insert(
                    "generated_anchor".to_string(),
                    json!({
                        "provider_boundary": provider.boundary_id,
                        "provider_sequence": provider.sequence,
                        "provider_mutation_boundary": parser.boundary_id,
                        "mutation": mutation,
                        "real_provider_parser_execution": true,
                        "actor": provider.actor_alias,
                    }),
                );
            Ok(contract_execution_boundary(
                &provider.actor_alias,
                &proof_id,
                at,
                execution,
            ))
        }
        "standard.native_tool_loop_reenters_model"
        | "standard.parallel_tool_results_checkpoint_once"
        | "standard.tool_failure_feedback_reenters_model"
        | "standard.max_turns_after_tool_result" => {
            let Some((tool, provider)) = generated_tool_then_same_actor_provider(events) else {
                return Err(FixedScriptRunnerError::Assertion(format!(
                    "could not anchor {contract} execution to tool result and same-actor provider continuation"
                )));
            };
            execution
                .as_object_mut()
                .expect("contract execution object")
                .insert(
                    "generated_anchor".to_string(),
                    json!({
                        "tool_boundary": tool.boundary_id,
                        "continuation_provider_boundary": provider.boundary_id,
                        "actor": tool.actor_alias,
                        "tool_sequence": tool.sequence,
                        "continuation_provider_sequence": provider.sequence,
                        "same_actor_continuation": tool.actor_alias == provider.actor_alias
                            && provider.sequence > tool.sequence,
                    }),
                );
            Ok(contract_execution_boundary(
                &tool.actor_alias,
                &proof_id,
                at,
                execution,
            ))
        }
        other => Err(FixedScriptRunnerError::Assertion(format!(
            "no Standard contract execution boundary anchor registered for `{other}`"
        ))),
    }
}

fn generated_tool_then_same_actor_provider(
    events: &[crate::scheduler::DeliveredBoundary],
) -> Option<(
    &crate::scheduler::DeliveredBoundary,
    &crate::scheduler::DeliveredBoundary,
)> {
    events
        .iter()
        .filter(|event| {
            event.kind == BoundaryKind::Tool
                && event.observed.get("runtime_tool_output").is_some()
                && event
                    .observed
                    .get("execution_count")
                    .and_then(Value::as_u64)
                    == Some(1)
        })
        .find_map(|tool| {
            events
                .iter()
                .filter(|provider| {
                    provider.kind == BoundaryKind::Provider
                        && provider.actor_alias == tool.actor_alias
                        && provider.sequence > tool.sequence
                        && provider.observed.get("success").and_then(Value::as_bool) == Some(true)
                })
                .min_by_key(|provider| provider.sequence)
                .map(|provider| (tool, provider))
        })
}

fn agent_contract_execution_boundary(
    events: &[crate::scheduler::DeliveredBoundary],
    at: u64,
    execution: Value,
) -> Result<BoundaryEvent, FixedScriptRunnerError> {
    let provider = first_successful_provider(events).ok_or_else(|| {
        FixedScriptRunnerError::Assertion(
            "could not anchor Agent contract execution to a successful generated provider boundary"
                .to_string(),
        )
    })?;
    let proof_id = execution
        .get("contract")
        .and_then(Value::as_str)
        .unwrap_or("agent.contract")
        .replace('.', "-")
        .replace('_', "-");
    Ok(contract_execution_boundary(
        &provider.actor_alias,
        &proof_id,
        at,
        execution,
    ))
}

fn rlm_protocol_execution_boundary(
    events: &[crate::scheduler::DeliveredBoundary],
    at: u64,
    execution: Value,
) -> Result<BoundaryEvent, FixedScriptRunnerError> {
    let provider = first_successful_provider(events).ok_or_else(|| {
        FixedScriptRunnerError::Assertion(
            "could not anchor RLM protocol contract execution to a successful generated provider boundary"
                .to_string(),
        )
    })?;
    let proof_id = execution
        .get("contract")
        .and_then(Value::as_str)
        .unwrap_or("rlm.protocol.contract")
        .replace('.', "-")
        .replace('_', "-");
    Ok(contract_execution_boundary(
        &provider.actor_alias,
        &proof_id,
        at,
        execution,
    ))
}

fn contract_execution_boundary(
    actor_alias: &str,
    proof_id: &str,
    at: u64,
    contract_execution: Value,
) -> BoundaryEvent {
    BoundaryEvent::new(
        format!("{actor_alias}:contract-execution:{proof_id}"),
        actor_alias.to_string(),
        BoundaryKind::Trigger,
        at,
        format!("contract-execution.{proof_id}"),
        json!({
            "session": actor_alias,
            "source_key": format!("contract-execution/{actor_alias}/{proof_id}"),
            "started_process": false,
            "contract_execution": contract_execution,
        }),
    )
}

fn standard_protocol_contract_executions() -> Result<Vec<Value>, FixedScriptRunnerError> {
    Ok(vec![
        standard_initial_request_projection_execution()?,
        standard_empty_provider_response_error_execution()?,
        standard_provider_error_without_checkpoint_execution()?,
        standard_native_tool_loop_reenters_model_execution()?,
        standard_parallel_tool_results_checkpoint_once_execution()?,
        standard_tool_failure_feedback_reenters_model_execution()?,
        standard_streamed_text_finalizes_once_execution()?,
        standard_max_turn_after_tool_result_execution()?,
    ])
}

fn standard_initial_request_projection_execution() -> Result<Value, FixedScriptRunnerError> {
    let result = run_standard_protocol_contract(
        "standard initial request projection",
        "hello standard protocol",
        None,
        vec![],
    )?;
    contract_execution_payload(
        "standard.initial_request_projection",
        "crates/lash-protocol-standard/tests/protocol_scenarios.rs",
        "standard_protocol_scenario_projects_initial_request",
        result,
    )
}

fn standard_empty_provider_response_error_execution() -> Result<Value, FixedScriptRunnerError> {
    let result = run_standard_protocol_contract(
        "standard empty provider response error",
        "answer with something",
        None,
        vec![StandardContractStep::Llm {
            text_streamed: false,
            parts: vec![],
        }],
    )?;
    contract_execution_payload(
        "standard.empty_provider_response_error",
        "crates/lash-protocol-standard/tests/protocol_scenarios.rs",
        "standard_protocol_scenario_empty_model_response_stops_provider_error",
        result,
    )
}

fn standard_provider_error_without_checkpoint_execution() -> Result<Value, FixedScriptRunnerError> {
    let result = run_standard_protocol_contract(
        "standard provider error without checkpoint",
        "trigger provider failure",
        None,
        vec![StandardContractStep::LlmError(
            "upstream provider unavailable",
        )],
    )?;
    contract_execution_payload(
        "standard.provider_error_without_checkpoint",
        "crates/lash-protocol-standard/tests/protocol_scenarios.rs",
        "standard_protocol_scenario_provider_error_stops_without_checkpoint",
        result,
    )
}

fn standard_native_tool_loop_reenters_model_execution() -> Result<Value, FixedScriptRunnerError> {
    let result = run_standard_protocol_contract(
        "standard native tool loop reenters model",
        "read file",
        None,
        vec![
            StandardContractStep::Llm {
                text_streamed: false,
                parts: vec![
                    standard_text_part("Let me read that."),
                    standard_tool_call_part("tc1", "read_file", r#"{"path":"foo.txt"}"#),
                ],
            },
            StandardContractStep::ToolResults(vec![StandardContractToolResult::ok(
                "tc1",
                "read_file",
                json!("file contents"),
                "file contents",
            )]),
            StandardContractStep::Checkpoint,
        ],
    )?;
    contract_execution_payload(
        "standard.native_tool_loop_reenters_model",
        "crates/lash-protocol-standard/tests/protocol_scenarios.rs",
        "standard_protocol_scenario_native_tool_loop_reenters_model_after_checkpoint",
        result,
    )
}

fn standard_parallel_tool_results_checkpoint_once_execution()
-> Result<Value, FixedScriptRunnerError> {
    let result = run_standard_protocol_contract(
        "standard parallel tool results checkpoint once",
        "read two files",
        None,
        vec![
            StandardContractStep::Llm {
                text_streamed: false,
                parts: vec![
                    standard_tool_call_part("tc1", "read_file", r#"{"path":"left.txt"}"#),
                    standard_tool_call_part("tc2", "read_file", r#"{"path":"right.txt"}"#),
                ],
            },
            StandardContractStep::ToolResults(vec![
                StandardContractToolResult::ok(
                    "tc1",
                    "read_file",
                    json!("left contents"),
                    "left contents",
                ),
                StandardContractToolResult::ok(
                    "tc2",
                    "read_file",
                    json!("right contents"),
                    "right contents",
                ),
            ]),
            StandardContractStep::Checkpoint,
        ],
    )?;
    contract_execution_payload(
        "standard.parallel_tool_results_checkpoint_once",
        "crates/lash-protocol-standard/tests/protocol_scenarios.rs",
        "standard_protocol_scenario_parallel_tool_results_checkpoint_once",
        result,
    )
}

fn standard_tool_failure_feedback_reenters_model_execution() -> Result<Value, FixedScriptRunnerError>
{
    let result = run_standard_protocol_contract(
        "standard tool failure feedback reenters model",
        "search docs",
        None,
        vec![
            StandardContractStep::Llm {
                text_streamed: false,
                parts: vec![standard_tool_call_part(
                    "tc1",
                    "search",
                    r#"{"query":"missing term"}"#,
                )],
            },
            StandardContractStep::ToolResults(vec![StandardContractToolResult::failure(
                "tc1",
                "search",
                "search_failed",
                "index unavailable",
                "search failed: index unavailable",
            )]),
            StandardContractStep::Checkpoint,
        ],
    )?;
    contract_execution_payload(
        "standard.tool_failure_feedback_reenters_model",
        "crates/lash-protocol-standard/tests/protocol_scenarios.rs",
        "standard_protocol_scenario_tool_failure_feedback_reenters_model_after_checkpoint",
        result,
    )
}

fn standard_streamed_text_finalizes_once_execution() -> Result<Value, FixedScriptRunnerError> {
    let result = run_standard_protocol_contract(
        "standard streamed text finalizes once",
        "answer directly",
        None,
        vec![
            StandardContractStep::Llm {
                text_streamed: true,
                parts: vec![standard_text_part("streamed done")],
            },
            StandardContractStep::Checkpoint,
        ],
    )?;
    contract_execution_payload(
        "standard.streamed_text_finalizes_once",
        "crates/lash-protocol-standard/tests/protocol_scenarios.rs",
        "standard_protocol_scenario_streamed_text_finishes_without_duplicate_delta",
        result,
    )
}

fn standard_max_turn_after_tool_result_execution() -> Result<Value, FixedScriptRunnerError> {
    let result = run_standard_protocol_contract(
        "standard max turns after tool result",
        "use a tool once",
        Some(1),
        vec![
            StandardContractStep::Llm {
                text_streamed: false,
                parts: vec![standard_tool_call_part("tc1", "test", "{}")],
            },
            StandardContractStep::ToolResults(vec![StandardContractToolResult::ok(
                "tc1",
                "test",
                json!("ok"),
                "ok",
            )]),
        ],
    )?;
    contract_execution_payload(
        "standard.max_turns_after_tool_result",
        "crates/lash-protocol-standard/tests/protocol_scenarios.rs",
        "standard_protocol_scenario_max_turns_terminates_after_tool_result",
        result,
    )
}

#[derive(Clone)]
enum StandardContractStep {
    Llm {
        text_streamed: bool,
        parts: Vec<LlmOutputPart>,
    },
    LlmError(&'static str),
    ToolResults(Vec<StandardContractToolResult>),
    Checkpoint,
}

#[derive(Clone)]
struct StandardContractToolResult {
    call_id: &'static str,
    tool_name: &'static str,
    output: lash_core::ToolCallOutput,
    model_return_text: &'static str,
    status: &'static str,
    error_code: Option<&'static str>,
}

impl StandardContractToolResult {
    fn ok(
        call_id: &'static str,
        tool_name: &'static str,
        output: Value,
        model_return_text: &'static str,
    ) -> Self {
        Self {
            call_id,
            tool_name,
            output: lash_core::ToolCallOutput::success(output),
            model_return_text,
            status: "success",
            error_code: None,
        }
    }

    fn failure(
        call_id: &'static str,
        tool_name: &'static str,
        code: &'static str,
        message: &'static str,
        model_return_text: &'static str,
    ) -> Self {
        Self {
            call_id,
            tool_name,
            output: lash_core::ToolCallOutput::failure(lash_core::ToolFailure::tool(
                lash_core::ToolFailureClass::Execution,
                code,
                message,
            )),
            model_return_text,
            status: "failure",
            error_code: Some(code),
        }
    }

    fn completed_call(&self, args: Value) -> lash_core::sansio::CompletedToolCall {
        lash_core::sansio::CompletedToolCall {
            call_id: self.call_id.to_string(),
            tool_name: self.tool_name.to_string(),
            args,
            output: self.output.clone(),
            model_return: lash_core::ModelToolReturn {
                call_id: self.call_id.to_string(),
                tool_name: self.tool_name.to_string(),
                parts: vec![lash_core::ModelToolReturnPart::text(self.model_return_text)],
            },
            duration_ms: 1,
            replay: None,
        }
    }

    fn summary(&self) -> Value {
        json!({
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "status": self.status,
            "error_code": self.error_code,
            "model_return_text": self.model_return_text,
        })
    }
}

#[derive(Default)]
struct StandardContractObserved {
    initial_request_text: Option<String>,
    tool_calls: Vec<Value>,
    tool_results: Vec<Value>,
    checkpoints: Vec<&'static str>,
    llm_response_full_texts: Vec<String>,
    llm_response_parts: Vec<Vec<Value>>,
    llm_call_count: usize,
    text_deltas: Vec<String>,
    errors: Vec<String>,
    turn_outcomes: Vec<lash_core::TurnOutcome>,
}

impl StandardContractObserved {
    fn record(&mut self, effects: &[lash_core::Effect]) {
        for effect in effects {
            match effect {
                lash_core::Effect::LlmCall { request, .. } => {
                    if self.initial_request_text.is_none() {
                        self.initial_request_text = Some(format!("{:?}", request.messages));
                    }
                    self.llm_call_count += 1;
                }
                lash_core::Effect::ToolCalls { calls, .. } => {
                    self.tool_calls.extend(calls.iter().map(|call| {
                        json!({
                            "call_id": call.call_id,
                            "tool_name": call.tool_name,
                            "args": call.args,
                        })
                    }));
                }
                lash_core::Effect::Checkpoint { checkpoint, .. } => {
                    self.checkpoints.push(checkpoint_kind_name(*checkpoint));
                }
                lash_core::Effect::Emit(lash_core::SessionEvent::TextDelta { content }) => {
                    self.text_deltas.push(content.clone());
                }
                lash_core::Effect::Emit(lash_core::SessionEvent::Error { message, .. }) => {
                    self.errors.push(message.clone());
                }
                lash_core::Effect::Emit(lash_core::SessionEvent::TurnOutcome { outcome }) => {
                    self.turn_outcomes.push(outcome.clone());
                }
                _ => {}
            }
        }
    }
}

fn run_standard_protocol_contract(
    scenario_name: &'static str,
    user_message: &'static str,
    max_turns: Option<usize>,
    steps: Vec<StandardContractStep>,
) -> Result<Value, FixedScriptRunnerError> {
    let mut config = standard_contract_turn_machine_config();
    config.max_turns = max_turns;
    let mut machine = lash_core::TurnMachine::new(
        config,
        vec![contract_user_message(user_message)],
        Arc::new(Vec::new()),
        0,
    );
    let mut observed = StandardContractObserved::default();
    let mut effects = drain_contract_turn_machine_effects(&mut machine);
    observed.record(&effects);

    for step in steps {
        match step {
            StandardContractStep::Llm {
                text_streamed,
                parts,
            } => {
                let llm_id = *find_contract_llm_call(&effects).ok_or_else(|| {
                    FixedScriptRunnerError::Assertion(format!(
                        "{scenario_name} expected a pending LLM call"
                    ))
                })?;
                let expected_parts = parts.clone();
                let expected_full_text = standard_full_text(&expected_parts);
                let expected_part_summary = llm_output_parts_contract_summary(&expected_parts);
                let response = llm_response_with_parts(expected_full_text.clone(), parts);
                require(
                    response.full_text == expected_full_text,
                    format!(
                        "{scenario_name} provider response full_text changed: expected {:?}, got {:?}",
                        expected_full_text, response.full_text
                    ),
                )?;
                let response_part_summary = llm_output_parts_contract_summary(&response.parts);
                require(
                    response_part_summary == expected_part_summary,
                    format!(
                        "{scenario_name} provider response parts changed: expected {:?}, got {:?}",
                        expected_part_summary, response_part_summary
                    ),
                )?;
                observed
                    .llm_response_full_texts
                    .push(response.full_text.clone());
                observed.llm_response_parts.push(response_part_summary);
                machine.handle_response(lash_core::sansio::Response::LlmComplete {
                    id: llm_id,
                    text_streamed,
                    result: Ok(response),
                });
            }
            StandardContractStep::LlmError(message) => {
                let llm_id = *find_contract_llm_call(&effects).ok_or_else(|| {
                    FixedScriptRunnerError::Assertion(format!(
                        "{scenario_name} expected a pending LLM call before provider error"
                    ))
                })?;
                machine.handle_response(lash_core::sansio::Response::LlmComplete {
                    id: llm_id,
                    text_streamed: false,
                    result: Err(standard_llm_error(message)),
                });
            }
            StandardContractStep::ToolResults(results) => {
                let (tool_id, calls) = effects
                    .iter()
                    .find_map(|effect| match effect {
                        lash_core::Effect::ToolCalls { id, calls } => Some((*id, calls.clone())),
                        _ => None,
                    })
                    .ok_or_else(|| {
                        FixedScriptRunnerError::Assertion(format!(
                            "{scenario_name} expected pending native tool calls"
                        ))
                    })?;
                require(
                    calls.len() == results.len()
                        && calls.iter().zip(&results).all(|(call, result)| {
                            call.call_id == result.call_id && call.tool_name == result.tool_name
                        }),
                    format!("{scenario_name} native tool-call shape changed"),
                )?;
                observed
                    .tool_results
                    .extend(results.iter().map(StandardContractToolResult::summary));
                machine.handle_response(lash_core::sansio::Response::ToolResults {
                    id: tool_id,
                    results: calls
                        .iter()
                        .zip(results)
                        .map(|(call, result)| result.completed_call(call.args.clone()))
                        .collect(),
                });
            }
            StandardContractStep::Checkpoint => {
                let checkpoint_id = effects
                    .iter()
                    .find_map(|effect| match effect {
                        lash_core::Effect::Checkpoint { id, .. } => Some(*id),
                        _ => None,
                    })
                    .ok_or_else(|| {
                        FixedScriptRunnerError::Assertion(format!(
                            "{scenario_name} expected pending checkpoint"
                        ))
                    })?;
                machine.handle_response(lash_core::sansio::Response::Checkpoint {
                    id: checkpoint_id,
                    delivery: lash_core::sansio::CheckpointDelivery::default(),
                });
            }
        }
        effects = drain_contract_turn_machine_effects(&mut machine);
        observed.record(&effects);
    }

    Ok(json!({
        "execution_api": "lash_core::sansio::TurnMachine",
        "driver": "lash_protocol_standard::StandardDriver",
        "scenario_name": scenario_name,
        "user_message": user_message,
        "max_turns": max_turns,
        "initial_request_contains_user_message": observed
            .initial_request_text
            .as_deref()
            .is_some_and(|request| request.contains(user_message)),
        "llm_call_count": observed.llm_call_count,
        "llm_response_full_texts": observed.llm_response_full_texts,
        "llm_response_parts": observed.llm_response_parts,
        "done": machine.is_done(),
        "tool_calls": observed.tool_calls,
        "tool_results": observed.tool_results,
        "checkpoints": observed.checkpoints,
        "text_delta_count": observed.text_deltas.len(),
        "text_deltas": observed.text_deltas,
        "errors": observed.errors,
        "turn_outcomes": observed.turn_outcomes.iter().map(turn_outcome_contract_json).collect::<Vec<_>>(),
    }))
}

fn standard_text_part(text: &str) -> LlmOutputPart {
    LlmOutputPart::Text {
        text: text.to_string(),
        response_meta: None,
    }
}

fn standard_tool_call_part(call_id: &str, tool_name: &str, input_json: &str) -> LlmOutputPart {
    LlmOutputPart::ToolCall {
        call_id: call_id.to_string(),
        tool_name: tool_name.to_string(),
        input_json: input_json.to_string(),
        replay: None,
    }
}

fn standard_full_text(parts: &[LlmOutputPart]) -> String {
    parts
        .iter()
        .filter_map(|part| match part {
            LlmOutputPart::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn llm_response_with_parts(full_text: String, parts: Vec<LlmOutputPart>) -> LlmResponse {
    let mut response = LlmResponse::default();
    response.full_text = full_text;
    response.parts = parts;
    response
}

fn text_llm_response(text: impl Into<String>) -> LlmResponse {
    let text = text.into();
    llm_response_with_parts(
        text.clone(),
        vec![LlmOutputPart::Text {
            text,
            response_meta: None,
        }],
    )
}

fn tool_call_llm_response(call_id: &str, tool_name: &str, input_json: &str) -> LlmResponse {
    let mut response = LlmResponse::default();
    response.parts = vec![LlmOutputPart::ToolCall {
        call_id: call_id.to_string(),
        tool_name: tool_name.to_string(),
        input_json: input_json.to_string(),
        replay: None,
    }];
    response
}

fn llm_output_parts_contract_summary(parts: &[LlmOutputPart]) -> Vec<Value> {
    parts
        .iter()
        .map(|part| match part {
            LlmOutputPart::Text { text, .. } => json!({
                "kind": "text",
                "text": text,
            }),
            LlmOutputPart::Reasoning { text, .. } => json!({
                "kind": "reasoning",
                "text": text,
            }),
            LlmOutputPart::ToolCall {
                call_id,
                tool_name,
                input_json,
                ..
            } => json!({
                "kind": "tool_call",
                "call_id": call_id,
                "tool_name": tool_name,
                "input_json": input_json,
            }),
        })
        .collect()
}

fn response_text_part(response: &LlmResponse) -> Option<&str> {
    response.parts.iter().find_map(|part| match part {
        LlmOutputPart::Text { text, .. } => Some(text.as_str()),
        _ => None,
    })
}

fn standard_llm_error(message: &str) -> lash_core::LlmCallError {
    lash_core::LlmCallError {
        message: message.to_string(),
        retryable: false,
        raw: None,
        code: Some("test_provider_error".to_string()),
        terminal_reason: LlmTerminalReason::ProviderError,
        request_body: None,
    }
}

pub(crate) fn replay_contract_execution(contract: &str) -> Result<Value, FixedScriptRunnerError> {
    match contract {
        other if other.starts_with("standard.") => standard_protocol_contract_executions()?
            .into_iter()
            .find(|execution| execution.get("contract").and_then(Value::as_str) == Some(other))
            .ok_or_else(|| {
                FixedScriptRunnerError::Assertion(format!(
                    "no replayable fixed Standard contract execution registered for `{other}`"
                ))
            }),
        other if other.starts_with("rlm.") => rlm_protocol_contract_executions()?
            .into_iter()
            .find(|execution| execution.get("contract").and_then(Value::as_str) == Some(other))
            .ok_or_else(|| {
                FixedScriptRunnerError::Assertion(format!(
                    "no replayable fixed RLM contract execution registered for `{other}`"
                ))
            }),
        other if other.starts_with("agent.") => replay_agent_contract_execution(other),
        other => Err(FixedScriptRunnerError::Assertion(format!(
            "contract execution replay is not registered for `{other}`"
        ))),
    }
}

fn replay_agent_contract_execution(contract: &str) -> Result<Value, FixedScriptRunnerError> {
    let contract = contract.to_string();
    let runner = agent_contract_runner(&contract)?;
    run_on_sim_harness_stack(
        format!("replay-{contract}-contract"),
        SIM_HARNESS_STACK_LIMIT_BYTES,
        move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(FixedScriptRunnerError::Io)?;
            runner(&runtime)
        },
    )
}

fn rlm_protocol_contract_executions() -> Result<Vec<Value>, FixedScriptRunnerError> {
    Ok(vec![
        rlm_natural_prose_finalizes_execution()?,
        rlm_typed_prose_requires_finish_execution()?,
        rlm_finish_required_max_turn_stop_execution()?,
        rlm_exec_error_max_turn_stop_execution()?,
        rlm_typed_finish_emits_outcome_and_done_execution()?,
        rlm_finish_required_diagnostic_counts_execution()?,
        rlm_natural_diagnostic_counts_execution()?,
        rlm_cell_diagnostic_counts_execution()?,
        rlm_retired_marker_plain_lashlang_text_execution()?,
        rlm_lashlang_cell_exec_continues_execution()?,
        rlm_empty_options_natural_default_execution()?,
        rlm_exec_result_no_tool_call_replay_execution()?,
        rlm_exec_tool_control_frame_switch_terminal_execution()?,
        rlm_exec_tool_control_fail_terminal_execution()?,
        rlm_natural_allows_finish_value_execution()?,
        rlm_typed_schema_mismatch_repair_loop_execution()?,
        rlm_typed_schema_any_of_mismatch_execution()?,
    ])
}

fn rlm_natural_prose_finalizes_execution() -> Result<Value, FixedScriptRunnerError> {
    let result = run_rlm_protocol_contract(
        "rlm natural prose finalizes",
        "hello",
        RlmTermination::Natural,
        None,
        None,
        vec![
            RlmContractStep::Llm(vec![rlm_text_part("Hello there!")]),
            RlmContractStep::Checkpoint,
        ],
    )?;
    contract_execution_payload(
        "rlm.natural_prose_finalizes",
        "crates/lash-protocol-rlm/tests/protocol_drivers/scenarios.rs",
        "rlm_protocol_scenario_prose_only_response_finishes_by_default",
        result,
    )
}

fn rlm_typed_prose_requires_finish_execution() -> Result<Value, FixedScriptRunnerError> {
    let result = run_rlm_protocol_contract(
        "rlm typed prose requires finish",
        "hello",
        RlmTermination::FinishRequired { schema: None },
        None,
        None,
        vec![
            RlmContractStep::Llm(vec![rlm_text_part("Hello there!")]),
            RlmContractStep::Checkpoint,
        ],
    )?;
    contract_execution_payload(
        "rlm.typed_prose_requires_finish",
        "crates/lash-protocol-rlm/tests/protocol_drivers/scenarios.rs",
        "rlm_protocol_scenario_typed_prose_only_response_requests_finish",
        result,
    )
}

fn rlm_finish_required_max_turn_stop_execution() -> Result<Value, FixedScriptRunnerError> {
    let result = run_rlm_protocol_contract(
        "rlm finish-required prose max-turn stop",
        "hello",
        RlmTermination::FinishRequired { schema: None },
        Some(1),
        None,
        vec![RlmContractStep::Llm(vec![rlm_text_part(
            "plain prose cannot finish finish-required RLM",
        )])],
    )?;
    contract_execution_payload(
        "rlm.finish_required_max_turn_stop",
        "crates/lash-protocol-rlm/tests/protocol_drivers/scenarios.rs",
        "rlm_protocol_scenario_finish_required_prose_at_max_turns_stops_without_retry_prompt",
        result,
    )
}

fn rlm_exec_error_max_turn_stop_execution() -> Result<Value, FixedScriptRunnerError> {
    let result = run_rlm_protocol_contract(
        "rlm finish-required exec error max-turn stop",
        "run bad code",
        RlmTermination::FinishRequired { schema: None },
        Some(1),
        None,
        vec![
            RlmContractStep::Llm(vec![rlm_text_part(&rlm_lashlang_block("missing_name"))]),
            RlmContractStep::Exec(rlm_exec_response(
                &[],
                Some("unknown variable `missing_name`"),
                None,
            )),
        ],
    )?;
    contract_execution_payload(
        "rlm.exec_error_max_turn_stop",
        "crates/lash-protocol-rlm/tests/protocol_drivers/scenarios.rs",
        "rlm_protocol_scenario_finish_required_exec_error_at_max_turns_stops_without_retry",
        result,
    )
}

fn rlm_typed_finish_emits_outcome_and_done_execution() -> Result<Value, FixedScriptRunnerError> {
    let result = run_rlm_protocol_contract(
        "rlm typed finish emits outcome and done",
        "return typed data",
        RlmTermination::FinishRequired {
            schema: Some(json!({
                "type": "object",
                "properties": {
                    "ok": { "type": "boolean" }
                },
                "required": ["ok"],
                "additionalProperties": false
            })),
        },
        None,
        None,
        vec![
            RlmContractStep::Llm(vec![rlm_text_part(&rlm_lashlang_block(
                "finish { ok: true }",
            ))]),
            RlmContractStep::Exec(rlm_exec_response(&[], None, Some(json!({ "ok": true })))),
            RlmContractStep::Checkpoint,
        ],
    )?;
    contract_execution_payload(
        "rlm.typed_finish_emits_outcome_and_done",
        "crates/lash-protocol-rlm/tests/protocol_drivers/scenarios.rs",
        "rlm_protocol_scenario_typed_finish_emits_turn_outcome_and_done",
        result,
    )
}

fn rlm_finish_required_diagnostic_counts_execution() -> Result<Value, FixedScriptRunnerError> {
    let result = run_rlm_protocol_contract(
        "rlm finish-required diagnostic counts",
        "hello",
        RlmTermination::FinishRequired { schema: None },
        None,
        None,
        vec![RlmContractStep::Llm(vec![rlm_text_part("Hello there!")])],
    )?;
    contract_execution_payload(
        "rlm.finish_required_diagnostic_counts",
        "crates/lash-protocol-rlm/tests/protocol_drivers/scenarios.rs",
        "rlm_protocol_scenario_finish_required_prose_only_diagnostic_has_clean_counts",
        result,
    )
}

fn rlm_natural_diagnostic_counts_execution() -> Result<Value, FixedScriptRunnerError> {
    let result = run_rlm_protocol_contract(
        "rlm natural diagnostic counts",
        "hello",
        RlmTermination::Natural,
        None,
        None,
        vec![RlmContractStep::Llm(vec![rlm_text_part("Hello there!")])],
    )?;
    contract_execution_payload(
        "rlm.natural_diagnostic_counts",
        "crates/lash-protocol-rlm/tests/protocol_drivers/scenarios.rs",
        "rlm_protocol_scenario_natural_prose_only_diagnostic_has_clean_counts",
        result,
    )
}

fn rlm_cell_diagnostic_counts_execution() -> Result<Value, FixedScriptRunnerError> {
    let result = run_rlm_protocol_contract(
        "rlm cell diagnostic counts",
        "run some code",
        RlmTermination::Natural,
        None,
        None,
        vec![
            RlmContractStep::Llm(vec![
                rlm_reasoning_part("Checking state."),
                rlm_text_part(&rlm_lashlang_block_with_prose("Ready.", "print \"hi\"")),
            ]),
            RlmContractStep::Exec(rlm_exec_response(&["hi\n"], None, None)),
        ],
    )?;
    contract_execution_payload(
        "rlm.cell_diagnostic_counts",
        "crates/lash-protocol-rlm/tests/protocol_drivers/scenarios.rs",
        "rlm_protocol_scenario_cell_reasoning_prose_code_diagnostic_has_clean_counts",
        result,
    )
}

fn rlm_retired_marker_plain_lashlang_text_execution() -> Result<Value, FixedScriptRunnerError> {
    let assistant_prose = "First.";
    let code = "text = \"%%lashlang is just source here\"\nprint text";
    let result = run_rlm_protocol_contract(
        "rlm retired marker plain LashLang text",
        "run some code",
        RlmTermination::Natural,
        None,
        None,
        vec![RlmContractStep::Llm(vec![rlm_text_part(
            &rlm_lashlang_block_with_prose(assistant_prose, code),
        )])],
    )?;
    contract_execution_payload(
        "rlm.retired_marker_plain_lashlang_text",
        "crates/lash-protocol-rlm/tests/protocol_drivers/scenarios.rs",
        "rlm_protocol_scenario_retired_percent_marker_inside_source_is_plain_lashlang_text",
        result,
    )
}

fn rlm_lashlang_cell_exec_continues_execution() -> Result<Value, FixedScriptRunnerError> {
    let result = run_rlm_protocol_contract(
        "rlm LashLang cell exec continues",
        "run some code",
        RlmTermination::Natural,
        None,
        None,
        vec![
            RlmContractStep::Llm(vec![rlm_text_part(&rlm_lashlang_block_with_prose(
                "Quick check.\n",
                "print \"hi\"",
            ))]),
            RlmContractStep::Exec(rlm_exec_response(&["hi\n"], None, None)),
            RlmContractStep::Checkpoint,
        ],
    )?;
    contract_execution_payload(
        "rlm.lashlang_cell_exec_continues",
        "crates/lash-protocol-rlm/tests/protocol_drivers/scenarios.rs",
        "rlm_protocol_scenario_lashlang_cell_runs_exec_and_continues",
        result,
    )
}

fn rlm_empty_options_natural_default_execution() -> Result<Value, FixedScriptRunnerError> {
    let result = run_rlm_protocol_contract(
        "rlm empty options natural default",
        "finish",
        RlmTermination::Natural,
        None,
        Some(lash_core::ProtocolTurnOptions::empty()),
        vec![
            RlmContractStep::Llm(vec![rlm_text_part(&rlm_lashlang_block("finish \"done\""))]),
            RlmContractStep::Exec(rlm_exec_response(&[], None, Some(json!("done")))),
            RlmContractStep::Checkpoint,
        ],
    )?;
    contract_execution_payload(
        "rlm.empty_options_natural_default",
        "crates/lash-protocol-rlm/tests/protocol_drivers/scenarios.rs",
        "rlm_protocol_scenario_empty_turn_options_use_natural_default",
        result,
    )
}

fn rlm_exec_result_no_tool_call_replay_execution() -> Result<Value, FixedScriptRunnerError> {
    let result = run_rlm_protocol_contract(
        "rlm exec result no tool-call replay",
        "run a tool",
        RlmTermination::Natural,
        None,
        None,
        vec![
            RlmContractStep::Llm(vec![rlm_text_part(&rlm_lashlang_block(
                "x = await tools.read_file({ path: \"foo\" })?",
            ))]),
            RlmContractStep::Exec(rlm_exec_response_with_tool_calls(
                &[],
                None,
                None,
                vec![rlm_tool_call_record(
                    "rlm-call-1",
                    "read_file",
                    json!({ "path": "foo" }),
                    lash_core::ToolCallOutput::success(json!("contents")),
                    7,
                )],
                7,
            )),
        ],
    )?;
    contract_execution_payload(
        "rlm.exec_result_no_tool_call_replay",
        "crates/lash-protocol-rlm/tests/protocol_drivers/scenarios.rs",
        "rlm_protocol_scenario_exec_result_does_not_store_tool_call_ids_or_replay_tool_events",
        result,
    )
}

fn rlm_exec_tool_control_frame_switch_terminal_execution() -> Result<Value, FixedScriptRunnerError>
{
    let result = run_rlm_protocol_contract(
        "rlm exec tool-control frame switch terminal",
        "run a custom frame-switch tool",
        RlmTermination::Natural,
        None,
        None,
        vec![
            RlmContractStep::Llm(vec![rlm_text_part(&rlm_lashlang_block(
                "x = await tools.custom_frame_switch({})?",
            ))]),
            RlmContractStep::Exec(rlm_exec_response_with_tool_calls(
                &[],
                None,
                None,
                vec![rlm_tool_call_record(
                    "custom-call-1",
                    "custom_frame_switch",
                    json!({}),
                    lash_core::ToolCallOutput::success(json!({ "ok": true })).with_control(
                        lash_core::ToolControl::SwitchAgentFrame {
                            frame_id: "next-frame".to_string(),
                            initial_nodes: Vec::new(),
                            task: Some("continue".to_string()),
                        },
                    ),
                    3,
                )],
                3,
            )),
            RlmContractStep::Checkpoint,
        ],
    )?;
    contract_execution_payload(
        "rlm.exec_tool_control_frame_switch_terminal",
        "crates/lash-protocol-rlm/tests/protocol_drivers/scenarios.rs",
        "rlm_protocol_scenario_exec_any_tool_control_frame_switch_is_terminal",
        result,
    )
}

fn rlm_exec_tool_control_fail_terminal_execution() -> Result<Value, FixedScriptRunnerError> {
    let result = run_rlm_protocol_contract(
        "rlm exec tool-control fail terminal",
        "run a custom failure tool",
        RlmTermination::Natural,
        None,
        None,
        vec![
            RlmContractStep::Llm(vec![rlm_text_part(&rlm_lashlang_block(
                "x = await tools.custom_fail({})?",
            ))]),
            RlmContractStep::Exec(rlm_exec_response_with_tool_calls(
                &[],
                None,
                None,
                vec![rlm_tool_call_record(
                    "custom-call-1",
                    "custom_fail",
                    json!({}),
                    lash_core::ToolCallOutput::success(json!({ "ok": true })).with_control(
                        lash_core::ToolControl::Fail {
                            failure: lash_core::ToolFailure::tool(
                                lash_core::ToolFailureClass::Execution,
                                "custom_fail",
                                "no valid result",
                            ),
                        },
                    ),
                    3,
                )],
                3,
            )),
            RlmContractStep::Checkpoint,
        ],
    )?;
    contract_execution_payload(
        "rlm.exec_tool_control_fail_terminal",
        "crates/lash-protocol-rlm/tests/protocol_drivers/scenarios.rs",
        "rlm_protocol_scenario_exec_any_tool_control_fail_is_terminal_error",
        result,
    )
}

fn rlm_natural_allows_finish_value_execution() -> Result<Value, FixedScriptRunnerError> {
    let result = run_rlm_protocol_contract(
        "rlm natural allows finish value",
        "return typed data",
        RlmTermination::Natural,
        None,
        None,
        vec![
            RlmContractStep::Llm(vec![rlm_text_part(&rlm_lashlang_block(
                "finish { ok: true }",
            ))]),
            RlmContractStep::Exec(rlm_exec_response(&[], None, Some(json!({ "ok": true })))),
            RlmContractStep::Checkpoint,
        ],
    )?;
    contract_execution_payload(
        "rlm.natural_allows_finish_value",
        "crates/lash-protocol-rlm/tests/protocol_drivers/scenarios.rs",
        "rlm_protocol_scenario_natural_allows_finish_value",
        result,
    )
}

fn rlm_typed_schema_mismatch_repair_loop_execution() -> Result<Value, FixedScriptRunnerError> {
    let result = run_rlm_protocol_contract(
        "rlm typed schema mismatch repair loop",
        "return typed data",
        RlmTermination::FinishRequired {
            schema: Some(json!({
                "type": "object",
                "properties": {
                    "ok": { "type": "boolean" }
                },
                "required": ["ok"]
            })),
        },
        None,
        None,
        vec![
            RlmContractStep::Llm(vec![rlm_text_part(&rlm_lashlang_block(
                "finish { missing: true }",
            ))]),
            RlmContractStep::Exec(rlm_exec_response(
                &[],
                None,
                Some(json!({ "missing": true })),
            )),
            RlmContractStep::Checkpoint,
        ],
    )?;
    contract_execution_payload(
        "rlm.typed_schema_mismatch_repair_loop",
        "crates/lash-protocol-rlm/tests/protocol_drivers/scenarios.rs",
        "rlm_protocol_scenario_typed_schema_mismatch_loops_with_feedback",
        result,
    )
}

fn rlm_typed_schema_any_of_mismatch_execution() -> Result<Value, FixedScriptRunnerError> {
    let result = run_rlm_protocol_contract(
        "rlm typed schema anyOf mismatch",
        "return typed data",
        RlmTermination::FinishRequired {
            schema: Some(json!({
                "anyOf": [
                    { "type": "string" },
                    { "type": "integer" }
                ]
            })),
        },
        None,
        None,
        vec![
            RlmContractStep::Llm(vec![rlm_text_part(&rlm_lashlang_block("finish true"))]),
            RlmContractStep::Exec(rlm_exec_response(&[], None, Some(json!(true)))),
        ],
    )?;
    contract_execution_payload(
        "rlm.typed_schema_any_of_mismatch",
        "crates/lash-protocol-rlm/tests/protocol_drivers/scenarios.rs",
        "rlm_protocol_scenario_typed_schema_mismatch_checks_any_of",
        result,
    )
}

#[derive(Clone)]
enum RlmContractStep {
    Llm(Vec<LlmOutputPart>),
    Exec(lash_core::ExecResponse),
    Checkpoint,
}

#[derive(Default)]
struct RlmContractObserved {
    initial_request_tools_empty: Option<bool>,
    exec_codes: Vec<String>,
    checkpoints: Vec<&'static str>,
    llm_response_full_texts: Vec<String>,
    llm_response_part_counts: Vec<usize>,
    llm_response_parts: Vec<Vec<Value>>,
    llm_call_count: usize,
    turn_outcomes: Vec<lash_core::TurnOutcome>,
    final_message_event: bool,
    tool_call_event: bool,
    assistant_conversation_progress: bool,
}

impl RlmContractObserved {
    fn record(&mut self, effects: &[lash_core::Effect]) {
        for effect in effects {
            match effect {
                lash_core::Effect::LlmCall { request, .. } => {
                    if self.initial_request_tools_empty.is_none() {
                        self.initial_request_tools_empty = Some(request.tools.is_empty());
                    }
                    self.llm_call_count += 1;
                }
                lash_core::Effect::ExecCode { code, .. } => {
                    self.exec_codes.push(code.clone());
                }
                lash_core::Effect::Checkpoint { checkpoint, .. } => {
                    self.checkpoints.push(checkpoint_kind_name(*checkpoint));
                }
                lash_core::Effect::Emit(lash_core::SessionEvent::TurnOutcome { outcome }) => {
                    self.turn_outcomes.push(outcome.clone());
                }
                lash_core::Effect::Emit(lash_core::SessionEvent::Message { kind, .. })
                    if kind == "final" =>
                {
                    self.final_message_event = true;
                }
                lash_core::Effect::Emit(lash_core::SessionEvent::ToolCall { .. }) => {
                    self.tool_call_event = true;
                }
                lash_core::Effect::Progress { event_delta, .. } => {
                    self.assistant_conversation_progress |= event_delta.iter().any(|event| {
                        matches!(
                            event,
                            lash_core::SessionEventRecord::Conversation(record)
                                if record.to_message().role == lash_core::MessageRole::Assistant
                        )
                    });
                }
                _ => {}
            }
        }
    }
}

fn run_rlm_protocol_contract(
    scenario_name: &'static str,
    user_message: &'static str,
    termination: RlmTermination,
    max_turns: Option<usize>,
    protocol_turn_options: Option<lash_core::ProtocolTurnOptions>,
    steps: Vec<RlmContractStep>,
) -> Result<Value, FixedScriptRunnerError> {
    let termination_declared = if protocol_turn_options.is_some() {
        json!({ "kind": "empty_protocol_turn_options" })
    } else {
        serde_json::to_value(&termination)?
    };
    let mut config = match protocol_turn_options {
        Some(options) => rlm_contract_config_with_turn_options(options),
        None => rlm_contract_config(termination),
    }?;
    config.max_turns = max_turns;
    let mut machine = lash_core::TurnMachine::new(
        config,
        vec![contract_user_message(user_message)],
        Arc::new(Vec::new()),
        0,
    );
    let mut observed = RlmContractObserved::default();
    let mut effects = drain_rlm_contract_effects(&mut machine);
    observed.record(&effects);
    for step in steps {
        match step {
            RlmContractStep::Llm(parts) => {
                let llm_id = *find_contract_llm_call(&effects).ok_or_else(|| {
                    FixedScriptRunnerError::Assertion(format!(
                        "{scenario_name} expected a pending LLM call"
                    ))
                })?;
                let expected_parts = parts.clone();
                let expected_full_text = rlm_full_text(&expected_parts);
                let expected_part_summary = llm_output_parts_contract_summary(&expected_parts);
                let response = llm_response_with_parts(expected_full_text.clone(), parts);
                require(
                    response.full_text == expected_full_text,
                    format!(
                        "{scenario_name} provider response full_text changed: expected {:?}, got {:?}",
                        expected_full_text, response.full_text
                    ),
                )?;
                require(
                    response.parts.len() == expected_parts.len() && !response.parts.is_empty(),
                    format!(
                        "{scenario_name} provider response parts changed: expected {} parts, got {}",
                        expected_parts.len(),
                        response.parts.len()
                    ),
                )?;
                let response_part_summary = llm_output_parts_contract_summary(&response.parts);
                require(
                    response_part_summary == expected_part_summary,
                    format!(
                        "{scenario_name} provider response parts changed: expected {:?}, got {:?}",
                        expected_part_summary, response_part_summary
                    ),
                )?;
                observed
                    .llm_response_full_texts
                    .push(response.full_text.clone());
                observed.llm_response_part_counts.push(response.parts.len());
                observed.llm_response_parts.push(response_part_summary);
                machine.handle_response(lash_core::sansio::Response::LlmComplete {
                    id: llm_id,
                    text_streamed: false,
                    result: Ok(response),
                });
            }
            RlmContractStep::Exec(result) => {
                let exec_id = effects
                    .iter()
                    .find_map(|effect| match effect {
                        lash_core::Effect::ExecCode { id, .. } => Some(*id),
                        _ => None,
                    })
                    .ok_or_else(|| {
                        FixedScriptRunnerError::Assertion(format!(
                            "{scenario_name} expected pending exec code"
                        ))
                    })?;
                machine.handle_response(lash_core::sansio::Response::ExecResult {
                    id: exec_id,
                    result: Ok(result),
                });
            }
            RlmContractStep::Checkpoint => {
                let checkpoint_id = effects
                    .iter()
                    .find_map(|effect| match effect {
                        lash_core::Effect::Checkpoint { id, .. } => Some(*id),
                        _ => None,
                    })
                    .ok_or_else(|| {
                        FixedScriptRunnerError::Assertion(format!(
                            "{scenario_name} expected pending checkpoint"
                        ))
                    })?;
                machine.handle_response(lash_core::sansio::Response::Checkpoint {
                    id: checkpoint_id,
                    delivery: lash_core::sansio::CheckpointDelivery::default(),
                });
            }
        }
        effects = drain_rlm_contract_effects(&mut machine);
        observed.record(&effects);
    }
    Ok(json!({
        "execution_api": "lash_core::sansio::TurnMachine",
        "driver": "lash_protocol_rlm::RlmDriver",
        "scenario_name": scenario_name,
        "user_message": user_message,
        "termination": termination_declared,
        "max_turns": max_turns,
        "initial_request_tools_empty": observed.initial_request_tools_empty,
        "llm_call_count": observed.llm_call_count,
        "llm_response_full_texts": observed.llm_response_full_texts,
        "llm_response_part_counts": observed.llm_response_part_counts,
        "llm_response_parts": observed.llm_response_parts,
        "done": machine.is_done(),
        "checkpoints": observed.checkpoints,
        "exec_codes": observed.exec_codes,
        "turn_outcomes": observed.turn_outcomes.iter().map(turn_outcome_contract_json).collect::<Vec<_>>(),
        "final_message_event": observed.final_message_event,
        "tool_call_event": observed.tool_call_event,
        "assistant_conversation_progress": observed.assistant_conversation_progress,
        "llm_extraction_diagnostics": rlm_contract_llm_extraction_diagnostics(&machine),
        "trajectory": rlm_contract_trajectory(&machine),
        "system_messages": rlm_contract_system_messages(&machine),
    }))
}

fn rlm_contract_config(
    termination: RlmTermination,
) -> Result<lash_core::TurnMachineConfig, FixedScriptRunnerError> {
    let options = lash_core::ProtocolTurnOptions::typed(RlmCreateExtras {
        termination,
        final_answer_format: None,
    })
    .map_err(|err| FixedScriptRunnerError::Assertion(err.to_string()))?;
    rlm_contract_config_with_turn_options(options)
}

fn rlm_contract_config_with_turn_options(
    termination: lash_core::ProtocolTurnOptions,
) -> Result<lash_core::TurnMachineConfig, FixedScriptRunnerError> {
    let protocol_driver: Arc<
        dyn lash_core::sansio::ProtocolDriverHandle<lash_core::HostTurnProtocol>,
    > = Arc::new(lash_protocol_rlm::RlmDriver);
    Ok(lash_core::TurnMachineConfig {
        protocol_driver,
        projector: Arc::new(lash_core::sansio::ChatContextProjector),
        sync_execution_environment: true,
        model: "rlm-contract".to_string(),
        max_context_tokens: None,
        max_turns: None,
        model_variant: None,
        generation: lash_core::GenerationOptions::default(),
        run_session_id: None,
        autonomous: false,
        tool_specs: Vec::new().into(),
        system_prompt: std::sync::Arc::from(""),
        session_id: "rlm-contract".to_string(),
        emit_llm_trace: false,
        termination,
        turn_limit_final_message: Arc::new(contract_turn_limit_final_message),
    })
}

fn drain_rlm_contract_effects(machine: &mut lash_core::TurnMachine) -> Vec<lash_core::Effect> {
    let mut effects = Vec::new();
    while let Some(effect) = machine.poll_effect() {
        if let lash_core::Effect::SyncExecutionEnvironment { id, .. } = effect {
            effects.push(effect);
            machine.handle_response(lash_core::sansio::Response::ExecutionEnvironmentSynced {
                id,
                result: Ok(Some(lash_core::sansio::ExecutionEnvironmentSync {
                    system_prompt: std::sync::Arc::from(""),
                    tool_specs: Arc::new(Vec::new()),
                })),
            });
            continue;
        }
        effects.push(effect);
    }
    effects
}

fn rlm_text_part(text: &str) -> LlmOutputPart {
    LlmOutputPart::Text {
        text: text.to_string(),
        response_meta: None,
    }
}

fn rlm_reasoning_part(text: &str) -> LlmOutputPart {
    LlmOutputPart::Reasoning {
        text: text.to_string(),
        replay: None,
    }
}

fn rlm_full_text(parts: &[LlmOutputPart]) -> String {
    parts
        .iter()
        .filter_map(|part| match part {
            LlmOutputPart::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

fn rlm_lashlang_block(code: &str) -> String {
    format!("<lashlang>\n{code}\n</lashlang>")
}

fn rlm_lashlang_block_with_prose(prose: &str, code: &str) -> String {
    format!("{prose}\n{}", rlm_lashlang_block(code))
}

fn rlm_exec_response(
    output: &[&str],
    error: Option<&str>,
    terminal_finish: Option<Value>,
) -> lash_core::ExecResponse {
    lash_core::ExecResponse {
        observations: output.iter().map(|value| (*value).to_string()).collect(),
        observation_truncation: Vec::new(),
        tool_calls: Vec::new(),
        images: Vec::new(),
        printed_images: Vec::new(),
        error: error.map(str::to_string),
        duration_ms: 1,
        terminal_finish,
    }
}

fn rlm_exec_response_with_tool_calls(
    output: &[&str],
    error: Option<&str>,
    terminal_finish: Option<Value>,
    tool_calls: Vec<lash_core::ToolCallRecord>,
    duration_ms: u64,
) -> lash_core::ExecResponse {
    lash_core::ExecResponse {
        observations: output.iter().map(|value| (*value).to_string()).collect(),
        observation_truncation: Vec::new(),
        tool_calls,
        images: Vec::new(),
        printed_images: Vec::new(),
        error: error.map(str::to_string),
        duration_ms,
        terminal_finish,
    }
}

fn rlm_tool_call_record(
    call_id: &str,
    tool: &str,
    args: Value,
    output: lash_core::ToolCallOutput,
    duration_ms: u64,
) -> lash_core::ToolCallRecord {
    lash_core::ToolCallRecord {
        call_id: Some(call_id.to_string()),
        tool: tool.to_string(),
        args,
        output,
        duration_ms,
    }
}

fn checkpoint_kind_name(checkpoint: lash_core::CheckpointKind) -> &'static str {
    match checkpoint {
        lash_core::CheckpointKind::BeforeCompletion => "before_completion",
        lash_core::CheckpointKind::AfterWork => "after_work",
    }
}

fn rlm_contract_llm_extraction_diagnostics(machine: &lash_core::TurnMachine) -> Vec<Value> {
    machine
        .events()
        .iter()
        .filter_map(|event| match event {
            lash_core::SessionEventRecord::Protocol(event) => {
                match lash_protocol_rlm::decode_rlm_protocol_event(event) {
                    Some(RlmProtocolEvent::RlmDiagnostic(diagnostic))
                        if diagnostic.phase == "llm_extraction" =>
                    {
                        Some(diagnostic.payload)
                    }
                    _ => None,
                }
            }
            _ => None,
        })
        .collect()
}

fn rlm_contract_trajectory(machine: &lash_core::TurnMachine) -> Vec<Value> {
    machine
        .events()
        .iter()
        .filter_map(|event| match event {
            lash_core::SessionEventRecord::Protocol(event) => {
                match lash_protocol_rlm::decode_rlm_protocol_event(event) {
                    Some(RlmProtocolEvent::RlmTrajectoryEntry(entry)) => {
                        serde_json::to_value(entry).ok()
                    }
                    _ => None,
                }
            }
            _ => None,
        })
        .collect()
}

fn rlm_contract_system_messages(machine: &lash_core::TurnMachine) -> Vec<String> {
    machine
        .messages()
        .iter()
        .filter(|message| message.role == lash_core::MessageRole::System)
        .flat_map(|message| message.parts.iter().map(|part| part.content.clone()))
        .collect()
}

async fn agent_tuple_json_array_execution() -> Result<Value, FixedScriptRunnerError> {
    let expected = json!({
        "first": "left",
        "tail": ["right"],
        "seen": ["left", "right"],
        "tuple": ["left", "right"],
        "nested": { "pair": ["left", "right"] }
    });
    let result = facade_final_value_execution(
        "lash_runtime agent tuple final value",
        "sim-agent-tuple-json-array-contract",
        "Use tuple values and finish the derived result.",
        r#"<lashlang>
pair = "left", "right"
tail = slice(pair, 1, null)
seen = []
for item in pair {
  seen = push(seen, item)
}
finish {
  first: pair[0],
  tail: tail,
  seen: seen,
  tuple: pair,
  nested: { pair: pair }
}
</lashlang>"#,
        &expected,
    )
    .await?;
    contract_execution_payload(
        "agent.tuple_values_finish_as_json_arrays",
        "crates/lash/src/tests/agent_scenarios/cases.rs",
        "agent_scenario_tuple_values_finish_as_json_arrays",
        result,
    )
}

async fn agent_contract_executions() -> Result<Vec<Value>, FixedScriptRunnerError> {
    // Aggregating every fixed Agent execution is simulation-harness work used by
    // generated proof/minimizer packages. It may use the bounded harness stack;
    // individual product facade executions are separately probed at 2 MiB.
    run_on_sim_harness_stack(
        "agent-contract-executions-aggregate",
        SIM_HARNESS_STACK_LIMIT_BYTES,
        || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(FixedScriptRunnerError::Io)?;
            let mut executions = Vec::new();
            for contract in FIXED_AGENT_PRODUCT_CONTRACTS {
                let runner = agent_contract_runner(contract)?;
                executions.push(runner(&runtime)?);
            }
            Ok(executions)
        },
    )
}

pub const FIXED_AGENT_PRODUCT_CONTRACTS: &[&str] = &[
    "agent.foreground_tool_call_round_trip",
    "agent.started_process_tool_call_graph",
    "agent.durable_input_suspension_resolution",
    "agent.shell_results_are_data",
    "agent.shell_output_print_projection_survives",
    "agent.started_process_subagent_spawn",
    "agent.nested_process_start_await",
    "agent.session_turn_process_child",
    "agent.failed_child_preserves_failure_graph",
    "agent.parallel_spawn_and_join",
    "agent.tuple_values_finish_as_json_arrays",
];

pub fn run_agent_contract_product_stack_probe(
    contract: &str,
    stack_bytes: usize,
) -> Result<(), FixedScriptRunnerError> {
    let contract = contract.to_string();
    let runner = agent_contract_runner(&contract)?;
    run_on_product_stack(
        format!("product-agent-contract-probe-{contract}"),
        stack_bytes,
        move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(FixedScriptRunnerError::Io)?;
            runner(&runtime).map(|_| ())
        },
    )
}

type AgentContractRunner = fn(&tokio::runtime::Runtime) -> Result<Value, FixedScriptRunnerError>;

fn agent_contract_runner(contract: &str) -> Result<AgentContractRunner, FixedScriptRunnerError> {
    match contract {
        "agent.foreground_tool_call_round_trip" => Ok(run_agent_foreground_tool_call_round_trip),
        "agent.started_process_tool_call_graph" => Ok(run_agent_started_process_tool_call_graph),
        "agent.durable_input_suspension_resolution" => {
            Ok(run_agent_durable_input_suspension_resolution)
        }
        "agent.shell_results_are_data" => Ok(run_agent_shell_results_are_data),
        "agent.shell_output_print_projection_survives" => {
            Ok(run_agent_shell_output_print_projection_survives)
        }
        "agent.started_process_subagent_spawn" => Ok(run_agent_started_process_subagent_spawn),
        "agent.nested_process_start_await" => Ok(run_agent_nested_process_start_await),
        "agent.session_turn_process_child" => Ok(run_agent_session_turn_process_child),
        "agent.failed_child_preserves_failure_graph" => {
            Ok(run_agent_failed_child_preserves_failure_graph)
        }
        "agent.parallel_spawn_and_join" => Ok(run_agent_parallel_spawn_and_join),
        "agent.tuple_values_finish_as_json_arrays" => Ok(run_agent_tuple_json_array),
        other => Err(FixedScriptRunnerError::Assertion(format!(
            "no replayable fixed Agent contract execution registered for `{other}`"
        ))),
    }
}

fn run_agent_foreground_tool_call_round_trip(
    runtime: &tokio::runtime::Runtime,
) -> Result<Value, FixedScriptRunnerError> {
    runtime.block_on(agent_foreground_tool_call_round_trip_execution())
}

fn run_agent_started_process_tool_call_graph(
    runtime: &tokio::runtime::Runtime,
) -> Result<Value, FixedScriptRunnerError> {
    runtime.block_on(agent_started_process_tool_call_graph_execution())
}

fn run_agent_durable_input_suspension_resolution(
    runtime: &tokio::runtime::Runtime,
) -> Result<Value, FixedScriptRunnerError> {
    runtime.block_on(agent_durable_input_suspension_resolution_execution())
}

fn run_agent_shell_results_are_data(
    runtime: &tokio::runtime::Runtime,
) -> Result<Value, FixedScriptRunnerError> {
    runtime.block_on(agent_shell_results_are_data_execution())
}

fn run_agent_shell_output_print_projection_survives(
    runtime: &tokio::runtime::Runtime,
) -> Result<Value, FixedScriptRunnerError> {
    runtime.block_on(agent_shell_output_print_projection_survives_execution())
}

fn run_agent_started_process_subagent_spawn(
    runtime: &tokio::runtime::Runtime,
) -> Result<Value, FixedScriptRunnerError> {
    runtime.block_on(agent_started_process_subagent_spawn_execution())
}

fn run_agent_nested_process_start_await(
    runtime: &tokio::runtime::Runtime,
) -> Result<Value, FixedScriptRunnerError> {
    runtime.block_on(agent_nested_process_start_await_execution())
}

fn run_agent_session_turn_process_child(
    runtime: &tokio::runtime::Runtime,
) -> Result<Value, FixedScriptRunnerError> {
    runtime.block_on(agent_session_turn_process_child_execution())
}

fn run_agent_failed_child_preserves_failure_graph(
    runtime: &tokio::runtime::Runtime,
) -> Result<Value, FixedScriptRunnerError> {
    runtime.block_on(agent_failed_child_preserves_failure_graph_execution())
}

fn run_agent_parallel_spawn_and_join(
    runtime: &tokio::runtime::Runtime,
) -> Result<Value, FixedScriptRunnerError> {
    runtime.block_on(agent_parallel_spawn_and_join_execution())
}

fn run_agent_tuple_json_array(
    runtime: &tokio::runtime::Runtime,
) -> Result<Value, FixedScriptRunnerError> {
    runtime.block_on(agent_tuple_json_array_execution())
}

async fn agent_foreground_tool_call_round_trip_execution() -> Result<Value, FixedScriptRunnerError>
{
    let expected = json!({ "ok": true });
    let result = facade_final_value_execution_with_tools(
        "lash_runtime agent foreground tool",
        "sim-agent-foreground-tool-contract",
        "Call the app lookup tool and finish its value.",
        vec![
            r#"<lashlang>
@label(title: "Lookup app state")
value = await tools.app_lookup({})?
finish value
</lashlang>"#,
        ],
        &expected,
        Some(Arc::new(ContractAppTools) as Arc<dyn lash_core::ToolProvider>),
    )
    .await?;
    require(
        result.get("tool_completed_count").and_then(Value::as_u64) == Some(1)
            && result
                .get("tool_completed_outputs")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .any(|entry| {
                    entry.get("name").and_then(Value::as_str) == Some("app_lookup")
                        && entry.get("value") == Some(&expected)
                }),
        "agent foreground tool execution did not record a concrete app_lookup completion",
    )?;
    contract_execution_payload(
        "agent.foreground_tool_call_round_trip",
        "crates/lash/src/tests/agent_scenarios/cases.rs",
        "agent_scenario_foreground_labeled_tool_call",
        result,
    )
}

async fn agent_started_process_tool_call_graph_execution() -> Result<Value, FixedScriptRunnerError>
{
    let expected = json!({ "ok": true });
    let result = facade_agent_process_execution(
        "lash_runtime agent started process tool",
        "sim-agent-started-process-tool-contract",
        "Start a process that calls the app lookup tool.",
        vec![
            r#"<lashlang>
process lookup(tools: Tools) {
  @label(title: "Lookup app state in process")
  value = await tools.app_lookup({})?
  finish value
}
handle = start lookup(tools: tools)
result = (await handle)?
finish result
</lashlang>"#,
        ],
        &expected,
        Some(Arc::new(ContractAppTools) as Arc<dyn lash_core::ToolProvider>),
    )
    .await?;
    contract_execution_payload(
        "agent.started_process_tool_call_graph",
        "crates/lash/src/tests/agent_scenarios/cases.rs",
        "agent_scenario_started_process_labeled_tool_call",
        result,
    )
}

async fn agent_durable_input_suspension_resolution_execution()
-> Result<Value, FixedScriptRunnerError> {
    let result = facade_agent_durable_input_execution().await?;
    contract_execution_payload(
        "agent.durable_input_suspension_resolution",
        "crates/lash/src/tests/agent_scenarios/cases.rs",
        "agent_scenario_process_durable_input_request_tool",
        result,
    )
}

async fn agent_shell_results_are_data_execution() -> Result<Value, FixedScriptRunnerError> {
    let expected = json!({
        "pipe_exit": 0,
        "pipe_output": "line\nline\nline\n",
        "missing_exit": 1,
        "missing_status": "completed"
    });
    let result = facade_final_value_execution_with_tools(
        "lash_runtime agent shell results data",
        "sim-agent-shell-results-data-contract",
        "Run shell commands and report their result metadata.",
        vec![
            r#"<lashlang>
pipe = await shell.exec({ cmd: "yes line | head -n 3", login: false })?
missing = await shell.exec({ cmd: "test -f /tmp/agent-scenario-definitely-missing-file", login: false })?
finish {
  pipe_exit: pipe.exit_code,
  pipe_output: pipe.output,
  missing_exit: missing.exit_code,
  missing_status: missing.status
}
</lashlang>"#,
        ],
        &expected,
        Some(Arc::new(lash_tools::shell::shell_provider(
            lash_tools::shell::StandardShell::new(),
        )) as Arc<dyn lash_core::ToolProvider>),
    )
    .await?;
    contract_execution_payload(
        "agent.shell_results_are_data",
        "crates/lash/src/tests/agent_scenarios/cases.rs",
        "agent_scenario_shell_nonzero_and_pipeline_results_are_data",
        result,
    )
}

async fn agent_nested_process_start_await_execution() -> Result<Value, FixedScriptRunnerError> {
    let expected = json!({ "parent": "done" });
    let result = facade_agent_process_execution(
        "lash_runtime agent nested process",
        "sim-agent-nested-process-contract",
        "Start a parent process that starts and awaits a child process.",
        vec![
            r#"<lashlang>
process child() {
  finish { child: "done" }
}
process parent() {
  @label(title: "Start nested child process")
  handle = start child()
  result = (await handle)?
  finish { parent: result.child }
}
handle = start parent()
result = (await handle)?
finish result
</lashlang>"#,
        ],
        &expected,
        None,
    )
    .await?;
    contract_execution_payload(
        "agent.nested_process_start_await",
        "crates/lash/src/tests/agent_scenarios/cases.rs",
        "agent_scenario_nested_process_start_await",
        result,
    )
}

async fn agent_shell_output_print_projection_survives_execution()
-> Result<Value, FixedScriptRunnerError> {
    let expected = json!({
        "chars": 60000,
        "tail": "x\nx\n",
        "has_full_output_path": true
    });
    let result = facade_final_value_execution_with_tools(
        "lash_runtime agent shell output projection",
        "sim-agent-shell-output-projection-contract",
        "Run a large shell command, inspect it, then report retained metadata.",
        vec![
            r#"<lashlang>
big = await shell.exec({ cmd: "yes x | head -c 60000", login: false })?
print big.output
</lashlang>"#,
            r#"<lashlang>
finish {
  chars: len(big.output),
  tail: slice(big.output, 59996, null),
  has_full_output_path: big.full_output_path == null ? false : len(big.full_output_path) > 0
}
</lashlang>"#,
        ],
        &expected,
        Some(Arc::new(lash_tools::shell::shell_provider(
            lash_tools::shell::StandardShell::new(),
        )) as Arc<dyn lash_core::ToolProvider>),
    )
    .await?;
    contract_execution_payload(
        "agent.shell_output_print_projection_survives",
        "crates/lash/src/tests/agent_scenarios/cases.rs",
        "agent_scenario_shell_output_survives_print_projection_in_variable",
        result,
    )
}

async fn agent_started_process_subagent_spawn_execution() -> Result<Value, FixedScriptRunnerError> {
    let expected = json!({ "len": 2 });
    let result = facade_agent_process_execution_with_options(
        "lash_runtime agent started process subagent",
        "sim-agent-started-process-subagent-contract",
        "Run a Lashlang process that spawns a subagent and returns its value.",
        vec![
            r#"<lashlang>
process spawn_child() {
  @label(title: "Spawn subagent with web search")
  result = await agents.spawn({
    capability: "default",
    task: "Finish `{ len: len(chunk) }` using the seeded `chunk` variable.",
    seed: { chunk: ["a", "b"] },
    output: Type { len: int }
  })?
  finish result
}
handle = start spawn_child()
result = (await handle)?
finish result
</lashlang>"#,
            r#"<lashlang>
finish { len: len(chunk) }
</lashlang>"#,
        ],
        &expected,
        None,
        true,
        None,
    )
    .await?;
    contract_execution_payload(
        "agent.started_process_subagent_spawn",
        "crates/lash/src/tests/agent_scenarios/cases.rs",
        "agent_scenario_started_process_labeled_subagent_spawn",
        result,
    )
}

async fn agent_session_turn_process_child_execution() -> Result<Value, FixedScriptRunnerError> {
    let expected = json!({ "child": "done" });
    let result = facade_final_value_execution(
        "lash_runtime agent session-turn process child",
        "sim-agent-session-turn-process-child-contract",
        "Start a child process and await its result.",
        r#"<lashlang>
process child() {
  finish { child: "done" }
}
handle = start child()
result = (await handle)?
finish result
</lashlang>"#,
        &expected,
    )
    .await?;
    contract_execution_payload(
        "agent.session_turn_process_child",
        "crates/lash/src/tests/agent_scenarios/cases.rs",
        "agent_scenario_session_turn_process_child",
        result,
    )
}

async fn agent_failed_child_preserves_failure_graph_execution()
-> Result<Value, FixedScriptRunnerError> {
    let (core, graph_store) = agent_process_contract_core_with_options(
        "lash_runtime agent failed child graph",
        vec![
            r#"<lashlang>
@label(title: "Spawn failing subagent")
result = await agents.spawn({
  capability: "default",
  task: "Fail with reason child boom.",
  seed: {},
  output: Type { reason: str }
})?
finish result
</lashlang>"#,
            r#"<lashlang>
await task.fail({ reason: "child boom" })?
</lashlang>"#,
            r#"<lashlang>
await task.fail({ reason: "parent observed child failure" })?
</lashlang>"#,
        ],
        None,
        true,
        Some(1),
    )?;
    let session = core
        .session("sim-agent-failed-child-contract")
        .open_fresh()
        .await
        .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
    let events = Arc::new(RuntimeProofRecordingEvents::default());
    let result = session
        .turn(lash::TurnInput::text(
            "Spawn a child that fails and preserve its execution graph.",
        ))
        .stream_to(events.as_ref())
        .await
        .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
    session
        .processes()
        .await_all()
        .await
        .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
    let recorded = events.snapshot().await;
    let process_observations = agent_contract_process_observations(&core).await?;
    let process_facts = agent_contract_process_facts(&process_observations);
    let graph_facts = agent_contract_graph_facts(&graph_store.graphs(), &result.state.session_id);
    let failure = agent_failed_child_activity_facts(&result, &recorded);
    let payload = json!({
        "execution_api": "lash::RlmCore facade",
        "provider_kind": "lash_runtime agent failed child graph",
        "session_id": result.state.session_id,
        "turn_index": result.state.turn_index,
        "done": true,
        "turn_outcome": turn_outcome_contract_json(&result.outcome),
        "final_value": Value::Null,
        "processes": process_observations
            .iter()
            .map(|process| process.observed.clone())
            .collect::<Vec<_>>(),
        "process_facts": process_facts,
        "graph_facts": graph_facts,
        "failure": failure,
    });
    contract_execution_payload(
        "agent.failed_child_preserves_failure_graph",
        "crates/lash/src/tests/agent_scenarios/cases.rs",
        "agent_scenario_failed_child_preserves_failure_graph",
        payload,
    )
}

async fn agent_parallel_spawn_and_join_execution() -> Result<Value, FixedScriptRunnerError> {
    let expected = json!({ "joined": ["left", "right"] });
    let result = facade_final_value_execution(
        "lash_runtime agent parallel process join",
        "sim-agent-parallel-spawn-join-contract",
        "Start two processes, await both, and finish their joined result.",
        r#"<lashlang>
process child(value: str) {
  finish value
}
@label(title: "Start left process")
left = start child(value: "left")
@label(title: "Start right process")
right = start child(value: "right")
left_value = (await left)?
right_value = (await right)?
finish { joined: [left_value, right_value] }
</lashlang>"#,
        &expected,
    )
    .await?;
    contract_execution_payload(
        "agent.parallel_spawn_and_join",
        "crates/lash/src/tests/agent_scenarios/cases.rs",
        "agent_scenario_parallel_spawn_and_join",
        result,
    )
}

async fn facade_final_value_execution(
    provider_kind: &'static str,
    session_id: &'static str,
    prompt: &'static str,
    provider_response: &'static str,
    expected_final_value: &Value,
) -> Result<Value, FixedScriptRunnerError> {
    facade_final_value_execution_with_tools(
        provider_kind,
        session_id,
        prompt,
        vec![provider_response],
        expected_final_value,
        None,
    )
    .await
}

async fn facade_final_value_execution_with_tools(
    provider_kind: &'static str,
    session_id: &'static str,
    prompt: &'static str,
    provider_responses: Vec<&'static str>,
    expected_final_value: &Value,
    tools: Option<Arc<dyn lash_core::ToolProvider>>,
) -> Result<Value, FixedScriptRunnerError> {
    facade_final_value_execution_inner(
        provider_kind,
        session_id,
        prompt,
        provider_responses,
        expected_final_value.clone(),
        tools,
    )
    .await
}

async fn facade_final_value_execution_inner(
    provider_kind: &'static str,
    session_id: &'static str,
    prompt: &'static str,
    provider_responses: Vec<&'static str>,
    expected_final_value: Value,
    tools: Option<Arc<dyn lash_core::ToolProvider>>,
) -> Result<Value, FixedScriptRunnerError> {
    let events = Arc::new(RuntimeProofRecordingEvents::default());
    let mut builder = lash::RlmCore::builder()
        .effect_host(Arc::new(lash::durability::InlineEffectHost::default()))
        .lashlang_artifact_store(Arc::new(
            lash::persistence::InMemoryLashlangArtifactStore::new(),
        ))
        .attachment_store(Arc::new(lash::persistence::InMemoryAttachmentStore::new()))
        .process_env_store(Arc::new(
            lash::persistence::InMemoryProcessExecutionEnvStore::new(),
        ))
        .store_factory(Arc::new(
            lash::persistence::InMemorySessionStoreFactory::new(),
        ))
        .process_registry(Arc::new(lash_core::TestLocalProcessRegistry::default())
            as Arc<dyn lash_core::ProcessRegistry>)
        .provider(fixed_texts_provider(provider_kind, provider_responses))
        .model(
            lash_core::ModelSpec::from_token_limits(provider_kind, None, 200_000, None)
                .map_err(FixedScriptRunnerError::Assertion)?,
        );
    if let Some(tools) = tools {
        builder = builder.tools(tools);
    }
    let core = builder
        .build()
        .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
    let session = core
        .session(session_id)
        .open_fresh()
        .await
        .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
    let result = session
        .turn(lash::TurnInput::text(prompt))
        .require_finish()
        .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?
        .stream_to(events.as_ref())
        .await
        .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
    let final_value = result.final_value().cloned().ok_or_else(|| {
        FixedScriptRunnerError::Assertion(format!(
            "{provider_kind} finished without TurnFinish::FinalValue: {:?}",
            result.outcome
        ))
    })?;
    require(
        final_value == expected_final_value,
        "facade final value execution produced an unexpected semantic value",
    )?;
    let recorded = events.snapshot().await;
    let final_value_events = events.final_value_events().await;
    let assistant_prose_delta_count = events.assistant_prose_delta_count().await;
    let tool_completed_count = events.tool_completed_count().await;
    let tool_completed_outputs = events
        .tool_completed_outputs()
        .await
        .into_iter()
        .map(
            |(name, value)| json!({ "name": name, "value": normalize_contract_tool_output(value) }),
        )
        .collect::<Vec<_>>();
    let facts = runtime_final_value_invariant_facts(&result, &recorded);
    require(
        facts.passed
            && facts.outcome_kind == "final_value"
            && facts.semantic_value.as_ref() == Some(&final_value)
            && final_value_events.iter().any(|value| value == &final_value)
            && result.assistant_message().is_none(),
        "facade final value execution did not produce concrete final-value outcome/event facts",
    )?;
    Ok(json!({
        "execution_api": "lash::RlmCore facade",
        "provider_kind": provider_kind,
        "session_id": result.state.session_id,
        "turn_index": result.state.turn_index,
        "done": true,
        "turn_outcome": {
            "kind": "final_value",
        },
        "final_value": final_value,
        "no_final_message_event": result.assistant_message().is_none(),
        "runtime_final_value_facts": facts,
        "final_value_event_count": final_value_events.len(),
        "assistant_prose_delta_count": assistant_prose_delta_count,
        "tool_completed_count": tool_completed_count,
        "tool_completed_outputs": tool_completed_outputs,
    }))
}

async fn facade_agent_process_execution(
    provider_kind: &'static str,
    session_id: &'static str,
    prompt: &'static str,
    provider_responses: Vec<&'static str>,
    expected_final_value: &Value,
    tools: Option<Arc<dyn lash_core::ToolProvider>>,
) -> Result<Value, FixedScriptRunnerError> {
    facade_agent_process_execution_with_options(
        provider_kind,
        session_id,
        prompt,
        provider_responses,
        expected_final_value,
        tools,
        false,
        None,
    )
    .await
}

async fn facade_agent_process_execution_with_options(
    provider_kind: &'static str,
    session_id: &'static str,
    prompt: &'static str,
    provider_responses: Vec<&'static str>,
    expected_final_value: &Value,
    tools: Option<Arc<dyn lash_core::ToolProvider>>,
    install_subagents: bool,
    max_turns: Option<usize>,
) -> Result<Value, FixedScriptRunnerError> {
    let (core, graph_store) = agent_process_contract_core_with_options(
        provider_kind,
        provider_responses,
        tools,
        install_subagents,
        max_turns,
    )?;
    let session = core
        .session(session_id)
        .open_fresh()
        .await
        .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
    let events = Arc::new(RuntimeProofRecordingEvents::default());
    let result = session
        .turn(lash::TurnInput::text(prompt))
        .stream_to(events.as_ref())
        .await
        .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
    session
        .processes()
        .await_all()
        .await
        .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
    agent_process_execution_result(
        &core,
        &graph_store,
        result,
        events,
        provider_kind,
        expected_final_value,
        None,
        false,
    )
    .await
}

async fn facade_agent_durable_input_execution() -> Result<Value, FixedScriptRunnerError> {
    let (key_tx, mut key_rx) =
        tokio::sync::oneshot::channel::<Result<lash_core::AwaitEventKey, String>>();
    let tools = Arc::new(ContractDurableInputTools::new(key_tx));
    let (core, graph_store) = agent_process_contract_core(
        "lash_runtime agent durable input",
        vec![
            r#"<lashlang>
process request_answer(tools: Tools) {
  result = await tools.mock_input_request({ question: "Need input?" })?
  finish result
}
handle = start request_answer(tools: tools)
result = (await handle)?
finish result.answer
</lashlang>"#,
            r#"<lashlang>
finish { recovered: true }
</lashlang>"#,
        ],
        Some(Arc::clone(&tools) as Arc<dyn lash_core::ToolProvider>),
    )?;
    let session = core
        .session("sim-agent-durable-input-contract")
        .open_fresh()
        .await
        .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
    let events = Arc::new(RuntimeProofRecordingEvents::default());
    let turn_session = session.clone();
    let turn_events = Arc::clone(&events);
    let turn = tokio::spawn(async move {
        turn_session
            .turn(lash::TurnInput::text(
                "Start a process that asks for durable input.",
            ))
            .stream_to(turn_events.as_ref())
            .await
            .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))
    });
    let key = wait_for_contract_durable_input_key(&mut key_rx, &turn).await?;
    let completed_before_resolution = events.tool_completed_count().await;
    let suspended_before_resolution = !turn.is_finished() && completed_before_resolution == 0;
    let await_custom_key = match &key.wait {
        lash_core::AwaitEventWaitIdentity::Custom { key } => key.clone(),
        other => {
            return Err(FixedScriptRunnerError::Assertion(format!(
                "durable input used non-custom await key `{other:?}`"
            )));
        }
    };
    let resolve_outcome = core
        .completions()
        .resolve(
            key,
            lash_core::Resolution::Ok(json!({ "answer": "approved" })),
        )
        .await
        .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
    let result = turn.await.map_err(|err| {
        FixedScriptRunnerError::Runtime(format!("durable input turn task failed to join: {err}"))
    })??;
    session
        .processes()
        .await_all()
        .await
        .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
    let completed_after_resolution = events.tool_completed_count().await;
    let durable_input = json!({
        "await_custom_key": await_custom_key,
        "suspended_before_resolution": suspended_before_resolution,
        "completed_event_count_before_resolution": completed_before_resolution,
        "completed_event_count_after_resolution": completed_after_resolution,
        "resolve_accepted": matches!(resolve_outcome, lash_core::ResolveOutcome::Accepted),
        "durable_step_count": tools.step_count(),
    });
    agent_process_execution_result(
        &core,
        &graph_store,
        result,
        events,
        "lash_runtime agent durable input",
        &json!("approved"),
        Some(("durable_input", durable_input)),
        true,
    )
    .await
}

fn agent_process_contract_core(
    provider_kind: &'static str,
    provider_responses: Vec<&'static str>,
    tools: Option<Arc<dyn lash_core::ToolProvider>>,
) -> Result<(lash::RlmCore, Arc<lash::tracing::TraceLashlangGraphStore>), FixedScriptRunnerError> {
    agent_process_contract_core_with_options(provider_kind, provider_responses, tools, false, None)
}

fn agent_process_contract_core_with_options(
    provider_kind: &'static str,
    provider_responses: Vec<&'static str>,
    tools: Option<Arc<dyn lash_core::ToolProvider>>,
    install_subagents: bool,
    max_turns: Option<usize>,
) -> Result<(lash::RlmCore, Arc<lash::tracing::TraceLashlangGraphStore>), FixedScriptRunnerError> {
    let graph_store = Arc::new(lash::tracing::TraceLashlangGraphStore::default());
    let mut builder = lash::RlmCore::builder()
        .effect_host(Arc::new(lash::durability::InlineEffectHost::default()))
        .lashlang_artifact_store(Arc::new(
            lash::persistence::InMemoryLashlangArtifactStore::new(),
        ))
        .attachment_store(Arc::new(lash::persistence::InMemoryAttachmentStore::new()))
        .process_env_store(Arc::new(
            lash::persistence::InMemoryProcessExecutionEnvStore::new(),
        ))
        .store_factory(Arc::new(
            lash::persistence::InMemorySessionStoreFactory::new(),
        ))
        .process_registry(Arc::new(lash_core::TestLocalProcessRegistry::default())
            as Arc<dyn lash_core::ProcessRegistry>)
        .lashlang_execution_sink(Arc::clone(&graph_store) as Arc<dyn lash::tracing::TraceSink>)
        .provider(fixed_texts_provider(provider_kind, provider_responses))
        .model(
            lash_core::ModelSpec::from_token_limits(provider_kind, None, 200_000, None)
                .map_err(FixedScriptRunnerError::Assertion)?,
        );
    if let Some(tools) = tools {
        builder = builder.tools(tools);
    }
    if install_subagents {
        builder = builder.plugin(agent_contract_subagents_plugin());
    }
    if let Some(max_turns) = max_turns {
        builder = builder.max_turns(max_turns);
    }
    let core = builder
        .build()
        .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
    Ok((core, graph_store))
}

fn agent_contract_subagents_plugin() -> Arc<dyn lash_core::PluginFactory> {
    Arc::new(lash_subagents::SubagentsPluginFactory::new(Arc::new(
        lash_subagents::CapabilityRegistry::new().with(Arc::new(
            lash_subagents::StaticCapability::new("default", lash_core::SessionSpec::inherit()),
        )),
    )))
}

async fn wait_for_contract_durable_input_key(
    key_rx: &mut tokio::sync::oneshot::Receiver<Result<lash_core::AwaitEventKey, String>>,
    turn: &tokio::task::JoinHandle<Result<lash::TurnResult, FixedScriptRunnerError>>,
) -> Result<lash_core::AwaitEventKey, FixedScriptRunnerError> {
    for _ in 0..MAX_PROVIDER_EVENT_POLL_YIELDS {
        match key_rx.try_recv() {
            Ok(Ok(key)) => return Ok(key),
            Ok(Err(err)) => return Err(FixedScriptRunnerError::Runtime(err)),
            Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                if turn.is_finished() {
                    return Err(FixedScriptRunnerError::Assertion(
                        "durable input turn completed before publishing await key".to_string(),
                    ));
                }
                tokio::task::yield_now().await;
            }
            Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                return Err(FixedScriptRunnerError::Assertion(
                    "durable input tool dropped await-key sender".to_string(),
                ));
            }
        }
    }
    Err(FixedScriptRunnerError::Assertion(
        "durable input tool did not publish await key within bounded scheduler yields".to_string(),
    ))
}

async fn agent_process_execution_result(
    core: &lash::RlmCore,
    graph_store: &lash::tracing::TraceLashlangGraphStore,
    result: lash::TurnResult,
    events: Arc<RuntimeProofRecordingEvents>,
    provider_kind: &'static str,
    expected_final_value: &Value,
    extra: Option<(&'static str, Value)>,
    include_process_events: bool,
) -> Result<Value, FixedScriptRunnerError> {
    let final_value = result.final_value().cloned().ok_or_else(|| {
        FixedScriptRunnerError::Assertion(format!(
            "{provider_kind} finished without TurnFinish::FinalValue: {:?}",
            result.outcome
        ))
    })?;
    require(
        final_value == *expected_final_value,
        "agent process execution produced an unexpected semantic value",
    )?;
    let recorded = events.snapshot().await;
    let final_value_events = events.final_value_events().await;
    let assistant_prose_delta_count = events.assistant_prose_delta_count().await;
    let tool_completed_count = events.tool_completed_count().await;
    let tool_completed_outputs = events
        .tool_completed_outputs()
        .await
        .into_iter()
        .map(
            |(name, value)| json!({ "name": name, "value": normalize_contract_tool_output(value) }),
        )
        .collect::<Vec<_>>();
    let facts = runtime_final_value_invariant_facts(&result, &recorded);
    require(
        facts.passed
            && facts.outcome_kind == "final_value"
            && facts.semantic_value.as_ref() == Some(&final_value)
            && final_value_events.iter().any(|value| value == &final_value)
            && result.assistant_message().is_none(),
        "agent process execution did not produce concrete final-value outcome/event facts",
    )?;
    let process_observations = agent_contract_process_observations(core).await?;
    let process_facts = agent_contract_process_facts(&process_observations);
    let process_events = if include_process_events {
        agent_contract_process_event_facts(core, &process_observations).await?
    } else {
        Vec::new()
    };
    let graph_facts = agent_contract_graph_facts(&graph_store.graphs(), &result.state.session_id);
    let mut payload = json!({
        "execution_api": "lash::RlmCore facade",
        "provider_kind": provider_kind,
        "session_id": result.state.session_id,
        "turn_index": result.state.turn_index,
        "done": true,
        "turn_outcome": {
            "kind": "final_value",
        },
        "final_value": final_value,
        "no_final_message_event": result.assistant_message().is_none(),
        "runtime_final_value_facts": facts,
        "final_value_event_count": final_value_events.len(),
        "assistant_prose_delta_count": assistant_prose_delta_count,
        "tool_completed_count": tool_completed_count,
        "tool_completed_outputs": tool_completed_outputs,
        "processes": process_observations
            .iter()
            .map(|process| process.observed.clone())
            .collect::<Vec<_>>(),
        "process_facts": process_facts,
        "process_events": process_events,
        "graph_facts": graph_facts,
    });
    if let Some((key, value)) = extra
        && let Some(object) = payload.as_object_mut()
    {
        object.insert(key.to_string(), value);
    }
    Ok(payload)
}

struct AgentContractProcessObservation {
    raw_process_id: String,
    process_ref: String,
    observed: Value,
}

async fn agent_contract_process_observations(
    core: &lash::RlmCore,
) -> Result<Vec<AgentContractProcessObservation>, FixedScriptRunnerError> {
    let mut observed = core
        .processes()
        .list(&lash_core::ProcessListFilter {
            definition: None,
            status: lash_core::ProcessStatusFilter::Any,
            waiting: None,
        })
        .await
        .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?
        .into_iter()
        .map(|process| {
            let process_ref = agent_contract_process_ref(&process);
            AgentContractProcessObservation {
                raw_process_id: process.process_id.clone(),
                process_ref: process_ref.clone(),
                observed: json!({
                    "process_ref": process_ref,
                    "kind": process.kind,
                    "label": process.label,
                    "status": process.lifecycle.label(),
                    "terminal": process.terminal,
                    "definition_present": process.identity.definition.is_some(),
                    "child_session_present": process.child_session_id.is_some(),
                }),
            }
        })
        .collect::<Vec<_>>();
    observed.sort_by(|left, right| left.process_ref.cmp(&right.process_ref));
    Ok(observed)
}

fn agent_contract_process_ref(process: &lash_core::ObservedProcess) -> String {
    let kind = process.kind.as_str();
    let label = process.label.as_str();
    let status = process.lifecycle.label();
    let terminal = process.terminal.to_string();
    let definition_present = process.identity.definition.is_some().to_string();
    let child_session_present = process.child_session_id.is_some().to_string();
    let mut hasher = Sha256::new();
    hasher.update(kind.as_bytes());
    hasher.update([0]);
    hasher.update(label.as_bytes());
    hasher.update([0]);
    hasher.update(status.as_bytes());
    hasher.update([0]);
    hasher.update(terminal.as_bytes());
    hasher.update([0]);
    hasher.update(definition_present.as_bytes());
    hasher.update([0]);
    hasher.update(child_session_present.as_bytes());
    let digest = hasher.finalize();
    format!("process-ref-{}", hex_prefix(&digest, 12))
}

fn hex_prefix(bytes: &[u8], len: usize) -> String {
    let full = bytes
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    full.chars().take(len).collect()
}

fn agent_contract_process_facts(processes: &[AgentContractProcessObservation]) -> Value {
    let mut completed_entries = BTreeSet::new();
    let mut completed_lashlang_process_refs = BTreeSet::new();
    let mut statuses = BTreeMap::<String, usize>::new();
    let mut kinds = BTreeMap::<String, usize>::new();
    for process in processes {
        let status = process
            .observed
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("");
        *statuses.entry(status.to_string()).or_default() += 1;
        let kind = process
            .observed
            .get("kind")
            .and_then(Value::as_str)
            .unwrap_or("");
        *kinds.entry(kind.to_string()).or_default() += 1;
        if status == "completed" {
            if let Some(label) = process.observed.get("label").and_then(Value::as_str) {
                completed_entries.insert(label.to_string());
            }
            if process
                .observed
                .get("kind")
                .and_then(Value::as_str)
                .is_some_and(|kind| kind == lash_lashlang_runtime::LASHLANG_ENGINE_KIND)
            {
                completed_lashlang_process_refs.insert(process.process_ref.clone());
            }
        }
    }
    json!({
        "process_count": processes.len(),
        "terminal_count": processes
            .iter()
            .filter(|process| process.observed.get("terminal").and_then(Value::as_bool) == Some(true))
            .count(),
        "completed_entries": completed_entries.into_iter().collect::<Vec<_>>(),
        "completed_lashlang_process_count": completed_lashlang_process_refs.len(),
        "completed_lashlang_process_refs": completed_lashlang_process_refs.into_iter().collect::<Vec<_>>(),
        "status_counts": statuses,
        "kind_counts": kinds,
        "all_terminal": processes
            .iter()
            .all(|process| process.observed.get("terminal").and_then(Value::as_bool) == Some(true)),
    })
}

async fn agent_contract_process_event_facts(
    core: &lash::RlmCore,
    processes: &[AgentContractProcessObservation],
) -> Result<Vec<Value>, FixedScriptRunnerError> {
    let mut events = Vec::new();
    for process in processes {
        for event in core
            .processes()
            .events(&process.raw_process_id, 0)
            .await
            .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?
        {
            events.push(json!({
                "process_ref": process.process_ref.clone(),
                "sequence": event.sequence,
                "event_type": event.event_type,
                "payload": normalize_contract_process_event_payload(event.payload),
            }));
        }
    }
    events.sort_by(|left, right| {
        (
            left.get("process_ref").and_then(Value::as_str),
            left.get("sequence").and_then(Value::as_u64),
        )
            .cmp(&(
                right.get("process_ref").and_then(Value::as_str),
                right.get("sequence").and_then(Value::as_u64),
            ))
    });
    Ok(events)
}

fn normalize_contract_process_event_payload(payload: Value) -> Value {
    let mut payload = payload;
    if let Some(object) = payload.as_object_mut() {
        object.remove("await_key_id");
    }
    payload
}

fn agent_contract_graph_facts(
    graphs: &[lash::tracing::TraceLashlangGraph],
    root_session_id: &str,
) -> Value {
    let mut completed_process_entries = BTreeSet::new();
    let mut completed_labeled_resources = BTreeSet::new();
    let mut failed_labeled_resources = BTreeSet::new();
    let mut completed_labeled_nodes = BTreeSet::new();
    let mut child_links = BTreeSet::new();
    let mut graph_status_counts = BTreeMap::<String, usize>::new();
    let mut child_session_exec_completed_count = 0usize;
    let mut child_session_exec_failed_count = 0usize;
    for graph in graphs {
        *graph_status_counts
            .entry(trace_lashlang_status_label(graph.status).to_string())
            .or_default() += 1;
        if graph.scope.session_id != root_session_id
            && matches!(
                &graph.subject,
                lash::tracing::TraceRuntimeSubject::Effect { kind, .. } if kind == "exec_code"
            )
        {
            match graph.status {
                lash::tracing::TraceLashlangStatus::Completed => {
                    child_session_exec_completed_count += 1;
                }
                lash::tracing::TraceLashlangStatus::Failed => {
                    child_session_exec_failed_count += 1;
                }
                _ => {}
            }
        }
        if graph.entry_kind == "process"
            && matches!(
                graph.subject,
                lash::tracing::TraceRuntimeSubject::Process { .. }
            )
            && graph.status == lash::tracing::TraceLashlangStatus::Completed
        {
            completed_process_entries.insert(graph.entry_name.clone());
        }
        for node in &graph.nodes {
            let title = node
                .label_metadata
                .as_ref()
                .map(|label| label.title.as_str());
            if node.kind == "resource_operation"
                && node.status == lash::tracing::TraceLashlangNodeStatus::Completed
                && let Some(title) = title
            {
                completed_labeled_resources.insert(title.to_string());
            }
            if node.kind == "resource_operation"
                && node.status == lash::tracing::TraceLashlangNodeStatus::Failed
                && let Some(title) = title
            {
                failed_labeled_resources.insert(title.to_string());
            }
            if node.status == lash::tracing::TraceLashlangNodeStatus::Completed
                && let Some(title) = title
            {
                completed_labeled_nodes.insert(title.to_string());
            }
        }
        for child in &graph.children {
            child_links.insert(format!(
                "{}->{}",
                graph.entry_name,
                child.child_entry_name.as_deref().unwrap_or("<unknown>")
            ));
        }
    }
    json!({
        "graph_count": graphs.len(),
        "status_counts": graph_status_counts,
        "completed_process_entries": completed_process_entries.into_iter().collect::<Vec<_>>(),
        "completed_labeled_resources": completed_labeled_resources.into_iter().collect::<Vec<_>>(),
        "failed_labeled_resources": failed_labeled_resources.into_iter().collect::<Vec<_>>(),
        "completed_labeled_nodes": completed_labeled_nodes.into_iter().collect::<Vec<_>>(),
        "child_links": child_links.into_iter().collect::<Vec<_>>(),
        "child_session_exec_completed_count": child_session_exec_completed_count,
        "child_session_exec_failed_count": child_session_exec_failed_count,
    })
}

fn agent_failed_child_activity_facts(
    result: &lash::TurnResult,
    events: &[lash::TurnActivity],
) -> Value {
    let mut failed_code_block_errors = Vec::new();
    let mut turn_error_messages = Vec::new();
    let mut final_value_event_count = 0usize;
    for activity in events {
        match &activity.event {
            lash::TurnEvent::CodeBlockCompleted {
                success: false,
                error: Some(error),
                ..
            } => failed_code_block_errors.push(error.clone()),
            lash::TurnEvent::Error { message } => turn_error_messages.push(message.clone()),
            lash::TurnEvent::FinalValue { .. } => final_value_event_count += 1,
            _ => {}
        }
    }
    let event_debug = format!("{events:#?}");
    json!({
        "turn_success": result.is_success(),
        "final_value_present": result.final_value().is_some(),
        "final_value_event_count": final_value_event_count,
        "failed_code_block_count": failed_code_block_errors.len(),
        "failed_code_block_errors": failed_code_block_errors,
        "turn_error_messages": turn_error_messages,
        "provider_exhaustion_observed": event_debug.contains("provider exhausted"),
        "child_task_fail_reason_observed": event_debug.contains("child boom"),
        "parent_task_fail_reason_observed": event_debug.contains("parent observed child failure"),
    })
}

fn trace_lashlang_status_label(status: lash::tracing::TraceLashlangStatus) -> &'static str {
    match status {
        lash::tracing::TraceLashlangStatus::Running => "running",
        lash::tracing::TraceLashlangStatus::Completed => "completed",
        lash::tracing::TraceLashlangStatus::Failed => "failed",
        lash::tracing::TraceLashlangStatus::Cancelled => "cancelled",
    }
}

fn normalize_contract_tool_output(value: Value) -> Value {
    let Some(object) = value.as_object() else {
        return value;
    };
    if !object.contains_key("full_output_path") {
        if object.contains_key("wall_time_seconds")
            && object.contains_key("status")
            && object.contains_key("output")
        {
            return json!({
                "status": object.get("status").cloned().unwrap_or(Value::Null),
                "done": object.get("done").cloned().unwrap_or(Value::Null),
                "running": object.get("running").cloned().unwrap_or(Value::Null),
                "exit_code": object.get("exit_code").cloned().unwrap_or(Value::Null),
                "output": object.get("output").cloned().unwrap_or(Value::Null),
            });
        }
        return value;
    }
    let output = object.get("output").and_then(Value::as_str).unwrap_or("");
    let tail_start = output.len().saturating_sub(4);
    json!({
        "status": object.get("status").cloned().unwrap_or(Value::Null),
        "exit_code": object.get("exit_code").cloned().unwrap_or(Value::Null),
        "output_len": output.len(),
        "output_tail": &output[tail_start..],
        "full_output_path_present": object
            .get("full_output_path")
            .and_then(Value::as_str)
            .is_some_and(|path| !path.is_empty()),
    })
}

fn contract_execution_payload(
    contract: &'static str,
    source_path: &'static str,
    source_scenario: &'static str,
    result: Value,
) -> Result<Value, FixedScriptRunnerError> {
    let result_body = serde_json::to_vec(&result)?;
    let result_sha256 = sha256_hex(&result_body);
    let source_material = format!("{source_path}:{source_scenario}:{result_sha256}");
    let source_hash = sha256_hex(source_material.as_bytes());
    Ok(json!({
        "contract": contract,
        "source": {
            "kind": "fixed_dst_api_execution",
            "path": source_path,
            "scenario": source_scenario,
            "source_hash": source_hash,
            "result_sha256": result_sha256,
        },
        "result": result,
    }))
}

fn fixed_texts_provider(kind: &'static str, responses: Vec<&'static str>) -> ProviderHandle {
    let responses = Arc::new(tokio::sync::Mutex::new(
        responses
            .into_iter()
            .map(str::to_string)
            .collect::<VecDeque<_>>(),
    ));
    lash_core::testing::TestProvider::builder()
        .kind(kind)
        .complete(move |_request| {
            let responses = Arc::clone(&responses);
            async move {
                let Some(text) = responses.lock().await.pop_front() else {
                    return Err(LlmTransportError::new(format!(
                        "{kind} provider exhausted its fixed response"
                    )));
                };
                let expected_text = text.clone();
                let response = text_llm_response(text);
                let response_part_text = response_text_part(&response);
                if response.full_text != expected_text
                    || response_part_text != Some(expected_text.as_str())
                {
                    return Err(LlmTransportError::new(format!(
                        "{kind} fixed response shape changed: expected full_text and text part {:?}, got full_text {:?} parts {:?}",
                        expected_text, response.full_text, response.parts
                    )));
                }
                Ok(response)
            }
        })
        .build()
        .into_handle()
}

struct ContractAppTools;

#[async_trait::async_trait]
impl lash_core::ToolProvider for ContractAppTools {
    fn tool_manifests(&self) -> Vec<lash_core::ToolManifest> {
        vec![contract_app_lookup_definition().manifest()]
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<lash_core::ToolContract>> {
        (name == "app_lookup").then(|| Arc::new(contract_app_lookup_definition().contract()))
    }

    async fn execute(&self, call: lash_core::ToolCall<'_>) -> lash_core::ToolResult {
        if call.name == "app_lookup" {
            lash_core::ToolResult::ok(json!({ "ok": true }))
        } else {
            lash_core::ToolResult::err_fmt(format!("Unknown contract app tool: {}", call.name))
        }
    }
}

fn contract_app_lookup_definition() -> lash_core::ToolDefinition {
    lash_core::ToolDefinition::raw(
        "tool:app_lookup",
        "app_lookup",
        "Lookup deterministic app state.",
        json!({
            "type": "object",
            "additionalProperties": false
        }),
        json!({
            "type": "object",
            "properties": {
                "ok": { "type": "boolean" }
            },
            "required": ["ok"],
            "additionalProperties": false
        }),
    )
    .with_lashlang_binding(lash_lashlang_runtime::LashlangToolBinding::new(
        ["tools"],
        "app_lookup",
    ))
}

struct ContractDurableInputTools {
    key_tx: Mutex<Option<tokio::sync::oneshot::Sender<Result<lash_core::AwaitEventKey, String>>>>,
    step_count: Mutex<usize>,
}

impl ContractDurableInputTools {
    fn new(key_tx: tokio::sync::oneshot::Sender<Result<lash_core::AwaitEventKey, String>>) -> Self {
        Self {
            key_tx: Mutex::new(Some(key_tx)),
            step_count: Mutex::new(0),
        }
    }

    fn step_count(&self) -> usize {
        *self.step_count.lock().expect("durable step count")
    }

    fn increment_step_count(&self) {
        *self.step_count.lock().expect("durable step count") += 1;
    }

    fn send_key_result(&self, result: Result<lash_core::AwaitEventKey, String>) {
        if let Some(tx) = self.key_tx.lock().expect("durable input key sender").take() {
            let _ = tx.send(result);
        }
    }
}

#[async_trait::async_trait]
impl lash_core::ToolProvider for ContractDurableInputTools {
    fn tool_manifests(&self) -> Vec<lash_core::ToolManifest> {
        vec![contract_durable_input_definition().manifest()]
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<lash_core::ToolContract>> {
        (name == "mock_input_request")
            .then(|| Arc::new(contract_durable_input_definition().contract()))
    }

    async fn execute(&self, call: lash_core::ToolCall<'_>) -> lash_core::ToolResult {
        if call.name != "mock_input_request" {
            return lash_core::ToolResult::err_fmt(format!(
                "Unknown durable input tool: {}",
                call.name
            ));
        }
        let durable = match call.context.durable_effects() {
            Ok(durable) => durable,
            Err(err) => {
                self.send_key_result(Err(err.to_string()));
                return lash_core::ToolResult::err_fmt(err);
            }
        };
        let question = call
            .args
            .get("question")
            .and_then(Value::as_str)
            .unwrap_or("answer")
            .to_string();
        let opened = match durable
            .run_json(
                "create",
                json!({ "question": question }),
                |input| async move {
                    Ok(json!({
                        "request_id": "request-1",
                        "question": input["question"].clone(),
                    }))
                },
            )
            .await
        {
            Ok(value) => {
                self.increment_step_count();
                value
            }
            Err(err) => {
                self.send_key_result(Err(err.to_string()));
                return lash_core::ToolResult::err_fmt(err);
            }
        };
        let key = match durable
            .external_event_key("mock-input-request:request-1")
            .await
        {
            Ok(key) => key,
            Err(err) => {
                self.send_key_result(Err(err.to_string()));
                return lash_core::ToolResult::err_fmt(err);
            }
        };
        if let Err(err) = durable
            .emit_process_event(
                "process.yield",
                json!({
                    "type": "work.input_request.opened",
                    "request_id": opened["request_id"].clone(),
                    "question": opened["question"].clone(),
                    "await_key_id": key.key_id,
                }),
            )
            .await
        {
            self.send_key_result(Err(err.to_string()));
            return lash_core::ToolResult::err_fmt(err);
        }
        self.send_key_result(Ok(key.clone()));

        let resolved = match durable.await_event_json(key).await {
            Ok(value) => value,
            Err(err) => return lash_core::ToolResult::err_fmt(err),
        };
        match durable
            .run_json(
                "complete",
                json!({
                    "request_id": opened["request_id"].clone(),
                    "answer": resolved["answer"].clone(),
                }),
                |input| async move {
                    Ok(json!({
                        "request_id": input["request_id"].clone(),
                        "answer": input["answer"].clone(),
                    }))
                },
            )
            .await
        {
            Ok(value) => {
                self.increment_step_count();
                lash_core::ToolResult::ok(value)
            }
            Err(err) => lash_core::ToolResult::err_fmt(err),
        }
    }
}

fn contract_durable_input_definition() -> lash_core::ToolDefinition {
    lash_core::ToolDefinition::raw(
        "tool:mock_input_request",
        "mock_input_request",
        "Open a durable input request and wait for the answer.",
        json!({
            "type": "object",
            "properties": {
                "question": { "type": "string" }
            },
            "required": ["question"],
            "additionalProperties": false
        }),
        json!({
            "type": "object",
            "properties": {
                "request_id": { "type": "string" },
                "answer": {}
            },
            "required": ["request_id", "answer"],
            "additionalProperties": true
        }),
    )
    .with_lashlang_binding(lash_lashlang_runtime::LashlangToolBinding::new(
        ["tools"],
        "mock_input_request",
    ))
}

fn standard_contract_turn_machine_config() -> lash_core::TurnMachineConfig {
    let protocol_driver: Arc<
        dyn lash_core::sansio::ProtocolDriverHandle<lash_core::HostTurnProtocol>,
    > = Arc::new(lash_protocol_standard::StandardDriver);
    lash_core::TurnMachineConfig {
        protocol_driver,
        projector: Arc::new(lash_core::sansio::ChatContextProjector),
        sync_execution_environment: false,
        model: "standard-max-turn-contract".to_string(),
        max_context_tokens: None,
        max_turns: None,
        model_variant: None,
        generation: lash_core::GenerationOptions::default(),
        run_session_id: None,
        autonomous: false,
        tool_specs: Vec::new().into(),
        system_prompt: std::sync::Arc::from(""),
        session_id: "standard-max-turn-contract".to_string(),
        emit_llm_trace: false,
        termination: lash_core::ProtocolTurnOptions::empty(),
        turn_limit_final_message: Arc::new(contract_turn_limit_final_message),
    }
}

fn contract_turn_limit_final_message(message_id: String, max_turns: usize) -> lash_core::Message {
    lash_core::Message {
        id: message_id.clone(),
        role: lash_core::MessageRole::System,
        parts: lash_core::shared_parts(vec![lash_core::Part {
            id: format!("{message_id}.p0"),
            kind: lash_core::PartKind::Error,
            content: format!("Turn limit reached ({max_turns}) before a final test response."),
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            tool_replay: None,
            prune_state: lash_core::PruneState::Intact,
            reasoning_meta: None,
            response_meta: None,
        }]),
        origin: None,
    }
}

fn contract_user_message(content: &str) -> lash_core::Message {
    lash_core::Message {
        id: "m0".to_string(),
        role: lash_core::MessageRole::User,
        parts: vec![lash_core::Part {
            id: "m0.p0".to_string(),
            kind: lash_core::PartKind::Text,
            content: content.to_string(),
            attachment: None,
            tool_call_id: None,
            tool_name: None,
            tool_replay: None,
            prune_state: lash_core::PruneState::Intact,
            reasoning_meta: None,
            response_meta: None,
        }]
        .into(),
        origin: None,
    }
}

fn drain_contract_turn_machine_effects(
    machine: &mut lash_core::TurnMachine,
) -> Vec<lash_core::Effect> {
    let mut effects = Vec::new();
    while let Some(effect) = machine.poll_effect() {
        effects.push(effect);
    }
    effects
}

fn find_contract_llm_call(effects: &[lash_core::Effect]) -> Option<&lash_core::sansio::EffectId> {
    effects.iter().find_map(|effect| match effect {
        lash_core::Effect::LlmCall { id, .. } => Some(id),
        _ => None,
    })
}

fn turn_outcome_contract_json(outcome: &lash_core::TurnOutcome) -> Value {
    match outcome {
        lash_core::TurnOutcome::Stopped(lash_core::TurnStop::MaxTurns) => json!({
            "kind": "stopped",
            "stop_reason": "max_turns",
        }),
        lash_core::TurnOutcome::Stopped(other) => json!({
            "kind": "stopped",
            "stop_reason": format!("{other:?}"),
        }),
        lash_core::TurnOutcome::Finished(lash_core::TurnFinish::FinalValue { value }) => json!({
            "kind": "final_value",
            "value": value,
        }),
        lash_core::TurnOutcome::Finished(other) => json!({
            "kind": "finished",
            "finish": format!("{other:?}"),
        }),
        lash_core::TurnOutcome::AgentFrameSwitch { frame_id, task } => json!({
            "kind": "agent_frame_switch",
            "frame_id": frame_id,
            "task": task,
        }),
    }
}

fn first_successful_provider(
    events: &[crate::scheduler::DeliveredBoundary],
) -> Option<&crate::scheduler::DeliveredBoundary> {
    events
        .iter()
        .filter(|event| {
            event.kind == BoundaryKind::Provider
                && event.observed.get("success").and_then(Value::as_bool) == Some(true)
        })
        .min_by_key(|event| event.sequence)
}

/// The serialized in-memory reference summary for the cross-backend check.
///
/// This drives the workload through the SAME `serialize_provider_turns` discipline
/// as the durable re-run, differing ONLY in the backend store (ephemeral
/// in-memory here vs the real durable store in `replay_workload_on_sqlite`). That
/// is what makes the comparison a well-posed durable-state equivalence: both runs
/// share one scheduling discipline, so any difference is a real store divergence
/// rather than an artifact of serialized-vs-concurrent execution. (The
/// concurrency-preserving SEARCH lane summary — `run_generated_workload`'s
/// `serialize_provider_turns == false` run — is a different discipline and is used
/// for oracle fuzzing, not for this backend equivalence comparison.)
pub async fn replay_workload_serialized_reference(
    workload: &GeneratedWorkload,
) -> Result<AbstractWorldSummary, FixedScriptRunnerError> {
    let mut world = GeneratedRuntimeWorld::with_backend(
        Arc::new(lash::persistence::InMemorySessionStoreFactory::new()),
        RuntimeEffectReplayStore::Memory,
        Arc::new(lash::persistence::InMemoryAttachmentStore::new()),
        Arc::new(lash::persistence::InMemoryProcessExecutionEnvStore::new()),
        true,
    );
    let (_events, final_summary) = drive_generated_workload(&mut world, workload).await?;
    Ok(final_summary)
}

/// Cross-backend check: re-drive the SAME generated workload through the SAME
/// dynamic runtime driver under the SAME serialized-provider-turn discipline, but
/// backed by the real `lash-sqlite-store` session store factory and the SQLite
/// durable-effect replay controller, and return the resulting abstract world
/// summary. The caller compares it against the serialized in-memory reference
/// (`replay_workload_serialized_reference`); equality proves the SQLite store
/// reproduces identical observable runtime behavior. Both runs serialize provider
/// turns and the serialized driver holds boundary delivery across a live turn's
/// completion, so the delivery order is fully determined by the workload and is
/// independent of the store's async timing — there is no fixed-order,
/// exchange-count-gated re-drive to deadlock, and no concurrency- or
/// async-timing-induced divergence.
pub async fn replay_workload_on_sqlite(
    workload: &GeneratedWorkload,
    db_root: &Path,
) -> Result<AbstractWorldSummary, FixedScriptRunnerError> {
    if db_root.exists() {
        if db_root.is_dir() {
            std::fs::remove_dir_all(db_root)?;
        } else {
            std::fs::remove_file(db_root)?;
        }
    }
    std::fs::create_dir_all(db_root)?;
    let store_factory: Arc<dyn SessionStoreFactory> = Arc::new(
        lash_sqlite_store::SqliteSessionStoreFactory::new(db_root.to_path_buf()),
    );
    let effect_replay_store =
        RuntimeEffectReplayStore::sqlite_file(db_root.join("runtime-effects.sqlite"));
    // A durable session store requires durable attachment + process-env stores,
    // so back them with the real SQLite/file stores (the in-memory reference uses
    // their ephemeral counterparts). These facets are not under cross-backend
    // comparison; only the session store's observable Lash state is.
    let attachment_store: Arc<dyn lash::persistence::AttachmentStore> = Arc::new(
        lash::persistence::FileAttachmentStore::new(db_root.join("attachments")),
    );
    let process_env_store: Arc<dyn lash::persistence::ProcessExecutionEnvStore> = Arc::new(
        lash_sqlite_store::Store::open(&db_root.join("process-env.sqlite"))
            .await
            .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?,
    );
    let mut world = GeneratedRuntimeWorld::with_backend(
        store_factory,
        effect_replay_store,
        attachment_store,
        process_env_store,
        // Serialize live provider turns for the durable re-run so async-store
        // interleaving cannot change committed outcomes vs the sync in-memory
        // reference; the comparison is then a well-posed durable-state equivalence.
        true,
    );
    let (_events, final_summary) = drive_generated_workload(&mut world, workload).await?;
    Ok(final_summary)
}

/// Cross-backend check for Postgres using the same dynamic generated workload
/// driver and serialized-provider-turn discipline as the SQLite equivalence
/// lane. This is intentionally not fixed-order trace replay: generated traces
/// can contain provider-exchange scheduling that only the generated driver owns.
pub async fn replay_workload_on_postgres(
    workload: &GeneratedWorkload,
    database_url: &str,
    artifact_root: &Path,
) -> Result<AbstractWorldSummary, FixedScriptRunnerError> {
    let storage = Arc::new(
        lash_postgres_store::PostgresStorage::connect(database_url)
            .await
            .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?,
    );
    crate::postgres_replay::reset_postgres_for_replay(storage.as_ref())
        .await
        .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
    let attachment_root = artifact_root.join("attachments");
    if attachment_root.exists() {
        std::fs::remove_dir_all(&attachment_root)?;
    }
    std::fs::create_dir_all(&attachment_root)?;
    let store_factory: Arc<dyn SessionStoreFactory> = Arc::new(storage.session_store_factory());
    let effect_replay_store = RuntimeEffectReplayStore::postgres(Arc::clone(&storage));
    let attachment_store: Arc<dyn lash::persistence::AttachmentStore> =
        Arc::new(lash::persistence::FileAttachmentStore::new(attachment_root));
    let process_env_store: Arc<dyn lash::persistence::ProcessExecutionEnvStore> =
        Arc::new(storage.process_env_store());
    let mut world = GeneratedRuntimeWorld::with_backend(
        store_factory,
        effect_replay_store,
        attachment_store,
        process_env_store,
        true,
    );
    let (_events, final_summary) = drive_generated_workload(&mut world, workload).await?;
    Ok(final_summary)
}

pub async fn run_generated_postgres_replay_for_seeds(
    artifact_root: impl AsRef<Path>,
    profile: &str,
    seed_values: &[u64],
    max_boundaries: usize,
    database_url: &str,
) -> Result<GeneratedPostgresReplayReport, FixedScriptRunnerError> {
    validate_workload_profile(profile)?;
    if seed_values.is_empty() {
        return Err(FixedScriptRunnerError::Assertion(
            "generated Postgres replay requires at least one seed".to_string(),
        ));
    }
    let artifact_root = artifact_root.as_ref();
    std::fs::create_dir_all(artifact_root)?;
    let boundary_limit = max_boundaries.max(1);
    let mut cases = Vec::new();
    let mut passed = 0;
    let mut failed = 0;
    let mut first_failure = None;
    for seed in seed_values.iter().copied() {
        let workload = generate_workload(seed, profile, boundary_limit)?;
        let reference = replay_workload_serialized_reference(&workload).await?;
        let case_dir = artifact_root.join(format!("seed-{seed:016x}"));
        std::fs::create_dir_all(&case_dir)?;
        let actual = replay_workload_on_postgres(&workload, database_url, &case_dir).await?;
        let verdict = replay_determinism(&reference, &actual);
        let report_path = case_dir.join("postgres-generated-rerun.json");
        let matches_reference = verdict.is_passed();
        let case_report = if matches_reference {
            json!({
                "schema": "lash.sim.postgres-generated-rerun.v1",
                "seed": seed,
                "profile": profile,
                "backend": "lash_postgres_store",
                "driver": "unified_generated_runtime_world",
                "matches_reference": true,
                "reference_digest": reference.digest.clone(),
                "actual_digest": actual.digest.clone(),
                "verdict": verdict.clone(),
                "final_summary": actual,
            })
        } else {
            json!({
                "schema": "lash.sim.postgres-generated-rerun.v1",
                "seed": seed,
                "profile": profile,
                "backend": "lash_postgres_store",
                "driver": "unified_generated_runtime_world",
                "matches_reference": false,
                "reference_digest": reference.digest.clone(),
                "actual_digest": actual.digest.clone(),
                "verdict": verdict.clone(),
                "reference_summary": reference,
                "actual_summary": actual,
            })
        };
        std::fs::write(&report_path, serde_json::to_vec_pretty(&case_report)?)?;
        let report_sha256 = file_sha256(&report_path)?;
        if matches_reference {
            passed += 1;
        } else {
            failed += 1;
        }
        cases.push(GeneratedPostgresReplayCase {
            seed,
            trace_alias: format!("seed-{seed:016x}"),
            status: if matches_reference {
                "passed"
            } else {
                "failed"
            },
            report_path: relative_path(artifact_root, &report_path),
            report_sha256,
            reference_digest: case_report["reference_digest"]
                .as_str()
                .unwrap_or_default()
                .to_string(),
            actual_digest: case_report["actual_digest"]
                .as_str()
                .unwrap_or_default()
                .to_string(),
            verdict: verdict.clone(),
        });
        if !matches_reference && first_failure.is_none() {
            first_failure = Some(format!(
                "generated Postgres re-run for seed {seed} ({profile}) diverged from the serialized in-memory reference: {}; wrote {}",
                verdict.message,
                report_path.display()
            ));
        }
    }
    let summary_path = artifact_root.join("summary.json");
    let report = GeneratedPostgresReplayReport {
        schema: "lash.sim.postgres-generated-rerun-summary.v1",
        status: if failed == 0 { "passed" } else { "failed" },
        profile: profile.to_string(),
        configured_max_boundaries: boundary_limit,
        database_url_redacted: crate::postgres_replay::redact_database_url(database_url),
        cases,
        counts: GeneratedPostgresReplayCounts {
            seeds: seed_values.len(),
            passed,
            failed,
        },
        summary_path: summary_path.clone(),
    };
    std::fs::write(&summary_path, serde_json::to_vec_pretty(&report)?)?;
    if let Some(message) = first_failure {
        return Err(FixedScriptRunnerError::Assertion(message));
    }
    Ok(report)
}

async fn run_generated_workload(
    workload: GeneratedWorkload,
    script_bundle_hash: &str,
    trace_path: &Path,
) -> Result<SimulationTrace, FixedScriptRunnerError> {
    let mut world = GeneratedRuntimeWorld::new();
    let (events, final_summary) = drive_generated_workload(&mut world, &workload).await?;
    // Per-seed live provider FAILURE turns: real `session.turn().run()`s that
    // stream valid prose then a non-retryable malformed chunk, released through a
    // real BoundaryScheduler, across >1 provider kind and >1 fault position.
    let live_failure_facts = drive_live_provider_failure_turns(workload.seed).await?;
    let mut oracles = vec![
        live_provider_failure_coverage(&live_failure_facts),
        scheduler_controlled_delivery(&events),
        scheduler_owned_runtime_completions(&events),
        state_machine_semantic_invariants(&events, &final_summary),
        operational_coverage(&events, &final_summary),
        ingress_sessions_opened(&final_summary),
        queued_ingress_observed(&final_summary, &events),
        cancellation_observed(&final_summary, &events),
        trigger_delivery_observed(&final_summary, &events),
        observer_reconnect_observed(&final_summary, &events),
        backend_failure_observed(&final_summary, &events),
        provider_mutation_rejected(&final_summary, &events),
        provider_transport_mutation_classified(&events),
        generated_runtime_provider_matrix_oracle(&events),
        provider_turn_interleaving_depth(&events),
        process_wake_observed(&final_summary, &events),
        tool_boundary_observed(&final_summary, &events),
        exec_code_observed(&final_summary, &events),
        cross_session_isolation(&final_summary),
        observer_convergence(&final_summary),
        runtime_session_graph_contract(&final_summary),
        runtime_graph_acyclic(&events),
        runtime_single_active_agent_frame(&events),
        runtime_usage_monotonic(&events),
        durable_effect_exactly_once(&final_summary),
        worker_stale_completion_rejected(&final_summary),
        worker_failover_continues_work(&events),
        lease_time_monotonic(&events),
        generated_suspend_resume(&events),
        generated_final_value_semantic_channel(&events),
    ];
    oracles.extend(scenario_contract_mini_oracles(&events, &final_summary));
    oracles.extend(scenario_contract_oracles(&events, &final_summary));
    let oracle = combine_oracles(&oracles);
    if !oracle.is_passed() {
        return Err(FixedScriptRunnerError::Assertion(oracle.message.clone()));
    }
    Ok(SimulationTrace::new(
        workload.seed,
        workload.generator_version,
        workload.profile,
        "0/1",
        workload.workload_family,
        workload.workload_id,
        script_bundle_hash,
        workload.aliases.into_map(),
        events,
        oracle,
        oracles,
        final_summary,
        trace_path,
    ))
}

#[derive(Default)]
struct RuntimeCompletionState {
    opened_sessions: BTreeSet<String>,
    queued_boundaries: BTreeSet<String>,
    provider_completions_by_session: BTreeMap<String, usize>,
    active_provider_turns_by_session: BTreeMap<String, usize>,
    durable_first_completions: BTreeSet<String>,
    /// When set, at most ONE live provider turn is admitted across ALL sessions:
    /// a provider turn becomes ready only when no provider turn is in flight
    /// anywhere. This is enabled ONLY for the cross-backend durable re-run
    /// (SQLite/Postgres), where preserved cross-session concurrency would let the
    /// async store's mid-op await points interleave differently from the
    /// synchronous in-memory reference and change a timing-sensitive
    /// `next_turn` `claim_and_run_pending` lease race — drifting the exchange
    /// count. Durable-state equivalence is well-posed under serial execution. The
    /// model-store generated SEARCH lane leaves this OFF and keeps full preserved
    /// concurrency (and its interleaving oracle) for concurrency fuzzing.
    serialize_provider_turns: bool,
}

impl RuntimeCompletionState {
    fn provider_started(&mut self, actor_alias: &str) {
        *self
            .active_provider_turns_by_session
            .entry(actor_alias.to_string())
            .or_default() += 1;
    }

    fn observe(&mut self, event: &crate::scheduler::DeliveredBoundary) {
        match event.kind {
            BoundaryKind::Ingress => {
                self.opened_sessions.insert(event.actor_alias.clone());
            }
            BoundaryKind::QueuedIngress => {
                self.queued_boundaries.insert(event.boundary_id.clone());
            }
            BoundaryKind::Provider => {
                *self
                    .provider_completions_by_session
                    .entry(event.actor_alias.clone())
                    .or_default() += 1;
                if let Some(active) = self
                    .active_provider_turns_by_session
                    .get_mut(&event.actor_alias)
                {
                    *active = active.saturating_sub(1);
                }
            }
            BoundaryKind::DurableEffect => {
                if event
                    .observed
                    .get("replayed")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    return;
                }
                if let Some(durable_key) = durable_key(event) {
                    self.durable_first_completions.insert(durable_key);
                }
            }
            _ => {}
        }
    }

    fn session_opened(&self, actor_alias: &str) -> bool {
        self.opened_sessions.contains(actor_alias)
    }

    fn provider_completed(&self, actor_alias: &str) -> bool {
        self.provider_completed_count(actor_alias) > 0
    }

    fn provider_completed_count(&self, actor_alias: &str) -> usize {
        self.provider_completions_by_session
            .get(actor_alias)
            .copied()
            .unwrap_or(0)
    }

    fn provider_active(&self, actor_alias: &str) -> bool {
        self.active_provider_turns_by_session
            .get(actor_alias)
            .copied()
            .unwrap_or(0)
            > 0
    }

    fn any_provider_active(&self) -> bool {
        self.active_provider_turns_by_session
            .values()
            .any(|&count| count > 0)
    }

    fn next_provider_turn_ready(&self, event: &BoundaryEvent) -> bool {
        if !self.session_opened(&event.actor_alias) || self.provider_active(&event.actor_alias) {
            return false;
        }
        let completed = self
            .provider_completions_by_session
            .get(&event.actor_alias)
            .copied()
            .unwrap_or(0);
        let Some(turn_index) = event.payload.get("turn_index").and_then(Value::as_u64) else {
            return true;
        };
        turn_index as usize == completed.saturating_add(1)
    }

    fn queued_boundary_exists(&self, boundary_id: &str) -> bool {
        self.queued_boundaries.contains(boundary_id)
    }

    fn durable_completed(&self, durable_key: &str) -> bool {
        self.durable_first_completions.contains(durable_key)
    }
}

fn split_runtime_completion_boundaries(
    boundaries: Vec<BoundaryEvent>,
) -> (Vec<BoundaryEvent>, RuntimeCompletionQueue) {
    let mut initial = Vec::new();
    let mut completions = Vec::new();
    for boundary in boundaries {
        if is_scheduler_owned_runtime_completion(boundary.kind) {
            completions.push(boundary);
        } else {
            initial.push(boundary);
        }
    }
    (initial, RuntimeCompletionQueue::new(completions))
}

fn is_scheduler_owned_runtime_completion(kind: BoundaryKind) -> bool {
    matches!(
        kind,
        BoundaryKind::Provider
            | BoundaryKind::Cancellation
            | BoundaryKind::BackendFailure
            | BoundaryKind::ProviderMutation
            | BoundaryKind::Tool
            | BoundaryKind::ExecCode
            | BoundaryKind::DurableEffect
            | BoundaryKind::Worker
            | BoundaryKind::ProcessWake
            | BoundaryKind::Observer
    )
}

fn register_ready_runtime_completions(
    queue: &mut RuntimeCompletionQueue,
    state: &mut RuntimeCompletionState,
    scheduler: &mut BoundaryScheduler,
    registered_after: &crate::scheduler::DeliveredBoundary,
    world: &mut GeneratedRuntimeWorld,
) -> Result<(), FixedScriptRunnerError> {
    let ready = queue.take_ready(|event| runtime_completion_ready(event, state));
    for event in ready {
        if !runtime_completion_ready(&event, state) {
            queue.defer(event);
            continue;
        }
        let family = runtime_completion_family(&event);
        let units = runtime_completion_units(&event)?;
        if event.kind == BoundaryKind::Provider {
            let turn_event = event.clone();
            let actor_alias = event.actor_alias.clone();
            let (_pending, completion_event) =
                queue.register_pending_event(event, registered_after, family, units);
            world.start_provider_turn(turn_event, completion_event, scheduler)?;
            state.provider_started(&actor_alias);
        } else {
            queue.register(scheduler, event, registered_after, family, units);
        }
    }
    Ok(())
}

fn runtime_completion_ready(event: &BoundaryEvent, state: &RuntimeCompletionState) -> bool {
    match event.kind {
        BoundaryKind::Provider => {
            state.next_provider_turn_ready(event)
                // Cross-backend durable re-run only: admit a provider turn only
                // when none is live anywhere, so live turns never overlap and the
                // backend's async-vs-sync store timing cannot change committed
                // outcomes. The generated SEARCH lane leaves this off.
                && (!state.serialize_provider_turns || !state.any_provider_active())
        }
        BoundaryKind::Observer => {
            if !state.session_opened(&event.actor_alias) {
                return false;
            }
            let expected_turn_index = event
                .payload
                .get("turn_index")
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize;
            state.provider_completed_count(&event.actor_alias) >= expected_turn_index
        }
        BoundaryKind::BackendFailure | BoundaryKind::ProviderMutation => {
            state.session_opened(&event.actor_alias) && !state.provider_active(&event.actor_alias)
        }
        BoundaryKind::Worker => {
            let session = completion_session_alias(event);
            state.session_opened(&session) && !state.provider_active(&session)
        }
        BoundaryKind::Cancellation => event
            .payload
            .get("target")
            .and_then(Value::as_str)
            .is_some_and(|target| state.queued_boundary_exists(target)),
        BoundaryKind::Tool | BoundaryKind::ExecCode => {
            state.provider_completed(&event.actor_alias)
                && !state.provider_active(&event.actor_alias)
        }
        BoundaryKind::DurableEffect => {
            if state.provider_active(&event.actor_alias) {
                return false;
            }
            if event.label.contains("replay") {
                durable_key_from_event(event)
                    .is_some_and(|durable_key| state.durable_completed(&durable_key))
            } else {
                state.session_opened(&event.actor_alias)
            }
        }
        BoundaryKind::ProcessWake => {
            let session = completion_session_alias(event);
            state.session_opened(&session) && !state.provider_active(&session)
        }
        _ => false,
    }
}

fn completion_session_alias(event: &BoundaryEvent) -> String {
    event
        .payload
        .get("session")
        .and_then(Value::as_str)
        .unwrap_or(&event.actor_alias)
        .to_string()
}

fn runtime_completion_family(event: &BoundaryEvent) -> &'static str {
    match event.kind {
        BoundaryKind::Provider => "provider_turn_completion",
        BoundaryKind::Cancellation => "queued_input_cancellation",
        BoundaryKind::BackendFailure => "backend_retry_or_failure",
        BoundaryKind::ProviderMutation => "provider_script_mutation",
        BoundaryKind::Tool => "tool_return",
        BoundaryKind::ExecCode => "exec_result",
        BoundaryKind::DurableEffect => "durable_effect_completion",
        BoundaryKind::Worker => "worker_lease_completion",
        BoundaryKind::ProcessWake => "process_wake",
        BoundaryKind::Observer => "observer_snapshot",
        _ => "runtime_completion",
    }
}

fn runtime_completion_units(
    event: &BoundaryEvent,
) -> Result<Vec<RuntimeCompletionUnit>, FixedScriptRunnerError> {
    if event.kind == BoundaryKind::Provider {
        let provider_kind = event
            .payload
            .get("provider_kind")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                FixedScriptRunnerError::Assertion(format!(
                    "provider runtime completion `{}` missing provider_kind",
                    event.boundary_id
                ))
            })?;
        let text = event
            .payload
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or("");
        let script = runtime_script_for_text(provider_kind, text)
            .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
        return Ok(script
            .timeline
            .iter()
            .enumerate()
            .map(|(index, wire_event)| {
                RuntimeCompletionUnit::new(
                    format!("provider:{}:{index:02}", wire_event.event_name()),
                    wire_event.at(),
                )
            })
            .collect());
    }

    let unit = match event.kind {
        BoundaryKind::Cancellation => "runtime:cancel_pending_turn_input",
        BoundaryKind::BackendFailure => {
            if event
                .payload
                .get("retryable")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                "runtime:backend_retry_attempt"
            } else {
                "runtime:backend_terminal_failure"
            }
        }
        BoundaryKind::ProviderMutation => "provider:mutated_script_parser_rejection",
        BoundaryKind::Tool => "runtime:tool_attempt_return",
        BoundaryKind::ExecCode => "runtime:exec_code_result",
        BoundaryKind::DurableEffect => {
            if event.label.contains("replay") {
                "runtime:durable_effect_replay"
            } else {
                "runtime:durable_effect_local_completion"
            }
        }
        BoundaryKind::Worker => "runtime:worker_stale_completion",
        BoundaryKind::ProcessWake => "runtime:process_wake_delivery",
        BoundaryKind::Observer => "runtime:observer_snapshot",
        _ => "runtime:completion",
    };
    Ok(vec![RuntimeCompletionUnit::new(unit, event.at)])
}

fn durable_key(event: &crate::scheduler::DeliveredBoundary) -> Option<String> {
    event
        .observed
        .get("durable_key")
        .or_else(|| event.payload.get("durable_key"))
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn durable_key_from_event(event: &BoundaryEvent) -> Option<String> {
    event
        .payload
        .get("durable_key")
        .and_then(Value::as_str)
        .map(str::to_string)
}

struct GeneratedRuntimeWorld {
    sessions: BTreeMap<String, GeneratedRuntimeSession>,
    queued_inputs: BTreeMap<String, String>,
    lease_ticks: BTreeMap<String, Vec<u64>>,
    backend_faults: SimBackendFaultInjector,
    provider_mutations: SimProviderMutationHarness,
    trigger_harness: SimTriggerHarness,
    store_factory: Arc<dyn SessionStoreFactory>,
    attachment_store: Arc<dyn lash::persistence::AttachmentStore>,
    process_env_store: Arc<dyn lash::persistence::ProcessExecutionEnvStore>,
    runtime_boundaries: RuntimeBoundaryHarness,
    peak_concurrent_live_turns: usize,
    suspending_turns: BTreeMap<String, SuspendingTurn>,
    /// When set, the driver admits at most one live provider turn at a time
    /// (see `RuntimeCompletionState::serialize_provider_turns`). Enabled for the
    /// cross-backend durable re-run; left off for the in-memory reference/search.
    serialize_provider_turns: bool,
}

/// A real generated turn that parks mid-flight on a tool/durable/exec await key
/// and is resumed only when the boundary scheduler delivers the matching
/// completion. This generalizes the fixed `prove_pending_tool_completion_through_turn`
/// proof into the live, interleaved generated search.
struct SuspendingTurn {
    core: lash::StandardCore,
    handle: tokio::task::JoinHandle<Result<lash::TurnResult, FixedScriptRunnerError>>,
    events: Arc<RuntimeProofRecordingEvents>,
    key_slot: Arc<tokio::sync::Mutex<Option<lash_core::AwaitEventKey>>>,
    suspend_kind: BoundaryKind,
    tool_name: String,
    resolution: Value,
    /// `true` once the world has observed the turn parked on its await key
    /// (the tool registered its completion key and the turn future is not yet
    /// finished). Recorded before any resolution so the oracle can prove the
    /// turn suspended rather than running synchronously.
    suspended_before_completion: Option<bool>,
    resolution_scheduled: bool,
    completed_before_resolution: usize,
}

struct GeneratedRuntimeSession {
    _core: lash::StandardCore,
    session: lash::LashSession,
    transport: Arc<ScriptedLlmHttpTransport>,
    provider_schedule: ScriptedTransportSchedule,
    provider_scripts: Vec<ProviderWireScript>,
    provider_kind: String,
    active_provider_turns: BTreeMap<String, ActiveProviderTurn>,
    finished_provider_turns: BTreeMap<String, Value>,
}

struct ActiveProviderTurn {
    completion_event: BoundaryEvent,
    handle: tokio::task::JoinHandle<Result<Value, FixedScriptRunnerError>>,
    final_ready_at: u64,
}

impl GeneratedRuntimeWorld {
    fn new() -> Self {
        // The in-memory reference / generated SEARCH lane keeps full preserved
        // cross-session concurrency (serialize_provider_turns = false).
        Self::with_backend(
            Arc::new(lash::persistence::InMemorySessionStoreFactory::new()),
            RuntimeEffectReplayStore::Memory,
            Arc::new(lash::persistence::InMemoryAttachmentStore::new()),
            Arc::new(lash::persistence::InMemoryProcessExecutionEnvStore::new()),
            false,
        )
    }

    /// Build the generated runtime world over an explicit backend (session store
    /// factory + durable-effect replay store + attachment/process-env stores). The
    /// reference in-memory run and the cross-backend SQLite re-run drive the SAME
    /// workload through the SAME scheduler-driven, concurrency-faithful driver,
    /// differing ONLY in this backend. That makes the cross-backend comparison
    /// genuinely apples-to-apples: any divergence is a real store divergence, not
    /// an artifact of a separate, fixed-order, provider-event-gated re-drive. A
    /// durable session store requires durable attachment/process-env stores, so
    /// those are supplied per backend rather than hard-coded to in-memory.
    fn with_backend(
        store_factory: Arc<dyn SessionStoreFactory>,
        effect_replay_store: RuntimeEffectReplayStore,
        attachment_store: Arc<dyn lash::persistence::AttachmentStore>,
        process_env_store: Arc<dyn lash::persistence::ProcessExecutionEnvStore>,
        serialize_provider_turns: bool,
    ) -> Self {
        Self {
            sessions: BTreeMap::new(),
            queued_inputs: BTreeMap::new(),
            lease_ticks: BTreeMap::new(),
            backend_faults: SimBackendFaultInjector::default(),
            provider_mutations: SimProviderMutationHarness::default(),
            trigger_harness: SimTriggerHarness::default(),
            runtime_boundaries: RuntimeBoundaryHarness::new(
                Arc::clone(&store_factory),
                effect_replay_store,
            ),
            store_factory,
            attachment_store,
            process_env_store,
            peak_concurrent_live_turns: 0,
            suspending_turns: BTreeMap::new(),
            serialize_provider_turns,
        }
    }

    fn pending_suspend_turn_count(&self) -> usize {
        self.suspending_turns
            .values()
            .filter(|turn| !turn.resolution_scheduled)
            .count()
    }

    /// Sample the number of provider-turn futures that are spawned and not yet
    /// joined, tracking the runtime-observed interleaving highwater. This is the
    /// true count of live turn futures (a superset of the event-derived measure,
    /// which only counts a turn live once its first provider chunk releases).
    fn sample_live_turn_highwater(&mut self) {
        let live = self.active_provider_turn_count();
        self.peak_concurrent_live_turns = self.peak_concurrent_live_turns.max(live);
    }

    async fn deliver_boundary(
        &mut self,
        event: &BoundaryEvent,
    ) -> Result<Value, FixedScriptRunnerError> {
        if event.payload.get("suspend_resume").and_then(Value::as_bool) == Some(true) {
            return self.resolve_suspended_turn(event).await;
        }
        match event.kind {
            BoundaryKind::Ingress => {
                if event.payload.get("suspend_kind").is_some() {
                    self.open_suspending_session(event).await
                } else {
                    self.open_runtime_session(event).await
                }
            }
            BoundaryKind::QueuedIngress => self.queue_turn_input(event).await,
            BoundaryKind::Provider => self.finish_provider_turn(event).await,
            BoundaryKind::ProviderEvent => self.release_provider_event(event).await,
            BoundaryKind::Observer => self.observe_session(event),
            BoundaryKind::Cancellation => self.cancel_queued_input(event).await,
            BoundaryKind::Trigger => self.trigger_harness.deliver(event).await,
            BoundaryKind::BackendFailure => Ok(self.backend_faults.inject(event)),
            BoundaryKind::ProviderMutation => self.provider_mutations.reject(event).await,
            BoundaryKind::DurableEffect
            | BoundaryKind::ProcessWake
            | BoundaryKind::Worker
            | BoundaryKind::Tool
            | BoundaryKind::ExecCode => self
                .runtime_boundaries
                .deliver(event)
                .await
                .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string())),
            BoundaryKind::LeaseTime => self.advance_lease_time(event).await,
        }
    }

    async fn open_runtime_session(
        &mut self,
        event: &BoundaryEvent,
    ) -> Result<Value, FixedScriptRunnerError> {
        let provider_texts = event
            .payload
            .get("provider_texts")
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .ok_or_else(|| {
                FixedScriptRunnerError::Assertion(format!(
                    "ingress boundary `{}` missing provider_texts",
                    event.boundary_id
                ))
            })?;
        if provider_texts.is_empty() {
            return Err(FixedScriptRunnerError::Assertion(format!(
                "ingress boundary `{}` provided no runtime provider scripts",
                event.boundary_id
            )));
        }
        let provider_kind = event
            .payload
            .get("provider_kind")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                FixedScriptRunnerError::Assertion(format!(
                    "ingress boundary `{}` missing provider_kind",
                    event.boundary_id
                ))
            })?;
        let scripts = runtime_provider_scripts_for_texts(provider_kind, &provider_texts)
            .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
        let provider_scripts = scripts.clone();
        let provider_schedule = ScriptedTransportSchedule::new();
        let (core, transport, provider_kind) = runtime_core_for_scripts(
            scripts,
            Arc::clone(&self.store_factory),
            Arc::clone(&self.attachment_store),
            Arc::clone(&self.process_env_store),
            Some(provider_schedule.clone()),
            // The generated harness owns provider execution through explicit
            // `Provider` boundaries. Each modeled success turn gets one scripted
            // exchange slot and one scheduler-owned release sequence. The runtime
            // queued-work driver would run next-turn inputs autonomously against the
            // same scripted transport, consuming exchange slots and shifting later
            // modeled turns onto unreleased gates. Queued-work behavior is exercised
            // by dedicated runtime boundary facts; this scripted provider core keeps
            // queued work inert so modeled provider boundaries remain the only
            // provider exchanges in the session.
            true,
        )?;
        let session = core
            .session(event.actor_alias.clone())
            .open_fresh()
            .await
            .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
        if session.session_id() != event.actor_alias {
            return Err(FixedScriptRunnerError::Assertion(format!(
                "ingress opened session `{}`, expected `{}`",
                session.session_id(),
                event.actor_alias
            )));
        }
        self.sessions.insert(
            event.actor_alias.clone(),
            GeneratedRuntimeSession {
                _core: core,
                session,
                transport,
                provider_schedule,
                provider_scripts,
                provider_kind,
                active_provider_turns: BTreeMap::new(),
                finished_provider_turns: BTreeMap::new(),
            },
        );
        Ok(json!({
            "session": event.actor_alias,
            "opened": true,
            "ingress_count": 1,
        }))
    }

    async fn queue_turn_input(
        &mut self,
        event: &BoundaryEvent,
    ) -> Result<Value, FixedScriptRunnerError> {
        let runtime_session = self.sessions.get(&event.actor_alias).ok_or_else(|| {
            FixedScriptRunnerError::Assertion(format!(
                "queued ingress boundary `{}` ran before ingress for `{}`",
                event.boundary_id, event.actor_alias
            ))
        })?;
        let text = event
            .payload
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or("queued input");
        let source_key = event
            .payload
            .get("source_key")
            .and_then(Value::as_str)
            .unwrap_or(&event.boundary_id);
        let mut enqueue = runtime_session
            .session
            .enqueue(lash::TurnInput::text(text.to_string()))
            .id(source_key);
        let observed_active_turn_id = event
            .payload
            .get("active_turn_id")
            .and_then(Value::as_str)
            .map(str::to_string);
        if event.payload.get("ingress_mode").and_then(Value::as_str) == Some("active_turn") {
            let active_turn_id = observed_active_turn_id
                .as_deref()
                .unwrap_or(&event.boundary_id);
            enqueue = enqueue.ingress(lash_core::TurnInputIngress::active_turn(
                active_turn_id,
                lash_core::TurnInputCheckpointBoundary::AfterWork,
            ));
        }
        let pending = enqueue
            .send()
            .await
            .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
        self.queued_inputs
            .insert(event.boundary_id.clone(), pending.input_id.clone());
        Ok(json!({
            "session": event.actor_alias,
            "queued_ingress": true,
            "source_key": source_key,
            "input_id": pending.input_id,
            "input_state": pending.state.as_str(),
            "ingress_mode": event
                .payload
                .get("ingress_mode")
                .and_then(Value::as_str)
                .unwrap_or("next_turn"),
            "active_turn_id": observed_active_turn_id,
        }))
    }

    fn start_provider_turn(
        &mut self,
        event: BoundaryEvent,
        completion_event: BoundaryEvent,
        scheduler: &mut BoundaryScheduler,
    ) -> Result<(), FixedScriptRunnerError> {
        let runtime_session = self.sessions.get_mut(&event.actor_alias).ok_or_else(|| {
            FixedScriptRunnerError::Assertion(format!(
                "provider boundary `{}` ran before ingress for `{}`",
                event.boundary_id, event.actor_alias
            ))
        })?;
        let expected_turn_index = event
            .payload
            .get("turn_index")
            .and_then(Value::as_u64)
            .unwrap_or(1) as usize;
        let script = runtime_session
            .provider_scripts
            .get(expected_turn_index.saturating_sub(1))
            .cloned()
            .ok_or_else(|| {
                FixedScriptRunnerError::Assertion(format!(
                    "provider boundary `{}` had no runtime provider script for turn {}",
                    event.boundary_id, expected_turn_index
                ))
            })?;
        let exchange_index = expected_turn_index.saturating_sub(1);
        if runtime_session
            .active_provider_turns
            .contains_key(&event.boundary_id)
        {
            return Err(FixedScriptRunnerError::Assertion(format!(
                "provider boundary `{}` was already active",
                event.boundary_id
            )));
        }

        let turn_started_at = completion_event.at;
        let mut final_ready_at = turn_started_at.saturating_add(1);
        for (event_index, wire_event) in script.timeline.iter().enumerate() {
            let release_at = turn_started_at.saturating_add(wire_event.at());
            final_ready_at = final_ready_at.max(release_at.saturating_add(1));
            scheduler.schedule(provider_release_boundary(
                &completion_event,
                &script,
                exchange_index,
                event_index,
                wire_event,
                release_at,
            ));
        }

        let mut completion_event = completion_event;
        completion_event.at = final_ready_at;
        set_runtime_completion_ready_at(&mut completion_event, final_ready_at);

        let session = runtime_session.session.clone();
        let transport = Arc::clone(&runtime_session.transport);
        let provider_kind = runtime_session.provider_kind.clone();
        let task_event = event.clone();
        let handle = tokio::spawn(async move {
            run_provider_turn_task(session, transport, provider_kind, task_event).await
        });
        runtime_session.active_provider_turns.insert(
            event.boundary_id.clone(),
            ActiveProviderTurn {
                completion_event,
                handle,
                final_ready_at,
            },
        );
        Ok(())
    }

    async fn release_provider_event(
        &self,
        event: &BoundaryEvent,
    ) -> Result<Value, FixedScriptRunnerError> {
        let turn_boundary_id = event
            .payload
            .get("turn_boundary_id")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                FixedScriptRunnerError::Assertion(format!(
                    "provider event `{}` missing turn_boundary_id",
                    event.boundary_id
                ))
            })?;
        let event_index = event
            .payload
            .get("event_index")
            .and_then(Value::as_u64)
            .ok_or_else(|| {
                FixedScriptRunnerError::Assertion(format!(
                    "provider event `{}` missing event_index",
                    event.boundary_id
                ))
            })? as usize;
        let exchange_index = event
            .payload
            .get("exchange_index")
            .and_then(Value::as_u64)
            .ok_or_else(|| {
                FixedScriptRunnerError::Assertion(format!(
                    "provider event `{}` missing exchange_index",
                    event.boundary_id
                ))
            })? as usize;
        let event_name = event
            .payload
            .get("event_name")
            .and_then(Value::as_str)
            .unwrap_or("provider_event");
        let runtime_session = self.sessions.get(&event.actor_alias).ok_or_else(|| {
            FixedScriptRunnerError::Assertion(format!(
                "provider event `{}` ran before ingress for `{}`",
                event.boundary_id, event.actor_alias
            ))
        })?;
        let active_turn_pending = runtime_session
            .active_provider_turns
            .contains_key(turn_boundary_id);
        let release = active_turn_pending.then(|| {
            runtime_session.provider_schedule.release(
                exchange_index,
                event_index,
                event_name,
                event.at,
            )
        });
        let mut observed = json!({
            "session": event.actor_alias,
            "provider_event_release": true,
            "turn_boundary_id": turn_boundary_id,
            "exchange_index": exchange_index,
            "event_index": event_index,
            "event_name": event_name,
            "provider_kind": runtime_session.provider_kind,
        });
        if let Some(release) = release {
            observed["active_turn_pending_before_release"] = json!(active_turn_pending);
            observed["released_while_turn_pending"] = json!(active_turn_pending);
            observed["scripted_transport_release"] = json!({
                "exchange_index": release.exchange_index,
                "event_index": release.event_index,
                "event_name": release.event_name,
                "at": release.at,
                "blocked_before_release": release.blocked_before_release,
            });
        } else {
            observed["provider_event_release_noop_turn_finished"] = json!(true);
        }
        Ok(observed)
    }

    async fn finish_provider_turn(
        &mut self,
        event: &BoundaryEvent,
    ) -> Result<Value, FixedScriptRunnerError> {
        let runtime_session = self.sessions.get_mut(&event.actor_alias).ok_or_else(|| {
            FixedScriptRunnerError::Assertion(format!(
                "provider completion `{}` ran before ingress for `{}`",
                event.boundary_id, event.actor_alias
            ))
        })?;
        runtime_session
            .finished_provider_turns
            .remove(&event.boundary_id)
            .ok_or_else(|| {
                FixedScriptRunnerError::Assertion(format!(
                    "provider completion `{}` was delivered before its turn future completed",
                    event.boundary_id
                ))
            })
    }

    async fn schedule_finished_provider_turns(
        &mut self,
        scheduler: &mut BoundaryScheduler,
    ) -> Result<(), FixedScriptRunnerError> {
        tokio::task::yield_now().await;
        let session_aliases = self.sessions.keys().cloned().collect::<Vec<_>>();
        for session_alias in session_aliases {
            let finished_ids = self
                .sessions
                .get(&session_alias)
                .into_iter()
                .flat_map(|session| {
                    session
                        .active_provider_turns
                        .iter()
                        .filter(|(_, active)| active.handle.is_finished())
                        .map(|(id, _)| id.clone())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            for turn_id in finished_ids {
                let active = self
                    .sessions
                    .get_mut(&session_alias)
                    .and_then(|session| session.active_provider_turns.remove(&turn_id))
                    .ok_or_else(|| {
                        FixedScriptRunnerError::Assertion(format!(
                            "finished provider turn `{turn_id}` disappeared before scheduling completion"
                        ))
                    })?;
                let ActiveProviderTurn {
                    completion_event,
                    handle,
                    final_ready_at,
                } = active;
                let observed = handle.await.map_err(|err| {
                    FixedScriptRunnerError::Runtime(format!(
                        "provider turn `{turn_id}` task failed to join: {err}"
                    ))
                })??;
                let runtime_session = self.sessions.get_mut(&session_alias).ok_or_else(|| {
                    FixedScriptRunnerError::Assertion(format!(
                        "provider turn `{turn_id}` session `{session_alias}` disappeared"
                    ))
                })?;
                runtime_session
                    .finished_provider_turns
                    .insert(turn_id, observed);
                debug_assert_eq!(completion_event.at, final_ready_at);
                scheduler.schedule(completion_event);
            }
        }
        Ok(())
    }

    fn active_provider_turn_count(&self) -> usize {
        self.sessions
            .values()
            .map(|session| session.active_provider_turns.len())
            .sum()
    }

    /// The earliest completion time (`final_ready_at`) across all live provider
    /// turns, or `None` when none is live. In serialize mode this is the delivery
    /// barrier: the driver holds back any boundary scheduled at or after this time
    /// until the turn finishes and its completion lands in the scheduler, so the
    /// completion is always delivered at its own `at` ahead of later boundaries —
    /// making the delivery order independent of how long the (sync in-memory vs
    /// async durable) store takes to drive the turn to completion.
    fn min_active_final_ready_at(&self) -> Option<u64> {
        self.sessions
            .values()
            .flat_map(|session| session.active_provider_turns.values())
            .map(|active| active.final_ready_at)
            .min()
    }

    fn observe_session(&self, event: &BoundaryEvent) -> Result<Value, FixedScriptRunnerError> {
        let runtime_session = self.sessions.get(&event.actor_alias).ok_or_else(|| {
            FixedScriptRunnerError::Assertion(format!(
                "observer boundary `{}` ran before ingress for `{}`",
                event.boundary_id, event.actor_alias
            ))
        })?;
        let expected_turn_index = event
            .payload
            .get("turn_index")
            .and_then(Value::as_u64)
            .unwrap_or(1) as usize;
        let observation = runtime_session.session.observe().current_observation();
        let read_view = observation.read_view;
        let graph_node_count = read_view.session_graph().nodes.len();
        let transcript_message_count = read_view.messages().len();
        let graph_non_empty = graph_node_count > 0;
        let observer_ok = read_view.session_id() == event.actor_alias
            && read_view.turn_index() == expected_turn_index
            && graph_non_empty;
        if !observer_ok {
            return Err(FixedScriptRunnerError::Assertion(format!(
                "observer invariants failed for `{}`: session_id={} turn_index={} graph_nodes={}",
                event.boundary_id,
                read_view.session_id(),
                read_view.turn_index(),
                read_view.session_graph().nodes.len()
            )));
        }
        Ok(json!({
            "session": event.actor_alias,
            "turn_index": expected_turn_index,
            "reconnected": event.payload
                .get("reconnect")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            "graph_node_count": graph_node_count,
            "transcript_message_count": transcript_message_count,
            "observer_invariants": {
                "session_id": true,
                "turn_index_converged": true,
                "graph_non_empty": true,
                "transcript_message_count_converged": transcript_message_count >= expected_turn_index * 2,
            },
        }))
    }

    async fn cancel_queued_input(
        &mut self,
        event: &BoundaryEvent,
    ) -> Result<Value, FixedScriptRunnerError> {
        let runtime_session = self.sessions.get(&event.actor_alias).ok_or_else(|| {
            FixedScriptRunnerError::Assertion(format!(
                "cancellation boundary `{}` ran before ingress for `{}`",
                event.boundary_id, event.actor_alias
            ))
        })?;
        let target = event
            .payload
            .get("target")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                FixedScriptRunnerError::Assertion(format!(
                    "cancellation boundary `{}` missing target",
                    event.boundary_id
                ))
            })?;
        let input_id = self.queued_inputs.get(target).cloned().ok_or_else(|| {
            FixedScriptRunnerError::Assertion(format!(
                "cancellation boundary `{}` target `{target}` was not queued",
                event.boundary_id
            ))
        })?;
        let outcome = runtime_session
            .session
            .cancel_pending_turn_input(&input_id)
            .await
            .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
        let (cancelled, cancel_outcome) = match &outcome {
            lash::persistence::PendingTurnInputCancelOutcome::Cancelled(_) => (true, "cancelled"),
            lash::persistence::PendingTurnInputCancelOutcome::AlreadyClaimed { .. } => {
                (false, "already_claimed")
            }
            lash::persistence::PendingTurnInputCancelOutcome::AlreadyCompleted(_) => {
                (false, "already_completed")
            }
            lash::persistence::PendingTurnInputCancelOutcome::AlreadyCancelled(_) => {
                (false, "already_cancelled")
            }
            lash::persistence::PendingTurnInputCancelOutcome::NotFound => (false, "not_found"),
        };
        Ok(json!({
            "session": event.actor_alias,
            "target": target,
            "cancelled": cancelled,
            "cancel_outcome": cancel_outcome,
        }))
    }

    async fn advance_lease_time(
        &mut self,
        event: &BoundaryEvent,
    ) -> Result<Value, FixedScriptRunnerError> {
        let tick = event
            .payload
            .get("tick")
            .and_then(Value::as_u64)
            .unwrap_or(event.at);
        let ticks = self
            .lease_ticks
            .entry(event.actor_alias.clone())
            .or_default();
        let monotonic = ticks
            .last()
            .copied()
            .is_none_or(|previous| previous <= tick);
        ticks.push(tick);
        // Ground the lease-time tick in a real session-execution-lease fencing
        // token from the store. This token (not the generator-fed `tick`) is
        // what the lease-time-monotonic oracle now asserts; the field is
        // normalized away on cross-backend replay because the abstract model
        // store cannot reproduce a real lease fence.
        let lease_fencing_token = self
            .runtime_boundaries
            .lease_probe_fencing_token(&event.actor_alias)
            .await
            .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
        Ok(json!({
            "session": event.actor_alias,
            "lease_time_tick": tick,
            "monotonic": monotonic,
            "runtime_lease_probe": {
                "session_execution_lease_fencing_token": lease_fencing_token,
                "real_lease_store": true,
            },
        }))
    }

    /// Open a suspend session and spawn its real turn. The turn calls a sim tool
    /// that registers its await key and returns `ToolResult::pending`, so the
    /// turn future parks mid-flight and cannot finish until the scheduler later
    /// delivers the matching completion boundary. The observed masquerades as a
    /// normal ingress for the abstract store; suspend evidence lives in a
    /// normalized-away field so cross-backend replay stays green.
    async fn open_suspending_session(
        &mut self,
        event: &BoundaryEvent,
    ) -> Result<Value, FixedScriptRunnerError> {
        let suspend_kind_label = event
            .payload
            .get("suspend_kind")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                FixedScriptRunnerError::Assertion(format!(
                    "suspend ingress `{}` missing suspend_kind",
                    event.boundary_id
                ))
            })?;
        let suspend_kind = match suspend_kind_label {
            "tool" => BoundaryKind::Tool,
            "durable_effect" => BoundaryKind::DurableEffect,
            "exec_code" => BoundaryKind::ExecCode,
            other => {
                return Err(FixedScriptRunnerError::Assertion(format!(
                    "suspend ingress `{}` had unknown suspend_kind `{other}`",
                    event.boundary_id
                )));
            }
        };
        let session_alias = event.actor_alias.clone();
        let tool_name = format!("await_{suspend_kind_label}");
        let resolution = json!({
            "ok": true,
            "suspend_kind": suspend_kind_label,
            "session": session_alias,
            "resolved_by": "lash-sim-boundary-scheduler",
        });

        let key_slot = Arc::new(tokio::sync::Mutex::new(None));
        let events = Arc::new(RuntimeProofRecordingEvents::default());
        // Route the parked turn through the real openai-compatible provider wire
        // transport (not a TestProvider), so both the tool-call exchange that
        // suspends the turn and the post-resume exchange exercise real provider
        // wire parsing.
        let suspend_scripts = suspend_roundtrip_scripts(&tool_name)
            .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
        let transport = Arc::new(ScriptedLlmHttpTransport::from_scripts(suspend_scripts));
        let (provider_handle, model, _provider_kind) =
            runtime_provider_components(OPENAI_COMPATIBLE, &transport)
                .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
        let core = lash::StandardCore::builder()
            .effect_host(Arc::new(lash::durability::InlineEffectHost::default()))
            .attachment_store(Arc::new(lash::persistence::InMemoryAttachmentStore::new()))
            .process_env_store(Arc::new(
                lash::persistence::InMemoryProcessExecutionEnvStore::new(),
            ))
            .store_factory(Arc::new(
                lash::persistence::InMemorySessionStoreFactory::new(),
            ))
            .process_registry(Arc::new(lash_core::TestLocalProcessRegistry::default())
                as Arc<dyn lash_core::ProcessRegistry>)
            .provider(provider_handle)
            .model(model)
            .tools(Arc::new(SuspendToolProvider::new(
                tool_name.clone(),
                Arc::clone(&key_slot),
            )) as Arc<dyn lash_core::ToolProvider>)
            .build()
            .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
        let session = core
            .session(session_alias.clone())
            .open_fresh()
            .await
            .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
        let turn_session = session.clone();
        let turn_events = Arc::clone(&events);
        let suspend_label = suspend_kind_label.to_string();
        let handle = tokio::spawn(async move {
            turn_session
                .turn(lash::TurnInput::text(format!(
                    "await {suspend_label} completion"
                )))
                .stream_to(turn_events.as_ref())
                .await
                .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))
        });
        self.suspending_turns.insert(
            session_alias.clone(),
            SuspendingTurn {
                core,
                handle,
                events,
                key_slot,
                suspend_kind,
                tool_name,
                resolution,
                suspended_before_completion: None,
                resolution_scheduled: false,
                completed_before_resolution: 0,
            },
        );
        Ok(json!({
            "session": session_alias,
            "opened": true,
            "ingress_count": 1,
            "runtime_suspend": {
                "suspend_kind": suspend_kind_label,
                "spawned": true,
            },
        }))
    }

    /// Poll the spawned suspend turns. Once a turn has registered its await key
    /// (it parked on the tool) and is still in flight, schedule the matching
    /// completion boundary into the scheduler — mirroring how finished provider
    /// turns schedule their completion. The completion is the only thing that
    /// can resume the parked turn.
    async fn schedule_parked_suspend_resolutions(
        &mut self,
        scheduler: &mut BoundaryScheduler,
        ready_at: u64,
    ) -> Result<(), FixedScriptRunnerError> {
        tokio::task::yield_now().await;
        for (session_alias, turn) in self.suspending_turns.iter_mut() {
            if turn.resolution_scheduled {
                continue;
            }
            let key_present = turn.key_slot.lock().await.is_some();
            if !key_present {
                continue;
            }
            // The await key exists, so the tool parked the turn. Record that the
            // turn suspended before any completion was delivered.
            let suspended =
                !turn.handle.is_finished() && turn.events.tool_completed_count().await == 0;
            turn.suspended_before_completion = Some(suspended);
            turn.completed_before_resolution = turn.events.tool_completed_count().await;
            let boundary_id = format!("{session_alias}:suspend-resume:001");
            let label = format!("suspend.{}.resume", boundary_kind_label(turn.suspend_kind));
            scheduler.schedule(BoundaryEvent::new(
                boundary_id,
                session_alias.clone(),
                turn.suspend_kind,
                ready_at,
                label,
                json!({
                    "suspend_resume": true,
                    "tool": turn.tool_name,
                    "output": turn.resolution,
                    "session": session_alias,
                }),
            ));
            turn.resolution_scheduled = true;
        }
        Ok(())
    }

    /// Resolve a parked suspend turn via `core.completions().resolve(...)` when
    /// the scheduler delivers its completion boundary, then await the resumed
    /// turn to completion. The observed masquerades as the matching runtime
    /// boundary (tool/exec/durable) for the abstract store, with suspend/resume
    /// evidence in a normalized-away field.
    async fn resolve_suspended_turn(
        &mut self,
        event: &BoundaryEvent,
    ) -> Result<Value, FixedScriptRunnerError> {
        let turn = self
            .suspending_turns
            .remove(&event.actor_alias)
            .ok_or_else(|| {
                FixedScriptRunnerError::Assertion(format!(
                    "suspend resume `{}` had no parked turn for `{}`",
                    event.boundary_id, event.actor_alias
                ))
            })?;
        let key = turn.key_slot.lock().await.take().ok_or_else(|| {
            FixedScriptRunnerError::Assertion(format!(
                "suspend resume `{}` delivered before the turn registered its await key",
                event.boundary_id
            ))
        })?;
        let suspended_before_completion = turn.suspended_before_completion.unwrap_or(false);
        let completed_before = turn.completed_before_resolution;
        let resolution = turn.resolution.clone();
        let accepted = turn
            .core
            .completions()
            .resolve(key, lash_core::Resolution::Ok(resolution.clone()))
            .await
            .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
        let result = turn.handle.await.map_err(|err| {
            FixedScriptRunnerError::Runtime(format!(
                "suspend turn `{}` task failed to join: {err}",
                event.actor_alias
            ))
        })??;
        let completed_after = turn.events.tool_completed_count().await;
        let assistant_message = result.assistant_message().unwrap_or_default().to_string();
        let resumed_after_completion = completed_after > completed_before
            && matches!(
                &result.outcome,
                lash_core::TurnOutcome::Finished(lash_core::TurnFinish::AssistantMessage { .. })
            );
        let resolve_accepted = matches!(accepted, lash_core::ResolveOutcome::Accepted);
        Ok(json!({
            "session": event.actor_alias,
            "tool_output": resolution,
            "tool_name": turn.tool_name,
            "tool_call_id": event.boundary_id,
            "execution_count": 1,
            "runtime_tool_output": lash_core::ToolCallOutput::success(resolution.clone()),
            "runtime_suspend": {
                "suspend_kind": boundary_kind_label(turn.suspend_kind),
                "turn_suspended_before_completion": suspended_before_completion,
                "scheduler_delivered_completion": true,
                "resolve_accepted": resolve_accepted,
                "resumed_after_completion": resumed_after_completion,
                "completed_event_count_before_resolution": completed_before,
                "completed_event_count_after_resolution": completed_after,
                "final_assistant_message": assistant_message,
            },
        }))
    }
}

fn boundary_kind_label(kind: BoundaryKind) -> &'static str {
    match kind {
        BoundaryKind::Tool => "tool",
        BoundaryKind::DurableEffect => "durable_effect",
        BoundaryKind::ExecCode => "exec_code",
        _ => "unknown",
    }
}

fn provider_release_boundary(
    turn_event: &BoundaryEvent,
    script: &ProviderWireScript,
    exchange_index: usize,
    event_index: usize,
    wire_event: &ProviderWireEvent,
    at: u64,
) -> BoundaryEvent {
    BoundaryEvent::new(
        format!(
            "{}:provider-event:{event_index:03}:{}",
            turn_event.boundary_id,
            wire_event.event_name()
        ),
        turn_event.actor_alias.clone(),
        BoundaryKind::ProviderEvent,
        at,
        format!("provider.{}", wire_event.event_name()),
        json!({
            "turn_boundary_id": turn_event.boundary_id,
            "provider_kind": turn_event
                .payload
                .get("provider_kind")
                .cloned()
                .unwrap_or_else(|| json!(script.provider_kind.clone())),
            "script": turn_event.payload.get("script").cloned().unwrap_or(Value::Null),
            "script_name": script.name.clone(),
            "exchange_index": exchange_index,
            "event_index": event_index,
            "event_name": wire_event.event_name(),
            "wire_at": wire_event.at(),
        }),
    )
}

fn set_runtime_completion_ready_at(event: &mut BoundaryEvent, ready_at: u64) {
    if let Some(completion) = event
        .payload
        .get_mut("runtime_completion")
        .and_then(Value::as_object_mut)
    {
        completion.insert("ready_at".to_string(), json!(ready_at));
    }
}

async fn run_provider_turn_task(
    session: lash::LashSession,
    transport: Arc<ScriptedLlmHttpTransport>,
    provider_kind: String,
    event: BoundaryEvent,
) -> Result<Value, FixedScriptRunnerError> {
    let expected_text = event
        .payload
        .get("text")
        .and_then(Value::as_str)
        .unwrap_or("");
    let expected_turn_index = event
        .payload
        .get("turn_index")
        .and_then(Value::as_u64)
        .unwrap_or(1) as usize;
    let output = session
        .turn(lash::TurnInput::text(format!(
            "Run generated provider turn {}.",
            event.boundary_id
        )))
        .turn_id(event.boundary_id.clone())
        .run()
        .await
        .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
    let assistant_message = output.assistant_message().unwrap_or_default().to_string();
    let read_view = output.result.state.read_view();
    let graph_node_count = output.result.state.session_graph.nodes.len();
    let transcript_message_count = read_view.messages().len();
    let provider_exchange_count = transport_exchanges(transport.as_ref())?.len();
    let graph_invariant = runtime_graph_invariant_facts(&output.result.state.session_graph);
    let agent_frame_invariant = runtime_agent_frame_invariant_facts(&output.result.state);
    let usage_invariant = runtime_usage_invariant_facts(&output.result, &output.activities);
    let final_value_invariant =
        runtime_final_value_invariant_facts(&output.result, &output.activities);
    if !output.is_success() {
        return Err(FixedScriptRunnerError::Assertion(format!(
            "runtime turn `{}` did not succeed; turn_index={} outcome={:?} activities={:?}",
            event.boundary_id,
            output.result.state.turn_index,
            output.result.outcome,
            output.activities
        )));
    }
    let observation = RuntimeTurnObservation {
        session_id: output.result.state.session_id.clone(),
        turn_index: output.result.state.turn_index,
        assistant_message: assistant_message.clone(),
        graph_node_count,
        transcript_message_count,
        activity_count: output.activities.len(),
        provider_exchange_count,
        graph_invariant: Some(graph_invariant.clone()),
        agent_frame_invariant: Some(agent_frame_invariant.clone()),
        usage_invariant: Some(usage_invariant.clone()),
    };
    let expected_exchange_count = event
        .payload
        .get("expected_provider_exchange_count")
        .and_then(Value::as_u64)
        .unwrap_or(expected_turn_index as u64) as usize;
    let runtime_contract = runtime_turn_contract(
        &observation,
        &event.actor_alias,
        expected_turn_index,
        expected_text,
        expected_exchange_count,
    );
    if let Err(message) = require_passed(&runtime_contract) {
        return Err(FixedScriptRunnerError::Assertion(format!(
            "runtime invariants failed for `{}`: {message}; success={} session_id={} turn_index={} graph_nodes={} transcript_messages={} activities={}",
            event.boundary_id,
            output.is_success(),
            output.result.state.session_id,
            output.result.state.turn_index,
            graph_node_count,
            transcript_message_count,
            output.activities.len()
        )));
    }
    Ok(json!({
        "session": event.actor_alias,
        "runtime_session_id": event.actor_alias,
        "turn_index": expected_turn_index,
        "success": true,
        "provider_output": assistant_message,
        "provider_script": event.payload.get("script").cloned().unwrap_or(Value::Null),
        "provider_exchange_count": provider_exchange_count,
        "graph_node_count": graph_node_count,
        "transcript_message_count": transcript_message_count,
        "activity_count_nonzero": !output.activities.is_empty(),
        "provider_kind": provider_kind,
        "runtime_invariants": {
            "session_id": true,
            "turn_index": true,
            "graph_non_empty": graph_node_count > 0,
            "graph_acyclic": graph_invariant.passed,
            "single_active_agent_frame": agent_frame_invariant.passed,
            "usage_monotonic": usage_invariant.passed,
            "transcript_contains_provider_output": read_view.messages().iter().any(|message| {
                message.parts.iter().any(|part| part.content.contains(expected_text))
            }),
            "activity_count_nonzero": !output.activities.is_empty(),
        },
        "runtime_invariant_facts": {
            "graph": graph_invariant,
            "agent_frame": agent_frame_invariant,
            "usage": usage_invariant,
        },
        "runtime_final_value_facts": final_value_invariant,
        "runtime_contract": runtime_contract,
    }))
}

#[derive(Default)]
struct SimBackendFaultInjector {
    attempts_by_operation: BTreeMap<String, usize>,
}

impl SimBackendFaultInjector {
    fn inject(&mut self, event: &BoundaryEvent) -> Value {
        let operation = event
            .payload
            .get("operation")
            .and_then(Value::as_str)
            .unwrap_or("backend_operation")
            .to_string();
        let attempts = self
            .attempts_by_operation
            .entry(operation.clone())
            .or_insert(0);
        *attempts += 1;
        let retryable = event
            .payload
            .get("retryable")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        backend_fault_observation(
            event
                .payload
                .get("session")
                .cloned()
                .unwrap_or_else(|| json!(event.actor_alias)),
            operation,
            *attempts,
            retryable,
        )
    }
}

#[derive(Default)]
struct SimProviderMutationHarness {
    rejected_mutations: BTreeSet<String>,
    matrix_cache: ProviderMutationMatrixCache,
}

impl SimProviderMutationHarness {
    async fn reject(&mut self, event: &BoundaryEvent) -> Result<Value, FixedScriptRunnerError> {
        let mutation = event
            .payload
            .get("mutation")
            .and_then(Value::as_str)
            .unwrap_or("unknown_mutation")
            .to_string();
        let mutation_key = format!("{}:{mutation}", event.actor_alias);
        let first_rejection = self.rejected_mutations.insert(mutation_key);
        let observed = json!({
            "session": event.actor_alias,
            "provider_mutation": true,
            "mutation": mutation,
            "rejected": true,
            "first_rejection": first_rejection,
            "oracle": event.payload.get("oracle").cloned().unwrap_or(Value::Null),
        });
        self.matrix_cache
            .augment_observation(event, observed)
            .await
            .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))
    }
}

struct SimTriggerHarness {
    store: Arc<lash_core::InMemoryTriggerStore>,
    registered_source_keys: BTreeSet<String>,
}

impl Default for SimTriggerHarness {
    fn default() -> Self {
        Self {
            store: Arc::new(lash_core::InMemoryTriggerStore::default()),
            registered_source_keys: BTreeSet::new(),
        }
    }
}

impl SimTriggerHarness {
    async fn deliver(&mut self, event: &BoundaryEvent) -> Result<Value, FixedScriptRunnerError> {
        let session = event
            .payload
            .get("session")
            .and_then(Value::as_str)
            .unwrap_or(&event.actor_alias)
            .to_string();
        let source_key = event
            .payload
            .get("source_key")
            .and_then(Value::as_str)
            .unwrap_or(&event.boundary_id)
            .to_string();
        let source_type = "sim.trigger";
        if self.registered_source_keys.insert(source_key.clone()) {
            let draft = lash_core::TriggerSubscriptionDraft::for_process(
                lash_core::ProcessOriginator::session(lash_core::SessionScope::new(
                    session.clone(),
                )),
                lash_core::ProcessExecutionEnvRef::new("process-env:sim-trigger"),
                source_type,
                source_key.clone(),
                lash_core::ProcessInput::External {
                    metadata: json!({
                        "trigger_boundary": event.boundary_id,
                    }),
                },
                lash_core::ProcessIdentity::new("sim-trigger").with_label(Some("sim trigger")),
            )
            .with_wake_target(lash_core::SessionScope::new(session.clone()));
            self.store
                .register_subscription(draft)
                .await
                .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
        }
        let occurrence = self
            .store
            .record_occurrence(
                lash_core::TriggerOccurrenceRequest::new(
                    source_type,
                    source_key.clone(),
                    json!({
                        "boundary_id": event.boundary_id,
                        "session": session,
                    }),
                    format!("sim-trigger:{}", event.boundary_id),
                )
                .with_source(json!({"sim": true})),
            )
            .await
            .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
        let reservations = self
            .store
            .reserve_matching_deliveries(&occurrence.occurrence_id)
            .await
            .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
        Ok(json!({
            "session": session,
            "trigger_delivered": true,
            "source_key": source_key,
            "occurrence_id": occurrence.occurrence_id,
            "reservation_count": reservations.len(),
            "started_process": event.payload.get("started_process").cloned().unwrap_or(Value::Bool(true)),
        }))
    }
}

fn write_provider_script_manifest(
    artifact_root: &Path,
    fixed_manifest: &FixedScriptManifest,
) -> Result<(), FixedScriptRunnerError> {
    let manifest = ProviderScriptProfileManifest {
        schema: "lash.sim.provider-script-manifest.v1",
        fixed_script_manifest: "provider-corpus/fixed-script-manifest.json".to_string(),
        fixed_script_summary: "provider-corpus/summary.json".to_string(),
        script_bundle_hash: fixed_manifest.script_bundle_hash.clone(),
        scripts: fixed_manifest.scripts.clone(),
        provider_matrix: fixed_manifest.provider_matrix.clone(),
        provider_transport_exclusions: fixed_manifest.provider_transport_exclusions.clone(),
    };
    std::fs::write(
        artifact_root.join(GENERATED_SIM_PROVIDER_MANIFEST),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(())
}

fn write_failure_artifact_shape(artifact_root: &Path) -> Result<(), FixedScriptRunnerError> {
    let path = artifact_root.join(GENERATED_SIM_FAILURE_SHAPE);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let shape = FailureArtifactShape {
        schema: "lash.sim.failure-artifact-shape.v1",
        directory: "failures/<failure-id>/",
        trace: "trace.json",
        replay_report: "replay.json",
        oracle: "oracle.json",
        final_summary: "final-summary.json",
    };
    std::fs::write(path, serde_json::to_vec_pretty(&shape)?)?;
    Ok(())
}

fn runtime_core_for_scripts(
    scripts: Vec<ProviderWireScript>,
    store_factory: Arc<dyn SessionStoreFactory>,
    attachment_store: Arc<dyn lash::persistence::AttachmentStore>,
    process_env_store: Arc<dyn lash::persistence::ProcessExecutionEnvStore>,
    provider_schedule: Option<ScriptedTransportSchedule>,
    disable_inline_queued_work_driver: bool,
) -> Result<(lash::StandardCore, Arc<ScriptedLlmHttpTransport>, String), FixedScriptRunnerError> {
    let provider_kind = scripts
        .first()
        .ok_or_else(|| {
            FixedScriptRunnerError::Assertion(
                "runtime core requires at least one script".to_string(),
            )
        })?
        .provider_kind
        .clone();
    if scripts
        .iter()
        .any(|script| script.provider_kind != provider_kind)
    {
        return Err(FixedScriptRunnerError::Assertion(
            "runtime provider scripts for a session must use one provider kind".to_string(),
        ));
    }
    let mut transport = ScriptedLlmHttpTransport::from_scripts(scripts);
    if let Some(schedule) = provider_schedule {
        transport = transport.with_event_schedule(schedule);
    }
    let transport = Arc::new(transport);
    let (provider_handle, model, provider_kind) =
        runtime_provider_components(&provider_kind, &transport)
            .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
    let mut builder = lash::StandardCore::builder()
        .effect_host(Arc::new(lash::durability::InlineEffectHost::default()))
        .attachment_store(attachment_store)
        .process_env_store(process_env_store)
        .store_factory(store_factory)
        .provider(provider_handle)
        .model(model);
    if disable_inline_queued_work_driver {
        builder = builder.disable_queued_work_driver();
    }
    let core = builder
        .build()
        .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
    Ok((core, transport, provider_kind))
}

async fn prove_runtime_facade_turn() -> Result<RuntimeFacadeProof, FixedScriptRunnerError> {
    let script = runtime_script_for_text(OPENAI_COMPATIBLE, "Runtime scripted answer.")
        .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
    let transport = Arc::new(ScriptedLlmHttpTransport::from_scripts([script]));
    let (provider_handle, model, provider_kind) =
        runtime_provider_components(OPENAI_COMPATIBLE, &transport)
            .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
    let core = lash::StandardCore::builder()
        .effect_host(Arc::new(lash::durability::InlineEffectHost::default()))
        .attachment_store(Arc::new(lash::persistence::InMemoryAttachmentStore::new()))
        .process_env_store(Arc::new(
            lash::persistence::InMemoryProcessExecutionEnvStore::new(),
        ))
        .provider(provider_handle)
        .model(model)
        .build()
        .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
    let session = core
        .session("sim-runtime-session")
        .open_fresh()
        .await
        .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
    let output = session
        .turn(lash::TurnInput::text("Run the scripted runtime proof."))
        .run()
        .await
        .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
    let assistant_message = output.assistant_message().unwrap_or_default().to_string();
    let runtime_ok = output.is_success()
        && output.result.state.session_id == "sim-runtime-session"
        && output.result.state.turn_index == 1;
    let provider_ok = assistant_message == "Runtime scripted answer.";
    let provider_exchange_count = transport_exchanges(transport.as_ref())?.len();
    require(
        runtime_ok,
        "runtime facade turn did not finish with the expected session state",
    )?;
    require(
        provider_ok && provider_exchange_count == 1,
        "runtime facade turn did not consume the expected scripted provider output",
    )?;
    let pending_tool_completion = prove_pending_tool_completion_through_turn().await?;
    let final_value_semantic_channel = prove_final_value_semantic_channel().await?;
    Ok(RuntimeFacadeProof {
        schema: "lash.sim.runtime-facade-proof.v1",
        name: "standard-facade-openai-compatible-scripted-turn",
        provider_kind,
        session_id: output.result.state.session_id,
        turn_index: output.result.state.turn_index,
        assistant_message,
        provider_exchange_count,
        runtime_invariant: runtime_provider_turn(
            runtime_ok,
            "turn finished once and advanced the expected session state",
        ),
        provider_output_invariant: runtime_provider_turn(
            provider_ok,
            "provider output matched the scripted OpenAI-compatible stream",
        ),
        pending_tool_completion,
        final_value_semantic_channel,
    })
}

/// The (provider kind, valid-prose-deltas-before-fault) combos exercised for the
/// live-provider-failure oracle EVERY seed. Covers more than one provider kind
/// and more than one fault position so `live_provider_failure_coverage` cannot
/// pass vacuously on a single degenerate case.
const LIVE_PROVIDER_FAILURE_COMBOS: &[(&str, usize)] = &[
    (OPENAI_COMPATIBLE, 1),
    (OPENAI_COMPATIBLE, 2),
    (ANTHROPIC, 1),
];

/// Drive every live-provider-failure combo for a seed, collecting the observed
/// facts for the per-seed coverage oracle.
async fn drive_live_provider_failure_turns(
    seed: u64,
) -> Result<Vec<LiveProviderFailureFacts>, FixedScriptRunnerError> {
    let mut facts = Vec::with_capacity(LIVE_PROVIDER_FAILURE_COMBOS.len());
    for (provider_kind, prose_deltas) in LIVE_PROVIDER_FAILURE_COMBOS.iter().copied() {
        let script = live_failure_script(provider_kind, prose_deltas)
            .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
        facts.push(
            run_live_turn_facts(
                seed,
                provider_kind,
                script,
                "malformed_sse_chunk",
                prose_deltas,
            )
            .await?,
        );
    }
    Ok(facts)
}

/// Run a real `session.turn().run()` against `script`, releasing its
/// scripted-transport SSE events through a REAL `BoundaryScheduler` (the same
/// provider-event release path generated turns use — NOT an ad-hoc index loop),
/// and record whether the turn terminalized without committing any output.
/// Shared by the failure driver and by the end-to-end negative test (which feeds
/// a SUCCESS script to prove the committed-output assertion bites).
async fn run_live_turn_facts(
    seed: u64,
    provider_kind: &str,
    script: ProviderWireScript,
    fault_kind: &str,
    offered_prose_deltas: usize,
) -> Result<LiveProviderFailureFacts, FixedScriptRunnerError> {
    let schedule = ScriptedTransportSchedule::new();
    let transport = Arc::new(
        ScriptedLlmHttpTransport::from_scripts([script.clone()])
            .with_event_schedule(schedule.clone()),
    );
    let (provider_handle, model, provider_kind) =
        runtime_provider_components(provider_kind, &transport)
            .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
    let core = lash::StandardCore::builder()
        .effect_host(Arc::new(lash::durability::InlineEffectHost::default()))
        .attachment_store(Arc::new(lash::persistence::InMemoryAttachmentStore::new()))
        .process_env_store(Arc::new(
            lash::persistence::InMemoryProcessExecutionEnvStore::new(),
        ))
        .store_factory(Arc::new(
            lash::persistence::InMemorySessionStoreFactory::new(),
        ))
        .provider(provider_handle)
        .model(model)
        .build()
        .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
    let session_id = format!("sim-live-failure-{provider_kind}-{offered_prose_deltas}");
    let session = core
        .session(session_id.clone())
        .open_fresh()
        .await
        .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;

    let events = Arc::new(RuntimeProofRecordingEvents::default());
    let turn_session = session.clone();
    let turn_events = Arc::clone(&events);
    let turn = tokio::spawn(async move {
        turn_session
            .turn(lash::TurnInput::text("Run the live provider failure turn."))
            .stream_to(turn_events.as_ref())
            .await
    });

    // Schedule the provider-event releases as real boundaries and deliver them
    // through a REAL BoundaryScheduler (seeded), exactly as generated provider
    // turns do.
    let turn_event = BoundaryEvent::new(
        format!("{session_id}:provider:001"),
        session_id.clone(),
        BoundaryKind::Provider,
        0,
        "provider.chat.stream.live-failure",
        json!({ "provider_kind": provider_kind, "turn_index": 1 }),
    );
    let release_boundaries = script
        .timeline
        .iter()
        .enumerate()
        .map(|(event_index, wire_event)| {
            provider_release_boundary(
                &turn_event,
                &script,
                0,
                event_index,
                wire_event,
                wire_event.at(),
            )
        })
        .collect::<Vec<_>>();
    let mut scheduler = BoundaryScheduler::with_events(seed, release_boundaries);

    // Observe the turn live and parked on the first gate before any release.
    let mut turn_was_live_parked = false;
    let mut polls = 0u64;
    loop {
        if schedule.is_blocked(0, 0) {
            turn_was_live_parked = true;
            break;
        }
        if turn.is_finished() || polls >= MAX_PROVIDER_EVENT_POLL_YIELDS {
            break;
        }
        polls += 1;
        tokio::task::yield_now().await;
    }

    // Deliver each release boundary through the BoundaryScheduler; release the gate
    // it names (bounded so it can never hang once the turn terminalizes).
    loop {
        if turn.is_finished() {
            break;
        }
        let Some(delivered) = scheduler.deliver_next(Value::Null) else {
            let mut idle = 0u64;
            while !turn.is_finished() && idle < MAX_PROVIDER_EVENT_POLL_YIELDS {
                idle += 1;
                tokio::task::yield_now().await;
            }
            break;
        };
        let event = delivered.as_event();
        let exchange_index = event
            .payload
            .get("exchange_index")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;
        let event_index = event
            .payload
            .get("event_index")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;
        let event_name = event
            .payload
            .get("event_name")
            .and_then(Value::as_str)
            .unwrap_or("provider_event")
            .to_string();
        if !turn.is_finished() {
            schedule.release(exchange_index, event_index, &event_name, event.at);
        }
    }

    let result = turn.await.map_err(|err| {
        FixedScriptRunnerError::Runtime(format!(
            "live provider failure turn task failed to join: {err}"
        ))
    })?;

    let streamed_prose_deltas = events.assistant_prose_delta_count().await;
    // COMMITTED output is the durable turn result + session transcript, NOT
    // transient stream deltas: a correct runtime may STREAM partial prose and then
    // DISCARD it on terminal failure. We require it commit none of that prose.
    let (terminalized_failure, committed_assistant_message_nonempty, committed_final_values) =
        match &result {
            Ok(turn_result) => (
                !turn_result.is_success(),
                turn_result
                    .assistant_message()
                    .is_some_and(|message| !message.is_empty()),
                usize::from(turn_result.final_value().is_some()),
            ),
            Err(_) => (true, false, 0),
        };
    let committed_prose_in_transcript =
        committed_transcript_contains(&session, LIVE_FAILURE_LEAK_PROSE);
    Ok(LiveProviderFailureFacts {
        provider_kind,
        fault_kind: fault_kind.to_string(),
        offered_prose_deltas,
        streamed_prose_deltas,
        turn_was_live_parked,
        terminalized_failure,
        committed_assistant_message_nonempty,
        committed_final_values,
        committed_prose_in_transcript,
    })
}

/// Whether the session's COMMITTED transcript contains `needle` — used to detect
/// partial prose leaked into durable state on a terminal failure.
fn committed_transcript_contains(session: &lash::LashSession, needle: &str) -> bool {
    let observation = session.observe().current_observation();
    observation.read_view.messages().iter().any(|message| {
        message
            .parts
            .iter()
            .any(|part| part.content.contains(needle))
    })
}

async fn prove_pending_tool_completion_through_turn()
-> Result<PendingToolCompletionProof, FixedScriptRunnerError> {
    let (key_tx, key_rx) = tokio::sync::oneshot::channel();
    let events = Arc::new(RuntimeProofRecordingEvents::default());
    let core = lash::StandardCore::builder()
        .effect_host(Arc::new(lash::durability::InlineEffectHost::default()))
        .attachment_store(Arc::new(lash::persistence::InMemoryAttachmentStore::new()))
        .process_env_store(Arc::new(
            lash::persistence::InMemoryProcessExecutionEnvStore::new(),
        ))
        .store_factory(Arc::new(
            lash::persistence::InMemorySessionStoreFactory::new(),
        ))
        .process_registry(Arc::new(lash_core::TestLocalProcessRegistry::default())
            as Arc<dyn lash_core::ProcessRegistry>)
        .provider(pending_tool_roundtrip_provider())
        .model(
            lash_core::ModelSpec::from_token_limits("mock-model", None, 200_000, None)
                .map_err(FixedScriptRunnerError::Assertion)?,
        )
        .tools(Arc::new(PendingToolProvider::new(key_tx)) as Arc<dyn lash_core::ToolProvider>)
        .build()
        .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
    let session = core
        .session("sim-pending-tool-session")
        .open_fresh()
        .await
        .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
    let turn_session = session.clone();
    let turn_events = Arc::clone(&events);
    let turn = tokio::spawn(async move {
        turn_session
            .turn(lash::TurnInput::text("use async tool"))
            .stream_to(turn_events.as_ref())
            .await
    });

    let key = key_rx.await.map_err(|_| {
        FixedScriptRunnerError::Runtime("pending tool did not send completion key".to_string())
    })?;
    for _ in 0..8 {
        tokio::task::yield_now().await;
    }
    let completed_before = events.tool_completed_count().await;
    let suspended_before_completion = !turn.is_finished() && completed_before == 0;
    require(
        suspended_before_completion,
        "pending tool turn completed before the scheduler-delivered completion boundary",
    )?;

    let resolved_payload = json!({
        "ok": true,
        "async": true,
        "resolved_by": "lash-sim-boundary-scheduler",
    });
    let completion_boundary_id = "sim-pending-tool-session:tool-completion:001";
    let mut scheduler = BoundaryScheduler::with_events(
        0x5eed_7001,
        [BoundaryEvent::new(
            completion_boundary_id,
            "sim-pending-tool-session",
            BoundaryKind::Tool,
            1,
            "tool.pending-completion.resolve",
            json!({
                "tool": "app_lookup",
                "resolution": resolved_payload,
                "completion_key_observed": true,
            }),
        )],
    );
    let mut delivered = scheduler.deliver_next(Value::Null).ok_or_else(|| {
        FixedScriptRunnerError::Assertion(
            "pending tool completion boundary was not scheduled".to_string(),
        )
    })?;
    let event = delivered.as_event();
    let resolution = event.payload.get("resolution").cloned().ok_or_else(|| {
        FixedScriptRunnerError::Assertion(
            "pending tool boundary missing resolution payload".to_string(),
        )
    })?;
    let accepted = core
        .completions()
        .resolve(key.clone(), lash_core::Resolution::Ok(resolution.clone()))
        .await
        .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
    delivered.observed = json!({
        "session": event.actor_alias,
        "tool": event.payload.get("tool").cloned().unwrap_or(Value::Null),
        "scheduler_delivered_tool_completion": true,
        "completion_key_observed": event.payload.get("completion_key_observed").cloned().unwrap_or(Value::Bool(false)),
        "resolve_outcome": accepted.clone(),
    });

    let result = turn
        .await
        .map_err(|err| {
            FixedScriptRunnerError::Runtime(format!("pending tool turn task failed: {err}"))
        })?
        .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
    let completed_after = events.tool_completed_count().await;
    let completed_outputs = events.tool_completed_outputs().await;
    let assistant_message = result.assistant_message().unwrap_or_default().to_string();
    let final_ok = matches!(
        &result.outcome,
        lash_core::TurnOutcome::Finished(lash_core::TurnFinish::AssistantMessage { .. })
    ) && assistant_message == "done"
        && completed_after > completed_before
        && completed_outputs
            .iter()
            .any(|(name, output)| name == "app_lookup" && output == &resolution);
    require(
        final_ok,
        "pending tool completion did not resume the turn to the scripted final answer",
    )?;
    let duplicate = core
        .completions()
        .resolve(
            key,
            lash_core::Resolution::Ok(json!({"ok": false, "duplicate": true})),
        )
        .await
        .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
    let session_id = result.state.session_id.clone();
    let turn_index = result.state.turn_index;

    Ok(PendingToolCompletionProof {
        schema: "lash.sim.pending-tool-completion-proof.v1",
        name: "standard-turn-pending-tool-scheduler-resolution",
        session_id,
        turn_index,
        assistant_message,
        tool_name: "app_lookup".to_string(),
        completion_boundary_id: delivered.boundary_id.clone(),
        scheduler_controlled: delivered.scheduler.scheduler_controlled,
        scheduler_sequence: delivered.sequence,
        turn_suspended_before_completion: suspended_before_completion,
        completed_event_count_before_resolution: completed_before,
        completed_event_count_after_resolution: completed_after,
        resolved_payload: resolution.clone(),
        completion_outcome: accepted.clone(),
        duplicate_completion_outcome: duplicate,
        turn_suspension_invariant: pending_tool_completion(
            suspended_before_completion,
            "pending ToolResult parked the live turn before external resolution",
        ),
        scheduler_resolution_invariant: pending_tool_completion(
            delivered.kind == BoundaryKind::Tool
                && delivered.scheduler.scheduler_controlled
                && matches!(accepted, lash_core::ResolveOutcome::Accepted),
            "BoundaryScheduler delivered the Tool boundary that resolved the await key",
        ),
        final_result_invariant: pending_tool_completion(
            final_ok,
            "tool completion produced ToolCallCompleted evidence and the second provider response finalized the turn",
        ),
    })
}

#[derive(Default)]
struct RuntimeProofRecordingEvents {
    events: tokio::sync::Mutex<Vec<lash::TurnActivity>>,
}

impl RuntimeProofRecordingEvents {
    async fn snapshot(&self) -> Vec<lash::TurnActivity> {
        self.events.lock().await.clone()
    }

    async fn tool_completed_count(&self) -> usize {
        self.events
            .lock()
            .await
            .iter()
            .filter(|activity| matches!(activity.event, lash::TurnEvent::ToolCallCompleted { .. }))
            .count()
    }

    async fn tool_completed_outputs(&self) -> Vec<(String, Value)> {
        self.events
            .lock()
            .await
            .iter()
            .filter_map(|activity| match &activity.event {
                lash::TurnEvent::ToolCallCompleted { name, output, .. } => {
                    Some((name.clone(), output.value_for_projection()))
                }
                _ => None,
            })
            .collect()
    }

    async fn final_value_events(&self) -> Vec<Value> {
        self.events
            .lock()
            .await
            .iter()
            .filter_map(|activity| match &activity.event {
                lash::TurnEvent::FinalValue { value } => Some(value.clone()),
                _ => None,
            })
            .collect()
    }

    async fn assistant_prose_delta_count(&self) -> usize {
        self.events
            .lock()
            .await
            .iter()
            .filter(|activity| {
                matches!(activity.event, lash::TurnEvent::AssistantProseDelta { .. })
            })
            .count()
    }
}

#[async_trait::async_trait]
impl lash::TurnActivitySink for RuntimeProofRecordingEvents {
    async fn emit(&self, activity: lash::TurnActivity) {
        self.events.lock().await.push(activity);
    }
}

async fn prove_final_value_semantic_channel()
-> Result<FinalValueSemanticProof, FixedScriptRunnerError> {
    let events = Arc::new(RuntimeProofRecordingEvents::default());
    let core = lash::RlmCore::builder()
        .effect_host(Arc::new(lash::durability::InlineEffectHost::default()))
        .lashlang_artifact_store(Arc::new(
            lash::persistence::InMemoryLashlangArtifactStore::new(),
        ))
        .attachment_store(Arc::new(lash::persistence::InMemoryAttachmentStore::new()))
        .process_env_store(Arc::new(
            lash::persistence::InMemoryProcessExecutionEnvStore::new(),
        ))
        .store_factory(Arc::new(
            lash::persistence::InMemorySessionStoreFactory::new(),
        ))
        .process_registry(Arc::new(lash_core::TestLocalProcessRegistry::default())
            as Arc<dyn lash_core::ProcessRegistry>)
        .provider(rlm_final_value_provider())
        .model(
            lash_core::ModelSpec::from_token_limits("mock-rlm-final-value", None, 200_000, None)
                .map_err(FixedScriptRunnerError::Assertion)?,
        )
        .build()
        .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
    let session = core
        .session("sim-final-value-session")
        .open_fresh()
        .await
        .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
    let result = session
        .turn(lash::TurnInput::text("produce a semantic final value"))
        .require_finish()
        .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?
        .stream_to(events.as_ref())
        .await
        .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
    let final_value = result.final_value().cloned().ok_or_else(|| {
        FixedScriptRunnerError::Assertion(format!(
            "final-value proof finished without TurnFinish::FinalValue: {:?}",
            result.outcome
        ))
    })?;
    let recorded = events.snapshot().await;
    let final_value_events = events.final_value_events().await;
    let assistant_prose_delta_count = events.assistant_prose_delta_count().await;
    let facts = runtime_final_value_invariant_facts(&result, &recorded);
    let transcript_text = result
        .state
        .read_view()
        .messages()
        .iter()
        .flat_map(|message| message.parts.iter())
        .map(|part| part.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let transcript_contains_final_value =
        transcript_text.contains("semantic-channel") || transcript_text.contains("\"count\"");
    let semantic_ok = facts.passed
        && facts.outcome_kind == "final_value"
        && facts.semantic_value.as_ref() == Some(&final_value)
        && final_value_events.iter().any(|value| value == &final_value)
        && !facts.transcript_inference_required
        && result.assistant_message().is_none()
        && !transcript_contains_final_value;
    require(
        semantic_ok,
        "final-value proof did not observe a semantic TurnOutcome and FinalValue event",
    )?;
    Ok(FinalValueSemanticProof {
        schema: "lash.sim.final-value-semantic-proof.v1",
        name: "rlm-turn-final-value-semantic-channel",
        session_id: result.state.session_id,
        turn_index: result.state.turn_index,
        final_value,
        assistant_output_text: result.assistant_output.safe_text,
        final_value_event_count: final_value_events.len(),
        assistant_prose_delta_count,
        facts,
        semantic_channel_invariant: runtime_final_value_semantic(
            semantic_ok,
            "final value was read from TurnFinish::FinalValue and TurnEvent::FinalValue, not transcript prose",
        ),
    })
}

fn rlm_final_value_provider() -> ProviderHandle {
    const RAW_FINAL: &str = "Visible prose before semantic value.\n<lashlang>\nfinish { source: \"semantic-channel\", ok: true, count: 3 }\n</lashlang>";
    const CHUNKS: &[&str] = &[
        "Visible prose",
        " before semantic value.\n<lash",
        "lang>\nfinish { source: ",
        "\"semantic-channel\", ok: true, count: 3 }",
        "\n</lashlang>",
    ];
    lash_core::testing::TestProvider::builder()
        .kind("lash-sim-rlm-final-value")
        .requires_streaming(true)
        .complete(|request| async move {
            let stream = request.stream_events.ok_or_else(|| {
                LlmTransportError::new("rlm final-value proof requires provider streaming")
            })?;
            for chunk in CHUNKS {
                stream.send(LlmStreamEvent::Delta((*chunk).to_string()));
            }
            let response = text_llm_response(RAW_FINAL);
            if response.full_text != RAW_FINAL || response_text_part(&response) != Some(RAW_FINAL)
            {
                return Err(LlmTransportError::new(format!(
                    "rlm final-value fixed response shape changed: expected {:?}, got full_text {:?} parts {:?}",
                    RAW_FINAL, response.full_text, response.parts
                )));
            }
            Ok(response)
        })
        .build()
        .into_handle()
}

struct PendingToolProvider {
    key_tx: Mutex<Option<tokio::sync::oneshot::Sender<lash_core::AwaitEventKey>>>,
}

impl PendingToolProvider {
    fn new(key_tx: tokio::sync::oneshot::Sender<lash_core::AwaitEventKey>) -> Self {
        Self {
            key_tx: Mutex::new(Some(key_tx)),
        }
    }
}

#[async_trait::async_trait]
impl lash_core::ToolProvider for PendingToolProvider {
    fn tool_manifests(&self) -> Vec<lash_core::ToolManifest> {
        vec![pending_tool_definition().manifest()]
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<lash_core::ToolContract>> {
        (name == "app_lookup").then(|| Arc::new(pending_tool_definition().contract()))
    }

    async fn execute(&self, call: lash_core::ToolCall<'_>) -> lash_core::ToolResult {
        if call.name != "app_lookup" {
            return lash_core::ToolResult::err_fmt(format_args!("unknown tool {}", call.name));
        }
        let key = match call.context.completion_key().await {
            Ok(key) => key,
            Err(err) => return lash_core::ToolResult::err_fmt(err),
        };
        if let Some(tx) = self.key_tx.lock().expect("pending tool key sender").take() {
            let _ = tx.send(key);
        }
        lash_core::ToolResult::pending(lash_core::PendingCompletion::new())
    }
}

fn pending_tool_definition() -> lash_core::ToolDefinition {
    lash_core::ToolDefinition::raw(
        "tool:app_lookup",
        "app_lookup",
        "Look up app state.",
        json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false
        }),
        json!({ "type": "object" }),
    )
}

fn pending_tool_roundtrip_provider() -> ProviderHandle {
    let responses = Arc::new(tokio::sync::Mutex::new(VecDeque::from([
        tool_call_llm_response("call-1", "app_lookup", "{}"),
        text_llm_response("done"),
    ])));
    lash_core::testing::TestProvider::builder()
        .kind("lash-sim-pending-tool")
        .complete(move |_request| {
            let responses = Arc::clone(&responses);
            async move {
                responses.lock().await.pop_front().ok_or_else(|| {
                    LlmTransportError::new("pending tool roundtrip provider exhausted")
                })
            }
        })
        .build()
        .into_handle()
}

/// A sim tool that registers its await key in a shared slot the generated world
/// can read, then returns `ToolResult::pending` so the calling turn parks until
/// the scheduler resolves the key. Generalizes `PendingToolProvider` for the
/// generated suspend sessions (Tool / DurableEffect / ExecCode).
struct SuspendToolProvider {
    tool_name: String,
    key_slot: Arc<tokio::sync::Mutex<Option<lash_core::AwaitEventKey>>>,
}

impl SuspendToolProvider {
    fn new(
        tool_name: String,
        key_slot: Arc<tokio::sync::Mutex<Option<lash_core::AwaitEventKey>>>,
    ) -> Self {
        Self {
            tool_name,
            key_slot,
        }
    }

    fn definition(&self) -> lash_core::ToolDefinition {
        lash_core::ToolDefinition::raw(
            format!("tool:{}", self.tool_name),
            self.tool_name.clone(),
            "Await an externally-resolved completion.",
            json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
            json!({ "type": "object" }),
        )
    }
}

#[async_trait::async_trait]
impl lash_core::ToolProvider for SuspendToolProvider {
    fn tool_manifests(&self) -> Vec<lash_core::ToolManifest> {
        vec![self.definition().manifest()]
    }

    fn resolve_contract(&self, name: &str) -> Option<Arc<lash_core::ToolContract>> {
        (name == self.tool_name).then(|| Arc::new(self.definition().contract()))
    }

    async fn execute(&self, call: lash_core::ToolCall<'_>) -> lash_core::ToolResult {
        if call.name != self.tool_name {
            return lash_core::ToolResult::err_fmt(format_args!("unknown tool {}", call.name));
        }
        let key = match call.context.completion_key().await {
            Ok(key) => key,
            Err(err) => return lash_core::ToolResult::err_fmt(err),
        };
        *self.key_slot.lock().await = Some(key);
        lash_core::ToolResult::pending(lash_core::PendingCompletion::new())
    }
}

fn generated_seed(profile: &str, seed_index: usize) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(profile.as_bytes());
    hasher.update(seed_index.to_le_bytes());
    let digest = hasher.finalize();
    let mut bytes = [0_u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    u64::from_le_bytes(bytes)
}

fn file_sha256(path: &Path) -> Result<String, FixedScriptRunnerError> {
    Ok(sha256_hex(&std::fs::read(path)?))
}

fn relative_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace(std::path::MAIN_SEPARATOR, "/")
}

fn lane_summary_for_manifest(manifest: &FixedScriptManifest) -> FixedScriptLaneSummary {
    let mut provider_set = manifest
        .scripts
        .iter()
        .map(|script| script.provider_kind.clone())
        .collect::<Vec<_>>();
    provider_set.sort();
    provider_set.dedup();

    FixedScriptLaneSummary {
        schema: "lash.sim.summary.v1",
        profile: FIXED_SCRIPT_PROFILE,
        generator_version: "fixed-provider-scripts.v1",
        replay_schema_version: manifest.schema,
        script_bundle_hash: manifest.script_bundle_hash.clone(),
        provider_set,
        backend_set: Vec::new(),
        counts: FixedScriptLaneCounts {
            generated_seeds: 0,
            fixed_script_events: manifest.event_count,
            fixed_replays: manifest.summary.total_proofs,
            minimized_replays: 0,
            backend_replays: 0,
            boundary_events: 0,
            oracle_passes: manifest.summary.passed,
            oracle_failures: manifest.summary.total_proofs - manifest.summary.passed,
            divergences: 0,
        },
        fixed_script_manifest: FIXED_SCRIPT_MANIFEST,
        fixed_script_events: FIXED_SCRIPT_EVENTS,
    }
}

fn provider_matrix(
    scripts: &[ScriptHashManifest],
    proof_runs: &[ProofRun],
) -> Vec<ProviderMatrixRow> {
    let mut by_provider: BTreeMap<String, ProviderMatrixRow> = BTreeMap::new();
    for script in scripts {
        let row = by_provider
            .entry(script.provider_kind.clone())
            .or_insert_with(|| ProviderMatrixRow {
                provider_kind: script.provider_kind.clone(),
                script_names: Vec::new(),
                proof_names: Vec::new(),
                endpoints: Vec::new(),
                success_proofs: 0,
                error_proofs: 0,
                cancelled_proofs: 0,
            });
        row.script_names.push(script.name.clone());
    }
    for proof in proof_runs {
        let row = by_provider
            .entry(proof.provider_kind.clone())
            .or_insert_with(|| ProviderMatrixRow {
                provider_kind: proof.provider_kind.clone(),
                script_names: Vec::new(),
                proof_names: Vec::new(),
                endpoints: Vec::new(),
                success_proofs: 0,
                error_proofs: 0,
                cancelled_proofs: 0,
            });
        row.proof_names.push(proof.name.clone());
        row.endpoints.push(proof.endpoint.clone());
        match proof.transcript.terminal.classification {
            "success" => row.success_proofs += 1,
            "error" => row.error_proofs += 1,
            "cancelled_before_response_start" => row.cancelled_proofs += 1,
            _ => {}
        }
    }
    by_provider
        .into_values()
        .map(|mut row| {
            row.script_names.sort();
            row.script_names.dedup();
            row.proof_names.sort();
            row.proof_names.dedup();
            row.endpoints.sort();
            row.endpoints.dedup();
            row
        })
        .collect()
}

fn generated_runtime_provider_matrix(
    event_lines: &[TraceEventLine],
) -> Vec<GeneratedRuntimeProviderMatrixRow> {
    let mut by_provider: BTreeMap<String, GeneratedRuntimeProviderMatrixRow> = BTreeMap::new();
    for line in event_lines {
        match line.event.kind {
            BoundaryKind::Ingress => {
                let Some(provider_kind) = line
                    .event
                    .payload
                    .get("provider_kind")
                    .and_then(Value::as_str)
                else {
                    continue;
                };
                let script = line
                    .event
                    .payload
                    .get("provider_script")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_string();
                let row = by_provider
                    .entry(provider_kind.to_string())
                    .or_insert_with(|| GeneratedRuntimeProviderMatrixRow {
                        provider_kind: provider_kind.to_string(),
                        script_names: Vec::new(),
                        runtime_session_count: 0,
                        runtime_provider_turn_count: 0,
                    });
                row.runtime_session_count += 1;
                row.script_names.push(script);
            }
            BoundaryKind::Provider => {
                let Some(provider_kind) = line
                    .event
                    .payload
                    .get("provider_kind")
                    .and_then(Value::as_str)
                else {
                    continue;
                };
                let script = line
                    .event
                    .payload
                    .get("script")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_string();
                let row = by_provider
                    .entry(provider_kind.to_string())
                    .or_insert_with(|| GeneratedRuntimeProviderMatrixRow {
                        provider_kind: provider_kind.to_string(),
                        script_names: Vec::new(),
                        runtime_session_count: 0,
                        runtime_provider_turn_count: 0,
                    });
                row.runtime_provider_turn_count += 1;
                row.script_names.push(script);
            }
            _ => {}
        }
    }
    by_provider
        .into_values()
        .map(|mut row| {
            row.script_names.sort();
            row.script_names.dedup();
            row
        })
        .collect()
}

fn script_hash_manifest() -> Result<Vec<ScriptHashManifest>, FixedScriptRunnerError> {
    CANONICAL_SCRIPTS
        .iter()
        .map(|entry| {
            let script = ProviderWireScript::from_json_str(entry.content)?;
            Ok(ScriptHashManifest {
                path: entry.path.to_string(),
                name: script.name,
                provider_kind: script.provider_kind,
                sha256: sha256_hex(entry.content.as_bytes()),
                bytes: entry.content.len(),
            })
        })
        .collect()
}

fn script_bundle_hash(scripts: &[ScriptHashManifest]) -> String {
    let mut hasher = Sha256::new();
    for script in scripts {
        hasher.update(script.path.as_bytes());
        hasher.update([0]);
        hasher.update(script.sha256.as_bytes());
        hasher.update([0]);
    }
    let digest = hasher.finalize();
    hex_digest(&digest)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    hex_digest(&digest)
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn write_proof_artifacts(
    artifact_root: &Path,
    proof_runs: Vec<ProofRun>,
) -> Result<Vec<FixedScriptProof>, FixedScriptRunnerError> {
    let proof_dir = artifact_root.join("proofs");
    std::fs::create_dir_all(&proof_dir)?;
    proof_runs
        .into_iter()
        .map(|run| {
            let relative_path =
                PathBuf::from("proofs").join(format!("{}.json", proof_file_stem(&run.name)));
            let body = serde_json::to_vec_pretty(&run.transcript)?;
            std::fs::write(artifact_root.join(&relative_path), &body)?;
            Ok(FixedScriptProof {
                name: run.name,
                provider_kind: run.provider_kind,
                endpoint: run.endpoint,
                outcome: run.outcome,
                transcript_path: relative_path
                    .to_string_lossy()
                    .replace(std::path::MAIN_SEPARATOR, "/"),
                transcript_sha256: sha256_hex(&body),
                observed: run.observed,
            })
        })
        .collect()
}

fn fixed_script_events(proof_runs: &[ProofRun]) -> Vec<FixedScriptEvent> {
    let mut sequence = 0;
    let mut events = Vec::new();
    for (proof_index, run) in proof_runs.iter().enumerate() {
        let proof_alias = format!("proof-{proof_index:03}");
        let transcript_path = PathBuf::from("proofs")
            .join(format!("{}.json", proof_file_stem(&run.name)))
            .to_string_lossy()
            .replace(std::path::MAIN_SEPARATOR, "/");
        for (exchange_index, exchange) in run.transcript.http_exchanges.iter().enumerate() {
            events.push(FixedScriptEvent {
                schema: "lash.sim.fixed-script-event.v1",
                sequence,
                event: "fixed_script_exchange",
                profile: FIXED_SCRIPT_PROFILE,
                proof_alias: proof_alias.clone(),
                exchange_alias: format!("{proof_alias}.exchange-{exchange_index:03}"),
                proof_name: run.name.clone(),
                provider_kind: run.provider_kind.clone(),
                outcome: run.outcome.clone(),
                transcript_path: transcript_path.clone(),
                request: FixedScriptEventRequest {
                    method: exchange.request.method.clone(),
                    path: exchange.request.path.clone(),
                    header_names: exchange
                        .request
                        .headers
                        .iter()
                        .map(|header| header.name.clone())
                        .collect(),
                    body_bytes: exchange.request.body_bytes,
                    body_shape: exchange.request.body_shape.clone(),
                },
                response: FixedScriptEventResponse {
                    status: exchange.response.status,
                    header_names: exchange
                        .response
                        .headers
                        .iter()
                        .map(|header| header.name.clone())
                        .collect(),
                    event_names: exchange.response.event_names.clone(),
                },
                terminal_classification: run.transcript.terminal.classification,
            });
            sequence += 1;
        }
    }
    events
}

fn write_events_artifact(
    artifact_root: &Path,
    events: &[FixedScriptEvent],
) -> Result<String, FixedScriptRunnerError> {
    let mut body = Vec::new();
    for event in events {
        body.extend_from_slice(&serde_json::to_vec(event)?);
        body.push(b'\n');
    }
    std::fs::write(artifact_root.join(FIXED_SCRIPT_EVENTS), &body)?;
    Ok(sha256_hex(&body))
}

fn proof_file_stem(name: &str) -> String {
    name.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn transcript_for_script(
    proof_name: &str,
    provider_kind: &str,
    script_content: &str,
    http_exchanges: Vec<ScriptedLlmHttpExchange>,
    terminal: TranscriptTerminal,
    observed: serde_json::Value,
) -> Result<FixedScriptTranscript, FixedScriptRunnerError> {
    let script = ProviderWireScript::from_json_str(script_content)?;
    let body_paths = script.request_match.body.keys().cloned().collect();
    let headers = script.request_match.headers.keys().cloned().collect();
    let response_events = script.timeline.iter().map(transcript_event).collect();
    Ok(FixedScriptTranscript {
        schema: "lash.sim.fixed-script-proof-transcript.v1",
        proof: proof_name.to_string(),
        provider_kind: provider_kind.to_string(),
        endpoint: TranscriptEndpoint {
            method: script.endpoint.method,
            path: script.endpoint.path,
        },
        timeline_at_semantics: FIXED_SCRIPT_TIMELINE_AT_SEMANTICS,
        request_match: TranscriptRequestMatch {
            body_paths,
            headers,
        },
        http_exchanges,
        response_events,
        terminal,
        observed,
    })
}

fn transcript_event(event: &ProviderWireEvent) -> TranscriptResponseEvent {
    match event {
        ProviderWireEvent::ResponseStart {
            at,
            status,
            headers,
        } => TranscriptResponseEvent {
            at: *at,
            event: event.event_name(),
            provider_event: None,
            status: Some(*status),
            headers: transcript_headers(headers),
        },
        ProviderWireEvent::HttpError {
            at,
            status,
            headers,
            ..
        } => TranscriptResponseEvent {
            at: *at,
            event: event.event_name(),
            provider_event: None,
            status: Some(*status),
            headers: transcript_headers(headers),
        },
        ProviderWireEvent::Sse { at, data } => TranscriptResponseEvent {
            at: *at,
            event: event.event_name(),
            provider_event: provider_event_name(data),
            status: None,
            headers: Vec::new(),
        },
        ProviderWireEvent::Body { at, .. }
        | ProviderWireEvent::Chunk { at, .. }
        | ProviderWireEvent::End { at }
        | ProviderWireEvent::Disconnect { at, .. }
        | ProviderWireEvent::Timeout { at, .. }
        | ProviderWireEvent::TransportError { at, .. } => TranscriptResponseEvent {
            at: *at,
            event: event.event_name(),
            provider_event: None,
            status: None,
            headers: Vec::new(),
        },
    }
}

fn transcript_headers(headers: &[ProviderWireHeader]) -> Vec<TranscriptHeader> {
    headers
        .iter()
        .map(|header| TranscriptHeader {
            name: header.name.clone(),
            value: redacted_header_value(&header.name, &header.value),
        })
        .collect()
}

fn provider_event_name(data: &str) -> Option<String> {
    let trimmed = data.trim();
    if trimmed == "[DONE]" {
        return Some("[DONE]".to_string());
    }
    let value: Value = serde_json::from_str(trimmed).ok()?;
    value
        .get("type")
        .and_then(Value::as_str)
        .map(str::to_string)
}

async fn prove_openai_compatible_tool_stream() -> Result<ProofRun, FixedScriptRunnerError> {
    let (mut provider, transport) = openai_compatible_provider(OPENAI_COMPAT_TOOL_CALL)?;
    let response = provider.complete(openai_compatible_request(true)).await?;
    require(
        response.terminal_reason == LlmTerminalReason::ToolUse,
        "OpenAI-compatible tool stream terminal reason was not tool_use",
    )?;
    require(
        response.full_text == "café ",
        "OpenAI-compatible tool stream text did not preserve split UTF-8",
    )?;
    require(
        response.parts.iter().any(|part| {
            matches!(
                part,
                LlmOutputPart::ToolCall {
                    tool_name,
                    input_json,
                    ..
                } if tool_name == "lookup" && input_json == "{\"q\":\"x\"}"
            )
        }),
        "OpenAI-compatible tool stream did not produce normalized lookup tool call",
    )?;
    proof(
        "openai-compatible.chat-tool-call-split-stream",
        "openai-compatible",
        OPENAI_COMPAT_TOOL_CALL,
        transport_exchanges(transport.as_ref())?,
        success_terminal(&response),
        json!({
            "classification": "success",
            "terminal_reason": response.terminal_reason.code(),
            "full_text": response.full_text,
        }),
    )
}

async fn prove_openai_responses_text_stream() -> Result<ProofRun, FixedScriptRunnerError> {
    let transport = Arc::new(ScriptedLlmHttpTransport::from_json_str(
        OPENAI_RESPONSES_TEXT,
    )?);
    let mut provider =
        OpenAiProvider::new("test-key").with_transport(provider_transport(&transport));
    let response = provider.complete(openai_responses_request()).await?;
    require(
        response.terminal_reason == LlmTerminalReason::Stop,
        "OpenAI Responses stream terminal reason was not stop",
    )?;
    require(
        response.full_text == "Direct answer.",
        "OpenAI Responses stream did not produce expected text",
    )?;
    proof(
        "openai.responses-text-stream",
        "openai",
        OPENAI_RESPONSES_TEXT,
        transport_exchanges(transport.as_ref())?,
        success_terminal(&response),
        json!({
            "classification": "success",
            "terminal_reason": response.terminal_reason.code(),
            "full_text": response.full_text,
            "usage": response.usage,
        }),
    )
}

async fn prove_anthropic_messages_text_stream() -> Result<ProofRun, FixedScriptRunnerError> {
    let transport = Arc::new(ScriptedLlmHttpTransport::from_json_str(
        ANTHROPIC_MESSAGES_TEXT,
    )?);
    let mut provider = AnthropicProvider::new("test-key")
        .with_base_url(Some("https://anthropic.test".to_string()))
        .with_transport(provider_transport(&transport));
    let response = provider.complete(anthropic_messages_request()).await?;
    require(
        response.terminal_reason == LlmTerminalReason::Stop,
        "Anthropic Messages stream terminal reason was not stop",
    )?;
    require(
        response.full_text == "Anthropic scripted answer.",
        "Anthropic Messages stream did not produce expected text",
    )?;
    proof(
        "anthropic.messages-text-stream",
        "anthropic",
        ANTHROPIC_MESSAGES_TEXT,
        transport_exchanges(transport.as_ref())?,
        success_terminal(&response),
        json!({
            "classification": "success",
            "terminal_reason": response.terminal_reason.code(),
            "full_text": response.full_text,
            "usage": response.usage,
        }),
    )
}

async fn prove_google_stream_generate_text() -> Result<ProofRun, FixedScriptRunnerError> {
    let transport = Arc::new(ScriptedLlmHttpTransport::from_json_str(
        GOOGLE_STREAM_GENERATE_TEXT,
    )?);
    let mut provider = GoogleOAuthProvider::new("access-token", "refresh-token", 0)
        .with_project_id(Some("project-1".to_string()))
        .with_transport(provider_transport(&transport));
    let response = provider.complete(google_request(true)).await?;
    require(
        response.terminal_reason == LlmTerminalReason::Stop,
        "Google streamGenerateContent terminal reason was not stop",
    )?;
    require(
        response.full_text == "Google scripted answer.",
        "Google streamGenerateContent did not produce expected text",
    )?;
    require(
        response.usage.input_tokens == 6
            && response.usage.output_tokens == 3
            && response.usage.reasoning_tokens == 1,
        "Google streamGenerateContent did not normalize usage metadata",
    )?;
    proof(
        "google.stream-generate-content-text-stream",
        "google_oauth",
        GOOGLE_STREAM_GENERATE_TEXT,
        transport_exchanges(transport.as_ref())?,
        success_terminal(&response),
        json!({
            "classification": "success",
            "terminal_reason": response.terminal_reason.code(),
            "full_text": response.full_text,
            "usage": response.usage,
        }),
    )
}

async fn prove_google_generate_text() -> Result<ProofRun, FixedScriptRunnerError> {
    let transport = Arc::new(ScriptedLlmHttpTransport::from_json_str(
        GOOGLE_GENERATE_TEXT,
    )?);
    let mut provider = GoogleOAuthProvider::new("access-token", "refresh-token", 0)
        .with_project_id(Some("project-1".to_string()))
        .with_transport(provider_transport(&transport));
    let response = provider.complete(google_request(false)).await?;
    require(
        response.terminal_reason == LlmTerminalReason::Stop,
        "Google generateContent terminal reason was not stop",
    )?;
    require(
        response.full_text == "Google buffered answer.",
        "Google generateContent did not produce expected text",
    )?;
    require(
        response.usage.input_tokens == 6
            && response.usage.output_tokens == 3
            && response.usage.cached_input_tokens == 2,
        "Google generateContent did not normalize buffered usage metadata",
    )?;
    proof(
        "google.generate-content-text",
        "google_oauth",
        GOOGLE_GENERATE_TEXT,
        transport_exchanges(transport.as_ref())?,
        success_terminal(&response),
        json!({
            "classification": "success",
            "terminal_reason": response.terminal_reason.code(),
            "full_text": response.full_text,
            "usage": response.usage,
        }),
    )
}

async fn prove_google_generate_rate_limit() -> Result<ProofRun, FixedScriptRunnerError> {
    let transport = Arc::new(ScriptedLlmHttpTransport::from_json_str(
        GOOGLE_GENERATE_RATE_LIMIT,
    )?);
    let mut provider = GoogleOAuthProvider::new("access-token", "refresh-token", 0)
        .with_project_id(Some("project-1".to_string()))
        .with_transport(provider_transport(&transport));
    let err = provider
        .complete(google_request(false))
        .await
        .expect_err("Google rate-limit script should fail");
    require(
        err.status == Some(429),
        "Google rate-limit script did not preserve 429 status",
    )?;
    require(
        err.retry_after == Some(std::time::Duration::from_secs(5)),
        "Google rate-limit script did not preserve retry-after",
    )?;
    let classified = DefaultProviderFailureClassifier.classify(err.clone());
    proof(
        "google.generate-content-rate-limit-429",
        "google_oauth",
        GOOGLE_GENERATE_RATE_LIMIT,
        transport_exchanges(transport.as_ref())?,
        error_terminal(&err),
        json!({
            "status": err.status,
            "headers": redacted_headers(&err.headers),
            "raw_body_bytes": err.raw.as_ref().map(|body| body.len()),
            "retry_after_ms": err.retry_after.map(|duration| duration.as_millis() as u64),
            "request_body_snapshot": err.request_body.is_some(),
            "provider_error_retryable": err.retryable,
            "classification": failure_classification(&classified),
        }),
    )
}

async fn prove_openai_compatible_rate_limit() -> Result<ProofRun, FixedScriptRunnerError> {
    let (mut provider, transport) = openai_compatible_provider(OPENAI_COMPAT_RATE_LIMIT)?;
    let err = provider
        .complete(openai_compatible_request(false))
        .await
        .expect_err("rate-limit script should fail");
    require(
        err.status == Some(429),
        "OpenAI-compatible rate limit script did not preserve 429 status",
    )?;
    let classified = DefaultProviderFailureClassifier.classify(err.clone());
    proof(
        "openai-compatible.chat-rate-limit-429",
        "openai-compatible",
        OPENAI_COMPAT_RATE_LIMIT,
        transport_exchanges(transport.as_ref())?,
        error_terminal(&err),
        json!({
            "status": err.status,
            "headers": redacted_headers(&err.headers),
            "raw_body_bytes": err.raw.as_ref().map(|body| body.len()),
            "retry_after_ms": err.retry_after.map(|duration| duration.as_millis() as u64),
            "request_body_snapshot": err.request_body.is_some(),
            "provider_error_retryable": err.retryable,
            "classification": failure_classification(&classified),
        }),
    )
}

async fn prove_openai_compatible_validation() -> Result<ProofRun, FixedScriptRunnerError> {
    let (mut provider, transport) = openai_compatible_provider(OPENAI_COMPAT_VALIDATION)?;
    let err = provider
        .complete(openai_compatible_request(false))
        .await
        .expect_err("validation script should fail");
    require(
        err.status == Some(400),
        "OpenAI-compatible validation script did not preserve 400 status",
    )?;
    let classified = DefaultProviderFailureClassifier.classify(err.clone());
    proof(
        "openai-compatible.chat-validation-error",
        "openai-compatible",
        OPENAI_COMPAT_VALIDATION,
        transport_exchanges(transport.as_ref())?,
        error_terminal(&err),
        json!({
            "status": err.status,
            "headers": redacted_headers(&err.headers),
            "raw_body_bytes": err.raw.as_ref().map(|body| body.len()),
            "request_body_snapshot": err.request_body.is_some(),
            "provider_error_retryable": err.retryable,
            "classification": failure_classification(&classified),
        }),
    )
}

async fn prove_openai_compatible_disconnect() -> Result<ProofRun, FixedScriptRunnerError> {
    let (mut provider, transport) = openai_compatible_provider(OPENAI_COMPAT_DISCONNECT)?;
    let err = provider
        .complete(openai_compatible_request(true))
        .await
        .expect_err("disconnect script should fail");
    require(
        err.kind == ProviderFailureKind::Stream && err.retryable,
        "OpenAI-compatible disconnect script did not surface retryable stream failure",
    )?;
    let classified = DefaultProviderFailureClassifier.classify(err.clone());
    proof(
        "openai-compatible.chat-mid-stream-disconnect",
        "openai-compatible",
        OPENAI_COMPAT_DISCONNECT,
        transport_exchanges(transport.as_ref())?,
        error_terminal(&err),
        json!({
            "kind": format!("{:?}", err.kind),
            "retryable": err.retryable,
            "classification": failure_classification(&classified),
        }),
    )
}

async fn prove_openai_compatible_response_start_timeout() -> Result<ProofRun, FixedScriptRunnerError>
{
    let (mut provider, transport) =
        openai_compatible_provider(OPENAI_COMPAT_RESPONSE_START_TIMEOUT)?;
    let err = provider
        .complete(openai_compatible_request(true))
        .await
        .expect_err("response-start timeout script should fail");
    require(
        err.kind == ProviderFailureKind::Timeout
            && err.code.as_deref() == Some("timeout")
            && err.retryable
            && err.status.is_none(),
        "OpenAI-compatible response-start timeout did not match production timeout envelope",
    )?;
    proof(
        "openai-compatible.chat-response-start-timeout",
        "openai-compatible",
        OPENAI_COMPAT_RESPONSE_START_TIMEOUT,
        transport_exchanges(transport.as_ref())?,
        error_terminal(&err),
        json!({
            "classification": failure_classification(&err),
            "timeout_phase": "response_start",
            "reported_successful_partial_response": false,
        }),
    )
}

async fn prove_openai_compatible_stream_chunk_timeout() -> Result<ProofRun, FixedScriptRunnerError>
{
    let (events, sender) = event_collector();
    let (mut provider, transport) = openai_compatible_provider(OPENAI_COMPAT_STREAM_CHUNK_TIMEOUT)?;
    let err = provider
        .complete(openai_compatible_request_with_events(Some(sender)))
        .await
        .expect_err("stream chunk timeout script should fail");
    let committed_events = events.lock().expect("event collector lock").len();
    require(
        err.kind == ProviderFailureKind::Timeout
            && err.code.as_deref() == Some("timeout")
            && err.retryable
            && err.status.is_none(),
        "OpenAI-compatible stream chunk timeout did not match production timeout envelope",
    )?;
    require(
        committed_events == 0,
        "stream chunk timeout committed a partial provider success event",
    )?;
    proof(
        "openai-compatible.chat-stream-chunk-timeout",
        "openai-compatible",
        OPENAI_COMPAT_STREAM_CHUNK_TIMEOUT,
        transport_exchanges(transport.as_ref())?,
        error_terminal(&err),
        json!({
            "classification": failure_classification(&err),
            "timeout_phase": "stream_chunk",
            "stream_events_committed": committed_events,
            "reported_successful_partial_response": false,
        }),
    )
}

async fn prove_openai_compatible_cancel_before_response_start()
-> Result<ProofRun, FixedScriptRunnerError> {
    let schedule = ScriptedTransportSchedule::new();
    let transport = Arc::new(
        ScriptedLlmHttpTransport::from_json_str(OPENAI_COMPAT_TOOL_CALL)?
            .with_event_schedule(schedule.clone()),
    );
    let (events, sender) = event_collector();
    let mut provider = OpenAiCompatibleProvider::new("test-key", "https://provider.test")
        .with_transport(provider_transport(&transport));

    let task = tokio::spawn(async move {
        provider
            .complete(openai_compatible_request_with_events(Some(sender)))
            .await
    });
    schedule.wait_until_blocked(0, 0).await;
    task.abort();

    let join_err = task.await.expect_err("cancelled provider task");
    require(
        join_err.is_cancelled(),
        "cancellation before response start did not cancel the provider task",
    )?;
    let committed_events = events.lock().expect("event collector lock").len();
    require(
        committed_events == 0,
        "cancellation before response start committed stream events",
    )?;
    proof(
        "openai-compatible.cancel-before-response-start",
        "openai-compatible",
        OPENAI_COMPAT_TOOL_CALL,
        transport_exchanges(transport.as_ref())?,
        cancelled_terminal(),
        json!({
            "classification": "cancelled_before_response_start",
            "stream_events_committed": committed_events,
        }),
    )
}

async fn prove_openai_compatible_retry_exhaustion() -> Result<ProofRun, FixedScriptRunnerError> {
    let attempt_budget = 2;
    let rate_limit_script = ProviderWireScript::from_json_str(OPENAI_COMPAT_RATE_LIMIT)?;
    let transport = ScriptedLlmHttpTransport::from_scripts(
        (0..attempt_budget).map(|_| rate_limit_script.clone()),
    );
    let transport_for_assert = transport.clone();
    let retry_options = ProviderOptions {
        reliability: lash_core::provider::ProviderReliability::default()
            .max_attempts(attempt_budget)
            .base_delay_ms(0)
            .max_delay_ms(0)
            .retry_after_cap_ms(Some(0)),
        ..ProviderOptions::default()
    };
    let provider = OpenAiCompatibleProvider::new("test-key", "https://provider.test")
        .with_options(retry_options.clone())
        .with_transport(Arc::new(transport));
    let mut handle = ProviderHandle::new(provider.into_components());
    handle.set_options(retry_options);
    let err = handle
        .complete(openai_compatible_request(false))
        .await
        .expect_err("retry exhaustion should fail");
    require(
        err.status == Some(429) && err.retryable,
        format!(
            "retry exhaustion did not return classified retryable 429: status={:?} retryable={} kind={:?} code={:?} message={}",
            err.status, err.retryable, err.kind, err.code, err.message
        ),
    )?;
    require(
        transport_for_assert.remaining_scripts()? == 0,
        "retry exhaustion did not consume the isolated two-attempt retry script budget",
    )?;
    let exchanges = transport_exchanges(&transport_for_assert)?;
    require(
        exchanges.len() == attempt_budget as usize,
        format!(
            "retry exhaustion executed {} scripted HTTP exchanges, expected the isolated two-attempt retry budget {attempt_budget}",
            exchanges.len()
        ),
    )?;
    require(
        exchanges
            .iter()
            .all(|exchange| exchange.response.status == Some(429)),
        "retry exhaustion did not preserve 429 status on every scripted retry attempt",
    )?;
    proof(
        "openai-compatible.retry-exhaustion",
        "openai-compatible",
        OPENAI_COMPAT_RATE_LIMIT,
        exchanges,
        error_terminal(&err),
        json!({
            "status": err.status,
            "retryable": err.retryable,
            "classification": failure_classification(&err),
            "attempts_consumed": attempt_budget,
        }),
    )
}

fn proof(
    name: impl Into<String>,
    provider_kind: impl Into<String>,
    script_content: &str,
    http_exchanges: Vec<ScriptedLlmHttpExchange>,
    terminal: TranscriptTerminal,
    observed: serde_json::Value,
) -> Result<ProofRun, FixedScriptRunnerError> {
    let name = name.into();
    let provider_kind = provider_kind.into();
    let transcript = transcript_for_script(
        &name,
        &provider_kind,
        script_content,
        http_exchanges,
        terminal,
        observed.clone(),
    )?;
    let endpoint = transcript.endpoint.path.clone();
    Ok(ProofRun {
        name,
        provider_kind,
        endpoint,
        outcome: "passed".to_string(),
        transcript,
        observed,
    })
}

fn success_terminal(response: &LlmResponse) -> TranscriptTerminal {
    TranscriptTerminal {
        classification: "success",
        provider_result: Some(TranscriptProviderResult {
            terminal_reason: response.terminal_reason.code().to_string(),
            full_text_bytes: response.full_text.len(),
            part_count: response.parts.len(),
            usage: serde_json::to_value(&response.usage).unwrap_or(serde_json::Value::Null),
        }),
        error_envelope: None,
    }
}

fn cancelled_terminal() -> TranscriptTerminal {
    TranscriptTerminal {
        classification: "cancelled_before_response_start",
        provider_result: None,
        error_envelope: None,
    }
}

fn error_terminal(error: &LlmTransportError) -> TranscriptTerminal {
    TranscriptTerminal {
        classification: "error",
        provider_result: None,
        error_envelope: Some(TranscriptErrorEnvelope {
            kind: format!("{:?}", error.kind),
            code: error.code.clone(),
            status: error.status,
            retryable: error.retryable,
            terminal_reason: error.terminal_reason.code().to_string(),
            raw_body_bytes: error.raw.as_ref().map(|body| body.len()),
            headers: transcript_headers_from_pairs(&error.headers),
            retry_after_ms: error
                .retry_after
                .map(|duration| duration.as_millis() as u64),
            request_body_snapshot: error.request_body.is_some(),
        }),
    }
}

fn transport_exchanges(
    transport: &ScriptedLlmHttpTransport,
) -> Result<Vec<ScriptedLlmHttpExchange>, FixedScriptRunnerError> {
    Ok(transport.exchanges()?)
}

fn provider_transport(transport: &Arc<ScriptedLlmHttpTransport>) -> Arc<dyn LlmHttpTransport> {
    transport.clone()
}

fn failure_classification(failure: &LlmTransportError) -> serde_json::Value {
    json!({
        "kind": format!("{:?}", failure.kind),
        "retryable": failure.retryable,
        "status": failure.status,
        "terminal_reason": failure.terminal_reason.code(),
    })
}

fn transcript_headers_from_pairs(headers: &[(String, String)]) -> Vec<TranscriptHeader> {
    headers
        .iter()
        .map(|(name, value)| TranscriptHeader {
            name: name.clone(),
            value: redacted_header_value(name, value),
        })
        .collect()
}

fn redacted_headers(headers: &[(String, String)]) -> Vec<serde_json::Value> {
    headers
        .iter()
        .map(|(name, value)| json!({ "name": name, "value": redacted_header_value(name, value) }))
        .collect()
}

fn redacted_header_value(name: &str, value: &str) -> String {
    let lower_name = name.to_ascii_lowercase();
    let lower_value = value.to_ascii_lowercase();
    if matches!(
        lower_name.as_str(),
        "authorization" | "proxy-authorization" | "cookie" | "set-cookie" | "x-api-key"
    ) || lower_name.contains("api-key")
        || lower_name.contains("token")
        || lower_value.contains("bearer ")
        || lower_value.contains("sk-")
    {
        "[redacted]".to_string()
    } else {
        value.to_string()
    }
}

fn require(condition: bool, message: impl Into<String>) -> Result<(), FixedScriptRunnerError> {
    if condition {
        Ok(())
    } else {
        Err(FixedScriptRunnerError::Assertion(message.into()))
    }
}

fn openai_compatible_provider(
    script: &str,
) -> Result<(OpenAiCompatibleProvider, Arc<ScriptedLlmHttpTransport>), FixedScriptRunnerError> {
    let transport = Arc::new(ScriptedLlmHttpTransport::from_json_str(script)?);
    let provider = OpenAiCompatibleProvider::new("test-key", "https://provider.test")
        .with_transport(provider_transport(&transport));
    Ok((provider, transport))
}

fn openai_compatible_request(stream: bool) -> LlmRequest {
    openai_compatible_request_with_events(stream.then(|| LlmEventSender::new(|_event| {})))
}

fn openai_compatible_request_with_events(stream_events: Option<LlmEventSender>) -> LlmRequest {
    LlmRequest {
        model: "openai/gpt-5.4".to_string(),
        messages: vec![LlmMessage::text(LlmRole::User, "lookup x")],
        attachments: Vec::new(),
        tools: Arc::new(vec![LlmToolSpec {
            name: "lookup".to_string(),
            description: "Lookup".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "q": { "type": "string" }
                }
            })
            .into(),
            output_schema: json!({}).into(),
        }]),
        tool_choice: LlmToolChoice::Auto,
        model_variant: None,
        generation: lash_core::GenerationOptions::default(),
        session_id: Some("session-1".to_string()),
        output_spec: None,
        stream_events,
        provider_trace: None,
    }
}

fn event_collector() -> (Arc<Mutex<Vec<LlmStreamEvent>>>, LlmEventSender) {
    let events = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&events);
    let sender = LlmEventSender::new(move |event| {
        captured.lock().expect("event collector lock").push(event);
    });
    (events, sender)
}

fn openai_responses_request() -> LlmRequest {
    LlmRequest {
        model: "gpt-5.4".to_string(),
        messages: vec![LlmMessage::text(LlmRole::User, "answer directly")],
        attachments: Vec::new(),
        tools: Arc::new(Vec::new()),
        tool_choice: LlmToolChoice::Auto,
        model_variant: None,
        generation: lash_core::GenerationOptions::default(),
        session_id: Some("session-1".to_string()),
        output_spec: None,
        stream_events: Some(LlmEventSender::new(|_event| {})),
        provider_trace: None,
    }
}

fn anthropic_messages_request() -> LlmRequest {
    LlmRequest {
        model: "claude-sonnet-4-20250514".to_string(),
        messages: vec![LlmMessage::text(LlmRole::User, "answer directly")],
        attachments: Vec::new(),
        tools: Arc::new(Vec::new()),
        tool_choice: LlmToolChoice::Auto,
        model_variant: None,
        generation: lash_core::GenerationOptions::default(),
        session_id: Some("session-1".to_string()),
        output_spec: None,
        stream_events: Some(LlmEventSender::new(|_event| {})),
        provider_trace: None,
    }
}

fn google_request(stream: bool) -> LlmRequest {
    LlmRequest {
        model: "gemini-3.1-pro-preview".to_string(),
        messages: vec![LlmMessage::text(LlmRole::User, "answer directly")],
        attachments: Vec::new(),
        tools: Arc::new(Vec::new()),
        tool_choice: LlmToolChoice::Auto,
        model_variant: None,
        generation: lash_core::GenerationOptions::default(),
        session_id: Some("session-1".to_string()),
        output_spec: None,
        stream_events: stream.then(|| LlmEventSender::new(|_event| {})),
        provider_trace: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn divergent_seed_cross_backend_durable_state_agrees() {
        // Regression guard for full-random seed 14123330213291275571, whose durable
        // cross-backend re-run previously hung (a `next_turn` queued ingress ran an
        // unmodeled inline turn under serialized execution) and then diverged (a
        // slow async store let later boundaries overtake a live turn's completion,
        // drifting the seeded delivery order). The serialized in-memory reference
        // and the SQLite durable re-run share the serialize-provider-turn discipline
        // and differ only in the store, so their abstract durable-state summaries
        // must be byte-identical.
        let seed = 14_123_330_213_291_275_571u64;
        let workload = generate_workload(seed, "full-random", 384).expect("workload");
        let reference = replay_workload_serialized_reference(&workload)
            .await
            .expect("serialized in-memory reference");
        let tmp = tempfile::tempdir().expect("tempdir");
        let sqlite_summary = replay_workload_on_sqlite(&workload, &tmp.path().join("sqlite-store"))
            .await
            .expect("sqlite re-run");
        assert_eq!(
            reference, sqlite_summary,
            "cross-backend durable-state summaries diverged for seed {seed}"
        );
        println!(
            "OK seed={seed} sessions={} digest={}",
            reference.session_count, reference.digest
        );
    }

    #[test]
    fn full_random_seed_12_keeps_modeled_provider_exchange_slots_owned_by_scheduler() {
        let seed = generated_seed("full-random", 12);
        assert_eq!(seed, 8_740_143_186_674_533_974);
        let workload = generate_workload(seed, "full-random", 384).expect("workload");
        let provider_turn = workload
            .boundaries
            .iter()
            .find(|event| event.boundary_id == "session-001:provider:003")
            .expect("session-001 provider turn 3");
        assert_eq!(
            provider_turn.payload.get("script").and_then(Value::as_str),
            Some("openai-compatible.chat-runtime-text-stream")
        );
        assert_eq!(
            provider_turn
                .payload
                .get("expected_provider_exchange_count")
                .and_then(Value::as_u64),
            Some(3)
        );

        let trace = run_on_sim_harness_stack(
            "full-random-seed-12-modeled-provider-exchange-slots",
            SIM_HARNESS_STACK_LIMIT_BYTES,
            move || {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(FixedScriptRunnerError::Io)?;
                runtime.block_on(run_generated_workload_for_fixture(
                    workload,
                    "seed-12-regression",
                ))
            },
        )
        .expect("seed 12 generated workload");
        let delivered = trace
            .events
            .iter()
            .find(|event| event.boundary_id == "session-001:provider:003")
            .expect("delivered provider turn 3");
        assert_eq!(
            delivered.observed.get("success").and_then(Value::as_bool),
            Some(true),
            "success-required modeled provider turn must not be converted into provider-error terminalization"
        );
        assert_eq!(
            delivered
                .observed
                .get("provider_exchange_count")
                .and_then(Value::as_u64),
            Some(3),
            "autonomous queued turns must not consume provider scripts before modeled turn 3"
        );
    }

    #[tokio::test]
    async fn runtime_completion_serialization_mutation_guard() {
        let seed = generated_seed("full-random", 12);
        let workload = generate_workload(seed, "full-random", 384).expect("workload");
        let mut world = GeneratedRuntimeWorld::with_backend(
            Arc::new(lash::persistence::InMemorySessionStoreFactory::new()),
            RuntimeEffectReplayStore::Memory,
            Arc::new(lash::persistence::InMemoryAttachmentStore::new()),
            Arc::new(lash::persistence::InMemoryProcessExecutionEnvStore::new()),
            true,
        );

        let (events, _summary) = drive_generated_workload(&mut world, &workload)
            .await
            .expect("serialized generated workload");
        let provider_completions = events
            .iter()
            .filter(|event| event.kind == BoundaryKind::Provider)
            .count();

        assert!(
            provider_completions > 1,
            "serialization guard must exercise more than one provider turn"
        );
        assert_eq!(
            world.peak_concurrent_live_turns, 1,
            "serialized generated replay must initialize RuntimeCompletionState with serialize_provider_turns=true"
        );
        assert_eq!(
            peak_concurrent_live_turns(&events),
            1,
            "delivered evidence must not show overlapping provider turns under serialized replay"
        );
    }

    #[test]
    fn standard_protocol_full_text_mutation_guard() {
        let result = run_standard_protocol_contract(
            "standard.full_text_mutation_guard",
            "answer with two chunks",
            None,
            vec![
                StandardContractStep::Llm {
                    text_streamed: true,
                    parts: vec![
                        standard_text_part("first chunk"),
                        standard_text_part("second chunk"),
                    ],
                },
                StandardContractStep::Checkpoint,
            ],
        )
        .expect("standard full-text contract");

        assert_eq!(
            result
                .get("llm_response_full_texts")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default(),
            vec![json!("first chunk\nsecond chunk")],
            "fixed Standard execution must preserve LlmResponse.full_text, not only streamed parts"
        );
        assert_eq!(
            result
                .get("llm_response_parts")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default(),
            vec![json!([
                {"kind": "text", "text": "first chunk"},
                {"kind": "text", "text": "second chunk"},
            ])],
            "fixed Standard execution must preserve concrete response parts"
        );
        assert_eq!(
            result.get("llm_call_count").and_then(Value::as_u64),
            Some(1),
            "the guard must drive a real Standard LLM turn"
        );
        assert_eq!(
            result.get("done").and_then(Value::as_bool),
            Some(true),
            "the guarded Standard turn must still complete"
        );
    }

    #[test]
    fn rlm_protocol_response_shape_mutation_guard() {
        let result = run_rlm_protocol_contract(
            "rlm.response_shape_mutation_guard",
            "answer naturally",
            RlmTermination::Natural,
            None,
            None,
            vec![
                RlmContractStep::Llm(vec![rlm_text_part("RLM final prose")]),
                RlmContractStep::Checkpoint,
            ],
        )
        .expect("rlm response-shape contract");

        assert_eq!(
            result
                .get("llm_response_full_texts")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default(),
            vec![json!("RLM final prose")],
            "fixed RLM execution must preserve LlmResponse.full_text"
        );
        assert_eq!(
            result
                .get("llm_response_part_counts")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default(),
            vec![json!(1)],
            "fixed RLM execution must preserve concrete response parts"
        );
        assert_eq!(
            result
                .get("llm_response_parts")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default(),
            vec![json!([{"kind": "text", "text": "RLM final prose"}])],
            "fixed RLM execution must preserve the concrete text part"
        );
        assert_eq!(
            result.get("done").and_then(Value::as_bool),
            Some(true),
            "the guarded RLM turn must still complete"
        );
    }

    #[tokio::test]
    async fn fixed_texts_provider_response_shape_mutation_guard() {
        let mut provider =
            fixed_texts_provider("lash-sim-fixed-text-guard", vec!["facade response text"]);
        let response = provider
            .complete(openai_compatible_request(false))
            .await
            .expect("fixed text provider response");

        assert_eq!(response.full_text, "facade response text");
        assert!(
            matches!(
                response.parts.as_slice(),
                [LlmOutputPart::Text { text, .. }] if text == "facade response text"
            ),
            "fixed text provider must return a matching text part"
        );
    }

    #[tokio::test]
    async fn rlm_final_value_provider_response_shape_mutation_guard() {
        let mut provider = rlm_final_value_provider();
        let response = provider
            .complete(openai_compatible_request(true))
            .await
            .expect("rlm final-value provider response");

        assert!(response.full_text.contains("semantic-channel"));
        assert!(
            response_text_part(&response).is_some_and(|text| text.contains("semantic-channel")),
            "rlm final-value provider must return the semantic text part"
        );
    }

    #[tokio::test]
    async fn pending_tool_roundtrip_provider_response_shape_mutation_guard() {
        let mut provider = pending_tool_roundtrip_provider();
        let tool_response = provider
            .complete(openai_compatible_request(false))
            .await
            .expect("pending tool provider tool-call response");
        assert!(
            matches!(
                tool_response.parts.as_slice(),
                [LlmOutputPart::ToolCall { call_id, tool_name, input_json, .. }]
                    if call_id == "call-1" && tool_name == "app_lookup" && input_json == "{}"
            ),
            "pending tool provider must start with the concrete tool-call part"
        );

        let final_response = provider
            .complete(openai_compatible_request(false))
            .await
            .expect("pending tool provider final response");
        assert_eq!(final_response.full_text, "done");
        assert_eq!(response_text_part(&final_response), Some("done"));
    }

    #[tokio::test]
    async fn fixed_script_profile_writes_deterministic_manifest() {
        let tmp = tempfile::tempdir().expect("tempdir");

        let manifest = run_fixed_script_profile(tmp.path()).await.expect("profile");

        assert_eq!(manifest.profile, FIXED_SCRIPT_PROFILE);
        assert_eq!(
            manifest.timeline_at_semantics,
            FIXED_SCRIPT_TIMELINE_AT_SEMANTICS
        );
        assert_eq!(manifest.summary.total_scripts, 11);
        assert_eq!(manifest.summary.total_proofs, 13);
        assert_eq!(manifest.summary.total_events, 14);
        assert_eq!(manifest.summary.passed, 13);
        assert!(
            manifest
                .provider_transport_exclusions
                .iter()
                .any(|exclusion| exclusion.path.contains("codex.rs"))
        );
        assert!(manifest.manifest_path.ends_with(FIXED_SCRIPT_MANIFEST));
        assert!(manifest.summary_path.ends_with(FIXED_SCRIPT_SUMMARY));

        let body =
            std::fs::read_to_string(tmp.path().join(FIXED_SCRIPT_MANIFEST)).expect("manifest");
        assert!(body.contains("script_bundle_hash"));
        assert!(body.contains("anthropic.messages-text-stream"));
        assert!(body.contains("openai.responses-text-stream"));
        assert!(body.contains("openai-compatible.chat-response-start-timeout"));
        assert!(body.contains("openai-compatible.chat-stream-chunk-timeout"));
        assert!(body.contains("openai-compatible.cancel-before-response-start"));
        assert!(body.contains("openai-compatible.retry-exhaustion"));
        assert!(body.contains("google.stream-generate-content-text-stream"));
        assert!(body.contains("google.generate-content-text"));
        assert!(body.contains("google.generate-content-rate-limit-429"));

        let summary_body =
            std::fs::read_to_string(tmp.path().join(FIXED_SCRIPT_SUMMARY)).expect("summary");
        let summary: serde_json::Value = serde_json::from_str(&summary_body).expect("summary JSON");
        assert_eq!(summary["schema"], "lash.sim.summary.v1");
        assert_eq!(summary["profile"], FIXED_SCRIPT_PROFILE);
        assert_eq!(summary["fixed_script_manifest"], FIXED_SCRIPT_MANIFEST);
        assert_eq!(summary["counts"]["generated_seeds"], 0);
        assert_eq!(summary["counts"]["fixed_replays"], 13);
        assert_eq!(summary["counts"]["oracle_passes"], 13);
        assert_eq!(
            summary["provider_set"],
            json!(["anthropic", "google_oauth", "openai", "openai-compatible"])
        );
    }

    #[tokio::test]
    async fn fixed_script_manifest_schema_contains_required_proofs_and_artifact_fields() {
        let tmp = tempfile::tempdir().expect("tempdir");

        run_fixed_script_profile(tmp.path()).await.expect("profile");

        let body =
            std::fs::read_to_string(tmp.path().join(FIXED_SCRIPT_MANIFEST)).expect("manifest");
        let manifest: serde_json::Value = serde_json::from_str(&body).expect("manifest JSON");
        assert_eq!(manifest["schema"], "lash.sim.fixed-script-manifest.v1");
        assert_eq!(manifest["profile"], FIXED_SCRIPT_PROFILE);
        assert_eq!(
            manifest["timeline_at_semantics"],
            FIXED_SCRIPT_TIMELINE_AT_SEMANTICS
        );
        assert_eq!(
            manifest["summary"],
            json!({
                "total_scripts": 11,
                "total_proofs": 13,
                "total_events": 14,
                "passed": 13
            })
        );
        assert_eq!(
            manifest["script_bundle_hash"]
                .as_str()
                .expect("script bundle hash")
                .len(),
            64
        );

        let proofs = manifest["proofs"].as_array().expect("proofs array");
        let required = [
            "openai-compatible.chat-tool-call-split-stream",
            "openai.responses-text-stream",
            "anthropic.messages-text-stream",
            "openai-compatible.chat-rate-limit-429",
            "openai-compatible.chat-validation-error",
            "openai-compatible.chat-mid-stream-disconnect",
            "openai-compatible.chat-response-start-timeout",
            "openai-compatible.chat-stream-chunk-timeout",
            "openai-compatible.cancel-before-response-start",
            "openai-compatible.retry-exhaustion",
            "google.stream-generate-content-text-stream",
            "google.generate-content-text",
            "google.generate-content-rate-limit-429",
        ];
        for name in required {
            let proof = proofs
                .iter()
                .find(|proof| proof["name"] == name)
                .unwrap_or_else(|| panic!("missing proof {name}"));
            assert_eq!(proof["outcome"], "passed");
            assert!(
                proof["endpoint"]
                    .as_str()
                    .expect("endpoint")
                    .starts_with('/')
            );
            assert_eq!(
                proof["transcript_sha256"]
                    .as_str()
                    .expect("transcript hash")
                    .len(),
                64
            );
            let transcript_path = proof["transcript_path"].as_str().expect("transcript path");
            assert!(transcript_path.starts_with("proofs/"));

            let transcript_body =
                std::fs::read_to_string(tmp.path().join(transcript_path)).expect("transcript");
            let transcript: serde_json::Value =
                serde_json::from_str(&transcript_body).expect("transcript JSON");
            assert_eq!(
                transcript["schema"],
                "lash.sim.fixed-script-proof-transcript.v1"
            );
            assert_eq!(transcript["proof"], name);
            assert_eq!(
                transcript["timeline_at_semantics"],
                FIXED_SCRIPT_TIMELINE_AT_SEMANTICS
            );
            assert!(
                transcript["request_match"]["body_paths"]
                    .as_array()
                    .expect("body paths")
                    .iter()
                    .all(|path| path.as_str().is_some())
            );
            assert!(
                !transcript["response_events"]
                    .as_array()
                    .expect("response events")
                    .is_empty()
            );
            let exchanges = transcript["http_exchanges"]
                .as_array()
                .expect("http exchanges");
            assert!(
                !exchanges.is_empty(),
                "transcript {name} should contain a sanitized HTTP exchange"
            );
            for exchange in exchanges {
                assert_eq!(exchange["request"]["method"], "POST");
                assert!(
                    exchange["request"]["path"]
                        .as_str()
                        .expect("request path")
                        .starts_with('/')
                );
                assert!(
                    exchange["request"]["body_bytes"]
                        .as_u64()
                        .expect("body bytes")
                        > 0
                );
                assert_eq!(exchange["request"]["body_shape"]["type"], "object");
                let request_headers = exchange["request"]["headers"]
                    .as_array()
                    .expect("request headers");
                let auth_header = request_headers
                    .iter()
                    .find(|header| {
                        header["name"].as_str().is_some_and(|name| {
                            name.eq_ignore_ascii_case("authorization")
                                || name.eq_ignore_ascii_case("x-api-key")
                        })
                    })
                    .expect("provider auth header");
                assert_eq!(auth_header["value"], "[redacted]");
                assert!(
                    exchange["response"]["status"].is_null()
                        || exchange["response"]["status"].is_u64()
                );
                assert!(exchange["response"]["headers"].is_array());
                assert!(exchange["response"]["event_names"].is_array());
            }
            let terminal = &transcript["terminal"];
            match terminal["classification"].as_str().expect("terminal class") {
                "success" => {
                    assert!(terminal["provider_result"]["terminal_reason"].is_string());
                    assert!(terminal["provider_result"]["full_text_bytes"].is_u64());
                    assert!(terminal["provider_result"]["part_count"].is_u64());
                }
                "error" => {
                    let envelope = &terminal["error_envelope"];
                    assert!(envelope["kind"].is_string());
                    assert!(envelope["retryable"].is_boolean());
                    assert!(envelope["terminal_reason"].is_string());
                    assert!(envelope["headers"].is_array());
                }
                "cancelled_before_response_start" => {}
                other => panic!("unexpected terminal classification {other}"),
            }
            if name.ends_with("timeout") {
                let envelope = &terminal["error_envelope"];
                assert_eq!(envelope["kind"], "Timeout");
                assert_eq!(envelope["code"], "timeout");
                assert_eq!(envelope["retryable"], true);
            }
            if name == "openai-compatible.chat-stream-chunk-timeout" {
                assert_eq!(transcript["observed"]["stream_events_committed"], 0);
                assert_eq!(
                    transcript["observed"]["reported_successful_partial_response"],
                    false
                );
            }
            assert!(!transcript_body.contains("test-key"));
            assert!(!transcript_body.contains("Bearer"));
            assert!(!transcript_body.contains("lookup x"));
            assert!(!transcript_body.contains("answer directly"));
            assert!(
                transcript["observed"]["classification"].is_string()
                    || transcript["observed"]["classification"].is_object()
            );
        }
        let matrix = manifest["provider_matrix"]
            .as_array()
            .expect("provider matrix");
        let google = matrix
            .iter()
            .find(|row| row["provider_kind"] == "google_oauth")
            .expect("google provider matrix row");
        assert_eq!(google["success_proofs"], 2);
        assert_eq!(google["error_proofs"], 1);
        assert_eq!(
            google["endpoints"],
            json!([
                "/v1internal:generateContent",
                "/v1internal:streamGenerateContent"
            ])
        );
    }

    #[tokio::test]
    async fn fixed_script_timeout_proofs_preserve_timeout_envelopes() {
        let tmp = tempfile::tempdir().expect("tempdir");

        run_fixed_script_profile(tmp.path()).await.expect("profile");

        for name in [
            "openai-compatible.chat-response-start-timeout",
            "openai-compatible.chat-stream-chunk-timeout",
        ] {
            let transcript_path = tmp.path().join("proofs").join(format!("{name}.json"));
            let transcript_body =
                std::fs::read_to_string(transcript_path).expect("timeout transcript");
            let transcript: serde_json::Value =
                serde_json::from_str(&transcript_body).expect("timeout transcript JSON");
            let envelope = &transcript["terminal"]["error_envelope"];
            assert_eq!(envelope["kind"], "Timeout");
            assert_eq!(envelope["code"], "timeout");
            assert_eq!(envelope["retryable"], true);
            assert!(envelope["status"].is_null());
            assert_eq!(
                transcript["observed"]["reported_successful_partial_response"],
                false
            );
            if name == "openai-compatible.chat-stream-chunk-timeout" {
                assert_eq!(transcript["observed"]["stream_events_committed"], 0);
                assert_eq!(transcript["http_exchanges"][0]["response"]["status"], 200);
                assert!(
                    transcript["http_exchanges"][0]["response"]["event_names"]
                        .as_array()
                        .expect("event names")
                        .iter()
                        .any(|event| event == "timeout")
                );
            }
        }
    }

    #[tokio::test]
    async fn runtime_facade_turn_uses_scripted_transport_and_checks_invariants() {
        let proof = prove_runtime_facade_turn().await.expect("runtime proof");

        assert_eq!(proof.provider_kind, "openai-compatible");
        assert_eq!(proof.session_id, "sim-runtime-session");
        assert_eq!(proof.turn_index, 1);
        assert_eq!(proof.assistant_message, "Runtime scripted answer.");
        assert_eq!(proof.provider_exchange_count, 1);
        assert!(proof.runtime_invariant.is_passed());
        assert!(proof.provider_output_invariant.is_passed());
        assert!(
            proof
                .pending_tool_completion
                .turn_suspension_invariant
                .is_passed()
        );
        assert!(
            proof
                .pending_tool_completion
                .scheduler_resolution_invariant
                .is_passed()
        );
        assert!(
            proof
                .pending_tool_completion
                .final_result_invariant
                .is_passed()
        );
        assert!(
            proof
                .final_value_semantic_channel
                .semantic_channel_invariant
                .is_passed()
        );
    }

    #[tokio::test]
    async fn pending_tool_completion_proof_uses_scheduler_delivered_tool_boundary() {
        let proof = prove_pending_tool_completion_through_turn()
            .await
            .expect("pending tool proof");

        assert_eq!(proof.session_id, "sim-pending-tool-session");
        assert_eq!(proof.turn_index, 1);
        assert_eq!(proof.assistant_message, "done");
        assert_eq!(proof.tool_name, "app_lookup");
        assert!(proof.scheduler_controlled);
        assert!(proof.turn_suspended_before_completion);
        assert_eq!(proof.completed_event_count_before_resolution, 0);
        assert!(proof.completed_event_count_after_resolution > 0);
        assert!(matches!(
            proof.completion_outcome,
            lash_core::ResolveOutcome::Accepted
        ));
        assert!(matches!(
            proof.duplicate_completion_outcome,
            lash_core::ResolveOutcome::AlreadyResolved {
                terminal: lash_core::Resolution::Ok(_)
            }
        ));
        assert!(proof.turn_suspension_invariant.is_passed());
        assert!(proof.scheduler_resolution_invariant.is_passed());
        assert!(proof.final_result_invariant.is_passed());
    }

    #[tokio::test]
    async fn final_value_semantic_channel_proof_uses_runtime_outcome_and_event() {
        let proof = prove_final_value_semantic_channel()
            .await
            .expect("final value proof");

        assert_eq!(proof.session_id, "sim-final-value-session");
        assert_eq!(proof.turn_index, 1);
        assert_eq!(proof.facts.outcome_kind, "final_value");
        assert_eq!(
            proof.final_value,
            json!({
                "source": "semantic-channel",
                "ok": true,
                "count": 3,
            })
        );
        assert!(proof.final_value_event_count > 0);
        assert!(proof.assistant_prose_delta_count > 0);
        assert!(!proof.facts.transcript_inference_required);
        assert!(proof.semantic_channel_invariant.is_passed());
    }

    #[tokio::test]
    async fn live_provider_failure_oracle_bites_on_a_committing_turn() {
        // END-TO-END NEGATIVE: drive a REAL `session.turn().run()` against a VALID
        // success script that streams AND COMMITS the leak prose (the same prose a
        // failure turn must NOT commit). The live-failure oracle MUST fail on it —
        // proving the "no committed output" assertion bites end-to-end, not just on
        // synthetic facts.
        let script = runtime_script_for_text(OPENAI_COMPATIBLE, LIVE_FAILURE_LEAK_PROSE)
            .expect("valid success control script");
        let facts = run_live_turn_facts(7, OPENAI_COMPATIBLE, script, "success_control", 1)
            .await
            .expect("drive committing control turn");

        // The control turn really did commit the prose (the runtime CAN commit).
        assert!(
            facts.committed_assistant_message_nonempty,
            "the success control turn should have committed the assistant prose: {facts:?}"
        );
        assert!(
            !facts.terminalized_failure,
            "the success control turn should not terminalize as a failure: {facts:?}"
        );
        assert!(
            facts.committed_prose_in_transcript,
            "the success control turn should leave the prose in the transcript: {facts:?}"
        );

        // The oracle catches the committed output.
        let verdict = crate::oracles::live_provider_failure_terminalizes(&facts);
        assert!(
            !verdict.is_passed(),
            "a turn that commits output MUST fail the live-provider-failure oracle: {}",
            verdict.message
        );
    }

    #[test]
    fn generated_sim_profile_writes_trace_replay_and_provider_artifacts() {
        let tmp = tempfile::tempdir().expect("tempdir");

        let artifact_root = tmp.path().to_path_buf();
        let report = run_on_sim_harness_stack(
            "generated-sim-profile-test",
            SIM_HARNESS_STACK_LIMIT_BYTES,
            move || {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(FixedScriptRunnerError::Io)?;
                runtime.block_on(run_generated_sim_profile(
                    artifact_root,
                    "fast-random",
                    2,
                    24,
                ))
            },
        )
        .expect("generated sim");

        assert_eq!(report.profile, "fast-random");
        assert_eq!(report.counts.generated_seeds, 2);
        assert_eq!(report.counts.replay_reports, 2);
        assert_eq!(report.counts.minimized_replays, 2);
        assert!(report.counts.boundary_events >= 4);
        assert_eq!(report.counts.oracle_failures, 0);
        assert!(
            report.counts.interleaving_depth_max >= 2,
            "generated lane must drive >= 2 provider turns concurrently, got {}",
            report.counts.interleaving_depth_max
        );
        assert!(report.counts.interleaving_depth_min >= 1);
        assert!(report.counts.interleaving_depth_min <= report.counts.interleaving_depth_max);
        assert_eq!(report.scenario_contracts.len(), 4);
        let expected_contract_slices = report
            .scenario_contracts
            .iter()
            .map(|manifest| manifest.contract_count)
            .sum::<usize>();
        assert_eq!(
            report.scenario_contract_slices.len(),
            expected_contract_slices
        );
        assert_eq!(
            report.counts.scenario_contract_slices,
            expected_contract_slices
        );
        assert_eq!(
            report.scenario_contract_packages.len(),
            expected_contract_slices
        );
        assert_eq!(
            report.counts.scenario_contract_packages,
            expected_contract_slices
        );
        let package_oracle_ids = report
            .scenario_contract_packages
            .iter()
            .map(|package| package.oracle_id.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            package_oracle_ids.len(),
            report.scenario_contract_packages.len(),
            "scenario packages must not share backing oracle ids"
        );
        assert!(
            report.scenario_contract_packages.iter().all(|package| {
                package.classification == "per_contract_oracle"
                    && package.oracle_id.ends_with(package.test_name)
                    && !package.oracle_id.ends_with(":coverage-manifest")
            }),
            "every scenario package must be backed by its own per-contract oracle"
        );
        let mut package_fact_graphs = BTreeMap::new();
        for slice in &report.scenario_contract_slices {
            let fingerprint =
                transition_fact_graph_fingerprint(&slice.generated_shape.transition_facts);
            assert!(
                package_fact_graphs
                    .insert(fingerprint, slice.test_name)
                    .is_none(),
                "scenario packages must not share identical generated transition fact graphs"
            );
        }
        assert_eq!(report.generated_backend_regression_fixtures.len(), 8);
        assert_eq!(report.counts.generated_backend_regression_fixtures, 8);
        let backend_regression_ids = report
            .generated_backend_regression_fixtures
            .iter()
            .map(|fixture| fixture.fixture_id)
            .collect::<BTreeSet<_>>();
        for fixture_id in [
            "queued-active-turn-cancel-race",
            "trigger-wakeup-routes-process",
            "duplicate-process-wake-idempotency",
            "worker-stale-completion-fenced",
            "durable-effect-crash-reopen-replay",
            "backend-retry-terminalization",
            "provider-protocol-terminalization",
            "rlm-standard-protocol-terminal-boundaries",
        ] {
            assert!(
                backend_regression_ids.contains(fixture_id),
                "missing generated backend regression fixture {fixture_id}"
            );
        }
        for fixture in &report.generated_backend_regression_fixtures {
            assert_eq!(fixture.status, "generated_cross_backend_valid_trace");
            assert!(tmp.path().join(&fixture.trace_path).exists());
            assert!(tmp.path().join(&fixture.package_path).exists());
            assert_eq!(fixture.replay_backends, vec!["model"]);
            assert_eq!(
                fixture.static_backend_replay_policy,
                "not_claimed_for_generated_scheduler_traces"
            );
            assert!(
                tmp.path()
                    .join(&fixture.source_sqlite_replay_report_path)
                    .exists()
            );
            assert!(
                fixture
                    .semantic_oracles
                    .contains(&"sim.oracle.state-machine-semantic-invariants.v1")
            );
        }
        assert!(
            report
                .scenario_contracts
                .iter()
                .any(|manifest| manifest.suite == "runtime" && manifest.contract_count == 9)
        );
        for suite in ["runtime", "standard", "rlm", "agent"] {
            assert!(
                report
                    .scenario_contract_slices
                    .iter()
                    .any(|slice| slice.suite == suite),
                "missing generated scenario slice suite {suite}"
            );
            assert!(
                report
                    .scenario_contract_packages
                    .iter()
                    .any(|package| package.suite == suite),
                "missing generated scenario package suite {suite}"
            );
        }
        let negative_fixtures = report
            .scenario_contract_packages
            .iter()
            .map(|slice| (slice.suite, slice.negative.fixture_id))
            .collect::<BTreeSet<_>>();
        for (suite, fixture_id) in [
            ("runtime", "operational-coverage-missing-cancellation"),
            ("runtime", "queued-input-operational-missing"),
            ("runtime", "trigger-wakeup-operational-missing"),
            ("runtime", "process-wake-operational-missing"),
            ("standard", "standard-provider-error-missing-parser-matrix"),
            ("rlm", "rlm-lashlang-cell-missing-exec-outcome"),
            ("agent", "agent-parallel-join-missing-wake-session"),
        ] {
            assert!(
                negative_fixtures.contains(&(suite, fixture_id)),
                "missing {suite} negative fixture {fixture_id}"
            );
        }
        let operational_cases = report
            .scenario_contract_packages
            .iter()
            .flat_map(|package| package.operational_cases.iter().map(String::as_str))
            .collect::<BTreeSet<_>>();
        for case in [
            "queueing-inputs",
            "active-turn-input-queueing",
            "triggers-wakeups",
            "cancellation",
            "duplicate-replayed-inputs",
            "backend-retry",
            "lease-fencing",
            "provider-failure",
            "worker-failover",
            "rlm-lashlang-exec",
            "tool-loop",
            "durable-effect",
        ] {
            assert!(
                operational_cases.contains(case),
                "scenario packages did not cover operational case {case}"
            );
        }
        for package in &report.scenario_contract_packages {
            assert_eq!(package.status, "generated_replay_package_written");
            assert!(package.oracle_id.starts_with("sim.oracle.scenario."));
            assert!(!package.operational_cases.is_empty());
            assert!(!package.positive.selected_boundary_ids.is_empty());
            assert_eq!(package.positive.oracle_status, OracleStatus::Passed);
            assert!(package.positive.selected_event_count > 0);
            assert!(!package.positive.source_trace_paths.is_empty());
            for path in package
                .positive
                .source_trace_paths
                .iter()
                .chain(package.positive.replay_report_paths.iter())
                .chain(package.positive.sqlite_replay_report_paths.iter())
            {
                assert!(
                    tmp.path().join(path).exists(),
                    "missing positive replay artifact {path} for {}",
                    package.package_id
                );
            }
            assert!(
                package.negative.fixture_path.ends_with(".json"),
                "negative fixture path should be concrete for {}",
                package.package_id
            );
            let package_path = tmp.path().join(&package.package_path);
            assert!(
                package_path.exists(),
                "missing scenario package artifact {package_path:?}"
            );
            let body = std::fs::read_to_string(&package_path).expect("package artifact");
            let artifact: serde_json::Value = serde_json::from_str(&body).expect("package JSON");
            assert_eq!(artifact["schema"], "lash.sim.scenario-contract-package.v1");
            assert_eq!(artifact["package_id"], package.package_id);
            assert_eq!(artifact["classification"], "per_contract_oracle");
            assert_eq!(artifact["positive"]["oracle_status"], "passed");
            assert!(
                !artifact["generated_shape"]["transition_facts"]
                    .as_array()
                    .unwrap()
                    .is_empty(),
                "scenario package {} must include transition facts",
                package.package_id
            );
            assert!(!artifact["events"].as_array().unwrap().is_empty());
        }
        for slice in &report.scenario_contract_slices {
            assert_eq!(slice.status, "generated_trace_slice_written");
            assert!(slice.oracle_id.starts_with("sim.oracle.scenario."));
            assert_eq!(
                slice.generated_shape.schema,
                "lash.sim.scenario-contract-generated-shape.v1"
            );
            assert_eq!(slice.generated_shape.semantic_oracle, slice.semantic_oracle);
            assert!(
                slice.generated_shape.transition_kind.contains(slice.suite),
                "transition kind should name suite for {}",
                slice.test_name
            );
            assert!(
                slice
                    .generated_shape
                    .required_evidence
                    .iter()
                    .all(|evidence| evidence.selected_event_count > 0),
                "required evidence map must point to generated events for {}",
                slice.test_name
            );
            let fixture_path =
                std::path::Path::new(slice.generated_shape.negative_fixture.fixture_path);
            let manifest_relative_fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(
                slice
                    .generated_shape
                    .negative_fixture
                    .fixture_path
                    .strip_prefix("crates/lash-sim/")
                    .unwrap_or(slice.generated_shape.negative_fixture.fixture_path),
            );
            assert!(
                fixture_path.exists() || manifest_relative_fixture.exists(),
                "negative fixture path must exist for {}",
                slice.test_name
            );
            assert!(slice.selected_event_count > 0);
            assert!(!slice.selected_evidence.is_empty());
            let slice_path = tmp.path().join(&slice.artifact_path);
            assert!(slice_path.exists(), "missing slice artifact {slice_path:?}");
            let body = std::fs::read_to_string(&slice_path).expect("slice artifact");
            let artifact: serde_json::Value = serde_json::from_str(&body).expect("slice JSON");
            assert_eq!(artifact["schema"], "lash.sim.scenario-contract-slice.v1");
            assert_eq!(artifact["contract"]["suite"], slice.suite);
            assert_eq!(artifact["contract"]["test_name"], slice.test_name);
            assert_eq!(artifact["oracle_id"], slice.oracle_id);
            assert_eq!(
                artifact["generated_shape"]["transition_kind"],
                slice.generated_shape.transition_kind
            );
            assert!(
                !artifact["generated_shape"]["transition_facts"]
                    .as_array()
                    .unwrap()
                    .is_empty(),
                "scenario slice {} must include transition facts",
                slice.test_name
            );
            assert!(!artifact["events"].as_array().unwrap().is_empty());
            assert!(!artifact["verdicts"].as_array().unwrap().is_empty());
        }
        let runtime_transition_facts = report
            .scenario_contract_slices
            .iter()
            .filter(|slice| slice.suite == "runtime")
            .map(|slice| {
                (
                    slice.semantic_oracle,
                    slice
                        .generated_shape
                        .transition_facts
                        .iter()
                        .map(|fact| fact.fact.as_str())
                        .collect::<BTreeSet<_>>(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        assert_eq!(
            runtime_transition_facts.len(),
            9,
            "every runtime scenario contract must have generated transition facts"
        );
        for (semantic_oracle, facts) in &runtime_transition_facts {
            assert!(
                !facts.contains("generated_transition_evidence_present"),
                "runtime contract {semantic_oracle} must not use generic selected-event fallback"
            );
        }
        assert!(
            runtime_transition_facts
                .get("runtime.command_only_queue_drain")
                .is_some_and(|facts| facts.contains("command_queue_drains_with_real_lease_fence")),
            "command-only runtime contract must assert queued source keys plus real lease fencing"
        );
        assert!(
            runtime_transition_facts
                .get("runtime.observation_replay_preserves_input")
                .is_some_and(
                    |facts| facts.contains("observer_reconnect_replays_original_input_state")
                ),
            "observer replay runtime contract must assert concrete reconnect observation state"
        );
        let backend_linked_contracts = report
            .scenario_contract_slices
            .iter()
            .filter(|slice| slice.generated_shape.generated_backend_regression.is_some())
            .count();
        assert!(
            backend_linked_contracts >= 8,
            "high-risk scenario contracts should link to generated backend regression fixtures"
        );
        assert!(
            report
                .model_only_boundary_reviews
                .iter()
                .any(|review| review.boundary_kind == "worker")
        );
        assert!(
            report
                .provider_transport_exclusions
                .iter()
                .any(|exclusion| exclusion.path.contains("codex.rs"))
        );
        assert!(report.oracle_verdicts.iter().any(|verdict| {
            verdict.oracle_id == "sim.oracle.operational-coverage.v1"
                && verdict.status == OracleStatus::Passed
        }));
        assert!(tmp.path().join(GENERATED_SIM_SUMMARY).exists());
        assert!(tmp.path().join(GENERATED_SIM_EVENTS).exists());
        assert!(tmp.path().join(GENERATED_SIM_PROVIDER_MANIFEST).exists());
        assert!(tmp.path().join(GENERATED_SIM_FAILURE_SHAPE).exists());
        for replay in &report.replay_reports {
            assert!(tmp.path().join(&replay.trace_path).exists());
            assert!(tmp.path().join(&replay.replay_report_path).exists());
            assert!(tmp.path().join(&replay.minimized_trace_path).exists());
            assert!(tmp.path().join(&replay.failure_package_path).exists());
            assert!(tmp.path().join(&replay.minimize_report_path).exists());
            assert!(tmp.path().join(&replay.sqlite_database_path).exists());
            assert!(tmp.path().join(&replay.sqlite_replay_report_path).exists());
            assert!(
                replay
                    .replay_command
                    .contains("lash-sim --locked -- replay")
            );
            assert!(
                replay
                    .sqlite_replay_command
                    .contains("lash-sim --locked -- replay-sqlite")
            );
        }

        let summary_body =
            std::fs::read_to_string(tmp.path().join(GENERATED_SIM_SUMMARY)).expect("summary");
        let summary: serde_json::Value = serde_json::from_str(&summary_body).expect("summary JSON");
        assert_eq!(summary["schema"], "lash.sim.profile-summary.v1");
        assert_eq!(
            summary["runtime_proof"]["assistant_message"],
            "Runtime scripted answer."
        );
        assert_eq!(summary["counts"]["oracle_failures"], 0);
        assert_eq!(summary["counts"]["backend_replays"], 2);
        assert_eq!(summary["counts"]["minimized_replays"], 2);
        assert_eq!(summary["scenario_contracts"].as_array().unwrap().len(), 4);
        assert_eq!(
            summary["scenario_contract_slices"]
                .as_array()
                .unwrap()
                .len(),
            expected_contract_slices
        );
        assert_eq!(
            summary["counts"]["scenario_contract_slices"],
            expected_contract_slices
        );
        assert_eq!(
            summary["scenario_contract_packages"]
                .as_array()
                .unwrap()
                .len(),
            expected_contract_slices
        );
        assert_eq!(
            summary["counts"]["scenario_contract_packages"],
            expected_contract_slices
        );
        assert_eq!(
            summary["generated_backend_regression_fixtures"]
                .as_array()
                .unwrap()
                .len(),
            8
        );
        assert_eq!(
            summary["counts"]["generated_backend_regression_fixtures"],
            8
        );
        assert!(
            summary["model_only_boundary_reviews"]
                .as_array()
                .unwrap()
                .iter()
                .any(|review| review["boundary_kind"] == "worker")
        );
    }

    #[test]
    fn postgres_effect_history_native_claim_is_consistent_across_reviews_docs_and_gate() {
        let repo_root = repo_root_for_test();
        let mut corpus = vec![(
            "model_only_boundary_reviews".to_string(),
            serde_json::to_string(&model_only_boundary_reviews()).expect("reviews JSON"),
        )];
        for relative in [
            "scripts/confidence-gate.sh",
            "docs/deterministic-simulation-harness-plan.md",
            "docs/adr/0008-confidence-gate.md",
            "CONTEXT.md",
        ] {
            corpus.push((
                relative.to_string(),
                std::fs::read_to_string(repo_root.join(relative))
                    .unwrap_or_else(|err| panic!("read {relative}: {err}")),
            ));
        }

        for (label, body) in &corpus {
            for stale in [
                "Postgres has no native Postgres effect-history controller",
                "permanent_non_goal_without_postgres_runtime_effect_controller",
                "without_postgres_runtime_effect_controller",
                "not_available_in_lash_postgres_store",
                "postgres-effect-history-exclusion",
            ] {
                assert!(
                    !body.contains(stale),
                    "{label} still contains stale Postgres effect-history exclusion wording: {stale}"
                );
            }
        }

        let reviews = &corpus[0].1;
        assert!(reviews.contains("PostgresRuntimeEffectController"));
        assert!(reviews.contains("lash_runtime_effect_replay"));
        let script = corpus
            .iter()
            .find(|(label, _)| label == "scripts/confidence-gate.sh")
            .map(|(_, body)| body)
            .expect("confidence gate corpus");
        assert!(script.contains("native_postgres_runtime_effect_controller"));
        assert!(script.contains("postgres-effect-history-status.json"));
    }

    fn repo_root_for_test() -> std::path::PathBuf {
        let mut cursor = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        loop {
            if cursor.join("scripts/confidence-gate.sh").is_file()
                && cursor
                    .join("docs/deterministic-simulation-harness-plan.md")
                    .is_file()
            {
                return cursor;
            }
            if !cursor.pop() {
                panic!(
                    "could not locate repository root from CARGO_MANIFEST_DIR={}",
                    env!("CARGO_MANIFEST_DIR")
                );
            }
        }
    }

    #[test]
    fn runtime_completion_ready_gates_provider_tool_durable_worker_boundaries() {
        let mut state = RuntimeCompletionState::default();
        let provider_one = BoundaryEvent::new(
            "session-001:provider:001",
            "session-001",
            BoundaryKind::Provider,
            4,
            "provider.chat.stream",
            json!({"turn_index": 1, "provider_kind": "openai-compatible", "text": "one"}),
        );
        let provider_two = BoundaryEvent::new(
            "session-001:provider:002",
            "session-001",
            BoundaryKind::Provider,
            5,
            "provider.chat.stream",
            json!({"turn_index": 2, "provider_kind": "openai-compatible", "text": "two"}),
        );
        let tool = BoundaryEvent::new(
            "session-001:tool:001",
            "session-001",
            BoundaryKind::Tool,
            6,
            "tool.return",
            json!({}),
        );
        let durable_first = BoundaryEvent::new(
            "session-001:durable:001",
            "session-001",
            BoundaryKind::DurableEffect,
            7,
            "durable.effect.complete",
            json!({"durable_key": "sleep/session-001/001"}),
        );
        let durable_replay = BoundaryEvent::new(
            "session-001:durable:001:replay",
            "session-001",
            BoundaryKind::DurableEffect,
            8,
            "durable.effect.replay",
            json!({"durable_key": "sleep/session-001/001"}),
        );
        let worker = BoundaryEvent::new(
            "worker-001:stale-completion",
            "worker-001",
            BoundaryKind::Worker,
            9,
            "worker.stale-completion-rejected",
            json!({"session": "session-001"}),
        );

        assert!(!runtime_completion_ready(&provider_one, &state));
        assert!(!runtime_completion_ready(&tool, &state));
        assert!(!runtime_completion_ready(&worker, &state));
        state.observe(&test_delivered(
            0,
            "session-001:ingress",
            "session-001",
            BoundaryKind::Ingress,
            json!({}),
        ));
        assert!(runtime_completion_ready(&provider_one, &state));
        assert!(!runtime_completion_ready(&provider_two, &state));
        assert!(!runtime_completion_ready(&tool, &state));
        assert!(runtime_completion_ready(&durable_first, &state));
        assert!(!runtime_completion_ready(&durable_replay, &state));
        assert!(runtime_completion_ready(&worker, &state));

        state.observe(&test_delivered(
            1,
            "session-001:provider:001",
            "session-001",
            BoundaryKind::Provider,
            json!({"provider_output": "one"}),
        ));
        assert!(runtime_completion_ready(&provider_two, &state));
        assert!(runtime_completion_ready(&tool, &state));

        state.observe(&test_delivered(
            2,
            "session-001:durable:001",
            "session-001",
            BoundaryKind::DurableEffect,
            json!({
                "durable_key": "sleep/session-001/001",
                "replayed": false
            }),
        ));
        assert!(runtime_completion_ready(&durable_replay, &state));
    }

    #[test]
    fn provider_runtime_completion_registration_does_not_schedule_turn_completion_immediately() {
        let mut state = RuntimeCompletionState::default();
        state.observe(&test_delivered(
            0,
            "session-001:ingress",
            "session-001",
            BoundaryKind::Ingress,
            json!({}),
        ));
        let registered_after = test_delivered(
            0,
            "session-001:ingress",
            "session-001",
            BoundaryKind::Ingress,
            json!({}),
        );
        let mut queue = RuntimeCompletionQueue::new([
            BoundaryEvent::new(
                "session-001:provider:001",
                "session-001",
                BoundaryKind::Provider,
                2,
                "provider.chat.stream",
                json!({
                    "provider_kind": "openai-compatible",
                    "text": "scheduled answer",
                    "turn_index": 1
                }),
            ),
            BoundaryEvent::new(
                "session-001:provider:002",
                "session-001",
                BoundaryKind::Provider,
                3,
                "provider.chat.stream",
                json!({
                    "provider_kind": "openai-compatible",
                    "text": "later answer",
                    "turn_index": 2
                }),
            ),
        ]);
        let ready = queue.take_ready(|event| runtime_completion_ready(event, &state));
        assert_eq!(ready.len(), 1);
        let event = ready.into_iter().next().expect("provider ready");
        let units = runtime_completion_units(&event).expect("provider completion units");
        let (_pending, completion_event) = queue.register_pending_event(
            event,
            &registered_after,
            "provider_turn_completion",
            units,
        );

        assert_eq!(queue.registered_len(), 1);
        assert_eq!(queue.pending_ids(), vec!["session-001:provider:002"]);
        assert_eq!(completion_event.boundary_id, "session-001:provider:001");
        assert!(completion_event.payload.get("runtime_completion").is_some());
        assert_eq!(
            completion_event
                .payload
                .pointer("/runtime_completion/completion_family")
                .and_then(Value::as_str),
            Some("provider_turn_completion")
        );
        assert!(
            completion_event
                .payload
                .pointer("/runtime_completion/completion_units")
                .and_then(Value::as_array)
                .is_some_and(|units| units.iter().any(|unit| unit
                    .get("unit")
                    .and_then(Value::as_str)
                    .is_some_and(|name| name.contains("provider:"))))
        );
    }

    #[test]
    fn runtime_completion_backend_mutation_idle_session_mutation_guard() {
        let backend_failure = BoundaryEvent::new(
            "session-001:backend-failure:001",
            "session-001",
            BoundaryKind::BackendFailure,
            4,
            "backend.failure",
            json!({}),
        );
        let provider_mutation = BoundaryEvent::new(
            "session-001:provider-mutation:001",
            "session-001",
            BoundaryKind::ProviderMutation,
            5,
            "provider.mutation",
            json!({}),
        );
        let mut state = RuntimeCompletionState::default();

        assert!(
            !runtime_completion_ready(&backend_failure, &state),
            "backend failure must not run before the session opens"
        );
        assert!(
            !runtime_completion_ready(&provider_mutation, &state),
            "provider mutation must not run before the session opens"
        );

        state.observe(&test_delivered(
            0,
            "session-001:ingress",
            "session-001",
            BoundaryKind::Ingress,
            json!({}),
        ));
        assert!(
            runtime_completion_ready(&backend_failure, &state),
            "backend failure is ready once its session is open and idle"
        );
        assert!(
            runtime_completion_ready(&provider_mutation, &state),
            "provider mutation is ready once its session is open and idle"
        );

        state.provider_started("session-001");
        assert!(
            !runtime_completion_ready(&backend_failure, &state),
            "backend failure must not interleave with an active provider turn for the same session"
        );
        assert!(
            !runtime_completion_ready(&provider_mutation, &state),
            "provider mutation must not interleave with an active provider turn for the same session"
        );
    }

    #[test]
    fn script_bundle_hash_is_stable_for_current_bundle() {
        let scripts = script_hash_manifest().expect("scripts");
        let hash = script_bundle_hash(&scripts);

        assert_eq!(hash.len(), 64);
        assert_eq!(hash, script_bundle_hash(&scripts));
    }

    fn test_delivered(
        sequence: usize,
        boundary_id: &str,
        actor_alias: &str,
        kind: BoundaryKind,
        observed: Value,
    ) -> crate::scheduler::DeliveredBoundary {
        crate::scheduler::DeliveredBoundary {
            schema: crate::scheduler::BOUNDARY_EVENT_SCHEMA.to_string(),
            sequence,
            scheduler: crate::scheduler::SchedulerDeliveryEvidence {
                scheduler_controlled: true,
                delivered_at: sequence as u64,
                ..crate::scheduler::SchedulerDeliveryEvidence::default()
            },
            boundary_id: boundary_id.to_string(),
            actor_alias: actor_alias.to_string(),
            kind,
            at: sequence as u64,
            label: format!("{kind:?}"),
            payload: json!({}),
            observed,
        }
    }
}
