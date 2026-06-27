#![allow(clippy::result_large_err)]

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use lash_core::llm::transport::{LlmTransportError, ProviderFailureKind};
use lash_core::llm::types::{
    LlmEventSender, LlmMessage, LlmOutputPart, LlmRequest, LlmResponse, LlmRole, LlmStreamEvent,
    LlmTerminalReason, LlmToolChoice, LlmToolSpec,
};
use lash_core::provider::{
    DefaultProviderFailureClassifier, Provider, ProviderFailureClassifier, ProviderHandle,
    ProviderOptions, ProviderReliability, ProviderRetryPolicy,
};
use lash_llm_transport::LlmHttpTransport;
use lash_provider_anthropic::AnthropicProvider;
use lash_provider_openai::{OpenAiCompatibleProvider, OpenAiProvider};
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::effects::{DurableEffectJournal, sleep_effect_envelope};
use crate::generator::{GENERATOR_VERSION, GeneratedWorkload, generate_workload};
use crate::oracles::{
    combine_oracles, cross_session_isolation, durable_effect_exactly_once, ingress_sessions_opened,
    lease_time_monotonic, observer_convergence, runtime_provider_turn,
    runtime_session_graph_contract, worker_stale_completion_rejected,
};
use crate::provider::{
    ProviderWireEvent, ProviderWireHeader, ProviderWireScript, ScriptedLlmHttpExchange,
    ScriptedLlmHttpTransport, ScriptedTransportGate,
};
use crate::replay::{ReplayError, replay_trace};
use crate::runtime_contracts::{RuntimeTurnObservation, require_passed, runtime_turn_contract};
use crate::scheduler::{BoundaryDeliveryLog, BoundaryEvent, BoundaryKind, BoundaryScheduler};
use crate::sqlite_replay::{SqliteReplayError, replay_trace_to_sqlite};
use crate::store::ModelStore;
use crate::trace::{
    OracleStatus, OracleVerdict, SimulationTrace, TraceEventLine, TraceIoError, write_event_lines,
    write_replay_report, write_trace,
};
use crate::workers::SimWorkerTopology;

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

const OPENAI_COMPAT_TOOL_CALL: &str = include_str!(
    "../provider-scripts/canonical/openai-compatible.chat-tool-call-split-stream.json"
);
const OPENAI_COMPAT_RATE_LIMIT: &str =
    include_str!("../provider-scripts/canonical/openai-compatible.chat-rate-limit-429.json");
const OPENAI_COMPAT_VALIDATION: &str =
    include_str!("../provider-scripts/canonical/openai-compatible.chat-validation-error.json");
const OPENAI_COMPAT_DISCONNECT: &str =
    include_str!("../provider-scripts/canonical/openai-compatible.chat-mid-stream-disconnect.json");
const OPENAI_COMPAT_RESPONSE_START_TIMEOUT: &str = include_str!(
    "../provider-scripts/canonical/openai-compatible.chat-response-start-timeout.json"
);
const OPENAI_COMPAT_STREAM_CHUNK_TIMEOUT: &str =
    include_str!("../provider-scripts/canonical/openai-compatible.chat-stream-chunk-timeout.json");
const OPENAI_RESPONSES_TEXT: &str =
    include_str!("../provider-scripts/canonical/openai.responses-text-stream.json");
const ANTHROPIC_MESSAGES_TEXT: &str =
    include_str!("../provider-scripts/canonical/anthropic.messages-text-stream.json");
const OPENAI_COMPAT_RUNTIME_TEXT: &str =
    include_str!("../provider-scripts/runtime/openai-compatible.chat-runtime-text-stream.json");

#[derive(Clone, Copy)]
struct CanonicalScript {
    path: &'static str,
    content: &'static str,
}

const CANONICAL_SCRIPTS: &[CanonicalScript] = &[
    CanonicalScript {
        path: "provider-scripts/canonical/anthropic.messages-text-stream.json",
        content: ANTHROPIC_MESSAGES_TEXT,
    },
    CanonicalScript {
        path: "provider-scripts/canonical/openai-compatible.chat-mid-stream-disconnect.json",
        content: OPENAI_COMPAT_DISCONNECT,
    },
    CanonicalScript {
        path: "provider-scripts/canonical/openai-compatible.chat-rate-limit-429.json",
        content: OPENAI_COMPAT_RATE_LIMIT,
    },
    CanonicalScript {
        path: "provider-scripts/canonical/openai-compatible.chat-response-start-timeout.json",
        content: OPENAI_COMPAT_RESPONSE_START_TIMEOUT,
    },
    CanonicalScript {
        path: "provider-scripts/canonical/openai-compatible.chat-stream-chunk-timeout.json",
        content: OPENAI_COMPAT_STREAM_CHUNK_TIMEOUT,
    },
    CanonicalScript {
        path: "provider-scripts/canonical/openai-compatible.chat-tool-call-split-stream.json",
        content: OPENAI_COMPAT_TOOL_CALL,
    },
    CanonicalScript {
        path: "provider-scripts/canonical/openai-compatible.chat-validation-error.json",
        content: OPENAI_COMPAT_VALIDATION,
    },
    CanonicalScript {
        path: "provider-scripts/canonical/openai.responses-text-stream.json",
        content: OPENAI_RESPONSES_TEXT,
    },
];

#[derive(Debug)]
pub enum FixedScriptRunnerError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Provider(LlmTransportError),
    Trace(TraceIoError),
    Replay(ReplayError),
    SqliteReplay(SqliteReplayError),
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

#[derive(Clone, Debug, Serialize)]
pub struct FixedScriptManifest {
    pub schema: &'static str,
    pub profile: &'static str,
    pub timeline_at_semantics: &'static str,
    pub script_bundle_hash: String,
    pub events_path: &'static str,
    pub events_sha256: String,
    pub event_count: usize,
    pub scripts: Vec<ScriptHashManifest>,
    pub proofs: Vec<FixedScriptProof>,
    pub summary: FixedScriptSummary,
    #[serde(skip)]
    pub manifest_path: PathBuf,
    #[serde(skip)]
    pub summary_path: PathBuf,
}

#[derive(Clone, Debug, Serialize)]
pub struct ScriptHashManifest {
    pub path: String,
    pub name: String,
    pub provider_kind: String,
    pub sha256: String,
    pub bytes: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct FixedScriptProof {
    pub name: String,
    pub provider_kind: String,
    pub endpoint: String,
    pub outcome: String,
    pub transcript_path: String,
    pub transcript_sha256: String,
    pub observed: serde_json::Value,
}

#[derive(Clone, Debug, Serialize)]
pub struct FixedScriptSummary {
    pub total_scripts: usize,
    pub total_proofs: usize,
    pub total_events: usize,
    pub passed: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct GeneratedSimProfileReport {
    pub schema: &'static str,
    pub profile: String,
    pub generator_version: &'static str,
    pub script_bundle_hash: String,
    pub provider_manifest_path: &'static str,
    pub events_path: &'static str,
    pub events_sha256: String,
    pub replay_reports: Vec<GeneratedReplayArtifact>,
    pub runtime_proof: RuntimeFacadeProof,
    pub counts: GeneratedSimCounts,
    pub oracle_verdicts: Vec<OracleVerdict>,
    pub failure_artifact_shape: &'static str,
    #[serde(skip)]
    pub summary_path: PathBuf,
}

#[derive(Clone, Debug, Serialize)]
pub struct GeneratedReplayArtifact {
    pub seed: u64,
    pub trace_path: String,
    pub trace_sha256: String,
    pub replay_report_path: String,
    pub replay_report_sha256: String,
    pub sqlite_database_path: String,
    pub sqlite_replay_report_path: String,
    pub sqlite_replay_report_sha256: String,
    pub replay_command: String,
    pub sqlite_replay_command: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct GeneratedSimCounts {
    pub generated_seeds: usize,
    pub boundary_events: usize,
    pub fixed_provider_proofs: usize,
    pub runtime_proofs: usize,
    pub replay_reports: usize,
    pub backend_replays: usize,
    pub oracle_passes: usize,
    pub oracle_failures: usize,
    pub model_store_sessions: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct RuntimeFacadeProof {
    pub schema: &'static str,
    pub name: &'static str,
    pub provider_kind: String,
    pub session_id: String,
    pub turn_index: usize,
    pub assistant_message: String,
    pub provider_exchange_count: usize,
    pub runtime_invariant: OracleVerdict,
    pub provider_output_invariant: OracleVerdict,
}

#[derive(Clone, Debug, Serialize)]
struct ProviderScriptProfileManifest {
    schema: &'static str,
    fixed_script_manifest: String,
    fixed_script_summary: String,
    script_bundle_hash: String,
    scripts: Vec<ScriptHashManifest>,
}

#[derive(Clone, Debug, Serialize)]
struct FailureArtifactShape {
    schema: &'static str,
    directory: &'static str,
    trace: &'static str,
    replay_report: &'static str,
    oracle: &'static str,
    final_summary: &'static str,
}

#[derive(Clone, Debug, Serialize)]
struct FixedScriptLaneSummary {
    schema: &'static str,
    profile: &'static str,
    generator_version: &'static str,
    replay_schema_version: &'static str,
    script_bundle_hash: String,
    provider_set: Vec<String>,
    backend_set: Vec<String>,
    counts: FixedScriptLaneCounts,
    fixed_script_manifest: &'static str,
    fixed_script_events: &'static str,
}

#[derive(Clone, Debug, Serialize)]
struct FixedScriptLaneCounts {
    generated_seeds: usize,
    fixed_script_events: usize,
    fixed_replays: usize,
    minimized_replays: usize,
    backend_replays: usize,
    boundary_events: usize,
    oracle_passes: usize,
    oracle_failures: usize,
    divergences: usize,
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
    ];
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

pub async fn run_generated_sim_profile(
    artifact_root: impl AsRef<Path>,
    profile: &str,
    seeds: usize,
    max_boundaries: usize,
) -> Result<GeneratedSimProfileReport, FixedScriptRunnerError> {
    let artifact_root = artifact_root.as_ref();
    std::fs::create_dir_all(artifact_root)?;

    let provider_dir = artifact_root.join("provider-corpus");
    let fixed_manifest = run_fixed_script_profile(&provider_dir).await?;
    write_provider_script_manifest(artifact_root, &fixed_manifest)?;

    let runtime_proof = prove_runtime_facade_turn().await?;
    let seed_count = seeds.max(1);
    let boundary_limit = max_boundaries.max(1);
    let replay_dir = artifact_root.join("replays");
    std::fs::create_dir_all(&replay_dir)?;

    let mut event_lines = Vec::new();
    let mut replay_reports = Vec::new();
    let mut oracle_verdicts = Vec::new();
    let mut model_store_sessions = 0;
    let mut boundary_events = 0;
    let mut runtime_turn_proofs = 0;

    for seed_index in 0..seed_count {
        let seed = generated_seed(profile, seed_index);
        let workload = generate_workload(seed, profile, boundary_limit);
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
        let sqlite_database_path = replay_dir.join(format!("seed-{seed:016x}.sqlite.db"));
        let sqlite_replay_report_path =
            replay_dir.join(format!("seed-{seed:016x}.sqlite-replay.json"));
        let sqlite_replay = replay_trace_to_sqlite(
            &trace_path,
            &trace,
            &sqlite_database_path,
            Some(&sqlite_replay_report_path),
        )?;
        oracle_verdicts.push(sqlite_replay.terminal_verdict.clone());
        let sqlite_replay_report_sha256 = file_sha256(&sqlite_replay_report_path)?;
        replay_reports.push(GeneratedReplayArtifact {
            seed,
            trace_path: relative_path(artifact_root, &trace_path),
            trace_sha256,
            replay_report_path: relative_path(artifact_root, &replay_report_path),
            replay_report_sha256,
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

    let events_sha256 = write_event_lines(&artifact_root.join(GENERATED_SIM_EVENTS), &event_lines)?;
    write_failure_artifact_shape(artifact_root)?;

    let oracle_passes = oracle_verdicts
        .iter()
        .filter(|verdict| verdict.status == OracleStatus::Passed)
        .count();
    let oracle_failures = oracle_verdicts.len() - oracle_passes;
    let summary_path = artifact_root.join(GENERATED_SIM_SUMMARY);
    let report = GeneratedSimProfileReport {
        schema: "lash.sim.profile-summary.v1",
        profile: profile.to_string(),
        generator_version: GENERATOR_VERSION,
        script_bundle_hash: fixed_manifest.script_bundle_hash.clone(),
        provider_manifest_path: GENERATED_SIM_PROVIDER_MANIFEST,
        events_path: GENERATED_SIM_EVENTS,
        events_sha256,
        replay_reports,
        runtime_proof,
        counts: GeneratedSimCounts {
            generated_seeds: seed_count,
            boundary_events,
            fixed_provider_proofs: fixed_manifest.summary.total_proofs,
            runtime_proofs: runtime_turn_proofs + 1,
            replay_reports: seed_count,
            backend_replays: seed_count,
            oracle_passes,
            oracle_failures,
            model_store_sessions,
        },
        oracle_verdicts,
        failure_artifact_shape: GENERATED_SIM_FAILURE_SHAPE,
        summary_path: summary_path.clone(),
    };
    std::fs::write(&summary_path, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

#[cfg(test)]
pub(crate) async fn run_generated_workload_for_test(
    workload: GeneratedWorkload,
    script_bundle_hash: &str,
) -> Result<SimulationTrace, FixedScriptRunnerError> {
    run_generated_workload(workload, script_bundle_hash, Path::new("trace.json")).await
}

async fn run_generated_workload(
    workload: GeneratedWorkload,
    script_bundle_hash: &str,
    trace_path: &Path,
) -> Result<SimulationTrace, FixedScriptRunnerError> {
    let mut scheduler = BoundaryScheduler::with_events(workload.seed, workload.boundaries.clone());
    let mut store = ModelStore::default();
    let mut world = GeneratedRuntimeWorld::default();
    let mut log = BoundaryDeliveryLog::default();
    while let Some(mut delivered) = scheduler.deliver_next(Value::Null) {
        let event = delivered.as_event();
        let observed = world.deliver_boundary(&event).await?;
        store.apply_observed_boundary(&event, &observed);
        delivered.observed = observed;
        log.push(delivered);
    }
    let events = log.into_vec();
    let final_summary = store.summary();
    let oracles = vec![
        ingress_sessions_opened(&final_summary),
        cross_session_isolation(&final_summary),
        observer_convergence(&final_summary),
        runtime_session_graph_contract(&final_summary),
        durable_effect_exactly_once(&final_summary),
        worker_stale_completion_rejected(&final_summary),
        lease_time_monotonic(&final_summary),
    ];
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
struct GeneratedRuntimeWorld {
    sessions: BTreeMap<String, GeneratedRuntimeSession>,
    durable_journal: DurableEffectJournal,
    worker_topology: SimWorkerTopology,
    lease_ticks: BTreeMap<String, Vec<u64>>,
}

#[derive(Clone)]
struct GeneratedRuntimeSession {
    _core: lash::StandardCore,
    session: lash::LashSession,
    transport: Arc<ScriptedLlmHttpTransport>,
    provider_kind: String,
}

impl GeneratedRuntimeWorld {
    async fn deliver_boundary(
        &mut self,
        event: &BoundaryEvent,
    ) -> Result<Value, FixedScriptRunnerError> {
        match event.kind {
            BoundaryKind::Ingress => self.open_runtime_session(event).await,
            BoundaryKind::Provider => self.run_provider_turn(event).await,
            BoundaryKind::Observer => self.observe_session(event),
            BoundaryKind::DurableEffect => Ok(self.complete_durable_effect(event)),
            BoundaryKind::Worker => Ok(self.run_worker_boundary(event)),
            BoundaryKind::LeaseTime => Ok(self.advance_lease_time(event)),
            BoundaryKind::Tool => Ok(project_tool_boundary(event)),
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
        let scripts = runtime_scripts_for_texts(&provider_texts)?;
        let (core, transport, provider_kind) = runtime_core_for_scripts(scripts)?;
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
                provider_kind,
            },
        );
        Ok(json!({
            "session": event.actor_alias,
            "opened": true,
            "ingress_count": 1,
        }))
    }

    async fn run_provider_turn(
        &self,
        event: &BoundaryEvent,
    ) -> Result<Value, FixedScriptRunnerError> {
        let runtime_session = self.sessions.get(&event.actor_alias).ok_or_else(|| {
            FixedScriptRunnerError::Assertion(format!(
                "provider boundary `{}` ran before ingress for `{}`",
                event.boundary_id, event.actor_alias
            ))
        })?;
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
        let output = runtime_session
            .session
            .turn(lash::TurnInput::text(format!(
                "Run generated provider turn {}.",
                event.boundary_id
            )))
            .run()
            .await
            .map_err(|err| FixedScriptRunnerError::Runtime(err.to_string()))?;
        let assistant_message = output.assistant_message().unwrap_or_default().to_string();
        let read_view = output.result.state.read_view();
        let graph_node_count = output.result.state.session_graph.nodes.len();
        let transcript_message_count = read_view.messages().len();
        let provider_exchange_count =
            transport_exchanges(runtime_session.transport.as_ref())?.len();
        if !output.is_success() {
            return Err(FixedScriptRunnerError::Assertion(format!(
                "runtime turn `{}` did not succeed",
                event.boundary_id
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
            "provider_kind": runtime_session.provider_kind,
            "runtime_invariants": {
                "session_id": true,
                "turn_index": true,
                "graph_non_empty": graph_node_count > 0,
                "transcript_contains_provider_output": read_view.messages().iter().any(|message| {
                    message.parts.iter().any(|part| part.content.contains(expected_text))
                }),
                "activity_count_nonzero": !output.activities.is_empty(),
            },
            "runtime_contract": runtime_contract,
        }))
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

    fn complete_durable_effect(&mut self, event: &BoundaryEvent) -> Value {
        let durable_key = event
            .payload
            .get("durable_key")
            .and_then(Value::as_str)
            .unwrap_or(&event.boundary_id)
            .to_string();
        let result = event
            .payload
            .get("result")
            .cloned()
            .unwrap_or_else(|| json!({"completed": true}));
        let runtime_effect = event.payload.get("runtime_effect");
        if let Some(runtime_effect) = runtime_effect {
            let effect_id = runtime_effect
                .get("effect_id")
                .and_then(Value::as_str)
                .unwrap_or(&event.boundary_id);
            let duration_ms = runtime_effect
                .get("duration_ms")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let envelope = sleep_effect_envelope(
                event.actor_alias.clone(),
                effect_id.to_string(),
                durable_key,
                duration_ms,
            );
            self.durable_journal
                .complete_runtime_effect(envelope, result)
        } else {
            self.durable_journal.complete(durable_key, result)
        }
    }

    fn run_worker_boundary(&mut self, event: &BoundaryEvent) -> Value {
        let session_alias = event
            .payload
            .get("session")
            .and_then(Value::as_str)
            .unwrap_or(&event.actor_alias)
            .to_string();
        self.worker_topology
            .run_stale_completion_script(event.actor_alias.clone(), session_alias)
    }

    fn advance_lease_time(&mut self, event: &BoundaryEvent) -> Value {
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
            .map_or(true, |previous| previous <= tick);
        ticks.push(tick);
        json!({
            "session": event.actor_alias,
            "lease_time_tick": tick,
            "monotonic": monotonic,
        })
    }
}

fn project_tool_boundary(event: &BoundaryEvent) -> Value {
    json!({
        "session": event.actor_alias,
        "tool_output": event.payload.get("output").cloned().unwrap_or(Value::Null),
        "tool_name": event.payload.get("tool").cloned().unwrap_or(Value::Null),
    })
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

fn runtime_script_for_text(text: &str) -> Result<String, FixedScriptRunnerError> {
    let mut script: Value = serde_json::from_str(OPENAI_COMPAT_RUNTIME_TEXT)?;
    script["timeline"][1]["data"] = Value::String(
        json!({
            "choices": [
                {
                    "delta": {
                        "content": text,
                    }
                }
            ]
        })
        .to_string(),
    );
    script["expected_provider"]["text"] = Value::String(text.to_string());
    Ok(serde_json::to_string(&script)?)
}

fn runtime_scripts_for_texts(
    texts: &[String],
) -> Result<Vec<ProviderWireScript>, FixedScriptRunnerError> {
    texts
        .iter()
        .map(|text| {
            runtime_script_for_text(text)
                .and_then(|script| Ok(ProviderWireScript::from_json_str(&script)?))
        })
        .collect()
}

fn runtime_core_for_scripts(
    scripts: Vec<ProviderWireScript>,
) -> Result<(lash::StandardCore, Arc<ScriptedLlmHttpTransport>, String), FixedScriptRunnerError> {
    let transport = Arc::new(ScriptedLlmHttpTransport::from_scripts(scripts));
    let provider = OpenAiCompatibleProvider::new("test-key", "https://provider.test")
        .with_transport(provider_transport(&transport));
    let provider_kind = provider.kind().to_string();
    let provider_handle = ProviderHandle::new(provider.into_components());
    let model = lash::ModelSpec::from_token_limits("openai/gpt-5.4", None, 200_000, None)
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
    Ok((core, transport, provider_kind))
}

async fn prove_runtime_facade_turn() -> Result<RuntimeFacadeProof, FixedScriptRunnerError> {
    let transport = Arc::new(ScriptedLlmHttpTransport::from_json_str(
        OPENAI_COMPAT_RUNTIME_TEXT,
    )?);
    let provider = OpenAiCompatibleProvider::new("test-key", "https://provider.test")
        .with_transport(provider_transport(&transport));
    let provider_kind = provider.kind().to_string();
    let provider_handle = ProviderHandle::new(provider.into_components());
    let model = lash::ModelSpec::from_token_limits("openai/gpt-5.4", None, 200_000, None)
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
    })
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
    let gate = ScriptedTransportGate::closed();
    let transport = Arc::new(
        ScriptedLlmHttpTransport::from_json_str(OPENAI_COMPAT_TOOL_CALL)?
            .with_response_start_gate(gate.clone()),
    );
    let (events, sender) = event_collector();
    let mut provider = OpenAiCompatibleProvider::new("test-key", "https://provider.test")
        .with_transport(provider_transport(&transport));

    let task = tokio::spawn(async move {
        provider
            .complete(openai_compatible_request_with_events(Some(sender)))
            .await
    });
    gate.wait_until_blocked().await;
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
    let transport = ScriptedLlmHttpTransport::from_scripts([
        ProviderWireScript::from_json_str(OPENAI_COMPAT_RATE_LIMIT)?,
        ProviderWireScript::from_json_str(OPENAI_COMPAT_RATE_LIMIT)?,
    ]);
    let transport_for_assert = transport.clone();
    let provider = OpenAiCompatibleProvider::new("test-key", "https://provider.test")
        .with_options(ProviderOptions {
            reliability: ProviderReliability {
                retry: ProviderRetryPolicy {
                    max_attempts: 2,
                    base_delay_ms: 0,
                    max_delay_ms: 0,
                    jitter_ms: 0,
                    retry_after_cap_ms: Some(0),
                    enabled: true,
                },
                ..ProviderReliability::default()
            },
            ..ProviderOptions::default()
        })
        .with_transport(Arc::new(transport));
    let mut handle = ProviderHandle::new(provider.into_components());
    let err = handle
        .complete(openai_compatible_request(false))
        .await
        .expect_err("retry exhaustion should fail");
    require(
        err.status == Some(429) && err.retryable,
        "retry exhaustion did not return classified retryable 429",
    )?;
    require(
        transport_for_assert.remaining_scripts()? == 0,
        "retry exhaustion did not consume both scripted attempts",
    )?;
    proof(
        "openai-compatible.retry-exhaustion",
        "openai-compatible",
        OPENAI_COMPAT_RATE_LIMIT,
        transport_exchanges(&transport_for_assert)?,
        error_terminal(&err),
        json!({
            "status": err.status,
            "retryable": err.retryable,
            "classification": failure_classification(&err),
            "attempts_consumed": 2,
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

fn require(condition: bool, message: &'static str) -> Result<(), FixedScriptRunnerError> {
    if condition {
        Ok(())
    } else {
        Err(FixedScriptRunnerError::Assertion(message.to_string()))
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
            }),
            output_schema: json!({}),
            input_schema_projections: Vec::new(),
            output_schema_projections: Vec::new(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn fixed_script_profile_writes_deterministic_manifest() {
        let tmp = tempfile::tempdir().expect("tempdir");

        let manifest = run_fixed_script_profile(tmp.path()).await.expect("profile");

        assert_eq!(manifest.profile, FIXED_SCRIPT_PROFILE);
        assert_eq!(
            manifest.timeline_at_semantics,
            FIXED_SCRIPT_TIMELINE_AT_SEMANTICS
        );
        assert_eq!(manifest.summary.total_scripts, 8);
        assert_eq!(manifest.summary.total_proofs, 10);
        assert_eq!(manifest.summary.total_events, 11);
        assert_eq!(manifest.summary.passed, 10);
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

        let summary_body =
            std::fs::read_to_string(tmp.path().join(FIXED_SCRIPT_SUMMARY)).expect("summary");
        let summary: serde_json::Value = serde_json::from_str(&summary_body).expect("summary JSON");
        assert_eq!(summary["schema"], "lash.sim.summary.v1");
        assert_eq!(summary["profile"], FIXED_SCRIPT_PROFILE);
        assert_eq!(summary["fixed_script_manifest"], FIXED_SCRIPT_MANIFEST);
        assert_eq!(summary["counts"]["generated_seeds"], 0);
        assert_eq!(summary["counts"]["fixed_replays"], 10);
        assert_eq!(summary["counts"]["oracle_passes"], 10);
        assert_eq!(
            summary["provider_set"],
            json!(["anthropic", "openai", "openai-compatible"])
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
                "total_scripts": 8,
                "total_proofs": 10,
                "total_events": 11,
                "passed": 10
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
    }

    #[tokio::test]
    async fn generated_sim_profile_writes_trace_replay_and_provider_artifacts() {
        let tmp = tempfile::tempdir().expect("tempdir");

        let report = run_generated_sim_profile(tmp.path(), "fast-random", 2, 24)
            .await
            .expect("generated sim");

        assert_eq!(report.profile, "fast-random");
        assert_eq!(report.counts.generated_seeds, 2);
        assert_eq!(report.counts.replay_reports, 2);
        assert!(report.counts.boundary_events >= 4);
        assert_eq!(report.counts.oracle_failures, 0);
        assert!(tmp.path().join(GENERATED_SIM_SUMMARY).exists());
        assert!(tmp.path().join(GENERATED_SIM_EVENTS).exists());
        assert!(tmp.path().join(GENERATED_SIM_PROVIDER_MANIFEST).exists());
        assert!(tmp.path().join(GENERATED_SIM_FAILURE_SHAPE).exists());
        for replay in &report.replay_reports {
            assert!(tmp.path().join(&replay.trace_path).exists());
            assert!(tmp.path().join(&replay.replay_report_path).exists());
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
    }

    #[test]
    fn script_bundle_hash_is_stable_for_current_bundle() {
        let scripts = script_hash_manifest().expect("scripts");
        let hash = script_bundle_hash(&scripts);

        assert_eq!(hash.len(), 64);
        assert_eq!(hash, script_bundle_hash(&scripts));
    }
}
