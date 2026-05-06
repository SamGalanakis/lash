use std::collections::{BTreeMap, HashMap};
use std::io;
use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::Context;
use chrono::Utc;
use lash::llm::types::{LlmOutputPart, LlmResponse, LlmStreamEvent, LlmUsage};
use lash::runtime::{RuntimeTurnPhase, RuntimeTurnPhaseProbe};
use lash::testing::TestProvider;
use lash::*;
use lash_default_tools::{
    DefaultToolPluginOptions, DefaultToolSurfaceProfile, tool_plugin_factories,
};
use lash_provider_openai::OpenAiGenericProvider;
use lash_rlm_types::{RlmGlobalsPatchPluginBody, RlmModeEvent};
use serde::Serialize;
use stats_alloc::Stats;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_util::sync::CancellationToken;

const OM_ACTIVE_STATE_PLUGIN_TYPE: &str = "lash.context.observational_memory.state";
const DEFAULT_PROMPT: &str =
    "Inspect the current state and reply with exactly: runtime perf benchmark ok";
const HISTORY_EXCHANGES: usize = 18;
const OPENAI_COMPAT_STREAM_CHUNK_COUNT: usize = 256;
const OPENAI_COMPAT_STREAM_CHUNK_BYTES: usize = 96;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum RuntimePerfScenario {
    Standard,
    Rlm,
    RlmToolCalls,
    RlmGlobals,
    ObservationalMemory,
    OpenAiCompatStream,
    EmbedStandard,
    EmbedRlm,
}

impl RuntimePerfScenario {
    const DEFAULTS: [Self; 8] = [
        Self::Standard,
        Self::Rlm,
        Self::RlmToolCalls,
        Self::RlmGlobals,
        Self::ObservationalMemory,
        Self::OpenAiCompatStream,
        Self::EmbedStandard,
        Self::EmbedRlm,
    ];
    const KNOWN: [Self; 8] = [
        Self::Standard,
        Self::Rlm,
        Self::RlmToolCalls,
        Self::RlmGlobals,
        Self::ObservationalMemory,
        Self::OpenAiCompatStream,
        Self::EmbedStandard,
        Self::EmbedRlm,
    ];

    fn parse(value: &str) -> Option<Self> {
        match value {
            "standard" => Some(Self::Standard),
            "rlm" => Some(Self::Rlm),
            "rlm_tool_calls" => Some(Self::RlmToolCalls),
            "rlm_globals" => Some(Self::RlmGlobals),
            "observational_memory" => Some(Self::ObservationalMemory),
            "openai_compat_stream" => Some(Self::OpenAiCompatStream),
            "embed_standard" => Some(Self::EmbedStandard),
            "embed_rlm" => Some(Self::EmbedRlm),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Rlm => "rlm",
            Self::RlmToolCalls => "rlm_tool_calls",
            Self::RlmGlobals => "rlm_globals",
            Self::ObservationalMemory => "observational_memory",
            Self::OpenAiCompatStream => "openai_compat_stream",
            Self::EmbedStandard => "embed_standard",
            Self::EmbedRlm => "embed_rlm",
        }
    }

    fn execution_mode(self) -> ExecutionMode {
        match self {
            Self::Standard
            | Self::ObservationalMemory
            | Self::OpenAiCompatStream
            | Self::EmbedStandard => ExecutionMode::standard(),
            Self::Rlm | Self::RlmToolCalls | Self::RlmGlobals | Self::EmbedRlm => {
                ExecutionMode::new("rlm")
            }
        }
    }

    fn standard_context_approach(self) -> Option<StandardContextApproach> {
        match self.execution_mode() {
            mode if mode != ExecutionMode::standard() => None,
            _ => Some(match self {
                Self::ObservationalMemory => StandardContextApproach::ObservationalMemory(
                    ObservationalMemoryConfig::default(),
                ),
                _ => StandardContextApproach::RollingHistory(RollingHistoryConfig),
            }),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RuntimePerfRunResult {
    scenario: String,
    chat_turns: usize,
    build_runtime_ms: f64,
    seed_state_ms: f64,
    run_turn_ms: f64,
    await_background_work_ms: f64,
    export_state_ms: f64,
    total_ms: f64,
    session_nodes: usize,
    active_path_messages: usize,
    memory: RuntimePerfMemoryRunResult,
    allocations: RuntimePerfAllocationRunResult,
    phase_profile: BTreeMap<String, RuntimePerfPhaseRunResult>,
    turns: Vec<RuntimePerfTurnResult>,
    cumulative_usage: SessionUsageReport,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RuntimePerfTurnResult {
    turn_index: usize,
    run_turn_ms: f64,
    await_background_work_ms: f64,
    total_ms: f64,
    memory: RuntimePerfTurnMemoryRunResult,
    allocations: RuntimePerfTurnAllocationRunResult,
    phase_profile: BTreeMap<String, RuntimePerfPhaseRunResult>,
    turn_usage: TokenUsage,
    usage_delta: SessionUsageReport,
    cumulative_usage: SessionUsageReport,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RuntimePerfTurnMemoryRunResult {
    rss_before_kb: Option<u64>,
    rss_after_turn_kb: Option<u64>,
    rss_after_await_kb: Option<u64>,
    peak_hwm_before_kb: Option<u64>,
    peak_hwm_after_await_kb: Option<u64>,
    rss_growth_kb: Option<i64>,
    hwm_growth_kb: Option<i64>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RuntimePerfTurnAllocationRunResult {
    run_turn: RuntimePerfAllocationDelta,
    await_background_work: RuntimePerfAllocationDelta,
    total: RuntimePerfAllocationDelta,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RuntimePerfMetricSummary {
    min: f64,
    median: f64,
    max: f64,
    mean: f64,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RuntimePerfMemoryRunResult {
    rss_before_kb: Option<u64>,
    rss_after_build_kb: Option<u64>,
    rss_after_seed_kb: Option<u64>,
    rss_after_turn_kb: Option<u64>,
    rss_after_await_kb: Option<u64>,
    rss_after_export_kb: Option<u64>,
    peak_hwm_before_kb: Option<u64>,
    peak_hwm_after_export_kb: Option<u64>,
    rss_growth_kb: Option<i64>,
    hwm_growth_kb: Option<i64>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RuntimePerfAllocationDelta {
    allocations: usize,
    deallocations: usize,
    reallocations: usize,
    bytes_allocated: usize,
    bytes_deallocated: usize,
    bytes_reallocated: isize,
    net_live_bytes: i64,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RuntimePerfAllocationRunResult {
    build_runtime: RuntimePerfAllocationDelta,
    seed_state: RuntimePerfAllocationDelta,
    run_turn: RuntimePerfAllocationDelta,
    await_background_work: RuntimePerfAllocationDelta,
    export_state: RuntimePerfAllocationDelta,
    total: RuntimePerfAllocationDelta,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RuntimePerfPhaseRunResult {
    duration_ms: f64,
    allocations: RuntimePerfAllocationDelta,
    rss_growth_kb: Option<i64>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RuntimePerfPhaseSummary {
    duration_ms: RuntimePerfMetricSummary,
    alloc_bytes: RuntimePerfMetricSummary,
    live_bytes: RuntimePerfMetricSummary,
    rss_growth_kb: Option<RuntimePerfMetricSummary>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RuntimePerfScenarioSummary {
    scenario: String,
    runs: usize,
    chat_turns: usize,
    build_runtime_ms: RuntimePerfMetricSummary,
    seed_state_ms: RuntimePerfMetricSummary,
    run_turn_ms: RuntimePerfMetricSummary,
    await_background_work_ms: RuntimePerfMetricSummary,
    export_state_ms: RuntimePerfMetricSummary,
    total_ms: RuntimePerfMetricSummary,
    rss_after_export_kb: Option<RuntimePerfMetricSummary>,
    rss_growth_kb: Option<RuntimePerfMetricSummary>,
    hwm_growth_kb: Option<RuntimePerfMetricSummary>,
    build_runtime_alloc_bytes: RuntimePerfMetricSummary,
    build_runtime_live_bytes: RuntimePerfMetricSummary,
    seed_state_alloc_bytes: RuntimePerfMetricSummary,
    seed_state_live_bytes: RuntimePerfMetricSummary,
    run_turn_alloc_bytes: RuntimePerfMetricSummary,
    run_turn_live_bytes: RuntimePerfMetricSummary,
    await_background_work_alloc_bytes: RuntimePerfMetricSummary,
    await_background_work_live_bytes: RuntimePerfMetricSummary,
    export_state_alloc_bytes: RuntimePerfMetricSummary,
    export_state_live_bytes: RuntimePerfMetricSummary,
    total_alloc_bytes: RuntimePerfMetricSummary,
    total_live_bytes: RuntimePerfMetricSummary,
    phase_summary: BTreeMap<String, RuntimePerfPhaseSummary>,
    first_turn: RuntimePerfTurnSummary,
    steady_state_turn: Option<RuntimePerfTurnSummary>,
    last_turn: RuntimePerfTurnSummary,
    sample_session_nodes: usize,
    sample_active_path_messages: usize,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RuntimePerfTurnSummary {
    total_ms: RuntimePerfMetricSummary,
    run_turn_ms: RuntimePerfMetricSummary,
    await_background_work_ms: RuntimePerfMetricSummary,
    rss_growth_kb: Option<RuntimePerfMetricSummary>,
    total_alloc_bytes: RuntimePerfMetricSummary,
    total_live_bytes: RuntimePerfMetricSummary,
    phase_summary: BTreeMap<String, RuntimePerfPhaseSummary>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RuntimePerfReport {
    created_at: String,
    version: String,
    warmups: usize,
    runs: usize,
    chat_turns: usize,
    scenarios: Vec<String>,
    dhat_out: Option<PathBuf>,
    results: Vec<RuntimePerfRunResult>,
    summary: Vec<RuntimePerfScenarioSummary>,
}

struct BenchmarkRuntime {
    runtime: LashRuntime,
    _openai_compat_server: Option<OpenAiCompatBenchServer>,
}

impl Deref for BenchmarkRuntime {
    type Target = LashRuntime;

    fn deref(&self) -> &Self::Target {
        &self.runtime
    }
}

impl DerefMut for BenchmarkRuntime {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.runtime
    }
}

struct OpenAiCompatBenchServer {
    base_url: String,
    shutdown: CancellationToken,
    accept_task: tokio::task::JoinHandle<()>,
}

#[derive(Clone)]
struct BenchmarkStreamProfile {
    full_text: String,
    deltas: Vec<String>,
}

#[derive(Default)]
struct RuntimePerfStore {
    next_blob_id: AtomicU64,
    session_head_meta: Mutex<Option<SessionHeadMeta>>,
    session_graph: Mutex<SessionGraph>,
    usage_deltas: Mutex<Vec<TokenLedgerEntry>>,
    session_meta: Mutex<Option<store::SessionMeta>>,
}

impl RuntimePerfStore {
    fn graph_node_count(&self) -> usize {
        self.session_graph
            .lock()
            .expect("lock perf graph")
            .nodes
            .len()
    }
}

#[derive(Clone)]
struct RuntimePerfStoreFactory {
    store: Arc<RuntimePerfStore>,
}

impl SessionStoreFactory for RuntimePerfStoreFactory {
    fn create_store(
        &self,
        _request: &SessionStoreCreateRequest,
    ) -> Result<Arc<dyn RuntimePersistence>, String> {
        Ok(Arc::clone(&self.store) as Arc<dyn RuntimePersistence>)
    }
}

#[async_trait::async_trait]
impl RuntimePersistence for RuntimePerfStore {
    async fn load_session(
        &self,
        scope: SessionReadScope,
    ) -> Result<Option<PersistedSessionRead>, store::StoreError> {
        let Some(meta) = self
            .session_head_meta
            .lock()
            .expect("lock perf session head meta")
            .clone()
        else {
            return Ok(None);
        };
        let mut graph = self.session_graph.lock().expect("lock perf graph").clone();
        match scope {
            SessionReadScope::FullGraph => {}
            SessionReadScope::ActivePath { leaf_node_id } => {
                if let Some(leaf_node_id) = leaf_node_id.or_else(|| meta.leaf_node_id.clone()) {
                    graph.set_leaf_node_id(Some(leaf_node_id));
                }
                graph = graph.fork_current_path();
            }
        }
        Ok(Some(PersistedSessionRead {
            session_id: meta.session_id,
            head_revision: meta.head_revision,
            config: meta.config,
            graph,
            checkpoint_ref: meta.checkpoint_ref,
            checkpoint: None,
            token_ledger: self
                .usage_deltas
                .lock()
                .expect("lock perf usage deltas")
                .clone(),
        }))
    }

    async fn load_node(
        &self,
        node_id: &str,
    ) -> Result<Option<SessionNodeRecord>, store::StoreError> {
        Ok(self
            .session_graph
            .lock()
            .expect("lock perf graph")
            .find_node(node_id)
            .cloned())
    }

    async fn commit_runtime_state(
        &self,
        commit: RuntimeCommit,
    ) -> Result<RuntimeCommitResult, store::StoreError> {
        let mut meta_guard = self
            .session_head_meta
            .lock()
            .expect("lock perf session head meta");
        let actual = meta_guard.as_ref().map_or(0, |meta| meta.head_revision);
        if commit.expected_head_revision.is_some() && commit.expected_head_revision != Some(actual)
        {
            return Err(store::StoreError::HeadRevisionConflict {
                expected: commit.expected_head_revision,
                actual,
            });
        }
        let mut graph = self.session_graph.lock().expect("lock perf graph");
        let leaf_node_id = match &commit.graph {
            GraphCommitDelta::Unchanged { leaf_node_id } => leaf_node_id.clone(),
            GraphCommitDelta::Append {
                nodes,
                leaf_node_id,
            } => {
                graph.extend_node_records(nodes.iter().cloned());
                leaf_node_id.clone()
            }
            GraphCommitDelta::ReplaceFull(next) => {
                *graph = next.clone();
                next.leaf_node_id.clone()
            }
        };
        if !commit.usage_deltas.is_empty() {
            self.usage_deltas
                .lock()
                .expect("lock perf usage deltas")
                .extend(commit.usage_deltas.iter().cloned());
        }
        let id = self.next_blob_id.fetch_add(1, Ordering::Relaxed);
        let checkpoint_ref = BlobRef(format!("perf-checkpoint-{id}"));
        let manifest = SessionCheckpoint {
            turn_state: commit.checkpoint.turn_state,
            dynamic_state_ref: commit.checkpoint.dynamic_state_ref,
            plugin_snapshot_ref: commit.checkpoint.plugin_snapshot_ref,
            plugin_snapshot_revision: commit.checkpoint.plugin_snapshot_revision,
            execution_state_ref: commit.checkpoint.execution_state_ref,
        };
        let head_revision = actual + 1;
        *meta_guard = Some(SessionHeadMeta {
            session_id: commit.session_id,
            head_revision,
            config: commit.config,
            checkpoint_ref: Some(checkpoint_ref.clone()),
            leaf_node_id,
            graph_node_count: graph.nodes.len(),
            token_ledger: Vec::new(),
        });
        Ok(RuntimeCommitResult {
            head_revision,
            checkpoint_ref,
            manifest,
        })
    }

    async fn save_session_meta(&self, meta: store::SessionMeta) -> Result<(), store::StoreError> {
        *self.session_meta.lock().expect("lock perf session meta") = Some(meta);
        Ok(())
    }

    async fn load_session_meta(&self) -> Result<Option<store::SessionMeta>, store::StoreError> {
        Ok(self
            .session_meta
            .lock()
            .expect("lock perf session meta")
            .clone())
    }

    async fn tombstone_nodes(&self, _ids: &[String]) -> Result<(), store::StoreError> {
        Ok(())
    }

    async fn vacuum(&self) -> Result<VacuumReport, store::StoreError> {
        Ok(VacuumReport::default())
    }

    async fn gc_unreachable(&self) -> Result<GcReport, store::StoreError> {
        Ok(GcReport::default())
    }
}

impl OpenAiCompatBenchServer {
    async fn start(profile: BenchmarkStreamProfile) -> anyhow::Result<Self> {
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .context("bind local OpenAI-compatible perf server")?;
        let address = listener
            .local_addr()
            .context("resolve local OpenAI-compatible perf server address")?;
        let response_body = Arc::new(openai_compat_sse_body(&profile));
        let shutdown = CancellationToken::new();
        let accept_shutdown = shutdown.clone();
        let accept_task = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = accept_shutdown.cancelled() => break,
                    accepted = listener.accept() => {
                        let Ok((stream, _)) = accepted else { break; };
                        let body = Arc::clone(&response_body);
                        tokio::spawn(async move {
                            let _ = serve_openai_compat_connection(stream, body).await;
                        });
                    }
                }
            }
        });
        Ok(Self {
            base_url: format!("http://{address}/v1"),
            shutdown,
            accept_task,
        })
    }
}

impl Drop for OpenAiCompatBenchServer {
    fn drop(&mut self) {
        self.shutdown.cancel();
        self.accept_task.abort();
    }
}

#[derive(Clone, Copy)]
struct PhaseStart {
    started_at: Instant,
    alloc_before: Stats,
    memory_before: ProcessMemorySample,
}

#[derive(Default)]
struct RuntimePerfPhaseProbeState {
    started: HashMap<RuntimeTurnPhase, PhaseStart>,
    completed: BTreeMap<String, RuntimePerfPhaseRunResult>,
}

#[derive(Default)]
struct RuntimePerfPhaseProbe {
    state: Mutex<RuntimePerfPhaseProbeState>,
}

impl RuntimePerfPhaseProbe {
    fn take_completed(&self) -> BTreeMap<String, RuntimePerfPhaseRunResult> {
        let mut state = self.state.lock().expect("phase probe lock");
        std::mem::take(&mut state.completed)
    }
}

impl RuntimeTurnPhaseProbe for RuntimePerfPhaseProbe {
    fn begin(&self, phase: RuntimeTurnPhase) {
        let mut state = self.state.lock().expect("phase probe lock");
        state.started.insert(
            phase,
            PhaseStart {
                started_at: Instant::now(),
                alloc_before: allocator_stats(),
                memory_before: process_memory_sample(),
            },
        );
    }

    fn end(&self, phase: RuntimeTurnPhase) {
        let mut state = self.state.lock().expect("phase probe lock");
        let Some(start) = state.started.remove(&phase) else {
            return;
        };
        let alloc_after = allocator_stats();
        let memory_after = process_memory_sample();
        state.completed.insert(
            phase_name(phase).to_string(),
            RuntimePerfPhaseRunResult {
                duration_ms: elapsed_ms(start.started_at),
                allocations: alloc_delta(start.alloc_before, alloc_after),
                rss_growth_kb: diff_opt_i64(start.memory_before.rss_kb, memory_after.rss_kb),
            },
        );
    }
}

fn benchmark_provider(scenario: RuntimePerfScenario) -> TestProvider {
    TestProvider::builder()
        .kind("benchmark")
        .default_model("mock-model")
        .requires_streaming(true)
        .complete(move |req| async move {
            let profile = benchmark_stream_profile(scenario);
            let usage = LlmUsage {
                input_tokens: 1_024,
                output_tokens: 64,
                cached_input_tokens: 512,
                reasoning_tokens: 48,
            };
            if let Some(tx) = req.stream_events.as_ref() {
                for delta in &profile.deltas {
                    tx.send(LlmStreamEvent::Delta(delta.clone()));
                }
                tx.send(LlmStreamEvent::Usage(usage.clone()));
            }
            Ok(LlmResponse {
                full_text: profile.full_text.clone(),
                deltas: profile.deltas.clone(),
                parts: vec![LlmOutputPart::Text {
                    text: profile.full_text,
                    response_meta: None,
                }],
                usage,
                provider_usage: None,
                request_body: None,
                http_summary: None,
            })
        })
        .build()
}

struct BenchmarkEchoTool;

#[async_trait::async_trait]
impl ToolProvider for BenchmarkEchoTool {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![
            ToolDefinition::new(
                "benchmark_echo",
                "Return the input payload with a tiny async yield for runtime profiling.",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "value": { "type": ["string", "number", "boolean", "object", "array", "null"] },
                        "ordinal": { "type": "integer" }
                    },
                    "additionalProperties": true
                }),
                serde_json::json!({ "type": "object", "additionalProperties": true }),
            )
            .with_execution_mode(ToolExecutionMode::Parallel),
        ]
    }

    async fn execute(&self, name: &str, args: &serde_json::Value) -> ToolResult {
        if name != "benchmark_echo" {
            return ToolResult::err_fmt(format_args!("Unknown benchmark tool: {name}"));
        }
        tokio::task::yield_now().await;
        ToolResult::ok(serde_json::json!({
            "value": args.get("value").cloned().unwrap_or(serde_json::Value::Null),
            "ordinal": args.get("ordinal").cloned().unwrap_or(serde_json::Value::Null),
        }))
    }
}

pub(crate) fn default_output_path() -> PathBuf {
    let stamp = Utc::now().format("%Y%m%dT%H%M%SZ");
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("lash-cli crate should live under repo root")
        .join(".benchmarks")
        .join("runtime-perf")
        .join(format!("{stamp}.json"))
}

fn default_dhat_output_path(report_out: &Path) -> PathBuf {
    let stem = report_out
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("runtime-perf");
    report_out.with_file_name(format!("{stem}.dhat.json"))
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_cli(
    out: Option<PathBuf>,
    enable_dhat: bool,
    dhat_out: Option<PathBuf>,
    dhat_frames: Option<usize>,
    runs: usize,
    warmups: usize,
    scenario_filters: Vec<String>,
    chat_turns: usize,
    version: &str,
) -> anyhow::Result<()> {
    if dhat_out.is_some() && !enable_dhat {
        anyhow::bail!("--runtime-perf-dhat-out requires --runtime-perf-dhat");
    }
    let scenarios = resolve_scenarios(&scenario_filters)?;
    let runs = runs.max(1);
    let chat_turns = chat_turns.max(1);

    for _ in 0..warmups {
        for scenario in &scenarios {
            let _ = run_once(*scenario, chat_turns).await?;
        }
    }

    let out_path = out.unwrap_or_else(default_output_path);
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create benchmark output dir {}", parent.display()))?;
    }
    let dhat_out_path = resolve_dhat_output_path(enable_dhat, &out_path, dhat_out);
    if let Some(ref path) = dhat_out_path
        && let Some(parent) = path.parent()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create dhat output dir {}", parent.display()))?;
    }

    let profiler = start_dhat_profiler(dhat_out_path.clone(), dhat_frames)?;
    let mut results = Vec::with_capacity(runs * scenarios.len());
    for _ in 0..runs {
        for scenario in &scenarios {
            results.push(run_once(*scenario, chat_turns).await?);
        }
    }
    finish_dhat_profiler(profiler);

    let report = RuntimePerfReport {
        created_at: Utc::now().to_rfc3339(),
        version: version.to_string(),
        warmups,
        runs,
        chat_turns,
        scenarios: scenarios
            .iter()
            .map(|scenario| scenario.name().to_string())
            .collect(),
        dhat_out: dhat_out_path.clone(),
        summary: summarize(&results, &scenarios, chat_turns),
        results,
    };

    std::fs::write(&out_path, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("write benchmark report {}", out_path.display()))?;

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "out": out_path,
            "dhat_out": report.dhat_out,
            "summary": report.summary,
        }))?
    );
    Ok(())
}

fn resolve_dhat_output_path(
    enable_dhat: bool,
    report_out: &Path,
    dhat_out: Option<PathBuf>,
) -> Option<PathBuf> {
    if enable_dhat {
        Some(dhat_out.unwrap_or_else(|| default_dhat_output_path(report_out)))
    } else {
        None
    }
}

#[cfg(feature = "dhat-heap")]
fn start_dhat_profiler(
    dhat_out: Option<PathBuf>,
    dhat_frames: Option<usize>,
) -> anyhow::Result<Option<dhat::Profiler>> {
    let Some(path) = dhat_out else {
        return Ok(None);
    };
    let profiler = dhat::Profiler::builder()
        .file_name(path)
        .trim_backtraces(dhat_frames)
        .build();
    Ok(Some(profiler))
}

#[cfg(not(feature = "dhat-heap"))]
fn start_dhat_profiler(
    dhat_out: Option<PathBuf>,
    _dhat_frames: Option<usize>,
) -> anyhow::Result<Option<()>> {
    if dhat_out.is_some() {
        anyhow::bail!(
            "runtime perf dhat profiling requires a lash-cli build with --features dhat-heap"
        );
    }
    Ok(None)
}

#[cfg(feature = "dhat-heap")]
fn finish_dhat_profiler(profiler: Option<dhat::Profiler>) {
    drop(profiler);
}

#[cfg(not(feature = "dhat-heap"))]
fn finish_dhat_profiler(_profiler: Option<()>) {}

fn resolve_scenarios(filters: &[String]) -> anyhow::Result<Vec<RuntimePerfScenario>> {
    if filters.is_empty() {
        return Ok(RuntimePerfScenario::DEFAULTS.to_vec());
    }

    let mut scenarios = Vec::with_capacity(filters.len());
    for filter in filters {
        if filter == "all" {
            for scenario in RuntimePerfScenario::KNOWN {
                if !scenarios.contains(&scenario) {
                    scenarios.push(scenario);
                }
            }
            continue;
        }
        let scenario = RuntimePerfScenario::parse(filter).ok_or_else(|| {
            anyhow::anyhow!(
                "unknown runtime perf scenario `{filter}`; expected one of: {}, all",
                RuntimePerfScenario::KNOWN
                    .iter()
                    .map(|scenario| scenario.name())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
        if !scenarios.contains(&scenario) {
            scenarios.push(scenario);
        }
    }
    Ok(scenarios)
}

async fn run_once(
    scenario: RuntimePerfScenario,
    chat_turns: usize,
) -> anyhow::Result<RuntimePerfRunResult> {
    if matches!(
        scenario,
        RuntimePerfScenario::EmbedStandard | RuntimePerfScenario::EmbedRlm
    ) {
        return run_once_embed(scenario, chat_turns).await;
    }

    let total_started = Instant::now();
    let before_memory = process_memory_sample();
    let total_before_alloc = allocator_stats();

    let build_before_alloc = allocator_stats();
    let build_started = Instant::now();
    let mut runtime = build_runtime(scenario).await?;
    let build_runtime_ms = elapsed_ms(build_started);
    let build_runtime_alloc = alloc_delta(build_before_alloc, allocator_stats());
    let after_build_memory = process_memory_sample();

    let seed_before_alloc = allocator_stats();
    let seed_started = Instant::now();
    seed_runtime_state(&mut runtime, scenario).await?;
    let seed_state_ms = elapsed_ms(seed_started);
    let seed_state_alloc = alloc_delta(seed_before_alloc, allocator_stats());
    let after_seed_memory = process_memory_sample();

    let mut turns = Vec::with_capacity(chat_turns);
    for turn_index in 0..chat_turns {
        prepare_turn(&mut runtime, scenario, turn_index).await?;

        let phase_probe = Arc::new(RuntimePerfPhaseProbe::default());
        runtime.set_turn_phase_probe(phase_probe.clone());

        let before_turn_usage = runtime.usage_report();
        let turn_before_alloc = allocator_stats();
        let turn_before_memory = process_memory_sample();
        let turn_started = Instant::now();
        let turn = runtime
            .run_turn_assembled(
                TurnInput {
                    items: vec![InputItem::Text {
                        text: benchmark_prompt(scenario, turn_index),
                    }],
                    image_blobs: Default::default(),
                    user_input: None,
                    mode: Some(RunMode::Normal),
                    mode_turn_options: None,
                },
                CancellationToken::new(),
            )
            .await
            .with_context(|| {
                format!(
                    "run runtime perf scenario {} turn {}",
                    scenario.name(),
                    turn_index + 1
                )
            })?;
        let run_turn_ms = elapsed_ms(turn_started);
        let run_turn_alloc = alloc_delta(turn_before_alloc, allocator_stats());
        let after_turn_memory = process_memory_sample();

        let await_before_alloc = allocator_stats();
        let background_started = Instant::now();
        runtime.await_background_work().await.with_context(|| {
            format!(
                "await background work for {} turn {}",
                scenario.name(),
                turn_index + 1
            )
        })?;
        let await_background_work_ms = elapsed_ms(background_started);
        let await_background_work_alloc = alloc_delta(await_before_alloc, allocator_stats());
        let after_await_memory = process_memory_sample();
        let turn_total_alloc =
            sum_allocation_deltas([&run_turn_alloc, &await_background_work_alloc]);

        let cumulative_usage = runtime.usage_report();
        let usage_delta_entries = lash::diff_usage_reports(&before_turn_usage, &cumulative_usage)
            .map_err(anyhow::Error::msg)?;
        turns.push(RuntimePerfTurnResult {
            turn_index,
            run_turn_ms,
            await_background_work_ms,
            total_ms: round3(run_turn_ms + await_background_work_ms),
            memory: RuntimePerfTurnMemoryRunResult {
                rss_before_kb: turn_before_memory.rss_kb,
                rss_after_turn_kb: after_turn_memory.rss_kb,
                rss_after_await_kb: after_await_memory.rss_kb,
                peak_hwm_before_kb: turn_before_memory.hwm_kb,
                peak_hwm_after_await_kb: after_await_memory.hwm_kb,
                rss_growth_kb: diff_opt_i64(turn_before_memory.rss_kb, after_await_memory.rss_kb),
                hwm_growth_kb: diff_opt_i64(turn_before_memory.hwm_kb, after_await_memory.hwm_kb),
            },
            allocations: RuntimePerfTurnAllocationRunResult {
                run_turn: run_turn_alloc,
                await_background_work: await_background_work_alloc,
                total: turn_total_alloc,
            },
            phase_profile: phase_probe.take_completed(),
            turn_usage: turn.token_usage,
            usage_delta: SessionUsageReport::from_entries(&usage_delta_entries),
            cumulative_usage,
        });
    }

    let export_before_alloc = allocator_stats();
    let export_started = Instant::now();
    let state = runtime.export_state();
    let cumulative_usage = runtime.usage_report();
    let export_state_ms = elapsed_ms(export_started);
    let export_state_alloc = alloc_delta(export_before_alloc, allocator_stats());
    let after_export_memory = process_memory_sample();
    let total_alloc = alloc_delta(total_before_alloc, allocator_stats());
    let last_turn_memory = turns.last().map(|turn| &turn.memory);

    Ok(RuntimePerfRunResult {
        scenario: scenario.name().to_string(),
        chat_turns,
        build_runtime_ms,
        seed_state_ms,
        run_turn_ms: round3(turns.iter().map(|turn| turn.run_turn_ms).sum()),
        await_background_work_ms: round3(
            turns.iter().map(|turn| turn.await_background_work_ms).sum(),
        ),
        export_state_ms,
        total_ms: elapsed_ms(total_started),
        session_nodes: state.session_graph.nodes.len(),
        active_path_messages: state.read_view().messages().len(),
        memory: RuntimePerfMemoryRunResult {
            rss_before_kb: before_memory.rss_kb,
            rss_after_build_kb: after_build_memory.rss_kb,
            rss_after_seed_kb: after_seed_memory.rss_kb,
            rss_after_turn_kb: last_turn_memory.and_then(|memory| memory.rss_after_turn_kb),
            rss_after_await_kb: last_turn_memory.and_then(|memory| memory.rss_after_await_kb),
            rss_after_export_kb: after_export_memory.rss_kb,
            peak_hwm_before_kb: before_memory.hwm_kb,
            peak_hwm_after_export_kb: after_export_memory.hwm_kb,
            rss_growth_kb: diff_opt_i64(before_memory.rss_kb, after_export_memory.rss_kb),
            hwm_growth_kb: diff_opt_i64(before_memory.hwm_kb, after_export_memory.hwm_kb),
        },
        allocations: RuntimePerfAllocationRunResult {
            build_runtime: build_runtime_alloc,
            seed_state: seed_state_alloc,
            run_turn: sum_allocation_deltas(turns.iter().map(|turn| &turn.allocations.run_turn)),
            await_background_work: sum_allocation_deltas(
                turns
                    .iter()
                    .map(|turn| &turn.allocations.await_background_work),
            ),
            export_state: export_state_alloc,
            total: total_alloc,
        },
        phase_profile: sum_phase_profiles(turns.iter().map(|turn| &turn.phase_profile)),
        turns,
        cumulative_usage,
    })
}

async fn run_once_embed(
    scenario: RuntimePerfScenario,
    chat_turns: usize,
) -> anyhow::Result<RuntimePerfRunResult> {
    let total_started = Instant::now();
    let before_memory = process_memory_sample();
    let total_before_alloc = allocator_stats();

    let build_before_alloc = allocator_stats();
    let build_started = Instant::now();
    let store = Arc::new(RuntimePerfStore::default());
    let core = build_embed_core(scenario, Arc::clone(&store))?;
    let session = core
        .session(format!("runtime-perf-{}", scenario.name()))
        .mode(lash_embed::ModeId::new(
            scenario.execution_mode().plugin_id(),
        ))
        .open()
        .await
        .with_context(|| format!("open embed session for {}", scenario.name()))?;
    let build_runtime_ms = elapsed_ms(build_started);
    let build_runtime_alloc = alloc_delta(build_before_alloc, allocator_stats());
    let after_build_memory = process_memory_sample();

    let seed_before_alloc = allocator_stats();
    let seed_started = Instant::now();
    let seed_state_ms = elapsed_ms(seed_started);
    let seed_state_alloc = alloc_delta(seed_before_alloc, allocator_stats());
    let after_seed_memory = process_memory_sample();

    let mut turns = Vec::with_capacity(chat_turns);
    for turn_index in 0..chat_turns {
        let before_turn_usage = SessionUsageReport::default();
        let turn_before_alloc = allocator_stats();
        let turn_before_memory = process_memory_sample();
        let turn_started = Instant::now();
        let turn = session
            .run(lash_embed::Input::text(benchmark_prompt(
                scenario, turn_index,
            )))
            .await
            .with_context(|| {
                format!(
                    "run embed runtime perf scenario {} turn {}",
                    scenario.name(),
                    turn_index + 1
                )
            })?;
        let run_turn_ms = elapsed_ms(turn_started);
        let run_turn_alloc = alloc_delta(turn_before_alloc, allocator_stats());
        let after_turn_memory = process_memory_sample();

        let await_before_alloc = allocator_stats();
        let background_started = Instant::now();
        let await_background_work_ms = elapsed_ms(background_started);
        let await_background_work_alloc = alloc_delta(await_before_alloc, allocator_stats());
        let after_await_memory = process_memory_sample();
        let turn_total_alloc =
            sum_allocation_deltas([&run_turn_alloc, &await_background_work_alloc]);

        turns.push(RuntimePerfTurnResult {
            turn_index,
            run_turn_ms,
            await_background_work_ms,
            total_ms: round3(run_turn_ms + await_background_work_ms),
            memory: RuntimePerfTurnMemoryRunResult {
                rss_before_kb: turn_before_memory.rss_kb,
                rss_after_turn_kb: after_turn_memory.rss_kb,
                rss_after_await_kb: after_await_memory.rss_kb,
                peak_hwm_before_kb: turn_before_memory.hwm_kb,
                peak_hwm_after_await_kb: after_await_memory.hwm_kb,
                rss_growth_kb: diff_opt_i64(turn_before_memory.rss_kb, after_await_memory.rss_kb),
                hwm_growth_kb: diff_opt_i64(turn_before_memory.hwm_kb, after_await_memory.hwm_kb),
            },
            allocations: RuntimePerfTurnAllocationRunResult {
                run_turn: run_turn_alloc,
                await_background_work: await_background_work_alloc,
                total: turn_total_alloc,
            },
            phase_profile: BTreeMap::new(),
            turn_usage: turn.usage,
            usage_delta: before_turn_usage,
            cumulative_usage: SessionUsageReport::default(),
        });
    }

    let export_before_alloc = allocator_stats();
    let export_started = Instant::now();
    let read_view = session.read_view().await;
    let export_state_ms = elapsed_ms(export_started);
    let export_state_alloc = alloc_delta(export_before_alloc, allocator_stats());
    let after_export_memory = process_memory_sample();
    let total_alloc = alloc_delta(total_before_alloc, allocator_stats());
    let last_turn_memory = turns.last().map(|turn| &turn.memory);

    Ok(RuntimePerfRunResult {
        scenario: scenario.name().to_string(),
        chat_turns,
        build_runtime_ms,
        seed_state_ms,
        run_turn_ms: round3(turns.iter().map(|turn| turn.run_turn_ms).sum()),
        await_background_work_ms: round3(
            turns.iter().map(|turn| turn.await_background_work_ms).sum(),
        ),
        export_state_ms,
        total_ms: elapsed_ms(total_started),
        session_nodes: store.graph_node_count(),
        active_path_messages: read_view.messages().len(),
        memory: RuntimePerfMemoryRunResult {
            rss_before_kb: before_memory.rss_kb,
            rss_after_build_kb: after_build_memory.rss_kb,
            rss_after_seed_kb: after_seed_memory.rss_kb,
            rss_after_turn_kb: last_turn_memory.and_then(|memory| memory.rss_after_turn_kb),
            rss_after_await_kb: last_turn_memory.and_then(|memory| memory.rss_after_await_kb),
            rss_after_export_kb: after_export_memory.rss_kb,
            peak_hwm_before_kb: before_memory.hwm_kb,
            peak_hwm_after_export_kb: after_export_memory.hwm_kb,
            rss_growth_kb: diff_opt_i64(before_memory.rss_kb, after_export_memory.rss_kb),
            hwm_growth_kb: diff_opt_i64(before_memory.hwm_kb, after_export_memory.hwm_kb),
        },
        allocations: RuntimePerfAllocationRunResult {
            build_runtime: build_runtime_alloc,
            seed_state: seed_state_alloc,
            run_turn: sum_allocation_deltas(turns.iter().map(|turn| &turn.allocations.run_turn)),
            await_background_work: sum_allocation_deltas(
                turns
                    .iter()
                    .map(|turn| &turn.allocations.await_background_work),
            ),
            export_state: export_state_alloc,
            total: total_alloc,
        },
        phase_profile: BTreeMap::new(),
        turns,
        cumulative_usage: SessionUsageReport::default(),
    })
}

fn build_embed_core(
    scenario: RuntimePerfScenario,
    store: Arc<RuntimePerfStore>,
) -> anyhow::Result<lash_embed::LashCore> {
    let mut builder = match scenario {
        RuntimePerfScenario::EmbedStandard => lash_embed::LashCore::standard(),
        RuntimePerfScenario::EmbedRlm => lash_embed::LashCore::rlm()
            .tools(Arc::new(BenchmarkEchoTool))
            .default_mode(lash_embed::ModeId::rlm()),
        _ => anyhow::bail!("{} is not an embed scenario", scenario.name()),
    };
    builder = builder
        .provider(benchmark_provider(scenario).into_handle())
        .model("mock-model")
        .max_context_tokens(200_000)
        .store_factory(Arc::new(RuntimePerfStoreFactory { store }));
    builder.build().map_err(anyhow::Error::from)
}

async fn build_runtime(scenario: RuntimePerfScenario) -> anyhow::Result<BenchmarkRuntime> {
    let execution_mode = scenario.execution_mode();
    let standard_context_approach = scenario.standard_context_approach();
    let openai_compat_server = if matches!(scenario, RuntimePerfScenario::OpenAiCompatStream) {
        Some(OpenAiCompatBenchServer::start(benchmark_stream_profile(scenario)).await?)
    } else {
        None
    };
    let base_url = openai_compat_server
        .as_ref()
        .map(|server| server.base_url.clone())
        .unwrap_or_else(|| "https://example.invalid/v1".to_string());
    let provider: ProviderHandle = match scenario {
        RuntimePerfScenario::OpenAiCompatStream => ProviderHandle::new(
            OpenAiGenericProvider::new("test-key", base_url.clone()).into_components(),
        ),
        _ => benchmark_provider(scenario).into_handle(),
    };
    let policy = SessionPolicy {
        model: "mock-model".to_string(),
        provider,
        max_context_tokens: Some(200_000),
        execution_mode: execution_mode.clone(),
        standard_context_approach: standard_context_approach.clone(),
        ..SessionPolicy::default()
    };

    let profile =
        DefaultToolSurfaceProfile::for_runtime(standard_context_approach.as_ref(), false, false);
    let store = Arc::new(RuntimePerfStore::default()) as Arc<dyn RuntimePersistence>;
    let mut factories = tool_plugin_factories(DefaultToolPluginOptions {
        execution_mode,
        standard_context_approach: standard_context_approach.clone(),
        bundles: profile.bundles,
        tavily_api_key: None,
        instruction_source: None,
    });
    factories.push(Arc::new(lash::BuiltinTaskControlsPluginFactory::new()));
    factories.push(Arc::new(lash::BuiltinMonitorToolPluginFactory::new()));
    factories.push(Arc::new(lash::plugin::StaticPluginFactory::new(
        "runtime_perf_tools",
        PluginSpec::new().with_tool_provider(Arc::new(BenchmarkEchoTool)),
    )));
    factories.push(Arc::new(
        lash_mode_standard::BuiltinStandardModePluginFactory,
    ));
    factories.push(Arc::new(
        lash_mode_rlm::BuiltinRlmModePluginFactory::default(),
    ));
    let plugin_host = PluginHost::new(factories).with_dynamic_tools();
    let builder = LashRuntime::builder()
        .with_policy(policy.clone())
        .with_store(Arc::clone(&store))
        .with_session_task_executor(Arc::new(TokioSessionTaskExecutor::default()))
        .with_plugin_host(plugin_host);
    let runtime = builder.build().await?;
    Ok(BenchmarkRuntime {
        runtime,
        _openai_compat_server: openai_compat_server,
    })
}

async fn seed_runtime_state(
    runtime: &mut LashRuntime,
    scenario: RuntimePerfScenario,
) -> anyhow::Result<()> {
    let mut nodes = Vec::with_capacity(HISTORY_EXCHANGES * 2);
    for index in 0..HISTORY_EXCHANGES {
        nodes.push(SessionAppendNode::message(PluginMessage::text(
            MessageRole::User,
            format!(
                "Historical user turn {index}: trace the performance-sensitive path through runtime/session graph/tool prep."
            ),
        )));
        nodes.push(SessionAppendNode::message(PluginMessage::text(
            MessageRole::Assistant,
            format!(
                "Historical assistant turn {index}: inspected runtime.rs, turn_runner.rs, and token ledger export surfaces."
            ),
        )));
    }

    runtime
        .append_session_nodes(AppendSessionNodesRequest {
            nodes,
            requires_ancestor_node_id: None,
        })
        .await
        .map_err(|err| anyhow::anyhow!("seed historical messages: {err}"))?;

    if matches!(scenario, RuntimePerfScenario::RlmGlobals) {
        let mut set = serde_json::Map::new();
        set.insert(
            "benchmark".to_string(),
            serde_json::json!({
                "name": "runtime_perf",
                "scenario": scenario.name(),
            }),
        );
        set.insert(
            "input".to_string(),
            serde_json::json!({
                "goal": "measure runtime overhead",
                "path": "lash/src/runtime",
            }),
        );
        runtime
            .append_session_nodes(AppendSessionNodesRequest {
                nodes: vec![SessionAppendNode::event(SessionEventRecord::Mode(
                    lash::ModeEvent::rlm(RlmModeEvent::RlmGlobalsPatch(
                        RlmGlobalsPatchPluginBody {
                            set,
                            unset: Vec::new(),
                        },
                    )),
                ))],
                requires_ancestor_node_id: None,
            })
            .await
            .map_err(|err| anyhow::anyhow!("seed RLM globals patch: {err}"))?;
    }

    if matches!(scenario, RuntimePerfScenario::ObservationalMemory) {
        let observed_through_message_id = runtime
            .read_view()
            .messages()
            .last()
            .map(|message| message.id.clone())
            .ok_or_else(|| anyhow::anyhow!("OM scenario expected seeded messages"))?;
        runtime
            .append_session_nodes(AppendSessionNodesRequest {
                nodes: vec![SessionAppendNode::plugin(
                    OM_ACTIVE_STATE_PLUGIN_TYPE,
                    serde_json::json!({
                        "observed_through_message_id": observed_through_message_id,
                        "observations": "Date: Apr 13, 2026\n* 🔴 User is evaluating Lash runtime performance without model inference.\n* 🟡 Historical inspection focused on runtime, token ledger export, and tool initialization overhead.\n* ✅ Shared process-wide search cache was added for indexed grep state.",
                        "current_task": "Primary: Benchmark runtime overhead.\nSecondary: Compare standard, rlm, and observational memory paths.",
                        "suggested_response": "Report the runtime benchmark numbers and the dominant overheads directly."
                    }),
                )],
                requires_ancestor_node_id: None,
            })
            .await
            .map_err(|err| anyhow::anyhow!("seed OM reflection node: {err}"))?;
    }

    Ok(())
}

async fn prepare_turn(
    runtime: &mut LashRuntime,
    scenario: RuntimePerfScenario,
    turn_index: usize,
) -> anyhow::Result<()> {
    if !matches!(scenario, RuntimePerfScenario::RlmGlobals) {
        return Ok(());
    }

    let mut set = serde_json::Map::new();
    set.insert(
        "input".to_string(),
        serde_json::json!({
            "turn": turn_index + 1,
            "goal": "measure runtime overhead across a longer same-session chat",
            "focus": [
                "checkpoint reuse",
                "persist_turn allocation churn",
                "steady-state graph growth"
            ],
        }),
    );
    set.insert(
        "chat".to_string(),
        serde_json::json!({
            "turn_count": turn_index + 1,
            "scenario": scenario.name(),
            "mode": "runtime_perf",
        }),
    );

    runtime
        .append_session_nodes(AppendSessionNodesRequest {
            nodes: vec![SessionAppendNode::event(SessionEventRecord::Mode(
                lash::ModeEvent::rlm(RlmModeEvent::RlmGlobalsPatch(RlmGlobalsPatchPluginBody {
                    set,
                    unset: Vec::new(),
                })),
            ))],
            requires_ancestor_node_id: None,
        })
        .await
        .map(|_| ())
        .map_err(|err| anyhow::anyhow!("seed RLM globals patch for turn {}: {err}", turn_index + 1))
}

fn benchmark_prompt(scenario: RuntimePerfScenario, turn_index: usize) -> String {
    match scenario {
        RuntimePerfScenario::Standard | RuntimePerfScenario::EmbedStandard => format!(
            "Turn {} of a longer runtime benchmark conversation. Inspect the state and reply with exactly: {}",
            turn_index + 1,
            DEFAULT_PROMPT
                .rsplit_once(": ")
                .map(|(_, text)| text)
                .unwrap_or("runtime perf benchmark ok")
        ),
        RuntimePerfScenario::Rlm | RuntimePerfScenario::EmbedRlm => format!(
            "Turn {} in RLM mode. Continue the benchmark chat and reply with exactly: {}",
            turn_index + 1,
            DEFAULT_PROMPT
                .rsplit_once(": ")
                .map(|(_, text)| text)
                .unwrap_or("runtime perf benchmark ok")
        ),
        RuntimePerfScenario::RlmToolCalls => format!(
            "Turn {} in RLM mode. Exercise the benchmark_echo tool path and reply with exactly: {}",
            turn_index + 1,
            DEFAULT_PROMPT
                .rsplit_once(": ")
                .map(|(_, text)| text)
                .unwrap_or("runtime perf benchmark ok")
        ),
        RuntimePerfScenario::RlmGlobals => format!(
            "Turn {} in RLM mode with bound variables updated for this turn. Inspect the current state and reply with exactly: {}",
            turn_index + 1,
            DEFAULT_PROMPT
                .rsplit_once(": ")
                .map(|(_, text)| text)
                .unwrap_or("runtime perf benchmark ok")
        ),
        RuntimePerfScenario::ObservationalMemory => format!(
            "Turn {} in observational memory mode. Continue the same longer benchmark conversation and reply with exactly: {}",
            turn_index + 1,
            DEFAULT_PROMPT
                .rsplit_once(": ")
                .map(|(_, text)| text)
                .unwrap_or("runtime perf benchmark ok")
        ),
        RuntimePerfScenario::OpenAiCompatStream => format!(
            "Turn {} in OpenAI-compatible streaming benchmark mode. Continue the benchmark chat and reply with exactly: runtime perf benchmark ok",
            turn_index + 1
        ),
    }
}

fn benchmark_stream_profile(scenario: RuntimePerfScenario) -> BenchmarkStreamProfile {
    match scenario {
        RuntimePerfScenario::OpenAiCompatStream => {
            let alphabet = "abcdefghijklmnopqrstuvwxyz0123456789";
            let mut deltas = Vec::with_capacity(OPENAI_COMPAT_STREAM_CHUNK_COUNT + 1);
            for index in 0..OPENAI_COMPAT_STREAM_CHUNK_COUNT {
                let prefix = format!("chunk-{index:03}: ");
                let fill_len = OPENAI_COMPAT_STREAM_CHUNK_BYTES.saturating_sub(prefix.len() + 1);
                let body: String = alphabet
                    .chars()
                    .cycle()
                    .skip(index % alphabet.len())
                    .take(fill_len)
                    .collect();
                deltas.push(format!("{prefix}{body}\n"));
            }
            deltas.push("runtime perf benchmark ok".to_string());
            BenchmarkStreamProfile {
                full_text: deltas.concat(),
                deltas,
            }
        }
        RuntimePerfScenario::Rlm
        | RuntimePerfScenario::RlmGlobals
        | RuntimePerfScenario::EmbedRlm => {
            let text = "```lashlang\nsubmit \"runtime perf benchmark ok\"\n```".to_string();
            BenchmarkStreamProfile {
                full_text: text.clone(),
                deltas: vec![text],
            }
        }
        RuntimePerfScenario::RlmToolCalls => {
            let text = r#"```lashlang
fanout = parallel {
  a: call benchmark_echo { value: "runtime perf benchmark ok", ordinal: 1 }
  b: call benchmark_echo { value: "runtime perf benchmark ok", ordinal: 2 }
  c: call benchmark_echo { value: "runtime perf benchmark ok", ordinal: 3 }
  d: call benchmark_echo { value: "runtime perf benchmark ok", ordinal: 4 }
}
first = fanout.a?
submit first.value
```"#
                .to_string();
            BenchmarkStreamProfile {
                full_text: text.clone(),
                deltas: vec![text],
            }
        }
        _ => {
            let text = "runtime perf benchmark ok".to_string();
            BenchmarkStreamProfile {
                full_text: text.clone(),
                deltas: vec![text],
            }
        }
    }
}

fn openai_compat_sse_body(profile: &BenchmarkStreamProfile) -> Vec<u8> {
    let mut body = String::new();
    let message_id = "msg_runtime_perf";
    body.push_str("data: ");
    body.push_str(
        &serde_json::json!({
            "type": "response.output_item.added",
            "item": {
                "type": "message",
                "id": message_id,
                "status": "in_progress",
                "role": "assistant",
                "content": []
            }
        })
        .to_string(),
    );
    body.push_str("\n\n");
    for delta in &profile.deltas {
        body.push_str("data: ");
        body.push_str(
            &serde_json::json!({
                "type": "response.output_text.delta",
                "item_id": message_id,
                "delta": delta,
            })
            .to_string(),
        );
        body.push_str("\n\n");
    }
    body.push_str("data: ");
    body.push_str(
        &serde_json::json!({
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "id": message_id,
                "status": "completed",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": profile.full_text,
                    "annotations": []
                }]
            }
        })
        .to_string(),
    );
    body.push_str("\n\n");
    body.push_str("data: ");
    body.push_str(
        &serde_json::json!({
            "type": "response.completed",
            "response": {
                "id": "resp_runtime_perf",
                "status": "completed",
                "output": [{
                    "type": "message",
                    "id": message_id,
                    "status": "completed",
                    "role": "assistant",
                    "content": [{
                        "type": "output_text",
                        "text": profile.full_text,
                        "annotations": []
                    }]
                }],
                "usage": {
                    "input_tokens": 1024,
                    "output_tokens": 64,
                    "input_tokens_details": {
                        "cached_tokens": 512,
                    },
                    "output_tokens_details": {
                        "reasoning_tokens": 48,
                    }
                },
            }
        })
        .to_string(),
    );
    body.push_str("\n\n");
    body.push_str("data: [DONE]\n\n");
    body.into_bytes()
}

async fn serve_openai_compat_connection(
    mut stream: tokio::net::TcpStream,
    response_body: Arc<Vec<u8>>,
) -> io::Result<()> {
    drain_http_request(&mut stream).await?;
    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nContent-Length: {}\r\n\r\n",
        response_body.len()
    );
    stream.write_all(header.as_bytes()).await?;
    stream.write_all(response_body.as_slice()).await?;
    stream.shutdown().await
}

async fn drain_http_request(stream: &mut tokio::net::TcpStream) -> io::Result<()> {
    let mut headers = Vec::new();
    let mut chunk = [0u8; 4096];
    let mut content_length = None;
    let mut header_len = None;
    let mut total_read = 0usize;
    loop {
        let read = stream.read(&mut chunk).await?;
        if read == 0 {
            return Ok(());
        }
        total_read += read;
        if header_len.is_none() {
            headers.extend_from_slice(&chunk[..read]);
        }
        if header_len.is_none()
            && let Some(index) = find_bytes(&headers, b"\r\n\r\n")
        {
            let end = index + 4;
            let header_text = String::from_utf8_lossy(&headers);
            header_len = Some(end);
            content_length = Some(parse_content_length(&header_text).unwrap_or(0));
            headers.clear();
        }
        if let (Some(header_len), Some(content_length)) = (header_len, content_length)
            && total_read >= header_len + content_length
        {
            return Ok(());
        }
    }
}

fn parse_content_length(headers: &str) -> Option<usize> {
    headers.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        if name.trim().eq_ignore_ascii_case("content-length") {
            value.trim().parse::<usize>().ok()
        } else {
            None
        }
    })
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn summarize(
    results: &[RuntimePerfRunResult],
    scenarios: &[RuntimePerfScenario],
    chat_turns: usize,
) -> Vec<RuntimePerfScenarioSummary> {
    scenarios
        .iter()
        .filter_map(|scenario| {
            let matching = results
                .iter()
                .filter(|result| result.scenario == scenario.name())
                .collect::<Vec<_>>();
            if matching.is_empty() {
                return None;
            }
            Some(RuntimePerfScenarioSummary {
                scenario: scenario.name().to_string(),
                runs: matching.len(),
                chat_turns,
                build_runtime_ms: summarize_metric(
                    matching
                        .iter()
                        .map(|result| result.build_runtime_ms)
                        .collect::<Vec<_>>(),
                ),
                seed_state_ms: summarize_metric(
                    matching
                        .iter()
                        .map(|result| result.seed_state_ms)
                        .collect::<Vec<_>>(),
                ),
                run_turn_ms: summarize_metric(
                    matching
                        .iter()
                        .map(|result| result.run_turn_ms)
                        .collect::<Vec<_>>(),
                ),
                await_background_work_ms: summarize_metric(
                    matching
                        .iter()
                        .map(|result| result.await_background_work_ms)
                        .collect::<Vec<_>>(),
                ),
                export_state_ms: summarize_metric(
                    matching
                        .iter()
                        .map(|result| result.export_state_ms)
                        .collect::<Vec<_>>(),
                ),
                total_ms: summarize_metric(
                    matching
                        .iter()
                        .map(|result| result.total_ms)
                        .collect::<Vec<_>>(),
                ),
                rss_after_export_kb: summarize_optional_metric(
                    matching
                        .iter()
                        .filter_map(|result| {
                            result.memory.rss_after_export_kb.map(|value| value as f64)
                        })
                        .collect::<Vec<_>>(),
                ),
                rss_growth_kb: summarize_optional_metric(
                    matching
                        .iter()
                        .filter_map(|result| result.memory.rss_growth_kb.map(|value| value as f64))
                        .collect::<Vec<_>>(),
                ),
                hwm_growth_kb: summarize_optional_metric(
                    matching
                        .iter()
                        .filter_map(|result| result.memory.hwm_growth_kb.map(|value| value as f64))
                        .collect::<Vec<_>>(),
                ),
                build_runtime_alloc_bytes: summarize_metric(
                    matching
                        .iter()
                        .map(|result| result.allocations.build_runtime.bytes_allocated as f64)
                        .collect::<Vec<_>>(),
                ),
                build_runtime_live_bytes: summarize_metric(
                    matching
                        .iter()
                        .map(|result| result.allocations.build_runtime.net_live_bytes as f64)
                        .collect::<Vec<_>>(),
                ),
                seed_state_alloc_bytes: summarize_metric(
                    matching
                        .iter()
                        .map(|result| result.allocations.seed_state.bytes_allocated as f64)
                        .collect::<Vec<_>>(),
                ),
                seed_state_live_bytes: summarize_metric(
                    matching
                        .iter()
                        .map(|result| result.allocations.seed_state.net_live_bytes as f64)
                        .collect::<Vec<_>>(),
                ),
                run_turn_alloc_bytes: summarize_metric(
                    matching
                        .iter()
                        .map(|result| result.allocations.run_turn.bytes_allocated as f64)
                        .collect::<Vec<_>>(),
                ),
                run_turn_live_bytes: summarize_metric(
                    matching
                        .iter()
                        .map(|result| result.allocations.run_turn.net_live_bytes as f64)
                        .collect::<Vec<_>>(),
                ),
                await_background_work_alloc_bytes: summarize_metric(
                    matching
                        .iter()
                        .map(|result| {
                            result.allocations.await_background_work.bytes_allocated as f64
                        })
                        .collect::<Vec<_>>(),
                ),
                await_background_work_live_bytes: summarize_metric(
                    matching
                        .iter()
                        .map(|result| {
                            result.allocations.await_background_work.net_live_bytes as f64
                        })
                        .collect::<Vec<_>>(),
                ),
                export_state_alloc_bytes: summarize_metric(
                    matching
                        .iter()
                        .map(|result| result.allocations.export_state.bytes_allocated as f64)
                        .collect::<Vec<_>>(),
                ),
                export_state_live_bytes: summarize_metric(
                    matching
                        .iter()
                        .map(|result| result.allocations.export_state.net_live_bytes as f64)
                        .collect::<Vec<_>>(),
                ),
                total_alloc_bytes: summarize_metric(
                    matching
                        .iter()
                        .map(|result| result.allocations.total.bytes_allocated as f64)
                        .collect::<Vec<_>>(),
                ),
                total_live_bytes: summarize_metric(
                    matching
                        .iter()
                        .map(|result| result.allocations.total.net_live_bytes as f64)
                        .collect::<Vec<_>>(),
                ),
                phase_summary: summarize_phase_profiles(
                    &matching
                        .iter()
                        .map(|result| result.phase_profile.clone())
                        .collect::<Vec<_>>(),
                ),
                first_turn: summarize_turn_group(
                    &matching
                        .iter()
                        .filter_map(|result| result.turns.first().cloned())
                        .collect::<Vec<_>>(),
                ),
                steady_state_turn: summarize_optional_turn_group(
                    &matching
                        .iter()
                        .filter_map(|result| mean_turn_result(&result.turns[1..]))
                        .collect::<Vec<_>>(),
                ),
                last_turn: summarize_turn_group(
                    &matching
                        .iter()
                        .filter_map(|result| result.turns.last().cloned())
                        .collect::<Vec<_>>(),
                ),
                sample_session_nodes: matching[0].session_nodes,
                sample_active_path_messages: matching[0].active_path_messages,
            })
        })
        .collect()
}

fn summarize_phase_profiles(
    profiles: &[BTreeMap<String, RuntimePerfPhaseRunResult>],
) -> BTreeMap<String, RuntimePerfPhaseSummary> {
    let mut by_phase: BTreeMap<String, Vec<&RuntimePerfPhaseRunResult>> = BTreeMap::new();
    for profile in profiles {
        for (phase, metrics) in profile {
            by_phase.entry(phase.clone()).or_default().push(metrics);
        }
    }

    by_phase
        .into_iter()
        .map(|(phase, metrics)| {
            let summary = RuntimePerfPhaseSummary {
                duration_ms: summarize_metric(
                    metrics.iter().map(|metric| metric.duration_ms).collect(),
                ),
                alloc_bytes: summarize_metric(
                    metrics
                        .iter()
                        .map(|metric| metric.allocations.bytes_allocated as f64)
                        .collect(),
                ),
                live_bytes: summarize_metric(
                    metrics
                        .iter()
                        .map(|metric| metric.allocations.net_live_bytes as f64)
                        .collect(),
                ),
                rss_growth_kb: summarize_optional_metric(
                    metrics
                        .iter()
                        .filter_map(|metric| metric.rss_growth_kb.map(|value| value as f64))
                        .collect(),
                ),
            };
            (phase, summary)
        })
        .collect()
}

fn summarize_turn_group(turns: &[RuntimePerfTurnResult]) -> RuntimePerfTurnSummary {
    RuntimePerfTurnSummary {
        total_ms: summarize_metric(turns.iter().map(|turn| turn.total_ms).collect()),
        run_turn_ms: summarize_metric(turns.iter().map(|turn| turn.run_turn_ms).collect()),
        await_background_work_ms: summarize_metric(
            turns
                .iter()
                .map(|turn| turn.await_background_work_ms)
                .collect(),
        ),
        rss_growth_kb: summarize_optional_metric(
            turns
                .iter()
                .filter_map(|turn| turn.memory.rss_growth_kb.map(|value| value as f64))
                .collect(),
        ),
        total_alloc_bytes: summarize_metric(
            turns
                .iter()
                .map(|turn| turn.allocations.total.bytes_allocated as f64)
                .collect(),
        ),
        total_live_bytes: summarize_metric(
            turns
                .iter()
                .map(|turn| turn.allocations.total.net_live_bytes as f64)
                .collect(),
        ),
        phase_summary: summarize_phase_profiles(
            &turns
                .iter()
                .map(|turn| turn.phase_profile.clone())
                .collect::<Vec<_>>(),
        ),
    }
}

fn summarize_optional_turn_group(
    turns: &[RuntimePerfTurnResult],
) -> Option<RuntimePerfTurnSummary> {
    if turns.is_empty() {
        None
    } else {
        Some(summarize_turn_group(turns))
    }
}

fn mean_turn_result(turns: &[RuntimePerfTurnResult]) -> Option<RuntimePerfTurnResult> {
    if turns.is_empty() {
        return None;
    }

    let count = turns.len() as f64;
    Some(RuntimePerfTurnResult {
        turn_index: turns[0].turn_index,
        run_turn_ms: round3(turns.iter().map(|turn| turn.run_turn_ms).sum::<f64>() / count),
        await_background_work_ms: round3(
            turns
                .iter()
                .map(|turn| turn.await_background_work_ms)
                .sum::<f64>()
                / count,
        ),
        total_ms: round3(turns.iter().map(|turn| turn.total_ms).sum::<f64>() / count),
        memory: RuntimePerfTurnMemoryRunResult {
            rss_before_kb: None,
            rss_after_turn_kb: None,
            rss_after_await_kb: None,
            peak_hwm_before_kb: None,
            peak_hwm_after_await_kb: None,
            rss_growth_kb: mean_option_i64(turns.iter().map(|turn| turn.memory.rss_growth_kb)),
            hwm_growth_kb: mean_option_i64(turns.iter().map(|turn| turn.memory.hwm_growth_kb)),
        },
        allocations: RuntimePerfTurnAllocationRunResult {
            run_turn: mean_allocation_delta(turns.iter().map(|turn| &turn.allocations.run_turn)),
            await_background_work: mean_allocation_delta(
                turns
                    .iter()
                    .map(|turn| &turn.allocations.await_background_work),
            ),
            total: mean_allocation_delta(turns.iter().map(|turn| &turn.allocations.total)),
        },
        phase_profile: mean_phase_profiles(turns.iter().map(|turn| &turn.phase_profile)),
        turn_usage: mean_token_usage(turns.iter().map(|turn| &turn.turn_usage)),
        usage_delta: SessionUsageReport::default(),
        cumulative_usage: SessionUsageReport::default(),
    })
}

fn sum_phase_profiles<'a>(
    profiles: impl IntoIterator<Item = &'a BTreeMap<String, RuntimePerfPhaseRunResult>>,
) -> BTreeMap<String, RuntimePerfPhaseRunResult> {
    let mut totals: BTreeMap<String, RuntimePerfPhaseRunResult> = BTreeMap::new();
    for profile in profiles {
        for (phase, metrics) in profile {
            let entry = totals
                .entry(phase.clone())
                .or_insert_with(|| RuntimePerfPhaseRunResult {
                    duration_ms: 0.0,
                    allocations: zero_allocation_delta(),
                    rss_growth_kb: Some(0),
                });
            entry.duration_ms = round3(entry.duration_ms + metrics.duration_ms);
            entry.allocations = sum_allocation_deltas([&entry.allocations, &metrics.allocations]);
            entry.rss_growth_kb = sum_optional_i64(entry.rss_growth_kb, metrics.rss_growth_kb);
        }
    }
    totals
}

fn mean_phase_profiles<'a>(
    profiles: impl IntoIterator<Item = &'a BTreeMap<String, RuntimePerfPhaseRunResult>>,
) -> BTreeMap<String, RuntimePerfPhaseRunResult> {
    let profiles = profiles.into_iter().collect::<Vec<_>>();
    if profiles.is_empty() {
        return BTreeMap::new();
    }
    let count = profiles.len() as f64;
    let sums = sum_phase_profiles(profiles);
    sums.into_iter()
        .map(|(phase, metrics)| {
            (
                phase,
                RuntimePerfPhaseRunResult {
                    duration_ms: round3(metrics.duration_ms / count),
                    allocations: scale_allocation_delta(&metrics.allocations, count),
                    rss_growth_kb: metrics
                        .rss_growth_kb
                        .map(|value| ((value as f64) / count).round() as i64),
                },
            )
        })
        .collect()
}

fn sum_allocation_deltas<'a>(
    deltas: impl IntoIterator<Item = &'a RuntimePerfAllocationDelta>,
) -> RuntimePerfAllocationDelta {
    let mut total = zero_allocation_delta();
    for delta in deltas {
        total.allocations += delta.allocations;
        total.deallocations += delta.deallocations;
        total.reallocations += delta.reallocations;
        total.bytes_allocated += delta.bytes_allocated;
        total.bytes_deallocated += delta.bytes_deallocated;
        total.bytes_reallocated += delta.bytes_reallocated;
        total.net_live_bytes += delta.net_live_bytes;
    }
    total
}

fn mean_allocation_delta<'a>(
    deltas: impl IntoIterator<Item = &'a RuntimePerfAllocationDelta>,
) -> RuntimePerfAllocationDelta {
    let deltas = deltas.into_iter().collect::<Vec<_>>();
    if deltas.is_empty() {
        return zero_allocation_delta();
    }
    let count = deltas.len() as f64;
    scale_allocation_delta(&sum_allocation_deltas(deltas), count)
}

fn scale_allocation_delta(
    delta: &RuntimePerfAllocationDelta,
    divisor: f64,
) -> RuntimePerfAllocationDelta {
    RuntimePerfAllocationDelta {
        allocations: ((delta.allocations as f64) / divisor).round() as usize,
        deallocations: ((delta.deallocations as f64) / divisor).round() as usize,
        reallocations: ((delta.reallocations as f64) / divisor).round() as usize,
        bytes_allocated: ((delta.bytes_allocated as f64) / divisor).round() as usize,
        bytes_deallocated: ((delta.bytes_deallocated as f64) / divisor).round() as usize,
        bytes_reallocated: ((delta.bytes_reallocated as f64) / divisor).round() as isize,
        net_live_bytes: ((delta.net_live_bytes as f64) / divisor).round() as i64,
    }
}

fn zero_allocation_delta() -> RuntimePerfAllocationDelta {
    RuntimePerfAllocationDelta {
        allocations: 0,
        deallocations: 0,
        reallocations: 0,
        bytes_allocated: 0,
        bytes_deallocated: 0,
        bytes_reallocated: 0,
        net_live_bytes: 0,
    }
}

fn mean_token_usage<'a>(usages: impl IntoIterator<Item = &'a TokenUsage>) -> TokenUsage {
    let usages = usages.into_iter().collect::<Vec<_>>();
    if usages.is_empty() {
        return TokenUsage::default();
    }
    let count = usages.len() as i64;
    TokenUsage {
        input_tokens: usages.iter().map(|usage| usage.input_tokens).sum::<i64>() / count,
        output_tokens: usages.iter().map(|usage| usage.output_tokens).sum::<i64>() / count,
        cached_input_tokens: usages
            .iter()
            .map(|usage| usage.cached_input_tokens)
            .sum::<i64>()
            / count,
        reasoning_tokens: usages
            .iter()
            .map(|usage| usage.reasoning_tokens)
            .sum::<i64>()
            / count,
    }
}

fn mean_option_i64(values: impl IntoIterator<Item = Option<i64>>) -> Option<i64> {
    let values = values.into_iter().flatten().collect::<Vec<_>>();
    if values.is_empty() {
        None
    } else {
        Some((values.iter().sum::<i64>() as f64 / values.len() as f64).round() as i64)
    }
}

fn sum_optional_i64(left: Option<i64>, right: Option<i64>) -> Option<i64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left + right),
        (Some(left), None) => Some(left),
        (None, Some(right)) => Some(right),
        (None, None) => None,
    }
}

fn summarize_metric(mut values: Vec<f64>) -> RuntimePerfMetricSummary {
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let min = *values.first().unwrap_or(&0.0);
    let max = *values.last().unwrap_or(&0.0);
    let median = if values.is_empty() {
        0.0
    } else if values.len().is_multiple_of(2) {
        (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
    } else {
        values[values.len() / 2]
    };
    let mean = if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    };
    RuntimePerfMetricSummary {
        min: round3(min),
        median: round3(median),
        max: round3(max),
        mean: round3(mean),
    }
}

fn summarize_optional_metric(values: Vec<f64>) -> Option<RuntimePerfMetricSummary> {
    if values.is_empty() {
        None
    } else {
        Some(summarize_metric(values))
    }
}

fn phase_name(phase: RuntimeTurnPhase) -> &'static str {
    match phase {
        RuntimeTurnPhase::ContextTransform => "context_transform",
        RuntimeTurnPhase::BeforeTurnHooks => "before_turn_hooks",
        RuntimeTurnPhase::PromptBuild => "prompt_build",
        RuntimeTurnPhase::EffectLoop => "effect_loop",
        RuntimeTurnPhase::FinalizeTurn => "finalize_turn",
        RuntimeTurnPhase::PersistTurn => "persist_turn",
    }
}

fn elapsed_ms(started: Instant) -> f64 {
    round3(started.elapsed().as_secs_f64() * 1000.0)
}

fn round3(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_scenarios_cover_all_synthetic_runtime_paths() {
        assert_eq!(
            resolve_scenarios(&[]).unwrap(),
            RuntimePerfScenario::KNOWN.to_vec()
        );
        assert_eq!(
            resolve_scenarios(&["all".to_string()]).unwrap(),
            RuntimePerfScenario::KNOWN.to_vec()
        );
        assert_eq!(
            resolve_scenarios(&["standard".to_string(), "all".to_string()])
                .unwrap()
                .len(),
            RuntimePerfScenario::KNOWN.len()
        );
    }

    #[test]
    fn openai_compat_stream_fixture_uses_responses_sse_shape() {
        let profile = benchmark_stream_profile(RuntimePerfScenario::OpenAiCompatStream);
        let body = String::from_utf8(openai_compat_sse_body(&profile)).unwrap();
        assert!(
            body.contains(r#""type":"response.output_item.added""#)
                || body.contains(r#""type": "response.output_item.added""#)
        );
        assert!(
            body.contains(r#""type":"response.output_text.delta""#)
                || body.contains(r#""type": "response.output_text.delta""#)
        );
        assert!(
            body.contains(r#""type":"response.completed""#)
                || body.contains(r#""type": "response.completed""#)
        );
        assert!(!body.contains(r#""choices""#));
    }
}

#[cfg(not(feature = "dhat-heap"))]
fn allocator_stats() -> Stats {
    crate::GLOBAL_ALLOCATOR.stats()
}

#[cfg(feature = "dhat-heap")]
fn allocator_stats() -> Stats {
    Stats::default()
}

fn alloc_delta(before: Stats, after: Stats) -> RuntimePerfAllocationDelta {
    let diff = after - before;
    RuntimePerfAllocationDelta {
        allocations: diff.allocations,
        deallocations: diff.deallocations,
        reallocations: diff.reallocations,
        bytes_allocated: diff.bytes_allocated,
        bytes_deallocated: diff.bytes_deallocated,
        bytes_reallocated: diff.bytes_reallocated,
        net_live_bytes: diff.bytes_allocated as i64 - diff.bytes_deallocated as i64,
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ProcessMemorySample {
    rss_kb: Option<u64>,
    hwm_kb: Option<u64>,
}

fn process_memory_sample() -> ProcessMemorySample {
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return ProcessMemorySample::default();
    };

    let mut sample = ProcessMemorySample::default();
    for line in status.lines() {
        if sample.rss_kb.is_none()
            && let Some(value) = parse_status_kb(line, "VmRSS:")
        {
            sample.rss_kb = Some(value);
        }
        if sample.hwm_kb.is_none()
            && let Some(value) = parse_status_kb(line, "VmHWM:")
        {
            sample.hwm_kb = Some(value);
        }
        if sample.rss_kb.is_some() && sample.hwm_kb.is_some() {
            break;
        }
    }
    sample
}

fn parse_status_kb(line: &str, key: &str) -> Option<u64> {
    let value = line.strip_prefix(key)?.trim();
    value.split_whitespace().next()?.parse::<u64>().ok()
}

fn diff_opt_i64(before: Option<u64>, after: Option<u64>) -> Option<i64> {
    Some(after? as i64 - before? as i64)
}
