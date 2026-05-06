mod bench_tools;
mod dataset;

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, bail};
use bench_tools::{BenchmarkQuestionContext, LongMemEvalSessionTools};
use chrono::Utc;
use clap::{ArgAction, Parser, ValueEnum};
use dataset::{LongMemEvalQuestion, load_questions};
use lash::plugin::{PluginFactory, PluginSpec, StaticPluginFactory};
use lash::{
    AppendSessionNodesRequest, BackgroundRuntimeHost, BuiltinToolResultProjectionPluginFactory,
    EmbeddedRuntimeHost, EventSink, ExecutionMode, InputItem, LashRuntime, PersistedSessionState,
    PersistentRuntimeServices, PluginHost, PromptSlot, PromptTemplate, PromptTemplateEntry,
    PromptTemplateSection, ProviderHandle, RuntimeCoreConfig, RuntimePersistence,
    SessionAppendNode, SessionEvent, SessionPolicy, SessionUsageReport, StandardContextApproach,
    TokioSessionTaskExecutor, TurnInjectionBridge, TurnInput, TurnInputInjectionBridge,
    diff_usage_reports,
};
use lash_llm_tools::LlmToolsPluginFactory;
use lash_plugin_observational_memory::ObservationalMemoryPluginFactory;
use lash_plugin_rolling_history::RollingHistoryPluginFactory;
use lash_provider_openai::OPENROUTER_BASE_URL;
use lash_rlm_types::{RlmGlobalsPatchPluginBody, RlmModeEvent};
use lash_sqlite_store::Store;
use lash_subagents::{LocalSubagentHost, SubagentHost, SubagentsPluginFactory};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

const STATE_ROOT: &str = ".benchmarks/longmemeval-rlm";
const DEFAULT_MODEL: &str = "google/gemini-3-flash-preview";
const DEFAULT_PROVIDER_ID: &str = "openai-compatible";
const DEFAULT_CONTEXT_APPROACH: &str = "rolling_history";
const DEFAULT_EXECUTION_MODE: &str = "rlm";
const DEFAULT_MAX_CONTEXT_TOKENS: usize = 1_000_000;
const DEFAULT_MAX_QUESTION_CONTEXT_TOKENS: i64 = 3_000_000;
const CLEANED_S_URL: &str = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json";
const FLASH_FAILURES_64_URL: &str = "https://raw.githubusercontent.com/rawwerks/longmemeval-rlm/master/data/longmemeval_s_flash_failures_64.json";
const DISCORDANT_110_URL: &str =
    "https://raw.githubusercontent.com/rawwerks/longmemeval-rlm/master/data/discordant_110.json";

#[derive(Clone, Copy, Debug, ValueEnum)]
enum PromptProfile {
    Baseline,
    TemporalObservations,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum DatasetPreset {
    #[value(name = "cleaned-s")]
    CleanedS,
    #[value(name = "flash-failures-64")]
    FlashFailures64,
    #[value(name = "discordant-110")]
    Discordant110,
}

impl DatasetPreset {
    fn file_name(self) -> &'static str {
        match self {
            Self::CleanedS => "longmemeval_s_cleaned.json",
            Self::FlashFailures64 => "longmemeval_s_flash_failures_64.json",
            Self::Discordant110 => "discordant_110.json",
        }
    }

    fn default_url(self) -> &'static str {
        match self {
            Self::CleanedS => CLEANED_S_URL,
            Self::FlashFailures64 => FLASH_FAILURES_64_URL,
            Self::Discordant110 => DISCORDANT_110_URL,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::CleanedS => "cleaned_s",
            Self::FlashFailures64 => "flash_failures_64",
            Self::Discordant110 => "discordant_110",
        }
    }
}

#[derive(Parser, Debug, Clone)]
#[command(name = "bench-longmemeval-rlm")]
#[command(about = "Run LongMemEval through Lash as an RLM-style memory benchmark.")]
struct Args {
    #[arg(long, value_enum, default_value_t = DatasetPreset::CleanedS)]
    dataset_preset: DatasetPreset,

    #[arg(long)]
    dataset_url: Option<String>,

    #[arg(long)]
    dataset: Option<PathBuf>,

    #[arg(long)]
    run_id: Option<String>,

    #[arg(long)]
    output_dir: Option<PathBuf>,

    #[arg(long, default_value = DEFAULT_MODEL)]
    model: String,

    #[arg(long, default_value = DEFAULT_PROVIDER_ID)]
    provider_id: String,

    #[arg(long)]
    variant: Option<String>,

    #[arg(long)]
    api_key: Option<String>,

    #[arg(long)]
    base_url: Option<String>,

    #[arg(long, default_value = DEFAULT_EXECUTION_MODE)]
    execution_mode: String,

    #[arg(long)]
    standard_context_approach: Option<String>,

    #[arg(long, default_value_t = DEFAULT_MAX_CONTEXT_TOKENS)]
    max_context_tokens: usize,

    #[arg(long, default_value_t = DEFAULT_MAX_QUESTION_CONTEXT_TOKENS)]
    max_question_context_tokens: i64,

    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    session_tools: bool,

    #[arg(long)]
    no_session_tools: bool,

    #[arg(long, value_enum, default_value_t = PromptProfile::Baseline)]
    prompt_profile: PromptProfile,

    #[arg(long)]
    limit: Option<usize>,

    #[arg(long, default_value_t = 0)]
    offset: usize,

    #[arg(long)]
    question_id: Vec<String>,

    #[arg(long)]
    resume: bool,

    #[arg(long)]
    await_background_work: bool,

    #[arg(long, default_value_t = 10)]
    batch_size: usize,

    #[arg(long)]
    dry_run: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RunManifest {
    run_id: String,
    dataset_preset: String,
    dataset: String,
    dataset_url: Option<String>,
    model: String,
    provider_id: String,
    variant: Option<String>,
    execution_mode: String,
    standard_context_approach: Option<String>,
    prompt_profile: String,
    session_tools: bool,
    batch_size: usize,
    max_question_context_tokens: i64,
    question_count: usize,
    created_at: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct QuestionResult {
    question_id: String,
    hypothesis: String,
    question_type: Option<String>,
    elapsed_seconds: f64,
    status: String,
    done_reason: String,
    iterations: usize,
    llm_calls: usize,
    retry_count: usize,
    error_count: usize,
    failure_reason: Option<String>,
    partial_output: Option<String>,
    observed_context_tokens: i64,
    token_budget_limit: Option<i64>,
    token_budget_exceeded: bool,
    provider_cost: ProviderCostSummary,
    usage: SessionUsageReport,
    tool_calls: usize,
    trace_path: String,
    session_db_path: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RunSummary {
    run_id: String,
    started_at: String,
    finished_at: String,
    duration_seconds: i64,
    question_count: usize,
    result_count: usize,
    completed_question_count: usize,
    failed_question_count: usize,
    interrupted_question_count: usize,
    status_counts: BTreeMap<String, usize>,
    llm_calls: usize,
    iterations: usize,
    retry_count: usize,
    questions_with_retries: usize,
    error_count: usize,
    questions_with_errors: usize,
    token_budget_exceeded_question_count: usize,
    provider_cost: ProviderCostSummary,
    usage: SessionUsageReport,
    results: Vec<QuestionResult>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct ProviderCostSummary {
    total_cost_credits: f64,
    total_upstream_inference_cost_credits: f64,
    cost_entry_count: usize,
    upstream_inference_cost_entry_count: usize,
}

#[derive(Clone, Debug)]
struct DatasetSpec {
    preset: DatasetPreset,
    path: PathBuf,
    url: Option<String>,
}

#[derive(Clone, Debug, Default)]
struct TraceMetrics {
    iterations: usize,
    llm_calls: usize,
    provider_cost: ProviderCostSummary,
}

#[derive(Clone, Debug)]
struct SinkErrorRecord {
    message: String,
    kind: Option<String>,
    code: Option<String>,
    raw: Option<String>,
}

#[derive(Clone, Debug, Default)]
struct LiveTokenBudget {
    max_context_tokens: Option<i64>,
    observed_context_tokens: i64,
    exceeded: bool,
}

impl LiveTokenBudget {
    fn new(max_context_tokens: i64) -> Self {
        Self {
            max_context_tokens: (max_context_tokens > 0).then_some(max_context_tokens),
            observed_context_tokens: 0,
            exceeded: false,
        }
    }

    fn record(&mut self, usage: &lash::TokenUsage) -> bool {
        self.observed_context_tokens = self
            .observed_context_tokens
            .saturating_add(context_tokens_for_usage(usage));
        if let Some(limit) = self.max_context_tokens
            && self.observed_context_tokens > limit
        {
            self.exceeded = true;
            return true;
        }
        false
    }
}

#[derive(Clone, Debug)]
struct TokenBudgetExceeded {
    observed_context_tokens: i64,
    max_context_tokens: i64,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _ = dotenvy::dotenv();
    let mut args = Args::parse();
    if args.no_session_tools {
        args.session_tools = false;
    }

    let state_root = PathBuf::from(STATE_ROOT);
    let data_dir = state_root.join("data");
    let runs_dir = state_root.join("runs");
    fs::create_dir_all(&data_dir).with_context(|| format!("create {}", data_dir.display()))?;
    fs::create_dir_all(&runs_dir).with_context(|| format!("create {}", runs_dir.display()))?;

    let dataset = resolve_dataset_spec(&args, &data_dir);
    ensure_dataset(&dataset).await?;
    let questions = select_questions(load_questions(&dataset.path)?, &args)?;
    if questions.is_empty() {
        bail!("no benchmark entries selected");
    }

    let run_id = args
        .run_id
        .clone()
        .unwrap_or_else(|| format!("run-{}", Utc::now().format("%Y%m%dT%H%M%SZ")));
    let output_dir = args
        .output_dir
        .clone()
        .unwrap_or_else(|| runs_dir.join(&run_id));
    fs::create_dir_all(&output_dir).with_context(|| format!("create {}", output_dir.display()))?;

    let execution_mode = parse_execution_mode(&args.execution_mode)?;
    let standard_context_approach = resolve_standard_context_approach(
        &execution_mode,
        args.standard_context_approach.as_deref(),
    )?;

    let manifest = RunManifest {
        run_id: run_id.clone(),
        dataset_preset: dataset.preset.label().to_string(),
        dataset: dataset.path.display().to_string(),
        dataset_url: dataset.url.clone(),
        model: args.model.clone(),
        provider_id: args.provider_id.clone(),
        variant: args.variant.clone(),
        execution_mode: execution_mode_label(&execution_mode).to_string(),
        standard_context_approach: standard_context_approach
            .as_ref()
            .map(standard_context_approach_label)
            .map(str::to_string),
        prompt_profile: match args.prompt_profile {
            PromptProfile::Baseline => "baseline",
            PromptProfile::TemporalObservations => "temporal_observations",
        }
        .to_string(),
        session_tools: args.session_tools,
        batch_size: args.batch_size.max(1),
        max_question_context_tokens: args.max_question_context_tokens,
        question_count: questions.len(),
        created_at: Utc::now().to_rfc3339(),
    };
    write_json(output_dir.join("run.json"), &manifest)?;

    if args.dry_run {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "manifest": manifest,
                "questions": questions.iter().map(|q| json!({
                    "question_id": q.question_id,
                    "question_type": q.question_type,
                    "question_date": q.question_date,
                })).collect::<Vec<_>>(),
            }))?
        );
        return Ok(());
    }

    let provider = resolve_provider(&args)?;

    let started_at = Utc::now();
    let hypotheses_path = output_dir.join("hypotheses.jsonl");
    let completed = if args.resume {
        load_completed_ids(&hypotheses_path)?
    } else {
        BTreeSet::new()
    };
    let pending_questions = questions
        .into_iter()
        .enumerate()
        .filter(|(_, question)| !completed.contains(&question.question_id))
        .collect::<Vec<_>>();
    let total_selected = manifest.question_count;
    let args = Arc::new(args);
    let provider = Arc::new(provider);
    let output_dir = Arc::new(output_dir);
    let semaphore = Arc::new(Semaphore::new(manifest.batch_size.max(1)));
    let mut join_set = JoinSet::new();
    for (index, question) in pending_questions {
        eprintln!(
            "[{}/{}] {} ({})",
            index + 1,
            total_selected,
            question.question_id,
            question
                .question_type
                .clone()
                .unwrap_or_else(|| "unknown".to_string())
        );
        let permit = semaphore
            .clone()
            .acquire_owned()
            .await
            .context("acquire benchmark batch slot")?;
        let output_dir = output_dir.clone();
        let provider = provider.clone();
        let args = args.clone();
        let standard_context_approach = standard_context_approach.clone();
        let execution_mode = execution_mode.clone();
        join_set.spawn(async move {
            let _permit = permit;
            let result = run_question(
                output_dir.as_ref(),
                provider.as_ref(),
                args.as_ref(),
                execution_mode,
                standard_context_approach.as_ref(),
                question,
            )
            .await;
            (index, result)
        });
    }

    let mut indexed_results = Vec::new();
    while let Some(joined) = join_set.join_next().await {
        let (index, result) = match joined {
            Ok(value) => value,
            Err(err) => {
                join_set.abort_all();
                return Err(anyhow::anyhow!("benchmark task failed: {err}"));
            }
        };
        let result = match result {
            Ok(result) => result,
            Err(err) => {
                join_set.abort_all();
                return Err(err);
            }
        };
        append_jsonl(
            &hypotheses_path,
            &json!({
                "question_id": result.question_id,
                "hypothesis": result.hypothesis,
            }),
        )?;
        indexed_results.push((index, result));
    }
    indexed_results.sort_by_key(|(index, _)| *index);
    let results = indexed_results
        .into_iter()
        .map(|(_, result)| result)
        .collect::<Vec<_>>();

    let finished_at = Utc::now();
    let usage = aggregate_usage(results.iter().map(|result| result.usage.clone()));
    let status_counts = aggregate_status_counts(&results);
    let summary = RunSummary {
        run_id,
        started_at: started_at.to_rfc3339(),
        finished_at: finished_at.to_rfc3339(),
        duration_seconds: (finished_at - started_at).num_seconds(),
        question_count: manifest.question_count,
        result_count: results.len(),
        completed_question_count: *status_counts.get("completed").unwrap_or(&0),
        failed_question_count: *status_counts.get("failed").unwrap_or(&0),
        interrupted_question_count: *status_counts.get("interrupted").unwrap_or(&0),
        status_counts,
        llm_calls: results.iter().map(|result| result.llm_calls).sum(),
        iterations: results.iter().map(|result| result.iterations).sum(),
        retry_count: results.iter().map(|result| result.retry_count).sum(),
        questions_with_retries: results
            .iter()
            .filter(|result| result.retry_count > 0)
            .count(),
        error_count: results.iter().map(|result| result.error_count).sum(),
        questions_with_errors: results
            .iter()
            .filter(|result| result.error_count > 0)
            .count(),
        token_budget_exceeded_question_count: results
            .iter()
            .filter(|result| result.token_budget_exceeded)
            .count(),
        provider_cost: aggregate_provider_cost(results.iter().map(|result| &result.provider_cost)),
        usage,
        results,
    };
    write_json(output_dir.join("summary.json"), &summary)?;
    write_summary_text(output_dir.join("summary.txt"), &summary)?;
    Ok(())
}

fn resolve_dataset_spec(args: &Args, data_dir: &Path) -> DatasetSpec {
    DatasetSpec {
        preset: args.dataset_preset,
        path: args
            .dataset
            .clone()
            .unwrap_or_else(|| data_dir.join(args.dataset_preset.file_name())),
        url: args
            .dataset_url
            .clone()
            .or_else(|| Some(args.dataset_preset.default_url().to_string())),
    }
}

async fn ensure_dataset(dataset: &DatasetSpec) -> anyhow::Result<()> {
    if dataset.path.exists() {
        return Ok(());
    }
    let url = dataset.url.as_deref().ok_or_else(|| {
        anyhow::anyhow!(
            "dataset {} does not exist locally and no download URL was provided",
            dataset.path.display()
        )
    })?;
    if let Some(parent) = dataset.path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let response = Client::new()
        .get(url)
        .send()
        .await
        .with_context(|| format!("download dataset from {url}"))?
        .error_for_status()
        .with_context(|| format!("dataset download failed from {url}"))?;
    let bytes = response.bytes().await.context("read dataset body")?;
    fs::write(&dataset.path, &bytes)
        .with_context(|| format!("write dataset {}", dataset.path.display()))?;
    Ok(())
}

fn select_questions(
    questions: Vec<LongMemEvalQuestion>,
    args: &Args,
) -> anyhow::Result<Vec<LongMemEvalQuestion>> {
    let filtered = if args.question_id.is_empty() {
        questions
    } else {
        let wanted = args.question_id.iter().cloned().collect::<BTreeSet<_>>();
        questions
            .into_iter()
            .filter(|question| wanted.contains(&question.question_id))
            .collect()
    };
    let sliced = filtered
        .into_iter()
        .skip(args.offset)
        .take(args.limit.unwrap_or(usize::MAX))
        .collect::<Vec<_>>();
    Ok(sliced)
}

async fn run_question(
    output_dir: &Path,
    provider: &ProviderHandle,
    args: &Args,
    execution_mode: ExecutionMode,
    standard_context_approach: Option<&StandardContextApproach>,
    question: LongMemEvalQuestion,
) -> anyhow::Result<QuestionResult> {
    let question_dir = output_dir.join("questions").join(&question.question_id);
    fs::create_dir_all(&question_dir)
        .with_context(|| format!("create {}", question_dir.display()))?;
    write_json(question_dir.join("question.json"), &question)?;

    let benchmark_context = BenchmarkQuestionContext::new(question.clone());
    let prompt = build_prompt(&benchmark_context, args.prompt_profile, args.session_tools);
    fs::write(question_dir.join("prompt.txt"), &prompt)
        .with_context(|| format!("write {}", question_dir.join("prompt.txt").display()))?;

    let store_path = question_dir.join("session.db");
    let trace_path = question_dir.join("session.trace.jsonl");
    let store = Arc::new(
        Store::open(&store_path).with_context(|| format!("open {}", store_path.display()))?,
    );
    let policy = SessionPolicy {
        model: args.model.clone(),
        provider: provider.clone(),
        max_context_tokens: Some(args.max_context_tokens),
        execution_mode: execution_mode.clone(),
        standard_context_approach: standard_context_approach.cloned(),
        model_variant: args.variant.clone(),
        ..SessionPolicy::default()
    };
    let root_plugins = build_plugin_session(
        execution_mode,
        standard_context_approach.cloned(),
        args.session_tools,
        benchmark_context,
        &policy,
    )?;
    let services = PersistentRuntimeServices::new_with_bridges(
        root_plugins,
        TurnInjectionBridge::new(),
        TurnInputInjectionBridge::new(),
        store.clone() as Arc<dyn RuntimePersistence>,
    );
    let host = BackgroundRuntimeHost::new(
        EmbeddedRuntimeHost::new(
            RuntimeCoreConfig::default()
                .with_trace_jsonl_path(Some(trace_path.clone()))
                .with_prompt_template(prompt_template(args.prompt_profile, args.session_tools)),
        ),
        Arc::new(TokioSessionTaskExecutor::default()),
    );
    let mut runtime = LashRuntime::from_persistent_background_state(
        policy.clone(),
        host,
        services,
        PersistedSessionState {
            session_id: "root".to_string(),
            policy,
            ..PersistedSessionState::default()
        },
    )
    .await?;

    runtime
        .append_session_nodes(AppendSessionNodesRequest {
            nodes: vec![SessionAppendNode::event(lash::SessionEventRecord::Mode(
                lash::ModeEvent::rlm(RlmModeEvent::RlmGlobalsPatch(build_globals_patch(
                    &question,
                ))),
            ))],
            requires_ancestor_node_id: None,
        })
        .await
        .context("append benchmark globals patch")?;

    let before_usage = runtime.usage_report();
    let cancel = tokio_util::sync::CancellationToken::new();
    let sink = JsonlEventSink::new(
        question_dir.join("events.jsonl"),
        args.max_question_context_tokens,
        cancel.clone(),
    )?;
    let started_at = std::time::Instant::now();
    let turn = runtime
        .stream_turn(
            TurnInput {
                items: vec![InputItem::Text { text: prompt }],
                image_blobs: Default::default(),
                user_input: None,
                mode: None,
                mode_turn_options: None,
            },
            &sink,
            cancel,
        )
        .await
        .context("run benchmark question")?;
    if args.await_background_work {
        runtime.await_background_work().await?;
    }
    let elapsed_seconds = started_at.elapsed().as_secs_f64();
    let after_usage = runtime.usage_report();
    let usage = diff_usage_reports(&before_usage, &after_usage)
        .map(|rows| SessionUsageReport::from_entries(&rows))
        .map_err(anyhow::Error::msg)
        .context("diff usage reports")?;
    let token_budget = sink.token_budget();
    let partial_output = sink
        .last_llm_response()
        .or_else(|| non_empty_text(&turn.assistant_output.safe_text));
    let answer = if turn_completed(&turn.outcome) && token_budget.is_none() {
        partial_output.clone().unwrap_or_default()
    } else {
        String::new()
    };
    fs::write(question_dir.join("answer.txt"), format!("{answer}\n"))
        .with_context(|| format!("write {}", question_dir.join("answer.txt").display()))?;
    if let Some(partial_output) = partial_output.as_ref().filter(|value| **value != answer) {
        fs::write(
            question_dir.join("partial_output.txt"),
            format!("{partial_output}\n"),
        )
        .with_context(|| {
            format!(
                "write {}",
                question_dir.join("partial_output.txt").display()
            )
        })?;
    }

    let trace_metrics = collect_trace_metrics(&trace_path).context("collect trace metrics")?;
    let error_records = sink.error_records();
    let failure_reason = if let Some(budget) = token_budget.as_ref() {
        Some(format!(
            "token budget exceeded: observed_context_tokens={} limit={}",
            budget.observed_context_tokens, budget.max_context_tokens
        ))
    } else {
        format_failure_reason(&turn, &error_records)
    };
    if let Some(reason) = &failure_reason {
        fs::write(question_dir.join("failure.txt"), format!("{reason}\n"))
            .with_context(|| format!("write {}", question_dir.join("failure.txt").display()))?;
    }
    let result = QuestionResult {
        question_id: question.question_id.clone(),
        hypothesis: answer.clone(),
        question_type: question.question_type.clone(),
        elapsed_seconds,
        status: if token_budget.is_some() {
            "failed".to_string()
        } else {
            turn_status_label(&turn.outcome).to_string()
        },
        done_reason: if token_budget.is_some() {
            "token_budget".to_string()
        } else {
            done_reason_label(&turn.outcome).to_string()
        },
        iterations: trace_metrics.iterations.max(sink.iteration_count()),
        llm_calls: trace_metrics.llm_calls.max(sink.llm_call_count()),
        retry_count: sink.retry_count(),
        error_count: error_records.len(),
        failure_reason,
        partial_output: partial_output.filter(|value| *value != answer),
        observed_context_tokens: sink.observed_context_tokens(),
        token_budget_limit: sink.max_question_context_tokens(),
        token_budget_exceeded: token_budget.is_some(),
        provider_cost: trace_metrics.provider_cost,
        usage,
        tool_calls: turn.tool_calls.len(),
        trace_path: trace_path.display().to_string(),
        session_db_path: store_path.display().to_string(),
    };
    write_json(question_dir.join("result.json"), &result)?;
    Ok(result)
}

fn build_plugin_session(
    execution_mode: ExecutionMode,
    standard_context_approach: Option<StandardContextApproach>,
    session_tools: bool,
    benchmark_context: BenchmarkQuestionContext,
    session_policy: &SessionPolicy,
) -> anyhow::Result<Arc<lash::PluginSession>> {
    let mut factories: Vec<Arc<dyn PluginFactory>> =
        vec![Arc::new(BuiltinToolResultProjectionPluginFactory::default())];
    if let Some(standard_context_approach) = &standard_context_approach {
        match standard_context_approach {
            StandardContextApproach::RollingHistory(_) => {
                factories.push(Arc::new(RollingHistoryPluginFactory::default()));
            }
            StandardContextApproach::ObservationalMemory(_) => {
                factories.push(Arc::new(ObservationalMemoryPluginFactory));
            }
        }
    }
    let mut subagent_models = std::collections::BTreeMap::new();
    subagent_models.insert("explore".to_string(), session_policy.model.clone());
    subagent_models.insert("peer".to_string(), session_policy.model.clone());
    let registry = std::sync::Arc::new(lash_subagents::default_registry(
        &subagent_models,
        ExecutionMode::standard(),
    ));
    let subagent_host: Arc<dyn SubagentHost> = Arc::new(LocalSubagentHost::default());
    factories.push(Arc::new(LlmToolsPluginFactory));
    factories.push(Arc::new(SubagentsPluginFactory::new(
        session_policy.clone(),
        registry,
        subagent_host,
    )));
    if session_tools {
        factories.push(Arc::new(StaticPluginFactory::new(
            "longmemeval_tools",
            PluginSpec::new()
                .with_tool_provider(Arc::new(LongMemEvalSessionTools::new(benchmark_context))),
        )));
    }
    let plugin_host = PluginHost::new(factories);
    plugin_host
        .build_session("root", execution_mode, standard_context_approach, None)
        .context("build plugin session")
}

fn build_globals_patch(question: &LongMemEvalQuestion) -> RlmGlobalsPatchPluginBody {
    let mut set = serde_json::Map::new();
    set.insert(
        "benchmark".to_string(),
        json!({
            "name": "LongMemEval",
            "question_id": question.question_id,
            "question_type": question.question_type,
            "question_date": question.question_date,
        }),
    );
    set.insert(
        "input".to_string(),
        json!({
            "question": question.question,
            "question_type": question.question_type,
            "question_date": question.question_date,
            "haystack_dates": question.haystack_dates,
            "haystack_session_ids": question.haystack_session_ids,
            "haystack_sessions": question.haystack_sessions,
        }),
    );
    RlmGlobalsPatchPluginBody {
        set,
        set_default: serde_json::Map::new(),
        unset: Vec::new(),
    }
}

fn build_prompt(
    question: &BenchmarkQuestionContext,
    profile: PromptProfile,
    session_tools: bool,
) -> String {
    let profile_guidance = match profile {
        PromptProfile::Baseline => None,
        PromptProfile::TemporalObservations => {
            Some("If useful, keep a short internal evidence ledger before finalizing.")
        }
    };
    let mut prompt = format!(
        "Question: {user_question}\nAsked on: {question_date}\nType: {question_type}\n\nRequirements:\n- prefer the most recent relevant fact\n- verify dates and entities before answering\n- if the history does not support an answer, say \"I don't know\"\n- the submitted final response must be plain prose",
        question_date = question
            .question
            .question_date
            .as_deref()
            .unwrap_or("unknown"),
        question_type = question
            .question
            .question_type
            .as_deref()
            .unwrap_or("unknown"),
        user_question = question.question.question,
    );
    if !session_tools {
        prompt.push_str(
            "\n- do not invent retrieval or search tools\n- start by judging the size of the provided history and plan accordingly\n- if the history is large, narrow to likely sessions or date ranges before inspecting details\n- use `spawn_agent` for focused subproblems; keep each child task bounded and concrete\n- avoid observing or printing the entire history unless it is already small",
        );
    }
    if let Some(extra) = profile_guidance {
        prompt.push_str("\n\n");
        prompt.push_str(extra);
    }
    prompt
}

fn prompt_template(profile: PromptProfile, session_tools: bool) -> PromptTemplate {
    let mut execution_entries = vec![PromptTemplateEntry::text(if session_tools {
        r#"In this mode you work by writing `lashlang` code inside your response and the runtime executes it.

Format each work step like this:

````
Brief reasoning here in plain prose.

```lashlang
result = (call tool_name { arg: value })?
print result
```
````

- Wrap each work step in exactly one ` ```lashlang ` fenced block. Only the first block runs per turn.
- Keep prose short. It is only a compact reasoning trace.
- After each result, either write another fenced block to continue or call `submit <value>` from inside a fenced `lashlang` block to end the turn.
- When you are done, call `submit <value>` from inside a fenced `lashlang` block. Do not end in prose without a fenced block.
- Variables persist across iterations.
- You can update variable-rooted collection paths: `record.field = value`, `record[key] = value`, `list[i] = value`, and nested forms. Record assignment inserts/replaces fields; list assignment replaces an existing integer index only. Dynamic record reads return `null` when missing, so `counts[g] = counts[g] + 1` works for histograms.
- If the prompt includes bound variables, use them directly.
- Call tools with `call tool_name { arg: expr }`. Tool calls return `{ ok, value/error }` wrappers; use `(call tool_name { arg: expr })?` for the normal fail-fast path.
- Start background work with `start call tool_name { arg: expr }`, wait with `await handle`, and stop it with `cancel handle`. Prefer `await { name: handle }` for multiple handles so results are named. Use `(await handle)?` for the normal fail-fast path.
- Use `print expr` to inspect a value mid-turn (keeps running). Use `submit <expr>` to end with a final answer.
- In `for` loops, `break` exits the nearest loop and `continue` skips to the next iteration. `submit` still exits the whole turn.
- Break large tasks into smaller, bounded steps instead of brute-force scanning."#
    } else {
        r#"In this mode you work by writing `lashlang` code inside your response and the runtime executes it.

Format each work step like this:

````
Brief reasoning here in plain prose.

```lashlang
candidate = start call spawn_agent { agent_name: "narrow_candidates", task: "narrow the search to likely sessions", capability: "explore" }
result = (await candidate)?
print result
```
````

- Wrap each work step in exactly one ` ```lashlang ` fenced block. Only the first block runs per turn.
- Keep prose short. It is only a compact reasoning trace.
- After each result, either write another fenced block to continue or call `submit <value>` from inside a fenced `lashlang` block to end the turn.
- When you are done, call `submit <value>` from inside a fenced `lashlang` block. Do not end in prose without a fenced block.
- Variables persist across iterations.
- You can update variable-rooted collection paths: `record.field = value`, `record[key] = value`, `list[i] = value`, and nested forms. Record assignment inserts/replaces fields; list assignment replaces an existing integer index only. Dynamic record reads return `null` when missing, so `counts[g] = counts[g] + 1` works for histograms.
- If the prompt includes bound variables, use them directly.
- In this run, do not assume any retrieval or search tools exist. The only helper tools are the subagent tools.
- If there is no Available Tools section, do not invent tool names.
- Tool calls return `{ ok, value/error }` wrappers; use `(call tool_name { arg: expr })?` for the normal fail-fast path.
- Start by checking the size and shape of the bound input so you can plan the search.
- If the history is large, work hierarchically: narrow candidate sessions or date ranges first, then inspect those candidates, then verify the final answer.
- Use `spawn_agent` for focused recursive subproblems such as narrowing candidate sessions, extracting date candidates, or verifying one hypothesis.
- Keep each child task bounded and concrete. Do not fan out one agent per session unless the narrowed candidate set is already small.
- Use `print expr` to inspect a value mid-turn (keeps running). Use `submit <expr>` for the final answer.
- In `for` loops, `break` exits the nearest loop and `continue` skips to the next iteration. `submit` still exits the whole turn.
- Avoid printing the entire haystack unless it is already small."#
    })];
    if matches!(profile, PromptProfile::TemporalObservations) {
        execution_entries.push(PromptTemplateEntry::text(
            "Before answering, explicitly ground your reasoning in session/date evidence and resolve entity ambiguity before producing the final answer.",
        ));
    }
    PromptTemplate::new(vec![
        PromptTemplateSection::untitled(vec![PromptTemplateEntry::text(
            "You are answering a memory question over prior conversation history.",
        )]),
        PromptTemplateSection::titled("Execution", execution_entries),
        PromptTemplateSection::titled(
            "Guidance",
            vec![
                PromptTemplateEntry::slot(PromptSlot::ProjectInstructions),
                PromptTemplateEntry::slot(PromptSlot::Guidance),
            ],
        ),
        PromptTemplateSection::titled(
            "Environment",
            vec![
                PromptTemplateEntry::slot(PromptSlot::RuntimeContext),
                PromptTemplateEntry::slot(PromptSlot::Environment),
            ],
        ),
    ])
}

fn resolve_provider(args: &Args) -> anyhow::Result<ProviderHandle> {
    match args.provider_id.as_str() {
        "openai-compatible" | "openai-generic" => {
            let api_key = resolve_api_key(args).ok_or_else(|| {
                anyhow::anyhow!(
                    "missing API key for LongMemEval runner; set OPENROUTER_API_KEY or OPENAI_COMPATIBLE_API_KEY in .env, or pass --api-key"
                )
            })?;
            let provider =
                lash_provider_openai::OpenAiGenericProvider::new(api_key, resolve_base_url(args));
            Ok(ProviderHandle::new(provider.into_components()))
        }
        other => bail!(
            "provider `{other}` is not supported by this harness; use the OpenAI-compatible path with an API key from .env"
        ),
    }
}

fn resolve_api_key(args: &Args) -> Option<String> {
    args.api_key
        .clone()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| read_env_var("OPENAI_COMPATIBLE_API_KEY"))
        .or_else(|| read_env_var("OPENROUTER_API_KEY"))
}

fn resolve_base_url(args: &Args) -> String {
    args.base_url
        .clone()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| read_env_var("OPENAI_COMPATIBLE_BASE_URL"))
        .or_else(|| read_env_var("OPENROUTER_BASE_URL"))
        .unwrap_or_else(|| OPENROUTER_BASE_URL.to_string())
}

fn read_env_var(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn parse_execution_mode(raw: &str) -> anyhow::Result<ExecutionMode> {
    match raw {
        "rlm" => Ok(ExecutionMode::new("rlm")),
        "standard" => Ok(ExecutionMode::standard()),
        _ => bail!("unsupported execution mode `{raw}`"),
    }
}

fn parse_standard_context_approach(raw: &str) -> anyhow::Result<StandardContextApproach> {
    match raw {
        "rolling_history" => Ok(StandardContextApproach::RollingHistory(Default::default())),
        "observational_memory" => Ok(StandardContextApproach::ObservationalMemory(
            Default::default(),
        )),
        _ => bail!("unsupported context approach `{raw}`"),
    }
}

fn resolve_standard_context_approach(
    execution_mode: &ExecutionMode,
    raw: Option<&str>,
) -> anyhow::Result<Option<StandardContextApproach>> {
    if *execution_mode == ExecutionMode::standard() {
        return parse_standard_context_approach(raw.unwrap_or(DEFAULT_CONTEXT_APPROACH)).map(Some);
    }
    if raw.is_some() {
        bail!("--context-approach only applies to --execution-mode standard");
    }
    Ok(None)
}

fn execution_mode_label(mode: &ExecutionMode) -> &str {
    mode.plugin_id()
}

fn standard_context_approach_label(approach: &StandardContextApproach) -> &'static str {
    match approach {
        StandardContextApproach::RollingHistory(_) => "rolling_history",
        StandardContextApproach::ObservationalMemory(_) => "observational_memory",
    }
}

fn write_json(path: PathBuf, value: &impl Serialize) -> anyhow::Result<()> {
    let text = serde_json::to_string_pretty(value)?;
    fs::write(&path, format!("{text}\n")).with_context(|| format!("write {}", path.display()))
}

fn append_jsonl(path: &Path, value: &Value) -> anyhow::Result<()> {
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("open {}", path.display()))?;
    writeln!(file, "{}", serde_json::to_string(value)?)
        .with_context(|| format!("append {}", path.display()))
}

fn load_completed_ids(path: &Path) -> anyhow::Result<BTreeSet<String>> {
    if !path.exists() {
        return Ok(BTreeSet::new());
    }
    let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut out = BTreeSet::new();
    for line in raw.lines().filter(|line| !line.trim().is_empty()) {
        let value: Value = serde_json::from_str(line)
            .with_context(|| format!("parse completed row from {}", path.display()))?;
        if let Some(question_id) = value.get("question_id").and_then(Value::as_str) {
            out.insert(question_id.to_string());
        }
    }
    Ok(out)
}

fn aggregate_usage(reports: impl IntoIterator<Item = SessionUsageReport>) -> SessionUsageReport {
    let mut total = BTreeMap::<(String, String), lash::TokenUsage>::new();
    for report in reports {
        for row in report.by_source_model {
            let key = (row.source.clone(), row.model.clone());
            let entry = total.entry(key).or_default();
            entry.input_tokens += row.usage.input_tokens;
            entry.output_tokens += row.usage.output_tokens;
            entry.cached_input_tokens += row.usage.cached_input_tokens;
            entry.reasoning_tokens += row.usage.reasoning_tokens;
        }
    }
    let entries = total
        .into_iter()
        .map(|((source, model), usage)| lash::TokenLedgerEntry {
            source,
            model,
            usage,
        })
        .collect::<Vec<_>>();
    SessionUsageReport::from_entries(&entries)
}

fn aggregate_status_counts(results: &[QuestionResult]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for result in results {
        *counts.entry(result.status.clone()).or_insert(0) += 1;
    }
    counts
}

fn write_summary_text(path: PathBuf, summary: &RunSummary) -> anyhow::Result<()> {
    let mut lines = vec![
        format!("Run: {}", summary.run_id),
        format!(
            "Questions completed: {}/{}",
            summary.completed_question_count, summary.question_count
        ),
        format!("Questions failed: {}", summary.failed_question_count),
        format!(
            "Questions interrupted: {}",
            summary.interrupted_question_count
        ),
        format!("Result rows: {}", summary.result_count),
        format!("LLM calls: {}", summary.llm_calls),
        format!("Iterations: {}", summary.iterations),
        format!(
            "Retries: {} across {} question(s)",
            summary.retry_count, summary.questions_with_retries
        ),
        format!(
            "Errors: {} across {} question(s)",
            summary.error_count, summary.questions_with_errors
        ),
        format!(
            "Token budget exceeded: {} question(s)",
            summary.token_budget_exceeded_question_count
        ),
        format!("Started: {}", summary.started_at),
        format!("Finished: {}", summary.finished_at),
        format!("Duration seconds: {}", summary.duration_seconds),
        String::new(),
        "By status:".to_string(),
    ];
    for (status, count) in &summary.status_counts {
        lines.push(format!("- {}: {}", status, count));
    }
    lines.extend([
        String::new(),
        format_usage_line("Total", &summary.usage.usage),
        format_provider_cost_line("Provider cost", &summary.provider_cost),
        String::new(),
        "By source:".to_string(),
    ]);
    for (source, usage) in &summary.usage.by_source {
        lines.push(format!("- {}", format_usage_line(source, usage)));
    }
    lines.push(String::new());
    lines.push("By model:".to_string());
    for (model, usage) in &summary.usage.by_model {
        lines.push(format!("- {}", format_usage_line(model, usage)));
    }
    fs::write(&path, lines.join("\n") + "\n").with_context(|| format!("write {}", path.display()))
}

fn format_usage_line(label: &str, usage: &lash::UsageTotals) -> String {
    format!(
        "{label}: input={} cached={} output={} reasoning={} total={} context_total={}",
        usage.input_tokens,
        usage.cached_input_tokens,
        usage.output_tokens,
        usage.reasoning_tokens,
        usage.total_tokens,
        usage.context_total_tokens
    )
}

fn format_provider_cost_line(label: &str, cost: &ProviderCostSummary) -> String {
    format!(
        "{label}: cost_credits={:.6} upstream_inference_cost_credits={:.6} cost_entries={} upstream_entries={}",
        cost.total_cost_credits,
        cost.total_upstream_inference_cost_credits,
        cost.cost_entry_count,
        cost.upstream_inference_cost_entry_count
    )
}

fn aggregate_provider_cost<'a>(
    summaries: impl IntoIterator<Item = &'a ProviderCostSummary>,
) -> ProviderCostSummary {
    let mut total = ProviderCostSummary::default();
    for summary in summaries {
        total.total_cost_credits += summary.total_cost_credits;
        total.total_upstream_inference_cost_credits +=
            summary.total_upstream_inference_cost_credits;
        total.cost_entry_count += summary.cost_entry_count;
        total.upstream_inference_cost_entry_count += summary.upstream_inference_cost_entry_count;
    }
    total
}

fn collect_trace_metrics(path: &Path) -> anyhow::Result<TraceMetrics> {
    if !path.exists() {
        return Ok(TraceMetrics::default());
    }
    wait_for_stable_file(path, std::time::Duration::from_secs(10));
    let mut last_err = None;
    for _ in 0..3 {
        match collect_trace_metrics_once(path) {
            Ok(metrics) => return Ok(metrics),
            Err(err) => {
                last_err = Some(err);
                wait_for_stable_file(path, std::time::Duration::from_secs(2));
            }
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("trace metrics unavailable")))
}

fn collect_trace_metrics_once(path: &Path) -> anyhow::Result<TraceMetrics> {
    let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut turns = BTreeSet::new();
    let mut metrics = TraceMetrics::default();
    for line in raw.lines().filter(|line| !line.trim().is_empty()) {
        let value: Value = serde_json::from_str(line)
            .with_context(|| format!("parse trace row from {}", path.display()))?;
        if value.get("type").and_then(Value::as_str) != Some("llm_call_completed") {
            continue;
        }
        metrics.llm_calls += 1;
        if let Some(turn) = value
            .get("context")
            .and_then(|context| context.get("iteration"))
            .and_then(Value::as_u64)
        {
            turns.insert(turn);
        }
        if let Some(provider_usage) = value.get("provider_usage") {
            if let Some(cost) = provider_usage.get("cost").and_then(Value::as_f64) {
                metrics.provider_cost.total_cost_credits += cost;
                metrics.provider_cost.cost_entry_count += 1;
            }
            if let Some(cost) = provider_usage
                .get("cost_details")
                .and_then(|details| details.get("upstream_inference_cost"))
                .and_then(Value::as_f64)
            {
                metrics.provider_cost.total_upstream_inference_cost_credits += cost;
                metrics.provider_cost.upstream_inference_cost_entry_count += 1;
            }
        }
    }
    metrics.iterations = turns.len();
    Ok(metrics)
}

fn wait_for_stable_file(path: &Path, timeout: std::time::Duration) {
    let start = std::time::Instant::now();
    let mut stable_polls = 0usize;
    let mut last_len = None;
    while start.elapsed() < timeout {
        let len = fs::metadata(path).ok().map(|meta| meta.len());
        if len.is_some() && len == last_len {
            stable_polls += 1;
            if stable_polls >= 3 {
                return;
            }
        } else {
            stable_polls = 0;
            last_len = len;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

struct JsonlEventSink {
    file: Mutex<File>,
    last_llm_response: Mutex<Option<String>>,
    llm_call_count: Mutex<usize>,
    llm_iterations: Mutex<BTreeSet<usize>>,
    retry_count: Mutex<usize>,
    error_records: Mutex<Vec<SinkErrorRecord>>,
    token_budget: Mutex<LiveTokenBudget>,
    cancel: tokio_util::sync::CancellationToken,
}

impl JsonlEventSink {
    fn new(
        path: PathBuf,
        max_question_context_tokens: i64,
        cancel: tokio_util::sync::CancellationToken,
    ) -> anyhow::Result<Self> {
        let file = File::create(&path).with_context(|| format!("create {}", path.display()))?;
        Ok(Self {
            file: Mutex::new(file),
            last_llm_response: Mutex::new(None),
            llm_call_count: Mutex::new(0),
            llm_iterations: Mutex::new(BTreeSet::new()),
            retry_count: Mutex::new(0),
            error_records: Mutex::new(Vec::new()),
            token_budget: Mutex::new(LiveTokenBudget::new(max_question_context_tokens)),
            cancel,
        })
    }

    fn last_llm_response(&self) -> Option<String> {
        self.last_llm_response
            .lock()
            .ok()
            .and_then(|value| value.clone())
    }

    fn llm_call_count(&self) -> usize {
        self.llm_call_count
            .lock()
            .map(|value| *value)
            .unwrap_or_default()
    }

    fn iteration_count(&self) -> usize {
        self.llm_iterations
            .lock()
            .map(|turns| turns.len())
            .unwrap_or_default()
    }

    fn retry_count(&self) -> usize {
        self.retry_count
            .lock()
            .map(|value| *value)
            .unwrap_or_default()
    }

    fn error_records(&self) -> Vec<SinkErrorRecord> {
        self.error_records
            .lock()
            .map(|value| value.clone())
            .unwrap_or_default()
    }

    fn observed_context_tokens(&self) -> i64 {
        self.token_budget
            .lock()
            .map(|value| value.observed_context_tokens)
            .unwrap_or_default()
    }

    fn max_question_context_tokens(&self) -> Option<i64> {
        self.token_budget
            .lock()
            .ok()
            .and_then(|value| value.max_context_tokens)
    }

    fn token_budget(&self) -> Option<TokenBudgetExceeded> {
        self.token_budget.lock().ok().and_then(|value| {
            value.exceeded.then_some(TokenBudgetExceeded {
                observed_context_tokens: value.observed_context_tokens,
                max_context_tokens: value.max_context_tokens.unwrap_or_default(),
            })
        })
    }
}

#[async_trait::async_trait]
impl EventSink for JsonlEventSink {
    async fn emit(&self, event: SessionEvent) {
        if let SessionEvent::LlmRequest { iteration, .. } = &event {
            if let Ok(mut count) = self.llm_call_count.lock() {
                *count += 1;
            }
            if let Ok(mut iterations) = self.llm_iterations.lock() {
                iterations.insert(*iteration);
            }
        }
        if let SessionEvent::LlmResponse { content, .. } = &event
            && let Ok(mut last) = self.last_llm_response.lock()
        {
            *last = Some(content.trim().to_string());
        }
        if matches!(event, SessionEvent::RetryStatus { .. })
            && let Ok(mut count) = self.retry_count.lock()
        {
            *count += 1;
        }
        if let SessionEvent::Error { message, envelope } = &event
            && let Ok(mut errors) = self.error_records.lock()
        {
            errors.push(SinkErrorRecord {
                message: message.clone(),
                kind: envelope.as_ref().map(|value| value.kind.clone()),
                code: envelope.as_ref().and_then(|value| value.code.clone()),
                raw: envelope.as_ref().and_then(|value| value.raw.clone()),
            });
        }
        if let SessionEvent::TokenUsage { usage, .. } | SessionEvent::ChildTokenUsage { usage, .. } =
            &event
            && let Ok(mut budget) = self.token_budget.lock()
            && budget.record(usage)
        {
            self.cancel.cancel();
        }
        if let Ok(line) = serde_json::to_string(&event)
            && let Ok(mut file) = self.file.lock()
        {
            let _ = writeln!(file, "{line}");
        }
    }
}

fn turn_completed(outcome: &lash::TurnOutcome) -> bool {
    matches!(
        outcome,
        lash::TurnOutcome::Finished(_) | lash::TurnOutcome::Handoff { .. }
    )
}

fn turn_status_label(outcome: &lash::TurnOutcome) -> &'static str {
    match outcome {
        lash::TurnOutcome::Finished(_) | lash::TurnOutcome::Handoff { .. } => "completed",
        lash::TurnOutcome::Stopped(lash::TurnStop::Cancelled) => "interrupted",
        lash::TurnOutcome::Stopped(_) => "failed",
    }
}

fn done_reason_label(outcome: &lash::TurnOutcome) -> &'static str {
    match outcome {
        lash::TurnOutcome::Finished(lash::TurnFinish::AssistantMessage { .. }) => {
            "assistant_message"
        }
        lash::TurnOutcome::Finished(lash::TurnFinish::Submission { .. }) => "submission",
        lash::TurnOutcome::Handoff { .. } => "handoff",
        lash::TurnOutcome::Stopped(lash::TurnStop::Cancelled) => "cancelled",
        lash::TurnOutcome::Stopped(lash::TurnStop::InvalidInput) => "invalid_input",
        lash::TurnOutcome::Stopped(lash::TurnStop::MaxTurns) => "max_turns",
        lash::TurnOutcome::Stopped(lash::TurnStop::ToolFailure) => "tool_failure",
        lash::TurnOutcome::Stopped(lash::TurnStop::ProviderError) => "provider_error",
        lash::TurnOutcome::Stopped(lash::TurnStop::PluginAbort) => "plugin_abort",
        lash::TurnOutcome::Stopped(lash::TurnStop::RuntimeError) => "runtime_error",
        lash::TurnOutcome::Stopped(lash::TurnStop::SubmittedError { .. }) => "submitted_error",
    }
}

fn context_tokens_for_usage(usage: &lash::TokenUsage) -> i64 {
    usage
        .input_tokens
        .max(0)
        .saturating_add(usage.output_tokens.max(0))
        .saturating_add(usage.reasoning_tokens.max(0))
        .saturating_add(usage.cached_input_tokens.max(0))
}

fn non_empty_text(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn format_failure_reason(
    turn: &lash::AssembledTurn,
    error_records: &[SinkErrorRecord],
) -> Option<String> {
    if turn_completed(&turn.outcome) {
        return None;
    }
    if let Some(error) = error_records.last() {
        let mut reason = error.message.clone();
        if let Some(kind) = &error.kind {
            reason = format!("{kind}: {reason}");
        }
        if let Some(code) = &error.code {
            reason.push_str(&format!(" [code={code}]"));
        }
        if let Some(raw) = &error.raw {
            reason.push_str(&format!(" raw={raw}"));
        }
        return Some(reason);
    }
    turn.errors
        .first()
        .map(|error| error.message.clone())
        .or_else(|| {
            Some(format!(
                "turn ended with status={} reason={}",
                turn_status_label(&turn.outcome),
                done_reason_label(&turn.outcome)
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_token_budget_counts_context_tokens() {
        let mut budget = LiveTokenBudget::new(100);
        assert!(!budget.record(&lash::TokenUsage {
            input_tokens: 40,
            output_tokens: 5,
            cached_input_tokens: 10,
            reasoning_tokens: 0,
        }));
        assert_eq!(budget.observed_context_tokens, 55);
        assert!(!budget.exceeded);
    }

    #[test]
    fn live_token_budget_trips_after_combined_root_and_child_usage() {
        let mut budget = LiveTokenBudget::new(100);
        assert!(!budget.record(&lash::TokenUsage {
            input_tokens: 45,
            output_tokens: 5,
            cached_input_tokens: 0,
            reasoning_tokens: 0,
        }));
        assert!(budget.record(&lash::TokenUsage {
            input_tokens: 40,
            output_tokens: 0,
            cached_input_tokens: 20,
            reasoning_tokens: 0,
        }));
        assert_eq!(budget.observed_context_tokens, 110);
        assert!(budget.exceeded);
    }
}
