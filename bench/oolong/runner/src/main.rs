mod dataset;

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, bail};
use async_trait::async_trait;
use chrono::Utc;
use clap::{Parser, ValueEnum};
use dataset::{OolongQuestion, OolongSuite, default_dataset_path, load_questions};
use lash::{
    LashCore, ModeId, ModePreset, SessionSpec, TurnInput,
    advanced::{
        EventSink, ExecutionMode, ModeTurnOptions, TurnContext, TurnFinish, TurnOutcome, TurnStop,
    },
    plugins::{PluginFactory, PluginHost, PluginSpec, StaticPluginFactory},
    prompt::{
        PromptBuiltin, PromptSlot, PromptTemplate, PromptTemplateEntry, PromptTemplateSection,
    },
    provider::{ProviderHandle, ProviderOptions},
    tools::{ToolCall, ToolDefinition, ToolExecutionMode, ToolProvider, ToolResult},
    usage::{SessionUsageReport, TokenLedgerEntry, TokenUsage, diff_usage_reports},
};
use lash_cli::config::LashConfig;
use lash_core::{InputItem, SessionEvent, ToolOutputBudgetPluginFactory};
use lash_export::{ExportFormat, export};
use lash_llm_tools::LlmToolsPluginFactory;
use lash_mode_rlm::{
    BuiltinRlmModePluginFactory, RlmModePluginConfig, RlmPromptFeatures, RlmTurnInputExt,
};
use lash_provider_openai::OPENROUTER_BASE_URL;
use lash_rlm_types::RlmTermination;
use lash_sqlite_store::Store;
use lash_subagents::{
    CapabilityRegistry, LocalSubagentHost, StaticCapability, SubagentHost, SubagentsPluginFactory,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

const STATE_ROOT: &str = ".benchmarks/oolong";
const EXECUTION_MODE_LABEL: &str = "rlm";
const SUBAGENT_CAPABILITY: &str = "default";
const DEFAULT_MODEL: &str = "openai/gpt-5";
const DEFAULT_VARIANT: &str = "medium";
const DEFAULT_MAX_TURNS: usize = 50;
const DEFAULT_MAX_CONTEXT_TOKENS: usize = 1_000_000;
const DEFAULT_MAX_OUTPUT_TOKENS: u64 = 125_000;
const DEFAULT_BATCH_SIZE: usize = 4;

const OOLONG_USER_DIRECTIVE: &str = concat!(
    "Answer the OOLONG aggregation question bound as `input.question` using the long context bound as `input.context`. ",
    "Do not guess or approximate. Inspect the context in slices, classify/count exactly, and use recursive subagents when useful. ",
    "End by submitting only the final answer value with `submit` from a fenced `lashlang` block."
);

#[derive(Clone, Copy, Debug, ValueEnum)]
enum SuiteArg {
    #[value(name = "synth")]
    Synth,
    #[value(name = "synth-with-labels")]
    SynthWithLabels,
    #[value(name = "real")]
    Real,
}

impl From<SuiteArg> for OolongSuite {
    fn from(value: SuiteArg) -> Self {
        match value {
            SuiteArg::Synth => OolongSuite::Synth,
            SuiteArg::SynthWithLabels => OolongSuite::SynthWithLabels,
            SuiteArg::Real => OolongSuite::Real,
        }
    }
}

#[derive(Parser, Debug, Clone)]
#[command(name = "bench-oolong")]
#[command(about = "Run OOLONG through Lash as an RLM-style aggregation benchmark.")]
struct Args {
    #[arg(long, value_enum, default_value_t = SuiteArg::Synth)]
    suite: SuiteArg,

    #[arg(long)]
    dataset: Option<PathBuf>,

    #[arg(long)]
    run_id: Option<String>,

    #[arg(long)]
    output_dir: Option<PathBuf>,

    #[arg(long, default_value = DEFAULT_MODEL)]
    model: String,

    #[arg(long)]
    child_model: Option<String>,

    #[arg(long, default_value = "openai-compatible")]
    provider_id: String,

    #[arg(long, default_value = DEFAULT_VARIANT)]
    variant: String,

    #[arg(long)]
    child_variant: Option<String>,

    #[arg(long)]
    api_key: Option<String>,

    #[arg(long)]
    base_url: Option<String>,

    #[arg(long, default_value_t = DEFAULT_MAX_TURNS)]
    max_turns: usize,

    #[arg(long)]
    child_max_turns: Option<usize>,

    #[arg(long, default_value_t = DEFAULT_MAX_CONTEXT_TOKENS)]
    max_context_tokens: usize,

    #[arg(long, default_value_t = DEFAULT_MAX_OUTPUT_TOKENS)]
    max_output_tokens: u64,

    #[arg(long)]
    soft_continue_as_tokens: Option<usize>,

    #[arg(long)]
    forced_continue_as_tokens: Option<usize>,

    #[arg(long)]
    disable_continue_as_fallback: bool,

    #[arg(long)]
    question_id: Vec<String>,

    #[arg(long)]
    dataset_name: Option<String>,

    #[arg(long)]
    context_len: Option<u64>,

    #[arg(long)]
    task_group: Option<String>,

    #[arg(long)]
    task: Option<String>,

    #[arg(long, default_value_t = 0)]
    offset: usize,

    #[arg(long)]
    max_questions: Option<usize>,

    #[arg(long)]
    shuffle_seed: Option<u64>,

    #[arg(long)]
    resume: bool,

    #[arg(long)]
    await_background_work: bool,

    #[arg(long, default_value_t = DEFAULT_BATCH_SIZE)]
    batch_size: usize,

    #[arg(long)]
    dry_run: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RunManifest {
    run_id: String,
    created_at: String,
    suite: String,
    dataset_path: String,
    model: String,
    child_model: Option<String>,
    provider_id: String,
    variant: Option<String>,
    child_variant: Option<String>,
    base_url: String,
    execution_mode: String,
    max_turns: usize,
    child_max_turns: Option<usize>,
    max_context_tokens: usize,
    max_output_tokens: u64,
    batch_size: usize,
    selection: SelectionSnapshot,
    selected_count: usize,
    predictions_path: String,
    reference: ReferenceSettings,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SelectionSnapshot {
    question_ids: Vec<String>,
    dataset_name: Option<String>,
    context_len: Option<u64>,
    task_group: Option<String>,
    task: Option<String>,
    offset: usize,
    max_questions: Option<usize>,
    shuffle_seed: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ReferenceSettings {
    paper: String,
    oolong_paper: String,
    hf_synth: String,
    hf_real: String,
    note: String,
}

impl Default for ReferenceSettings {
    fn default() -> Self {
        Self {
            paper: "https://arxiv.org/abs/2512.24601".to_string(),
            oolong_paper: "https://arxiv.org/abs/2511.02817".to_string(),
            hf_synth: "https://huggingface.co/datasets/oolongbench/oolong-synth".to_string(),
            hf_real: "https://huggingface.co/datasets/oolongbench/oolong-real".to_string(),
            note: "RLM Table 1 uses the OOLONG trec_coarse split with 50 tasks at the 131K-token setting. Prepare that slice with `bench/oolong/setup.sh --suite synth --dataset trec_coarse --context-len 131072 --limit 50`.".to_string(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct QuestionResult {
    question_id: String,
    suite: String,
    dataset: Option<String>,
    context_len: Option<u64>,
    task_group: Option<String>,
    task: Option<String>,
    answer_type: Option<String>,
    prediction: Value,
    prediction_text: String,
    answer: Value,
    correct: bool,
    model: String,
    usage: SessionUsageReport,
    elapsed_seconds: f64,
    iterations: usize,
    metrics: QuestionMetrics,
    status: String,
    done_reason: String,
    failure_reason: Option<String>,
    artifacts: QuestionArtifacts,
    lash: LashRunSnapshot,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct QuestionArtifacts {
    question_json: String,
    result_json: String,
    answer_txt: String,
    prompt_txt: String,
    system_prompt_txt: String,
    events_jsonl: String,
    session_db: String,
    trace_jsonl: String,
    trace_html: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct LashRunSnapshot {
    execution_mode: String,
    variant: Option<String>,
    child_model: Option<String>,
    child_variant: Option<String>,
    max_turns: usize,
    child_max_turns: Option<usize>,
    max_output_tokens: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RunSummary {
    run_id: String,
    started_at: String,
    finished_at: String,
    duration_seconds: i64,
    question_count: usize,
    result_count: usize,
    correct: usize,
    accuracy: f64,
    failed: usize,
    by_task_group: BTreeMap<String, Bucket>,
    by_dataset: BTreeMap<String, Bucket>,
    iterations: usize,
    wall_clock_seconds: f64,
    timing: TimingSummary,
    metrics: RunMetrics,
    usage: SessionUsageReport,
    predictions_path: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Bucket {
    count: usize,
    correct: usize,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct QuestionMetrics {
    wall_clock_seconds: f64,
    root_llm_calls: usize,
    child_llm_calls: usize,
    token_usage_events: usize,
    child_usage_events: usize,
    trace_llm_calls: usize,
    trace_llm_failures: usize,
    trace_llm_duration_ms: u64,
    trace_llm_calls_by_session: BTreeMap<String, usize>,
    tool_calls_by_name: BTreeMap<String, usize>,
    tool_call_duration_ms_by_name: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct RunMetrics {
    root_llm_calls: usize,
    child_llm_calls: usize,
    token_usage_events: usize,
    child_usage_events: usize,
    trace_llm_calls: usize,
    trace_llm_failures: usize,
    trace_llm_duration_ms: u64,
    trace_llm_calls_by_session: BTreeMap<String, usize>,
    tool_calls_by_name: BTreeMap<String, usize>,
    tool_call_duration_ms_by_name: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct TimingSummary {
    total_question_seconds: f64,
    mean_question_seconds: f64,
    min_question_seconds: f64,
    max_question_seconds: f64,
    p50_question_seconds: f64,
    p95_question_seconds: f64,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _ = dotenvy::dotenv();
    lash_providers_builtin::register_all();

    let args = Args::parse();
    let suite: OolongSuite = args.suite.into();
    let state_root = PathBuf::from(STATE_ROOT);
    let dataset_path = args
        .dataset
        .clone()
        .unwrap_or_else(|| default_dataset_path(&state_root, suite));
    if !dataset_path.exists() {
        bail!(
            "OOLONG dataset slice not found at {} — run bench/oolong/setup.sh first",
            dataset_path.display()
        );
    }
    let runs_dir = state_root.join("runs");
    fs::create_dir_all(&runs_dir).with_context(|| format!("create {}", runs_dir.display()))?;

    let questions = select_questions(load_questions(&dataset_path)?, &args);
    if questions.is_empty() {
        bail!("no OOLONG questions selected");
    }

    let run_id = args
        .run_id
        .clone()
        .unwrap_or_else(|| Utc::now().format("%Y%m%dT%H%M%SZ").to_string());
    let output_dir = args
        .output_dir
        .clone()
        .unwrap_or_else(|| runs_dir.join(&run_id));
    fs::create_dir_all(&output_dir).with_context(|| format!("create {}", output_dir.display()))?;
    let predictions_path = output_dir.join("predictions.jsonl");

    let manifest = RunManifest {
        run_id: run_id.clone(),
        created_at: Utc::now().to_rfc3339(),
        suite: suite.label().to_string(),
        dataset_path: dataset_path.display().to_string(),
        model: args.model.clone(),
        child_model: args.child_model.clone(),
        provider_id: args.provider_id.clone(),
        variant: Some(args.variant.clone()),
        child_variant: args.child_variant.clone(),
        base_url: resolve_base_url(&args),
        execution_mode: EXECUTION_MODE_LABEL.to_string(),
        max_turns: args.max_turns,
        child_max_turns: args.child_max_turns,
        max_context_tokens: args.max_context_tokens,
        max_output_tokens: args.max_output_tokens,
        batch_size: args.batch_size.max(1),
        selection: SelectionSnapshot {
            question_ids: args.question_id.clone(),
            dataset_name: args.dataset_name.clone(),
            context_len: args.context_len,
            task_group: args.task_group.clone(),
            task: args.task.clone(),
            offset: args.offset,
            max_questions: args.max_questions,
            shuffle_seed: args.shuffle_seed,
        },
        selected_count: questions.len(),
        predictions_path: predictions_path.display().to_string(),
        reference: ReferenceSettings::default(),
    };
    write_json(&output_dir.join("manifest.json"), &manifest)?;

    if args.dry_run {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "manifest": manifest,
                "questions": questions.iter().map(question_preview).collect::<Vec<_>>(),
            }))?
        );
        return Ok(());
    }

    let provider = resolve_provider(&args)?;
    let completed = if args.resume {
        load_completed_ids(&predictions_path)?
    } else {
        BTreeSet::new()
    };
    let pending = questions
        .iter()
        .filter(|q| !completed.contains(&q.question_id))
        .cloned()
        .collect::<Vec<_>>();

    eprintln!("OOLONG run_id={run_id}");
    eprintln!("  suite:            {}", manifest.suite);
    eprintln!("  selected:         {}", questions.len());
    eprintln!("  pending:          {}", pending.len());
    eprintln!("  model:            {}", args.model);
    eprintln!(
        "  child_model:      {}",
        args.child_model.as_deref().unwrap_or("<inherit>")
    );
    eprintln!("  execution-mode:   {}", manifest.execution_mode);
    eprintln!("  max_turns:        {}", args.max_turns);
    eprintln!("  batch_size:       {}", args.batch_size.max(1));
    eprintln!("  predictions:      {}", predictions_path.display());
    if !completed.is_empty() {
        eprintln!("  resuming:         skipping {} ids", completed.len());
    }

    if pending.is_empty() {
        eprintln!("nothing to run — predictions already cover every selected question");
        return Ok(());
    }

    let started_at = Utc::now();
    let started_instant = std::time::Instant::now();
    let semaphore = Arc::new(Semaphore::new(args.batch_size.max(1)));
    let provider = Arc::new(provider);
    let args_shared = Arc::new(args.clone());
    let output_dir_shared = Arc::new(output_dir.clone());
    let predictions_path_shared = Arc::new(predictions_path.clone());
    let total = pending.len();
    let mut join_set = JoinSet::new();
    for (index, question) in pending.into_iter().enumerate() {
        let permit = semaphore
            .clone()
            .acquire_owned()
            .await
            .context("acquire benchmark slot")?;
        let provider = provider.clone();
        let args = args_shared.clone();
        let output_dir = output_dir_shared.clone();
        let predictions_path = predictions_path_shared.clone();
        join_set.spawn(async move {
            let _permit = permit;
            let result = run_question(
                output_dir.as_ref(),
                provider.as_ref(),
                args.as_ref(),
                question,
            )
            .await;
            if let Ok(row) = &result {
                let _ = append_response_row(predictions_path.as_ref(), row);
            }
            (index, result)
        });
    }

    let mut indexed_results = Vec::<(usize, QuestionResult)>::new();
    let mut completed_count = 0usize;
    while let Some(joined) = join_set.join_next().await {
        let (index, result) = joined.context("benchmark task panicked")?;
        let result = match result {
            Ok(value) => value,
            Err(err) => {
                join_set.abort_all();
                return Err(err);
            }
        };
        completed_count += 1;
        eprintln!(
            "  [{}/{}] {} dataset={} task={} correct={} status={} t={:.1}s iters={}",
            completed_count,
            total,
            result.question_id,
            result.dataset.as_deref().unwrap_or("-"),
            result.task_group.as_deref().unwrap_or("-"),
            if result.correct { "y" } else { "n" },
            result.status,
            result.elapsed_seconds,
            result.iterations,
        );
        indexed_results.push((index, result));
    }
    indexed_results.sort_by_key(|(idx, _)| *idx);
    let results = indexed_results
        .into_iter()
        .map(|(_, result)| result)
        .collect::<Vec<_>>();

    let finished_at = Utc::now();
    let correct = results.iter().filter(|r| r.correct).count();
    let summary = RunSummary {
        run_id: run_id.clone(),
        started_at: started_at.to_rfc3339(),
        finished_at: finished_at.to_rfc3339(),
        duration_seconds: (finished_at - started_at).num_seconds(),
        question_count: questions.len(),
        result_count: results.len(),
        correct,
        accuracy: if results.is_empty() {
            0.0
        } else {
            correct as f64 / results.len() as f64
        },
        failed: results.iter().filter(|r| !r.status.eq("completed")).count(),
        by_task_group: aggregate_by(results.iter().map(|r| {
            (
                r.task_group
                    .clone()
                    .unwrap_or_else(|| "unknown".to_string()),
                r.correct,
            )
        })),
        by_dataset: aggregate_by(results.iter().map(|r| {
            (
                r.dataset.clone().unwrap_or_else(|| "unknown".to_string()),
                r.correct,
            )
        })),
        iterations: results.iter().map(|r| r.iterations).sum(),
        wall_clock_seconds: started_instant.elapsed().as_secs_f64(),
        timing: summarize_timing(&results),
        metrics: aggregate_metrics(results.iter().map(|r| &r.metrics)),
        usage: aggregate_usage(results.iter().map(|r| r.usage.clone())),
        predictions_path: predictions_path.display().to_string(),
    };
    write_json(&output_dir.join("results.json"), &summary)?;
    write_trace_index(&output_dir, &run_id, &results)?;

    eprintln!();
    eprintln!("Run summary:");
    eprintln!("  run_dir:        {}", output_dir.display());
    eprintln!("  predictions:    {}", predictions_path.display());
    eprintln!(
        "  correct:        {}/{}",
        summary.correct, summary.result_count
    );
    eprintln!("  accuracy:       {:.3}", summary.accuracy);
    eprintln!("  iterations:     {}", summary.iterations);
    eprintln!("  wall_clock:     {:.1}s", summary.wall_clock_seconds);
    eprintln!("  root_llm_calls: {}", summary.metrics.root_llm_calls);
    eprintln!("  child_llm_calls: {}", summary.metrics.child_llm_calls);
    eprintln!();
    eprintln!("Evaluate with:");
    eprintln!("  bench/oolong/evaluate.sh {}", output_dir.display());
    Ok(())
}

fn question_preview(q: &OolongQuestion) -> Value {
    json!({
        "question_id": q.question_id,
        "suite": q.suite.label(),
        "dataset": q.dataset,
        "context_len": q.context_len,
        "task_group": q.task_group,
        "task": q.task,
        "prompt_chars": q.prompt.chars().count(),
    })
}

fn select_questions(mut questions: Vec<OolongQuestion>, args: &Args) -> Vec<OolongQuestion> {
    if !args.question_id.is_empty() {
        let wanted: BTreeSet<&str> = args.question_id.iter().map(String::as_str).collect();
        questions.retain(|q| wanted.contains(q.question_id.as_str()));
    }
    if let Some(dataset_name) = args.dataset_name.as_deref() {
        questions.retain(|q| q.dataset.as_deref() == Some(dataset_name));
    }
    if let Some(context_len) = args.context_len {
        questions.retain(|q| q.context_len == Some(context_len));
    }
    if let Some(task_group) = args.task_group.as_deref() {
        questions.retain(|q| q.task_group.as_deref() == Some(task_group));
    }
    if let Some(task) = args.task.as_deref() {
        questions.retain(|q| q.task.as_deref() == Some(task));
    }
    if let Some(seed) = args.shuffle_seed {
        simple_shuffle(&mut questions, seed);
    }
    if args.offset > 0 {
        questions = questions.into_iter().skip(args.offset).collect();
    }
    if let Some(limit) = args.max_questions {
        questions.truncate(limit);
    }
    questions
}

fn simple_shuffle<T>(items: &mut [T], seed: u64) {
    let mut state = seed;
    for i in (1..items.len()).rev() {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        items.swap(i, (z as usize) % (i + 1));
    }
}

async fn run_question(
    output_dir: &Path,
    provider: &ProviderHandle,
    args: &Args,
    question: OolongQuestion,
) -> anyhow::Result<QuestionResult> {
    let question_dir = output_dir.join("questions").join(&question.question_id);
    fs::create_dir_all(&question_dir)
        .with_context(|| format!("create {}", question_dir.display()))?;
    write_json(&question_dir.join("question.json"), &question)?;
    fs::write(question_dir.join("prompt.txt"), &question.prompt)
        .with_context(|| format!("write {}", question_dir.join("prompt.txt").display()))?;

    let store_path = question_dir.join("session.db");
    let trace_path = question_dir.join("session.trace.jsonl");
    let store = Arc::new(
        Store::open(&store_path).with_context(|| format!("open {}", store_path.display()))?,
    );
    let execution_mode = ExecutionMode::new(EXECUTION_MODE_LABEL);
    let core = LashCore::builder()
        .install_mode(ModePreset::rlm())
        .default_mode(ModeId::rlm())
        .provider(provider.clone())
        .model(args.model.clone(), Some(args.variant.clone()))
        .max_context_tokens(args.max_context_tokens)
        .max_turns(args.max_turns)
        .prompt_template(oolong_prompt_template())
        .trace_jsonl_path(Some(trace_path.clone()))
        .advanced()
        .plugin_host(build_plugin_host(execution_mode.clone(), args))
        .build()?;
    let session = core
        .session("root")
        .rlm()
        .store(store.clone())
        .open()
        .await?;

    let before_usage = session.usage_report();
    let started = std::time::Instant::now();
    let cancel = tokio_util::sync::CancellationToken::new();
    let sink = Arc::new(OolongEventSink::new(question_dir.join("events.jsonl"))?);
    let sink_trait: Arc<dyn EventSink> = sink.clone();
    let input = TurnInput {
        items: vec![InputItem::Text {
            text: OOLONG_USER_DIRECTIVE.to_string(),
        }],
        image_blobs: Default::default(),
        mode_turn_options: Some(ModeTurnOptions::typed(
            execution_mode,
            RlmTermination::SubmitRequired {
                schema: Some(oolong_answer_schema(&question)),
            },
        )?),
        trace_turn_id: None,
        mode_extension: None,
        turn_context: TurnContext::default(),
    }
    .rlm_project(build_projected_bindings(&question)?)?;
    let turn_result = session
        .turn(input)
        .cancel(cancel)
        .collect_session_events_with(sink_trait.as_ref())
        .await;
    let background_result = if args.await_background_work && turn_result.is_ok() {
        session.background_tasks().await_all().await
    } else {
        Ok(())
    };
    let cancelled = session.background_tasks().cancel_all().await?;
    if !cancelled.is_empty() {
        eprintln!(
            "  cancelled {} background task(s) after {}",
            cancelled.len(),
            question.question_id
        );
    }
    background_result?;
    let turn = turn_result.context("run OOLONG question")?;
    let elapsed_seconds = started.elapsed().as_secs_f64();
    let trace_metrics = read_trace_metrics(&trace_path).unwrap_or_default();
    let metrics = sink.metrics(elapsed_seconds, trace_metrics);
    let usage = diff_usage_reports(&before_usage, &session.usage_report())
        .map(|rows| SessionUsageReport::from_entries(&rows))
        .map_err(anyhow::Error::msg)
        .context("diff usage reports")?;

    let prediction = prediction_from_turn(&turn.outcome, &turn.assistant_output.safe_text);
    let prediction_text = answer_text(&prediction);
    let correct = answer_matches(&prediction, &question.answer);
    let status = turn_status_label(&turn.outcome).to_string();
    let done_reason = done_reason_label(&turn.outcome).to_string();
    let failure_reason = if turn_completed(&turn.outcome) {
        None
    } else {
        turn.errors
            .first()
            .map(|e| e.message.clone())
            .or_else(|| sink.last_error())
            .or_else(|| Some(format!("status={status} reason={done_reason}")))
    };

    fs::write(
        question_dir.join("answer.txt"),
        format!("{prediction_text}\n"),
    )
    .with_context(|| format!("write {}", question_dir.join("answer.txt").display()))?;

    let result = QuestionResult {
        question_id: question.question_id.clone(),
        suite: question.suite.label().to_string(),
        dataset: question.dataset.clone(),
        context_len: question.context_len,
        task_group: question.task_group.clone(),
        task: question.task.clone(),
        answer_type: question.answer_type.clone(),
        prediction,
        prediction_text,
        answer: question.answer.clone(),
        correct,
        model: args.model.clone(),
        usage,
        elapsed_seconds,
        iterations: sink.iteration_count(),
        metrics,
        status,
        done_reason,
        failure_reason,
        artifacts: question_artifacts(&question_dir),
        lash: LashRunSnapshot {
            execution_mode: EXECUTION_MODE_LABEL.to_string(),
            variant: Some(args.variant.clone()),
            child_model: args.child_model.clone(),
            child_variant: args.child_variant.clone(),
            max_turns: args.max_turns,
            child_max_turns: args.child_max_turns,
            max_output_tokens: args.max_output_tokens,
        },
    };
    write_json(&question_dir.join("result.json"), &result)?;

    let html_trace_path = question_dir.join("trace.html");
    if let Err(err) = export(
        &store_path,
        &trace_path,
        ExportFormat::Html,
        Some(&html_trace_path),
    ) {
        eprintln!(
            "warn: failed to render HTML trace for {}: {err:#}",
            question.question_id
        );
    }
    if let Err(err) = write_system_prompt_snapshot(&trace_path, &question_dir) {
        eprintln!(
            "warn: failed to snapshot system prompt for {}: {err:#}",
            question.question_id
        );
    }

    Ok(result)
}

fn oolong_answer_schema(question: &OolongQuestion) -> Value {
    let answer_type = question
        .answer_type
        .as_deref()
        .unwrap_or_default()
        .to_ascii_uppercase();
    if answer_type.contains("NUMERIC") {
        return json!({ "type": "integer" });
    }
    if answer_type.contains("COMPARISON") {
        return json!({
            "type": "string",
            "enum": ["more common than", "less common than", "same frequency as"]
        });
    }
    if answer_type.contains("LABEL") {
        let choices = label_choices_from_question(&question.question);
        if !choices.is_empty() {
            return scalar_or_array_schema(json!({ "type": "string", "enum": choices }));
        }
        return scalar_or_array_schema(json!({ "type": "string" }));
    }
    if answer_type.contains("USER") {
        return scalar_or_array_schema(json!({
            "anyOf": [
                { "type": "integer" },
                { "type": "string", "pattern": "^[A-Za-z0-9_-]+$" }
            ]
        }));
    }
    if answer_type.contains("DATE") {
        return scalar_or_array_schema(json!({
            "type": "string",
            "pattern": "^\\d{2}/\\d{2}/\\d{4}$"
        }));
    }
    if answer_type.contains("MONTH_YEAR") {
        return scalar_or_array_schema(json!({
            "type": "string",
            "pattern": "^(January|February|March|April|May|June|July|August|September|October|November|December) \\d{4}$"
        }));
    }
    json!({
        "anyOf": [
            { "type": "string" },
            { "type": "number" },
            { "type": "integer" },
            { "type": "boolean" },
            { "type": "array", "items": { "type": ["string", "number", "integer", "boolean"] } },
            { "type": "object" }
        ]
    })
}

fn scalar_or_array_schema(item_schema: Value) -> Value {
    json!({
        "anyOf": [
            item_schema.clone(),
            {
                "type": "array",
                "items": item_schema,
                "minItems": 1,
                "uniqueItems": true
            }
        ]
    })
}

fn label_choices_from_question(question: &str) -> Vec<String> {
    let Some((_, after_marker)) = question.split_once("one of the labels:") else {
        return Vec::new();
    };
    let choices_text = after_marker
        .split_once('.')
        .map(|(choices, _)| choices)
        .unwrap_or(after_marker);
    choices_text
        .split(',')
        .map(str::trim)
        .map(|choice| choice.trim_matches(|c: char| c == '\'' || c == '"' || c == '`'))
        .filter(|choice| !choice.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn prediction_from_turn(outcome: &TurnOutcome, assistant_text: &str) -> Value {
    if let TurnOutcome::Finished(TurnFinish::SubmittedValue { value }) = outcome {
        return value.clone();
    }
    parse_answerish_value(assistant_text)
        .unwrap_or_else(|| Value::String(assistant_text.trim().to_string()))
}

fn parse_answerish_value(text: &str) -> Option<Value> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
        return Some(value);
    }
    let after_prefix = trimmed
        .split_once(':')
        .map(|(_, rhs)| rhs.trim())
        .unwrap_or(trimmed);
    if let Ok(value) = serde_json::from_str::<Value>(after_prefix) {
        return Some(value);
    }
    Some(Value::String(after_prefix.to_string()))
}

fn answer_matches(prediction: &Value, answer: &Value) -> bool {
    let expected = normalized_answer_items(answer);
    let predicted = normalized_answer_items(prediction);
    if expected.is_empty() || predicted.is_empty() {
        return normalized_answer_scalar(prediction) == normalized_answer_scalar(answer);
    }
    expected == predicted
}

fn normalized_answer_items(value: &Value) -> Vec<String> {
    let mut items = match value {
        Value::Array(values) => values
            .iter()
            .map(normalized_answer_scalar)
            .filter(|v| !v.is_empty())
            .collect::<Vec<_>>(),
        _ => vec![normalized_answer_scalar(value)],
    };
    items.sort();
    items.dedup();
    items
}

fn normalized_answer_scalar(value: &Value) -> String {
    match value {
        Value::String(text) => normalize_answer_text(text),
        Value::Number(n) => normalize_answer_text(&n.to_string()),
        Value::Bool(b) => b.to_string(),
        Value::Array(_) => normalized_answer_items(value).join("|"),
        Value::Object(_) => normalize_answer_text(&value.to_string()),
        Value::Null => String::new(),
    }
}

fn normalize_answer_text(text: &str) -> String {
    let mut s = text.trim().to_string();
    if let Some((prefix, rest)) = s.split_once(':') {
        let p = prefix.trim().to_ascii_lowercase();
        if matches!(p.as_str(), "answer" | "label" | "user") {
            s = rest.trim().to_string();
        }
    }
    s.trim_matches(|c: char| c == '"' || c == '\'' || c == '[' || c == ']')
        .trim()
        .to_ascii_lowercase()
}

fn answer_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.trim().to_string(),
        other => other.to_string(),
    }
}

fn write_system_prompt_snapshot(trace_path: &Path, question_dir: &Path) -> anyhow::Result<()> {
    if !trace_path.exists() {
        return Ok(());
    }
    let raw =
        fs::read_to_string(trace_path).with_context(|| format!("read {}", trace_path.display()))?;
    for line in raw.lines().filter(|l| !l.trim().is_empty()) {
        let Ok(record) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if record.get("type").and_then(Value::as_str) != Some("llm_call_started") {
            continue;
        }
        let Some(request) = record.get("request") else {
            continue;
        };
        let request_value: Value = match request {
            Value::String(s) => serde_json::from_str(s).unwrap_or(Value::Null),
            v => v.clone(),
        };
        let Some(messages) = request_value.get("messages").and_then(Value::as_array) else {
            continue;
        };
        let system = messages
            .iter()
            .find(|m| m.get("role").and_then(Value::as_str) == Some("system"));
        if let Some(system) = system {
            let text = extract_text(system.get("blocks").or_else(|| system.get("content")));
            fs::write(question_dir.join("system_prompt.txt"), text).with_context(|| {
                format!("write {}", question_dir.join("system_prompt.txt").display())
            })?;
            return Ok(());
        }
    }
    Ok(())
}

fn extract_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn build_plugin_host(execution_mode: ExecutionMode, args: &Args) -> PluginHost {
    let child_spec = child_session_spec(args, execution_mode.clone());
    let llm_tools = match (&args.child_model, &args.child_variant) {
        (Some(model), variant) => {
            LlmToolsPluginFactory::default().with_model(model.clone(), variant.clone())
        }
        (None, Some(variant)) => {
            LlmToolsPluginFactory::default().with_model_variant(variant.clone())
        }
        (None, None) => LlmToolsPluginFactory::default(),
    };
    let factories: Vec<Arc<dyn PluginFactory>> = vec![
        Arc::new(ToolOutputBudgetPluginFactory::default()),
        Arc::new(BuiltinRlmModePluginFactory::new(rlm_config(args))),
        Arc::new(llm_tools),
        Arc::new(StaticPluginFactory::new(
            "oolong_async_handles",
            PluginSpec::new().with_tool_provider(Arc::new(OolongAsyncHandlesTool)),
        )),
        Arc::new(
            SubagentsPluginFactory::new(
                Arc::new(
                    CapabilityRegistry::new().with(Arc::new(StaticCapability::new(
                        SUBAGENT_CAPABILITY,
                        child_spec,
                    ))),
                ),
                Arc::new(LocalSubagentHost::default()) as Arc<dyn SubagentHost>,
            )
            .with_session_spec(SessionSpec::inherit()),
        ),
    ];
    PluginHost::new(factories)
}

fn child_session_spec(args: &Args, execution_mode: ExecutionMode) -> SessionSpec {
    let mut spec = SessionSpec::inherit().mode(execution_mode);
    if let Some(child_model) = args.child_model.as_ref() {
        spec = spec.model(child_model, args.child_variant.clone());
    } else if let Some(child_variant) = args.child_variant.as_ref() {
        spec = spec.model_variant(child_variant);
    }
    if let Some(max_turns) = args.child_max_turns {
        spec = spec.max_turns(max_turns);
    }
    spec
}

fn rlm_config(args: &Args) -> RlmModePluginConfig {
    let mut config = RlmModePluginConfig {
        prompt_features: RlmPromptFeatures {
            images: false,
            ..RlmPromptFeatures::default()
        },
        ..RlmModePluginConfig::default()
    };
    if args.disable_continue_as_fallback {
        config.continue_as_soft_warn_tokens = None;
        config.continue_as_forced_fallback_tokens = None;
    } else {
        if let Some(value) = args.soft_continue_as_tokens {
            config.continue_as_soft_warn_tokens = Some(value);
        }
        if let Some(value) = args.forced_continue_as_tokens {
            config.continue_as_forced_fallback_tokens = Some(value);
        }
    }
    config
}

struct OolongAsyncHandlesTool;

#[async_trait]
impl ToolProvider for OolongAsyncHandlesTool {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![oolong_list_async_handles_tool_definition()]
    }

    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        ToolResult::err_fmt(format_args!(
            "`{}` is handled by the RLM session runtime and cannot run directly",
            call.name
        ))
    }
}

fn oolong_list_async_handles_tool_definition() -> ToolDefinition {
    ToolDefinition::raw(
        "list_async_handles",
        "List live lashlang async handles only. Returns `{ monitor: { monitor_id: handle }, subagent: { name: handle }, tool: { id: handle } }`; terminal, awaited, or cancelled handles are omitted. Use this after fan-out via `spawn_agent`.",
        ToolDefinition::default_input_schema(),
        json!({
            "type": "object",
            "properties": {
                "monitor": { "type": "object" },
                "subagent": { "type": "object" },
                "tool": { "type": "object" }
            },
            "required": ["monitor", "subagent", "tool"]
        }),
    )
    .with_examples(vec![r#"handles = (call list_async_handles {})?"#.into()])
    .with_execution_mode(ToolExecutionMode::Parallel)
}

fn oolong_prompt_template() -> PromptTemplate {
    PromptTemplate::new(vec![
        PromptTemplateSection::untitled(vec![
            PromptTemplateEntry::text(
                "You are solving an OOLONG long-context aggregation task. The context and question are host-projected bindings, not ordinary prompt text. Use symbolic access to inspect and decompose them.",
            ),
            PromptTemplateEntry::slot(PromptSlot::Intro),
        ]),
        PromptTemplateSection::titled(
            "Execution",
            vec![
                PromptTemplateEntry::builtin(PromptBuiltin::ExecutionInstructions),
                PromptTemplateEntry::slot(PromptSlot::Execution),
            ],
        ),
        PromptTemplateSection::titled(
            "OOLONG Strategy",
            vec![PromptTemplateEntry::text(OOLONG_DECOMPOSITION_GUIDANCE)],
        ),
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

const OOLONG_DECOMPOSITION_GUIDANCE: &str = r#"OOLONG rewards exact aggregation, not broad summarization. The context is available as `input.context`, the natural-language question as `input.question`, and metadata as `benchmark`.

Default pattern:
1. Inspect metadata first: dataset, task_group, task, answer_type, context_len.
2. Inspect the question and enough of the context to understand its structure before deciding the strategy. Identify the record boundary and the label/user/date fields before counting.
3. Cover the full context before final submission. Chunk by observed record boundaries or character ranges, and keep chunks large enough to preserve local evidence instead of making unnecessarily tiny fragments.
4. For contexts with many records, dispatch independent `spawn_agent` calls with `capability: "default"` over disjoint ranges. Pass `seed: { context: slice(input.context, start, end), question: input.question, metadata: benchmark }`.
5. Maintain buffers of per-chunk counts, labels, evidence, or partial answers in lashlang data structures. Aggregate those buffers programmatically, then verify totals and tie-breaking before final submission.
6. Submit only the final answer value. If the required output schema lists choices, submit one of those raw values without prefixes like `Label:` or `Answer:`. If the expected answer is a label or user, submit a string; if it asks for multiple entries, submit an array; if it asks for a count, submit a number.

Avoid copying the whole context into prose or printing whole document lists. Use short slices and compact projections to inspect structure and evidence. If a subtask requires its own multi-step inspection, use `spawn_agent`; if it is a small direct classification, extraction, summarization, or judgment over supplied data, `llm_query` is enough. Use `start call spawn_agent` plus `await`/`list_async_handles` for parallel fan-out when chunks are independent."#;

fn build_projected_bindings(
    question: &OolongQuestion,
) -> anyhow::Result<lash_mode_rlm::RlmProjectedBindings> {
    Ok(lash_mode_rlm::RlmProjectedBindings::new()
        .bind_json(
            "benchmark",
            json!({
                "name": "OOLONG",
                "suite": question.suite.label(),
                "question_id": question.question_id,
                "dataset": question.dataset,
                "config": question.config,
                "context_len": question.context_len,
                "context_window_id": question.context_window_id,
                "task_group": question.task_group,
                "task": question.task,
                "answer_type": question.answer_type,
                "input_subset": question.input_subset,
            }),
        )?
        .bind_json(
            "input",
            json!({
                "context": question.context,
                "question": question.question,
                "prompt": question.prompt,
                "question_id": question.question_id,
            }),
        )?)
}

fn resolve_provider(args: &Args) -> anyhow::Result<ProviderHandle> {
    let mut provider = match args.provider_id.as_str() {
        "openai-compatible" => {
            let api_key = resolve_api_key(args).ok_or_else(|| {
                anyhow::anyhow!(
                    "missing API key — set OPENROUTER_API_KEY or OPENAI_COMPATIBLE_API_KEY in .env, or pass --api-key"
                )
            })?;
            let provider = lash_provider_openai::OpenAiCompatibleProvider::new(
                api_key,
                resolve_base_url(args),
            );
            Ok(ProviderHandle::new(provider.into_components()))
        }
        other => {
            let config_path = lash_home().join("config.json");
            let mut config = LashConfig::load(&config_path).ok_or_else(|| {
                anyhow::anyhow!(
                    "missing or invalid {} — initialise a lash config to use provider `{other}`",
                    config_path.display()
                )
            })?;
            config
                .set_active_provider_kind(other)
                .map_err(anyhow::Error::msg)?;
            config.build_active_provider().map_err(anyhow::Error::msg)
        }
    }?;
    let mut options: ProviderOptions = provider.options();
    options.max_output_tokens = (args.max_output_tokens > 0).then_some(args.max_output_tokens);
    provider.set_options(options);
    Ok(provider)
}

fn lash_home() -> PathBuf {
    env::var_os("LASH_HOME")
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|home| home.join(".lash")))
        .unwrap_or_else(|| Path::new(".lash").to_path_buf())
}

fn resolve_api_key(args: &Args) -> Option<String> {
    args.api_key
        .clone()
        .filter(|v| !v.trim().is_empty())
        .or_else(|| read_env_var("OPENAI_COMPATIBLE_API_KEY"))
        .or_else(|| read_env_var("OPENROUTER_API_KEY"))
}

fn resolve_base_url(args: &Args) -> String {
    args.base_url
        .clone()
        .filter(|v| !v.trim().is_empty())
        .or_else(|| read_env_var("OPENAI_COMPATIBLE_BASE_URL"))
        .or_else(|| read_env_var("OPENROUTER_BASE_URL"))
        .unwrap_or_else(|| OPENROUTER_BASE_URL.to_string())
}

fn read_env_var(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

fn turn_completed(outcome: &TurnOutcome) -> bool {
    matches!(
        outcome,
        TurnOutcome::Finished(_) | TurnOutcome::Handoff { .. }
    )
}

fn turn_status_label(outcome: &TurnOutcome) -> &'static str {
    match outcome {
        TurnOutcome::Finished(_) | TurnOutcome::Handoff { .. } => "completed",
        TurnOutcome::Stopped(TurnStop::Cancelled) => "interrupted",
        TurnOutcome::Stopped(_) => "failed",
    }
}

fn done_reason_label(outcome: &TurnOutcome) -> &'static str {
    match outcome {
        TurnOutcome::Finished(TurnFinish::AssistantMessage { .. }) => "assistant_message",
        TurnOutcome::Finished(TurnFinish::SubmittedValue { .. }) => "submitted_value",
        TurnOutcome::Finished(TurnFinish::ToolValue { .. }) => "tool_value",
        TurnOutcome::Handoff { .. } => "handoff",
        TurnOutcome::Stopped(TurnStop::Cancelled) => "cancelled",
        TurnOutcome::Stopped(TurnStop::InvalidInput) => "invalid_input",
        TurnOutcome::Stopped(TurnStop::MaxTurns) => "max_turns",
        TurnOutcome::Stopped(TurnStop::ToolFailure) => "tool_failure",
        TurnOutcome::Stopped(TurnStop::ProviderError) => "provider_error",
        TurnOutcome::Stopped(TurnStop::PluginAbort) => "plugin_abort",
        TurnOutcome::Stopped(TurnStop::RuntimeError) => "runtime_error",
        TurnOutcome::Stopped(TurnStop::SubmittedError { .. }) => "submitted_error",
        TurnOutcome::Stopped(TurnStop::ToolError { .. }) => "tool_error",
    }
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> anyhow::Result<()> {
    let text = serde_json::to_string_pretty(value)?;
    fs::write(path, format!("{text}\n")).with_context(|| format!("write {}", path.display()))
}

fn question_artifacts(question_dir: &Path) -> QuestionArtifacts {
    let path = |name: &str| question_dir.join(name).display().to_string();
    QuestionArtifacts {
        question_json: path("question.json"),
        result_json: path("result.json"),
        answer_txt: path("answer.txt"),
        prompt_txt: path("prompt.txt"),
        system_prompt_txt: path("system_prompt.txt"),
        events_jsonl: path("events.jsonl"),
        session_db: path("session.db"),
        trace_jsonl: path("session.trace.jsonl"),
        trace_html: path("trace.html"),
    }
}

fn append_response_row(path: &Path, row: &QuestionResult) -> anyhow::Result<()> {
    let line = serde_json::to_string(row)?;
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("open {}", path.display()))?;
    writeln!(file, "{line}").with_context(|| format!("append {}", path.display()))?;
    Ok(())
}

fn load_completed_ids(path: &Path) -> anyhow::Result<BTreeSet<String>> {
    if !path.exists() {
        return Ok(BTreeSet::new());
    }
    let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut out = BTreeSet::new();
    for line in raw.lines().filter(|l| !l.trim().is_empty()) {
        let value: Value = serde_json::from_str(line)
            .with_context(|| format!("parse row from {}", path.display()))?;
        if let Some(qid) = value.get("question_id").and_then(Value::as_str) {
            out.insert(qid.to_string());
        }
    }
    Ok(out)
}

fn aggregate_usage(reports: impl IntoIterator<Item = SessionUsageReport>) -> SessionUsageReport {
    let mut total = BTreeMap::<(String, String), TokenUsage>::new();
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
        .map(|((source, model), usage)| TokenLedgerEntry {
            source,
            model,
            usage,
        })
        .collect::<Vec<_>>();
    SessionUsageReport::from_entries(&entries)
}

fn aggregate_metrics<'a>(metrics: impl IntoIterator<Item = &'a QuestionMetrics>) -> RunMetrics {
    let mut out = RunMetrics::default();
    for metric in metrics {
        out.root_llm_calls += metric.root_llm_calls;
        out.child_llm_calls += metric.child_llm_calls;
        out.token_usage_events += metric.token_usage_events;
        out.child_usage_events += metric.child_usage_events;
        out.trace_llm_calls += metric.trace_llm_calls;
        out.trace_llm_failures += metric.trace_llm_failures;
        out.trace_llm_duration_ms += metric.trace_llm_duration_ms;
        merge_counts(
            &mut out.trace_llm_calls_by_session,
            &metric.trace_llm_calls_by_session,
        );
        merge_counts(&mut out.tool_calls_by_name, &metric.tool_calls_by_name);
        merge_durations(
            &mut out.tool_call_duration_ms_by_name,
            &metric.tool_call_duration_ms_by_name,
        );
    }
    out
}

fn summarize_timing(results: &[QuestionResult]) -> TimingSummary {
    if results.is_empty() {
        return TimingSummary::default();
    }
    let mut values = results
        .iter()
        .map(|r| r.elapsed_seconds)
        .collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    let total = values.iter().sum::<f64>();
    TimingSummary {
        total_question_seconds: total,
        mean_question_seconds: total / values.len() as f64,
        min_question_seconds: values[0],
        max_question_seconds: values[values.len() - 1],
        p50_question_seconds: percentile(&values, 0.50),
        p95_question_seconds: percentile(&values, 0.95),
    }
}

fn percentile(sorted_values: &[f64], percentile: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    let rank = ((sorted_values.len() - 1) as f64 * percentile).ceil() as usize;
    sorted_values[rank.min(sorted_values.len() - 1)]
}

fn merge_counts(target: &mut BTreeMap<String, usize>, source: &BTreeMap<String, usize>) {
    for (key, value) in source {
        *target.entry(key.clone()).or_default() += value;
    }
}

fn merge_durations(target: &mut BTreeMap<String, u64>, source: &BTreeMap<String, u64>) {
    for (key, value) in source {
        *target.entry(key.clone()).or_default() += value;
    }
}

#[derive(Clone, Debug, Default)]
struct TraceMetrics {
    llm_calls: usize,
    llm_failures: usize,
    llm_duration_ms: u64,
    llm_calls_by_session: BTreeMap<String, usize>,
}

fn read_trace_metrics(path: &Path) -> anyhow::Result<TraceMetrics> {
    if !path.exists() {
        return Ok(TraceMetrics::default());
    }
    let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut metrics = TraceMetrics::default();
    for line in raw.lines().filter(|line| !line.trim().is_empty()) {
        let value: Value = serde_json::from_str(line)
            .with_context(|| format!("parse trace row from {}", path.display()))?;
        let event_type = value.get("type").and_then(Value::as_str);
        let session_id = value
            .get("context")
            .and_then(|ctx| ctx.get("session_id"))
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        match event_type {
            Some("llm_call_completed") => {
                metrics.llm_calls += 1;
                *metrics
                    .llm_calls_by_session
                    .entry(session_id.to_string())
                    .or_default() += 1;
                metrics.llm_duration_ms += value
                    .get("response")
                    .and_then(|response| response.get("duration_ms"))
                    .and_then(Value::as_u64)
                    .unwrap_or_default();
            }
            Some("llm_call_failed") => {
                metrics.llm_failures += 1;
                *metrics
                    .llm_calls_by_session
                    .entry(session_id.to_string())
                    .or_default() += 1;
            }
            _ => {}
        }
    }
    Ok(metrics)
}

fn aggregate_by(items: impl IntoIterator<Item = (String, bool)>) -> BTreeMap<String, Bucket> {
    let mut out = BTreeMap::<String, Bucket>::new();
    for (key, correct) in items {
        let bucket = out.entry(key).or_default();
        bucket.count += 1;
        if correct {
            bucket.correct += 1;
        }
    }
    out
}

fn write_trace_index(
    output_dir: &Path,
    run_id: &str,
    results: &[QuestionResult],
) -> anyhow::Result<()> {
    let rows: String = results
        .iter()
        .map(|r| {
            let qid = html_escape(&r.question_id);
            let dataset = html_escape(r.dataset.as_deref().unwrap_or("-"));
            let task = html_escape(r.task_group.as_deref().unwrap_or("-"));
            let status = html_escape(&r.status);
            let badge_class = if r.correct { "ok" } else { "fail" };
            format!(
                "<tr>\
                   <td><a href=\"questions/{qid}/trace.html\">{qid}</a></td>\
                   <td>{dataset}</td>\
                   <td>{task}</td>\
                   <td class=\"{badge_class}\">{correct}</td>\
                   <td>{status}</td>\
                   <td>{iters}</td>\
                   <td>{seconds:.1}s</td>\
                   <td><a href=\"questions/{qid}/system_prompt.txt\">system</a> · \
                       <a href=\"questions/{qid}/prompt.txt\">prompt</a> · \
                       <a href=\"questions/{qid}/answer.txt\">answer</a> · \
                       <a href=\"questions/{qid}/events.jsonl\">events</a> · \
                       <a href=\"questions/{qid}/session.trace.jsonl\">trace.jsonl</a> · \
                       <a href=\"questions/{qid}/trace.html\">trace.html</a> · \
                       <a href=\"questions/{qid}/session.db\">session.db</a></td>\
                 </tr>",
                correct = if r.correct { "correct" } else { "wrong" },
                iters = r.iterations,
                seconds = r.elapsed_seconds,
            )
        })
        .collect();

    let html = format!(
        r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>OOLONG run {run_id}</title>
<style>
  body {{ font: 14px/1.45 ui-sans-serif, system-ui, sans-serif; max-width: 1200px; margin: 2rem auto; padding: 0 1rem; color: #111; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 0.2rem; }}
  p.meta {{ color: #555; margin-top: 0; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border-bottom: 1px solid #eee; padding: 6px 10px; text-align: left; font-variant-numeric: tabular-nums; }}
  th {{ background: #fafafa; position: sticky; top: 0; }}
  td.ok {{ color: #1a7f37; font-weight: 600; }}
  td.fail {{ color: #cf222e; font-weight: 600; }}
  a {{ color: #0366d6; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  code {{ background: #f6f8fa; padding: 1px 4px; border-radius: 3px; }}
</style>
</head>
<body>
<h1>OOLONG run <code>{run_id}</code></h1>
<p class="meta">{count} questions · see <a href="results.json">results.json</a> / <a href="manifest.json">manifest.json</a></p>
<table>
  <thead>
    <tr>
      <th>question_id</th><th>dataset</th><th>task</th><th>score</th>
      <th>status</th><th>iters</th><th>elapsed</th><th>artifacts</th>
    </tr>
  </thead>
  <tbody>
    {rows}
  </tbody>
</table>
</body>
</html>
"#,
        count = results.len(),
    );
    fs::write(output_dir.join("index.html"), html)
        .with_context(|| format!("write {}", output_dir.join("index.html").display()))?;
    Ok(())
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

struct OolongEventSink {
    file: Mutex<File>,
    last_error: Mutex<Option<String>>,
    iteration_count: Mutex<BTreeSet<usize>>,
    root_llm_calls: Mutex<usize>,
    child_llm_calls: Mutex<usize>,
    token_usage_events: Mutex<usize>,
    child_usage_events: Mutex<usize>,
    tool_calls_by_name: Mutex<BTreeMap<String, usize>>,
    tool_call_duration_ms_by_name: Mutex<BTreeMap<String, u64>>,
}

impl OolongEventSink {
    fn new(path: PathBuf) -> anyhow::Result<Self> {
        let file = File::create(&path).with_context(|| format!("create {}", path.display()))?;
        Ok(Self {
            file: Mutex::new(file),
            last_error: Mutex::new(None),
            iteration_count: Mutex::new(BTreeSet::new()),
            root_llm_calls: Mutex::new(0),
            child_llm_calls: Mutex::new(0),
            token_usage_events: Mutex::new(0),
            child_usage_events: Mutex::new(0),
            tool_calls_by_name: Mutex::new(BTreeMap::new()),
            tool_call_duration_ms_by_name: Mutex::new(BTreeMap::new()),
        })
    }

    fn iteration_count(&self) -> usize {
        self.iteration_count
            .lock()
            .map(|turns| turns.len())
            .unwrap_or_default()
    }

    fn last_error(&self) -> Option<String> {
        self.last_error.lock().ok().and_then(|v| v.clone())
    }

    fn metrics(&self, wall_clock_seconds: f64, trace: TraceMetrics) -> QuestionMetrics {
        QuestionMetrics {
            wall_clock_seconds,
            root_llm_calls: self.root_llm_calls.lock().map(|v| *v).unwrap_or_default(),
            child_llm_calls: self.child_llm_calls.lock().map(|v| *v).unwrap_or_default(),
            token_usage_events: self
                .token_usage_events
                .lock()
                .map(|v| *v)
                .unwrap_or_default(),
            child_usage_events: self
                .child_usage_events
                .lock()
                .map(|v| *v)
                .unwrap_or_default(),
            trace_llm_calls: trace.llm_calls,
            trace_llm_failures: trace.llm_failures,
            trace_llm_duration_ms: trace.llm_duration_ms,
            trace_llm_calls_by_session: trace.llm_calls_by_session,
            tool_calls_by_name: self
                .tool_calls_by_name
                .lock()
                .map(|v| v.clone())
                .unwrap_or_default(),
            tool_call_duration_ms_by_name: self
                .tool_call_duration_ms_by_name
                .lock()
                .map(|v| v.clone())
                .unwrap_or_default(),
        }
    }
}

#[async_trait]
impl EventSink for OolongEventSink {
    async fn emit(&self, event: SessionEvent) {
        match &event {
            SessionEvent::LlmRequest { mode_iteration, .. } => {
                if let Ok(mut turns) = self.iteration_count.lock() {
                    turns.insert(*mode_iteration);
                }
                if let Ok(mut calls) = self.root_llm_calls.lock() {
                    *calls += 1;
                }
            }
            SessionEvent::LlmResponse { .. } => {}
            SessionEvent::TokenUsage { .. } => {
                if let Ok(mut count) = self.token_usage_events.lock() {
                    *count += 1;
                }
            }
            SessionEvent::ChildTokenUsage { .. } => {
                if let Ok(mut count) = self.child_usage_events.lock() {
                    *count += 1;
                }
                if let Ok(mut calls) = self.child_llm_calls.lock() {
                    *calls += 1;
                }
            }
            SessionEvent::ToolCall {
                name, duration_ms, ..
            } => {
                if let Ok(mut counts) = self.tool_calls_by_name.lock() {
                    *counts.entry(name.clone()).or_default() += 1;
                }
                if let Ok(mut durations) = self.tool_call_duration_ms_by_name.lock() {
                    *durations.entry(name.clone()).or_default() += *duration_ms;
                }
            }
            SessionEvent::Error { message, .. } => {
                if let Ok(mut last) = self.last_error.lock() {
                    *last = Some(message.clone());
                }
            }
            _ => {}
        }
        if let Ok(line) = serde_json::to_string(&event)
            && let Ok(mut file) = self.file.lock()
        {
            let _ = writeln!(file, "{line}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn answer_matching_accepts_label_prefixes() {
        assert!(answer_matches(
            &Value::String("Label: spam".to_string()),
            &json!(["spam"])
        ));
    }

    #[test]
    fn answer_matching_compares_array_sets() {
        assert!(answer_matches(&json!(["b", "a"]), &json!(["a", "b"])));
    }

    fn question_with_answer_type(answer_type: &str, question: &str) -> OolongQuestion {
        OolongQuestion {
            question_id: "q".to_string(),
            suite: OolongSuite::Synth,
            split: "validation".to_string(),
            dataset: Some("spam".to_string()),
            config: None,
            context_len: Some(1024),
            context_window_id: Some(1),
            task_group: Some("counting".to_string()),
            task: None,
            answer_type: Some(answer_type.to_string()),
            input_subset: None,
            prompt: String::new(),
            context: String::new(),
            question: question.to_string(),
            answer: Value::Null,
            source: Value::Null,
        }
    }

    #[test]
    fn oolong_answer_schema_uses_label_choices_from_question() {
        let question = question_with_answer_type(
            "ANSWER_TYPE.LABEL",
            "Give your final answer in the form 'Label: answer' where answer is one of the labels: ham, spam.",
        );

        let schema = oolong_answer_schema(&question);

        assert_eq!(schema["anyOf"][0]["enum"], json!(["ham", "spam"]));
        assert_eq!(schema["anyOf"][1]["items"]["enum"], json!(["ham", "spam"]));
    }

    #[test]
    fn oolong_answer_schema_uses_numeric_type() {
        let question = question_with_answer_type(
            "ANSWER_TYPE.NUMERIC",
            "Give your final answer in the form 'Answer: number'.",
        );

        let schema = oolong_answer_schema(&question);

        assert_eq!(schema, json!({ "type": "integer" }));
    }

    #[test]
    fn oolong_answer_schema_uses_comparison_enum() {
        let question = question_with_answer_type(
            "ANSWER_TYPE.COMPARISON",
            "Is label 'ham' more common, less common, or the same frequency as label 'spam'?",
        );

        let schema = oolong_answer_schema(&question);

        assert_eq!(
            schema["enum"],
            json!(["more common than", "less common than", "same frequency as"])
        );
    }
}
