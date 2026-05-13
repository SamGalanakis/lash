use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

use anyhow::{Context, bail};
use async_trait::async_trait;
use chrono::Utc;
use clap::Parser;
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

const STATE_ROOT: &str = ".benchmarks/frontier-cs";
const DEFAULT_SOURCE_DIR: &str = ".benchmarks/frontier-cs/source";
const DEFAULT_MODEL: &str = "openai/gpt-5.2";
const DEFAULT_VARIANT: &str = "high";
const DEFAULT_MAX_TURNS: usize = 50;
const DEFAULT_MAX_CONTEXT_TOKENS: usize = 1_000_000;
const DEFAULT_MAX_OUTPUT_TOKENS: u64 = 125_000;
const DEFAULT_BATCH_SIZE: usize = 1;
const EXECUTION_MODE_LABEL: &str = "rlm";
const SUBAGENT_CAPABILITY: &str = "default";

const FRONTIER_USER_DIRECTIVE: &str = concat!(
    "Solve the Frontier-CS problem bound as `problem`. ",
    "Reason carefully, decompose with subagents when useful, and submit only the complete source code as a string. ",
    "Do not wrap the final answer in markdown fences."
);

#[derive(Parser, Debug, Clone)]
#[command(name = "bench-frontier-cs")]
#[command(about = "Run Frontier-CS through Lash as an RLM benchmark.")]
struct Args {
    #[arg(long, default_value = "algorithmic", value_parser = ["algorithmic", "research"])]
    track: String,

    #[arg(long, default_value = DEFAULT_SOURCE_DIR)]
    source_dir: PathBuf,

    #[arg(long)]
    problem_id: Vec<String>,

    #[arg(long)]
    problems_file: Option<PathBuf>,

    #[arg(long, default_value_t = 0)]
    offset: usize,

    #[arg(long)]
    max_problems: Option<usize>,

    #[arg(long)]
    shuffle_seed: Option<u64>,

    #[arg(long)]
    run_id: Option<String>,

    #[arg(long)]
    output_dir: Option<PathBuf>,

    #[arg(long)]
    resume: bool,

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

    #[arg(long, default_value_t = DEFAULT_BATCH_SIZE)]
    batch_size: usize,

    #[arg(long)]
    await_background_work: bool,

    #[arg(long)]
    no_evaluate: bool,

    #[arg(long)]
    backend: Option<String>,

    #[arg(long, default_value = "http://localhost:8081")]
    judge_url: String,

    #[arg(long)]
    keep_cluster: bool,

    #[arg(long)]
    dry_run: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct FrontierProblem {
    problem_id: String,
    track: String,
    statement: String,
    source_path: String,
    solution_ext: String,
    language: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RunManifest {
    run_id: String,
    created_at: String,
    source_dir: String,
    track: String,
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
    problem_ids: Vec<String>,
    problems_file: Option<String>,
    offset: usize,
    max_problems: Option<usize>,
    shuffle_seed: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ReferenceSettings {
    upstream_repo: String,
    paper: String,
    note: String,
}

impl Default for ReferenceSettings {
    fn default() -> Self {
        Self {
            upstream_repo: "https://github.com/FrontierCS/Frontier-CS".to_string(),
            paper: "https://arxiv.org/abs/2512.15699".to_string(),
            note: "This harness generates Frontier-CS solutions with Lash RLM and scores them with Frontier-CS's official evaluator. Algorithmic is the default local Docker/go-judge track; research can be selected when its required backend resources are available.".to_string(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ProblemResult {
    problem_id: String,
    track: String,
    language: String,
    model: String,
    solution_path: String,
    successful: bool,
    score: Option<f64>,
    score_unbounded: Option<f64>,
    evaluation_status: String,
    evaluation_message: Option<String>,
    evaluation_duration_seconds: Option<f64>,
    generation_status: String,
    done_reason: String,
    failure_reason: Option<String>,
    usage: SessionUsageReport,
    elapsed_seconds: f64,
    iterations: usize,
    metrics: ProblemMetrics,
    artifacts: ProblemArtifacts,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ProblemArtifacts {
    problem_txt: String,
    solution: String,
    answer_txt: String,
    result_json: String,
    evaluation_json: String,
    events_jsonl: String,
    session_db: String,
    trace_jsonl: String,
    trace_html: String,
    system_prompt_txt: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct ProblemMetrics {
    wall_clock_seconds: f64,
    root_llm_calls: usize,
    child_llm_calls: usize,
    token_usage_events: usize,
    child_usage_events: usize,
    tool_calls_by_name: BTreeMap<String, usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RunSummary {
    run_id: String,
    started_at: String,
    finished_at: String,
    duration_seconds: i64,
    problem_count: usize,
    result_count: usize,
    successful: usize,
    evaluated: usize,
    average_score: f64,
    by_track: BTreeMap<String, Bucket>,
    iterations: usize,
    wall_clock_seconds: f64,
    usage: SessionUsageReport,
    predictions_path: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Bucket {
    count: usize,
    successful: usize,
    score_sum: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct FrontierEvalRow {
    problem_id: String,
    score: Option<f64>,
    score_unbounded: Option<f64>,
    status: String,
    message: Option<String>,
    duration_seconds: Option<f64>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _ = dotenvy::dotenv();
    lash_providers_builtin::register_all();

    let args = Args::parse();
    if !args.source_dir.join("pyproject.toml").exists() {
        bail!(
            "Frontier-CS source not found at {} - run bench/frontier-cs/setup.sh first",
            args.source_dir.display()
        );
    }

    let mut problems = load_problems(&args)?;
    if let Some(seed) = args.shuffle_seed {
        simple_shuffle(&mut problems, seed);
    }
    if args.offset > 0 {
        problems = problems.into_iter().skip(args.offset).collect();
    }
    if let Some(limit) = args.max_problems {
        problems.truncate(limit);
    }
    if problems.is_empty() {
        bail!("no Frontier-CS problems selected");
    }

    let run_id = args
        .run_id
        .clone()
        .unwrap_or_else(|| Utc::now().format("%Y%m%dT%H%M%SZ").to_string());
    let runs_dir = PathBuf::from(STATE_ROOT).join("runs");
    fs::create_dir_all(&runs_dir).with_context(|| format!("create {}", runs_dir.display()))?;
    let output_dir = args
        .output_dir
        .clone()
        .unwrap_or_else(|| runs_dir.join(&run_id));
    fs::create_dir_all(&output_dir).with_context(|| format!("create {}", output_dir.display()))?;
    let predictions_path = output_dir.join("predictions.jsonl");

    let manifest = RunManifest {
        run_id: run_id.clone(),
        created_at: Utc::now().to_rfc3339(),
        source_dir: args.source_dir.display().to_string(),
        track: args.track.clone(),
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
            problem_ids: args.problem_id.clone(),
            problems_file: args
                .problems_file
                .as_ref()
                .map(|path| path.display().to_string()),
            offset: args.offset,
            max_problems: args.max_problems,
            shuffle_seed: args.shuffle_seed,
        },
        selected_count: problems.len(),
        predictions_path: predictions_path.display().to_string(),
        reference: ReferenceSettings::default(),
    };
    write_json(&output_dir.join("manifest.json"), &manifest)?;

    if args.dry_run {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "manifest": manifest,
                "problems": problems.iter().map(problem_preview).collect::<Vec<_>>(),
            }))?
        );
        return Ok(());
    }

    let provider = Arc::new(resolve_provider(&args)?);
    let completed = if args.resume {
        load_completed_ids(&predictions_path)?
    } else {
        BTreeSet::new()
    };
    let pending = problems
        .iter()
        .filter(|problem| !completed.contains(&result_key(problem)))
        .cloned()
        .collect::<Vec<_>>();

    eprintln!("Frontier-CS run_id={run_id}");
    eprintln!("  track:          {}", args.track);
    eprintln!("  selected:       {}", problems.len());
    eprintln!("  pending:        {}", pending.len());
    eprintln!("  model:          {}", args.model);
    eprintln!("  batch_size:     {}", args.batch_size.max(1));
    eprintln!("  source:         {}", args.source_dir.display());
    eprintln!("  predictions:    {}", predictions_path.display());

    let started_at = Utc::now();
    let started_instant = std::time::Instant::now();
    let semaphore = Arc::new(Semaphore::new(args.batch_size.max(1)));
    let args = Arc::new(args);
    let output_dir = Arc::new(output_dir);
    let predictions_path = Arc::new(predictions_path);
    let total = pending.len();
    let mut join_set = JoinSet::new();
    for (index, problem) in pending.into_iter().enumerate() {
        let permit = semaphore
            .clone()
            .acquire_owned()
            .await
            .context("acquire benchmark slot")?;
        let provider = provider.clone();
        let args = args.clone();
        let output_dir = output_dir.clone();
        let predictions_path = predictions_path.clone();
        join_set.spawn(async move {
            let _permit = permit;
            let result = run_problem(
                output_dir.as_ref(),
                provider.as_ref(),
                args.as_ref(),
                problem,
            )
            .await;
            if let Ok(row) = &result {
                let _ = append_response_row(predictions_path.as_ref(), row);
            }
            (index, result)
        });
    }

    let mut indexed = Vec::<(usize, ProblemResult)>::new();
    let mut done = 0usize;
    while let Some(joined) = join_set.join_next().await {
        let (index, result) = joined.context("benchmark task panicked")?;
        let result = match result {
            Ok(value) => value,
            Err(err) => {
                join_set.abort_all();
                return Err(err);
            }
        };
        done += 1;
        eprintln!(
            "  [{}/{}] {}/{} score={} eval={} gen={} t={:.1}s",
            done,
            total,
            result.track,
            result.problem_id,
            result
                .score
                .map(|value| format!("{value:.4}"))
                .unwrap_or_else(|| "-".to_string()),
            result.evaluation_status,
            result.generation_status,
            result.elapsed_seconds,
        );
        indexed.push((index, result));
    }
    indexed.sort_by_key(|(idx, _)| *idx);
    let results = indexed
        .into_iter()
        .map(|(_, result)| result)
        .collect::<Vec<_>>();

    let finished_at = Utc::now();
    let score_sum = results.iter().filter_map(|r| r.score).sum::<f64>();
    let summary = RunSummary {
        run_id: run_id.clone(),
        started_at: started_at.to_rfc3339(),
        finished_at: finished_at.to_rfc3339(),
        duration_seconds: (finished_at - started_at).num_seconds(),
        problem_count: problems.len(),
        result_count: results.len(),
        successful: results.iter().filter(|r| r.successful).count(),
        evaluated: results
            .iter()
            .filter(|r| r.evaluation_status == "success")
            .count(),
        average_score: if results.is_empty() {
            0.0
        } else {
            score_sum / results.len() as f64
        },
        by_track: aggregate_by_track(&results),
        iterations: results.iter().map(|r| r.iterations).sum(),
        wall_clock_seconds: started_instant.elapsed().as_secs_f64(),
        usage: aggregate_usage(results.iter().map(|r| r.usage.clone())),
        predictions_path: predictions_path.display().to_string(),
    };
    write_json(&output_dir.join("results.json"), &summary)?;
    write_trace_index(&output_dir, &run_id, &results)?;

    eprintln!();
    eprintln!("Run summary:");
    eprintln!("  run_dir:      {}", output_dir.display());
    eprintln!(
        "  successful:   {}/{}",
        summary.successful, summary.result_count
    );
    eprintln!(
        "  evaluated:    {}/{}",
        summary.evaluated, summary.result_count
    );
    eprintln!("  avg_score:    {:.4}", summary.average_score);
    eprintln!("  iterations:   {}", summary.iterations);
    eprintln!("  wall_clock:   {:.1}s", summary.wall_clock_seconds);
    eprintln!();
    eprintln!("Evaluate with:");
    eprintln!("  bench/frontier-cs/evaluate.sh {}", output_dir.display());
    Ok(())
}

fn load_problems(args: &Args) -> anyhow::Result<Vec<FrontierProblem>> {
    let ids = if !args.problem_id.is_empty() {
        args.problem_id.clone()
    } else if let Some(path) = args.problems_file.as_ref() {
        fs::read_to_string(path)
            .with_context(|| format!("read {}", path.display()))?
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .map(ToString::to_string)
            .collect::<Vec<_>>()
    } else {
        list_problem_ids(&args.source_dir, &args.track)?
    };
    ids.into_iter()
        .map(|id| load_problem(&args.source_dir, &args.track, &id))
        .collect()
}

fn list_problem_ids(source_dir: &Path, track: &str) -> anyhow::Result<Vec<String>> {
    let problems_dir = source_dir.join(track).join("problems");
    let mut ids = Vec::new();
    for entry in
        fs::read_dir(&problems_dir).with_context(|| format!("read {}", problems_dir.display()))?
    {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            ids.push(entry.file_name().to_string_lossy().to_string());
        }
    }
    ids.sort_by(|a, b| sort_key(a).cmp(&sort_key(b)));
    Ok(ids)
}

fn sort_key(value: &str) -> (u8, u64, &str) {
    value
        .parse::<u64>()
        .map(|n| (0, n, ""))
        .unwrap_or((1, 0, value))
}

fn load_problem(
    source_dir: &Path,
    track: &str,
    problem_id: &str,
) -> anyhow::Result<FrontierProblem> {
    match track {
        "algorithmic" => {
            let path = source_dir
                .join("algorithmic")
                .join("problems")
                .join(problem_id)
                .join("statement.txt");
            let statement =
                fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
            Ok(FrontierProblem {
                problem_id: problem_id.to_string(),
                track: track.to_string(),
                statement,
                source_path: path.display().to_string(),
                solution_ext: "cpp".to_string(),
                language: "C++17".to_string(),
            })
        }
        "research" => {
            let base = source_dir
                .join("research")
                .join("problems")
                .join(problem_id);
            let path = ["readme", "README.md"]
                .into_iter()
                .map(|name| base.join(name))
                .find(|path| path.exists())
                .ok_or_else(|| anyhow::anyhow!("missing research readme for {problem_id}"))?;
            let statement =
                fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
            Ok(FrontierProblem {
                problem_id: problem_id.to_string(),
                track: track.to_string(),
                statement,
                source_path: path.display().to_string(),
                solution_ext: "py".to_string(),
                language: "Python".to_string(),
            })
        }
        other => bail!("unsupported track {other}"),
    }
}

fn problem_preview(problem: &FrontierProblem) -> Value {
    json!({
        "track": problem.track,
        "problem_id": problem.problem_id,
        "language": problem.language,
        "statement_chars": problem.statement.chars().count(),
        "source_path": problem.source_path,
    })
}

async fn run_problem(
    output_dir: &Path,
    provider: &ProviderHandle,
    args: &Args,
    problem: FrontierProblem,
) -> anyhow::Result<ProblemResult> {
    let problem_dir = output_dir
        .join("problems")
        .join(&problem.track)
        .join(safe_path_segment(&problem.problem_id));
    fs::create_dir_all(&problem_dir)
        .with_context(|| format!("create {}", problem_dir.display()))?;
    fs::write(problem_dir.join("problem.txt"), &problem.statement)
        .with_context(|| format!("write {}", problem_dir.join("problem.txt").display()))?;

    let store_path = problem_dir.join("session.db");
    let trace_path = problem_dir.join("session.trace.jsonl");
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
        .prompt_template(frontier_prompt_template())
        .trace_jsonl_path(Some(trace_path.clone()))
        .advanced()
        .plugin_host(build_plugin_host(execution_mode.clone(), args))
        .build()?;
    let session = core
        .session(format!("frontier-cs-{}", problem.problem_id))
        .rlm()
        .store(store.clone())
        .open()
        .await?;

    let before_usage = session.usage_report();
    let started = std::time::Instant::now();
    let cancel = tokio_util::sync::CancellationToken::new();
    let sink = Arc::new(FrontierEventSink::new(problem_dir.join("events.jsonl"))?);
    let sink_trait: Arc<dyn EventSink> = sink.clone();
    let input = TurnInput {
        items: vec![InputItem::Text {
            text: FRONTIER_USER_DIRECTIVE.to_string(),
        }],
        image_blobs: Default::default(),
        mode_turn_options: Some(ModeTurnOptions::typed(
            execution_mode,
            RlmTermination::SubmitRequired {
                schema: Some(json!({ "type": "string" })),
            },
        )?),
        trace_turn_id: None,
        mode_extension: None,
        turn_context: TurnContext::default(),
    }
    .rlm_project(build_projected_bindings(&problem)?)?;
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
            "  cancelled {} background task(s) after {}/{}",
            cancelled.len(),
            problem.track,
            problem.problem_id
        );
    }
    background_result?;
    let turn = turn_result.context("run Frontier-CS problem")?;
    let elapsed_seconds = started.elapsed().as_secs_f64();
    let usage = diff_usage_reports(&before_usage, &session.usage_report())
        .map(|rows| SessionUsageReport::from_entries(&rows))
        .map_err(anyhow::Error::msg)
        .context("diff usage reports")?;

    let solution = strip_code_fence(&solution_from_turn(
        &turn.outcome,
        &turn.assistant_output.safe_text,
    ));
    let solution_path = problem_dir.join(format!("solution.{}", problem.solution_ext));
    fs::write(&solution_path, &solution)
        .with_context(|| format!("write {}", solution_path.display()))?;
    fs::write(problem_dir.join("answer.txt"), &solution)
        .with_context(|| format!("write {}", problem_dir.join("answer.txt").display()))?;

    let eval = if args.no_evaluate {
        FrontierEvalRow {
            problem_id: problem.problem_id.clone(),
            score: None,
            score_unbounded: None,
            status: "skipped".to_string(),
            message: Some("--no-evaluate".to_string()),
            duration_seconds: None,
        }
    } else {
        evaluate_solution(args, &problem, &solution_path)
            .with_context(|| format!("evaluate {}/{}", problem.track, problem.problem_id))?
    };
    write_json(&problem_dir.join("evaluation.json"), &eval)?;

    let generation_status = turn_status_label(&turn.outcome).to_string();
    let done_reason = done_reason_label(&turn.outcome).to_string();
    let failure_reason = if turn_completed(&turn.outcome) {
        None
    } else {
        turn.errors
            .first()
            .map(|e| e.message.clone())
            .or_else(|| sink.last_error())
            .or_else(|| Some(format!("status={generation_status} reason={done_reason}")))
    };

    let result = ProblemResult {
        problem_id: problem.problem_id.clone(),
        track: problem.track.clone(),
        language: problem.language.clone(),
        model: args.model.clone(),
        solution_path: solution_path.display().to_string(),
        successful: eval.status == "success" && eval.score.unwrap_or_default() > 0.0,
        score: eval.score,
        score_unbounded: eval.score_unbounded,
        evaluation_status: eval.status,
        evaluation_message: eval.message,
        evaluation_duration_seconds: eval.duration_seconds,
        generation_status,
        done_reason,
        failure_reason,
        usage,
        elapsed_seconds,
        iterations: sink.iteration_count(),
        metrics: sink.metrics(elapsed_seconds),
        artifacts: problem_artifacts(&problem_dir, &problem.solution_ext),
    };
    write_json(&problem_dir.join("result.json"), &result)?;

    let html_trace_path = problem_dir.join("trace.html");
    if let Err(err) = export(
        &store_path,
        &trace_path,
        ExportFormat::Html,
        Some(&html_trace_path),
    ) {
        eprintln!(
            "warn: failed to render HTML trace for {}/{}: {err:#}",
            problem.track, problem.problem_id
        );
    }
    if let Err(err) = write_system_prompt_snapshot(&trace_path, &problem_dir) {
        eprintln!(
            "warn: failed to snapshot system prompt for {}/{}: {err:#}",
            problem.track, problem.problem_id
        );
    }

    Ok(result)
}

fn build_projected_bindings(
    problem: &FrontierProblem,
) -> anyhow::Result<lash_mode_rlm::RlmProjectedBindings> {
    Ok(lash_mode_rlm::RlmProjectedBindings::new().bind_json(
        "problem",
        json!({
            "benchmark": "Frontier-CS",
            "track": problem.track,
            "problem_id": problem.problem_id,
            "language": problem.language,
            "source_path": problem.source_path,
            "statement": problem.statement,
            "requirements": track_requirements(&problem.track),
        }),
    )?)
}

fn track_requirements(track: &str) -> &'static str {
    match track {
        "algorithmic" => {
            "Return one complete C++17 source file. It must read from stdin, write to stdout, and include any needed headers. Do not include explanations."
        }
        "research" => {
            "Return one complete Python solution.py file implementing the Solution interface requested by the problem readme. Do not include explanations."
        }
        _ => "Return one complete source file.",
    }
}

fn frontier_prompt_template() -> PromptTemplate {
    PromptTemplate::new(vec![
        PromptTemplateSection::untitled(vec![
            PromptTemplateEntry::text(
                "You are solving a Frontier-CS benchmark problem. The complete problem statement and metadata are bound as `problem`.",
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
            "Frontier-CS Strategy",
            vec![PromptTemplateEntry::text(FRONTIER_STRATEGY)],
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

const FRONTIER_STRATEGY: &str = r#"Frontier-CS rewards verifiable score, not prose. The problem is available as `problem.statement`; metadata and output requirements are in `problem`.

For algorithmic tasks:
1. Read the full statement and infer the input/output protocol exactly.
2. Design the strongest practical algorithm you can within the stated limits.
3. Use `llm_query` or `spawn_agent` to independently check tricky cases, proof obligations, and implementation details.
4. Submit exactly one complete C++17 file as a string. No markdown fences.

For research tasks:
1. Read the readme interface carefully and implement the required `Solution` class.
2. Prefer a robust baseline that runs inside the official evaluator over fragile, environment-specific tricks.
3. Submit exactly one complete Python file as a string. No markdown fences.

You do not have shell or filesystem tools during generation. The host will write the submitted source file and run Frontier-CS evaluation after generation."#;

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
        Arc::new(BuiltinRlmModePluginFactory::new(rlm_config())),
        Arc::new(llm_tools),
        Arc::new(StaticPluginFactory::new(
            "frontier_async_handles",
            PluginSpec::new().with_tool_provider(Arc::new(FrontierAsyncHandlesTool)),
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

fn rlm_config() -> RlmModePluginConfig {
    RlmModePluginConfig {
        prompt_features: RlmPromptFeatures {
            images: false,
            ..RlmPromptFeatures::default()
        },
        ..RlmModePluginConfig::default()
    }
}

struct FrontierAsyncHandlesTool;

#[async_trait]
impl ToolProvider for FrontierAsyncHandlesTool {
    fn definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition::raw(
            "list_async_handles",
            "List live lashlang async handles only. Returns `{ monitor: { monitor_id: handle }, subagent: { name: handle }, tool: { id: handle } }`; terminal, awaited, or cancelled handles are omitted.",
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
        .with_execution_mode(ToolExecutionMode::Parallel)]
    }

    async fn execute(&self, call: ToolCall<'_>) -> ToolResult {
        ToolResult::err_fmt(format_args!(
            "`{}` is handled by the RLM session runtime and cannot run directly",
            call.name
        ))
    }
}

fn evaluate_solution(
    args: &Args,
    problem: &FrontierProblem,
    solution_path: &Path,
) -> anyhow::Result<FrontierEvalRow> {
    let solution_path = solution_path
        .canonicalize()
        .with_context(|| format!("canonicalize {}", solution_path.display()))?;
    let backend = args.backend.clone().unwrap_or_else(|| {
        if problem.track == "research" {
            "skypilot".to_string()
        } else {
            "docker".to_string()
        }
    });
    let mut command = Command::new("uv");
    command
        .arg("run")
        .arg("frontier")
        .arg("eval")
        .arg(&problem.track)
        .arg(&problem.problem_id)
        .arg(&solution_path)
        .arg("--backend")
        .arg(&backend)
        .arg("--json")
        .arg("--judge-url")
        .arg(&args.judge_url)
        .current_dir(&args.source_dir);
    if args.keep_cluster {
        command.arg("--keep-cluster");
    }
    let output = command.output().context("run frontier eval")?;
    if !output.status.success() {
        return Ok(FrontierEvalRow {
            problem_id: problem.problem_id.clone(),
            score: None,
            score_unbounded: None,
            status: "error".to_string(),
            message: Some(format!(
                "frontier eval exited with {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            )),
            duration_seconds: None,
        });
    }
    let stdout = String::from_utf8(output.stdout).context("frontier eval stdout utf8")?;
    let rows: Vec<FrontierEvalRow> =
        serde_json::from_str(&stdout).with_context(|| format!("parse frontier JSON: {stdout}"))?;
    rows.into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("frontier eval returned no rows"))
}

fn solution_from_turn(outcome: &TurnOutcome, assistant_text: &str) -> String {
    if let TurnOutcome::Finished(TurnFinish::SubmittedValue { value }) = outcome {
        return match value {
            Value::String(text) => text.clone(),
            other => other.to_string(),
        };
    }
    assistant_text.trim().to_string()
}

fn strip_code_fence(text: &str) -> String {
    let trimmed = text.trim();
    if !trimmed.starts_with("```") {
        return trimmed.to_string();
    }
    let mut lines = trimmed.lines().collect::<Vec<_>>();
    if lines
        .first()
        .is_some_and(|line| line.trim_start().starts_with("```"))
    {
        lines.remove(0);
    }
    if lines.last().is_some_and(|line| line.trim() == "```") {
        lines.pop();
    }
    lines.join("\n").trim().to_string()
}

fn resolve_provider(args: &Args) -> anyhow::Result<ProviderHandle> {
    let mut provider = match args.provider_id.as_str() {
        "openai-compatible" => {
            let api_key = resolve_api_key(args).ok_or_else(|| {
                anyhow::anyhow!(
                    "missing API key - set OPENROUTER_API_KEY or OPENAI_COMPATIBLE_API_KEY in .env, or pass --api-key"
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
                    "missing or invalid {} - initialise a lash config to use provider `{other}`",
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

fn problem_artifacts(problem_dir: &Path, ext: &str) -> ProblemArtifacts {
    let path = |name: &str| problem_dir.join(name).display().to_string();
    ProblemArtifacts {
        problem_txt: path("problem.txt"),
        solution: path(&format!("solution.{ext}")),
        answer_txt: path("answer.txt"),
        result_json: path("result.json"),
        evaluation_json: path("evaluation.json"),
        events_jsonl: path("events.jsonl"),
        session_db: path("session.db"),
        trace_jsonl: path("session.trace.jsonl"),
        trace_html: path("trace.html"),
        system_prompt_txt: path("system_prompt.txt"),
    }
}

fn append_response_row(path: &Path, row: &ProblemResult) -> anyhow::Result<()> {
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
    for line in raw.lines().filter(|line| !line.trim().is_empty()) {
        let value: Value = serde_json::from_str(line)
            .with_context(|| format!("parse row from {}", path.display()))?;
        let track = value.get("track").and_then(Value::as_str);
        let problem_id = value.get("problem_id").and_then(Value::as_str);
        if let (Some(track), Some(problem_id)) = (track, problem_id) {
            out.insert(format!("{track}/{problem_id}"));
        }
    }
    Ok(out)
}

fn result_key(problem: &FrontierProblem) -> String {
    format!("{}/{}", problem.track, problem.problem_id)
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

fn aggregate_by_track(results: &[ProblemResult]) -> BTreeMap<String, Bucket> {
    let mut out = BTreeMap::<String, Bucket>::new();
    for result in results {
        let bucket = out.entry(result.track.clone()).or_default();
        bucket.count += 1;
        if result.successful {
            bucket.successful += 1;
        }
        bucket.score_sum += result.score.unwrap_or_default();
    }
    out
}

fn write_trace_index(
    output_dir: &Path,
    run_id: &str,
    results: &[ProblemResult],
) -> anyhow::Result<()> {
    let rows: String = results
        .iter()
        .map(|r| {
            let pid = html_escape(&r.problem_id);
            let track = html_escape(&r.track);
            let dir = format!("problems/{}/{}", track, safe_path_segment(&r.problem_id));
            let score = r
                .score
                .map(|value| format!("{value:.4}"))
                .unwrap_or_else(|| "-".to_string());
            let badge_class = if r.successful { "ok" } else { "fail" };
            format!(
                "<tr>\
                   <td><a href=\"{dir}/trace.html\">{track}/{pid}</a></td>\
                   <td>{language}</td>\
                   <td class=\"{badge_class}\">{score}</td>\
                   <td>{eval}</td>\
                   <td>{gen}</td>\
                   <td>{iters}</td>\
                   <td>{seconds:.1}s</td>\
                   <td><a href=\"{dir}/problem.txt\">problem</a> · \
                       <a href=\"{dir}/solution.{ext}\">solution</a> · \
                       <a href=\"{dir}/evaluation.json\">evaluation</a> · \
                       <a href=\"{dir}/events.jsonl\">events</a> · \
                       <a href=\"{dir}/session.trace.jsonl\">trace.jsonl</a> · \
                       <a href=\"{dir}/trace.html\">trace.html</a> · \
                       <a href=\"{dir}/session.db\">session.db</a></td>\
                 </tr>",
                language = html_escape(&r.language),
                eval = html_escape(&r.evaluation_status),
                gen = html_escape(&r.generation_status),
                iters = r.iterations,
                seconds = r.elapsed_seconds,
                ext = if r.language == "Python" { "py" } else { "cpp" },
            )
        })
        .collect();

    let html = format!(
        r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Frontier-CS run {run_id}</title>
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
<h1>Frontier-CS run <code>{run_id}</code></h1>
<p class="meta">{count} problems · see <a href="results.json">results.json</a> / <a href="manifest.json">manifest.json</a></p>
<table>
  <thead>
    <tr>
      <th>problem</th><th>language</th><th>score</th><th>evaluation</th>
      <th>generation</th><th>iters</th><th>elapsed</th><th>artifacts</th>
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

fn write_system_prompt_snapshot(trace_path: &Path, problem_dir: &Path) -> anyhow::Result<()> {
    if !trace_path.exists() {
        return Ok(());
    }
    let raw =
        fs::read_to_string(trace_path).with_context(|| format!("read {}", trace_path.display()))?;
    for line in raw.lines().filter(|line| !line.trim().is_empty()) {
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
            fs::write(problem_dir.join("system_prompt.txt"), text).with_context(|| {
                format!("write {}", problem_dir.join("system_prompt.txt").display())
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

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn safe_path_segment(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                ch
            } else {
                '_'
            }
        })
        .collect()
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

struct FrontierEventSink {
    file: Mutex<File>,
    last_error: Mutex<Option<String>>,
    iteration_count: Mutex<BTreeSet<usize>>,
    root_llm_calls: Mutex<usize>,
    child_llm_calls: Mutex<usize>,
    token_usage_events: Mutex<usize>,
    child_usage_events: Mutex<usize>,
    tool_calls_by_name: Mutex<BTreeMap<String, usize>>,
}

impl FrontierEventSink {
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

    fn metrics(&self, wall_clock_seconds: f64) -> ProblemMetrics {
        ProblemMetrics {
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
            tool_calls_by_name: self
                .tool_calls_by_name
                .lock()
                .map(|v| v.clone())
                .unwrap_or_default(),
        }
    }
}

#[async_trait]
impl EventSink for FrontierEventSink {
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
            SessionEvent::ToolCall { name, .. } => {
                if let Ok(mut counts) = self.tool_calls_by_name.lock() {
                    *counts.entry(name.clone()).or_default() += 1;
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
    fn strips_markdown_code_fence() {
        assert_eq!(
            strip_code_fence("```cpp\nint main() {}\n```"),
            "int main() {}"
        );
    }

    #[test]
    fn safe_path_segment_replaces_slashes() {
        assert_eq!(safe_path_segment("a/b:c"), "a_b_c");
    }
}
