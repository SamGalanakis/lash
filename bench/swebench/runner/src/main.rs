mod dataset;
mod run_json;

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use chrono::Utc;
use clap::Parser;
use dataset::{SweBenchInstance, load_instances};
use lash::plugin::PluginFactory;
use lash::provider::LashConfig;
use lash::{
    BackgroundRuntimeHost, BuiltinObservationalMemoryPluginFactory,
    BuiltinRollingHistoryPluginFactory, BuiltinToolResultProjectionPluginFactory, ContextApproach,
    EmbeddedRuntimeHost, EventSink, ExecutionMode, InputItem, LashRuntime, PersistedSessionState,
    PersistentRuntimeServices, PluginHost, Provider, RuntimeCoreConfig, RuntimeStore, SessionEvent,
    SessionPolicy, SessionUsageReport, Store, TokioSessionTaskExecutor, TurnInjectionBridge,
    TurnInput, TurnInputInjectionBridge, diff_usage_reports,
};
use lash_default_tools::{DefaultToolBundle, DefaultToolPluginOptions, tool_plugin_factories};
use lash_subagents::{
    CapabilityRegistry, LocalSubagentHost, SubagentHost, SubagentsPluginFactory, TierCapability,
    TierExecutionMode,
};
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

const STATE_ROOT: &str = ".benchmarks/swebench";
const DEFAULT_MAX_TURNS: usize = 60;
const DEFAULT_MAX_CONTEXT_TOKENS: usize = 1_000_000;
const DEFAULT_BATCH_SIZE: usize = 5;
const DEFAULT_EXECUTION_MODE: &str = "rlm";
const DEFAULT_CONTEXT_APPROACH: &str = "rolling_history";
const DEFAULT_DATASET: &str = ".benchmarks/swebench/verified.jsonl";

/// The user message sent to the model. The full problem statement is
/// appended verbatim; code-editing tools (shell, apply_patch, read,
/// grep, ls, glob) are registered on the session and the project root
/// is the instance's checked-out repo.
const TASK_PREAMBLE: &str = concat!(
    "You are fixing a real GitHub issue in an open-source repository. The repository is checked out at the issue's base commit in your current working directory — list it to orient yourself.\n\n",
    "Work steps:\n",
    "1. Read the problem statement end-to-end.\n",
    "2. Use `grep`/`ls`/`read_file` to locate the relevant source. Start narrow, don't dump whole files into context.\n",
    "3. Edit source files with `apply_patch` (or `shell` with redirection) to fix the issue. Keep changes minimal and surgical.\n",
    "4. Do NOT add, modify, or look at test files — the grader ships its own tests and runs them against your diff.\n",
    "5. Stop once the fix is in place. `git diff HEAD` in the project root will be captured as your submitted patch, so do not commit, push, or run `git reset`.\n",
);

#[derive(Parser, Debug, Clone)]
#[command(name = "bench-swebench")]
#[command(about = "Run SWE-bench Verified through Lash (direct runtime integration).")]
struct Args {
    /// JSONL or JSON file with SWE-bench instances. Produced by setup.sh from
    /// the upstream `SWE-bench/SWE-bench_Verified` dataset.
    #[arg(long, default_value = DEFAULT_DATASET)]
    dataset: PathBuf,

    #[arg(long, default_value = "SWE-bench/SWE-bench_Verified")]
    dataset_label: String,

    /// Shared workspace for cloned repos (they're shared across instances;
    /// each instance runs in its own `git worktree`).
    #[arg(long)]
    workspace_root: Option<PathBuf>,

    /// Run identifier; used as the output dir name under
    /// `.benchmarks/swebench/runs/`.
    #[arg(long)]
    run_id: Option<String>,

    /// Explicit output directory. Defaults to
    /// `.benchmarks/swebench/runs/<run_id>`.
    #[arg(long)]
    output_dir: Option<PathBuf>,

    /// Resume a previous run — skip instances whose predictions.jsonl row
    /// already exists.
    #[arg(long)]
    resume: bool,

    /// Run only these instance IDs (repeatable).
    #[arg(long)]
    instance_id: Vec<String>,

    #[arg(long)]
    limit: Option<usize>,

    #[arg(long, default_value_t = 0)]
    offset: usize,

    /// Model slug. If omitted, falls back to `~/.lash/config.json`'s active
    /// provider's default model (typically `gpt-5.4` on Codex).
    #[arg(long)]
    model: Option<String>,

    /// Reasoning effort variant (e.g. `high`, `xhigh`).
    #[arg(long, default_value = "high")]
    variant: String,

    /// Override the provider. By default the runner loads
    /// `~/.lash/config.json`; the active provider (Codex, OpenRouter, etc.)
    /// bills the run.
    #[arg(long)]
    provider_id: Option<String>,

    #[arg(long, default_value = DEFAULT_EXECUTION_MODE, value_parser = ["standard", "rlm"])]
    execution_mode: String,

    #[arg(long, default_value = DEFAULT_CONTEXT_APPROACH, value_parser = ["rolling_history", "observational_memory"])]
    context_approach: String,

    #[arg(long, default_value_t = DEFAULT_MAX_TURNS)]
    max_turns: usize,

    #[arg(long, default_value_t = DEFAULT_MAX_CONTEXT_TOKENS)]
    max_context_tokens: usize,

    #[arg(long, default_value_t = DEFAULT_BATCH_SIZE)]
    batch_size: usize,

    /// Label shown in the dashboard (e.g. `representative-10`).
    #[arg(long)]
    preset: Option<String>,

    #[arg(long)]
    dry_run: bool,

    /// Internal: run exactly one instance end-to-end and write its
    /// `result.json`. Used by the parent process to fan out one subprocess
    /// per instance so each has an isolated CWD (lash's file tools resolve
    /// paths against the process CWD).
    #[arg(long, hide = true)]
    single_instance: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct TokenTotals {
    pub input: u64,
    pub output: u64,
    pub reasoning: u64,
    pub cache: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InstanceResult {
    pub instance_id: String,
    pub repo: String,
    pub base_commit: String,
    pub model: String,
    pub execution_mode_label: String,
    pub model_patch: String,
    pub grade: String,
    pub failure_reason: Option<String>,
    pub assistant_text: String,
    pub iterations: u64,
    pub llm_calls: u64,
    pub tool_calls: u64,
    pub tool_breakdown: BTreeMap<String, u64>,
    pub tokens: TokenTotals,
    pub turn_status: String,
    pub done_reason: String,
    pub started_at: String,
    pub finished_at: String,
    pub elapsed_seconds: f64,
    pub checkout_seconds: f64,
    pub turn_seconds: f64,
}

pub struct RunSettings {
    pub run_id: String,
    pub dataset_label: String,
    pub model: String,
    pub variant: Option<String>,
    pub provider_kind: String,
    pub execution_mode_label: String,
    pub context_approach_label: String,
    pub batch_size: usize,
    pub preset: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();
    let args = Args::parse();

    if args.single_instance.is_some() {
        run_child(args).await
    } else {
        run_parent(args).await
    }
}

async fn run_parent(args: Args) -> Result<()> {
    let state_root = PathBuf::from(STATE_ROOT);
    fs::create_dir_all(&state_root).with_context(|| format!("create {}", state_root.display()))?;
    let workspace_root = args
        .workspace_root
        .clone()
        .unwrap_or_else(|| state_root.join("workspace"));
    fs::create_dir_all(&workspace_root)
        .with_context(|| format!("create {}", workspace_root.display()))?;

    if !args.dataset.exists() {
        bail!(
            "dataset {} not found — run bench/swebench/setup.sh first",
            args.dataset.display()
        );
    }
    let mut instances = load_instances(&args.dataset)?;
    instances = select_instances(instances, &args);
    if instances.is_empty() {
        bail!("no SWE-bench instances selected");
    }

    let run_id = args
        .run_id
        .clone()
        .unwrap_or_else(|| Utc::now().format("%Y%m%dT%H%M%SZ").to_string());
    let run_dir = args
        .output_dir
        .clone()
        .unwrap_or_else(|| state_root.join("runs").join(&run_id));
    fs::create_dir_all(&run_dir).with_context(|| format!("create {}", run_dir.display()))?;
    let run_dir = fs::canonicalize(&run_dir)
        .with_context(|| format!("canonicalize {}", run_dir.display()))?;
    let workspace_root = fs::canonicalize(&workspace_root)
        .with_context(|| format!("canonicalize {}", workspace_root.display()))?;
    let instances_root = run_dir.join("instances");
    fs::create_dir_all(&instances_root)
        .with_context(|| format!("create {}", instances_root.display()))?;
    let predictions_path = run_dir.join("predictions.jsonl");

    // Resolve provider + model in the parent only so the logged line
    // matches what each child will pick up from `~/.lash/config.json`.
    let (_provider, provider_kind, resolved_model) = resolve_provider(&args)?;
    let execution_mode = parse_execution_mode(&args.execution_mode)?;
    let context_approach = parse_context_approach(&args.context_approach)?;
    let execution_mode_label = execution_mode_label(execution_mode).to_string();
    let context_approach_label = context_approach_label(&context_approach).to_string();

    let completed = if args.resume {
        load_completed_ids(&predictions_path)?
    } else {
        BTreeSet::new()
    };
    let pending: Vec<SweBenchInstance> = instances
        .iter()
        .filter(|i| !completed.contains(&i.instance_id))
        .cloned()
        .collect();

    eprintln!("SWE-bench run_id={run_id}");
    eprintln!("  dataset:          {}", args.dataset.display());
    eprintln!("  selected:         {}", instances.len());
    eprintln!("  pending:          {}", pending.len());
    eprintln!("  model:            {resolved_model} (provider={provider_kind})");
    eprintln!("  variant:          {}", args.variant);
    eprintln!("  execution-mode:   {execution_mode_label}");
    eprintln!("  context-approach: {context_approach_label}");
    eprintln!("  batch_size:       {}", args.batch_size);
    eprintln!("  output:           {}", run_dir.display());

    if args.dry_run {
        for ins in &pending {
            eprintln!(
                "  [dry-run] {} ({} @ {})",
                ins.instance_id, ins.repo, ins.base_commit
            );
        }
        return Ok(());
    }
    if pending.is_empty() {
        eprintln!("nothing to run — predictions.jsonl already covers every selected instance");
        return Ok(());
    }

    let started_at = Utc::now();
    let started_instant = Instant::now();

    let child_exe = std::env::current_exe().context("resolve current_exe")?;
    let semaphore = Arc::new(Semaphore::new(args.batch_size.max(1)));
    let args_shared = Arc::new(args.clone());
    let workspace_root_shared = Arc::new(workspace_root);
    let run_dir_shared = Arc::new(run_dir.clone());
    let predictions_path_shared = Arc::new(predictions_path.clone());
    let child_exe_shared = Arc::new(child_exe);
    let append_mutex = Arc::new(tokio::sync::Mutex::new(()));
    let total = pending.len();
    let mut join_set: JoinSet<(usize, Result<InstanceResult>)> = JoinSet::new();
    for (index, instance) in pending.into_iter().enumerate() {
        let permit = semaphore
            .clone()
            .acquire_owned()
            .await
            .context("acquire benchmark slot")?;
        let args = args_shared.clone();
        let workspace_root = workspace_root_shared.clone();
        let run_dir = run_dir_shared.clone();
        let predictions_path = predictions_path_shared.clone();
        let child_exe = child_exe_shared.clone();
        let append_mutex = append_mutex.clone();
        join_set.spawn(async move {
            let _permit = permit;
            let result = spawn_child(
                &child_exe,
                &run_dir,
                &workspace_root,
                args.as_ref(),
                &instance,
            )
            .await;
            if let Ok(row) = &result {
                let _guard = append_mutex.lock().await;
                let _ = append_prediction(predictions_path.as_ref(), row);
            }
            (index, result)
        });
    }

    let mut indexed: Vec<(usize, InstanceResult)> = Vec::new();
    let mut failures: Vec<(String, anyhow::Error)> = Vec::new();
    let mut finished = 0usize;
    while let Some(joined) = join_set.join_next().await {
        let (index, result) = match joined {
            Ok(v) => v,
            Err(err) => {
                join_set.abort_all();
                return Err(anyhow::anyhow!("instance task panicked: {err}"));
            }
        };
        match result {
            Ok(row) => {
                finished += 1;
                eprintln!(
                    "  [{finished}/{total}] {} grade={} patch_bytes={} t={:.1}s iters={}",
                    row.instance_id,
                    row.grade,
                    row.model_patch.len(),
                    row.elapsed_seconds,
                    row.iterations
                );
                indexed.push((index, row));
            }
            Err(err) => {
                finished += 1;
                eprintln!("  [{finished}/{total}] ERROR: {err:#}");
                failures.push((format!("instance#{index}"), err));
            }
        }
    }
    indexed.sort_by_key(|(i, _)| *i);
    let results: Vec<InstanceResult> = indexed.into_iter().map(|(_, r)| r).collect();

    let finished_at = Utc::now();
    let duration_seconds = started_instant.elapsed().as_secs_f64();

    let run_settings = RunSettings {
        run_id: run_id.clone(),
        dataset_label: args_shared.dataset_label.clone(),
        model: resolved_model.clone(),
        variant: Some(args_shared.variant.clone()),
        provider_kind,
        execution_mode_label,
        context_approach_label,
        batch_size: args_shared.batch_size,
        preset: args_shared.preset.clone(),
    };
    run_json::write_dashboard_run_json(
        &run_dir,
        &run_settings,
        &results,
        &started_at.to_rfc3339(),
        &finished_at.to_rfc3339(),
        duration_seconds,
    )?;

    eprintln!();
    eprintln!("Run summary:");
    eprintln!("  run_dir:          {}", run_dir.display());
    eprintln!("  predictions:      {}", predictions_path.display());
    let produced = results
        .iter()
        .filter(|r| !r.model_patch.trim().is_empty())
        .count();
    eprintln!("  patches produced: {}/{}", produced, results.len());
    if !failures.is_empty() {
        eprintln!("  failures: {}", failures.len());
        for (id, err) in &failures {
            eprintln!("    {id}: {err:#}");
        }
    }
    eprintln!("  wall_clock:       {duration_seconds:.1}s");
    eprintln!();
    eprintln!("Evaluate with:");
    eprintln!("  bench/swebench/evaluate.sh {}", run_dir.display());
    Ok(())
}

async fn run_child(args: Args) -> Result<()> {
    let instance_id = args
        .single_instance
        .clone()
        .expect("single_instance required in child mode");
    let run_dir = args
        .output_dir
        .clone()
        .ok_or_else(|| anyhow::anyhow!("child requires --output-dir"))?;
    let workspace_root = args
        .workspace_root
        .clone()
        .ok_or_else(|| anyhow::anyhow!("child requires --workspace-root"))?;

    let instances = load_instances(&args.dataset)?;
    let instance = instances
        .into_iter()
        .find(|i| i.instance_id == instance_id)
        .ok_or_else(|| anyhow::anyhow!("instance {instance_id} not in dataset"))?;

    let (provider, _, resolved_model) = resolve_provider(&args)?;
    let execution_mode = parse_execution_mode(&args.execution_mode)?;
    let context_approach = parse_context_approach(&args.context_approach)?;

    let result = run_instance(
        &run_dir,
        &workspace_root,
        &provider,
        &args,
        &resolved_model,
        execution_mode,
        &context_approach,
        &instance,
    )
    .await
    .with_context(|| format!("run {}", instance_id))?;

    // result.json was already written by `run_instance`. Parent reads it
    // from instance_dir. Nothing more to do here.
    eprintln!(
        "child[{}] grade={} patch_bytes={} iters={} t={:.1}s",
        result.instance_id,
        result.grade,
        result.model_patch.len(),
        result.iterations,
        result.elapsed_seconds
    );
    Ok(())
}

async fn spawn_child(
    child_exe: &Path,
    run_dir: &Path,
    workspace_root: &Path,
    args: &Args,
    instance: &SweBenchInstance,
) -> Result<InstanceResult> {
    let instance_dir = run_dir.join("instances").join(&instance.instance_id);
    fs::create_dir_all(&instance_dir)
        .with_context(|| format!("create {}", instance_dir.display()))?;

    let mut cmd = tokio::process::Command::new(child_exe);
    cmd.arg("--single-instance").arg(&instance.instance_id);
    cmd.arg("--dataset").arg(&args.dataset);
    cmd.arg("--dataset-label").arg(&args.dataset_label);
    cmd.arg("--workspace-root").arg(workspace_root);
    cmd.arg("--output-dir").arg(run_dir);
    cmd.arg("--variant").arg(&args.variant);
    cmd.arg("--execution-mode").arg(&args.execution_mode);
    cmd.arg("--context-approach").arg(&args.context_approach);
    cmd.arg("--max-turns").arg(args.max_turns.to_string());
    cmd.arg("--max-context-tokens")
        .arg(args.max_context_tokens.to_string());
    if let Some(model) = args.model.as_deref() {
        cmd.arg("--model").arg(model);
    }
    if let Some(provider_id) = args.provider_id.as_deref() {
        cmd.arg("--provider-id").arg(provider_id);
    }
    // Pipe child stderr through a file in the instance dir so a crashed
    // child leaves a trail. stdout is unused but piped to /dev/null to
    // keep the parent terminal quiet.
    cmd.stdout(std::process::Stdio::null());
    let stderr_path = instance_dir.join("child.stderr.log");
    let stderr_file = std::fs::File::create(&stderr_path)
        .with_context(|| format!("create {}", stderr_path.display()))?;
    cmd.stderr(std::process::Stdio::from(stderr_file));

    let status = cmd
        .status()
        .await
        .with_context(|| format!("spawn child for {}", instance.instance_id))?;

    if !status.success() {
        let tail = read_tail(&stderr_path, 80).unwrap_or_default();
        bail!(
            "child exited with {} — last stderr lines:\n{}",
            status,
            tail
        );
    }

    let result_path = instance_dir.join("result.json");
    let raw = fs::read_to_string(&result_path)
        .with_context(|| format!("read {}", result_path.display()))?;
    let result: InstanceResult =
        serde_json::from_str(&raw).with_context(|| format!("parse {}", result_path.display()))?;
    Ok(result)
}

fn read_tail(path: &Path, max_lines: usize) -> Option<String> {
    let raw = fs::read_to_string(path).ok()?;
    let lines: Vec<&str> = raw.lines().collect();
    let start = lines.len().saturating_sub(max_lines);
    Some(lines[start..].join("\n"))
}

fn select_instances(mut instances: Vec<SweBenchInstance>, args: &Args) -> Vec<SweBenchInstance> {
    if !args.instance_id.is_empty() {
        let wanted: BTreeSet<&str> = args.instance_id.iter().map(String::as_str).collect();
        instances.retain(|i| wanted.contains(i.instance_id.as_str()));
    }
    if args.offset > 0 {
        instances = instances.into_iter().skip(args.offset).collect();
    }
    if let Some(limit) = args.limit {
        instances.truncate(limit);
    }
    instances
}

async fn run_instance(
    run_dir: &Path,
    workspace_root: &Path,
    provider: &Provider,
    args: &Args,
    model: &str,
    execution_mode: ExecutionMode,
    context_approach: &ContextApproach,
    instance: &SweBenchInstance,
) -> Result<InstanceResult> {
    let started_at = Utc::now();
    let started_instant = Instant::now();

    let instance_dir = run_dir.join("instances").join(&instance.instance_id);
    fs::create_dir_all(&instance_dir)
        .with_context(|| format!("create {}", instance_dir.display()))?;
    // Absolute path — `git worktree add` and the tool CWD both need a
    // stable location regardless of what the process CWD happens to be.
    let repo_dir = std::fs::canonicalize(&instance_dir)
        .with_context(|| format!("canonicalize {}", instance_dir.display()))?
        .join("repo");

    let checkout_started = Instant::now();
    checkout_instance(workspace_root, &repo_dir, instance)
        .with_context(|| format!("checkout {}", instance.instance_id))?;
    let checkout_seconds = checkout_started.elapsed().as_secs_f64();

    let prompt = format!(
        "{preamble}\n---\n## Problem statement\n\n{problem}\n",
        preamble = TASK_PREAMBLE,
        problem = instance.problem_statement.trim(),
    );
    fs::write(instance_dir.join("prompt.txt"), &prompt)
        .with_context(|| format!("write {}", instance_dir.join("prompt.txt").display()))?;
    fs::write(
        instance_dir.join("instance.json"),
        serde_json::to_string_pretty(instance)?,
    )
    .with_context(|| format!("write {}", instance_dir.join("instance.json").display()))?;

    let store_path = instance_dir.join("session.db");
    let llm_log_path = instance_dir.join("session.llm.jsonl");
    let events_path = instance_dir.join("events.jsonl");
    let store = Arc::new(
        Store::open(&store_path).with_context(|| format!("open {}", store_path.display()))?,
    );

    let policy = SessionPolicy {
        model: model.to_string(),
        provider: provider.clone(),
        max_context_tokens: Some(args.max_context_tokens),
        execution_mode,
        context_approach: context_approach.clone(),
        model_variant: Some(args.variant.clone()),
        max_turns: Some(args.max_turns),
        ..SessionPolicy::default()
    };
    let plugin_session = build_plugin_session(execution_mode, context_approach.clone(), &policy)?;
    let services = PersistentRuntimeServices::new_with_bridges(
        plugin_session,
        TurnInjectionBridge::new(),
        TurnInputInjectionBridge::new(),
        store.clone() as Arc<dyn RuntimeStore>,
    );
    let host = BackgroundRuntimeHost::new(
        EmbeddedRuntimeHost::new(
            RuntimeCoreConfig::default()
                .with_base_dir(&repo_dir)
                .with_llm_log_path(Some(llm_log_path.clone())),
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

    let sink = Arc::new(InstanceEventSink::new(events_path.clone())?);
    let sink_trait: Arc<dyn EventSink> = sink.clone();
    let cancel = tokio_util::sync::CancellationToken::new();

    let before_usage = runtime.usage_report();
    let turn_started = Instant::now();
    // Each instance runs in its own subprocess (see `spawn_child`), so we
    // own the process CWD for the duration of this turn. Lash's file tools
    // resolve paths against the process CWD rather than
    // `RuntimeCoreConfig::base_dir`, so this is the only way to pin them
    // to the instance's worktree.
    std::env::set_current_dir(&repo_dir)
        .with_context(|| format!("cd into {}", repo_dir.display()))?;
    let turn = runtime
        .stream_turn(
            TurnInput {
                items: vec![InputItem::Text { text: prompt }],
                image_blobs: Default::default(),
                user_input: None,
                mode: None,
                rlm_termination_override: None,
            },
            sink_trait.as_ref(),
            cancel,
        )
        .await
        .context("run swebench instance")?;
    let model_patch = capture_git_diff(&repo_dir)
        .with_context(|| format!("capture diff for {}", instance.instance_id))?;
    let turn_seconds = turn_started.elapsed().as_secs_f64();
    let after_usage = runtime.usage_report();
    let usage = diff_usage_reports(&before_usage, &after_usage)
        .map(|rows| SessionUsageReport::from_entries(&rows))
        .map_err(anyhow::Error::msg)
        .context("diff usage reports")?;
    fs::write(instance_dir.join("model.patch"), &model_patch)
        .with_context(|| format!("write {}", instance_dir.join("model.patch").display()))?;

    let turn_status = match turn.status {
        lash::TurnStatus::Completed => "completed",
        lash::TurnStatus::Interrupted => "interrupted",
        lash::TurnStatus::Failed => "failed",
    };
    let done_reason = match turn.done_reason {
        lash::DoneReason::ModelStop => "model_stop",
        lash::DoneReason::MaxTurns => "max_turns",
        lash::DoneReason::UserAbort => "user_abort",
        lash::DoneReason::ToolFailure => "tool_failure",
        lash::DoneReason::RuntimeError => "runtime_error",
    };

    let grade = if !matches!(turn.status, lash::TurnStatus::Completed) {
        "error"
    } else if model_patch.trim().is_empty() {
        "fail"
    } else {
        "no-reward"
    };

    let failure_reason = turn
        .errors
        .first()
        .map(|e| e.message.clone())
        .or_else(|| sink.last_error())
        .or_else(|| {
            if grade == "fail" {
                Some("empty model patch (no edits made)".to_string())
            } else {
                None
            }
        });

    let tool_breakdown = sink.tool_breakdown();
    let tool_calls = tool_breakdown.values().copied().sum::<u64>();
    let tokens = aggregate_usage(&usage);
    let assistant_text = sink
        .last_llm_response()
        .or_else(|| non_empty(&turn.assistant_output.safe_text))
        .unwrap_or_default();

    let result = InstanceResult {
        instance_id: instance.instance_id.clone(),
        repo: instance.repo.clone(),
        base_commit: instance.base_commit.clone(),
        model: model.to_string(),
        execution_mode_label: execution_mode_label(execution_mode).to_string(),
        model_patch,
        grade: grade.to_string(),
        failure_reason,
        assistant_text,
        iterations: sink.iteration_count() as u64,
        llm_calls: sink.llm_response_count() as u64,
        tool_calls,
        tool_breakdown,
        tokens,
        turn_status: turn_status.to_string(),
        done_reason: done_reason.to_string(),
        started_at: started_at.to_rfc3339(),
        finished_at: Utc::now().to_rfc3339(),
        elapsed_seconds: started_instant.elapsed().as_secs_f64(),
        checkout_seconds,
        turn_seconds,
    };
    fs::write(
        instance_dir.join("result.json"),
        serde_json::to_string_pretty(&result)?,
    )
    .with_context(|| format!("write {}", instance_dir.join("result.json").display()))?;

    // Repo worktree is gigabytes per instance on large repos — prune it.
    // The dedicated predictions + diff artifacts already capture everything we
    // need for the dashboard.
    let _ = remove_worktree(workspace_root, instance, &repo_dir);

    Ok(result)
}

fn checkout_instance(
    workspace_root: &Path,
    repo_dir: &Path,
    instance: &SweBenchInstance,
) -> Result<()> {
    let bare_dir = workspace_root.join(bare_repo_dirname(&instance.repo));
    ensure_bare_clone(&bare_dir, &instance.repo)?;
    ensure_commit_present(&bare_dir, &instance.base_commit)?;
    // Drop any stale registration for this path (prior crashed run) so
    // `worktree add` doesn't bail with "already exists".
    let _ = run_git(
        &bare_dir,
        &[
            "worktree",
            "remove",
            "--force",
            &repo_dir.display().to_string(),
        ],
    );
    let _ = run_git(&bare_dir, &["worktree", "prune"]);
    if repo_dir.exists() {
        fs::remove_dir_all(repo_dir).with_context(|| format!("remove {}", repo_dir.display()))?;
    }
    fs::create_dir_all(repo_dir.parent().unwrap_or(Path::new(".")))
        .with_context(|| format!("create parent of {}", repo_dir.display()))?;
    run_git(
        &bare_dir,
        &[
            "worktree",
            "add",
            "--detach",
            "-f",
            &repo_dir.display().to_string(),
            &instance.base_commit,
        ],
    )
    .with_context(|| format!("git worktree add {}", instance.base_commit))?;
    Ok(())
}

fn remove_worktree(
    workspace_root: &Path,
    instance: &SweBenchInstance,
    repo_dir: &Path,
) -> Result<()> {
    let bare_dir = workspace_root.join(bare_repo_dirname(&instance.repo));
    if !repo_dir.exists() {
        return Ok(());
    }
    let _ = run_git(
        &bare_dir,
        &[
            "worktree",
            "remove",
            "--force",
            &repo_dir.display().to_string(),
        ],
    );
    if repo_dir.exists() {
        fs::remove_dir_all(repo_dir).ok();
    }
    let _ = run_git(&bare_dir, &["worktree", "prune"]);
    Ok(())
}

fn bare_repo_dirname(repo: &str) -> String {
    let sanitized = repo.replace('/', "__");
    format!("{sanitized}.git")
}

fn ensure_bare_clone(bare_dir: &Path, repo: &str) -> Result<()> {
    if bare_dir.join("HEAD").exists() {
        return Ok(());
    }
    fs::create_dir_all(bare_dir.parent().unwrap_or(Path::new(".")))
        .with_context(|| format!("create parent of {}", bare_dir.display()))?;
    let url = format!("https://github.com/{repo}.git");
    eprintln!("  cloning {url} → {}", bare_dir.display());
    run_git(
        Path::new("."),
        &["clone", "--bare", &url, &bare_dir.display().to_string()],
    )
    .with_context(|| format!("git clone {url}"))?;
    Ok(())
}

fn ensure_commit_present(bare_dir: &Path, sha: &str) -> Result<()> {
    if run_git(bare_dir, &["cat-file", "-e", &format!("{sha}^{{commit}}")]).is_ok() {
        return Ok(());
    }
    run_git(bare_dir, &["fetch", "--tags", "origin", sha])
        .or_else(|_| run_git(bare_dir, &["fetch", "origin"]))
        .with_context(|| format!("fetch {sha}"))?;
    run_git(bare_dir, &["cat-file", "-e", &format!("{sha}^{{commit}}")])
        .with_context(|| format!("commit {sha} still missing after fetch"))?;
    Ok(())
}

fn run_git(cwd: &Path, args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(cwd)
        .output()
        .with_context(|| format!("spawn git {args:?} in {}", cwd.display()))?;
    if !output.status.success() {
        bail!(
            "git {args:?} in {} failed: {}",
            cwd.display(),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn capture_git_diff(repo_dir: &Path) -> Result<String> {
    // Stage everything so `git diff HEAD` captures new files too.
    // Using `HEAD` (not `--cached`) produces the conventional `a/`/`b/`
    // prefixes that SWE-bench's evaluator expects.
    run_git(repo_dir, &["add", "-A"]).ok();
    let output = Command::new("git")
        .args([
            "-c",
            "diff.mnemonicPrefix=false",
            "-c",
            "diff.noprefix=false",
            "diff",
            "HEAD",
            "--no-color",
            "--binary",
            "--src-prefix=a/",
            "--dst-prefix=b/",
        ])
        .current_dir(repo_dir)
        .output()
        .with_context(|| "git diff HEAD")?;
    if !output.status.success() {
        bail!(
            "git diff failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn build_plugin_session(
    execution_mode: ExecutionMode,
    context_approach: ContextApproach,
    policy: &SessionPolicy,
) -> Result<Arc<lash::PluginSession>> {
    let mut factories: Vec<Arc<dyn PluginFactory>> =
        vec![Arc::new(BuiltinToolResultProjectionPluginFactory::default())];
    match context_approach {
        ContextApproach::RollingHistory(_) => {
            factories.push(Arc::new(BuiltinRollingHistoryPluginFactory::default()));
        }
        ContextApproach::ObservationalMemory(_) => {
            factories.push(Arc::new(BuiltinObservationalMemoryPluginFactory));
        }
    }
    factories.push(Arc::new(
        lash_mode_standard::BuiltinStandardModePluginFactory,
    ));
    factories.push(Arc::new(
        lash_mode_rlm::BuiltinRlmModePluginFactory::default(),
    ));

    // Tool bundles only — core/context/mode plugins are registered
    // above, so we ask `tool_plugin_factories` for just the tool
    // surfaces (shell, apply_patch, read/ls/grep/glob). No `ask`
    // (autonomous run) and no web tools.
    let mut tool_factories = tool_plugin_factories(DefaultToolPluginOptions {
        execution_mode,
        context_approach: context_approach.clone(),
        bundles: vec![
            DefaultToolBundle::Shell,
            DefaultToolBundle::Editing,
            DefaultToolBundle::Files,
            DefaultToolBundle::Search,
        ],
        tavily_api_key: None,
        instruction_source: None,
    });
    factories.append(&mut tool_factories);

    let registry = Arc::new(CapabilityRegistry::new().with(Arc::new(TierCapability::new(
        "default",
        None,
        std::iter::empty::<String>(),
        TierExecutionMode::Inherit,
    ))));
    let subagent_host: Arc<dyn SubagentHost> = Arc::new(LocalSubagentHost::default());
    factories.push(Arc::new(SubagentsPluginFactory::new(
        policy.clone(),
        registry,
        subagent_host,
    )));

    let plugin_host = PluginHost::new(factories);
    plugin_host
        .build_session("root", execution_mode, context_approach, None)
        .context("build plugin session")
}

fn resolve_provider(args: &Args) -> Result<(Provider, String, String)> {
    let config = LashConfig::load();
    let provider = config
        .as_ref()
        .map(|c| c.active_provider().clone())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "~/.lash/config.json not found or invalid — set up a provider with `lash --provider` or re-login"
            )
        })?;
    let kind = provider.kind().id().to_string();
    let model = args
        .model
        .clone()
        .unwrap_or_else(|| provider.default_model().to_string());
    Ok((provider, kind, model))
}

fn parse_execution_mode(raw: &str) -> Result<ExecutionMode> {
    match raw {
        "rlm" => Ok(ExecutionMode::Rlm),
        "standard" => Ok(ExecutionMode::Standard),
        other => bail!("unsupported execution mode `{other}`"),
    }
}

fn parse_context_approach(raw: &str) -> Result<ContextApproach> {
    match raw {
        "rolling_history" => Ok(ContextApproach::RollingHistory(Default::default())),
        "observational_memory" => Ok(ContextApproach::ObservationalMemory(Default::default())),
        other => bail!("unsupported context approach `{other}`"),
    }
}

fn execution_mode_label(mode: ExecutionMode) -> &'static str {
    match mode {
        ExecutionMode::Rlm => "rlm",
        ExecutionMode::Standard => "standard",
    }
}

fn context_approach_label(approach: &ContextApproach) -> &'static str {
    match approach {
        ContextApproach::RollingHistory(_) => "rolling_history",
        ContextApproach::ObservationalMemory(_) => "observational_memory",
    }
}

fn non_empty(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn aggregate_usage(report: &SessionUsageReport) -> TokenTotals {
    let mut out = TokenTotals::default();
    for row in &report.by_source_model {
        out.input += row.usage.input_tokens.max(0) as u64;
        out.output += row.usage.output_tokens.max(0) as u64;
        out.cache += row.usage.cached_input_tokens.max(0) as u64;
        out.reasoning += row.usage.reasoning_tokens.max(0) as u64;
    }
    out
}

fn load_completed_ids(path: &Path) -> Result<BTreeSet<String>> {
    if !path.exists() {
        return Ok(BTreeSet::new());
    }
    let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut out = BTreeSet::new();
    for line in raw.lines().filter(|l| !l.trim().is_empty()) {
        let value: serde_json::Value = serde_json::from_str(line)
            .with_context(|| format!("parse row from {}", path.display()))?;
        if let Some(id) = value.get("instance_id").and_then(|v| v.as_str()) {
            out.insert(id.to_string());
        }
    }
    Ok(out)
}

fn append_prediction(path: &Path, row: &InstanceResult) -> Result<()> {
    let prediction = serde_json::json!({
        "instance_id": row.instance_id,
        "model_name_or_path": row.model,
        "model_patch": row.model_patch,
    });
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("open {}", path.display()))?;
    writeln!(file, "{}", serde_json::to_string(&prediction)?)
        .with_context(|| format!("append {}", path.display()))?;
    Ok(())
}

struct InstanceEventSink {
    file: Mutex<File>,
    last_llm_response: Mutex<Option<String>>,
    iterations: Mutex<BTreeSet<usize>>,
    last_error: Mutex<Option<String>>,
    llm_response_count: Mutex<u64>,
    tool_breakdown: Mutex<BTreeMap<String, u64>>,
}

impl InstanceEventSink {
    fn new(path: PathBuf) -> Result<Self> {
        let file = File::create(&path).with_context(|| format!("create {}", path.display()))?;
        Ok(Self {
            file: Mutex::new(file),
            last_llm_response: Mutex::new(None),
            iterations: Mutex::new(BTreeSet::new()),
            last_error: Mutex::new(None),
            llm_response_count: Mutex::new(0),
            tool_breakdown: Mutex::new(BTreeMap::new()),
        })
    }

    fn last_llm_response(&self) -> Option<String> {
        self.last_llm_response.lock().ok().and_then(|g| g.clone())
    }

    fn iteration_count(&self) -> usize {
        self.iterations.lock().map(|g| g.len()).unwrap_or(0)
    }

    fn llm_response_count(&self) -> u64 {
        *self
            .llm_response_count
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    fn tool_breakdown(&self) -> BTreeMap<String, u64> {
        self.tool_breakdown
            .lock()
            .map(|g| g.clone())
            .unwrap_or_default()
    }

    fn last_error(&self) -> Option<String> {
        self.last_error.lock().ok().and_then(|g| g.clone())
    }
}

#[async_trait::async_trait]
impl EventSink for InstanceEventSink {
    async fn emit(&self, event: SessionEvent) {
        match &event {
            SessionEvent::LlmRequest { iteration, .. } => {
                if let Ok(mut s) = self.iterations.lock() {
                    s.insert(*iteration);
                }
            }
            SessionEvent::LlmResponse { content, .. } => {
                if let Ok(mut last) = self.last_llm_response.lock() {
                    *last = Some(content.trim().to_string());
                }
                if let Ok(mut count) = self.llm_response_count.lock() {
                    *count += 1;
                }
            }
            SessionEvent::Error { message, .. } => {
                if let Ok(mut last) = self.last_error.lock() {
                    *last = Some(message.clone());
                }
            }
            SessionEvent::ToolCall { name, .. } => {
                if let Ok(mut map) = self.tool_breakdown.lock() {
                    *map.entry(name.clone()).or_insert(0) += 1;
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
